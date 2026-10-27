"""Launch selection, SDPA equivalence, and checkpoint contents on CPU."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from search_comp.trainer.launch import training_command
from search_comp.utils.distributed import save_beacon_checkpoint, setup_training
from test_beacon_qwen35_linear_memory import _tiny_model
from test_qwen3_whole_information import tiny_qwen3


@pytest.mark.parametrize("kind", ["beacon", "plain_sft"])
def test_multigpu_launch_uses_torchrun_and_zero2(monkeypatch, kind):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.delenv("NPROC_PER_NODE", raising=False)
    monkeypatch.delenv("DISTRIBUTED_BACKEND", raising=False)
    command = training_command(f"search_comp.trainer.{kind}_trainer", ["--config", "run.yaml"])
    assert "torch.distributed.run" in command
    assert "--nproc_per_node=2" in command
    assert command[-1].endswith("configs/train/ds_zero2.json")
    config = json.load(open(command[-1].split("=", 1)[1]))
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert config["gradient_accumulation_steps"] == "auto"


def test_zero3_rejected_for_dynamic_beacon(monkeypatch):
    monkeypatch.setenv("NPROC_PER_NODE", "2")
    monkeypatch.setenv("DISTRIBUTED_BACKEND", "zero3")
    with pytest.raises(ValueError, match="zero3"):
        training_command("search_comp.trainer.beacon_trainer", [])
    with pytest.raises(ValueError, match="ZeRO-3"):
        setup_training({"deepspeed": {"zero_optimization": {"stage": 3}}}, beacon=True)


@pytest.mark.parametrize("factory", [tiny_qwen3, _tiny_model])
@pytest.mark.parametrize("position", ["append", "intersect"])
def test_sdpa_preserves_loss_gradients_and_compressed_memory(factory, position):
    torch.manual_seed(17)
    eager = factory().train()
    config = eager.beacon_config
    config.beacon_information_mode = "whole"
    config.beacon_window_enabled = False
    config.beacon_keep_window = 0
    config.beacon_pos = position
    config.beacon_readout_distill_weight = 0
    config.beacon_output_distill_weight = 0
    eager.set_beacon_config(config)
    sdpa = copy.deepcopy(eager)
    config = copy.deepcopy(config)
    config.beacon_attention_backend = "sdpa"
    sdpa.set_beacon_config(config)
    ids = torch.arange(10, 28).unsqueeze(0)
    labels = ids.clone()
    labels[:, 2:8] = -100
    labels[:, 11:15] = -100
    batch = dict(input_ids=ids, labels=labels, regions=[(2, 8), (11, 15)])
    first, second = eager(**batch).loss, sdpa(**batch).loss
    torch.testing.assert_close(first, second)
    first.backward()
    second.backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), sdpa.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)
    for (key, value), (other_key, other_value) in zip(eager._mem._cache, sdpa._mem._cache):
        if key is not None:
            torch.testing.assert_close(key, other_key)
            torch.testing.assert_close(value, other_value)


@pytest.mark.parametrize("factory", [tiny_qwen3, _tiny_model])
@pytest.mark.parametrize("position", ["append", "intersect"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sdpa_bf16_aligns_mask_and_backpropagates(monkeypatch, factory, position, device):
    if device == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("Requires a CUDA GPU with BF16 support")
    model = factory().to(device=device, dtype=torch.bfloat16).train()
    config = copy.deepcopy(model.beacon_config)
    config.beacon_information_mode = "whole"
    config.beacon_window_enabled = False
    config.beacon_keep_window = 0
    config.beacon_pos = position
    config.beacon_attention_backend = "sdpa"
    config.beacon_readout_distill_weight = 0
    config.beacon_output_distill_weight = 0
    model.set_beacon_config(config)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def checked_sdpa(query, key, value, *, attn_mask=None, **kwargs):
        # CPU SDPA accepts FP32 masks with BF16 queries, whereas the CUDA
        # memory-efficient kernel rejects them. Check the contract on CPU too.
        assert query.dtype == key.dtype == value.dtype == torch.bfloat16
        assert attn_mask is not None
        assert attn_mask.dtype == query.dtype
        assert attn_mask.device == query.device
        assert (attn_mask == 0).any() and (attn_mask < 0).any()
        calls.append(query.shape)
        return sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", checked_sdpa)
    ids = torch.arange(10, 28, device=device).unsqueeze(0)
    labels = ids.clone()
    labels[:, 2:8] = -100
    labels[:, 11:15] = -100
    loss = model(input_ids=ids, labels=labels, regions=[(2, 8), (11, 15)]).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert len(calls) >= 5  # prefix, two compressed regions, intervening keeps
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_adapter_save_preserves_beacon_and_lora_weights(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    model = get_peft_model(_tiny_model(), LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    class Tokenizer:
        def save_pretrained(self, directory):
            pass
    trainer = SimpleNamespace(is_world_process_zero=lambda: True)
    save_beacon_checkpoint(trainer, model, Tokenizer(), tmp_path, True)
    state = torch.load(tmp_path / "beacon_state.pt", weights_only=True)
    expected = {name.removeprefix("base_model.model."): p for name, p in model.named_parameters()
                if "beacon" in name}
    assert state.keys() == expected.keys()
    for name, value in state.items():
        torch.testing.assert_close(value, expected[name])
    restored = PeftModel.from_pretrained(_tiny_model(), tmp_path)
    for name, param in restored.named_parameters():
        if "lora_" in name:
            torch.testing.assert_close(param, dict(model.named_parameters())[name])
