"""Two-process CPU smoke test for the actual Trainer / DDP / save path."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Trainer, TrainingArguments

from test_beacon_qwen35_linear_memory import _tiny_model, modeling_qwen3_5
from search_comp.trainer.beacon_distillation_trainer import BeaconDistillationTrainer
from search_comp.trainer.beacon_trainer import _SaveBeaconCheckpoints
from search_comp.utils.trainer_callbacks import JsonlMetricsCallback
from search_comp.utils.distributed import setup_training


def main():
    output, kind = sys.argv[1:]
    setup_training({})
    torch.manual_seed(42)
    model = _tiny_model()
    if kind == "plain":
        with patch.multiple(modeling_qwen3_5, chunk_gated_delta_rule=None,
                            fused_recurrent_gated_delta_rule=None,
                            causal_conv1d_fn=None, FusedRMSNormGated=None):
            model = Qwen3_5ForCausalLM(model.config)
    else:
        cfg = model.beacon_config
        cfg.beacon_information_mode = "whole"
        cfg.beacon_window_enabled = False
        cfg.beacon_attention_backend = "sdpa"
        cfg.beacon_output_distill_weight = 0
        cfg.beacon_readout_distill_weight = 0
        model.set_beacon_config(cfg)
    model = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    for name, param in model.named_parameters():
        if "beacon" in name:
            param.requires_grad_(True)
    model.train()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    dataset = []
    for index in range(8):
        ids = torch.arange(10, 22 + index % 3)
        row = dict(input_ids=ids, labels=ids.clone())
        if kind == "beacon":
            # Different ranks exercise different numbers of keep/compress segments.
            row["regions"] = [(2, 6)] if index % 2 else [(2, 5), (7, 10)]
        dataset.append(row)

    def collate(rows):
        row = rows[0]
        return {key: value.unsqueeze(0) if torch.is_tensor(value) else value
                for key, value in row.items()}

    args = TrainingArguments(
        output_dir=output, use_cpu=True, report_to=[], max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        logging_steps=1, save_steps=1, save_strategy="steps" if kind == "plain" else "no",
        remove_unused_columns=False, ddp_find_unused_parameters=True,
        gradient_checkpointing=kind == "plain", gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_tqdm=True,
    )
    callbacks = [JsonlMetricsCallback(output)]
    if kind == "beacon":
        callbacks.append(_SaveBeaconCheckpoints(output, 1, model, tokenizer, True, 2))
    cls = BeaconDistillationTrainer if kind == "beacon" else Trainer
    trainer = cls(model=model, args=args, train_dataset=dataset, data_collator=collate,
                  callbacks=callbacks, processing_class=tokenizer)
    for callback in callbacks:
        if isinstance(callback, _SaveBeaconCheckpoints):
            callback.trainer = trainer
    trainer.train()
    # Verify that parameters are synchronized after the optimizer updates.
    param = next(param for name, param in model.named_parameters() if "lora_B" in name)
    gathered = [torch.empty_like(param) for _ in range(2)]
    torch.distributed.all_gather(gathered, param.detach())
    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    assert param.detach().abs().sum() > 0
    trainer.accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
