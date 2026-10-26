"""Training placement and checkpoint helpers for torchrun / DeepSpeed."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path

import torch


def setup_training(cfg, beacon=False):
    """Initialize before loading models; never use device_map=auto in torchrun."""
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        from accelerate import PartialState
        PartialState()
    ds = cfg.get("deepspeed")
    if ds:
        if isinstance(ds, str):
            ds = json.loads(Path(ds).read_text())
        if beacon and ds.get("zero_optimization", {}).get("stage") == 3:
            raise ValueError("Beacon 的动态分段前向当前不支持 ZeRO-3，请使用 configs/train/ds_zero2.json")
        cfg["deepspeed"] = ds
        # The supplied DS config owns the optimizer, including CPU offload.
        cfg["optim"] = "adamw_torch"
        return "cpu"
    if distributed:
        return f"cuda:{int(os.environ['LOCAL_RANK'])}" if torch.cuda.is_available() else "cpu"
    return "auto"


def save_beacon_checkpoint(trainer, model, tokenizer, save_dir, use_lora):
    """Every rank participates in ZeRO gathers; only rank zero writes files."""
    if not use_lora:
        state = trainer.accelerator.get_state_dict(trainer.model_wrapped)
        if trainer.is_world_process_zero():
            model.save_pretrained(save_dir, state_dict=state)
            tokenizer.save_pretrained(save_dir)
        return

    # Gather just the adapter and Beacon tensors, one at a time, onto CPU.
    # Gathering the entire 4B base model would create a large save-time GPU peak.
    state = {}
    for name, param in model.named_parameters():
        if "lora_" not in name and "beacon" not in name:
            continue
        context = nullcontext()
        if hasattr(param, "ds_id"):
            import deepspeed
            context = deepspeed.zero.GatheredParameters([param], modifier_rank=None)
        with context:
            if trainer.is_world_process_zero():
                state[name] = param.detach().cpu().clone()
    if trainer.is_world_process_zero():
        model.save_pretrained(save_dir, state_dict=state)
        beacon_state = {name.removeprefix("base_model.model."): value
                        for name, value in state.items() if "beacon" in name}
        torch.save(beacon_state, Path(save_dir) / "beacon_state.pt")
        tokenizer.save_pretrained(save_dir)
