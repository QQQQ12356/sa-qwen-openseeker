"""Single-node training launcher shared by shell scripts and experiment.py."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def training_command(module, arguments, python=sys.executable):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    default_count = len([value for value in visible.split(",") if value.strip() and value != "-1"])
    count = int(os.environ.get("NPROC_PER_NODE", str(max(1, default_count))))
    if count < 1:
        raise ValueError("NPROC_PER_NODE 必须大于 0")
    if count == 1:
        return [python, "-u", "-m", module, *arguments]
    backend = os.environ.get("DISTRIBUTED_BACKEND", "zero2")
    if backend not in {"ddp", "zero2", "zero3"}:
        raise ValueError("DISTRIBUTED_BACKEND 必须为 ddp / zero2 / zero3")
    if backend == "zero3" and module.endswith("beacon_trainer"):
        raise ValueError("Beacon 动态分段训练请使用 zero2 或 ddp；zero3 仅支持 plain")
    overrides = []
    if backend != "ddp":
        config = Path(__file__).resolve().parents[2] / "configs" / "train" / f"ds_{backend}.json"
        overrides = ["--set", f"deepspeed={config}"]
    return [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={count}", "--module", module, *arguments, *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    args, arguments = parser.parse_known_args()
    command = training_command(args.module, arguments)
    print("[train-launch]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
