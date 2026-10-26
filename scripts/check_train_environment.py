"""Run on a GPU compute node before launching distributed training."""

import os
import re
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME


def main():
    print(f"torch={torch.__version__}, torch CUDA={torch.version.cuda}")
    print(f"CUDA_HOME env={os.environ.get('CUDA_HOME')}, resolved={CUDA_HOME}")
    if torch.__version__ != "2.7.0+cu126":
        raise RuntimeError("Expected torch==2.7.0+cu126")
    if os.environ.get("DS_SKIP_CUDA_CHECK", "0") != "0":
        raise RuntimeError("Unset DS_SKIP_CUDA_CHECK before verification")
    if not CUDA_HOME:
        raise RuntimeError("Set CUDA_HOME to the CUDA 12.6 toolkit directory")
    output = subprocess.check_output(
        [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    print(output.strip())
    match = re.search(r"release (\d+\.\d+)", output)
    if not match or match.group(1) != torch.version.cuda:
        raise RuntimeError("nvcc and torch.version.cuda must both be 12.6")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this check on a compute node with a visible GPU")
    print(f"GPU={torch.cuda.get_device_name(0)}")
    x = torch.ones(16, device="cuda", requires_grad=True)
    x.square().sum().backward()
    torch.cuda.synchronize()

    # Exercise this project's imports before downloading/loading a checkpoint.
    import peft  # noqa: F401
    from search_comp.models.beacon_qwen35 import BeaconLinearStateWriter  # noqa: F401
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    # Triggers the exact extension build in the reported traceback and checks
    # that the resulting optimizer can perform a real CPU parameter update.
    parameter = torch.nn.Parameter(torch.ones(16))
    optimizer = DeepSpeedCPUAdam([parameter], lr=0.01)
    parameter.square().sum().backward()
    optimizer.step()
    if not torch.isfinite(parameter).all() or torch.equal(parameter, torch.ones(16)):
        raise RuntimeError("CPUAdam parameter update failed")
    print("PASS: CUDA backward, Beacon imports, CPUAdam build and update")


if __name__ == "__main__":
    main()

