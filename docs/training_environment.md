# Qwen3.5 / DeepSpeed CUDA 12.6 环境

适用 Linux x86_64 NVIDIA GPU 服务器。此配置针对 `tmp.txt` 和此次 traceback，
保留 torch 2.7.0、Transformers 5.9.0、Accelerate 1.14.0、DeepSpeed 0.15.0，
统一使用 CUDA 12.6；PEFT 改用 0.18.0，与 Transformers 5.9.0 声明的 PEFT extra
下限一致。环境文件固定核心依赖，不是包含所有传递依赖和哈希的完整 lockfile。
尚未在目标服务器完成安装或 GPU 训练验证，需执行下述检查。

## 报错原因

`ds_zero2.json` 和 `ds_zero3.json` 都有 `offload_optimizer.device=cpu`，
DeepSpeed 因此选择 CPUAdam。虽然优化器参数在 CPU 上，其扩展构建仍检查 CUDA。
本次构建检测到 Toolkit 12.8，但 `torch.version.cuda` 是 12.6。
DeepSpeed 0.15.0 的 CUDA 次版本兼容白名单只列到 12.5，因此拒绝该组合。

`ds_opt_adam` 缺失是构造失败后析构函数再次访问未初始化成员；NCCL 清理警告和
`ChildFailedError` 是进程异常退出的后续信息，不是此次最先出现的故障。

`tmp.txt` 同时包含 CUDA 12 和 CUDA 13 的 NVIDIA pip 包，存在额外动态库冲突风险，
但这份列表本身不足以证明已加载了错误的动态库。本次直接原因由 traceback 中的
12.8 / 12.6 检查明确给出。不要原样复装整个 `tmp.txt`。

注意区分：`nvidia-smi` 中的 CUDA Version 是驱动支持能力；`nvcc --version`
是编译工具链；`torch.version.cuda` 是 PyTorch 的构建版本。后两者应在此方案中
都为 12.6。无需把驱动“降级到 12.6”；驱动只需支持 CUDA 12.6 和实际 GPU。

## 新建环境（在仓库根目录执行）

```bash
conda env create -f environment.train-cu126.yml
conda activate search-comp-qwen35-cu126
export PYTHON="$CONDA_PREFIX/bin/python"

# 每次新 shell 激活环境后、编译和训练前都要设置。
# 集群如自动加载 CUDA 12.8 module，应先卸载对应 module。
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export MAX_JOBS=4
unset DS_SKIP_CUDA_CHECK
# 使用新缓存目录，不删除其他环境的扩展缓存；多节点时应使用节点本地目录。
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/${USER}/torch-extensions-qwen35-cu126"
mkdir -p "$TORCH_EXTENSIONS_DIR"

python -m pip install 'torch==2.7.0+cu126' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.train-cu126.txt
DS_BUILD_OPS=0 python -m pip install --no-build-isolation 'deepspeed==0.15.0'
python -m pip check
python -m scripts.check_train_environment
```

`DS_BUILD_OPS=0` 只是不在安装阶段预编译全部扩展；CPUAdam 仍会在验证时 JIT 编译。
验证必须在分配到 GPU 的计算节点进行；成功输出 `PASS` 后再启动训练。
不要把系统 CUDA 或旧 cuDNN 的库路径强行放在 `LD_LIBRARY_PATH` 前面；若集群已有
此类设置，应按实际路径清理，保留集群通信所需的库路径。

## 可选：恢复线性注意力加速内核

基础配置不强制安装这些内核，Transformers 可以使用回退实现；长序列训练可能很慢。
如需与 `tmp.txt` 一样的 FLA / causal-conv1d，在上面的验证通过后安装：

```bash
python -m pip install -c requirements.train-cu126.txt \
  'fla-core==0.5.2' 'flash-linear-attention==0.5.2'
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-build-isolation \
  -c requirements.train-cu126.txt \
  'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@cd81f0413cad2fc1e6f17e785ac39f59aae690cd'
python -m pip check
python -c 'import fla, causal_conv1d; print("kernel imports OK")'
python -m scripts.check_train_environment
```

约束文件可防止安装可选包时悄悄替换 torch；若 pip 报依赖冲突，应保留错误信息，
不要通过 `--no-deps` 强行安装。这些可选内核的 GPU 前反向仍需用实际模型训练验证。
无需为当前 SDPA 配置另装 flash-attn，也无需照搬 `tmp.txt` 中的 torchao。

## 启动两卡训练

```bash
PYTHON="$CONDA_PREFIX/bin/python" CUDA_VISIBLE_DEVICES=0,1 bash scripts/22_beacon_train.sh \
  configs/train/beacon_qwen35_whole_searchr1.yaml
```

模型和数据路径仍需按目标服务器设置。`scripts/common.sh` 会优先查找默认旧环境
`search-comp-qwen3.5`，所以必须像上面一样指定 `PYTHON`，或设置
`CONDA_ENV=search-comp-qwen35-cu126`，避免激活新环境后启动脚本仍选择旧环境。具体多卡参数见
[multi_gpu_training.md](multi_gpu_training.md)。

## 原环境的最小修复

若不新建环境，可保留 `torch==2.7.0+cu126`，在集群切换到 CUDA 12.6 Toolkit：

```bash
# 把下面路径替换为真实安装路径；也可以先 module load 对应的 cuda/12.6。
export CUDA_HOME=/实际路径/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
unset DS_SKIP_CUDA_CHECK
python -m scripts.check_train_environment
```

这能修复本次版本检查，但不会清理原环境已混装的 NVIDIA 运行库。不要将
`DS_SKIP_CUDA_CHECK=1` 当作正式修复，它只跳过检查。另一条可行路线是把 torch
改为官方 cu128 wheel 并重装所有 CUDA 扩展；本方案选择 cu126 以减少变动。

## 依据

- [DeepSpeed 0.15.0 CUDA 检查源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.15.0/op_builder/builder.py)
- [PyTorch 官方历史版本安装命令](https://pytorch.org/get-started/previous-versions/#v270)
- [Transformers 5.9.0 依赖声明](https://github.com/huggingface/transformers/blob/v5.9.0/setup.py)
