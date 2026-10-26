# Beacon / plain 多卡训练

默认 Qwen3.5 Beacon 基础配置为 `configs/train/beacon_qwen35_whole_searchr1.yaml`，plain 为 `configs/train/qwen35_plain_sft.yaml`。现有单卡命令保持可用。

在训练用的 Python 环境中安装 DeepSpeed：

```bash
python -m pip install 'deepspeed>=0.15.0'
```

下面是单机两卡训练；四卡可改为 `CUDA_VISIBLE_DEVICES=0,1,2,3`。进程数默认等于显式列出的可见 GPU 数，也可设置 `NPROC_PER_NODE=2`。

```bash
# Beacon：whole 配置，默认 ZeRO-2 + CPU 优化器卸载
CUDA_VISIBLE_DEVICES=0,1 bash scripts/22_beacon_train.sh

# plain：默认 ZeRO-2 + CPU 优化器卸载
CUDA_VISIBLE_DEVICES=0,1 bash scripts/20_plain_train.sh

# OpenSeeker：修改脚本内 kind: beacon / plain；复用同一启动逻辑
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_openseeker.sh

# plain 可进一步分片参数：ZeRO-3 + CPU 参数和优化器卸载
CUDA_VISIBLE_DEVICES=0,1 DISTRIBUTED_BACKEND=zero3 bash scripts/20_plain_train.sh

# 只使用 DDP，无需 DeepSpeed
CUDA_VISIBLE_DEVICES=0,1 DISTRIBUTED_BACKEND=ddp bash scripts/22_beacon_train.sh
```

也可以直接启动训练器，配置 `deepspeed` 后 Trainer 会接管模型放置；不要使用 `device_map=auto` 将训练模型跨卡派发：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module search_comp.trainer.beacon_trainer \
  --config configs/train/beacon_qwen35_whole_searchr1.yaml \
  --set deepspeed=configs/train/ds_zero2.json
```

`DISTRIBUTED_BACKEND` 接受 `zero2`（多卡默认）、`zero3`（仅 plain）、`ddp`。启动器选择的 DeepSpeed 配置优先于传入的同名覆盖参数。单卡时如需 DeepSpeed，可显式传 `--set deepspeed=configs/train/ds_zero2.json`。它使用 CPU AdamW 优化器，替代基础配置中的 bitsandbytes 优化器；需要足够的 CPU 内存和 CPU/GPU 传输带宽。

## 显存边界

- DDP 每卡保留完整模型、梯度和优化器状态，主要增加吞吐，不降低单条样本的激活峰值。
- ZeRO-2 分片梯度、优化器状态，并将优化器卸载到 CPU；每卡仍保留完整模型参数。
- plain 的 ZeRO-3 进一步分片参数，并启用 CPU 参数卸载。加载先在 CPU 完成，再由 DeepSpeed 分片，所以启动时需要容纳完整模型的主机内存。
- Beacon 当前有随样本变化的 keep/compress 分段和投影调用。不同 rank 的模块调用次数不一致，尚未支持 ZeRO-3；入口会明确拒绝，以免集合通信顺序不一致导致挂起。
- whole Beacon 配置启用 `beacon_attention_backend: sdpa`，用 PyTorch SDPA 替代手写 attention score/softmax，避免代码主动生成 `[batch, heads, query_length, key_length]` 矩阵。实际 GPU 内核由 PyTorch 根据硬件、dtype 和 mask 选择；若回退到数学实现，显存收益可能受限。`eager` 可用于对照。
- 多卡或 DeepSpeed 下，`distill_teacher_device: auto` 解析为 CPU。teacher 不参与 ZeRO 分片，每个 rank 有一份 CPU teacher。输出蒸馏的目标 logits 仍会传到 student GPU；`strategy: all` 的大词表目标仍有显存成本。关闭蒸馏会改变训练目标，应按实验需要选择。
- 单条样本激活仍由一张卡承担，这里未实现张量并行或序列并行。因此不能保证任意长度都不 OOM。

有效 batch size 为 `GPU 数 × per_device_batch_size × grad_accum_steps`；Beacon 每卡 batch 固定为 1。比如原来单卡累积 4 次，双卡若要保持相同有效 batch，可设置 `--set grad_accum_steps=2`。

## 日志与保存

每个进程绑定自己的 GPU，Trainer 分片数据。仅全局 rank 0 写入共享 JSON、指标、tokenizer 和最终合并模型；Beacon 的 adapter checkpoint 同时包含 LoRA 与全部 Beacon 参数。训练器保存最终配置到 `models/<exp_name>/resolved_config.json`，其中包含启动器注入的 DeepSpeed 配置；实验入口生成的 `train.yaml` 位于实验目录。

## 验证

CPU 单元测试覆盖 SDPA 与 eager 的 loss、梯度和压缩缓存，以及 adapter 保存内容、启动参数选择。两进程 CPU 冒烟测试覆盖 DDP 更新、各 rank 参数同步和 checkpoint 保存：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/helpers/distributed_training_worker.py \
  /tmp/beacon-ddp-smoke beacon
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/helpers/distributed_training_worker.py \
  /tmp/plain-ddp-smoke plain
```

DeepSpeed GPU 训练与实际显存下降需要在目标服务器验证。配置依据：[Transformers DeepSpeed 文档](https://huggingface.co/docs/transformers/deepspeed)。
