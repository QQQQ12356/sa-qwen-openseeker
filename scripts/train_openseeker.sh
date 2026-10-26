#!/usr/bin/env bash
# 修改下方参数后执行：bash scripts/train_openseeker.sh
set -euo pipefail

# ===== 运行环境 =====
# 留空时按 common.sh 的规则选择 Python；也可填当前服务器的 Python 绝对路径。
PYTHON=""
CONDA_ENV="search-comp-qwen3.5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACTION="train"                       # train：生成新实验并训练；plan：只生成配置

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PY=$(resolve_python)
if [[ "$ACTION" != "train" && "$ACTION" != "plan" ]]; then
  echo "ACTION 必须是 train 或 plan" >&2
  exit 2
fi

# 临时配置会归档到实验目录的 source_config.yaml；结束后删除临时文件。
CONFIG_FILE=$(mktemp "${TMPDIR:-/tmp}/openseeker-train.XXXXXX")
trap 'rm -f -- "$CONFIG_FILE"' EXIT

# ===== 实验参数：直接修改下面的 YAML =====
# 路径相对于项目根目录，也可以使用绝对路径；与 configs/openseeker.yaml 无需同步。
cat > "$CONFIG_FILE" <<'YAML'
model_family: qwen35                 # qwen3 / qwen35
kind: beacon                         # beacon / plain
data_mode: openseeker
dataset: outputs/data/openseeker/polar_correct_qwen35_16k
models:
  # 可改为本地模型路径；必须与数据 manifest.json 中记录的 tokenizer 标识一致。
  qwen3: Qwen/Qwen3-4B-Thinking-2507
  qwen35: Qwen/Qwen3.5-4B
base_configs:
  qwen3:
    beacon: configs/train/beacon_qwen3_searchr1.yaml
    plain: configs/train/qwen3_plain_sft.yaml
  qwen35:
    beacon: configs/train/beacon_qwen35_whole_searchr1.yaml
    plain: configs/train/qwen35_plain_sft.yaml
training:
  max_length: 16384                  # 数据需提前 prepare；不在训练时截断
  learning_rate: 0.00005
  seed: 42
  num_epochs: 1
  max_train_steps: null              # null 按 epoch；正整数按优化器步数
  grad_accum_steps: 4
  per_device_batch_size: 1
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  optim: adamw_bnb_8bit
  use_bf16: true
  logging_steps: 5
  log_freq_steps: 5
  save_freq_steps: 200
  eval_steps: 200
  length_diagnostics_samples: 128
beacon:
  beacon_ratio: 16
  beacon_information_mode: whole
  beacon_window_enabled: false
  beacon_doc_region_split: false
  beacon_question_memory_v1: false
  # 逐 token 相对权重；{} 恢复全部为 1。以下为可调整的起始值。
  loss_segments:
    think_tag: 1.0
    think_content: 1.0
    tool_call_tag: 1.0
    tool_call_content: 1.0
    answer_tag: 1.0
    answer_content: 0.2
    final_answer_tag: 2.0
    final_answer_content: 10.0
    other: 1.0
  beacon_ce_loss_weight: 1.0
  beacon_continuation_loss_weight: 0.1
  beacon_information_loss: false
  beacon_output_distill_weight: 0.0  # >0 会额外加载 teacher
  beacon_output_distill_strategy: all
  beacon_readout_distill_weight: 0.0
  beacon_loss_chunk_size: 32
  beacon_checkpoint_loss: true
  beacon_cpu_offload_activations: true
evaluation:                         # 归档评测参数；本脚本只训练，不自动评测
  backend: replay
  corpus: null
  endpoint: null
  limit: 100
  max_context: 16384
  max_new_tokens: 8192
  max_tool_tokens: 16384
  max_turns: 32
YAML

echo "[openseeker] Python: $PY"
echo "[openseeker] GPU: $CUDA_VISIBLE_DEVICES; action: $ACTION"
# experiment.py 自动创建新实验，保存最终 train.yaml，并记录训练日志和退出码。
# 无需手动 plan 或复制 run_dir。
"$PY" scripts/experiment.py "$ACTION" --config "$CONFIG_FILE"
