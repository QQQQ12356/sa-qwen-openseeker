#!/usr/bin/env bash
# 修改下方参数后执行：bash scripts/test_openseeker.sh
set -euo pipefail

# ===== 运行环境 =====
# 留空时按 common.sh 的规则选择 Python；也可填 Python 绝对路径。
PYTHON=""
CONDA_ENV="search-comp-qwen3.5"
export CUDA_VISIBLE_DEVICES="0"      # 多卡示例："0,1,2,3"
NUM_WORKERS=1                        # 每卡一个完整模型副本；多卡并行时设为使用的 GPU 数
PROGRESS_INTERVAL=15                # 长时间加载/生成时，每隔多少秒打印当前阶段
DRY_RUN=0                            # 1：只检查配置并打印命令，不加载模型或创建输出

# ===== 模型与数据 =====
RUN_DIR=""                           # train_openseeker.sh 生成的 outputs/experiments/<实验名>
MODEL_PATH=""                        # 留空：从 RUN_DIR 的 train_summary.json 读取最终模型
KIND=""                              # beacon / plain；留空：读取实验配置，无 RUN_DIR 时用 beacon
SPLIT="test"                         # train / validation / test；训练集评测用于判断拟合能力
DATA=""                              # 留空：按 SPLIT 选择数据；非空时覆盖自动选择的路径
# 单独测试 checkpoint 时填写 MODEL_PATH；LoRA 请使用已合并的模型目录。
# 同时填写 RUN_DIR 和 MODEL_PATH 时，后者覆盖实验中的模型路径。

# ===== 评测参数 =====
BACKEND="replay"                      # replay：历史动作复现诊断；http：在线工具；corpus：本地语料
ENDPOINT=""                          # http 必填，例如 http://127.0.0.1:8080/tool
CORPUS=""                            # corpus 必填：独立语料 JSONL 路径
LIMIT=100                            # 最多测试的样本数
MAX_CONTEXT=16384
MAX_NEW_TOKENS=8192
MAX_TOOL_TOKENS=16384
MAX_TURNS=32
OUTPUT_DIR=""                        # 留空：RUN_DIR/evaluation 或 outputs/results/openseeker

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PY=$(resolve_python)

if ! [[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_WORKERS 必须是正整数" >&2
  exit 2
fi
if [[ "$SPLIT" != train && "$SPLIT" != validation && "$SPLIT" != test ]]; then
  echo "SPLIT 必须是 train、validation 或 test" >&2
  exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN 必须是 0 或 1" >&2
  exit 2
fi
case "$BACKEND" in
  replay) ;;
  http) [[ -n "$ENDPOINT" ]] || { echo "http 后端必须设置 ENDPOINT" >&2; exit 2; } ;;
  corpus) [[ -f "$CORPUS" ]] || { echo "corpus 后端必须设置有效的 CORPUS 文件" >&2; exit 2; } ;;
  *) echo "BACKEND 必须是 replay、http 或 corpus" >&2; exit 2 ;;
esac

# 使用实验元数据定位最终模型和测试集，不修改实验快照。
RESOLVED=$("$PY" - "$RUN_DIR" "$MODEL_PATH" "$KIND" "$DATA" "$SPLIT" <<'PY'
import json
from pathlib import Path
import sys

run, model, kind, data, split = sys.argv[1:]
if run:
    root = Path(run)
    spec = json.loads((root / "experiment.json").read_text())
    if not model:
        summary = root / "models/model/train_summary.json"
        if not summary.is_file():
            sys.exit(f"找不到 {summary}；请先完成训练，或显式设置 MODEL_PATH")
        model = json.loads(summary.read_text())["final_model_path"]
    kind = kind or spec["kind"]
    data_key = {"train": "train_data_path", "validation": "val_data_path", "test": "test_data_path"}[split]
    data = data or spec["training_config"][data_key]
if not model:
    sys.exit("请在脚本顶部填写 RUN_DIR 或 MODEL_PATH")
kind = kind or "beacon"
data = data or f"outputs/data/openseeker/polar_correct_qwen35_16k/{split}.jsonl"
if kind not in ("plain", "beacon"):
    sys.exit("KIND 必须是 plain 或 beacon")
if not Path(data).is_file():
    sys.exit(f"测试集不存在：{data}")
print(model)
print(kind)
print(data)
PY
)
mapfile -t RESOLVED_ARGS <<< "$RESOLVED"
MODEL_PATH=${RESOLVED_ARGS[0]}
KIND=${RESOLVED_ARGS[1]}
DATA=${RESOLVED_ARGS[2]}

OUTPUT_DIR=${OUTPUT_DIR:-${RUN_DIR:+$RUN_DIR/evaluation}}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/results/openseeker}
STAMP=$(date -u '+%Y%m%dT%H%M%S%NZ')
OUTPUT="$OUTPUT_DIR/predictions_${SPLIT}_$STAMP.jsonl"
LOG_PATH="$OUTPUT_DIR/eval_${SPLIT}_$STAMP.log"
COMMAND=(
  "$PY" scripts/evaluate_openseeker.py
  --model "$MODEL_PATH" --kind "$KIND" --data "$DATA"
  --output "$OUTPUT" --backend "$BACKEND"
  --data-split "$SPLIT" --num-workers "$NUM_WORKERS"
  --progress-interval "$PROGRESS_INTERVAL"
  --limit "$LIMIT" --max-context "$MAX_CONTEXT"
  --max-new-tokens "$MAX_NEW_TOKENS" --max-tool-tokens "$MAX_TOOL_TOKENS"
  --max-turns "$MAX_TURNS"
)
if [[ "$BACKEND" == http ]]; then COMMAND+=(--endpoint "$ENDPOINT"); fi
if [[ "$BACKEND" == corpus ]]; then COMMAND+=(--corpus "$CORPUS"); fi

echo "[openseeker-test] Python: $PY"
echo "[openseeker-test] GPU: $CUDA_VISIBLE_DEVICES; workers: $NUM_WORKERS; split: $SPLIT; backend: $BACKEND"
printf '[openseeker-test] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi

mkdir -p "$OUTPUT_DIR"
run_logged "$LOG_PATH" "${COMMAND[@]}"
echo "[openseeker-test] 预测: $OUTPUT"
echo "[openseeker-test] 指标: ${OUTPUT%.jsonl}.metrics.json"
echo "[openseeker-test] 日志: $LOG_PATH"
