#!/usr/bin/env bash
# 修改下方参数后执行：bash scripts/prepare_openseeker.sh
set -euo pipefail

# ===== 运行环境：仅加载 tokenizer，不加载模型权重，不需要 GPU =====
PYTHON=""
CONDA_ENV="search-comp-qwen3.5"

# ===== 输入与输出：输出目录必须不存在 =====
SOURCE="polar"                         # polar / zephyr / combined
INPUTS=(
  "outputs/data/raw/polar/openseeker_v1_data.jsonl"
)
OUTPUT_DIR="outputs/data/openseeker/polar_correct_qwen35_16k_custom"
MODELS=("Qwen/Qwen3.5-4B")              # 与后续训练使用的 tokenizer 一致
MAX_LENGTH=16384
WORKERS=4
OFFLINE=1                              # 1：仅使用本地 tokenizer 缓存；0：允许下载
REQUIRE_CORRECT=1                      # 1：剔除 Incorrect；0：保留错误轨迹

# ===== 数据划分：训练集比例自动为 1 - VAL_FRACTION - TEST_FRACTION =====
# 默认 train / validation / test = 96% / 2% / 2%。
# 例如 80% / 10% / 10%：将下方两个值都改为 0.10。
# 两个值必须非负且和小于 1；哈希划分的实际条数是近似比例。
# 输入文件名为 val/validation/test 时，已有题目优先保留原划分。
VAL_FRACTION=0.02
TEST_FRACTION=0.02
SEED=42                                # 固定 seed 和输入可复现划分

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PY=$(resolve_python)
ARGS=()
for input in "${INPUTS[@]}"; do ARGS+=(--input "$input"); done
if [[ "$OFFLINE" == 1 ]]; then ARGS+=(--offline); fi
if [[ "$REQUIRE_CORRECT" == 1 ]]; then
  ARGS+=(--require-correct)
else
  ARGS+=(--no-require-correct)
fi

echo "[openseeker-prepare] Python: $PY"
echo "[openseeker-prepare] validation=$VAL_FRACTION test=$TEST_FRACTION；train 为剩余部分"
"$PY" scripts/prepare_openseeker.py "${ARGS[@]}" \
  --source "$SOURCE" --output "$OUTPUT_DIR" --models "${MODELS[@]}" \
  --max-length "$MAX_LENGTH" --workers "$WORKERS" \
  --val-fraction "$VAL_FRACTION" --test-fraction "$TEST_FRACTION" --seed "$SEED"
