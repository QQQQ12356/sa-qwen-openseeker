#!/usr/bin/env bash
# 修改下方参数后执行：bash scripts/build_openseeker_corpus.sh
set -euo pipefail

# ===== 运行环境：只读轨迹，不下载网页、不加载模型，不需要 GPU =====
PYTHON=""
CONDA_ENV="search-comp-qwen3.5"

# ===== 语料来源 =====
# 默认合并同一数据版本的三个 split，为训练集拟合诊断和测试共用固定证据库。
# 也可只保留 test.jsonl；必须在实验报告中说明来源划分。
# 使用其他服务器的 my_correct_4k 时，将下方三条路径一起修改。
INPUTS=(
  "outputs/data/openseeker/polar_correct_qwen35_16k/train.jsonl"
  "outputs/data/openseeker/polar_correct_qwen35_16k/validation.jsonl"
  "outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl"
)
OUTPUT_DIR="outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence"
PROGRESS_EVERY=100                    # 每读取多少条轨迹打印一次进度

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PY=$(resolve_python)
ARGS=()
for input in "${INPUTS[@]}"; do ARGS+=(--input "$input"); done

echo "[openseeker-corpus] Python: $PY"
"$PY" scripts/build_openseeker_corpus.py "${ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" --progress-every "$PROGRESS_EVERY"
echo "[openseeker-corpus] 测试脚本设置 BACKEND=\"corpus\""
echo "[openseeker-corpus] 测试脚本设置 CORPUS=\"$OUTPUT_DIR/corpus.jsonl\""
