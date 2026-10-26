#!/usr/bin/env bash
# 修改顶部参数后直接运行本脚本，无需填写命令行参数。
set -euo pipefail

# ===== 运行环境：仅需 Python 3 标准库，不加载模型，不需要 GPU =====
PYTHON=""                            # 可填写 Python 的绝对路径
CONDA_ENV="search-comp-qwen3.5"

# ===== 输入：可填写多个 JSONL；相对路径以项目根目录为准 =====
INPUTS=(
  "outputs/data/tmp/sa-qwen3-v1-16-intersect-test_predictions-100qa.jsonl"
  # "outputs/experiments/你的实验/evaluation/predictions_test_时间戳.jsonl"
)

# ===== 输出：每个输入生成一个同名 .html =====
OUTPUT_DIR="outputs/reports"          # 留空：保存到各输入文件旁边
TITLE=""                             # 留空：使用输入文件名；非空：所有报告使用此标题
TEMPLATE="scripts/templates/predictions.html"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PY=$(resolve_python)

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "请在脚本顶部 INPUTS 中填写至少一个 JSONL 文件。" >&2
  exit 2
fi
for input in "${INPUTS[@]}"; do
  if [[ ! -f "$input" ]]; then
    echo "输入文件不存在：$input；请修改脚本顶部 INPUTS。" >&2
    exit 2
  fi
done

ARGS=("${INPUTS[@]}" --template "$TEMPLATE")
if [[ -n "$OUTPUT_DIR" ]]; then ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "$TITLE" ]]; then ARGS+=(--title "$TITLE"); fi

echo "[predictions-html] Python: $PY"
"$PY" scripts/render_predictions_html.py "${ARGS[@]}"
echo "[predictions-html] 已生成 HTML；下载到本机后可直接用浏览器打开。"
