#!/usr/bin/env bash
# 修改默认输入路径，或 bash scripts/render_dataset_html.sh /path/to/dataset [output.html]
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

INPUT="${1:-${DATASET_PATH:-outputs/data/openseeker/polar_correct_qwen35_16k}}"
OUTPUT="${2:-${OUTPUT_HTML:-outputs/reports/dataset.html}}"
SPLIT="${SPLIT:-train}"        # train / validation / test / rejected / all
LIMIT="${LIMIT:-0}"            # 0 为全部；大数据集可设为 100
OFFSET="${OFFSET:-0}"          # 跳过前 N 条，可分批导出

# 仅需 Python 3 标准库；不切换 conda 环境、不加载模型。
"${PYTHON:-python3}" scripts/render_dataset_html.py "$INPUT" \
  --output "$OUTPUT" --split "$SPLIT" --limit "$LIMIT" --offset "$OFFSET"
echo "[dataset-html] 已生成，下载 HTML 后可直接在浏览器中离线打开。"
