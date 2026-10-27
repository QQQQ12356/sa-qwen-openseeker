#!/usr/bin/env bash
# 一键剔除无观测正文轨迹；保留原数据，输出目录必须不存在。
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

INPUT="${1:-${DATASET_PATH:-outputs/data/openseeker/polar_correct_qwen35_6k}}"
OUTPUT="${2:-${OUTPUT_DIR:-${INPUT%/}_with_observations}}"
DATA_FORMAT="${DATA_FORMAT:-openseeker}"  # openseeker / searchr1

"${PYTHON:-python3}" scripts/filter_observation_dataset.py "$INPUT" \
  --output "$OUTPUT" --format "$DATA_FORMAT"
echo "[filter-observation] 原始文件未改动；剔除记录和统计见输出目录 removed/、filter_report.json。"
echo "[filter-observation] 请检查各 split 数量，并将训练脚本的 dataset 改为上述输出目录。"
