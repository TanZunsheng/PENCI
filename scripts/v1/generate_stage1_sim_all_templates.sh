#!/usr/bin/env bash
set -euo pipefail

# 顺序生成 Stage1 仿真数据，覆盖仿真专用 archive 中的全部真实模板。
#
# 默认策略：
# - 从 sim_leadfield_cache/fingerprint_registry.pt 读取当前可用模板
# - 按 n_channels 聚合，因为当前 sim_pretrain 仍是固定通道数数据集
# - 每个通道数单独生成一个数据集目录
# - 样本数按“每模板多少 train / val”自动乘以该通道数下模板个数
#
# 可通过环境变量覆盖：
#   PYTHON_BIN
#   REGISTRY_PATH
#   OUTPUT_ROOT
#   TRAIN_PER_TEMPLATE
#   VAL_PER_TEMPLATE
#   TIME_STEPS
#   PRESET
#   PRINT_EVERY
#   DRY_RUN=1   # 仅打印命令，不实际执行

PYTHON_BIN="${PYTHON_BIN:-/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
REGISTRY_PATH="${REGISTRY_PATH:-/work/2024/tanzunsheng/sim_leadfield_cache/fingerprint_registry.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work/2024/tanzunsheng/PENCI_sim_data/stage1_all_templates}"
TRAIN_PER_TEMPLATE="${TRAIN_PER_TEMPLATE:-4000}"
VAL_PER_TEMPLATE="${VAL_PER_TEMPLATE:-800}"
TIME_STEPS="${TIME_STEPS:-512}"
PRESET="${PRESET:-formal}"
PRINT_EVERY="${PRINT_EVERY:-500}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERATOR_SCRIPT="${REPO_ROOT}/scripts/v1/generate_stage1_sim_data.py"

if [[ ! -f "${REGISTRY_PATH}" ]]; then
  echo "registry 不存在: ${REGISTRY_PATH}" >&2
  exit 1
fi

if [[ ! -f "${GENERATOR_SCRIPT}" ]]; then
  echo "生成脚本不存在: ${GENERATOR_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

mapfile -t CHANNEL_LAYOUT_COUNTS < <(
  env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" "${PYTHON_BIN}" -u - <<'PY' "${REGISTRY_PATH}"
import sys
from collections import Counter

import torch

archive_path = sys.argv[1]
archive = torch.load(archive_path, map_location="cpu", weights_only=False)
configs = archive.get("configs", {})
counts = Counter(len(entry["channel_names"]) for entry in configs.values())
for n_channels in sorted(counts):
    print(f"{n_channels} {counts[n_channels]}")
PY
)

if [[ "${#CHANNEL_LAYOUT_COUNTS[@]}" -eq 0 ]]; then
  echo "archive 中没有可用模板: ${REGISTRY_PATH}" >&2
  exit 1
fi

echo "使用 registry: ${REGISTRY_PATH}"
echo "输出根目录: ${OUTPUT_ROOT}"
echo "每模板样本数: train=${TRAIN_PER_TEMPLATE}, val=${VAL_PER_TEMPLATE}"
echo "time_steps: ${TIME_STEPS}"
echo "preset: ${PRESET}"
echo

for line in "${CHANNEL_LAYOUT_COUNTS[@]}"; do
  read -r N_CHANNELS TEMPLATE_COUNT <<<"${line}"
  N_TRAIN=$(( TEMPLATE_COUNT * TRAIN_PER_TEMPLATE ))
  N_VAL=$(( TEMPLATE_COUNT * VAL_PER_TEMPLATE ))
  OUTPUT_DIR="${OUTPUT_ROOT}/${N_CHANNELS}ch"

  CMD=(
    env "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
    "${PYTHON_BIN}"
    -u
    "${GENERATOR_SCRIPT}"
    --output_dir "${OUTPUT_DIR}"
    --preset "${PRESET}"
    --n_train "${N_TRAIN}"
    --n_val "${N_VAL}"
    --n_sensors "${N_CHANNELS}"
    --time_steps "${TIME_STEPS}"
    --layout_mode real_only
    --registry_path "${REGISTRY_PATH}"
    --external_template_roots ""
    --print_every "${PRINT_EVERY}"
  )

  echo "============================================================"
  echo "生成 ${N_CHANNELS}ch 数据集"
  echo "模板数: ${TEMPLATE_COUNT}"
  echo "train=${N_TRAIN}, val=${N_VAL}"
  echo "输出目录: ${OUTPUT_DIR}"
  echo "命令:"
  printf '  %q' "${CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] 跳过实际执行"
    continue
  fi

  "${CMD[@]}"
done

echo
echo "全部生成完成。输出位于: ${OUTPUT_ROOT}"
