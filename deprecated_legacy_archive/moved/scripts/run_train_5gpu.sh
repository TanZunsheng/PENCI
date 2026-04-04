#!/bin/bash
# -*- coding: utf-8 -*-
# PENCI 5-GPU DDP 训练启动脚本 (GPU 0-4, A6000)
#
# 日志文件说明:
#   terminal_<timestamp>.log  — torchrun 所有进程的原始 stdout/stderr
#   train_<timestamp>.log     — rank 0 的结构化训练日志 (由 train.py 写入)
#
# 用法:
#   bash scripts/run_train_5gpu.sh
#   bash scripts/run_train_5gpu.sh --config configs/my_config.yaml
#   bash scripts/run_train_5gpu.sh --resume outputs/xxx/checkpoints/best_model.pt

set -uo pipefail

# ─── 参数透传 ──────────────────────────────────────────────────────────────
# 所有传给本脚本的参数都原样转发给 train.py
EXTRA_ARGS=("$@")

# ─── 路径设置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_ROOT}/outputs/train_5gpu_${TIMESTAMP}"
TERMINAL_LOG="${OUTPUT_DIR}/terminal_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"

# ─── 打印启动信息 ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  PENCI 5-GPU DDP 训练"
echo "  时间戳    : ${TIMESTAMP}"
echo "  输出目录  : ${OUTPUT_DIR}"
echo "  终端日志  : ${TERMINAL_LOG}"
echo "  训练日志  : ${OUTPUT_DIR}/train_${TIMESTAMP}.log  (由 train.py 写入)"
echo "  GPU       : CUDA_VISIBLE_DEVICES=0,1,2,3,4"
echo "============================================================"
echo ""
echo "实时查看终端日志:"
echo "  tail -f ${TERMINAL_LOG}"
echo ""
echo "实时查看训练日志:"
echo "  tail -f ${OUTPUT_DIR}/train_${TIMESTAMP}.log"
echo ""

# ─── 启动训练 ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
torchrun \
    --nproc_per_node=5 \
    --master_port=29500 \
    "${SCRIPT_DIR}/train.py" \
    --config "${PROJECT_ROOT}/configs/default.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --ddp_mode prod \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${TERMINAL_LOG}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "  训练完成 ✓"
else
    echo "  训练异常退出，exit_code=${EXIT_CODE}"
fi
echo "  终端日志: ${TERMINAL_LOG}"
echo "  训练日志: ${OUTPUT_DIR}/train_*.log"
echo "  检查点  : ${OUTPUT_DIR}/checkpoints/"
echo "============================================================"

exit "${EXIT_CODE}"
