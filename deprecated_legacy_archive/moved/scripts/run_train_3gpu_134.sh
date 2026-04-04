#!/bin/bash
# -*- coding: utf-8 -*-
# PENCI 3-GPU DDP 训练启动脚本 (node03: GPU 1,3,4, RTX 4090D)
#
# 日志文件说明:
#   terminal_<timestamp>.log  — torchrun 所有进程的原始 stdout/stderr
#   train_<timestamp>.log     — rank 0 的结构化训练日志 (由 train.py 写入)
#
# 用法:
#   bash scripts/run_train_3gpu_134.sh
#   bash scripts/run_train_3gpu_134.sh --config configs/my_config.yaml
#   bash scripts/run_train_3gpu_134.sh --resume outputs/xxx/checkpoints/best_model.pt
#   PREFETCH_WARMUP_GB=64 bash scripts/run_train_3gpu_134.sh
#   bash scripts/run_train_3gpu_134.sh --io_prefetch_warmup_gb 64

set -uo pipefail

# ─── 参数透传 ──────────────────────────────────────────────────────────────
# 所有传给本脚本的参数都原样转发给 train.py
EXTRA_ARGS=("$@")

# ─── 路径设置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/train_3gpu_134_safe.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_ROOT}/outputs/train_3gpu_134_${TIMESTAMP}"
TERMINAL_LOG="${OUTPUT_DIR}/terminal_${TIMESTAMP}.log"
PREFETCH_WARMUP_GB="${PREFETCH_WARMUP_GB:-80}"
CONFIG_PATH="${DEFAULT_CONFIG}"

for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
        --config)
            if ((i + 1 < ${#EXTRA_ARGS[@]})); then
                CONFIG_PATH="${EXTRA_ARGS[$((i + 1))]}"
            fi
            ;;
        --config=*)
            CONFIG_PATH="${EXTRA_ARGS[$i]#--config=}"
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

# ─── 打印启动信息 ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  PENCI 3-GPU DDP 训练 (node03 RTX 4090D: GPU 1,3,4)"
echo "  时间戳    : ${TIMESTAMP}"
echo "  输出目录  : ${OUTPUT_DIR}"
echo "  终端日志  : ${TERMINAL_LOG}"
echo "  训练日志  : ${OUTPUT_DIR}/train_${TIMESTAMP}.log  (由 train.py 写入)"
echo "  配置文件  : ${CONFIG_PATH}"
echo "  GPU       : CUDA_VISIBLE_DEVICES=1,3,4"
echo "  预读窗口  : ${PREFETCH_WARMUP_GB} GiB (可用 PREFETCH_WARMUP_GB 覆盖)"
echo "============================================================"
echo ""
echo "实时查看终端日志:"
echo "  tail -f ${TERMINAL_LOG}"
echo ""
echo "实时查看训练日志:"
echo "  tail -f ${OUTPUT_DIR}/train_${TIMESTAMP}.log"
echo ""

# ─── 启动训练 ──────────────────────────────────────────────────────────────
TORCHRUN="${TORCHRUN:-torchrun}"
if ! command -v "${TORCHRUN}" &>/dev/null; then
    TORCHRUN="/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/torchrun"
fi

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=1,3,4 \
"${TORCHRUN}" \
    --nproc_per_node=3 \
    --master_port=29504 \
    "${SCRIPT_DIR}/train.py" \
    --config "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --io_prefetch_warmup_gb "${PREFETCH_WARMUP_GB}" \
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
