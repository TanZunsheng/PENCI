#!/bin/bash
# -*- coding: utf-8 -*-
# PENCI 8-GPU DDP 训练启动脚本（8x RTX 4090D，稳态/NFS 友好版）
#
# 默认策略:
#   - 每卡 batch_size=12（由 configs/train_8gpu_12_safe.yaml 控制）
#   - AMP 关闭，优先验证稳定性
#   - 启用文件级调度 + HDF5 page cache 预热，降低训练时段 NFS 抖动
#
# 用法:
#   bash scripts/run_train_8gpu.sh
#   bash scripts/run_train_8gpu.sh --config configs/my_config.yaml
#   bash scripts/run_train_8gpu.sh --resume outputs/xxx/checkpoints/best_model.pt
#   PREFETCH_WARMUP_GB=160 bash scripts/run_train_8gpu.sh
#   GPU_LIST=0,1,2,3,4,5,6,7 MASTER_PORT=29508 bash scripts/run_train_8gpu.sh

set -euo pipefail

# ─── 参数透传 ──────────────────────────────────────────────────────────────
RAW_ARGS=("$@")
PASSTHROUGH_ARGS=()

# ─── 路径与默认参数 ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/train_8gpu_12_safe.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_PATH="${DEFAULT_CONFIG}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/train_8gpu_bs12_${TIMESTAMP}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29508}"
PREFETCH_WARMUP_GB="${PREFETCH_WARMUP_GB:-120}"

for ((i = 0; i < ${#RAW_ARGS[@]}; i++)); do
    case "${RAW_ARGS[$i]}" in
        --config)
            if ((i + 1 < ${#RAW_ARGS[@]})); then
                CONFIG_PATH="${RAW_ARGS[$((i + 1))]}"
                i=$((i + 1))
            fi
            ;;
        --config=*)
            CONFIG_PATH="${RAW_ARGS[$i]#--config=}"
            ;;
        --output_dir)
            if ((i + 1 < ${#RAW_ARGS[@]})); then
                OUTPUT_DIR="${RAW_ARGS[$((i + 1))]}"
                i=$((i + 1))
            fi
            ;;
        --output_dir=*)
            OUTPUT_DIR="${RAW_ARGS[$i]#--output_dir=}"
            ;;
        *)
            PASSTHROUGH_ARGS+=("${RAW_ARGS[$i]}")
            ;;
    esac
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARRAY[@]}
TERMINAL_LOG="${OUTPUT_DIR}/terminal_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${TERMINAL_LOG}")"

# ─── 打印启动信息 ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  PENCI 8-GPU DDP 训练 (RTX 4090D 稳态版)"
echo "  时间戳    : ${TIMESTAMP}"
echo "  输出目录  : ${OUTPUT_DIR}"
echo "  终端日志  : ${TERMINAL_LOG}"
echo "  训练日志  : ${OUTPUT_DIR}/train_*.log  (由 train.py 写入)"
echo "  预热细节  : ${OUTPUT_DIR}/logs/prefetch_detail_*.log"
echo "  配置文件  : ${CONFIG_PATH}"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${GPU_LIST}"
echo "  进程数    : ${NUM_GPUS}"
echo "  端口      : ${MASTER_PORT}"
echo "  预读窗口  : ${PREFETCH_WARMUP_GB} GiB (可用 PREFETCH_WARMUP_GB 覆盖)"
echo "============================================================"
echo ""
echo "实时查看终端日志:"
echo "  tail -f ${TERMINAL_LOG}"
echo ""
echo "实时查看训练日志:"
echo "  tail -f ${OUTPUT_DIR}/train_*.log"
echo ""
echo "实时查看预热细节日志:"
echo "  tail -f ${OUTPUT_DIR}/logs/prefetch_detail_*.log"
echo ""

# ─── 启动训练 ──────────────────────────────────────────────────────────────
TORCHRUN="${TORCHRUN:-torchrun}"
if ! command -v "${TORCHRUN}" &>/dev/null; then
    TORCHRUN="/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/torchrun"
fi

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${TORCHRUN}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --config "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --io_prefetch_warmup_gb "${PREFETCH_WARMUP_GB}" \
    --ddp_mode prod \
    "${PASSTHROUGH_ARGS[@]}" \
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
