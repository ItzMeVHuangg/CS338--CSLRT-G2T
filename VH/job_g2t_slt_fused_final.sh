#!/bin/bash
# ==============================================================================
# SLURM Job -- Final experimental fused SLT
#
# Pipeline:
#   Swin-T BiLSTM CTC
#   Swin-T Transformer CTC
#   ResNet34 BiLSTM CTC
#        -> confidence-weighted pseudo-gloss fusion
#        -> confidence-aware mBART G2T
#        -> confidence-aware latent CVAE G2T
#
# Usage:
#   sbatch job_g2t_slt_fused_final.sh full
#   sbatch job_g2t_slt_fused_final.sh export
#   sbatch job_g2t_slt_fused_final.sh fuse
#   sbatch job_g2t_slt_fused_final.sh train_mbart
#   sbatch job_g2t_slt_fused_final.sh train_cvae
#   sbatch job_g2t_slt_fused_final.sh export_smoke
# ==============================================================================

#SBATCH --job-name=g2t_fused
#SBATCH --chdir=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
#SBATCH --output=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/output/g2t_fused_%j.out
#SBATCH --error=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/output/g2t_fused_%j.err
#SBATCH --gres=mps:a100:2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00

set -euo pipefail

MODE="${1:-full}"
REQUIRED_VRAM="${REQUIRED_VRAM:-18000}"
CLUSTER_PROJECT_DIR="/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg"
SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${PROJECT_DIR:-}" ]; then
    if [ -d "${CLUSTER_PROJECT_DIR}" ]; then
        PROJECT_DIR="${CLUSTER_PROJECT_DIR}"
    else
        PROJECT_DIR="${SCRIPT_PROJECT_DIR}"
    fi
fi
cd "${PROJECT_DIR}"
mkdir -p output outputs checkpoints_g2t_slt

if command -v module >/dev/null 2>&1; then
    module clear -f || true
    module load shared python312 || true
fi

if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
unset CUDA_VISIBLE_DEVICES

set +e
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "${REQUIRED_VRAM}" "${SLURM_JOB_ID}")
EXIT_CODE=$?
set -e
if [ "${EXIT_CODE}" -eq 10 ]; then
    echo "${CHECK_OUT}"
    exit 0
elif [ "${EXIT_CODE}" -eq 11 ]; then
    echo "${CHECK_OUT}"
    exit 1
fi
BEST_GPU="${CHECK_OUT}"
echo "Job ${SLURM_JOB_ID} running on GPU: ${BEST_GPU}"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job${SLURM_JOB_ID}
rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
export CUDA_VISIBLE_DEVICES="${BEST_GPU}"

PYTHON_BIN="${PYTHON_BIN:-python}"

SWIN_BILSTM_CONFIG="${SWIN_BILSTM_CONFIG:-${PROJECT_DIR}/configs/config.yaml}"
SWIN_TRANSFORMER_CONFIG="${SWIN_TRANSFORMER_CONFIG:-${PROJECT_DIR}/configs/config.yaml}"
RESNET_CONFIG="${RESNET_CONFIG:-${PROJECT_DIR}/configs/config_resnet34.yaml}"

SWIN_BILSTM_CKPT="${SWIN_BILSTM_CKPT:-${PROJECT_DIR}/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth}"
SWIN_TRANSFORMER_CKPT="${SWIN_TRANSFORMER_CKPT:-${PROJECT_DIR}/checkpoints/ablation/Variant_H_path/checkpoint/checkpoint_75.pth}"
RESNET34_CKPT="${RESNET34_CKPT:-/datastore/cndt_phungdtm/KLTN_HoangBinh/MBinh/checkpoints/ablation/Variant_A_path/best_path/best_path_12.pth}"

MBART_FUSED_CONFIG="${MBART_FUSED_CONFIG:-${PROJECT_DIR}/configs/config_g2t_mbart_fused.yaml}"
CVAE_FUSED_CONFIG="${CVAE_FUSED_CONFIG:-${PROJECT_DIR}/configs/config_g2t_cvae_fused.yaml}"

PHOENIX_ROOT="${PHOENIX_ROOT:-}"
FRAMES_ROOT="${FRAMES_ROOT:-}"
ANNOT_ROOT="${ANNOT_ROOT:-}"
EXPORT_MAX_SAMPLES="${EXPORT_MAX_SAMPLES:-}"

SWIN_BILSTM_DEV="${SWIN_BILSTM_DEV:-${PROJECT_DIR}/outputs/fused_sources/pred_gloss_swin_bilstm_dev.jsonl}"
SWIN_BILSTM_TEST="${SWIN_BILSTM_TEST:-${PROJECT_DIR}/outputs/fused_sources/pred_gloss_swin_bilstm_test.jsonl}"
SWIN_TRANSFORMER_DEV="${SWIN_TRANSFORMER_DEV:-${PROJECT_DIR}/outputs/fused_sources/pred_gloss_swin_transformer_dev.jsonl}"
SWIN_TRANSFORMER_TEST="${SWIN_TRANSFORMER_TEST:-${PROJECT_DIR}/outputs/fused_sources/pred_gloss_swin_transformer_test.jsonl}"
RESNET34_DEV="${RESNET34_DEV:-${PROJECT_DIR}/outputs/fused_sources/pred_gloss_resnet34_dev.jsonl}"
RESNET34_TEST="${RESNET34_TEST:-${PROJECT_DIR}/outputs/fused_sources/pred_gloss_resnet34_test.jsonl}"

FUSED_DEV="${FUSED_DEV:-${PROJECT_DIR}/outputs/pred_gloss_fused_3cslr_dev.jsonl}"
FUSED_TEST="${FUSED_TEST:-${PROJECT_DIR}/outputs/pred_gloss_fused_3cslr_test.jsonl}"

RESNET_WEIGHT="${RESNET_WEIGHT:-1.20}"
SWIN_BILSTM_WEIGHT="${SWIN_BILSTM_WEIGHT:-1.00}"
SWIN_TRANSFORMER_WEIGHT="${SWIN_TRANSFORMER_WEIGHT:-0.90}"

echo "========================================================"
echo " Final fused G2T SLT"
echo " Mode: ${MODE}"
echo " Project: ${PROJECT_DIR}"
echo " Swin BiLSTM ckpt      : ${SWIN_BILSTM_CKPT}"
echo " Swin Transformer ckpt : ${SWIN_TRANSFORMER_CKPT}"
echo " ResNet34 ckpt         : ${RESNET34_CKPT}"
echo " Fused dev/test        : ${FUSED_DEV} | ${FUSED_TEST}"
echo "========================================================"

common_path_args=()
if [ -n "${PHOENIX_ROOT}" ]; then
    common_path_args+=(--phoenix_root "${PHOENIX_ROOT}")
fi
if [ -n "${FRAMES_ROOT}" ]; then
    common_path_args+=(--frames_root "${FRAMES_ROOT}")
fi
if [ -n "${ANNOT_ROOT}" ]; then
    common_path_args+=(--annot_root "${ANNOT_ROOT}")
fi
if [ -n "${EXPORT_MAX_SAMPLES}" ]; then
    common_path_args+=(--max_samples "${EXPORT_MAX_SAMPLES}")
fi

run_export() {
    echo "[Export] Swin-T BiLSTM dev/test"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        --config "${SWIN_BILSTM_CONFIG}" \
        --cslr_ckpt "${SWIN_BILSTM_CKPT}" \
        --encoder swin \
        --seq_model auto \
        --builder auto \
        --batch_size 1 \
        "${common_path_args[@]}" \
        --split dev \
        --output "${SWIN_BILSTM_DEV}"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        --config "${SWIN_BILSTM_CONFIG}" \
        --cslr_ckpt "${SWIN_BILSTM_CKPT}" \
        --encoder swin \
        --seq_model auto \
        --builder auto \
        --batch_size 1 \
        "${common_path_args[@]}" \
        --split test \
        --output "${SWIN_BILSTM_TEST}"

    echo "[Export] Swin-T Transformer dev/test"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        --config "${SWIN_TRANSFORMER_CONFIG}" \
        --cslr_ckpt "${SWIN_TRANSFORMER_CKPT}" \
        --encoder swin \
        --seq_model transformer \
        --builder ablation \
        --batch_size 1 \
        "${common_path_args[@]}" \
        --split dev \
        --output "${SWIN_TRANSFORMER_DEV}"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        --config "${SWIN_TRANSFORMER_CONFIG}" \
        --cslr_ckpt "${SWIN_TRANSFORMER_CKPT}" \
        --encoder swin \
        --seq_model transformer \
        --builder ablation \
        --batch_size 1 \
        "${common_path_args[@]}" \
        --split test \
        --output "${SWIN_TRANSFORMER_TEST}"

    echo "[Export] ResNet34 BiLSTM dev/test"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        --config "${RESNET_CONFIG}" \
        --cslr_ckpt "${RESNET34_CKPT}" \
        --encoder resnet34 \
        --batch_size 1 \
        "${common_path_args[@]}" \
        --split dev \
        --output "${RESNET34_DEV}"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        --config "${RESNET_CONFIG}" \
        --cslr_ckpt "${RESNET34_CKPT}" \
        --encoder resnet34 \
        --batch_size 1 \
        "${common_path_args[@]}" \
        --split test \
        --output "${RESNET34_TEST}"
}

run_fuse() {
    echo "[Fuse] dev"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/fuse_pred_gloss.py" \
        --source resnet34 "${RESNET34_DEV}" "${RESNET_WEIGHT}" \
        --source swin_bilstm "${SWIN_BILSTM_DEV}" "${SWIN_BILSTM_WEIGHT}" \
        --source swin_transformer "${SWIN_TRANSFORMER_DEV}" "${SWIN_TRANSFORMER_WEIGHT}" \
        --output "${FUSED_DEV}" \
        --require_all

    echo "[Fuse] test"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/fuse_pred_gloss.py" \
        --source resnet34 "${RESNET34_TEST}" "${RESNET_WEIGHT}" \
        --source swin_bilstm "${SWIN_BILSTM_TEST}" "${SWIN_BILSTM_WEIGHT}" \
        --source swin_transformer "${SWIN_TRANSFORMER_TEST}" "${SWIN_TRANSFORMER_WEIGHT}" \
        --output "${FUSED_TEST}" \
        --require_all
}

train_mbart() {
    local args=(--config "${MBART_FUSED_CONFIG}" --experiment_name g2t_final_fused_mbart)
    if [ -n "${ANNOT_ROOT}" ]; then
        args+=(--annot_root "${ANNOT_ROOT}")
    fi
    args+=(--pred_dev "${FUSED_DEV}" --pred_test "${FUSED_TEST}")
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/training/train_g2t_slt.py" "${args[@]}"
}

train_cvae() {
    local args=(--config "${CVAE_FUSED_CONFIG}" --experiment_name g2t_final_fused_cvae)
    if [ -n "${ANNOT_ROOT}" ]; then
        args+=(--annot_root "${ANNOT_ROOT}")
    fi
    args+=(--pred_dev "${FUSED_DEV}" --pred_test "${FUSED_TEST}")
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/training/train_g2t_slt.py" "${args[@]}"
}

case "${MODE}" in
    envcheck)
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/check_g2t_env.py"
        ;;
    export)
        run_export
        ;;
    export_smoke)
        EXPORT_MAX_SAMPLES="${EXPORT_MAX_SAMPLES:-8}"
        common_path_args+=(--max_samples "${EXPORT_MAX_SAMPLES}")
        run_export
        ;;
    fuse)
        run_fuse
        ;;
    train_mbart)
        train_mbart
        ;;
    train_cvae)
        train_cvae
        ;;
    train)
        train_mbart
        train_cvae
        ;;
    full)
        run_export
        run_fuse
        train_mbart
        train_cvae
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Use one of: envcheck, export, export_smoke, fuse, train_mbart, train_cvae, train, full"
        exit 2
        ;;
esac

echo "========================================================"
echo " Job complete."
echo "========================================================"
