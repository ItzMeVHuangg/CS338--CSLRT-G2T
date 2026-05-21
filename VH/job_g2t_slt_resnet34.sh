#!/bin/bash
# ==============================================================================
# SLURM Job -- Fast SLT comparison with frozen ResNet34 CSLR
# CSLR checkpoint: MBinh Variant A, ResNet34 + BiLSTM + CTC
#
# Usage:
#   sbatch job_g2t_slt_resnet34.sh export_smoke
#   sbatch job_g2t_slt_resnet34.sh export
#   sbatch job_g2t_slt_resnet34.sh all
#   sbatch job_g2t_slt_resnet34.sh full
#   sbatch job_g2t_slt_resnet34.sh confidence
#   sbatch job_g2t_slt_resnet34.sh full_confidence
#   sbatch job_g2t_slt_resnet34.sh full_mbart
#   sbatch job_g2t_slt_resnet34.sh full_cvae
#   sbatch job_g2t_slt_resnet34.sh test20
# ==============================================================================

#SBATCH --job-name=g2t_r34
#SBATCH --chdir=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
#SBATCH --output=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/output/g2t_resnet34_%j.out
#SBATCH --error=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/output/g2t_resnet34_%j.err
#SBATCH --gres=mps:a100:2
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

set -euo pipefail

MODE="${1:-all}"
REQUIRED_VRAM="${REQUIRED_VRAM:-12000}"
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

RESNET_CONFIG="${RESNET_CONFIG:-${PROJECT_DIR}/configs/config_resnet34.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-${PROJECT_DIR}/configs/config_g2t_baseline.yaml}"
PROPOSED_CONFIG="${PROPOSED_CONFIG:-${PROJECT_DIR}/configs/config_g2t_proposed.yaml}"
CONFIDENCE_CONFIG="${CONFIDENCE_CONFIG:-${PROJECT_DIR}/configs/config_g2t_confidence.yaml}"
MBART_CONFIG="${MBART_CONFIG:-${PROJECT_DIR}/configs/config_g2t_mbart.yaml}"
CVAE_CONFIG="${CVAE_CONFIG:-${PROJECT_DIR}/configs/config_g2t_cvae.yaml}"
DEFAULT_RESNET34_CKPT="/datastore/cndt_phungdtm/KLTN_HoangBinh/MBinh/checkpoints/ablation/Variant_A_path/best_path/best_path_12.pth"
CSLR_CKPT="${CSLR_CKPT:-${DEFAULT_RESNET34_CKPT}}"
G2T_CKPT="${G2T_CKPT:-${PROJECT_DIR}/checkpoints_g2t_slt/g2t_proposed_adat_gloss2text/best_g2t_slt.pth}"
PHOENIX_ROOT="${PHOENIX_ROOT:-}"
FRAMES_ROOT="${FRAMES_ROOT:-}"
ANNOT_ROOT="${ANNOT_ROOT:-}"
PRED_DEV="${PRED_DEV:-${PROJECT_DIR}/outputs/pred_gloss_resnet34_dev.jsonl}"
PRED_TEST="${PRED_TEST:-${PROJECT_DIR}/outputs/pred_gloss_resnet34_test.jsonl}"
EXPORT_MAX_SAMPLES="${EXPORT_MAX_SAMPLES:-}"
USE_CSLR_VOCAB="${USE_CSLR_VOCAB:-0}"

echo "========================================================"
echo " Fast G2T SLT -- ResNet34 CSLR"
echo " Project       : ${PROJECT_DIR}"
echo " Mode          : ${MODE}"
echo " Python        : ${PYTHON_BIN}"
echo " Required VRAM : ${REQUIRED_VRAM} MB"
echo " CSLR checkpoint: ${CSLR_CKPT}"
echo "========================================================"

run_export() {
    local common_args=(--config "${RESNET_CONFIG}" --cslr_ckpt "${CSLR_CKPT}" --encoder resnet34 --batch_size 1)
    if [ -n "${PHOENIX_ROOT}" ]; then
        common_args+=(--phoenix_root "${PHOENIX_ROOT}")
    fi
    if [ -n "${FRAMES_ROOT}" ]; then
        common_args+=(--frames_root "${FRAMES_ROOT}")
    fi
    if [ -n "${ANNOT_ROOT}" ]; then
        common_args+=(--annot_root "${ANNOT_ROOT}")
    fi
    if [ -n "${EXPORT_MAX_SAMPLES}" ]; then
        common_args+=(--max_samples "${EXPORT_MAX_SAMPLES}")
    fi

    echo "[Export ResNet34] Dev -> ${PRED_DEV}"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        "${common_args[@]}" \
        --split dev \
        --output "${PRED_DEV}"

    echo "[Export ResNet34] Test -> ${PRED_TEST}"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/export_swin_gloss.py" \
        "${common_args[@]}" \
        --split test \
        --output "${PRED_TEST}"
}

run_train() {
    local config_path="$1"
    local label="$2"
    local exp_name="$3"
    local args=(--config "${config_path}" --experiment_name "${exp_name}")

    if [ "${USE_CSLR_VOCAB}" = "1" ]; then
        args+=(--cslr_ckpt "${CSLR_CKPT}")
    fi
    if [ -n "${ANNOT_ROOT}" ]; then
        args+=(--annot_root "${ANNOT_ROOT}")
    fi
    if [ -f "${PRED_DEV}" ]; then
        args+=(--pred_dev "${PRED_DEV}")
    fi
    if [ -f "${PRED_TEST}" ]; then
        args+=(--pred_test "${PRED_TEST}")
    fi

    echo "========================================================"
    echo " Train ${label}"
    echo " Config: ${config_path}"
    echo " Experiment: ${exp_name}"
    echo "========================================================"
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/training/train_g2t_slt.py" "${args[@]}"
}

run_test20() {
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/test_20_cslr_g2t.py" \
        --config "${RESNET_CONFIG}" \
        --encoder resnet34 \
        --cslr_ckpt "${CSLR_CKPT}" \
        --g2t_ckpt "${G2T_CKPT}" \
        --split test \
        --n_samples 20 \
        --seed 42 \
        --output "${PROJECT_DIR}/outputs/test20_resnet34_variant_a.json"
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
        run_export
        ;;
    baseline)
        run_train "${BASELINE_CONFIG}" "baseline vanilla Transformer G2T (ResNet34 pred eval)" "g2t_baseline_resnet34"
        ;;
    proposed)
        run_train "${PROPOSED_CONFIG}" "proposed G2T (ResNet34 pred eval)" "g2t_proposed_resnet34"
        ;;
    confidence)
        run_train "${CONFIDENCE_CONFIG}" "CAPG-G2T confidence-aware pseudo-gloss (ResNet34 pred eval)" "g2t_confidence_resnet34"
        ;;
    mbart)
        run_train "${MBART_CONFIG}" "pretrained mBART Gloss-to-Text (ResNet34 pred eval)" "g2t_mbart_resnet34"
        ;;
    cvae)
        run_train "${CVAE_CONFIG}" "latent CVAE Gloss-to-Text (ResNet34 pred eval)" "g2t_cvae_resnet34"
        ;;
    all)
        run_train "${BASELINE_CONFIG}" "baseline vanilla Transformer G2T (ResNet34 pred eval)" "g2t_baseline_resnet34"
        run_train "${PROPOSED_CONFIG}" "proposed G2T (ResNet34 pred eval)" "g2t_proposed_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    compare3)
        run_train "${BASELINE_CONFIG}" "baseline vanilla Transformer G2T (ResNet34 pred eval)" "g2t_baseline_resnet34"
        run_train "${PROPOSED_CONFIG}" "proposed G2T (ResNet34 pred eval)" "g2t_proposed_resnet34"
        run_train "${CONFIDENCE_CONFIG}" "CAPG-G2T confidence-aware pseudo-gloss (ResNet34 pred eval)" "g2t_confidence_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    full)
        run_export
        run_train "${BASELINE_CONFIG}" "baseline vanilla Transformer G2T (ResNet34 pred eval)" "g2t_baseline_resnet34"
        run_train "${PROPOSED_CONFIG}" "proposed G2T (ResNet34 pred eval)" "g2t_proposed_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    full_confidence)
        run_export
        run_train "${CONFIDENCE_CONFIG}" "CAPG-G2T confidence-aware pseudo-gloss (ResNet34 pred eval)" "g2t_confidence_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    full_mbart)
        run_export
        run_train "${MBART_CONFIG}" "pretrained mBART Gloss-to-Text (ResNet34 pred eval)" "g2t_mbart_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    full_cvae)
        run_export
        run_train "${CVAE_CONFIG}" "latent CVAE Gloss-to-Text (ResNet34 pred eval)" "g2t_cvae_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    full_new)
        run_export
        run_train "${MBART_CONFIG}" "pretrained mBART Gloss-to-Text (ResNet34 pred eval)" "g2t_mbart_resnet34"
        run_train "${CVAE_CONFIG}" "latent CVAE Gloss-to-Text (ResNet34 pred eval)" "g2t_cvae_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    full3)
        run_export
        run_train "${BASELINE_CONFIG}" "baseline vanilla Transformer G2T (ResNet34 pred eval)" "g2t_baseline_resnet34"
        run_train "${PROPOSED_CONFIG}" "proposed G2T (ResNet34 pred eval)" "g2t_proposed_resnet34"
        run_train "${CONFIDENCE_CONFIG}" "CAPG-G2T confidence-aware pseudo-gloss (ResNet34 pred eval)" "g2t_confidence_resnet34"
        "${PYTHON_BIN}" -u "${PROJECT_DIR}/script/summarize_g2t_results.py" \
            --root "${PROJECT_DIR}/checkpoints_g2t_slt"
        ;;
    test20)
        run_test20
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Use one of: envcheck, export, export_smoke, baseline, proposed, confidence, mbart, cvae, all, compare3, full, full_confidence, full_mbart, full_cvae, full_new, full3, test20"
        exit 2
        ;;
esac

echo "========================================================"
echo " Job complete."
echo "========================================================"
