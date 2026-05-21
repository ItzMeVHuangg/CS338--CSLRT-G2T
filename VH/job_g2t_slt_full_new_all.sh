#!/bin/bash
# ==============================================================================
# Submit all new G2T/SLT variants:
#   1) Swin-T + BiLSTM + CTC       -> mBART G2T + CVAE G2T
#   2) Swin-T + Transformer + CTC  -> mBART G2T + CVAE G2T
#   3) ResNet34 + BiLSTM + CTC     -> mBART G2T + CVAE G2T
#
# Usage on cluster login node:
#   cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
#   bash job_g2t_slt_full_new_all.sh
#
# Optional overrides:
#   REQUIRED_VRAM=18000 bash job_g2t_slt_full_new_all.sh
#   MBART_CONFIG=configs/config_g2t_mbart.yaml CVAE_CONFIG=configs/config_g2t_cvae.yaml bash job_g2t_slt_full_new_all.sh
# ==============================================================================

set -euo pipefail

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

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found. Run this script on the SLURM login node."
    exit 1
fi

echo "========================================================"
echo " Submit full new SLT runs"
echo " Project: ${PROJECT_DIR}"
echo " Mode   : full_new = export predicted gloss + train mBART + train CVAE"
echo "========================================================"

submit_job() {
    local label="$1"
    local script="$2"
    echo "[Submit] ${label}: sbatch ${script} full_new"
    sbatch "${script}" full_new
}

submit_job "Swin-T BiLSTM CTC" "${PROJECT_DIR}/job_g2t_slt.sh"
submit_job "Swin-T Transformer CTC" "${PROJECT_DIR}/job_g2t_slt_swin_transformer.sh"
submit_job "ResNet34 BiLSTM CTC" "${PROJECT_DIR}/job_g2t_slt_resnet34.sh"

echo "========================================================"
echo " Submitted 3 jobs."
echo " Check queue: squeue -u \$USER"
echo " Summarize after jobs finish:"
echo "   python script/summarize_g2t_results.py --root checkpoints_g2t_slt"
echo "========================================================"
