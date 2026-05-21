#!/bin/bash
# ==============================================================================
# SLURM Job -- Fast SLT comparison with frozen Swin-T Transformer CSLR
# CSLR checkpoint: Variant H, Swin-T + Transformer + CTC
#
# Usage:
#   sbatch job_g2t_slt_swin_transformer.sh export_smoke
#   sbatch job_g2t_slt_swin_transformer.sh export
#   sbatch job_g2t_slt_swin_transformer.sh full3
#   sbatch job_g2t_slt_swin_transformer.sh full_confidence
#
# This wrapper reuses job_g2t_slt.sh, but fixes the CSLR checkpoint,
# sequence model, predicted-gloss files, and G2T experiment names for Variant H.
# ==============================================================================

#SBATCH --job-name=g2t_swtr
#SBATCH --chdir=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
#SBATCH --output=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/output/g2t_swin_transformer_%j.out
#SBATCH --error=/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/output/g2t_swin_transformer_%j.err
#SBATCH --gres=mps:a100:2
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

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

export PROJECT_DIR
export SWIN_CONFIG="${SWIN_CONFIG:-${PROJECT_DIR}/configs/config.yaml}"
# The old best_path_29 file exports around 48% WER in the current SLT export
# path. Variant-H training logs show the ~38% WER region after resume, around
# epoch 70-75, so default to checkpoint_75 unless overridden explicitly.
export CSLR_CKPT="${CSLR_CKPT:-${PROJECT_DIR}/checkpoints/ablation/Variant_H_path/checkpoint/checkpoint_75.pth}"
export CSLR_SEQ_MODEL="${CSLR_SEQ_MODEL:-transformer}"
export CSLR_BUILDER="${CSLR_BUILDER:-ablation}"
export CSLR_CASE_NAME="${CSLR_CASE_NAME:-Swin-T Transformer}"
export EXP_SUFFIX="${EXP_SUFFIX:-swin_transformer}"
export PRED_DEV="${PRED_DEV:-${PROJECT_DIR}/outputs/pred_gloss_swin_transformer_dev.jsonl}"
export PRED_TEST="${PRED_TEST:-${PROJECT_DIR}/outputs/pred_gloss_swin_transformer_test.jsonl}"

exec bash "${PROJECT_DIR}/job_g2t_slt.sh" "$@"
