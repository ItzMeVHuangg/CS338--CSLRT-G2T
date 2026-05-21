# Fast SLT Comparison: Frozen Swin CSLR -> Gloss-to-Text

This branch is the short-deadline setup. The trained Video Swin-T + BiLSTM +
CTC checkpoint is reused only for inference/export. CSLR is not retrained.

## Pipelines

Baseline:

```text
Video -> frozen Swin-T + BiLSTM + CTC -> predicted gloss
predicted/ground-truth gloss -> Vanilla Transformer G2T -> German text
```

Proposed:

```text
Video -> frozen Swin-T + BiLSTM + CTC -> predicted gloss
predicted/ground-truth gloss -> ADAT/Gloss2Text-inspired G2T -> German text
```

Additional CAPG-G2T variant:

```text
Video -> frozen Swin-T + BiLSTM + CTC -> predicted gloss + CTC confidence
predicted/ground-truth gloss + <conf_low>/<conf_mid>/<conf_high>
  -> confidence-aware pseudo-gloss G2T
  -> German text
```

The proposed G2T adds:

- a learned PHOENIX weather-domain prefix;
- rule-extracted weather slot features from gloss tokens;
- a lightweight Conv1D gated local/global encoder inspired by ADAT-style
  adaptive temporal modeling;
- label smoothing in the Gloss2Text-style training setup;
- optional confidence-aware pseudo-gloss conditioning for predicted CSLR gloss.

## Files Added For This Experiment

- `configs/config_g2t_baseline.yaml`: vanilla Transformer G2T baseline.
- `configs/config_g2t_proposed.yaml`: proposed ADAT/Gloss2Text-inspired G2T.
- `configs/config_g2t_confidence.yaml`: CAPG-G2T confidence-aware pseudo-gloss variant.
- `data/gloss_text_dataset.py`: annotation-only PHOENIX G2T dataset.
- `models/gloss_to_text.py`: G2T baseline/proposed model switches.
- `training/train_g2t_slt.py`: SLT-only trainer.
- `script/export_swin_gloss.py`: export predicted gloss and CTC confidence from frozen CSLR.
- `script/test_20_cslr_g2t.py`: test 20 samples for a CSLR checkpoint and optional G2T checkpoint.
- `script/summarize_g2t_results.py`: print slide-ready comparison table.
- `job_g2t_slt.sh`: SLURM job for export/train/compare.
- `job_g2t_slt_resnet34.sh`: same workflow for ResNet34 Variant A.
- `job_test20_cslr_variants.sh`: test 20 samples on both Swin Variant G and ResNet34 Variant A.

## 1. Train Oracle G2T First

This does not need video frames and does not need a CSLR checkpoint.

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
python training/train_g2t_slt.py --config configs/config_g2t_baseline.yaml
python training/train_g2t_slt.py --config configs/config_g2t_proposed.yaml
```

This gives the oracle comparison:

```text
ground-truth gloss -> text
```

## 2. Export Predicted Gloss From Frozen Swin CSLR

Use your trained `.pt`/`.pth` Swin-T BiLSTM CTC checkpoint:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
python script/export_swin_gloss.py \
  --config configs/config_swin.yaml \
  --cslr_ckpt /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --split dev \
  --output outputs/pred_gloss_swin_dev.jsonl \
  --encoder swin
```

For test:

```bash
python script/export_swin_gloss.py \
  --config configs/config_swin.yaml \
  --cslr_ckpt /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --split test \
  --output outputs/pred_gloss_swin_test.jsonl \
  --encoder swin
```

Each export also writes a metrics file such as:

```text
outputs/pred_gloss_swin_dev.jsonl.metrics.json
```

## 3. Train And Evaluate With Predicted Gloss

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
python training/train_g2t_slt.py \
  --config configs/config_g2t_baseline.yaml \
  --cslr_ckpt /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --pred_dev outputs/pred_gloss_swin_dev.jsonl \
  --pred_test outputs/pred_gloss_swin_test.jsonl

python training/train_g2t_slt.py \
  --config configs/config_g2t_proposed.yaml \
  --cslr_ckpt /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --pred_dev outputs/pred_gloss_swin_dev.jsonl \
  --pred_test outputs/pred_gloss_swin_test.jsonl

python training/train_g2t_slt.py \
  --config configs/config_g2t_confidence.yaml \
  --cslr_ckpt /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --pred_dev outputs/pred_gloss_swin_dev.jsonl \
  --pred_test outputs/pred_gloss_swin_test.jsonl
```

## 4. SLURM Submit

Oracle-only comparison:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt.sh baseline
sbatch job_g2t_slt.sh proposed
```

Full comparison with frozen CSLR:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt.sh full
```

CAPG-G2T only:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt.sh full_confidence
```

All three variants:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt.sh full3
```

Fast oracle comparison first, without exporting Swin predictions:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt.sh all
```

If the annotation path changes:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
ANNOT_ROOT=/path/to/PHOENIX-2014-T/annotations/manual \
sbatch job_g2t_slt.sh all
```

`job_g2t_slt.sh` uses this Swin Variant G checkpoint by default:

```text
/datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg/checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth
```

If export appears slow, test only a few samples first:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt.sh export_smoke
```

## ResNet34 Variant A

ResNet34 checkpoint:

```text
/datastore/cndt_phungdtm/KLTN_HoangBinh/MBinh/checkpoints/ablation/Variant_A_path/best_path/best_path_12.pth
```

Smoke export:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt_resnet34.sh export_smoke
```

Full ResNet34 predicted-gloss evaluation:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt_resnet34.sh full
```

ResNet34 with CAPG-G2T:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_g2t_slt_resnet34.sh full_confidence
```

## Test 20 Samples

Run both Swin Variant G and ResNet34 Variant A on 20 test samples:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
sbatch job_test20_cslr_variants.sh
```

Outputs:

```text
outputs/test20_swin_variant_g.json
outputs/test20_swin_variant_g.jsonl
outputs/test20_resnet34_variant_a.json
outputs/test20_resnet34_variant_a.jsonl
```

## Environment Check

On the cluster:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
source .venv/bin/activate
python script/check_g2t_env.py
```

If only non-torch packages are missing:

```bash
pip install -r requirements_g2t_slt.txt
```

Do not reinstall `torch` or `torchvision` from this requirements file. Use the
CUDA-compatible torch stack already installed in the cluster venv/module.

## 5. Results For Slides

After training:

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
python script/summarize_g2t_results.py --root checkpoints_g2t_slt
```

Main files:

```text
checkpoints_g2t_slt/g2t_baseline_vanilla_transformer/summary.json
checkpoints_g2t_slt/g2t_proposed_adat_gloss2text/summary.json
checkpoints_g2t_slt/g2t_confidence_pseudogloss/summary.json
```

Report the comparison as:

```text
Same frozen CSLR WER = 37.35%
Baseline SLT: predicted gloss -> Vanilla Transformer G2T
Proposed SLT: predicted gloss -> ADAT/Gloss2Text-inspired G2T
CAPG-G2T: predicted gloss + CTC confidence -> confidence-aware pseudo-gloss G2T
```
