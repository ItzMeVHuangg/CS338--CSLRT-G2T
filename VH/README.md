# Continuous Sign Language Recognition and Translation on RWTH-PHOENIX-Weather 2014T

Đồ án xây dựng pipeline nhận dạng và dịch ngôn ngữ ký hiệu liên tục từ video sang văn bản trên bộ dữ liệu **RWTH-PHOENIX-Weather 2014T**. Hệ thống được triển khai theo hướng two-stage:

```text
Video -> CSLR -> Gloss -> SLT / Gloss-to-Text -> German text
```

Trong đó, phần CSLR đã được huấn luyện trước bằng **Video Swin Transformer + BiLSTM + CTC**. Phần đóng góp mới của nhánh này tập trung vào giai đoạn **Sign Language Translation**, cụ thể là bài toán **Gloss-to-Text**.

## Proposed Pipeline

### Stage 1: CSLR

```text
Input video
  -> Video Swin-T encoder
  -> BiLSTM temporal model
  -> CTC classifier
  -> gloss sequence
```

Checkpoint CSLR tốt nhất được tái sử dụng:

```text
checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth
```

Trong giai đoạn SLT, CSLR được giữ cố định để giảm chi phí huấn luyện và để việc so sánh tập trung vào module dịch Gloss-to-Text.

### Stage 2: Baseline SLT

```text
Ground-truth / predicted gloss
  -> Vanilla Transformer Encoder-Decoder
  -> German text
```

Config baseline:

```text
configs/config_g2t_baseline.yaml
```

Baseline tắt các thành phần đề xuất:

```yaml
use_weather_slots: false
use_conv_gate: false
domain_prefix_len: 0
```

### Stage 3: Proposed SLT

```text
Ground-truth / predicted gloss
  -> gloss embedding
  -> learned weather-domain prefix
  -> weather-slot token
  -> Conv1D local temporal branch
  -> adaptive gate
  -> Transformer Encoder-Decoder
  -> German text
```

Config proposed:

```text
configs/config_g2t_proposed.yaml
```

Proposed bật các thành phần mới:

```yaml
use_weather_slots: true
use_conv_gate: true
domain_prefix_len: 1
```

### Stage 4: Confidence-Aware Pseudo-Gloss SLT

```text
Predicted gloss + CTC confidence
  -> prepend <conf_low>/<conf_mid>/<conf_high>
  -> confidence-aware pseudo-gloss augmentation
  -> ADAT/Gloss2Text-inspired G2T
  -> German text
```

Config:

```text
configs/config_g2t_confidence.yaml
```

Module mới này được đặt tên trong báo cáo là **CAPG-G2T**: Confidence-Aware Pseudo-Gloss Gloss-to-Text. Nó giữ nguyên frozen CSLR checkpoint, nhưng dùng confidence từ CTC để báo cho decoder biết chuỗi gloss đầu vào đáng tin ở mức nào. Trong train nhanh, model học thêm với các prefix confidence và nhiễu gloss nhẹ để bớt nhạy với lỗi từ CSLR.

## Method Origin and Implementation Files

| Component | Implementation file | Role |
|---|---|---|
| PHOENIX Gloss-to-Text dataset | `data/gloss_text_dataset.py` | Đọc annotation `orth -> translation`, không load video frame khi train G2T |
| Weather slot extraction | `data/gloss_text_dataset.py` | Trích các cue thời tiết như time, precipitation, wind, temperature, cloud, region từ gloss |
| Confidence-aware pseudo-gloss | `data/gloss_text_dataset.py` | Thêm `<conf_low>/<conf_mid>/<conf_high>` và augmentation lỗi gloss nhẹ cho CAPG-G2T |
| Baseline / proposed G2T model | `models/gloss_to_text.py` | Cài đặt `DomainSlotGlossToText` với các switch `use_conv_gate`, `use_weather_slots` |
| G2T trainer | `training/train_g2t_slt.py` | Train/evaluate baseline và proposed G2T, lưu `summary.json` |
| Predicted gloss export | `script/export_swin_gloss.py` | Dùng frozen CSLR checkpoint để sinh predicted gloss kèm CTC confidence |
| 20-sample CSLR/G2T test | `script/test_20_cslr_g2t.py` | In 20 mẫu ref/pred gloss và ref/pred text cho Swin hoặc ResNet34 |
| Result summary | `script/summarize_g2t_results.py` | In bảng kết quả BLEU dùng cho báo cáo |
| SLURM job | `job_g2t_slt.sh` | Chạy baseline/proposed hoặc full pipeline trên cluster |
| ResNet34 SLURM job | `job_g2t_slt_resnet34.sh` | Phiên bản tương tự cho ResNet34 Variant A checkpoint |
| 20-sample comparison job | `job_test20_cslr_variants.sh` | Test 20 mẫu cho cả Swin Variant G và ResNet34 Variant A |

## Reference Papers

### Main 2024 Reference

**Gloss2Text: Sign Language Gloss Translation using LLMs and Semantically Aware Label Smoothing**  
Findings of EMNLP 2024  
Paper: [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.947/)  
arXiv: [2407.01394](https://arxiv.org/abs/2407.01394)

This paper is used as the main 2024 reference because it directly studies the **Gloss-to-Text** setting on sign language translation.

### Additional 2025 Reference

**ADAT: Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation**  
arXiv 2025  
Paper: [arXiv:2504.11942](https://arxiv.org/abs/2504.11942)

This paper motivates the use of adaptive temporal modeling for sign language translation. The implementation in this repository is **ADAT-inspired**, not a full reproduction of the ADAT architecture.

### CSLR / Visual Backbone References

- **Video Swin Transformer**: [arXiv:2106.13230](https://arxiv.org/abs/2106.13230)
- **Visual Alignment Constraint for Continuous Sign Language Recognition**: [ICCV 2021 Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)

## Novelty

The original project already includes a CSLR pipeline based on visual encoders, temporal modeling, and CTC decoding. The new contribution is placed after CSLR:

```text
Frozen Swin-T BiLSTM CTC -> Gloss-to-Text SLT
```

The proposed G2T module differs from a vanilla Transformer baseline in three ways:

1. **Weather-domain prefix**
   A learned prefix token is prepended to the gloss sequence to bias the model toward the PHOENIX weather-report domain.

2. **Gloss-derived weather slots**
   Simple domain slots are extracted from gloss tokens, such as time, precipitation, wind, temperature, cloud, and region. These slots provide compact semantic cues for the decoder.

3. **ADAT-inspired local/global fusion**
   A lightweight Conv1D branch captures local temporal patterns in gloss sequences. An adaptive gate fuses this local representation with the original gloss embedding before the Transformer encoder.

4. **Confidence-aware pseudo-gloss**
   The exported CSLR gloss now carries a CTC confidence score. CAPG-G2T prepends `<conf_low>`, `<conf_mid>`, or `<conf_high>` to the gloss sequence and uses light pseudo-gloss corruption during G2T training. This targets the recognition-to-translation error propagation problem when CSLR WER is not low.

Compared with **Gloss2Text 2024**, this project does not fine-tune a large language model. Instead, it adapts the Gloss-to-Text idea into a lightweight Transformer model that can be trained quickly under limited GPU time. CAPG-G2T further adds a practical pseudo-gloss confidence signal for the two-stage CSLR -> SLT setting.

Compared with **ADAT 2025**, this project does not reproduce the full ADAT architecture. It takes the core intuition of adaptive time-series modeling and implements a compact Conv1D-gated encoder block suitable for a short-deadline SLT experiment.

## Experimental Setup

The comparison is focused on the SLT/G2T stage:

```text
Baseline: GT gloss -> Vanilla Transformer G2T
Proposed: GT gloss -> Domain/Slot + ConvGate Transformer G2T
```

Current oracle Gloss-to-Text results on dev:

| Input | Model | BLEU-4 |
|---|---|---:|
| Ground-truth gloss | Vanilla Transformer G2T | 21.73 |
| Ground-truth gloss | Proposed ADAT/Gloss2Text-inspired G2T | 22.57 |

Improvement:

```text
+0.84 BLEU-4
```

The CSLR checkpoint has WER approximately:

```text
Swin-T BiLSTM CTC WER = 37.35%
```

Because the table above uses ground-truth gloss, it should be reported as **oracle Gloss-to-Text evaluation**. For full video-to-text evaluation, first export predicted gloss from the frozen CSLR checkpoint, then run the same G2T models on predicted gloss.

## How to Run

### Environment

```bash
cd /datastore/cndt_phungdtm/KLTN_HoangBinh/VHuangg
source .venv/bin/activate
pip install -r requirements_g2t_slt.txt
```

Check the environment:

```bash
python script/check_g2t_env.py
```

### Train Oracle G2T

```bash
python training/train_g2t_slt.py --config configs/config_g2t_baseline.yaml
python training/train_g2t_slt.py --config configs/config_g2t_proposed.yaml
python training/train_g2t_slt.py --config configs/config_g2t_confidence.yaml
```

Or submit through SLURM:

```bash
sbatch job_g2t_slt.sh all
```

### Export Predicted Gloss

```bash
python script/export_swin_gloss.py \
  --config configs/config_swin.yaml \
  --cslr_ckpt checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --split dev \
  --output outputs/pred_gloss_swin_dev.jsonl \
  --encoder swin
```

Smoke test through SLURM:

```bash
sbatch job_g2t_slt.sh export_smoke
```

Full pipeline through SLURM:

```bash
sbatch job_g2t_slt.sh full
```

Run the new CAPG-G2T module only:

```bash
sbatch job_g2t_slt.sh full_confidence
```

Run all three SLT variants after exporting Swin predicted gloss:

```bash
sbatch job_g2t_slt.sh full3
```

### ResNet34 Variant A Checkpoint

The ResNet34 CSLR checkpoint is:

```text
/datastore/cndt_phungdtm/KLTN_HoangBinh/MBinh/checkpoints/ablation/Variant_A_path/best_path/best_path_12.pth
```

Export predicted gloss with ResNet34:

```bash
sbatch job_g2t_slt_resnet34.sh export_smoke
sbatch job_g2t_slt_resnet34.sh export
```

Run the ResNet34-based full pipeline:

```bash
sbatch job_g2t_slt_resnet34.sh full
```

Run ResNet34 with CAPG-G2T:

```bash
sbatch job_g2t_slt_resnet34.sh full_confidence
```

### Test 20 Samples

Test 20 samples for both CSLR checkpoints:

```bash
sbatch job_test20_cslr_variants.sh
```

Outputs:

```text
outputs/test20_swin_variant_g.json
outputs/test20_swin_variant_g.jsonl
outputs/test20_resnet34_variant_a.json
outputs/test20_resnet34_variant_a.jsonl
```

Run manually for Swin Variant G:

```bash
python script/test_20_cslr_g2t.py \
  --config configs/config_swin.yaml \
  --encoder swin \
  --cslr_ckpt checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth \
  --g2t_ckpt checkpoints_g2t_slt/g2t_proposed_adat_gloss2text/best_g2t_slt.pth \
  --split test \
  --n_samples 20 \
  --output outputs/test20_swin_variant_g.json
```

Run manually for ResNet34 Variant A:

```bash
python script/test_20_cslr_g2t.py \
  --config configs/config_resnet34.yaml \
  --encoder resnet34 \
  --cslr_ckpt /datastore/cndt_phungdtm/KLTN_HoangBinh/MBinh/checkpoints/ablation/Variant_A_path/best_path/best_path_12.pth \
  --g2t_ckpt checkpoints_g2t_slt/g2t_proposed_adat_gloss2text/best_g2t_slt.pth \
  --split test \
  --n_samples 20 \
  --output outputs/test20_resnet34_variant_a.json
```

### Summarize Results

```bash
python script/summarize_g2t_results.py --root checkpoints_g2t_slt
```

Output checkpoints and summaries:

```text
checkpoints_g2t_slt/g2t_baseline_vanilla_transformer/
checkpoints_g2t_slt/g2t_proposed_adat_gloss2text/
checkpoints_g2t_slt/g2t_confidence_pseudogloss/
```

## Reporting Statement

Recommended wording for the report:

> The CSLR model based on Video Swin-T, BiLSTM, and CTC is trained first and the best checkpoint is reused as a frozen recognition module. This work focuses on improving the SLT stage, specifically Gloss-to-Text translation. The proposed G2T module adds PHOENIX weather-domain cues and an ADAT-inspired local/global gated encoder, improving oracle dev BLEU-4 from 21.73 to 22.57 compared with the vanilla Transformer G2T baseline.

For the additional module:

> An additional CAPG-G2T variant adds confidence-aware pseudo-gloss conditioning. CTC confidence from the frozen CSLR checkpoint is converted into `<conf_low>`, `<conf_mid>`, and `<conf_high>` tokens prepended to the gloss sequence, while lightweight pseudo-gloss corruption is used during G2T training to improve robustness to recognition errors.

## Notes

- Do not claim this as end-to-end fine-tuning unless CSLR weights are updated during SLT training.
- Current main result is oracle Gloss-to-Text. Full video-to-text evaluation requires predicted gloss export.
- The proposed model is designed for fast training and report-level comparison, not for claiming SOTA on RWTH-PHOENIX-Weather 2014T.
