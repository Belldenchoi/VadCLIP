# VadCLIP Direct Class Prototypes

Thí nghiệm này thay Learnable Prompt và Visual Prompt của A-branch bằng class
prototype cố định. Temporal encoder, C-branch và MIL-Align hard Top-K gốc vẫn giữ nguyên.

## Pipeline cập nhật

Pipeline gồm hai giai đoạn độc lập: xây dựng prototype offline và huấn luyện VadCLIP.

### Giai đoạn A — Xây dựng class prototype offline

```text
14 class UCF-Crime
        │
        ▼
20 description cho mỗi class
(chủ thể + hành động, không a/an/the, tránh tính từ)
        │
        ▼
Frozen CLIP Text Encoder ViT-B/16
        │
        ▼
L2-normalized description embeddings
[14, 20, 512]
        │
        ├── gần class-name anchor
        ├── gần leave-one-out centroid cùng class
        └── xa centroid class khác gần nhất
        │
        ▼
Semantic score
score = 0.4 × name + 0.3 × intra - 0.3 × inter
        │
        ▼
Xếp hạng + diversity filtering
(cosine threshold = 0.90)
        │
        ▼
Top-5 description embeddings mỗi class
        │
        ▼
Weighted mean theo softmax(score / 0.10)
        │
        ▼
L2 normalization
        │
        ▼
Fixed class prototypes P [14, 512]
        │
        ├── ucf_crime.pt
        └── ucf_crime_selection.json
```

CLIP Text Encoder chỉ chạy ở bước offline này. Prototype sau khi tạo được cache và
không nhận gradient trong quá trình huấn luyện.

### Giai đoạn B — Huấn luyện với prototype

```text
Precomputed CLIP video features [B, T, 512]
                         │
                         ▼
               VadCLIP temporal encoder
                         │
                         ▼
               Visual features V [B, T, 512]
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
          C-branch                 A-branch
       classifier + MLP       normalize(V) @ normalize(P)ᵀ
             │                  / temperature 0.07
             ▼                       │
     anomaly logits1                 ▼
         [B, T, 1]          class logits2 [B, T, 14]
             │                       │
             ▼                       ▼
   Hard Top-K gốc theo T    Hard Top-K gốc theo T và class
   K = floor(Tvalid/16)+1   K = floor(Tvalid/16)+1
             │                       │
             ▼                       ▼
          loss1                   loss2
             │                       │
             └───────────┬───────────┘
                         ▼
                total loss = loss1 + loss2
```

MIL-Align không bị loại bỏ. A-branch vẫn chọn hard Top-K temporal logits riêng cho
từng class rồi tính weakly-supervised classification loss như baseline.

### So với VadCLIP gốc

Phần được thay đổi nằm trước bước tạo `logits2`. Hai khối **Learnable Prompt** và
**Visual Prompt** của pipeline gốc được thay bằng quá trình xây dựng **Fixed Class
Prototype** offline. Prototype cuối cùng chính là label feature mới.

```mermaid
flowchart LR
    subgraph ORIGINAL["VadCLIP gốc"]
        O1["Class name"] --> O2["Learnable Prompt"]
        O2 --> O3["Frozen CLIP<br/>Text Encoder"]
        O3 --> O4["Base text feature<br/>C x D"]
        O4 --> O5["Visual Prompt<br/>+ video context + FFN"]
        O5 --> O6["Video-conditioned<br/>label feature B x C x D"]
    end

    subgraph PROTOTYPE["Class Prototype thử nghiệm"]
        P1["20 descriptions / class"] --> P2["Frozen CLIP<br/>Text Encoder offline"]
        P2 --> P3["Scoring + diversity<br/>+ Top-K"]
        P3 --> P4["Weighted mean<br/>+ L2 normalize"]
        P4 --> P5["Fixed class prototype<br/>P = label feature C x D"]
    end

    O6 --> S1["Cosine với temporal<br/>visual features"]
    P5 --> S1
    S1 --> S2["Frame-level logits2<br/>B x T x C"]
    S2 --> S3["MIL-Align hard Top-K gốc"]
    S3 --> S4["loss2"]

    classDef removed fill:#ffd6d6,stroke:#c62828,color:#111;
    classDef added fill:#d8f3dc,stroke:#2d6a4f,color:#111;
    classDef shared fill:#dbeafe,stroke:#2563eb,color:#111;
    class O2,O5 removed;
    class P1,P2,P3,P4,P5 added;
    class S1,S2,S3,S4 shared;
```

Màu đỏ là cơ chế gốc bị loại khỏi cấu hình thử nghiệm; màu xanh lá là cơ chế
prototype thay thế; màu xanh dương là pipeline dùng chung được giữ nguyên.

```text
PHẦN GỐC BỊ THAY

Class name
    +
Learnable Prompt
    |
    v
Frozen CLIP Text Encoder
    |
    v
Base text feature [C, D]
    |
    v
Visual Prompt + video context + FFN
    |
    v
Video-conditioned label feature [B, C, D]

                  ĐƯỢC THAY BẰNG

Descriptions theo class
    |
    v
Frozen CLIP Text Encoder chạy offline
    |
    v
Scoring + diversity filtering + Top-K descriptions
    |
    v
Weighted mean + L2 normalization
    |
    v
Fixed class prototype P [C, D]
    = label feature mới

                  PHẦN SAU GIỮ NGUYÊN

Temporal visual features [B, T, D]
    +
Label feature [C, D]
    |
    v
Cosine similarity / 0.07
    |
    v
Frame-level logits2 [B, T, C]
    |
    v
MIL-Align hard Top-K theo class
    |
    v
loss2
```

Điểm cần phân biệt: pipeline mới **vẫn tạo label feature**. Tuy nhiên, label feature
mới là prototype cố định `[C, D]`, dùng chung cho mọi video; nó không còn là label
feature phụ thuộc video `[B, C, D]` như đầu ra của Visual Prompt gốc.

| Thành phần | VadCLIP gốc | Class Prototype thử nghiệm |
|---|---|---|
| Learnable Prompt | Có | Bỏ |
| CLIP Text Encoder | Frozen, chạy trong forward | Frozen, chỉ chạy offline |
| Visual Prompt | Có | Bỏ |
| Text representation | Phụ thuộc training và video | Prototype cố định |
| Temporal encoder | Có | Giữ nguyên |
| C-branch | Có | Giữ nguyên |
| MIL-Align hard Top-K | Có | Giữ nguyên |
| `loss1` | Có | Giữ nguyên |
| `loss2` | Có | Giữ nguyên |
| `loss3` text separation | Có | Tắt vì prototype cố định |

Thí nghiệm đầu tiên chỉ đánh giá Class Prototype với Top-K gốc. AIS và Temporal
Segment Top-K không được bật để tránh thay đổi đồng thời nhiều cơ chế.

## Description

`descriptions/ucf_crime_descriptions.json` chứa 20 description cho mỗi class, tập trung
vào chủ thể và hành động. Description không dùng mạo từ `a`, `an`, `the` và tránh tính từ.

## 1. Build prototype

```bash
python experiments/class_prototypes/build_prototypes.py \
  --output outputs/class_prototypes/ucf_crime.pt \
  --audit-output outputs/class_prototypes/ucf_crime_selection.json \
  --top-k 5 \
  --diversity-threshold 0.90 \
  --aggregation weighted_mean \
  --weight-temperature 0.10
```

Script dùng frozen CLIP ViT-B/16, chấm score theo class-name, intra-class và
hard-negative inter-class, sau đó diversity filtering và weighted mean.

Trên Kaggle:

```bash
!cd /kaggle/working/VadCLIP && \
mkdir -p outputs/class_prototypes && \
python experiments/class_prototypes/build_prototypes.py \
  --output outputs/class_prototypes/ucf_crime.pt \
  --audit-output outputs/class_prototypes/ucf_crime_selection.json \
  --top-k 5 \
  --diversity-threshold 0.90 \
  --aggregation weighted_mean \
  --weight-temperature 0.10
```

## 2. Train với MIL-Align Top-K gốc

```bash
python experiments/class_prototypes/train_ucf.py \
  --prototype-path outputs/class_prototypes/ucf_crime.pt \
  --topk-pooling mean \
  --model-path outputs/ucf_prototype.pth \
  --checkpoint-path outputs/ucf_prototype_checkpoint.pth \
  --log-path outputs/ucf_prototype.log
```

Không bật `--adaptive-instance-selection` hoặc `--temporal-segment-topk` trong thí
nghiệm đầu tiên. `loss3` được tắt vì prototype cố định; loss C-branch và MIL-Align
A-branch vẫn được giữ.

Trên Kaggle:

```bash
!cd /kaggle/working/VadCLIP && \
set -o pipefail && \
python experiments/class_prototypes/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --prototype-path outputs/class_prototypes/ucf_crime.pt \
  --topk-pooling mean \
  --model-path outputs/ucf_prototype.pth \
  --checkpoint-path outputs/ucf_prototype_checkpoint.pth \
  --log-path outputs/ucf_prototype.log \
  2>&1 | tee outputs/ucf_prototype_console.log
```

## 3. Test

```bash
python experiments/class_prototypes/test_ucf.py \
  --prototype-path outputs/class_prototypes/ucf_crime.pt \
  --model-path outputs/ucf_prototype.pth \
  --log-path outputs/ucf_prototype_test.log
```
