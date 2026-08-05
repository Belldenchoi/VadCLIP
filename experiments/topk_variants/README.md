# VadCLIP Top-K variants

Folder này chứa toàn bộ code thí nghiệm Top-K tách biệt khỏi baseline. Không
file nào trong `src/` cần chỉnh sửa để chạy các phương pháp ở đây.

## Cấu trúc

```text
experiments/topk_variants/
├── train_ucf.py                   # train mean/soft/multi-K/AIS
├── test_ucf.py                    # đánh giá và ghi log
├── options.py                     # CLI riêng của variants
├── dataset_variants.py            # lọc action/smoke-test riêng
├── topk_pooling.py                # mean, Soft Top-K, Multi-K
├── adaptive_instance_selection.py # AIS tách khỏi Soft Top-K
├── detection_map.py               # mAP có thể lọc action
├── training_log.py                # log ra terminal và file
├── inspect_topk_weights.py        # xem score/index/weight Top-K
└── tests/                         # unit tests riêng
```

Model, temporal adapter và CLIP vẫn được dùng trực tiếp từ `src/model.py` của
baseline gốc. Folder này không ghi đè hay monkey-patch code trong `src/`.

## Bốn cấu hình so sánh

### 1. Baseline gốc

Muốn tái hiện code paper, chạy trực tiếp file gốc:

```bash
python src/ucf_train.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv
```

### 2. Soft Top-K

Giữ nguyên K gốc `floor(Tvalid/16)+1`. Sau khi chọn K temporal score lớn nhất,
dùng Softmax theo score để tính weighted mean.

```bash
python experiments/topk_variants/train_ucf.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --topk-pooling soft \
  --c-topk-temperature 1.0 \
  --a-topk-temperature 1.0 \
  --model-path outputs/ucf_soft.pth \
  --checkpoint-path outputs/ucf_soft_checkpoint.pth \
  --log-path outputs/ucf_soft.log
```

### 3. Multi-K

Tính mean Top-K riêng tại bốn tỷ lệ Top-1%, Top-5%, Top-10% và Top-20%, sau
đó lấy trung bình bốn kết quả.

```bash
python experiments/topk_variants/train_ucf.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --topk-pooling multi_k \
  --multi-k-percentages 1 5 10 20 \
  --model-path outputs/ucf_multi_k.pth \
  --checkpoint-path outputs/ucf_multi_k_checkpoint.pth \
  --log-path outputs/ucf_multi_k.log
```

### 4. Adaptive Instance Selection

C-branch sinh anomaly probability cho từng temporal feature. AIS tính một K
cho mỗi cặp Normal/Anomaly rồi dùng cùng K cho mean Top-K của C- và A-branch.
AIS không nằm trong `topk_pooling.py` và không thêm layer học mới.

```bash
python experiments/topk_variants/train_ucf.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --topk-pooling mean \
  --adaptive-instance-selection \
  --ais-score-threshold 0.9 \
  --ais-min-k 1 \
  --model-path outputs/ucf_ais.pth \
  --checkpoint-path outputs/ucf_ais_checkpoint.pth \
  --log-path outputs/ucf_ais.log
```

AIS cố ý không cho chạy đồng thời với `soft` hoặc `multi_k`, giúp mỗi thí
nghiệm chỉ thay đổi một cơ chế pooling.

## Cấu hình mặc định để so sánh công bằng

```text
Epochs          10
Batch size      64 Normal + 64 anomaly
Learning rate   2e-5
Temporal length 256
LGT window      8
Seed            234
AMP             tắt
Accumulation    1
```

Đây là default UCF-Crime của code upstream. Nếu giảm batch hoặc bật AMP thì
kết quả không còn là so sánh hoàn toàn cùng cấu hình với baseline paper.

## Smoke test ba action

```bash
python experiments/topk_variants/train_ucf.py \
  --train-list list/ucf_CLIP_rgb.csv \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --train-actions Abuse Arson Arrest \
  --max-samples-per-label 8 \
  --max-epoch 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --amp \
  --skip-eval \
  --topk-pooling mean \
  --adaptive-instance-selection \
  --model-path outputs/smoke_ais.pth \
  --checkpoint-path outputs/smoke_ais_checkpoint.pth \
  --log-path outputs/smoke_ais.log
```

## Đánh giá

```bash
python experiments/topk_variants/test_ucf.py \
  --test-list list/ucf_CLIP_rgbtest.csv \
  --model-path outputs/ucf_ais.pth \
  --log-path outputs/ucf_ais_test.log
```

## Kiểm tra unit tests

```bash
python -B -m unittest discover \
  -s experiments/topk_variants/tests -v
```

## Xem trọng số Top-K

```bash
python experiments/topk_variants/inspect_topk_weights.py --help
```

Script này đọc model checkpoint và feature `.npy`, sau đó xuất temporal index,
score và weight của C/A branch. Dùng `--help` để xem đầy đủ tham số đầu vào.
