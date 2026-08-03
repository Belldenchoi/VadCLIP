# Hướng dẫn chạy VadCLIP + Adaptive Instance Selection trên Colab

Hướng dẫn này chạy pipeline UCF-Crime sau:

```text
CLIP features (.npy)
    -> chia Normal/Anomaly theo video label
    -> ghép một batch Normal với một batch Anomaly
    -> sample/pad mỗi chuỗi về T=256 và lưu valid length
    -> VadCLIP C-branch sinh anomaly score
    -> AIS tính một K cho mỗi cặp Normal/Anomaly
    -> dùng cùng K cho mean Top-K ở C-branch và A-branch
    -> tính loss và cập nhật mô hình
```

## 1. Chuẩn bị Google Drive

Trong Google Drive, đặt feature vào một thư mục có cấu trúc tương tự:

```text
MyDrive/
└── UCFClipFeatures/
    └── UCFClipFeatures/          # FEATURE_ROOT trỏ vào đây
        ├── Abuse/
        ├── Arrest/
        ├── Arson/
        ├── ...
        ├── Training_Normal_Videos_Anomaly/
        └── Testing_Normal_Videos_Anomaly/
```

`FEATURE_ROOT` phải là thư mục trực tiếp chứa `Abuse`, `Arrest`, `Arson`,...
Nếu Drive của bạn không có hai lớp thư mục `UCFClipFeatures`, chỉ cần sửa
`FEATURE_ROOT` ở bước 6.

Các file cần có trong repository:

```text
list/ucf_CLIP_rgb.csv
list/ucf_CLIP_rgbtest.csv
list/gt_ucf.npy
list/gt_segment_ucf.npy
list/gt_label_ucf.npy
```

## 2. Bật GPU

Trong Colab chọn:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Sau đó chạy:

```python
!nvidia-smi
```

## 3. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 4. Đưa code vào Colab

Nếu code đã được đẩy lên GitHub:

```python
!git clone YOUR_GITHUB_REPO_URL /content/VadCLIP
%cd /content/VadCLIP
```

Nếu repository là private, có thể upload file ZIP lên Drive rồi giải nén:

```python
!unzip -q /content/drive/MyDrive/VadCLIP.zip -d /content/
%cd /content/VadCLIP
```

Kiểm tra đúng thư mục:

```python
!pwd
!ls src/utils/adaptive_instance_selection.py
!ls list/gt_ucf.npy
```

## 5. Cài dependency

Colab đã có PyTorch và CUDA. Chỉ cài các package còn lại:

```python
!pip install -q ftfy regex tqdm pandas scipy scikit-learn
```

Không cần cài package `clip` khác vì repository đã chứa implementation CLIP
trong `src/clip`. Lần tạo model đầu tiên, code sẽ tự tải checkpoint CLIP
ViT-B/16.

## 6. Tạo CSV dùng đường dẫn Google Drive

Không sửa hai CSV gốc. Tạo hai CSV tạm trong `/content`:

```python
from pathlib import Path
import pandas as pd

REPO_ROOT = Path("/content/VadCLIP")
FEATURE_ROOT = Path(
    "/content/drive/MyDrive/UCFClipFeatures/UCFClipFeatures"
)
OLD_ROOT = "/home/xbgydx/Desktop/UCFClipFeatures"

assert FEATURE_ROOT.is_dir(), f"Không tìm thấy: {FEATURE_ROOT}"
assert (FEATURE_ROOT / "Abuse").is_dir(), (
    "FEATURE_ROOT phải trực tiếp chứa thư mục Abuse"
)

csv_jobs = [
    (
        REPO_ROOT / "list/ucf_CLIP_rgb.csv",
        Path("/content/ucf_train_colab.csv"),
    ),
    (
        REPO_ROOT / "list/ucf_CLIP_rgbtest.csv",
        Path("/content/ucf_test_colab.csv"),
    ),
]

for source, destination in csv_jobs:
    table = pd.read_csv(source)
    table["path"] = table["path"].str.replace(
        OLD_ROOT, str(FEATURE_ROOT), regex=False
    )
    table.to_csv(destination, index=False)

    missing = [path for path in table["path"] if not Path(path).is_file()]
    print(
        destination,
        "samples =", len(table),
        "missing =", len(missing),
    )
    if missing:
        print("Ví dụ file không tìm thấy:", missing[0])
    assert not missing, "Hãy sửa FEATURE_ROOT rồi chạy lại cell này"
```

Kiểm tra số lượng nhãn và kích thước feature:

```python
import numpy as np

train_table = pd.read_csv("/content/ucf_train_colab.csv")
print(train_table.groupby("label").size())

sample_path = train_table.iloc[0]["path"]
sample = np.load(sample_path, mmap_mode="r")
print("Sample:", sample_path)
print("Shape:", sample.shape)

assert sample.ndim == 2
assert sample.shape[1] == 512
```

Mỗi file phải có dạng `[Ti, 512]`. `Ti` có thể khác nhau giữa các file; dataset
loader sẽ sample hoặc pad về `[256, 512]`.

## 7. Kiểm tra code AIS

```python
%cd /content/VadCLIP
!python -B -m unittest discover -s tests -v
```

Kết quả mong đợi là tất cả test đều `OK`.

## 8. Smoke test trước khi chạy full

Smoke test dưới đây chỉ dùng `Abuse`, `Arson`, `Arrest`, giới hạn 8 mẫu mỗi
nhãn và chạy 1 epoch:

```python
from pathlib import Path

SMOKE_OUT = Path("/content/drive/MyDrive/VadCLIP_outputs/ais_smoke")
SMOKE_OUT.mkdir(parents=True, exist_ok=True)
```

```python
%cd /content/VadCLIP
!python src/ucf_train.py \
  --train-list /content/ucf_train_colab.csv \
  --test-list /content/ucf_test_colab.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --train-actions Abuse Arson Arrest \
  --max-samples-per-label 8 \
  --max-epoch 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --amp \
  --skip-eval \
  --topk-pooling mean \
  --adaptive-instance-selection \
  --ais-score-threshold 0.9 \
  --ais-min-k 1 \
  --log-interval 1 \
  --model-path {SMOKE_OUT}/model.pth \
  --checkpoint-path {SMOKE_OUT}/checkpoint.pth \
  --log-path {SMOKE_OUT}/train.log
```

Xem log:

```python
!tail -n 30 {SMOKE_OUT}/train.log
```

Trong log cần xuất hiện:

```text
adaptive_instance_selection=True
ais_k_mean=...
ais_k_min=...
ais_k_max=...
ais_omega_mean=...
ais_confident_mean=...
```

Ở những epoch đầu, `ais_k_mean` gần 1 là bình thường vì mô hình chưa có nhiều
C-score của positive instance vượt ngưỡng `0.9`.

## 9. Train full trên A100 theo cấu hình gần paper VadCLIP

Cấu hình dưới đây dùng toàn bộ 13 anomaly classes, 10 epoch, FP32, batch 64 cho
mỗi pool Normal và Anomaly:

```python
from pathlib import Path

FULL_OUT = Path("/content/drive/MyDrive/VadCLIP_outputs/ais_full")
FULL_OUT.mkdir(parents=True, exist_ok=True)
```

```python
%cd /content/VadCLIP
!python src/ucf_train.py \
  --train-list /content/ucf_train_colab.csv \
  --test-list /content/ucf_test_colab.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --max-epoch 10 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --lr 1e-5 \
  --attn-window 8 \
  --topk-pooling mean \
  --adaptive-instance-selection \
  --ais-score-threshold 0.9 \
  --ais-min-k 1 \
  --log-interval 10 \
  --model-path {FULL_OUT}/model_ucf_ais.pth \
  --checkpoint-path {FULL_OUT}/checkpoint_ucf_ais.pth \
  --log-path {FULL_OUT}/train.log
```

Không thêm `--amp` trong lệnh trên để giữ FP32. Nếu A100 báo hết VRAM, dùng cấu
hình tiết kiệm bộ nhớ sau:

```text
--batch-size 8 --gradient-accumulation-steps 8 --amp
```

Effective batch khi đó vẫn xấp xỉ 64 Normal + 64 anomaly cho mỗi lần optimizer
update, nhưng không phải tái hiện hoàn toàn batch vật lý 64 của code gốc.

Nên chạy cell train ở foreground và giữ tab Colab mở. Model, checkpoint và log
được đặt trên Google Drive nên không mất khi runtime Colab bị xóa.

## 10. Đánh giá model tốt nhất

```python
%cd /content/VadCLIP
!python src/ucf_test.py \
  --test-list /content/ucf_test_colab.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --model-path {FULL_OUT}/model_ucf_ais.pth \
  --log-path {FULL_OUT}/test.log
```

Xem kết quả:

```python
!cat {FULL_OUT}/test.log
```

Các chỉ số chính:

- `AUC1`: chất lượng phát hiện anomaly theo frame của C-branch.
- `AP1`: Average Precision nhị phân của C-branch.
- `mAP@0.1` đến `mAP@0.5`: chất lượng phân loại và định vị loại bất thường.
- `average_mAP`: trung bình mAP trên các IoU threshold.

## 11. Tiếp tục từ checkpoint

Nếu `checkpoint_ucf_ais.pth` đã tồn tại:

```python
%cd /content/VadCLIP
!python src/ucf_train.py \
  --train-list /content/ucf_train_colab.csv \
  --test-list /content/ucf_test_colab.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --max-epoch 10 \
  --batch-size 64 \
  --topk-pooling mean \
  --adaptive-instance-selection \
  --ais-score-threshold 0.9 \
  --ais-min-k 1 \
  --use-checkpoint True \
  --model-path {FULL_OUT}/model_ucf_ais.pth \
  --checkpoint-path {FULL_OUT}/checkpoint_ucf_ais.pth \
  --log-path {FULL_OUT}/train.log
```

Phải giữ nguyên pooling, AIS threshold, batch settings và đường dẫn CSV so với
lần chạy trước.
