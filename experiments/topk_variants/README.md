# VadCLIP Top-K variants

Folder này chứa toàn bộ code thí nghiệm Top-K tách biệt khỏi baseline. Không
file nào trong `src/` cần chỉnh sửa để chạy các phương pháp ở đây.

## Cấu trúc

```text
experiments/topk_variants/
├── train_ucf.py                   # train các Top-K và temporal regularizer
├── test_ucf.py                    # đánh giá, optional temporal post-process
├── options.py                     # CLI riêng của variants
├── dataset_variants.py            # lọc action/smoke-test riêng
├── topk_pooling.py                # mean/soft/multi-K/temporal segment
├── adaptive_instance_selection.py # AIS tách khỏi Soft Top-K
├── temporal_smoothness.py         # smoothness loss cho C/A probability
├── detection_map.py               # mAP có thể lọc action
├── training_log.py                # log ra terminal và file
├── inspect_topk_weights.py        # xem score/index/weight Top-K
└── tests/                         # unit tests riêng
```

Model, temporal adapter và CLIP vẫn được dùng trực tiếp từ `src/model.py` của
baseline gốc. Folder này không ghi đè hay monkey-patch code trong `src/`.

## Các cấu hình so sánh

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

### 5. Temporal Segment Top-K

Hard Top-K gốc chọn K temporal position mạnh nhất và cho phép chúng nằm rời
rạc. Temporal Segment Top-K giữ nguyên K nhưng buộc A-branch chọn một cửa sổ
liên tục dài K. Mỗi class chọn cửa sổ riêng.

```bash
python experiments/topk_variants/train_ucf.py \
  --topk-pooling mean \
  --temporal-segment-topk \
  --temporal-segment-start-epoch 6 \
  --temporal-smoothing-kernel 1 \
  --checkpoint-metric auc1 \
  --model-path outputs/ucf_temporal_segment.pth \
  --checkpoint-path outputs/ucf_temporal_segment_checkpoint.pth \
  --log-path outputs/ucf_temporal_segment.log
```

`--temporal-smoothing-kernel 1` là identity. Kernel `3` hoặc `5` dùng fixed
mean Conv1D trước khi chọn segment trong training và smooth A-branch logits
trước khi tính metric trong evaluation. Post-process vẫn giữ nguyên số temporal
position; nó không crop video về selected segment.

Biến thể Weighted Temporal Segment vẫn chọn cửa sổ liên tục bằng mean, nhưng
thay mean đều bên trong cửa sổ đã chọn bằng score-derived softmax weights:

```text
segment = argmax cửa sổ mean dài K
w[t] = softmax(segment_score[t] / temperature)
pooled = sum w[t] * segment_score[t]
```

```bash
--temporal-segment-weighted \
--temporal-segment-temperature 1.0
```

Temperature thấp tập trung trọng số vào timestep mạnh; temperature cao tiến
gần arithmetic mean. Đây là trọng số pooling bên trong selected segment, không
phải `temporal_smoothness_weight` và không được cộng như một auxiliary loss.

### 6. Temporal Smoothness Loss

Smoothness Loss khác với fixed Conv1D phía trên. Nó không thay score bằng moving
average mà thêm một regularizer lên chênh lệch xác suất liền kề:

```text
p_A[t] = 1 - softmax(logits2[t])[Normal]
L_smooth_A = mean |p_A[t+1] - p_A[t]|

L_total = loss1 + loss2 + loss3 + lambda_A * L_smooth_A
```

Mục đích là giảm spike rời rạc và khuyến khích anomaly score tạo thành vùng
liên tục. Weight vẫn cần để cân bằng regularizer với các loss chính; bỏ phép
nhân tương đương cố định weight bằng `1.0`.

```bash
python experiments/topk_variants/train_ucf.py \
  --topk-pooling mean \
  --temporal-smoothness-branch a \
  --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --checkpoint-metric auc1 \
  --model-path outputs/smooth_a_only.pth \
  --checkpoint-path outputs/smooth_a_only_checkpoint.pth \
  --log-path outputs/smooth_a_only.log
```

Trong cấu hình A-only, `loss_smooth_c=0` là đúng. `weighted_smooth_a` được tính
bằng `a_temporal_smoothness_weight * loss_smooth_a`; giá trị có thể rất nhỏ và
bị giao diện làm tròn nhưng không đồng nghĩa regularizer bị tắt.

Smoothness Loss chỉ tác động trong training. Vì vậy log A-only có thể đồng thời
ghi `temporal_smoothness_active=True` và `temporal_eval_active=False`. Dòng thứ
hai chỉ nói fixed Conv1D post-process không chạy trong evaluator; evaluator vẫn
nhận model đã được smoothness regularization cập nhật.

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

Evaluator luôn log `AUC1/AP1`, `AUC2/AP2` và detection mAP. Checkpoint tốt nhất
mặc định vẫn được chọn theo `AUC1`, khớp policy trước đây:

```text
--checkpoint-metric auc1
```

Có thể chọn `ap1`, `auc2`, `ap2` hoặc `average_map` cho ablation riêng, nhưng
phải giữ cùng policy giữa các run cần so sánh. Training tiếp tục từ model hiện
tại qua các epoch; best checkpoint chỉ được nạp để xuất `model-path` sau khi
training kết thúc.

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
