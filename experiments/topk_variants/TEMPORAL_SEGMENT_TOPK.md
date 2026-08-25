# Thiết kế Temporal Top-K cho VadCLIP

## 1. Mục tiêu

VadCLIP gốc dùng Hard Top-K: chọn K temporal position có score cao nhất dù chúng có
thể nằm rời rạc. Thiết kế này thử giả định anomaly thường tạo thành một đoạn thời gian
liên tục.

Phạm vi hiện tại:

- C-branch giữ Hard Top-K và có thể dùng fixed Conv1D trước khi chọn Top-K.
- A-branch chuyển sang Temporal Segment Top-K.
- Mỗi class trong A-branch chọn segment riêng.
- Có hai cách aggregate segment: mean đều và score-weighted.
- Temporal Smoothness Loss là regularizer độc lập, không phải pooling weight.
- Best checkpoint mặc định vẫn được chọn theo `AUC1`.

## 2. Ký hiệu

Với một video:

```text
T       valid temporal length
C       số class
S       A-branch logits, S ∈ R^(T×C)
K       min(T, floor(T/16) + 1)
c       class index
tau     temperature của weighted segment
```

Padding ngoài `T` không tham gia smoothing, segment selection hoặc loss.

## 3. Baseline Hard Top-K

### C-branch

```text
p_C[t] = sigmoid(logits1[t])
z_C    = mean(topK(p_C, K))
```

`z_C` được dùng để tính binary classification loss `L1`.

### A-branch

Mỗi class chọn K logits cao nhất độc lập:

```text
z_A[c] = mean(topK(S[:,c], K))
```

Các position được chọn không cần liên tục. Vector `z_A` được dùng để tính
multi-class MIL loss `L2`.

## 4. Temporal Segment Top-K

### 4.1. Chọn segment

Với mỗi class `c`, xét mọi cửa sổ liên tục dài K:

```text
m[s,c] = (1/K) × Σ(j=0..K-1) S[s+j,c]
s*[c]  = argmax_s m[s,c]
```

`s*[c]` là start index của segment tốt nhất cho class `c`.

Ví dụ:

```text
scores = [0.10, 0.92, 0.12, 0.78, 0.81, 0.09]
K = 2

Hard Top-K       → [0.92, 0.81], hai điểm rời rạc
Temporal Segment → [0.78, 0.81], một đoạn liên tục
```

### 4.2. Segment mean

Đây là Temporal Segment mặc định:

```text
z_A[c] = (1/K) × Σ(j=0..K-1) S[s*[c]+j,c]
```

Mọi position trong selected segment có weight bằng `1/K`.

### 4.3. Weighted Temporal Segment

Weighted Temporal Segment vẫn chọn `s*[c]` bằng window mean ở mục 4.1. Sau khi đã
chọn segment, K position bên trong segment nhận score-derived softmax weight:

```text
q[j,c] = S[s*[c]+j,c]

                 exp(q[j,c] / tau)
w[j,c] = ---------------------------------
          Σ(r=0..K-1) exp(q[r,c] / tau)

z_A[c] = Σ(j=0..K-1) w[j,c] × q[j,c]
```

Tính chất:

- `Σ_j w[j,c] = 1`.
- `tau` thấp làm weight tập trung vào peak mạnh.
- `tau` cao làm weight gần đều và tiến về segment mean.
- Mỗi class có segment và bộ weight riêng.
- Weight được suy ra từ logits, không phải tham số học mới.

CLI:

```text
--temporal-segment-weighted
--temporal-segment-temperature 1.0
```

Đây là biến thể chính theo yêu cầu hiện tại.

## 5. Fixed temporal smoothing trước pooling

Có thể dùng mean kernel cố định trước khi pooling:

```text
S_bar[t,c] = (1/H) × Σ(j=-r..r) S[clamp(t+j),c]
H = 2r + 1
```

A-branch thay `S` bằng `S_bar` trong toàn bộ công thức ở mục 4. C-branch áp
dụng cùng phép làm mượt lên `sigmoid(logits1)` rồi vẫn dùng Hard Top-K gốc.

```text
--c-temporal-smoothing-kernel 1  C identity, không smoothing
--c-temporal-smoothing-kernel 3  C mean 3 position trước Hard Top-K
--temporal-smoothing-kernel 1    A identity, không smoothing
--temporal-smoothing-kernel 3    A mean 3 position trước segment
```

Fixed smoothing không có loss weight và không thêm layer học mới.

## 6. Temporal Smoothness Loss

Temporal Smoothness Loss là auxiliary regularization. Nó không chọn Top-K, không
thay arithmetic mean trong segment và không dùng moving average.

### 6.1. Xác suất dùng để tính loss

C-branch:

```text
p_C[b,t] = sigmoid(logits1[b,t])
```

A-branch dùng cùng anomaly probability với evaluator:

```text
P_normal[b,t] = softmax(logits2[b,t,:])[Normal]
p_A[b,t]      = 1 - P_normal[b,t]
```

### 6.2. Smoothness của từng video

Với video `b` có valid length `T_b > 1`:

```text
TV_C[b] = 1/(T_b-1) × Σ(t=0..T_b-2) |p_C[b,t+1] - p_C[b,t]|

TV_A[b] = 1/(T_b-1) × Σ(t=0..T_b-2) |p_A[b,t+1] - p_A[b,t]|
```

Mỗi video được mean riêng để video dài không lấn át video ngắn. Video có `T_b=1`
không có adjacent pair nên không tham gia batch mean. Nếu toàn bộ batch đều có
`T_b=1`, smoothness loss bằng 0.

### 6.3. Batch loss

Gọi `B_valid` là tập video có `T_b > 1`:

```text
L_smooth_C = 1/|B_valid| × Σ(b∈B_valid) TV_C[b]
L_smooth_A = 1/|B_valid| × Σ(b∈B_valid) TV_A[b]
```

### 6.4. Total training loss

```text
L_total = L1 + L2 + L3
        + I_C × lambda_C × L_smooth_C
        + I_A × lambda_A × L_smooth_A
```

Trong đó:

```text
I_C = 1 nếu branch là c hoặc both và đã tới start epoch, ngược lại 0
I_A = 1 nếu branch là a hoặc both và đã tới start epoch, ngược lại 0
```

Các giá trị log:

```text
loss_smooth_a          = L_smooth_A
weighted_smooth_a      = lambda_A × L_smooth_A
loss_smooth_c          = L_smooth_C
weighted_smooth_c      = lambda_C × L_smooth_C
```

Ví dụ thực tế từ run A-only:

```text
L_smooth_A             = 0.002391
lambda_A               = 0.05
weighted_smooth_A      = 0.002391 × 0.05
                       = 0.00011955 ≈ 0.000120
```

Trong A-only, `weighted_smooth_c=0` là đúng. Bỏ phép nhân weight tương đương đặt
`lambda=1.0`, không phải loại bỏ khái niệm weight.

### 6.5. Mục đích và giới hạn

Smoothness Loss khuyến khích anomaly probability liền kề ổn định hơn:

```text
noisy:   0.05, 0.82, 0.10, 0.79, 0.08
smoother:0.05, 0.55, 0.68, 0.61, 0.10
```

Nó có thể giảm spike nhiễu và tạo vùng anomaly liên tục. Nếu bật quá sớm hoặc weight
quá lớn, nghiệm dễ là score phẳng; điều này làm mất anomaly ngắn và boundary.

Smoothness activation và Temporal Segment activation độc lập. Ví dụ sau là hợp lệ:

```text
temporal_smoothness_active=True
temporal_eval_active=False
```

Dòng thứ hai xác nhận evaluator luôn dùng raw logits của VadCLIP gốc;
Smoothness Loss vẫn đang cập nhật model trong training.

## 7. Training và evaluation

Training A-branch:

```text
logits2 [B,T,C]
    ↓ chọn contiguous segment theo class
selected segment [K,C]
    ↓ mean hoặc score-weighted pooling
video logits [B,C]
    ↓ weak label
L2
```

Evaluation giữ nguyên hoàn toàn evaluator VadCLIP gốc để tính AUC/AP/mAP:

- Segment selection và segment weights không chạy lúc test.
- Fixed Conv1D không chạy lúc test, kể cả khi training dùng kernel `3/5`.
- A-branch dùng trực tiếp `1 - softmax(raw_logits2)[Normal]`.
- C-branch dùng trực tiếp `sigmoid(raw_logits1)`.
- Log luôn ghi `temporal_eval_active=False evaluation_mode=original`.

Weighted Temporal Segment tác động trực tiếp trong training và gián tiếp tới metric
qua model weights đã học.

## 8. Activation theo epoch

T1 hiện tại kích hoạt Temporal Segment từ epoch 1:

```text
--temporal-segment-topk
--temporal-segment-start-epoch 1
```

Expected log:

```text
epoch=1 temporal_segment_active=True
c_branch_pooling=original_hard_topk
a_branch_pooling=weighted_temporal_segment
```

Smoothness Loss có start epoch riêng:

```text
--temporal-smoothness-start-epoch N
```

Không thay đổi hai start epoch cùng lúc trong một ablation.

## 9. Checkpoint policy

Best checkpoint mặc định vẫn chọn theo C-branch AUC:

```text
--checkpoint-metric auc1
```

Các metric vẫn được log đầy đủ:

```text
AUC1, AP1, AUC2, AP2, mAP@IoU, average_mAP
```

Training tiếp tục từ current model qua các epoch. Best-AUC checkpoint chỉ được dùng để
xuất `model-path` sau khi training hoàn tất.

## 10. Thí nghiệm tối thiểu

| ID | C-branch | A-branch pooling | Start | Smoothness Loss |
|---|---|---|---:|---:|
| T0 | Hard Top-K gốc | Hard Top-K gốc | — | Tắt |
| T1 | Hard Top-K gốc | Temporal Segment mean | 1 | Tắt |
| C-K3 | Conv1D `3` + Hard Top-K | Hard Top-K gốc | — | Tắt |
| CA-K3 | Conv1D `3` + Hard Top-K | Conv1D `3` + Segment mean | 1 | Tắt |
| S-A | Hard Top-K gốc | Hard Top-K gốc | — | A-only |

So sánh quan trọng:

```text
T0 → T1    ảnh hưởng của contiguous selection trên A
T0 → C-K3 ảnh hưởng riêng của fixed Conv1D trên C
T1 → CA-K3 ảnh hưởng thêm của fixed Conv1D trên cả C và A
T0 → S-A  ảnh hưởng riêng của Temporal Smoothness Loss
```

Trong CA-K3, C-branch dùng sigmoid score đã qua Conv1D để tính `L1`, còn A-branch
dùng logits đã qua Conv1D và Temporal Segment để tính `L2`. Evaluation vẫn dùng
raw C/A logits để giữ evaluator gốc.

## 11. Lệnh chạy T1W

```bash
python experiments/topk_variants/train_ucf.py \
  --topk-pooling mean \
  --temporal-segment-topk \
  --temporal-segment-start-epoch 1 \
  --temporal-smoothing-kernel 1 \
  --temporal-segment-weighted \
  --temporal-segment-temperature 1.0 \
  --temporal-smoothness-branch none \
  --checkpoint-metric auc1 \
  --model-path outputs/t1_weighted_temporal.pth \
  --checkpoint-path outputs/t1_weighted_temporal_checkpoint.pth \
  --log-path outputs/t1_weighted_temporal.log
```

Expected configuration log:

```text
temporal_segment_topk=True
temporal_segment_start_epoch=1
temporal_smoothing_kernel=1
temporal_segment_weighted=True
temporal_segment_temperature=1.0
temporal_smoothness_branch=none
checkpoint_metric=auc1
```

## 12. Unit tests

```bash
python -B -m unittest discover \
  -s experiments/topk_variants/tests -p "test_*.py"
```

Các test bao phủ:

- segment liên tục có window mean lớn nhất;
- segment selection riêng theo class;
- weighted aggregation đúng theo softmax;
- temperature phải dương;
- kernel 1 là identity;
- padding không tham gia;
- gradient hữu hạn;
- Smoothness Loss chỉ dùng valid adjacent pairs.
