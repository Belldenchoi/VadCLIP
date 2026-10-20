# Video–label semantic residual cho VadCLIP

Thí nghiệm UCF-Crime thêm nhánh R sau LGT, dùng video feature và feature của
tất cả nhãn để bổ trợ logits A/C. `src/model.py`, LGT và các script train gốc
được giữ nguyên. Trainer hiện có chỉ thêm ba dòng ghi cấu hình/chẩn đoán R.

Đây là code thí nghiệm đã kiểm tra bằng fixture; chưa có kết quả training
UCF-Crime thật và chưa có bằng chứng tăng AUC.

## Luồng tính toán

1. Gọi nguyên `CLIPVAD.forward(..., return_visual_features=True)` để nhận
   `text_features_ori`, `C0`, `A0`, `visual_features`.
2. Nhánh R nhận `visual_features` và `text_features_ori`. Nhánh này sử dụng
   toàn bộ nhãn ứng viên, không nhận ground-truth label của từng video.
3. Cộng hiệu chỉnh vào logits sau khi A/C gốc tính xong. Visual prompt của A
   vẫn dùng **C0**, không dùng C đã được R hiệu chỉnh; LGT chỉ chạy một lần.

```text
F = encode_video(features)                 # LGT gốc và lớp chiếu gốc
C0, A0 = original_C_and_A(F, text)          # giữ visual prompt C0 -> A0
delta_C, delta_A = R(F, text_features_ori)
C_final = C0 + 0.1 * delta_C                # khi bật R -> C
A_final = A0 + 0.1 * delta_A                # khi bật R -> A
```

Video/text được L2-normalize riêng trong R, rồi chiếu từ 512 xuống H=64.
Với mỗi cặp thời điểm t và nhãn c, R học:

```text
v = P_video(normalize(F[t]))
l = P_text(normalize(text_features_ori[c]))
h[t,c] = LayerNorm(GELU(Linear(concat(v, l, v*l))))
h'[t,c] = h[t,c] + GELU(depthwise_Conv1D_time(h[:,c]))
r[t,c] = Linear_score(h'[t,c])
delta_A[t,c] = r[t,c]
delta_C[t] = Linear_classes(r[t,:])
```

Implementation tính phép Linear trên ba thành phần riêng để tránh tạo tensor
concat `[B,T,K,3H]` lớn. Conv1D kernel 3 học trên feature tương tác, dùng cùng
bộ lọc cho từng lớp và không trộn các lớp theo thứ tự nhãn. Đây là Conv1D có
tham số học, khác với bộ lọc trung bình cố định trên score của Temporal-A.

`Linear_score` khởi tạo zero nên dự đoán đầu tiên khớp baseline chính xác.
Head này nhận gradient ngay từ update đầu; các lớp R phía trước bắt đầu nhận
gradient khi head đã khác zero. Scale A/C cố định 0.1, không có warm-up riêng;
R hoạt động từ epoch 1. C readout ban đầu lấy trung bình lớp bất thường trừ
lớp Normal, sau đó các trọng số này tiếp tục được học.

Padding bị mask trước projection, trước/sau Conv1D và sau các readout. Chunk
độ dài zero do evaluator gốc sinh ra có residual bằng zero. Việc này không
thay đổi cách LGT gốc xử lý padding.

Loss vẫn là MIL C + MIL-Align A + text regularization của trainer hiện tại,
tính trên logits cuối. Không thêm auxiliary loss. A/C dùng hard Top-K gốc;
không kết hợp Temporal-A, TV, PCA, ranking, shift, AIS hoặc Dual-K trong đợt này.

`--residual-branch none` không tạo module R và không thay output model. Khởi
tạo R bảo toàn trạng thái RNG sau khi tạo model gốc, giúp các cấu hình ghép
seed có cùng trọng số ban đầu và cùng trạng thái RNG cho việc trộn dữ liệu.

## Bốn cấu hình so sánh

| Tên | `--residual-branch` | Hiệu chỉnh trực tiếp |
| --- | --- | --- |
| B0 | none | Đối chứng cùng trainer, không có R |
| R_A | a | A |
| R_C | c | C |
| R_AC | both | A và C |

Đây là đối chứng **cùng trainer Top-K variants**, không phải tuyên bố tái hiện
bit-for-bit toàn bộ training script paper. Trainer này giữ các xử lý số học
FP32/clamp và chính sách checkpoint hiện có. Không dùng điểm Temporal-A R
trước đây làm baseline cho thí nghiệm mới.

Chỉ hiệu chỉnh một nhánh không có nghĩa nhánh kia bất biến suốt training:
encoder và prompt vẫn được tối ưu chung bằng loss hiện có. Trong cùng một
forward với cùng trọng số, output của nhánh không bật residual giữ nguyên.

Mặc định giữ nguyên **kiến trúc** LGT và tiếp tục tối ưu trọng số như baseline.
Nếu cần cố định cả trọng số LGT, thêm `--freeze-lgt` vào **cả bốn lệnh** và dùng
output directory mới. Không trộn kết quả freeze với kết quả không freeze.

## Lệnh Kaggle

Chạy tại checkout có code mới. Hai CSV dưới đây phải đã được sửa để cột `path`
trỏ tới `.npy` tồn tại trong mount Kaggle. Code không tự tìm hoặc đoán mount.

Cùng GPU, seed 234, batch mỗi loader 64 (tổng Normal + Anomaly là 128),
10 epoch, LR mặc định 2e-5, AMP tắt, accumulation 1. B0 và R_AC là cặp chạy
đầu tiên; R_A/R_C dùng để phân tích đóng góp. Mỗi lệnh có output riêng.

### B0

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/semantic_residual/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --topk-pooling mean \
  --residual-branch none --residual-hidden-dim 64 --residual-kernel-size 3 \
  --residual-a-scale 0.1 --residual-c-scale 0.1 \
  --model-path outputs/t4_semantic_residual_r1/B0_seed234/model.pth \
  --checkpoint-path outputs/t4_semantic_residual_r1/B0_seed234/checkpoint.pth \
  --log-path outputs/t4_semantic_residual_r1/B0_seed234/train.log
```

### R_AC

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/semantic_residual/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --topk-pooling mean \
  --residual-branch both --residual-hidden-dim 64 --residual-kernel-size 3 \
  --residual-a-scale 0.1 --residual-c-scale 0.1 \
  --model-path outputs/t4_semantic_residual_r1/R_AC_seed234/model.pth \
  --checkpoint-path outputs/t4_semantic_residual_r1/R_AC_seed234/checkpoint.pth \
  --log-path outputs/t4_semantic_residual_r1/R_AC_seed234/train.log
```

### R_A

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/semantic_residual/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --topk-pooling mean \
  --residual-branch a --residual-hidden-dim 64 --residual-kernel-size 3 \
  --residual-a-scale 0.1 --residual-c-scale 0.1 \
  --model-path outputs/t4_semantic_residual_r1/R_A_seed234/model.pth \
  --checkpoint-path outputs/t4_semantic_residual_r1/R_A_seed234/checkpoint.pth \
  --log-path outputs/t4_semantic_residual_r1/R_A_seed234/train.log
```

### R_C

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/semantic_residual/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --topk-pooling mean \
  --residual-branch c --residual-hidden-dim 64 --residual-kernel-size 3 \
  --residual-a-scale 0.1 --residual-c-scale 0.1 \
  --model-path outputs/t4_semantic_residual_r1/R_C_seed234/model.pth \
  --checkpoint-path outputs/t4_semantic_residual_r1/R_C_seed234/checkpoint.pth \
  --log-path outputs/t4_semantic_residual_r1/R_C_seed234/train.log
```

Để ghép thêm seed, thay mọi `234` bằng `3407`, rồi `2026`, cả seed và output.
Nếu chạy lại, đổi `t4_semantic_residual_r1` thành thư mục mới. CLI từ chối ghi
đè model, checkpoint, log hoặc model_cur khi bắt đầu một run mới.

Lần kiểm tra sau có thể thay `--residual-kernel-size 3` bằng `1` và dùng output
mới để tách ảnh hưởng của MLP tương tác với Conv1D. Kernel 1 bỏ hẳn Conv1D.

## Đọc log, checkpoint và đánh giá

Log có `semantic_residual_config`, `residual_active`,
`residual_a_weighted_abs_mean` / `residual_c_weighted_abs_mean` cho nhánh được
bật. Text source là `text_features_ori`; prompt source là `original_C_logits`.

Checkpoint và model export chứa `_extra_state` lưu cấu hình R, class order,
scale và trạng thái freeze. Load/resume sai cấu hình bị từ chối. Để resume,
chạy lại đúng lệnh cũ và thêm `--resume`; `--max-epoch` là tổng số epoch đích.
Resume giữ chính sách của trainer hiện tại: checkpoint khi đánh giá là
best-test AUC1; checkpoint smoke `--skip-eval` lưu trạng thái cuối epoch và RNG.

Evaluator vẫn dùng `sigmoid(C_final)` và `1-softmax(A_final)[Normal]`, repeat
16 để so với GT. `evaluation_mode=original` mô tả cách chuyển logits thành
score, không có nghĩa residual bị tắt. R hoạt động cả train lẫn inference.
Trainer chọn checkpoint theo AUC1, đồng thời log AUC2/AP1/AP2 và mAP. Kết quả
best-test/single-seed là sơ bộ; cần so các seed ghép để đánh giá độ ổn định.

Ví dụ đánh giá lại R_AC, dùng đúng cấu hình đã train:

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/semantic_residual/evaluate.py \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --residual-branch both --residual-hidden-dim 64 --residual-kernel-size 3 \
  --residual-a-scale 0.1 --residual-c-scale 0.1 \
  --model-path outputs/t4_semantic_residual_r1/R_AC_seed234/model.pth \
  --log-path outputs/t4_semantic_residual_r1/R_AC_seed234/evaluation.log
```

Không dùng `src/ucf_test.py` cho model R vì file đó dựng model gốc không có R.

## Kiểm tra local

```powershell
& 'C:\Users\Minh\AppData\Local\Programs\Python\Python312\python.exe' -B -m unittest discover -s experiments/semantic_residual/tests -v
& 'C:\Users\Minh\AppData\Local\Programs\Python\Python312\python.exe' -B -m unittest discover -s experiments/topk_variants/tests -v
git diff --check
```

Fixture dùng feature ngẫu nhiên và CLIP giả lập, chạy LGT, C/A, MIL, optimizer,
save/resume và evaluator thật. `DistanceAdj` gốc hard-code CUDA; fixture CPU
chỉ đổi device target trong hàm đó, giữ nguyên công thức. Không sửa LGT trong
code sản phẩm và không cần tải pretrained CLIP để chạy tests. Kiểm tra này
xác nhận wiring/gradient/serialization; không xác nhận chất lượng detection,
tốc độ hoặc VRAM trên T4. Train thật dùng môi trường CUDA và dependencies của
VadCLIP gốc.
