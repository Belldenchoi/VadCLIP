# Soft Top-K + loss xếp hạng trên branch C

## Thiết kế thử nghiệm

Giữ Soft Top-K trên C và A, nhiệt độ cố định 1.0. Giữ nguyên BCE của C,
CE của A và loss văn bản hiện có. Chỉ thêm loss xếp hạng vào C; không thay
evaluator, K, learning rate hoặc chính sách nạp lại model tốt nhất cuối epoch.
Đây là thử nghiệm mới, chưa có kết quả AUC và không bảo đảm cải thiện.

Với mỗi video, s là chính điểm sau sigmoid và Soft Top-K đã dùng cho BCE
(trước clamp bảo vệ BCE). Không tính pooling lần hai, không detach trọng số.
Chỉ dùng snippet hợp lệ; không gán nhãn bất thường cho toàn bộ snippet.

Gọi P là các video có nhãn bất thường và N là các video bình thường trong batch:

\[
L_{rank}=\frac{1}{|P||N|}\sum_{i\in P}\sum_{j\in N}
\max(0,m-s_i+s_j),\qquad
L=L_C+L_A+L_{text}+\lambda L_{rank}.
\]

Tính trên tất cả cặp, không ghép theo thứ tự và không chỉ chọn cặp khó nhất.
Batch mặc định gồm 64 video bình thường + 64 video bất thường, tức 4096 cặp.
Nếu thiếu một nhóm, trả zero có gradient và ghi số cặp bằng 0.
Loss tính FP32; trung bình theo cặp để không tự tăng hệ số khi đổi batch size.

Ví dụ m=0.2: s_bất_thường=0.6, s_bình_thường=0.55 cho loss 0.15;
với hai điểm 0.8 và 0.3, loss bằng 0. Lambda=0.1 làm phần cộng tương ứng
là 0.015 và 0. Margin nằm trong (0,1), lambda không âm, đều phải hữu hạn.

## Cấu hình và đối chứng

- `--c-ranking-loss`: bật riêng loss mới (mặc định tắt).
- `--c-ranking-margin 0.2`: khoảng cách điểm mong muốn.
- `--c-ranking-weight 0.1`: hệ số lambda cố định, không phải biến Dual-K.
- `--c-ranking-start-epoch 1`: epoch bắt đầu, tính từ 1.

Các giá trị trên là cấu hình thăm dò, chưa tối ưu. Cho phép weight=0 để kiểm
tra tương đương. Bản đầu yêu cầu pooling soft, nhiệt độ cố định, không kết hợp
AIS, Dual-K, Conv1D, temporal segment hoặc loss làm trơn để tách tác động.
Không có cờ mới nào cần dùng khi chạy phương pháp cũ.

Log ghi loss thô, loss đã nhân hệ số, điểm trung bình hai nhóm, khoảng cách
trung bình, số cặp và tỷ lệ cặp chưa đạt margin. Epoch summary ghi trung bình
loss; epoch active dùng số epoch tuyệt đối cả khi resume.
Checkpoint mới lưu cấu hình ranking và từ chối resume nếu cấu hình khác.
Checkpoint cũ không có cấu hình này chỉ được resume khi ranking tắt.
Không sửa hạn chế resume giữa epoch của vòng train cũ.

## Ánh xạ code

- `ranking_loss.py`: loss tất cả cặp, thống kê và kiểm tra cấu hình/checkpoint.
- `CLAS2` trong `train_ucf.py`: tùy chọn trả thêm điểm video, mặc định vẫn scalar BCE.
- `train`: cộng ranking ngoài BCE, giữ loss1/loss2/loss3 có ý nghĩa cũ.
- `options.py`: CLI tách riêng; `tests/test_ranking_loss.py`: regression tests.

## Lệnh Kaggle cho lần chạy mới

```bash
!cd /kaggle/working/VadCLIP && \
mkdir -p outputs && \
set -o pipefail && \
python experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 \
  --batch-size 64 \
  --max-epoch 10 \
  --topk-pooling soft \
  --c-topk-temperature 1.0 \
  --a-topk-temperature 1.0 \
  --a-topk-temperature-schedule constant \
  --c-ranking-loss \
  --c-ranking-margin 0.2 \
  --c-ranking-weight 0.1 \
  --c-ranking-start-epoch 1 \
  --temporal-smoothness-branch none \
  --model-path outputs/ucf_soft_c_ranking.pth \
  --checkpoint-path outputs/ucf_soft_c_ranking_checkpoint.pth \
  --log-path outputs/ucf_soft_c_ranking.log
```

Đối chứng: bỏ cờ `--c-ranking-loss`, đổi ba đường dẫn output sang tên riêng,
giữ nguyên các yếu tố khác và cùng phiên bản code. Không resume run nhiệt độ
thay đổi. So sánh AUC1 peak và các epoch lân cận; chọn cấu hình trên validation,
không liên tục tối ưu bằng test. Kết quả một seed chỉ có tính thăm dò.
