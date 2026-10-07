# Thí nghiệm snippet ranking và snippet ranking + PCA

Ngày: 07/10/2026. Hai hướng được triển khai để kiểm thử và chạy Kaggle;
chưa có kết quả training cho các loss mới. Đây là giả thuyết nghiên cứu,
không phải kết luận PCA sẽ tăng AUC hoặc giảm biến động seed.

## 1. Ba cấu hình so sánh

| ID | Cấu hình | Loss thêm |
|---|---|---|
| R | Conv1D trung bình k3 + đoạn K liên tiếp + TV ở A | Không |
| S | R + ranking từng snippet ở C | 0,1 × L_score |
| SP | S + PCA trên biểu diễn sau adapter | 0,01 × L_PCA |

Cả ba giữ C Hard Top-K mean, K=min(T, floor(T/16)+1); A segment và
TV bật từ epoch 1, TV A=0,05. Các loss mới bật từ epoch 3. Epoch 1–2 là
khởi động chung. Không đổi pooling sang Soft Top-K. Hinge là cấu hình chính;
Softplus là ablation tùy chọn, không ghép vào SP trong lần so sánh đầu.

## 2. Hướng S: từng snippet positive so với đỉnh negative

Với p=sigmoid(logits1) chưa làm trơn, lấy Top-K rời rạc của từng video
abnormal. So mỗi phần tử được chọn với max score hợp lệ của **từng** video
Normal trong batch, tức tất cả cặp positive–negative, không phụ thuộc thứ tự.

```math
L_{score}=\frac{1}{|P||N|}\sum_{a\in P,b\in N}
\frac{1}{K_a}\sum_{i\in TopK(p_a)}[0.1+\max_j p_{b,j}-p_{a,i}]_+.
```

Ví dụ positive Top-K=[0,9; 0,8; 0,5], negative max=0,6, margin=0,1:
loss=(0+0+0,2)/3=0,066667. Pooling trước rồi ranking sẽ cho loss bằng 0
trong ví dụ này, bỏ qua snippet 0,5. K phụ thuộc độ dài hợp lệ của video;
tính trung bình trong video trước để video dài không tự có trọng số lớn hơn.
Gradient đi qua score được chọn và max negative; không đi qua chỉ số Top-K.

Softplus tùy chọn thay [v]+ bằng tau*softplus(v/tau), tau=0,05.
Nó làm mềm hàm phạt, không làm mềm phép chọn Top-K. Margin và các trọng số
ở đây là giá trị khởi đầu để thử, chưa được tối ưu.

## 3. Hướng SP: thêm residual ngoài không gian Normal

Lấy h=normalize(encode_video(x)): đặc trưng sau temporal/graph adapter,
trước classifier C và phép tương đồng A. Không áp PCA lên score 1 chiều
hoặc feature CLIP cố định rồi mong loss đó cập nhật encoder.

Đầu epoch 3, model ở eval/no_grad; lấy tối đa 128 video Normal **train**,
32 vị trí hợp lệ cách đều/video. Chọn video bằng RNG riêng seed 0, giữ cùng
mẫu giữa các seed training. Fit PCA tâm hóa trên CPU float64, giữ r=64
thành phần. Không whitening. Mean mu và basis U được cố định sau lần fit này.
Khôi phục RNG và trạng thái train/eval sau calibration. Không đọc test hoặc
đặc trưng abnormal để fit. Nếu mẫu không đủ hạng, dừng với lỗi rõ ràng.

```math
e(h)=||(h-mu)-UU^T(h-mu)||_2^2,
L_{normal}=mean_{b\in N}\ mean_{j<T_b} e(h_{b,j}),
L_{residual}=mean_{a\in P,b\in N}\ mean_{i\in TopK(p_a)}
[0.1+\max_{j<T_b}e(h_{b,j})-e(h_{a,i})]_+,
L_{PCA}=L_{normal}+L_{residual}.
```

Top-K positive vẫn được chọn theo **score C**, không chọn theo residual.
Max residual negative được tính trên toàn bộ vị trí hợp lệ của video đó;
nó có thể khác vị trí max score. Chuẩn hóa h hạn chế việc tăng norm để né
loss. PCA basis detach nhưng residual giữ gradient đến encoder. Không kéo
mọi abnormal về không gian Normal. Basis cố định có thể trở nên kém phù hợp
khi encoder đổi; theo dõi residual và violation trong log để chẩn đoán.

PCA không thay score đánh giá, không làm inference smoothing và không cần
ở file model cuối dùng cho test. Basis và cấu hình được lưu trong checkpoint
training để kiểm tra khi resume. Basis giữ nguyên khi trainer nạp lại best
model ở cuối epoch; không refit theo test hoặc từng batch.

## 4. Ánh xạ code và ranh giới ablation

| File | Vai trò |
|---|---|
| experiments/topk_variants/snippet_pca_loss.py | Loss, fit PCA, validation, metadata checkpoint |
| experiments/topk_variants/options.py | Cờ opt-in, mặc định tắt cả hai loss |
| experiments/topk_variants/train_ucf.py | Gọi loss mới, calibration, log, lưu/nạp trạng thái |
| src/model.py | Thêm tùy chọn trả visual_features, giữ nguyên 3 output mặc định |
| experiments/topk_variants/tests/test_snippet_pca_loss.py | Kiểm tra số học, gradient, padding, PCA và checkpoint |

Không sửa c-ranking-loss cũ: loss đó dùng điểm video và dành cho ablation
Soft Top-K riêng. Cờ mới không được kết hợp c-ranking-loss, Dual-K, AIS,
Multi-K, smoothing/segment ở C. Temporal A vẫn được phép. Không thay evaluator.

## 5. Protocol chạy và đọc kết quả

Chạy ghép seed [234,3407,2026] cho R, S, SP: tổng 9 run, tuần tự,
10 epoch/run, cùng T4, batch-size=64 mỗi loader (128 video/batch), workers=0,
AdamW LR=2e-5, milestones 4/8, gamma=0,1; AMP tắt, accumulation=1.
Mỗi run có thư mục model/checkpoint/log riêng, bắt đầu mới.

Giữ protocol lịch sử để đối chiếu: test AUC1 chọn best checkpoint và trainer
nạp lại **model** best ở cuối epoch, không rollback optimizer. Vì test tham
gia quỹ đạo train, đây chưa phải protocol validation độc lập. Không đổi riêng
protocol cho một hướng rồi so với hướng khác. Resume từ best checkpoint
lịch sử không tương đương khôi phục đúng batch bị ngắt; lệnh chính không resume.

Báo từng seed, mean và sample std (ddof=1), delta S−R và SP−S ghép seed;
không chỉ chọn seed tốt nhất. Ghi AUC2/AP/mAP tại cùng checkpoint AUC1 chính,
cùng với epoch đỉnh và đường diễn biến loss. PCA fit variance/rank/sample count,
L_score, L_normal, L_residual, loss có trọng số và tỷ lệ vi phạm đều được log.
Chỉ 3 seed là sàng lọc; chưa kết luận tính ổn định nếu chênh lệch nhỏ.

Môi trường, version thư viện, GPU, CSV/feature và code phải giữ giống nhau
giữa ba hướng. Cùng seed không bảo đảm giống bitwise giữa GPU hoặc thư viện.
Các lệnh cụ thể nằm trong LENH_KAGGLE_SNIPPET_RANKING_PCA.md cùng thư mục.

## 6. Kiểm chứng local ngày 07/10/2026

Toàn bộ suite Top-K: **79 tests pass**, trong đó 14 tests mới cho snippet/PCA
(Python 3.12, PyTorch 2.9.0+cpu). Bao gồm ví dụ số học, padding, gradient,
trọng số theo video, Softplus, PCA chuẩn hóa, calibration giữ RNG, checkpoint
và resume. Kiểm thử tích hợp chạy vòng trainer với model/dữ liệu giả và hàm
forward thực được trích từ code, không tải pretrained CLIP hoặc UCF-Crime.
Đường đánh giá trong kiểm thử tích hợp dùng evaluator giả để kiểm tra lưu best
và nạp model; không đo AUC thật. Output/gradient của ba output mặc định khớp
chính xác khi bật tùy chọn trả thêm representation trong fixture này.

Ba mẫu lệnh đã được parse với cả ba seed: 9 cấu hình hợp lệ, checkpoint path
khác nhau, không AMP/skip-eval/resume. Chưa chạy training trên T4, chưa kiểm
chứng VRAM/thời gian thực hoặc độ cải thiện AUC/độ lệch chuẩn qua seed.

Lệnh kiểm thử từ gốc repo, với Python có cài torch:

```bash
python -B -m unittest discover -s experiments/topk_variants/tests -v
```
