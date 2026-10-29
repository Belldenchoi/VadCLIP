# CNN phụ học đặc trưng tương lai bằng MSE

**Bản mới theo yêu cầu học chung từng epoch:** xem [JOINT_MSE.md](JOINT_MSE.md),
mode `future_mse_joint`. Không chạy `pretrain.py` và không cần `encoder.pt`.
Tài liệu bên dưới giữ lại **bản hai giai đoạn cũ** để đối chiếu/resume.

Biến thể `--aux-mode future_mse` thuộc `experiments/cnn_memory`. Không sửa `src/`
hoặc notebook trong `Finetune-VadCLIP`. Đây là cách tổ chức học lấy cảm hứng từ
SSOC, dùng feature CLIP toàn cảnh có sẵn; không tái hiện mục tiêu ngoại hình và
bounding box của đối tượng trong SSOC. Chưa có kết quả train GPU/UCF-Crime.

## Hai giai đoạn

1. `pretrain.py` chỉ đọc **train CSV**, có cả Normal và Abnormal. Dùng cùng
   `process_feat()` như detector: gộp trung bình theo thời gian nếu dài hơn
   `visual_length`, hoặc pad nếu ngắn hơn. Nhãn video không tham gia MSE.
   Không đọc test CSV, frame GT hoặc annotation.
2. `train.py --aux-mode future_mse` nạp encoder đã học, bỏ đầu dự đoán tương lai,
   **đóng băng encoder**, học đầu chấm điểm CNN cùng các thành phần trainable
   gốc của VadCLIP bằng MIL. Kiến trúc LGT, classifier C, Visual Prompt và A gốc
   được giữ lại. Giữ nguyên kiến trúc không có nghĩa là đóng băng toàn bộ VadCLIP.

```text
Giai đoạn 1:
CLIP f [B,T,512] -> left padding -> Conv1D(512,D,k) -> GELU
                               -> Conv1D(D,D,1) -> LayerNorm -> GELU -> z
z -> Linear(D,512) -> dự đoán f[t+horizon]
                   -> MSE với f[t+horizon] cố định

Giai đoạn 2:
CLIP f -> LGT gốc -> C gốc -------------------------> C + scale*r
                   |   |
                   |   +-> aggregation gốc -> Visual Prompt -> text adaptation
                   +----------------------------------------> dot product -> A
CLIP f -> CNN encoder đã học, đóng băng -> z -> Linear(D,1) -> r
A Normal: giữ nguyên; A lớp bất thường: cộng scale*r
```

- CNN chỉ nhận feature **trước LGT**, chỉ nhìn bước hiện tại và quá khứ. Kernel 3
  có cửa sổ 3 bước feature; không trộn feature tương lai vào đầu vào dự đoán.
- Đầu dự đoán `Linear(D,512)` chỉ dùng ở giai đoạn 1, không có trong model inference.
- `horizon=1` là bước tiếp theo trên lưới feature sau `process_feat`, **không phải**
  một frame gốc hoặc luôn là 16 frame. Với video dài, mỗi bước có thể là trung bình
  của nhiều vector CLIP. Clip quá ngắn để có target được đếm và bỏ qua.
- MSE là trung bình theo mọi cặp thời gian hợp lệ và 512 chiều. Target được detach;
  padding không tạo cặp học và NaN trong padding bị loại trước convolution.
- Nhánh phụ mới không dùng text, prototype, ngân hàng bộ nhớ hoặc cosine matching.
  Nhánh A gốc vẫn giữ phép chuẩn hóa/dot product với text của VadCLIP.
- Visual Prompt luôn dùng logit C **trước** khi cộng residual; không có vòng phản hồi.
- `r` là logit Normal/Abnormal học bằng MIL, không phải sai số MSE. MSE thấp/cao
  không tự động được diễn giải thành Normal/Abnormal.
- Đầu chấm điểm khởi tạo bằng 0: lần forward hoạt động đầu tiên khớp baseline.
  `--residual-scale` áp dụng cho C và bias các lớp bất thường của A. Bản này không
  học residual riêng cho từng loại sự kiện; chênh lệch logits giữa các lớp bất
  thường tại cùng một thời điểm vẫn do A gốc quyết định.

Loss giai đoạn 2 giữ nguyên cách tính của thí nghiệm cũ:

```text
CLAS2(C_sau_bổ_trợ) + CLASM(A_sau_bổ_trợ) + loss_text_gốc
                    + memory_loss_weight * CLAS2(r)
```

Tên `loss_memory`/`weighted_memory` trong log được giữ để tương thích các parser
cũ. Trong mode `future_mse`, chúng là loss MIL video của đầu r; MSE chỉ xuất hiện
trong `pretrain.log` và `metrics.jsonl` của giai đoạn 1.

## Chạy Kaggle

Bản học chung mới dùng cell train trong [JOINT_MSE.md](JOINT_MSE.md).
Các lệnh riêng bên dưới giữ lại bản hai giai đoạn, dành cho notebook cũ đã
có hai CSV tên cố định. Chạy setup/CSV/GT cũ, bật GPU và Internet.
Code được cung cấp trong nhánh `main` của
[Belldenchoi/VadCLIP](https://github.com/Belldenchoi/VadCLIP);
clone mới hoặc cập nhật checkout trước khi chạy:

```bash
!git -C /kaggle/working/VadCLIP pull --ff-only origin main
```

Dùng hai CSV Kaggle cũ, giữ nguyên thứ tự test. Mỗi khối dưới đây là **một cell**.

Giai đoạn 1, pretrain 5 epoch, kernel 3, horizon 1:

```python
!cd /kaggle/working/VadCLIP && \
python experiments/cnn_memory/pretrain.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --cnn-kernel 3 --feature-dim 256 --horizon 1 \
  --seed 234 --batch-size 64 --max-epoch 5 \
  --output-dir outputs/cnn_memory_mse_pretrain_k3_h1_s234_r1
```

Giai đoạn 2, train VadCLIP + CNN phụ 10 epoch:

```python
!cd /kaggle/working/VadCLIP && \
python experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --aux-mode future_mse --aux-branch both \
  --pretrained-cnn outputs/cnn_memory_mse_pretrain_k3_h1_s234_r1/encoder.pt \
  --cnn-kernel 3 --feature-dim 256 --memory-start-epoch 1 \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_memory_mse_both_k3_h1_s234_r1
```

Nhánh phụ bật từ epoch 1 trong lệnh này vì đã pretrain; mặc định CLI cũ vẫn là
epoch 3. Control cần dùng cùng `--memory-start-epoch 1` để có cùng lịch chọn best.
`--aux-branch c` hoặc `a` chỉ thay nơi cộng residual, không thay Visual Prompt gốc.

Evaluate vẫn gọi evaluator Top-K/gốc, có đủ AUC1/AP1, AUC2/AP2 và mAP:

```python
!cd /kaggle/working/VadCLIP && \
python experiments/cnn_memory/evaluate.py \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --checkpoint outputs/cnn_memory_mse_both_k3_h1_s234_r1/best.pt \
  --log-path outputs/cnn_memory_mse_both_k3_h1_s234_r1/evaluate.log
```

Control cùng trainer, seed, CSV và lịch chọn checkpoint:

```python
!cd /kaggle/working/VadCLIP && \
python experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --aux-mode future_mse --aux-branch none --memory-start-epoch 1 \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_memory_mse_control_s234_r1
```

So sánh này đo tác dụng của cả nhánh CNN đã pretrain. Để quy riêng hiệu quả cho
MSE cần thêm control encoder ngẫu nhiên tương ứng; chưa thể quy toàn bộ gain cho MSE.
Best vẫn chọn theo **test AUC1**, như trainer hiện có; không phải validation độc lập.

## Quy tắc dự đoán video rồi định vị

`predict.py` là bước xuất kết quả tùy chọn, tách khỏi evaluator AUC/AP. Nó chỉ đọc
đường dẫn feature, bỏ qua nhãn trong CSV và không đọc GT. Từ score C sau bổ trợ,
lấy trung bình Top-K xác suất với `k=T//16+1` như CLAS2 để dự đoán video. Video
được dự đoán Normal luôn có `segments=[]`; video Abnormal mới ngưỡng hóa các
score theo thời gian và gộp thành đoạn liên tiếp.

```python
!cd /kaggle/working/VadCLIP && \
python experiments/cnn_memory/predict.py \
  --checkpoint outputs/cnn_memory_mse_both_k3_h1_s234_r1/best.pt \
  --video-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --video-threshold 0.5 --frame-threshold 0.5 \
  --output-path outputs/cnn_memory_mse_both_k3_h1_s234_r1/predictions.json
```

Ngưỡng 0.5 chỉ là giá trị chạy thử; chọn ngưỡng trên validation cho ứng dụng thực.
`--score-branch a` dùng `1-softmax(A)[Normal]` thay C nếu muốn xuất từ nhánh A.
Khoảng frame dùng quy ước repeat 16, `[start_frame, end_frame_exclusive)`; không
phải định vị độc lập từng frame. Nếu gate video dự đoán nhầm Normal, sự kiện sẽ
bị bỏ qua. Gate không thay raw score hoặc loại video Normal khỏi tính AUC/AP.

## Checkpoint, resume và kiểm tra

- Pretrain: `encoder.pt` chứa encoder và cấu hình/provenance, không chứa đầu P;
  `last.pt` có thêm đầu P, optimizer và RNG. Resume bằng cùng tùy chọn, tăng
  `--max-epoch`, thêm `--resume <pretrain-output>/last.pt`.
- Train: `best.pt`/`last.pt` chứa đủ encoder đã pretrain, đầu chấm điểm, VadCLIP,
  trạng thái kích hoạt và mode. Resume/evaluate không cần file pretrain bên ngoài.
  Khi resume train, giữ các tùy chọn, có thể bỏ `--pretrained-cnn`, thêm
  `--resume <train-output>/last.pt`. Checkpoint prototype cũ vẫn được đọc như trước.
- `pretraining_info` ghi hash checkpoint, hash train CSV, số epoch và horizon.
  Kernel, chiều feature, visual_length và nội dung train CSV phải khớp giữa hai
  giai đoạn. Hash CSV không thay thế việc kiểm soát train/test split của dữ liệu.
- `pretrain.py --preflight` kiểm tra dữ liệu train mà không train/tải CLIP;
  `train.py --preflight` giữ kiểm tra CSV/GT cũ. Không dùng test data cho MSE.

```powershell
python -B -m unittest discover -s experiments/cnn_memory/tests -v
```

Kiểm tra CPU dùng dữ liệu giả cho causal masking, MSE/gradient, encoder transfer,
đóng băng encoder, C/A + Visual Prompt, gate video, checkpoint và resume. Chúng
kiểm tra implementation, không xác nhận AUC thực tế trên UCF-Crime.
