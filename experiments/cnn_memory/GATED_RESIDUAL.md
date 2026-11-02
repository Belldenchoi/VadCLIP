# CNN + Gated Residual Fusion — baseline v1 đã chốt

Triển khai thiết kế người dùng chốt ngày 02/11/2026 trong
`reports/cnn_memory/VadCLIP_CNN_Gated_Residual_Fusion_Method.md`.
Mode mới: `--aux-mode gated_residual --aux-branch c`.
Code nằm trong thí nghiệm `cnn_memory`; không sửa `src/` hoặc notebook.

## Cách làm

```text
H sau LGT [B,T,512] ---------------------------> A gốc
     |
     +-> Conv1D(512,128,k3,s1,d1,p1) -> GELU
         -> Linear(128,512) -> F_cnn

g = sigmoid(Linear(concat(H,F_cnn))) [B,T,512]
H_enhanced = H + g * F_cnn
logits_C = classifier(H_enhanced + mlp2(H_enhanced))
```

- Đúng một Conv1D; không dùng Conv1D thứ hai, multi-scale, dilation mở rộng,
  temporal difference hoặc local deviation.
- Projection/gate là Linear theo từng vị trí thời gian, không phải các lớp
  convolution bổ sung. Gate theo từng chiều feature; bias khởi tạo -2 như gợi ý
  trong bản chốt. Các trọng số còn lại dùng khởi tạo PyTorch thông thường.
- Giữ nguyên `classifier` và residual `mlp2` hiện có, chỉ thay feature đầu vào.
  Không có classifier thứ hai, `logits_aux`, bank prototype hoặc đầu MSE.
- Module [CNNFeatureEnhancement](feature_fusion.py) chỉ nhận feature và lengths,
  không nhận text. Mask padding trước convolution; output padding được đặt 0.
  Đã kiểm tra tương đương chạy trên từng valid segment, kể cả chunk rỗng.

## Giữ đường A gốc

Original A có phụ thuộc vào C qua aggregation. Vì vậy giữ nguyên trọng số
aggregation tính từ H gốc, không đưa C đã enhancement ngược vào Visual Prompt:

```text
visual_prompt_weights = classifier(H + mlp2(H))
Visual Prompt = cơ chế aggregation/text adaptation gốc dùng H và các trọng số trên
logits_A = normalized(H) @ normalized(adapted_text).T / 0.07
```

Đây là tái sử dụng cùng classifier/MLP hiện có cho phép tính aggregation,
không phải thêm head, prediction output hoặc auxiliary classification loss.
Forward trả về bộ logits C đã enhancement và logits A gốc như interface cũ.
Với cùng H và cùng trọng số, bật/tắt hoặc thay trọng số CNN không đổi logits A.

Trong training, C/A vẫn chia sẻ các tham số vốn có trên đường aggregation;
nếu dùng `clip`, LGT cũng còn trainable. Vì thế AUC2 qua các epoch vẫn có thể
thay đổi gián tiếp, dù A không nhận H_enhanced hoặc gradient trực tiếp qua CNN.

## Đầu vào và loss

Người dùng xác nhận `.npy` đã xuất sau LGT: chọn `--input-features post_lgt`.
H được cấp trực tiếp cho C/A và CNN; không chạy LGT lần nữa. LGT offline không
được cập nhật. Nếu dùng feature CLIP chưa qua LGT, chọn `--input-features clip`;
CNN vẫn được gắn sau LGT online, gradient C có thể đi qua shared LGT.

```text
loss = CLAS2(logits_C, video_labels)
     + CLASM(logits_A, video_labels)
     + loss_text_original
```

Không tính loss memory, MSE hoặc loss phụ mới trong mode này. Các trường
`loss_memory`, `weighted_memory`, `loss_mse`, `weighted_mse`, `mse_pairs` được
giữ trong formatter cũ với giá trị 0 để đọc log tương thích.
CNN/projection/gate nằm cùng nhóm optimizer và cùng LR với các đầu gốc:
2e-5 trong lệnh dưới. `memory_lr` của các mode cũ không được dùng cho GRF.

## Cell train Kaggle — E2

Giữ notebook/setup/CSV/GT đang có, bật GPU và Internet. Hai CSV cần trỏ đúng
feature sau LGT đã xuất, giữ thứ tự và độ phân giải thời gian khớp GT.
Chỉ thay cell train cuối; mỗi khối dưới là một cell, một dấu ! ở đầu.

```python
!cd /kaggle/working/VadCLIP && \
git pull --ff-only origin main && \
python -u experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --input-features post_lgt \
  --aux-mode gated_residual --aux-branch c \
  --cnn-kernel 3 --cnn-channels 128 \
  --memory-start-epoch 1 \
  --memory-loss-weight 0 --mse-loss-weight 0 \
  --lr 2e-5 --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_grf_c_k3_ch128_s234_20261102_r1
```

`memory-start-epoch` là tên cờ phase cũ, trong mode này điều khiển bật feature
enhancement. Đặt 1 để học ngay từ epoch đầu, không có pretrain hoặc warm-up.
Trainer tự evaluate mỗi 10 update và cuối epoch bằng evaluator chung với Top-K/gốc.
Log mới có `feature_fusion=gated_residual`, `conv_layers=1`, `cnn_channels=128`,
`objective=vadclip_original`, `auxiliary_classifier=False mse=False`,
`a_features=original_h visual_prompt_weights=original_c`.

## Cell đối chứng — E0

Giữ mode, dữ liệu, input type, LR, seed, lịch eval và thời điểm chọn checkpoint;
chỉ đặt `--aux-branch none` để không tạo CNN/gate. Đây là baseline trên cùng
dataset cached sau LGT, không thêm LGT online.

```python
!cd /kaggle/working/VadCLIP && \
python -u experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --input-features post_lgt \
  --aux-mode gated_residual --aux-branch none \
  --cnn-kernel 3 --cnn-channels 128 \
  --memory-start-epoch 1 \
  --memory-loss-weight 0 --mse-loss-weight 0 \
  --lr 2e-5 --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_grf_e0_s234_20261102_r1
```

E1 bỏ gate, các kernel 5/7 và các hướng mở rộng khác trong bản thiết kế chưa
được bật trong baseline v1 này.

## Evaluate và checkpoint

```python
!cd /kaggle/working/VadCLIP && \
python -u experiments/cnn_memory/evaluate.py \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --checkpoint outputs/cnn_grf_c_k3_ch128_s234_20261102_r1/best.pt \
  --log-path outputs/cnn_grf_c_k3_ch128_s234_20261102_r1/evaluate.log
```

Evaluate/predict đọc mode, input type, cnn_channels và phase từ checkpoint;
CNN/gate tiếp tục chạy tại inference. Không có đầu tương lai cần bỏ đi.
`last.pt` chứa optimizer, scheduler và RNG để resume đúng run.
Không resume checkpoint memory/MSE bằng GRF hoặc đổi cnn_channels/input type.
Run mới cần output trống; đổi r1 thành r2 nếu đã có kết quả.

Các mode `prototype`, `future_mse`, `future_mse_joint` được giữ để tái hiện run
cũ; mặc định CLI chưa đổi. Cần chọn rõ `gated_residual` để chạy phương pháp mới.
Evaluator vẫn dùng raw score, repeat 16, AUC1/AP1, AUC2/AP2 và mAP; không video gate.
Trainer hiện vẫn chọn best theo **test AUC1**, giống protocol thí nghiệm trước;
đó không phải independent validation. Không dùng test để tune hyperparameter
hoặc khẳng định cải thiện từ một seed. Chưa có kết quả GPU/AUC mới của GRF.
