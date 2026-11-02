# CNN + Gated Residual Fusion — baseline v1 đã chốt

Triển khai thiết kế người dùng chốt ngày 02/11/2026 trong
`reports/cnn_memory/VadCLIP_CNN_Gated_Residual_Fusion_Method.md`.
Đây là phương pháp duy nhất trong `cnn_memory`, mặc định bật cho nhánh C.
Code nằm trong thí nghiệm `cnn_memory`; không sửa `src/` hoặc notebook.

## Cách làm

```text
Feature CLIP .npy tải sẵn -> LGT gốc -> H
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

Dataset tải từ repo gốc là **feature CLIP trước LGT**, như
[README tác giả](https://github.com/nwpu-zxr/VadCLIP#setup) mô tả. Trong
[forward gốc](https://github.com/nwpu-zxr/VadCLIP/blob/main/src/model.py), feature
còn đi qua `encode_video()` (temporal Transformer và graph). Vì vậy mặc định
và lệnh Kaggle dùng **`--input-features clip`**: LGT được học chung với C/A/CNN.
CNN vẫn nhận H sau LGT; không chạy CLIP image encoder lại trên video thô.

`--input-features post_lgt` chỉ dùng cho `.npy` tự xuất trực tiếp từ
`encode_video()` của một checkpoint. Chế độ đó bỏ qua và đóng băng LGT online.
Hai loại feature có thể đều 512 chiều; shape không xác định được loại đầu vào.
Tên CSV giống nhau cũng không chứng minh file bên trong đã qua LGT.

```text
loss = CLAS2(logits_C, video_labels)
     + CLASM(logits_A, video_labels)
     + loss_text_original
```

Chỉ tính ba loss gốc. Log giữ cấu trúc `epoch`, `batch`, `loss`, `loss1`,
`loss2`, `loss3`, `lr`, `peak_vram` như các thí nghiệm trước. Các trường loss
memory/MSE đã được bỏ khỏi log.
CNN/projection/gate nằm cùng nhóm optimizer và cùng LR với các đầu gốc:
2e-5 trong lệnh dưới.

## Cell train Kaggle — E2

Giữ notebook/setup/CSV/GT đang có, bật GPU và Internet. Hai CSV trỏ tới feature
CLIP tải sẵn, giữ thứ tự và độ phân giải thời gian khớp GT.
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
  --input-features clip \
  --aux-branch c \
  --cnn-kernel 3 --cnn-channels 128 \
  --cnn-start-epoch 1 \
  --lr 2e-5 --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_grf_clip_c_k3_ch128_s234_20261102_r3
```

`--cnn-start-epoch` mặc định 1: CNN/gate học ngay từ epoch đầu cùng C/A.
Trainer tự evaluate mỗi 10 update và cuối epoch bằng evaluator chung với Top-K/gốc.
Log mới có `feature_fusion=gated_residual`, `conv_layers=1`, `cnn_channels=128`,
`objective=vadclip_original`, `auxiliary_classifier=False mse=False`,
`a_features=original_h visual_prompt_weights=original_c`, `lgt_applied=True`,
`online_lgt=True lgt_trainable=True`.

## Cell đối chứng — E0

Giữ mode, dữ liệu, input type, LR, seed, lịch eval và thời điểm chọn checkpoint;
chỉ đặt `--aux-branch none` để không tạo CNN/gate. Đây là baseline cùng dataset
feature CLIP, có LGT online như E2.

```python
!cd /kaggle/working/VadCLIP && \
python -u experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --input-features clip \
  --aux-branch none \
  --cnn-kernel 3 --cnn-channels 128 \
  --cnn-start-epoch 1 \
  --lr 2e-5 --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_grf_clip_e0_s234_20261102_r3
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
  --checkpoint outputs/cnn_grf_clip_c_k3_ch128_s234_20261102_r3/best.pt \
  --log-path outputs/cnn_grf_clip_c_k3_ch128_s234_20261102_r3/evaluate.log
```

Evaluate/predict đọc mode, input type, cnn_channels và phase từ checkpoint;
CNN/gate tiếp tục chạy tại inference. Không có đầu tương lai cần bỏ đi.
`last.pt` chứa optimizer, scheduler và RNG để resume đúng run.
Không resume checkpoint memory/MSE bằng GRF hoặc đổi cnn_channels/input type.
Run mới cần output trống; tăng hậu tố `r3` nếu đã có kết quả.
Không đổi loại đầu vào rồi resume `last.pt` của run `post_lgt`: hãy train run mới.
Evaluate vẫn đọc loại đầu vào đã lưu, để tái hiện đúng checkpoint cũ.

Đã xóa implementation, pretrain, loss, cờ CLI và hướng dẫn chạy của các mode
`prototype`, `future_mse`, `future_mse_joint` khỏi `cnn_memory`. Lệnh cũ có các
cờ memory/MSE sẽ báo lỗi; dùng các cell trên. Checkpoint các phương pháp cũ
được từ chối rõ ràng. Checkpoint GRF format v1 được chuyển tên trạng thái sang
`enhancement_active` khi nạp; checkpoint mới dùng format v2. Khi resume GRF v1,
thay cờ `--memory-start-epoch` bằng `--cnn-start-epoch` và bỏ các cờ memory/MSE.

MSC là thí nghiệm độc lập: phần prototype mà nó cần đã được chuyển về
`experiments/msc_branch/prototype_*.py`, giữ nguyên phép tính/checkpoint của MSC.
Notebook và log cũ được giữ làm tư liệu; chỉ dùng các cell setup của notebook
cũ rồi thay cell train bằng lệnh trong tài liệu này. Code cũ có trong Git history.
Evaluator vẫn dùng raw score, repeat 16, AUC1/AP1, AUC2/AP2 và mAP; không video gate.
Trainer hiện vẫn chọn best theo **test AUC1**, giống protocol thí nghiệm trước;
đó không phải independent validation. Không dùng test để tune hyperparameter
hoặc khẳng định cải thiện từ một seed. Run `post_lgt` ở `logs/train.txt` đã
hoàn tất nhưng bỏ LGT trên feature CLIP; chưa có kết quả GPU của lệnh `clip`
đã sửa. Run đó không đủ để kết luận GRF làm giảm chất lượng khi giữ LGT.
