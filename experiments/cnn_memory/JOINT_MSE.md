# CNN + MSE học chung với C/A

Mode `future_mse_joint` trong `experiments/cnn_memory`. Với dataset người dùng
đã xuất **sau LGT**, dùng `--input-features post_lgt`: C/A và CNN nhận trực tiếp
cùng feature đã lưu, không chạy LGT lần thứ hai. Giữ nguyên src, classifier C,
aggregation/Visual Prompt và dot product của A. Không cần
`pretrain.py`, checkpoint baseline hoặc `--pretrained-cnn`. Không tạo notebook mới.

Mặc định CLI vẫn là `--input-features clip` để giữ cách chạy/checkpoint cũ:
CLIP feature đầu vào được đưa qua LGT online. Shape 512 chiều không đủ để tự
phân biệt hai loại feature; phải khai báo đúng nguồn dữ liệu.

## Cách học trong từng batch

```text
LGT checkpoint -> xuất trước thành F_cached.npy

F_cached -> C/A gốc -> cộng residual r -> loss C/A
         |
         +-> CNN -> z -> đầu chấm điểm -> r -> loss MIL video
                    |
                    +-> đầu P -> dự đoán F_cached[t+horizon] -> MSE
```

CNN, đầu chấm điểm, đầu P và các đầu C/A/prompt trainable cùng nằm trong một
optimizer; mỗi batch chỉ gọi backward/step một lần:

```text
loss = loss_C + loss_A + loss_text
     + memory_loss_weight * loss_CNN_video
     + mse_loss_weight * loss_MSE
```

- Encoder CNN không đóng băng. MSE và các loss phát hiện bất thường cùng học
  biểu diễn z; gradient MSE đi qua CNN/P, không đi qua LGT hoặc text gốc.
- Đầu chấm điểm r khởi tạo bằng 0 để forward đầu khớp baseline. Ở update đầu,
  MSE có thể cập nhật CNN trong khi đầu chấm điểm bắt đầu học; các update sau
  loss C/A và video có đường gradient qua đầu chấm điểm về CNN.
- Với `post_lgt`, CNN nhận chính feature sau LGT mà C/A đang dùng. Các module
  LGT được đóng băng và loại khỏi optimizer; checkpoint vẫn giữ các tensor này
  để tương thích state dict. Không thêm positional embedding hoặc temporal/GCN
  lên feature đã cache.
- Target MSE là `F_cached[t+horizon].detach()`. CNN chỉ dùng các bước feature
  hiện tại/quá khứ, nhưng bản thân F_cached có thể đã chứa ngữ cảnh tương lai
  từ LGT lúc xuất. Đây là loss phụ học biểu diễn, không bảo đảm nhân quả theo
  video gốc. Với `clip`, CNN/target vẫn dùng CLIP feature trước LGT như bản cũ.
- Mask padding trước C/A và CNN; chuẩn hóa có epsilon để chunk toàn padding
  hoặc vector zero không tạo NaN. Công thức C/A trên các feature hợp lệ giữ nguyên.
- Horizon tính theo bước feature sau process_feat, không phải frame gốc.
  Padding bị loại khỏi MSE. Batch không có cặp tương lai vẫn học MIL và ghi
  `mse_pairs=0`; không tạo target giả hoặc bỏ batch đó.
- Visual Prompt vẫn dùng logit C gốc trước residual. A giữ logit Normal và
  cộng cùng bias r vào các lớp bất thường; không thêm text/prototype vào CNN.
- Trong evaluate/inference, đầu P không chạy và không tính MSE. Đầu P vẫn nằm
  trong checkpoint để resume đầy đủ. Evaluator Top-K/gốc, raw score, repeat 16,
  AUC1/AP1, AUC2/AP2 và mAP không đổi.
- Nhánh phụ/loss mới bắt đầu tại memory_start_epoch; lệnh dưới đặt 1 để học
  chung ngay từ epoch đầu. CLI chung vẫn giữ mặc định 3 cho các lệnh cũ.

## Kaggle: giữ notebook và setup cũ

Dùng notebook `temporal-2-branch.ipynb` đang có, giữ các cell setup/CSV/GT.
Notebook tạo hai CSV `/kaggle/working/ucf_CLIP_rgb_kaggle.csv` và
`/kaggle/working/ucf_CLIP_rgbtest_kaggle.csv`; GT nằm trong `VadCLIP/list`.
Bật GPU và Internet. Repo là
[Belldenchoi/VadCLIP](https://github.com/Belldenchoi/VadCLIP).
Lệnh dưới cần checkout có hỗ trợ `--input-features post_lgt`.

Chỉ thay cell train cuối; không cần nạp ZIP hoặc sửa notebook setup.
Nếu checkout Kaggle đã clone từ trước, cập nhật code trước khi train:

```bash
!git -C /kaggle/working/VadCLIP pull --ff-only origin main
```

Không cần vá LR trong `src/ucf_option.py` để chạy thí nghiệm vì parser riêng
đã có `type=float`.

```python
!cd /kaggle/working/VadCLIP && \
python -u experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --aux-mode future_mse_joint --aux-branch both \
  --input-features post_lgt \
  --cnn-kernel 3 --feature-dim 256 \
  --future-horizon 1 --mse-loss-weight 0.1 \
  --memory-start-epoch 1 \
  --lr 2e-5 --memory-lr 2e-4 \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir outputs/cnn_memory_joint_mse_post_lgt_20261031_r1
```

CNN/MSE học chung với C/A từ epoch 1 trong 10 epoch. Trainer tự evaluate
mỗi 10 optimizer update và cuối epoch, dùng evaluator chung với Top-K/gốc;
không cần gọi một lệnh evaluate riêng để có AUC/AP/mAP trong log train.

Giữ một dấu `!` đầu cell và các dấu nối dòng `\`; không thêm dòng trống
sau `&& \`. MSE 0.1 là cấu hình thử, chưa tối ưu trên validation.
LR: VadCLIP 2e-5, CNN/đầu mới 2e-4; scheduler mốc 4/8.

Log/checkpoint nằm tại `/kaggle/working/VadCLIP/outputs/cnn_memory_joint_mse_post_lgt_20261031_r1/`:
`train.log`, `metrics.jsonl`, `best.pt`, `last.pt`, `config.json`.
Chạy mới lần nữa dùng output trống, ví dụ đổi `20261031_r1` thành `20261031_r2`.
Hai CSV phải trỏ tới feature sau LGT đã xuất, giữ chiều thời gian tương ứng
với GT. Preflight vẫn kiểm tra tổng số bước test nhân 16 khớp frame GT.

## Evaluate lại best.pt (tùy chọn)

Cell train đã ghi metric; cell riêng dưới chỉ dùng khi muốn đánh giá lại checkpoint.

```python
!cd /kaggle/working/VadCLIP && \
python -u experiments/cnn_memory/evaluate.py \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --checkpoint outputs/cnn_memory_joint_mse_post_lgt_20261031_r1/best.pt \
  --log-path outputs/cnn_memory_joint_mse_post_lgt_20261031_r1/evaluate.log
```

## Control tắt riêng MSE, vẫn giữ CNN trainable

Control này kiểm tra đóng góp của MSE: giữ CNN, residual và loss video, đặt
mse_loss_weight=0. Mô hình đối chứng không phải baseline VadCLIP không có CNN.

```python
!cd /kaggle/working/VadCLIP && \
python -u experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --aux-mode future_mse_joint --aux-branch both \
  --input-features post_lgt \
  --cnn-kernel 3 --feature-dim 256 \
  --future-horizon 1 --mse-loss-weight 0 --memory-start-epoch 1 \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --output-dir /kaggle/working/outputs/cnn_memory_joint_mse_post_lgt_off_20261031_r1
```

Baseline không CNN có thể dùng cùng mode, `--aux-branch none` và output riêng.
Giữ nguyên CSV, seed, scheduler và policy chọn checkpoint khi so sánh.

## Log và resume

- `loss1/2/3` và `loss_memory/weighted_memory` giữ nghĩa cũ.
- `loss_mse`, `weighted_mse`, `mse_pairs` xuất hiện ở dòng batch;
  epoch_summary có `avg_loss_mse`, `avg_weighted_mse` và tổng số cặp.
- Header ghi `input_features=post_lgt lgt_applied=False`; log bổ sung
  `online_lgt=False lgt_trainable=False ca_and_cnn_share_input=True` và
  `mse_target=post_lgt_feature` để xác nhận đúng đường đi.
- Trung bình loss theo batch khớp cách báo loss cũ. Không diễn giải nó thành
  MSE trung bình theo mọi cặp của toàn epoch nếu các batch có số cặp khác nhau.
- best.pt vẫn chọn theo **test AUC1**, không phải validation độc lập;
  last.pt lưu đầy đủ CNN, predictor P, C/A, optimizer, scheduler và RNG.
- Loại đầu vào được lưu trong `model_config` và `train_args`. Evaluate/predict
  tự đọc từ checkpoint, không cần thêm cờ. Checkpoint cũ thiếu trường này được
  hiểu là `clip`, không tự đổi kiến trúc. `post_lgt` hỗ trợ cả CPU và GPU;
  chế độ `clip` vẫn yêu cầu CUDA do LGT gốc.
- Resume dùng đúng loại đầu vào/mode/horizon/trọng số MSE và cài đặt cũ, cùng output, thêm
  `--resume <output>/last.pt`. Không resume checkpoint hai giai đoạn bằng mode joint.
- Không resume run `clip` cũ bằng `post_lgt` vì đường tính và nhóm optimizer
  đã thay đổi. Chạy một output mới để so sánh kết quả.
- `--aux-mode future_mse` vẫn giữ bản pretrain + encoder đóng băng;
  `--aux-mode prototype` giữ thí nghiệm prototype cũ. Mặc định CLI không đổi.

Chưa chạy GPU hoặc có kết quả AUC mới. Kiểm tra CPU chỉ xác nhận wiring, gradient,
masking, checkpoint/resume và tương thích evaluator.
