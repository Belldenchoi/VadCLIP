# Temporal-A/R + hai margin loss chuyển từ Finetune-VadCLIP

Hai công thức được chép nguyên từ `Finetune-VadCLIP/VadCLIP/src/ucf_train.py`
của repo [VanLinh2801/Finetune-VadCLIP](https://github.com/VanLinh2801/Finetune-VadCLIP),
checkout `330f3661edde133f09318ae1ed79e8af74894ca6`:

- `ranking_margin_loss`: hinge giữa **trung bình hai nhóm video** Normal/Abnormal.
- `abnormal_background_suppression_loss`: hinge giữa `min(high)` và `max(low)`
  trong từng video Abnormal. Tên hàm dùng chữ suppression nhưng công thức là separation.

Mã loss nằm trong [finetune_margin_loss.py](finetune_margin_loss.py), phần
ghép trainer/mask/warm-up/checkpoint trong [finetune_margin_adapter.py](finetune_margin_adapter.py).
Không chép trainer epoch-only, thay model, text prompt, dataset, CNN hoặc GRF từ repo nguồn.

## Mốc giữ nguyên

Baseline R theo `logs/segment_a_conv3_smooth_a.txt`, best-test AUC1 88,1147%:

- C: Hard Top-K mean gốc, K=floor(T/16)+1, không C-segment/smoothing.
- A: Conv1D mean cố định k3 + đoạn K liên tiếp tốt nhất theo lớp, từ epoch 1.
- TV-A=0,05 từ epoch 1; giữ LGT, Visual Prompt, loss C/A/text và LR 2e-5.
- Evaluator, lịch evaluate, chọn best-test AUC1 và reload model cuối epoch
  theo trainer Top-K hiện có. Không đổi sang policy epoch-only của Finetune.

Tổng loss là `L_R + lambda * L_imported`. Mặc định `--finetune-margin none`:
không tính loss mới, không thêm tham số model, không đổi baseline R.
Chỉ chọn **một** trong `ranking` hoặc `background` cho mỗi run.

## Tham số được giữ từ nguồn

Nhánh phụ dùng `sigmoid(C)` và `1-softmax(A)[Normal]` trên logits **raw**.
`both` tính riêng từng nhánh rồi trung bình hai loss, không trung bình score trước.
Mặc định `both` và ratio 0,15 như nguồn. Nếu muốn chạy C-only, đổi
`--regularize-score-branch classifier` và dùng output mới; A-only là `alignment`.

K_aux=max(1,floor(0,15*T)) chỉ dành cho loss mới. Với T=256, K_aux=38;
K của task C và độ dài đoạn A vẫn 17. Loss phụ không chuyển thành segment
pooling và không thay cơ chế segment của A.

Các cell dưới dùng đúng hệ số trong notebook nguồn
`code/train_description_guided_vadclip_colab.ipynb`:

| Run | Lambda | Margin | Warm-up ban đầu | Nhánh / ratio |
| --- | ---: | ---: | ---: | --- |
| Ranking Margin | 0,05 | 0,20 | 1 epoch | both / 0,15 |
| Background Separation | 0,05 | 0,15 | 1 epoch | both / 0,15 |

`warmup-epochs=1` nghĩa epoch 1 chưa cộng loss, **bắt đầu epoch 2**. Đây khác
gợi ý thử C-only/margin 0,1/warm-up 2 epoch trong bản phân tích ban đầu; các
cell này ưu tiên mang nguyên hai thí nghiệm nguồn sang baseline R.

## Kaggle: giữ notebook và các cell setup/CSV cũ

Bật GPU. Các CSV bên dưới trỏ tới feature CLIP tải sẵn, LGT vẫn chạy trong
model; không dùng `post_lgt`. Mỗi khối là một cell riêng, một dấu `!` đầu cell.
Chạy tuần tự; các tên output khác nhau. Nếu đã có output, tăng hậu tố `r1`.
Mỗi cell chỉ chạy **một** phương pháp, không tự chạy cả ba hoặc mọi seed.

### R0: đối chứng Temporal-A

```python
!cd /kaggle/working/VadCLIP && \
git pull --ff-only origin main && \
python -u experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --lr 2e-5 \
  --topk-pooling mean \
  --temporal-segment-topk --temporal-segment-start-epoch 1 \
  --c-temporal-smoothing-kernel 1 --temporal-smoothing-kernel 3 \
  --temporal-smoothness-branch a --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --model-path outputs/temporal_a_finetune_margin_20261106_r1/R_seed234/model.pth \
  --checkpoint-path outputs/temporal_a_finetune_margin_20261106_r1/R_seed234/checkpoint.pth \
  --log-path outputs/temporal_a_finetune_margin_20261106_r1/R_seed234/train.log
```

### R1: Temporal-A + Ranking Margin

```python
!cd /kaggle/working/VadCLIP && \
git pull --ff-only origin main && \
python -u experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --lr 2e-5 \
  --topk-pooling mean \
  --temporal-segment-topk --temporal-segment-start-epoch 1 \
  --c-temporal-smoothing-kernel 1 --temporal-smoothing-kernel 3 \
  --temporal-smoothness-branch a --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --finetune-margin ranking \
  --regularize-score-branch both --regularize-top-k-ratio 0.15 \
  --lambda-ranking-margin 0.05 --ranking-margin 0.20 --ranking-warmup-epochs 1 \
  --model-path outputs/temporal_a_finetune_margin_20261106_r1/R1_ranking_seed234/model.pth \
  --checkpoint-path outputs/temporal_a_finetune_margin_20261106_r1/R1_ranking_seed234/checkpoint.pth \
  --log-path outputs/temporal_a_finetune_margin_20261106_r1/R1_ranking_seed234/train.log
```

### R2: Temporal-A + Abnormal Background Separation

```python
!cd /kaggle/working/VadCLIP && \
git pull --ff-only origin main && \
python -u experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 --lr 2e-5 \
  --topk-pooling mean \
  --temporal-segment-topk --temporal-segment-start-epoch 1 \
  --c-temporal-smoothing-kernel 1 --temporal-smoothing-kernel 3 \
  --temporal-smoothness-branch a --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --finetune-margin background \
  --regularize-score-branch both --regularize-top-k-ratio 0.15 \
  --lambda-abnormal-bg 0.05 --abnormal-bg-margin 0.15 --abnormal-bg-warmup-epochs 1 \
  --model-path outputs/temporal_a_finetune_margin_20261106_r1/R2_background_seed234/model.pth \
  --checkpoint-path outputs/temporal_a_finetune_margin_20261106_r1/R2_background_seed234/checkpoint.pth \
  --log-path outputs/temporal_a_finetune_margin_20261106_r1/R2_background_seed234/train.log
```

Giữ cùng seed/GPU/feature/CSV cho R0/R1/R2; để ghép seed, thay `234` trong
tham số và output thành `3407` rồi `2026`, chạy thành các run riêng.

## Log và checkpoint

Header run mới ghi `finetune_margin_config`, epoch ghi mode và active True/False.
Tiến độ vẫn giữ `loss1/2/3`, `loss_smooth_a`, AUC1/AP1, AUC2/AP2 và mAP.
Các trường mới gồm loss thô, weighted loss, weight, gap, violation fraction và
số video đủ điều kiện. Tổng `loss` đã cộng cả TV-A và weighted margin.
Warm-up không tính loss mới, không tác động thứ tự dữ liệu/RNG.

Cấu hình mode/nhánh/ratio/margin/weight/warm-up được lưu trong checkpoint;
resume hoặc reload best khác cấu hình sẽ bị từ chối. Với resume cùng cấu hình,
giữ mọi cờ và đường dẫn, chỉ thêm `--use-checkpoint True`; `--max-epoch` có thể tăng.
Không dùng checkpoint R0 để resume thành R1/R2 hoặc đổi giữa hai loại margin.

Wrapper mới kiểm tra valid lengths và mask padding trước sigmoid/softmax;
không chọn padding. Khi không đủ video hai nhóm hoặc video quá ngắn, loss
bằng 0. Background giữ quy tắc nguồn: bỏ video có hai nhóm Top-K chồng lấn
theo số lượng; ratio > 0,5 bị từ chối sớm để tránh một run không có tác dụng.

`--c-ranking-loss` và `--snippet-ranking-loss` cũ vẫn giữ nguyên và mặc định
tắt. Không bật chung chúng, PCA, Dual-K, AIS hoặc tail shift với loss mới.
Normal suppression, hard-pair soft ranking và các nhận xét mở rộng min/max/
average trong Word chưa được thêm vào hai thí nghiệm này.

## Kiểm tra và giới hạn

```bash
python -B -m unittest discover -s experiments/topk_variants/tests -p test_finetune_margin.py -v
```

Kiểm tra CPU gồm công thức, gradient, lựa chọn nhánh, padding, warm-up,
resume và trainer Temporal-A. Các hàm nguồn được so trực tiếp về AST/value/
gradient khi checkout Finetune có sẵn; kiểm tra công thức vẫn chạy độc lập.
Chưa chạy GPU/Kaggle/UCF-Crime với hai kết hợp này, chưa có AUC mới.
88,1147% là best-test một run baseline, không phải cam kết kết quả chạy lại.
