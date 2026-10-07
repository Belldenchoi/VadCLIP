# Lệnh Kaggle: snippet ranking và PCA

Đọc [thiết kế và công thức](THIET_KE_SNIPPET_RANKING_PCA_2026-10-07.md).
Các lệnh dưới cần **code có thay đổi mới** trong notebook Kaggle; clone phiên
bản cũ sẽ chưa có các cờ snippet/PCA. Đây là lệnh train mới, chưa được chạy GPU.

## Cách chạy ghép seed

Chạy lần lượt R, S, SP với seed 234. Sau đó copy cả ba cell và thay mọi
`234` thành `3407`, rồi thành `2026` (cả seed và đường dẫn output): tổng 9 run.
Mỗi cell dùng một GPU. Không chạy đồng thời nhiều cell trên cùng GPU.
Nếu chạy lại cùng seed, đổi `t4_snippet_pca_r1` thành `t4_snippet_pca_r2`.
Không dùng thư mục có model/checkpoint/log từ lần chạy trước.

Kiểm tra GPU thực tế và lưu version torch/CUDA/cuDNN, commit + diff trước khi
chạy. Giữ cùng môi trường và các file feature/CSV giữa ba hướng. Lệnh giả định
repo và CSV đã có tại các đường dẫn dưới, như các lần Kaggle trước.
Không thêm AMP, accumulation, skip-eval hoặc thay LR vào bộ so sánh này.

## R — đối chứng temporal A

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --topk-pooling mean \
  --temporal-segment-topk --temporal-segment-start-epoch 1 \
  --temporal-smoothing-kernel 3 \
  --temporal-smoothness-branch a --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --model-path outputs/t4_snippet_pca_r1/R_seed234/model.pth \
  --checkpoint-path outputs/t4_snippet_pca_r1/R_seed234/checkpoint.pth \
  --log-path outputs/t4_snippet_pca_r1/R_seed234/train.log
```

## S — R + ranking từng snippet

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --topk-pooling mean \
  --temporal-segment-topk --temporal-segment-start-epoch 1 \
  --temporal-smoothing-kernel 3 \
  --temporal-smoothness-branch a --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --snippet-ranking-loss --snippet-ranking-mode hinge \
  --snippet-ranking-margin 0.1 --snippet-ranking-weight 0.1 \
  --snippet-ranking-start-epoch 3 \
  --model-path outputs/t4_snippet_pca_r1/S_seed234/model.pth \
  --checkpoint-path outputs/t4_snippet_pca_r1/S_seed234/checkpoint.pth \
  --log-path outputs/t4_snippet_pca_r1/S_seed234/train.log
```

## SP — S + PCA trên representation

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 --batch-size 64 --max-epoch 10 \
  --topk-pooling mean \
  --temporal-segment-topk --temporal-segment-start-epoch 1 \
  --temporal-smoothing-kernel 3 \
  --temporal-smoothness-branch a --temporal-smoothness-start-epoch 1 \
  --a-temporal-smoothness-weight 0.05 \
  --snippet-ranking-loss --snippet-ranking-mode hinge \
  --snippet-ranking-margin 0.1 --snippet-ranking-weight 0.1 \
  --snippet-ranking-start-epoch 3 \
  --pca-loss --pca-weight 0.01 --pca-margin 0.1 --pca-rank 64 \
  --pca-start-epoch 3 --pca-fit-videos 128 --pca-fit-snippets 32 --pca-fit-seed 0 \
  --model-path outputs/t4_snippet_pca_r1/SP_seed234/model.pth \
  --checkpoint-path outputs/t4_snippet_pca_r1/SP_seed234/checkpoint.pth \
  --log-path outputs/t4_snippet_pca_r1/SP_seed234/train.log
```

## Đọc log

Hai epoch đầu S/SP chưa có loss mới. Từ epoch 3:

- S/SP: `loss_snippet`, `weighted_snippet`, `snippet_violation` là loss thô,
  loss có trọng số và tỷ lệ vi phạm margin (trung bình trong video trước).
- SP: một dòng `pca_fit` ghi số mẫu, rank, phương sai giải thích và epoch fit;
  `pca_normal`, `pca_residual`, `weighted_pca`, `pca_violation` theo dõi loss mới.
- `epoch_summary` ghi các giá trị trung bình batch với tiền tố `avg_`.
- AUC1/AUC2/AP/mAP vẫn do evaluator hiện tại tính bằng score gốc.

So S−R và SP−S **trong từng seed** trước khi tổng hợp mean/std. Không lấy
đỉnh AUC1 seed 234 của phương pháp mới so với seed khác của đối chứng.
Không diễn giải loss giảm thành bằng chứng AUC tăng.

## Softplus tùy chọn sau bộ so sánh chính

Để kiểm tra việc làm mềm hàm phạt, copy cell S, đổi
`--snippet-ranking-mode hinge` thành
`--snippet-ranking-mode softplus --snippet-ranking-temperature 0.05`,
đổi cả ba thư mục output `S_seed...` thành `S_softplus_seed...`.
Giữ mọi tham số còn lại và chạy cùng các seed. Cờ này không bật Soft Top-K.

Checkpoint có cấu hình snippet/PCA và basis để resume đúng loss. Tuy nhiên
checkpoint best của protocol lịch sử không chứa đầy đủ trạng thái phục hồi
đúng batch bị ngắt. Bộ so sánh chính dùng train mới; không thêm
`--use-checkpoint` hoặc `--use-checkpoint False`.
