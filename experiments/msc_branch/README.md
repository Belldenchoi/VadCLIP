# Nhánh representation phụ sau LGT

MLP `512 → 256 → 128` học bằng contrastive thường hoặc mean-shifted contrastive,
song song với C/A và CNN memory. Gradient loss mới đi về LGT dùng chung.
C giữ BCE, A giữ cross-entropy; projection MLP không tham gia inference.

- [Thiết kế, công thức, tạo cặp và tâm Normal](../../reports/cnn_memory/THIET_KE_NHANH_MSC_2026-10-26.md).
- [Lệnh Kaggle cho none/standard/mean_shifted và ba seed](../../reports/cnn_memory/LENH_KAGGLE_NHANH_MSC_2026-10-26.md).

Code mới tách trong thư mục này. Dùng lại dataset, CNN memory và evaluator;
wrapper CNN có tùy chọn trả feature chung, mặc định giữ output cũ. Không sửa
`src/`. Cần code CNN memory trong cùng checkout để chạy.

Train và evaluate checkpoint dùng chung `experiments/topk_variants/test_ucf.py::test`
qua adapter CNN: giữ chunk/mask của bản gốc, raw sigmoid C và 1-softmax Normal
của A, repeat 16, detection mAP tại IoU 0.1–0.5 và format log Top-K.
MLP contrastive không chạy lúc test. Cần cập nhật đồng bộ `msc_branch`,
`cnn_memory` và evaluator Top-K trong cùng checkout.

```bash
python experiments/msc_branch/train.py --help
python -B -m unittest discover -s experiments/msc_branch/tests -v
```

Chạy mean-shifted trên Kaggle, sau khi có code mới và hai CSV đã remap:

```bash
!cd /kaggle/working/VadCLIP && \
python -u experiments/msc_branch/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --aux-branch both --cnn-kernel 3 --seed 234 --batch-size 64 --max-epoch 10 \
  --contrastive-mode mean_shifted --contrastive-start-epoch 3 \
  --contrastive-weight 0.01 --contrastive-temperature 0.1 \
  --representation-hidden 256 --representation-dim 128 --representation-lr 0.0002 \
  --center-videos 128 --center-snippets 32 --center-seed 0 \
  --output-dir outputs/msc_20261026_r1/mean_shifted_seed234
```

Đối chứng dùng cùng lệnh, đổi cả mode và output:

| Mode | Ý nghĩa | Output |
|---|---|---|
| `none` | Tắt loss mới | `outputs/msc_20261026_r1/none_seed234` |
| `standard` | Contrastive không trừ tâm | `outputs/msc_20261026_r1/standard_seed234` |
| `mean_shifted` | Contrastive có trừ tâm Normal | `outputs/msc_20261026_r1/mean_shifted_seed234` |

Chạy cả ba mode với seed 234, rồi thay mọi `234` bằng `3407`, sau đó `2026`.
Không ghi đè run cũ. Có thể thêm `--preflight` để kiểm tra dữ liệu trước train.
Mỗi khối lệnh là một cell, không chèn dòng trống sau ký tự nối dòng.
Resume cuối epoch: cùng lệnh và output, thêm `--resume outputs/.../last.pt`.
Test lại bằng `experiments/msc_branch/evaluate.py --checkpoint outputs/.../best.pt`
và cùng `--test-list`; evaluator lấy kiến trúc và tâm từ checkpoint mới.

Tâm chỉ lấy từ Normal train, cố định và lưu trong checkpoint. Cặp positive
phải cùng nhãn sự kiện và khác video nguồn; các crop `__0`, `__1` cùng nguồn
không được coi là hai positive độc lập. Có thể thêm cột `video_id` vào CSV
nếu tên feature không theo quy ước crop này. Loss mới bật từ epoch 3;
log ghi số anchor có/thiếu partner và số snippet quá gần tâm bị loại.

Trainer giữ policy CNN: train liên tục, eval mỗi 10 update và cuối epoch,
best-test AUC1 từ epoch 3; đây chưa phải đánh giá validation độc lập.
So delta ghép seed và mean/std; các log cũ khác trainer không phải control trực tiếp.
Các liên kết báo cáo ở trên là tài liệu local trong `reports/` (Git bỏ qua);
lệnh chạy và hướng dẫn chính cũng có sẵn trong README này.

Các tests dùng fixture CPU và CLIP giả để kiểm tra số học/gradient/resume.
14 kiểm thử MSC và 18 kiểm thử CNN đã qua. Mode none khớp từng bit với
trainer CNN trên fixture; resume mean-shifted khớp train liên tục, tâm không
fit lại. Fixture train dùng dữ liệu giả và mAP giả lập; fixture hồi quy evaluator
riêng tính cả mAP thật, khớp evaluator UCF gốc và Top-K trên dữ liệu giả đủ 14 lớp.
Chưa có kết quả huấn luyện GPU hoặc bằng chứng tăng AUC của nhánh mới.
