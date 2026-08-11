# Temporal-aware Segment Top-K (thí nghiệm đầu tiên)

## Phạm vi

- C-branch luôn giữ hard Top-K gốc của VadCLIP.
- Trước epoch kích hoạt, A-branch cũng dùng hard Top-K gốc.
- Từ epoch kích hoạt, A-branch chọn một đoạn liên tục dài
  `K = floor(T_valid / 16) + 1` thay cho K frame rời rạc.
- Mỗi class trong A-branch chọn đoạn riêng dựa trên logits của class đó.
- Conv1D smoothing là tùy chọn, dùng kernel trung bình cố định và không thêm tham số học.
- Không kết hợp phương pháp này với Soft Top-K, Multi-K hoặc AIS trong thí nghiệm đầu tiên.

## Công thức

Với A-branch logits `S` có shape `[T, C]`, class `c` chọn:

```text
t*_c = argmax_t mean(S_smooth[t:t+K, c])
z_c  = mean(S_smooth[t*_c:t*_c+K, c])
```

Nếu `--temporal-smoothing-kernel 1`, `S_smooth = S`. Với kernel lớn hơn 1,
smoothing dùng depthwise Conv1D theo thời gian và replicate padding.

## Lệnh chạy đề xuất

Không smoothing:

```bash
python experiments/topk_variants/train_ucf.py \
  --topk-pooling mean \
  --temporal-segment-topk \
  --temporal-segment-start-epoch 6 \
  --temporal-smoothing-kernel 1 \
  --model-path outputs/ucf_temporal_segment.pth \
  --checkpoint-path outputs/ucf_temporal_segment_checkpoint.pth \
  --log-path outputs/ucf_temporal_segment.log
```

Conv1D smoothing kernel 5:

```bash
python experiments/topk_variants/train_ucf.py \
  --topk-pooling mean \
  --temporal-segment-topk \
  --temporal-segment-start-epoch 6 \
  --temporal-smoothing-kernel 5 \
  --model-path outputs/ucf_temporal_segment_smooth5.pth \
  --checkpoint-path outputs/ucf_temporal_segment_smooth5_checkpoint.pth \
  --log-path outputs/ucf_temporal_segment_smooth5.log
```

Với lịch 10 epoch hiện tại, epoch 1–5 dùng baseline hard Top-K cho cả hai branch;
epoch 6–10 chỉ đổi A-branch sang temporal segment.

## Ablation tối thiểu

1. Baseline hard Top-K cho cả C/A branch.
2. Temporal segment từ epoch 6, không smoothing.
3. Temporal segment từ epoch 6, smoothing kernel 3.
4. Temporal segment từ epoch 6, smoothing kernel 5.

Giữ nguyên seed, dữ liệu, batch size, learning rate và evaluator giữa bốn cấu hình.
