# VadCLIP + CNN Gated Residual Fusion

Thư mục này chỉ triển khai phương pháp đã chốt trong
[bản thiết kế](../../reports/cnn_memory/VadCLIP_CNN_Gated_Residual_Fusion_Method.md).
Các implementation prototype, future MSE, joint MSE và bước pretrain đã được xóa.

```text
H sau LGT [B,T,512] ────────────────────────────────> nhánh A gốc
     │
     └─> Conv1D(512,128,k3) → GELU → Linear(128,512) → F_cnn
         g = sigmoid(Linear(concat(H, F_cnn)))
         H_enhanced = H + g * F_cnn
         → mlp2 + classifier C gốc → logits_C
```

Mặc định dùng feature đã xuất sau LGT (`post_lgt`), CNN chỉ tăng cường C và
học chung từ epoch 1. A giữ H và trọng số aggregation tính từ C gốc.
CNN/gate dùng cùng learning rate với các đầu gốc. Loss chỉ gồm
`CLAS2(C) + CLASM(A) + text separation` gốc.

Xem **[cell train Kaggle, E0 và evaluate](GATED_RESIDUAL.md)**. Giữ các cell
setup/CSV của notebook hiện có, thay cell train cuối bằng lệnh mới. Notebook
cũ được giữ nguyên làm tư liệu; các cờ memory/MSE trong cell train cũ đã hết hiệu lực.
Đối chứng `--aux-branch none` tắt CNN/gate để so sánh trên cùng pipeline.

Các file chính:

- [feature_fusion.py](feature_fusion.py): một Conv1D, projection, gate, residual và mask padding.
- [cnn_memory_model.py](cnn_memory_model.py): ghép feature vào C, giữ đường A gốc.
- [config.py](config.py), [train.py](train.py): cấu hình GRF và loss gốc.
- [evaluate.py](evaluate.py): evaluator chung với Top-K/gốc, raw scores, repeat 16, AUC/AP/mAP.
- [checkpoint.py](checkpoint.py): checkpoint GRF format v2; đọc được GRF v1, từ chối checkpoint phương pháp đã xóa.

Trainer chọn checkpoint theo **test AUC1**, không phải validation độc lập.
`last.pt` lưu optimizer, scheduler và RNG để resume tại epoch kế tiếp. Dùng
output trống cho run mới. Chưa có kết quả GPU/AUC mới của GRF.

Kiểm tra CPU:

```bash
python -B -m unittest discover -s experiments/cnn_memory/tests -v
```

`src/`, notebook và log cũ được giữ nguyên. Thí nghiệm MSC độc lập giữ control
prototype của nó trong `experiments/msc_branch/prototype_*.py`.
