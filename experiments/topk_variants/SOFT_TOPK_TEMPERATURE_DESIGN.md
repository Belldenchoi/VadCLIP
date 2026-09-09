# Soft Top-K với nhiệt độ thay đổi theo epoch

## Mục tiêu thử nghiệm

Chỉ thay đổi nhiệt độ pooling của branch A để kiểm tra ảnh hưởng lên quá trình
học và epoch đạt AUC1 cao nhất. Chưa có kết quả thực nghiệm; không bảo đảm peak
sẽ cao hơn hoặc xuất hiện muộn hơn.

Giữ nguyên số K, cách chọn K snippet, các loss BCE/CE, learning rate và evaluator.
Branch C vẫn dùng Soft Top-K với nhiệt độ cố định 1.0. Đây là đối chứng với
**Soft Top-K nhiệt độ cố định trên cả hai branch**, không phải baseline mean Top-K.
Thí nghiệm đầu không bật Conv1D, segment, smoothness loss, AIS hoặc Dual-K.

## Công thức và ý nghĩa

Với mỗi lớp c, chọn tập I_c gồm K logits lớn nhất trong các snippet hợp lệ.
Với logit z_tc của snippet t, trọng số và logit video là:

\[
w_{tc}(e)=\frac{\exp(z_{tc}/\tau_A(e))}
{\sum_{j\in I_c}\exp(z_{jc}/\tau_A(e))},\qquad
v_c(e)=\sum_{t\in I_c}w_{tc}(e)z_{tc}.
\]

Loss branch A vẫn là:

\[
L_A(e)=-\frac1B\sum_{b=1}^B\sum_c y_{bc}
\log\operatorname{softmax}(v_b(e))_c.
\]

B là số video trong batch; y là nhãn video đã chuẩn hóa như CLASM hiện có.
Nhiệt độ chỉ dùng tính trọng số theo thời gian, không chia logits lần nữa trong
softmax theo lớp của CE. Không thêm loss mới hoặc hệ số lambda.

Với e là epoch tính từ 1, s và f là epoch bắt đầu/kết thúc:

\[
r(e)=\operatorname{clip}\left(\frac{e-s}{f-s},0,1\right),\qquad
\tau_A(e)=(1-r(e))\tau_{start}+r(e)\tau_{end}.
\]

Cấu hình đầu: s=1, f=4, nhiệt độ đầu 2.0, cuối 1.0.

| Epoch | Nhiệt độ A | Nhiệt độ C |
|---|---:|---:|
| 1 | 2.0000 | 1.0000 |
| 2 | 1.6667 | 1.0000 |
| 3 | 1.3333 | 1.0000 |
| 4–10 | 1.0000 | 1.0000 |

Với cùng logits, nhiệt độ cao phân bố trọng số đều hơn trong tập K đã chọn;
nhiệt độ thấp tập trung hơn vào score lớn. Ví dụ hai logits [1, 3] có trọng số
xấp xỉ [0.269, 0.731] khi nhiệt độ 2 và [0.119, 0.881] khi nhiệt độ 1.
Đây vẫn là chọn tập bằng hard Top-K rồi tính tổng có trọng số, không phải bộ
chọn khả vi trên toàn bộ video. Khi logits thay đổi trong quá trình học, mức
tập trung thực tế không nhất thiết tăng đơn điệu theo epoch.

## Cấu hình và mã nguồn

- `temperature_schedule.py`: kiểm tra cấu hình, tính nhiệt độ từ epoch tuyệt đối.
- `options.py`: thêm lịch A `constant` (mặc định) hoặc `linear`.
- `train_ucf.py`: truyền nhiệt độ của epoch vào CLASM; CLAS2 giữ nhiệt độ C.
- `topk_pooling.py`: giữ nguyên cách chọn K, softmax và gradient qua trọng số.

Các cờ mới:

- `--a-topk-temperature-schedule linear`: bật lịch; chỉ dùng với pooling `soft`.
- `--a-topk-temperature-start 2.0`: nhiệt độ đầu.
- `--a-topk-temperature-start-epoch 1`: epoch bắt đầu, tính từ 1.
- `--a-topk-temperature-end-epoch 4`: epoch đạt nhiệt độ cuối, lớn hơn epoch đầu.
- `--a-topk-temperature 1.0`: cờ có sẵn, là nhiệt độ cuối khi bật lịch.

Không bật lịch thì hành vi pooling cũ được giữ nguyên. Trước epoch đầu giữ
nhiệt độ đầu; sau epoch cuối giữ nhiệt độ cuối. Có thể tăng nhiệt độ thay vì
giảm bằng cách đổi hai giá trị đầu/cuối; thí nghiệm đầu chỉ dùng 2 xuống 1.
Log ghi cấu hình, nhiệt độ hai branch ở đầu mỗi epoch và trong log batch.

Checkpoint mới lưu cấu hình nhiệt độ. Resume sẽ kiểm tra cấu hình có khớp
không và tính lịch từ epoch được resume, không khởi động lại lịch từ 1.
Checkpoint cũ thiếu cấu hình sẽ bị từ chối nếu bật lịch: hãy chạy mới với tên
output riêng. Cơ chế checkpoint hiện tại không bảo đảm resume chính xác giữa
epoch; thay đổi này không sửa cơ chế đó.

Lưu ý đối chứng: train hiện tại nạp lại trọng số model tốt nhất sau mỗi epoch
khi có đánh giá; learning-rate scheduler vẫn tiến tiếp. Giữ nguyên chính sách
này trong cả hai lần chạy, không diễn giải đường AUC như train liên tục thuần túy.

## Lệnh Kaggle

```bash
!cd /kaggle/working/VadCLIP && \
mkdir -p outputs && \
python experiments/topk_variants/train_ucf.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --seed 234 \
  --batch-size 64 \
  --max-epoch 10 \
  --topk-pooling soft \
  --c-topk-temperature 1.0 \
  --a-topk-temperature 1.0 \
  --a-topk-temperature-schedule linear \
  --a-topk-temperature-start 2.0 \
  --a-topk-temperature-start-epoch 1 \
  --a-topk-temperature-end-epoch 4 \
  --temporal-smoothness-branch none \
  --model-path outputs/ucf_soft_a_tau_2to1.pth \
  --checkpoint-path outputs/ucf_soft_a_tau_2to1_checkpoint.pth \
  --log-path outputs/ucf_soft_a_tau_2to1.log
```

Đối chứng: đổi `linear` thành `constant` và đổi ba đường dẫn output sang tên
riêng (ví dụ `ucf_soft_fixed`). Các tham số khác giữ nguyên. Không bật resume
cho hai lần chạy mới. CLI hiện chưa có `--num-workers` hoặc `--eval-steps`;
không thêm các cờ không tồn tại vào lệnh.

So sánh AUC1 peak, epoch đạt peak và các epoch lân cận của mỗi lần chạy.
Một seed chỉ là thăm dò, chưa đủ kết luận cải thiện ổn định. Không chọn lịch
tối ưu bằng cách liên tục điều chỉnh theo điểm test.
