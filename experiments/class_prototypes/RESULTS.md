# Kết quả thí nghiệm Direct Class Prototype trên UCF-Crime

## 1. Phạm vi báo cáo

Báo cáo này tổng hợp baseline VadCLIP gốc và log huấn luyện của năm cấu hình Class
Prototype A0–A4. Các thí nghiệm Temporal Segment Top-K, AIS và những biến thể Top-K
khác không thuộc phạm vi so sánh.

Class Prototype thay Learnable Prompt và Visual Prompt của A-branch bằng prototype cố
định được tạo offline. Temporal encoder, C-branch và MIL hard Top-K gốc được giữ lại.

## 2. Xây dựng Class Prototype

Class Prototype được tạo **offline trước khi huấn luyện**, không được sinh lại ở từng
batch và cũng không được cập nhật bởi optimizer. Toàn bộ quá trình gồm các bước sau.

### Bước 1 — Chuẩn bị description

UCF-Crime có 14 lớp và mỗi lớp được mô tả bằng 20 câu ngắn tập trung vào chủ thể và
hành động. Code kiểm tra description rỗng, câu trùng lặp và các mạo từ đã quy định
trước khi xử lý. Thứ tự 14 lớp trong file description được giữ nguyên để đồng bộ với
label của dataset và evaluator.

### Bước 2 — Mã hóa văn bản bằng CLIP

Tất cả description và tên lớp được mã hóa một lần bằng CLIP `ViT-B/16` Text Encoder.
CLIP được đóng băng và chạy trong chế độ inference, do đó bước này không học thêm tham
số. Embedding của từng câu được chuẩn hóa trước khi chuyển sang bước đánh giá.

Tên lớp cũng được mã hóa để làm mốc ngữ nghĩa. Riêng `roadAccidents` được chuyển thành
`road accidents` khi mã hóa để phù hợp với văn bản tự nhiên.

### Bước 3 — Chấm điểm và xếp hạng description

Mỗi description được đánh giá theo ba tiêu chí:

- `name_score`: câu có gần với ý nghĩa tên lớp hay không;
- `intra_score`: câu có đồng thuận với nội dung chung của các description còn lại trong
  cùng lớp hay không; khi đánh giá một câu, chính câu đó không được dùng để tạo nội
  dung tham chiếu;
- `inter_score`: câu có bị gần với một lớp khác hay không. Code tìm lớp khác gần nhất
  làm hard negative và dùng tín hiệu này để giảm ưu tiên cho câu dễ gây nhầm lớp.

Ba tín hiệu được kết hợp thành score dùng để xếp hạng. Trong các thí nghiệm A0–A4,
mức đóng góp lần lượt là `0.4` cho tên lớp, `0.3` cho đồng thuận nội lớp và `0.3` cho
phần phạt hard negative.

### Bước 4 — Chọn contributor

Các description được duyệt từ score cao xuống thấp:

- A0 giữ một câu đứng đầu;
- A1 và A3 giữ trực tiếp năm câu đứng đầu;
- A2 và A4 bật diversity: một câu bị bỏ qua nếu embedding của nó quá giống một câu đã
  được chọn, sau đó code tiếp tục tìm candidate tiếp theo cho đến khi đủ số lượng hoặc
  hết candidate phù hợp.

Diversity chỉ quyết định tập câu được giữ lại. Nó không trực tiếp tạo trọng số và không
thay đổi cách các embedding được gộp ở bước tiếp theo.

### Bước 5 — Gộp thành một prototype cho mỗi lớp

Với `mean`, mọi description được chọn đóng góp như nhau. Với `weighted_mean`, câu có
score cao hơn nhận trọng số lớn hơn; `weight_temperature=0.10` điều khiển mức chênh
lệch giữa các trọng số. A0 chỉ có một contributor nên `mean` và `weighted_mean` đều
cho cùng kết quả.

Sau khi gộp, vector được chuẩn hóa và trở thành một prototype 512 chiều đại diện cho
lớp. Ghép 14 vector theo đúng thứ tự class tạo ra prototype cache có shape `[14, 512]`.

### Bước 6 — Lưu cache và kiểm tra lại

Script lưu hai artifact:

- file `.pt` chứa tensor prototype, thứ tự class và cấu hình build;
- file audit JSON ghi description được chọn, score, hard-negative class, trọng số,
  các candidate bị diversity loại và norm của prototype.

Khi bắt đầu train, model kiểm tra thứ tự class và kích thước embedding rồi đăng ký cache
làm buffer cố định. A-branch tính độ tương đồng trực tiếp giữa visual feature và 14
prototype; Learnable Prompt, Visual Prompt và CLIP Text Encoder không còn nằm trong
forward của branch này. C-branch, temporal encoder và MIL hard Top-K gốc vẫn được giữ.

### Danh sách class và câu description đại diện

Thứ tự class phải giữ cố định giữa prototype cache, label vector và evaluator. Bảng
dưới liệt kê class text của VadCLIP gốc, class-name anchor dùng khi chấm score và câu
Top-1 được A0 chọn cho mỗi lớp.

| # | Class text | Class-name anchor | A0 Top-1 description |
|---:|---|---|---|
| 0 | `normal` | `normal` | `person uses phone` |
| 1 | `abuse` | `abuse` | `person pushes victim` |
| 2 | `arrest` | `arrest` | `police restrain suspect` |
| 3 | `arson` | `arson` | `person feeds material into fire` |
| 4 | `assault` | `assault` | `person hits person with object` |
| 5 | `burglary` | `burglary` | `person exits building carrying property` |
| 6 | `explosion` | `explosion` | `object explodes` |
| 7 | `fighting` | `fighting` | `people block and counterattack` |
| 8 | `roadAccidents` | `road accidents` | `vehicle swerves and crashes` |
| 9 | `robbery` | `robbery` | `person steals property using weapon` |
| 10 | `shooting` | `shooting` | `person shoots and flees` |
| 11 | `shoplifting` | `shoplifting` | `shopper hides merchandise in clothing` |
| 12 | `stealing` | `stealing` | `person grabs belongings and leaves` |
| 13 | `vandalism` | `vandalism` | `person breaks sign` |

A1–A4 dùng nhiều câu cho mỗi class. Danh sách contributor và trọng số đầy đủ nằm trong
các audit file [A1](../../a1_topk_mean.txt), [A2](../../a2_diverse_topk_mean.txt),
[A3](../../a3_weighted_topk.txt) và [A4](../../a4_diverse_weighted_topk.txt).

## 3. Các cấu hình ablation

| ID | K | Diversity | Aggregation | Mục tiêu |
|---|---:|---:|---|---|
| Original | – | – | Learnable + Visual Prompt | Baseline VadCLIP gốc |
| A0 | 1 | Không có tác dụng | Mean | Dùng description có score cao nhất |
| A1 | 5 | Tắt | Mean | Baseline multi-description |
| A2 | 5 | Ngưỡng `0.90` | Mean | Đo tác động riêng của diversity |
| A3 | 5 | Tắt | Weighted mean, temperature `0.10` | Đo tác động riêng của weighting |
| A4 | 5 | Metadata ghi `0.90`; audit cho thấy hành vi gần `0.95` | Weighted mean, temperature `0.10` | Kết hợp diversity và weighting |

Với A0, dù một lệnh build có vô tình bật diversity hoặc chọn `weighted_mean`, kết quả
vẫn là Top-1: candidate đứng đầu được chọn rồi vòng lặp dừng; softmax trên một phần tử
cho trọng số bằng 1.

### Kiểm tra cache và audit prototype

Năm audit file xác nhận:

- cùng CLIP model `ViT-B/16`;
- cùng description SHA-256
  `bec218ba72e2cea3930acda4a6f60a47a16c1bc8b2b464e23ef14bc677402bed`;
- cùng thứ tự 14 lớp;
- prototype shape `[14, 512]`;
- norm của mọi prototype xấp xỉ 1.

| Cấu hình | Số contributor mỗi lớp | Số rejection được ghi | Khoảng weight |
|---|---|---:|---:|
| A0 | 1 cho mọi lớp | 0 | `1.0` |
| A1 | 5 cho mọi lớp | 0 | `0.2` |
| A2 | 4 cho `abuse`, `arrest`; 5 cho lớp khác | 95 | `0.20–0.25` |
| A3 | 5 cho mọi lớp | 0 | `0.182702–0.227163` |
| A4 | 5 cho mọi lớp | 15 | `0.181299–0.227163` |

Có một bất nhất cần xử lý trước khi coi A2 và A4 là ablation factorial sạch:

- A2 ghi nhận rejection nhỏ nhất tại cosine `0.900272`, phù hợp threshold `0.90`;
- A4 ghi nhận rejection nhỏ nhất tại cosine `0.950722`, phù hợp threshold khoảng
  `0.95`, dù metadata vẫn ghi `0.90`;
- selection của A2 và A4 khác nhau, trong khi aggregation `mean/weighted_mean` được áp
  dụng sau selection và theo thiết kế không được làm thay đổi tập contributor.

Ví dụ ở lớp `abuse`, A2 chọn các index `1, 6, 19, 7` và không tìm đủ 5 contributor;
A4 lại chọn `1, 15, 16, 12, 5`. Vì vậy chênh lệch A2–A4 không thể chỉ quy cho weighted
mean. Cần kiểm tra lại Kaggle source/version hoặc build lại A4 bằng code hiện tại và
xác nhận audit có cùng selection với A2 trước khi so sánh aggregation.

## 4. Điều kiện huấn luyện chung

| Thành phần | Giá trị |
|---|---|
| Dataset | UCF-Crime |
| Training clips | 8.000 normal + 8.100 anomaly |
| Epoch | 10 |
| Batch size | 64 normal + 64 anomaly mỗi iteration |
| Optimizer | AdamW |
| Learning rate ban đầu | `2e-5` |
| AMP | Tắt |
| C/A pooling | Original hard Top-K, mean |
| AIS | Tắt |
| Temporal Segment Top-K | Tắt |
| `loss3` | Original: bật; A0–A4: tắt |
| Prototype logit temperature | A0–A4: `0.07` |
| Peak VRAM | A0–A4: khoảng `4.88–5.02 GB` |

Các run được thực hiện trong các Kaggle Version độc lập. Vì vậy việc dùng cùng tên
output trong từng version không làm các artifact ghi đè lẫn nhau.

Baseline gốc vẫn dùng Learnable Prompt, Visual Prompt và `loss3`; ở cuối training, log
baseline ghi `loss3 ≈ 0.0191`.

## 5. Metrics

- `AUC1`, `AP1`: frame-level anomaly detection từ C-branch (`logits1`).
- `AUC2`, `AP2`: frame-level anomaly detection từ xác suất không thuộc lớp `normal`
  của A-branch (`logits2`).
- `mAP@IoU`: class-wise temporal detection từ A-branch.
- `average mAP`: trung bình `mAP@0.1` đến `mAP@0.5`.

`AUC` và `AP` được biểu diễn trong khoảng `[0, 1]`; detection mAP được biểu diễn theo
phần trăm.

## 6. Kết quả tại cùng một mốc huấn luyện

Bảng dưới lấy cùng mốc epoch 4, sau dòng log `batch=120`, để hạn chế sai lệch do chọn
thời điểm tốt nhất riêng cho từng cấu hình.

| Cấu hình | AUC1 | AP1 | AUC2 | AP2 | Average mAP |
|---|---:|---:|---:|---:|---:|
| Original VadCLIP | **0.871007** | **0.306797** | **0.859129** | **0.268492** | 6.19% |
| A0 Top-1 | 0.865536 | 0.298404 | 0.854656 | 0.247945 | 7.51% |
| A1 Top-K Mean | 0.868865 | 0.301836 | 0.848042 | 0.257238 | 7.34% |
| A2 Diverse + Mean | 0.862504 | 0.275067 | 0.850484 | 0.250292 | **9.14%** |
| A3 Weighted Top-K | 0.869457 | 0.305718 | 0.848482 | 0.257331 | 7.43% |
| A4 Diverse + Weighted | 0.869321 | 0.298300 | 0.850173 | 0.261062 | 7.81% |

Chi tiết detection mAP tại cùng mốc:

| Cấu hình | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | Average |
|---|---:|---:|---:|---:|---:|---:|
| Original VadCLIP | 11.55% | 8.29% | 4.92% | 4.01% | 2.20% | 6.19% |
| A0 Top-1 | 12.91% | 9.51% | 6.59% | 4.92% | 3.61% | 7.51% |
| A1 Top-K Mean | 14.58% | 10.49% | 5.17% | 3.56% | 2.89% | 7.34% |
| A2 Diverse + Mean | **15.37%** | **12.39%** | **7.95%** | **5.59%** | **4.42%** | **9.14%** |
| A3 Weighted Top-K | 14.86% | 10.50% | 5.10% | 3.71% | 2.97% | 7.43% |
| A4 Diverse + Weighted | 14.40% | 10.99% | 6.04% | 4.84% | 2.79% | 7.81% |
