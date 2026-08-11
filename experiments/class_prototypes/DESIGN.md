# Thiết kế thí nghiệm: Class Prototype từ nhiều Description Embedding

## 1. Mục tiêu

Mỗi class không còn được biểu diễn bởi duy nhất class name hoặc một prompt. Thay vào đó,
mỗi class có một tập description, được mã hóa bằng CLIP Text Encoder, chấm điểm, lọc
trùng lặp và gộp thành một prototype ổn định.

Prototype cuối cùng có dạng `[num_classes, embed_dim]`, được L2-normalize và dùng trực
tiếp để tính cosine similarity với image/video embedding.

Đây sẽ là một nhánh thí nghiệm tách biệt với baseline. Không sửa hành vi mặc định của
`src/` để checkpoint và kết quả baseline hiện tại vẫn tái lập được.

## 2. Điểm cần lưu ý trong VadCLIP hiện tại

Trong `src/model.py`:

- CLIP backbone đã được freeze bằng `requires_grad = False`.
- Tuy nhiên, `text_prompt_embeddings` vẫn là tham số học được.
- `encode_textprompt()` trộn class text với learned context embedding.
- Trong `forward()`, text feature còn được điều chỉnh theo video qua `visual_attn` và
  `mlp1` trước khi tạo `logits2`.

Do yêu cầu prototype phải được tính trước, cache và dùng trực tiếp cho cosine similarity,
nhánh prototype đề xuất sẽ:

1. Dùng CLIP Text Encoder frozen để mã hóa description offline.
2. Không dùng `text_prompt_embeddings` khi tạo description embedding.
3. Không cộng `visual_attn` hoặc `mlp1` vào prototype ở cấu hình chính.
4. Giữ nguyên temporal encoder và C-branch của VadCLIP.
5. Tính A-branch bằng normalized visual feature nhân với normalized class prototype.

Việc giữ visual-conditioned text adaptation chỉ nên là một ablation riêng, không trộn
vào cấu hình chính vì khi đó prototype không còn được dùng "trực tiếp".

## 3. Dữ liệu description

Tạo một file JSON cho từng dataset, ví dụ:

```text
experiments/class_prototypes/descriptions/
├── ucf_crime_descriptions.json
└── xd_violence_descriptions.json
```

Schema đề xuất:

```json
{
  "dataset": "ucf-crime",
  "version": 1,
  "classes": {
    "normal": [
      "a surveillance video showing ordinary daily activity",
      "people moving normally without any dangerous event"
    ],
    "fighting": [
      "two or more people physically attacking each other",
      "a violent physical confrontation between people"
    ]
  }
}
```

Nguyên tắc sinh description:

- Số lượng ứng viên ban đầu nên đồng đều giữa các class, dự kiến 20–30 description/class.
- Mô tả nội dung có thể quan sát trong video, tránh kiến thức không xuất hiện trong ảnh.
- Tránh nhắc tên dataset, split, camera ID hoặc các shortcut đặc thù dữ liệu.
- Bao gồm nhiều cách diễn đạt, góc nhìn và mức độ chi tiết khác nhau.
- Giữ riêng file description gốc để có thể audit và tái tạo cache.
- Class `normal` phải được mô tả đủ đa dạng, không chỉ bằng phủ định như “not abnormal”.

## 4. Mã hóa và cache embedding

Với mỗi description `d_(c,i)` của class `c`:

```text
e_(c,i) = normalize(CLIP_text(d_(c,i)))
```

Class-name anchor được tính bằng:

```text
n_c = normalize(CLIP_text(class_name_prompt_c))
```

Mặc định dùng một template cố định, ví dụ `a video of {class_name}`, để mọi class được
đánh giá trong cùng một ngữ cảnh. Có thể đánh giá template ensemble ở ablation riêng.

Cache đề xuất:

```text
outputs/class_prototypes/<dataset>/<clip_model>/<description_hash>/
├── embeddings.pt
├── metadata.json
└── selection.json
```

`metadata.json` cần lưu tối thiểu:

- Dataset và thứ tự class.
- CLIP architecture và checkpoint identifier.
- Description file path và SHA-256.
- Tokenizer/context length.
- Dtype dùng khi encode và dtype lưu cache.
- Công thức score cùng toàn bộ hyperparameter.
- Thời điểm tạo cache và phiên bản code.

Cache chỉ hợp lệ khi description hash, class order, CLIP checkpoint và tokenizer khớp.
Embedding được tính trong `model.eval()` và `torch.no_grad()`.

## 5. Chấm điểm từng description embedding

Tất cả embedding phải được L2-normalize trước khi tính score.

### 5.1. Gần class name

```text
s_name(c,i) = cos(e_(c,i), n_c)
```

Thành phần này loại các description bị lệch khỏi ý nghĩa class chính.

### 5.2. Gần các embedding cùng class

Để tránh một embedding tự làm tăng score của chính nó, dùng leave-one-out centroid:

```text
mu_(c,-i) = normalize(mean({e_(c,j) | j != i}))
s_intra(c,i) = cos(e_(c,i), mu_(c,-i))
```

Nếu một class chỉ có một description thì đặt `s_intra = 0` và ghi cảnh báo.

### 5.3. Xa embedding của class khác

Tính centroid ban đầu cho từng class khác:

```text
mu_d = normalize(mean({e_(d,j)}))
s_inter(c,i) = max_{d != c} cos(e_(c,i), mu_d)
```

Dùng class khác gần nhất thay vì mean trên toàn bộ class khác, vì nhầm lẫn thường do
một hard-negative class cụ thể.

### 5.4. Score tổng hợp

```text
score(c,i) = alpha * s_name(c,i)
           + beta  * s_intra(c,i)
           - gamma * s_inter(c,i)
```

Default ban đầu:

```text
alpha = 0.4
beta  = 0.3
gamma = 0.3
```

Các trọng số này chỉ là điểm bắt đầu và phải được chọn trên validation set, không chọn
theo test set. `selection.json` sẽ lưu cả ba score thành phần và score tổng cho từng
description để có thể kiểm tra thủ công.

## 6. Top-K và diversity filtering

Không nên lấy đúng K trước rồi xóa các embedding gần nhau, vì một class có thể còn ít
hơn K prototype contributor. Quy trình đề xuất:

1. Xếp hạng toàn bộ description theo score giảm dần.
2. Duyệt theo thứ tự score.
3. Chấp nhận ứng viên đầu tiên.
4. Với ứng viên tiếp theo, chỉ giữ nếu cosine similarity với **mọi** embedding đã chọn
   nhỏ hơn ngưỡng `tau_diversity`.
5. Dừng khi đủ K hoặc đã duyệt hết danh sách.

Pseudo-code:

```python
selected = []
for candidate in ranked_candidates:
    if all(cos(candidate.embedding, x.embedding) < tau_diversity
           for x in selected):
        selected.append(candidate)
    if len(selected) == k:
        break
```

Default ban đầu:

```text
K = 5
tau_diversity = 0.90
```

Nếu không đủ K sau filtering, giữ số lượng thực tế và ghi rõ `selected_count`; không
hạ threshold âm thầm. Có thể thử threshold `0.85`, `0.90`, `0.95` trong ablation.

Greedy filtering được ưu tiên hơn clustering ở bản đầu vì đơn giản, deterministic và
dễ audit. Mỗi embedding bị loại phải lưu `duplicate_of` và cosine similarity tương ứng.

## 7. Tạo class prototype

### 7.1. Mean pooling

```text
p_c = normalize(mean({e_(c,i) | i thuộc selected_c}))
```

Đây là cấu hình đơn giản và là baseline chính của hướng multi-description.

### 7.2. Weighted mean

Chuyển score của các embedding đã chọn thành trọng số trong từng class:

```text
w_(c,i) = softmax(score(c,i) / tau_weight)
p_c = normalize(sum_i w_(c,i) * e_(c,i))
```

Default ban đầu: `tau_weight = 0.1`. Score chỉ được chuẩn hóa trong phạm vi cùng một
class. Cần log trọng số để phát hiện trường hợp một description chi phối gần như toàn bộ
prototype.

Không dùng raw score trực tiếp làm trọng số vì score có thể âm và tổng trọng số không ổn
định.

## 8. Tích hợp vào VadCLIP

Nên tạo toàn bộ code mới dưới:

```text
experiments/class_prototypes/
├── DESIGN.md
├── build_prototypes.py
├── prototype_model.py
├── train_ucf.py
├── test_ucf.py
├── options.py
├── descriptions/
└── tests/
```

Không sửa baseline trong `src/` ở bước đầu. `prototype_model.py` có thể kế thừa hoặc
wrap `CLIPVAD`, nhưng nhận prototype cache như input cố định.

Trong prototype mode:

```text
V = normalize(encode_video(video))       # [B, T, D]
P = normalize(class_prototypes)          # [C, D]
logits2 = V @ P^T / temperature          # [B, T, C]
```

Default giữ `temperature = 0.07` như nhánh similarity hiện tại. Class order trong `P`
phải khớp tuyệt đối với `prompt_text`, label vector và thứ tự mAP evaluator.

Prototype nên được load bằng `register_buffer` hoặc tensor không gradient:

```text
requires_grad = False
```

Không đưa prototype vào optimizer.

### 8.1. Ảnh hưởng tới loss hiện tại

`loss3` trong `ucf_train.py` hiện regularize khoảng cách giữa learned normal text feature
và các anomaly text feature. Với prototype tĩnh, loss này là hằng số và không truyền
gradient tới model. Vì vậy cấu hình prototype trực tiếp nên:

- Giữ `loss1` của C-branch.
- Giữ `loss2` của A-branch dùng prototype logits.
- Tắt `loss3` trong cấu hình chính.
- Chỉ thử một loss hình ảnh–prototype mới ở ablation riêng nếu cần.

Việc này phải được ghi rõ khi so sánh, vì đây không chỉ là thay prompt string mà còn thay
cơ chế text adaptation của baseline.

## 9. Kế hoạch thí nghiệm

Giữ nguyên dataset split, seed, temporal length, optimizer, learning rate, batch size và
evaluation của baseline. Chỉ thay cơ chế tạo text representation.

Thứ tự ablation đề xuất:

| ID | Text representation | Selection | Diversity | Aggregation |
|---|---|---|---|---|
| B0 | VadCLIP baseline | class name | không | learned prompt |
| P1 | Multi-description | không ranking | không | mean tất cả |
| P2 | Multi-description | Top-K theo score | không | mean |
| P3 | Multi-description | Top-K theo score | có | mean |
| P4 | Multi-description | Top-K theo score | có | weighted mean |

Sau đó mới ablate:

- `K ∈ {1, 3, 5, 8, 10}`.
- `tau_diversity ∈ {0.85, 0.90, 0.95}`.
- `tau_weight ∈ {0.05, 0.1, 0.2, 0.5}`.
- Bỏ từng thành phần `s_name`, `s_intra`, `s_inter`.
- Dùng mean-inter-class thay cho hardest negative.
- Static prototype trực tiếp so với prototype cộng visual-conditioned adaptation.

Metric chính giữ nguyên theo dataset: frame-level AUC/AP và detection mAP nếu evaluator
hiện tại hỗ trợ. Ngoài metric cuối, log thêm:

- Số description đầu vào và số được chọn cho từng class.
- Pairwise cosine trung bình/lớn nhất trước và sau diversity filtering.
- Khoảng cách prototype tới class-name anchor.
- Ma trận cosine giữa các class prototype.
- Description, score thành phần, rank và weight được chọn.

## 10. Kiểm thử bắt buộc trước khi train đầy đủ

1. Cùng input và cùng cache phải sinh prototype giống hệt nhau.
2. Mọi description embedding và prototype cuối có norm xấp xỉ 1.
3. Không có gradient trong CLIP Text Encoder hoặc prototype tensor.
4. Diversity filtering không giữ cặp có cosine lớn hơn hoặc bằng threshold.
5. Weighted mean có trọng số dương và tổng bằng 1 trong từng class.
6. Thứ tự class của cache khớp label map ở train và test.
7. `logits2` có shape `[batch, temporal_length, num_classes]`.
8. Cache sai description hash hoặc CLIP checkpoint phải bị từ chối.
9. Chạy smoke test nhỏ trước, sau đó mới chạy full training.

## 11. Cấu hình đầu tiên đề xuất

```yaml
candidate_descriptions_per_class: 20
class_name_template: "a video of {}"
score_weights:
  name: 0.4
  intra: 0.3
  inter: 0.3
top_k: 5
diversity_threshold: 0.90
aggregation: weighted_mean
weight_temperature: 0.10
logit_temperature: 0.07
freeze_clip_text_encoder: true
use_learned_prompt: false
use_visual_conditioned_text: false
use_text_separation_loss: false
```

Cấu hình này bám sát yêu cầu: nhiều description, CLIP Text Encoder frozen, embedding có
thể cache, chọn theo semantic quality, loại description trùng, weighted prototype được
normalize và dùng trực tiếp cho cosine similarity.

## 12. Tiêu chí hoàn thành triển khai

- Baseline trong `src/` không bị thay đổi.
- Có script build/cache prototype độc lập với training.
- Có file audit cho toàn bộ ranking, filtering và weighting.
- Có unit test cho scoring, diversity, aggregation và class order.
- Có smoke test chạy được trên UCF-Crime với cache nhỏ.
- Có lệnh train/test riêng và bảng kết quả so sánh B0–P4.
