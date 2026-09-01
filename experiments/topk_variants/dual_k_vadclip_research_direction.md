# Dual-\(K\) Adaptive Instance Selection for VadCLIP

## 1. Mục tiêu nghiên cứu

Thiết kế một cơ chế **adaptive instance selection** cho VadCLIP, trong đó hai branch sử dụng hai mức selection khác nhau:

- **C-branch**: học mức hỗ trợ cần thiết để phát hiện *đâu là vùng bất thường*.
- **A-branch**: học mức hỗ trợ cần thiết để xác định *đâu là các snippet giàu bằng chứng ngữ nghĩa cho class bất thường*.

Thay vì dùng một giá trị Top-\(K\) cố định cho cả hai branch, ta cho phép:

\[
K_C \neq K_A
\]

và quan trọng hơn, không đặt trực tiếp \(K_C\), \(K_A\) bằng hyperparameter cố định mà để chúng xuất hiện từ hai cơ chế soft selection riêng biệt.

> **Scope của hướng này:** không thay đổi visual prompt của VadCLIP. Visual prompt và đường truyền giữa C-branch và A-branch giữ nguyên như baseline. Chỉ thay phần instance selection / MIL aggregation.

---

# 2. Vấn đề của Top-\(K\) cố định trong VadCLIP

VadCLIP sử dụng Top-\(K\) ở hai nhiệm vụ có bản chất khác nhau.

## 2.1. C-branch

C-branch tạo anomaly confidence:

\[
a_t \in [0,1], \qquad t=1,\ldots,T
\]

và dùng các snippet có anomaly score cao nhất để tạo supervision ở mức video.

Nhiệm vụ của branch này là:

\[
\boxed{\text{Where is the anomaly?}}
\]

Do đó số snippet hữu ích có xu hướng liên quan tới:

- độ dài của anomaly;
- mức độ rõ của abnormal pattern;
- mức phân tán của anomaly trong video;
- độ tự tin của coarse anomaly classifier.

---

## 2.2. A-branch

A-branch tạo similarity giữa visual snippet và anomaly class:

\[
m_{t,c}
=
\operatorname{sim}(x_t,p_c)
\]

trong đó \(p_c\) là text/class representation.

Nhiệm vụ là:

\[
\boxed{\text{Which snippets best characterize anomaly class }c\text{?}}
\]

Không phải tất cả snippet bất thường đều giàu thông tin class-specific.

Ví dụ một video Fighting có thể có:

- 8 snippet được C-branch xem là abnormal;
- nhưng chỉ 3–4 snippet thể hiện rõ hành vi Fighting.

Do đó giả định:

\[
K_C = K_A
\]

không có lý do bắt buộc phải đúng.

---

# 3. Giả thuyết nghiên cứu

## H1 — Branch heterogeneity

Hai branch của VadCLIP cần lượng evidence khác nhau:

\[
K_C^* \neq K_A^*
\]

với nhiều video.

---

## H2 — Fixed \(K\) làm mất khả năng thích nghi

Một \(K\) cố định không thể đồng thời phù hợp với:

- anomaly ngắn và anomaly dài;
- anomaly dễ phát hiện và anomaly mơ hồ;
- coarse abnormal evidence và class-specific semantic evidence.

---

## H3 — Soft selection giúp gradient dày hơn

Hard Top-\(K\) tạo mask:

\[
h_t \in \{0,1\}
\]

và:

\[
\sum_t h_t = K
\]

nên chỉ selected snippets nhận tín hiệu trực tiếp từ MIL aggregation.

Ta thay bằng:

\[
\alpha_t \in [0,1]
\]

để nhiều snippet quanh decision boundary vẫn nhận gradient.

---

## H4 — Hai mức selection nên được điều khiển bằng hai constraint khác nhau

C-branch và A-branch có uncertainty khác bản chất.

Vì vậy ta dùng:

\[
\lambda_C
\]

và:

\[
\lambda_A
\]

là hai dual variables khác nhau để điều khiển độ nghiêm ngặt của selection.

---

# 4. Baseline formulation

Giả sử video có \(T\) snippets.

## 4.1. C-branch baseline

Anomaly scores:

\[
\mathbf a
=
[a_1,\ldots,a_T]
\]

Hard Top-\(K\):

\[
I_C
=
\operatorname{TopK}(\mathbf a,K)
\]

Video-level score:

\[
S_C^{base}
=
\frac{1}{K}
\sum_{t\in I_C} a_t
\]

---

## 4.2. A-branch baseline

Similarity matrix:

\[
M
\in
\mathbb R^{T\times C}
\]

với:

\[
M_{t,c}
=
\operatorname{sim}(x_t,p_c)
\]

Đối với class \(c\):

\[
I_{A,c}
=
\operatorname{TopK}(M_{:,c},K)
\]

Class score:

\[
S_{A,c}^{base}
=
\frac{1}{K}
\sum_{t\in I_{A,c}}
M_{t,c}
\]

---

# 5. Dual-\(K\) formulation

Thay hard mask bằng hai soft selectors:

\[
\alpha_t^C \in [0,1]
\]

và:

\[
\alpha_{t,c}^A \in [0,1]
\]

Sau đó định nghĩa **effective support size**:

\[
\boxed{
K_C^{eff}
=
\sum_{t=1}^{T}\alpha_t^C
}
\]

và:

\[
\boxed{
K_{A,c}^{eff}
=
\sum_{t=1}^{T}\alpha_{t,c}^A
}
\]

Trong formulation đơn giản hơn, có thể dùng một:

\[
K_A^{eff}
\]

cho toàn bộ A-branch.

Trong formulation class-conditioned mạnh hơn:

\[
K_{A,c}^{eff}
\]

có thể khác nhau giữa các class.

---

# 6. Ý nghĩa của hai \(K\)

## C-branch

\[
K_C^{eff}
\]

đại diện cho:

> **anomaly support size**

tức số lượng snippet mà mô hình cho rằng đáng đóng góp vào coarse anomaly decision.

---

## A-branch

\[
K_{A,c}^{eff}
\]

đại diện cho:

> **semantic evidence support size**

tức lượng snippet cần thiết để nhận diện class bất thường \(c\).

Do hai quantities khác nhau về bản chất:

\[
K_C^{eff}
\]

không cần bằng:

\[
K_{A,c}^{eff}
\]

---

# 7. Branch-specific evidence

## 7.1. C-branch evidence

Ta sử dụng:

\[
e_t^C = a_t
\]

Trong đó \(a_t\) là anomaly confidence.

---

## 7.2. A-branch evidence

Với ground-truth video-level anomaly class \(y\):

\[
e_t^A = M_{t,y}
\]

Hoặc class-wise:

\[
e_{t,c}^A = M_{t,c}
\]

---

# 8. Branch-specific uncertainty

Đây là thành phần giúp biến formulation thành constrained selection thay vì chỉ học hai \(K\) bằng MLP.

---

## 8.1. C-branch uncertainty

Một lựa chọn đơn giản là binary entropy:

\[
u_t^C
=
-a_t\log(a_t+\epsilon)
-(1-a_t)\log(1-a_t+\epsilon)
\]

Ý nghĩa:

- \(a_t \approx 0\) hoặc \(1\): uncertainty thấp.
- \(a_t \approx 0.5\): uncertainty cao.

---

## 8.2. A-branch uncertainty

Có thể sử dụng class margin.

Giả sử:

\[
P(c|x_t)
=
\operatorname{softmax}(M_{t,:})_c
\]

Với ground-truth class \(y\):

\[
margin_t
=
P(y|x_t)
-
\max_{c\neq y}P(c|x_t)
\]

Sau đó:

\[
u_t^A
=
1-margin_t
\]

Nếu Fighting và Abuse có score gần nhau, uncertainty sẽ cao.

---

# 9. Primal constrained optimization

Với branch:

\[
b\in\{C,A\}
\]

ta muốn chọn nhiều useful instances nhưng không cho phép uncertainty trung bình vượt quá budget.

Định nghĩa utility:

\[
J_b(\alpha^b)
=
\sum_t \alpha_t^b e_t^b
+
\rho_b
\sum_t \alpha_t^b
+
\tau_b H(\alpha^b)
\]

trong đó:

- \(e_t^b\): evidence;
- \(\rho_b\): khuyến khích coverage;
- \(H(\alpha^b)\): entropy regularization;
- \(\tau_b\): mức smoothing.

Constraint:

\[
R_b(\alpha^b)
=
\frac{
\sum_t \alpha_t^b u_t^b
}{
\sum_t \alpha_t^b + \epsilon
}
\leq
\delta_b
\]

Trong đó:

\[
\delta_C
\]

và:

\[
\delta_A
\]

là mức uncertainty chấp nhận được cho từng branch.

---

# 10. Dual formulation

Lagrangian của branch \(b\):

\[
\mathcal L_b
=
-J_b(\alpha^b)
+
\lambda_b
\left(
R_b(\alpha^b)-\delta_b
\right)
\]

với:

\[
\lambda_b\geq0
\]

Ta có hai dual variables:

\[
\boxed{\lambda_C}
\]

và:

\[
\boxed{\lambda_A}
\]

---

# 11. Ý nghĩa của dual variables

Nếu uncertainty constraint bị vi phạm:

\[
R_b > \delta_b
\]

thì:

\[
\lambda_b \uparrow
\]

selector trở nên nghiêm ngặt hơn.

Kết quả:

\[
K_b^{eff}\downarrow
\]

Ngược lại, nếu selected snippets có uncertainty thấp:

\[
R_b < \delta_b
\]

thì:

\[
\lambda_b\downarrow
\]

selector có thể chọn rộng hơn:

\[
K_b^{eff}\uparrow
\]

Do đó:

\[
\boxed{
\lambda_b
\text{ controls selection strictness}
}
\]

và:

\[
K_b^{eff}
\]

xuất hiện như kết quả của optimization.

---

# 12. Soft selector

Một dạng thực tế có thể dùng:

\[
\alpha_t^b
=
\sigma
\left(
\frac{
e_t^b
-
\lambda_b u_t^b
-
\theta_b
}{
\tau_b
}
\right)
\]

Trong đó:

- \(\theta_b\): selection threshold;
- \(\tau_b\): temperature;
- \(\lambda_b u_t^b\): uncertainty penalty.

Ta có:

\[
\alpha_t^b\in(0,1)
\]

---

# 13. Effective dual \(K\)

Sau khi có soft weights:

\[
\boxed{
K_C^{eff}
=
\sum_t \alpha_t^C
}
\]

và:

\[
\boxed{
K_{A,c}^{eff}
=
\sum_t \alpha_{t,c}^A
}
\]

Không cần round trong training.

Ví dụ:

\[
K_C^{eff}=5.73
\]

và:

\[
K_A^{eff}=2.41
\]

hoàn toàn hợp lệ.

---

# 14. Weighted MIL aggregation

## 14.1. C-branch

Thay:

\[
S_C^{base}
=
\frac{1}{K}
\sum_{t\in TopK}a_t
\]

bằng:

\[
\boxed{
S_C
=
\frac{
\sum_t \alpha_t^C a_t
}{
\sum_t \alpha_t^C+\epsilon
}
}
\]

---

## 14.2. A-branch

Với class \(c\):

\[
\boxed{
S_{A,c}
=
\frac{
\sum_t \alpha_{t,c}^A M_{t,c}
}{
\sum_t \alpha_{t,c}^A+\epsilon
}
}
\]

---

# 15. Không áp đặt quan hệ \(K_A<K_C\)

Một giả định dễ mắc phải là:

\[
K_A \leq K_C
\]

vì semantic evidence có vẻ là subset của coarse anomaly evidence.

Không nên hard-code điều này.

Có thể tồn tại:

\[
K_A > K_C
\]

khi A-branch nhận ra semantic anomaly trong các snippet mà C-branch chưa đủ tự tin.

Do đó hai selector cần được học độc lập ở mức constraint.

---

# 16. Ví dụ — Fighting

Giả sử video có 10 snippets.

C-branch anomaly scores:

\[
A=
[0.08,0.12,0.72,0.86,0.93,0.88,0.69,0.25,0.10,0.06]
\]

A-branch similarity với Fighting:

\[
M_F=
[0.10,0.15,0.32,0.67,0.91,0.84,0.38,0.22,0.11,0.08]
\]

C-branch có thể sinh:

\[
\alpha^C=
[0.02,0.05,0.66,0.88,0.96,0.91,0.71,0.18,0.04,0.01]
\]

nên:

\[
K_C^{eff}
=
4.42
\]

A-branch có thể sinh:

\[
\alpha^A=
[0.01,0.02,0.08,0.37,0.91,0.82,0.12,0.04,0.01,0.01]
\]

nên:

\[
K_A^{eff}
=
2.39
\]

Interpretation:

- C-branch cần vùng abnormal rộng.
- A-branch chỉ cần một số snippet semantic mạnh nhất.

---

# 17. Train-time formulation

Trong training:

\[
\alpha^C,\alpha^A
\]

được giữ continuous.

Do đó:

\[
0<\alpha_t<1
\]

và nhiều snippet quanh boundary vẫn nhận gradient.

Pipeline:

```text
Video features
    │
    ├── C-branch
    │      │
    │      ├── anomaly evidence
    │      ├── uncertainty
    │      ├── dual variable λ_C
    │      └── soft selector α_C
    │
    └── A-branch
           │
           ├── class similarity
           ├── semantic uncertainty
           ├── dual variable λ_A
           └── soft selector α_A
```

Sau đó:

```text
α_C → weighted C-MIL
α_A → weighted A-MIL
```

---

# 18. Inference

Có hai lựa chọn cần ablation.

## Option 1 — Soft inference

Giữ:

\[
\alpha^C,\alpha^A
\]

và dùng weighted aggregation như training.

Ưu điểm:

- không có train-test selection mismatch;
- không cần rounding \(K\).

---

## Option 2 — Soft-to-Hard inference

Training:

\[
\alpha_t\in[0,1]
\]

Inference:

\[
K_C^{hard}
=
\operatorname{round}(K_C^{eff})
\]

\[
K_A^{hard}
=
\operatorname{round}(K_A^{eff})
\]

sau đó:

\[
HardTopK_C
\]

và:

\[
HardTopK_A
\]

Điều này cho phép giữ behavior gần MIL Top-\(K\) truyền thống.

---

# 19. Dual update

Một primal-dual update đơn giản:

\[
\lambda_b
\leftarrow
\left[
\lambda_b
+
\eta_\lambda
\left(
R_b-\delta_b
\right)
\right]_+
\]

Trong đó:

\[
[\cdot]_+
=
\max(0,\cdot)
\]

Nếu:

\[
R_b>\delta_b
\]

thì:

\[
\lambda_b
\]

tăng.

Nếu:

\[
R_b<\delta_b
\]

thì:

\[
\lambda_b
\]

giảm.

---

# 20. Video-conditioned dual variables

Phiên bản đơn giản:

\[
\lambda_C,\lambda_A
\]

là global learnable variables.

Phiên bản mạnh hơn:

\[
\lambda_{v,C}
\]

và:

\[
\lambda_{v,A}
\]

phụ thuộc video.

Có thể viết:

\[
\lambda_{v,C}
=
\operatorname{softplus}(g_C(X_v))
\]

\[
\lambda_{v,A}
=
\operatorname{softplus}(g_A(X_v))
\]

Tuy nhiên cần cẩn thận:

nếu chỉ cho MLP dự đoán \(\lambda\) mà không còn dual constraint update thì formulation trở thành learned gate thông thường.

Do đó phiên bản paper nên ưu tiên interpretation:

\[
\lambda
\]

là dual variable của constrained optimization.

---

# 21. Loss tổng

Một formulation tối thiểu:

\[
L
=
L_{VadCLIP}
+
\beta_C L_{constraint}^C
+
\beta_A L_{constraint}^A
\]

Trong đó:

\[
L_{constraint}^b
=
\lambda_b
\left(
R_b-\delta_b
\right)
\]

Nếu cần regularization chống collapse:

\[
L
=
L_{VadCLIP}
+
L_{dual}
+
\gamma L_{support}
\]

---

# 22. Collapse problem

Một nguy cơ là selector học:

\[
K^{eff}\rightarrow1
\]

vì chỉ lấy snippet mạnh nhất có thể dễ tối ưu classification loss hơn.

Do đó cần theo dõi:

\[
K_C^{eff}
\]

và:

\[
K_A^{eff}
\]

trong quá trình training.

Có thể thêm minimum support:

\[
K_b^{eff}\geq K_{min}
\]

hoặc penalty:

\[
L_{min}
=
\max(
0,
K_{min}-K_b^{eff}
)
\]

Tuy nhiên không nên regularize quá mạnh khiến hệ thống quay lại fixed-\(K\).

---

# 23. Temperature

Temperature:

\[
\tau_b
\]

điều khiển độ mềm của selector.

Nếu:

\[
\tau_b \gg 0
\]

weights tương đối dense.

Nếu:

\[
\tau_b\rightarrow0
\]

selector tiến gần hard threshold.

Có thể dùng annealing:

\[
\tau_{start}>\tau_{end}
\]

để:

- đầu training: gradient dense;
- cuối training: selection sắc hơn.

---

# 24. Hai temperature riêng

Do hai branch có đặc tính khác nhau:

\[
\tau_C
\]

không nhất thiết bằng:

\[
\tau_A
\]

A-branch có thể cần sharper semantic selection hơn C-branch.

Đây là một ablation đáng chạy.

---

# 25. Formulation tối thiểu để triển khai trước

Không nên bắt đầu ngay với full primal-dual system.

### Version 0 — Original VadCLIP

\[
K_C=K_A=K
\]

Hard Top-\(K\).

---

### Version 1 — Dual Fixed-\(K\)

Dùng:

\[
K_C\neq K_A
\]

nhưng cả hai vẫn fixed.

Mục tiêu:

> Kiểm tra giả thuyết hai branch có thực sự cần cardinality khác nhau không.

---

### Version 2 — Dual Soft-\(K\)

Giữ:

\[
K_C,K_A
\]

fixed nhưng thay hard selection bằng differentiable/soft selection.

Mục tiêu:

> Kiểm tra benefit đến từ smooth gradient hay không.

---

### Version 3 — Learnable Dual-\(K\)

Cho:

\[
K_C=f_C(X)
\]

\[
K_A=f_A(X)
\]

Mục tiêu:

> Baseline adaptive-\(K\) đơn giản.

---

### Version 4 — Dual-Constrained Selection

Full proposed method:

\[
\lambda_C,\lambda_A
\]

điều khiển:

\[
\alpha^C,\alpha^A
\]

và:

\[
K_C^{eff},K_A^{eff}
\]

xuất hiện từ soft support.

---

# 26. Ablation table đề xuất

| Variant | \(K_C\neq K_A\) | Soft selection | Adaptive support | Dual constraint |
|---|---:|---:|---:|---:|
| VadCLIP | No | No | No | No |
| Dual Fixed-K | Yes | No | No | No |
| Dual Soft-K | Yes | Yes | No | No |
| Learned Dual-K | Yes | Yes | Yes | No |
| Proposed Dual-Constrained | Yes | Yes | Yes | Yes |

---

# 27. Những metric cần theo dõi ngoài AUC/AP

Không chỉ đo performance cuối.

Nên log:

\[
K_C^{eff}
\]

và:

\[
K_A^{eff}
\]

theo:

- anomaly category;
- video duration;
- anomaly duration;
- training epoch.

Ngoài ra:

### 1. Mean effective support

\[
\mathbb E[K_C^{eff}]
\]

\[
\mathbb E[K_A^{eff}]
\]

### 2. Support gap

\[
\Delta K
=
K_C^{eff}-K_A^{eff}
\]

### 3. Selection entropy

\[
H(\alpha^C)
\]

\[
H(\alpha^A)
\]

### 4. Branch constraint violation

\[
R_C-\delta_C
\]

\[
R_A-\delta_A
\]

### 5. Correlation với anomaly duration

Nếu có temporal ground truth ở test:

\[
corr(
K_C^{eff},
\text{true anomaly length}
)
\]

Đây là bằng chứng rất mạnh cho adaptive support.

---

# 28. Research questions

### RQ1

Có thật sự cần:

\[
K_C\neq K_A
\]

không?

---

### RQ2

Soft selection có cải thiện performance so với hard Top-\(K\) khi giữ nguyên \(K\) không?

---

### RQ3

Adaptive support có tốt hơn fixed dual-\(K\) không?

---

### RQ4

Dual-constrained optimization có tốt hơn một MLP trực tiếp predict \(K\) không?

---

### RQ5

Hai dual variables có học behavior khác nhau không?

Ví dụ:

\[
\lambda_A>\lambda_C
\]

ở các class dễ semantic confusion.

---

### RQ6

\(K_C^{eff}\) có tương quan với temporal anomaly duration không?

---

### RQ7

\(K_A^{eff}\) có nhỏ hơn ở những class có semantic signature ngắn và rõ không?

---

# 29. Expected behavior

## Fighting

Có thể:

\[
K_C^{eff}>K_A^{eff}
\]

vì vùng abnormal dài nhưng chỉ một số snippet thể hiện hành vi Fighting rõ.

---

## Explosion

Có thể:

\[
K_C^{eff}\approx3
\]

và:

\[
K_A^{eff}\approx1
\]

do semantic event rất ngắn.

---

## Shoplifting

Có thể:

\[
K_A^{eff}>K_C^{eff}
\]

nếu semantic similarity phát hiện chuỗi hành động mà coarse classifier chỉ đánh score cao ở một số thời điểm.

---

# 30. Pseudocode

```python
# X: T x D video snippet features

# ----- C branch -----
a = C_branch(X)                       # T

u_c = binary_entropy(a)

alpha_c = sigmoid(
    (a - lambda_c * u_c - theta_c) / tau_c
)

k_c_eff = alpha_c.sum()

score_c = (
    alpha_c * a
).sum() / (k_c_eff + eps)


# ----- A branch -----
M = A_branch(X)                       # T x C

P = softmax(M, dim=-1)

margin = (
    P[:, y] -
    max_other_class(P, y)
)

u_a = 1.0 - margin

e_a = M[:, y]

alpha_a = sigmoid(
    (e_a - lambda_a * u_a - theta_a) / tau_a
)

k_a_eff = alpha_a.sum()

score_a = (
    alpha_a * e_a
).sum() / (k_a_eff + eps)


# ----- original VadCLIP losses -----
loss_vadclip = vadclip_loss(
    score_c,
    score_a,
    ...
)

# ----- dual constraints -----
R_c = (
    alpha_c * u_c
).sum() / (k_c_eff + eps)

R_a = (
    alpha_a * u_a
).sum() / (k_a_eff + eps)

loss = (
    loss_vadclip
    + lambda_c * (R_c - delta_c)
    + lambda_a * (R_a - delta_a)
)
```

---

# 31. Dual update pseudocode

```python
with torch.no_grad():
    lambda_c = torch.clamp(
        lambda_c
        + eta_lambda * (R_c - delta_c),
        min=0.0
    )

    lambda_a = torch.clamp(
        lambda_a
        + eta_lambda * (R_a - delta_a),
        min=0.0
    )
```

---

# 32. Điểm novelty tiềm năng

Không nên claim:

> We replace fixed Top-K with adaptive K.

Claim này quá yếu.

Cũng không nên claim:

> We use two different K values.

Chỉ có hai \(K\) không đủ mạnh.

Claim nên ở mức:

> We formulate instance selection in VadCLIP's heterogeneous coarse-classification and semantic-alignment branches as two branch-specific constrained optimization problems. Their dual variables independently regulate selection strictness, yielding adaptive continuous support sizes for anomaly evidence and class-discriminative evidence.

Điểm chính:

\[
\boxed{
\text{branch-specific constrained selection}
}
\]

không phải chỉ:

\[
\boxed{
\text{adaptive Top-K}
}
\]

---

# 33. Contribution dự kiến

### Contribution 1 — Dual branch-specific selection

Thay shared fixed Top-\(K\) bằng hai selection processes tương ứng với hai nhiệm vụ khác nhau của VadCLIP.

---

### Contribution 2 — Adaptive effective support

Không dự đoán hard \(K\) trực tiếp.

Định nghĩa:

\[
K_b^{eff}
=
\sum_t \alpha_t^b
\]

để selection size xuất hiện từ soft support.

---

### Contribution 3 — Dual-controlled uncertainty constraint

Sử dụng:

\[
\lambda_C
\]

và:

\[
\lambda_A
\]

để thích nghi mức selection strictness theo uncertainty của từng branch.

---

### Contribution 4 — Denser optimization signal

Soft selection giúp các instances quanh Top-\(K\) boundary vẫn nhận gradient thay vì bị loại hoàn toàn.

---

# 34. Những claim chưa nên đưa ra trước khi có thực nghiệm

Không nên mặc định:

\[
K_C>K_A
\]

cho mọi video.

Không nên mặc định:

\[
K_A<K_C
\]

cho mọi anomaly class.

Không nên claim dual formulation tốt hơn learned-\(K\) trước khi so với:

\[
K_C=f_C(X),
\qquad
K_A=f_A(X)
\]

Không nên claim soft selection luôn tốt hơn hard selection vì dense gradient cũng có thể đưa noise vào MIL.

---

# 35. Failure modes cần kiểm tra

## 35.1. Support collapse

\[
K^{eff}\rightarrow1
\]

---

## 35.2. Support explosion

\[
K^{eff}\rightarrow T
\]

khi selector quá mềm.

---

## 35.3. Branch homogenization

Hai selector học gần như giống nhau:

\[
\alpha^C\approx\alpha^A
\]

Nếu điều này xảy ra thì giả thuyết dual-\(K\) yếu đi.

---

## 35.4. Unstable dual variables

\[
\lambda
\]

dao động mạnh hoặc diverge.

Cần:

- smaller dual learning rate;
- clipping;
- moving-average constraint estimate.

---

## 35.5. Semantic uncertainty không đáng tin

Nếu A-branch calibration kém, class margin có thể không phản ánh true semantic uncertainty.

Cần ablation nhiều uncertainty definitions.

---

# 36. Uncertainty ablation cho A-branch

Có thể thử:

### Margin uncertainty

\[
u_t^A
=
1-
(P_y-P_{2nd})
\]

### Entropy

\[
u_t^A
=
-\sum_c
P(c|x_t)\log P(c|x_t)
\]

### Top-2 ratio

\[
u_t^A
=
\frac{P_{2nd}}
{P_{1st}+\epsilon}
\]

### Energy uncertainty

Dựa trên logits/similarity energy.

---

# 37. Uncertainty ablation cho C-branch

### Binary entropy

\[
H(a_t)
\]

### Distance to decision boundary

\[
u_t^C
=
1-|2a_t-1|
\]

### Temporal inconsistency

\[
u_t^C
=
|a_t-a_{t-1}|
+
|a_t-a_{t+1}|
\]

### Predictive variance

Nếu có stochastic inference hoặc ensemble.

---

# 38. Minimal experiment roadmap

## Phase 1 — Xác nhận dual-\(K\) có ý nghĩa

Grid-search:

\[
(K_C,K_A)
\]

Ví dụ:

\[
K_C\in\{2,4,6,8\}
\]

\[
K_A\in\{2,4,6,8\}
\]

Nếu optimum nằm ngoài diagonal:

\[
K_C\neq K_A
\]

thì đây là bằng chứng đầu tiên cho hypothesis.

---

## Phase 2 — Soft selection

Thay hard Top-\(K\) bằng soft selector nhưng giữ:

\[
K_C,K_A
\]

fixed.

---

## Phase 3 — Learned adaptive support

Cho phép:

\[
K_C^{eff}
\]

và:

\[
K_A^{eff}
\]

thay đổi theo sample.

---

## Phase 4 — Full dual formulation

Thêm:

\[
\lambda_C,\lambda_A
\]

và uncertainty constraints.

---

# 39. Experiment table nên có

| Model | C selection | A selection | Adaptive? | AUC | AP |
|---|---|---|---:|---:|---:|
| VadCLIP | Hard Top-K | Hard Top-K | No | - | - |
| Dual Fixed-K | Hard \(K_C\) | Hard \(K_A\) | No | - | - |
| Dual Soft-K | Soft \(K_C\) | Soft \(K_A\) | No | - | - |
| Learned Dual-K | Learned | Learned | Yes | - | - |
| Dual-Constrained | Dual-controlled | Dual-controlled | Yes | - | - |

---

# 40. Key visualization cho paper

Một figure rất quan trọng:

```text
Time →
────────────────────────────────────────────

Ground truth anomaly:
      █████████████

C-branch α:
    ▂▄██████████▆▃

A-branch α:
        ▂█████▃

────────────────────────────────────────────

K_C_eff = 10.4
K_A_eff = 5.7
```

Figure này trực tiếp cho thấy:

- C-branch học vùng anomaly rộng;
- A-branch tập trung vào semantic core;
- hai support sizes khác nhau.

---

# 41. Câu hỏi quan trọng nhất cần chứng minh

Paper này chỉ có ý nghĩa nếu thực nghiệm chứng minh:

\[
\boxed{
\text{C-branch và A-branch thực sự cần selection behavior khác nhau}
}
\]

Nếu cuối cùng:

\[
K_C^{eff}\approx K_A^{eff}
\]

ở hầu hết video và class, contribution dual formulation sẽ yếu.

Do đó trước khi đầu tư full implementation, **grid-search \(K_C,K_A\) trên baseline** là experiment rẻ nhưng cực kỳ quan trọng.

---

# 42. Formulation cuối được đề xuất

\[
\boxed{
\begin{aligned}
\alpha_t^C
&=
\sigma
\left(
\frac{
a_t-\lambda_C u_t^C-\theta_C
}{
\tau_C
}
\right)
\\[4pt]
K_C^{eff}
&=
\sum_t\alpha_t^C
\\[8pt]
\alpha_{t,c}^A
&=
\sigma
\left(
\frac{
M_{t,c}-\lambda_Au_{t,c}^A-\theta_A
}{
\tau_A
}
\right)
\\[4pt]
K_{A,c}^{eff}
&=
\sum_t\alpha_{t,c}^A
\end{aligned}
}
\]

với:

\[
\lambda_C
\]

và:

\[
\lambda_A
\]

được cập nhật theo branch-specific uncertainty constraints.

MIL scores:

\[
\boxed{
S_C
=
\frac{
\sum_t\alpha_t^C a_t
}{
K_C^{eff}+\epsilon
}
}
\]

\[
\boxed{
S_{A,c}
=
\frac{
\sum_t\alpha_{t,c}^A M_{t,c}
}{
K_{A,c}^{eff}+\epsilon
}
}
\]

---

# 43. Tên tạm thời

Một số tên có thể dùng trong quá trình nghiên cứu:

- **Dual-Adaptive Instance Selection (DAIS)**
- **Dual-Support MIL for VadCLIP**
- **Branch-Adaptive Soft Top-K**
- **Dual-Constrained Instance Selection**
- **Dual Adaptive Support Estimation (DASE)**

Tên chỉ nên chốt sau khi formulation và novelty được xác minh đầy đủ.

---

# 44. Tóm tắt một câu

> Thay vì buộc C-branch và A-branch của VadCLIP dùng cùng một fixed Top-\(K\), phương pháp học hai continuous instance supports khác nhau thông qua branch-specific uncertainty-constrained soft selection, từ đó tạo ra \(K_C^{eff}\) cho coarse anomaly evidence và \(K_A^{eff}\) cho class-specific semantic evidence.

