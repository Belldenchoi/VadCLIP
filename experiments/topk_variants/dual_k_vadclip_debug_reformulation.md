# Debugging and Reformulating Dual-\(K\) Selection for VadCLIP

## 1. Mục tiêu của tài liệu

Tài liệu này phân tích failure mode quan sát được khi triển khai **Dual-\(K\) Adaptive Instance Selection** cho VadCLIP và đề xuất một formulation sửa đổi.

Phạm vi:

- Không thay đổi visual prompt.
- Không thay đổi backbone VadCLIP.
- Chỉ tập trung vào:
  - C-branch selection;
  - A-branch selection;
  - dual variables;
  - uncertainty constraints;
  - effective support size;
  - weighted MIL aggregation.

---

# 2. Hiện tượng quan sát được từ log

Run hiện tại cho thấy:

\[
K_C^{eff} \gg K_A^{eff}
\]

với C-branch thường chọn khoảng:

\[
K_C^{eff}\approx 45\sim60
\]

trong khi A-branch thường:

\[
K_A^{eff}\approx20\sim23.
\]

Điểm quan trọng hơn là:

\[
\boxed{
\lambda_C\rightarrow0,
\qquad
\lambda_A\rightarrow0
}
\]

rất sớm trong training.

Ví dụ cuối epoch 1:

\[
K_C^{eff}\approx46.23,
\qquad
K_A^{eff}\approx22.86
\]

nhưng:

\[
\lambda_C=0,
\qquad
\lambda_A=0.
\]

Đến epoch 10, behavior này vẫn giữ nguyên:

\[
K_C^{eff}\approx50,
\qquad
K_A^{eff}\approx20\sim22
\]

và:

\[
\lambda_C=\lambda_A=0.
\]

Do đó selector hiện tại thực tế không còn được điều khiển bởi dual variables.

---

# 3. Formulation hiện tại

Giả sử branch:

\[
b\in\{C,A\}.
\]

Ta có evidence:

\[
e_t^b
\]

và uncertainty:

\[
u_t^b.
\]

Selector hiện tại có dạng gần:

\[
\boxed{
\alpha_t^b
=
\sigma
\left(
\frac{
e_t^b-\lambda_bu_t^b-\theta_b
}{
\tau_b
}
\right)
}
\]

Trong đó:

- \(\alpha_t^b\in(0,1)\): soft instance weight;
- \(\lambda_b\): dual variable;
- \(\theta_b\): selection threshold;
- \(\tau_b\): temperature.

Effective support:

\[
\boxed{
K_b^{eff}
=
\sum_{t=1}^{T}\alpha_t^b
}
\]

Weighted MIL:

\[
\boxed{
S_b
=
\frac{
\sum_t\alpha_t^b e_t^b
}{
K_b^{eff}+\epsilon
}
}
\]

Dual update:

\[
\boxed{
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
}
\]

với:

\[
[x]_+=\max(0,x).
\]

Risk:

\[
\boxed{
R_b
=
\frac{
\sum_t\alpha_t^bu_t^b
}{
\sum_t\alpha_t^b+\epsilon
}
}
\]

Constraint:

\[
R_b\leq\delta_b.
\]

---

# 4. Failure mode 1 — Dual variables collapse về 0

Config hiện tại dùng:

\[
\delta_C=0.35
\]

và:

\[
\delta_A=0.8.
\]

Trong log, C-branch thường có:

\[
R_C\approx0.17\sim0.25
\]

nên:

\[
R_C-\delta_C<0.
\]

Do đó:

\[
\lambda_C
\leftarrow
[
\lambda_C+\eta(R_C-\delta_C)
]_+
\]

liên tục giảm và cuối cùng:

\[
\boxed{
\lambda_C=0.
}
\]

Tương tự A-branch:

\[
R_A\approx0.75\sim0.79
\]

trong khi:

\[
\delta_A=0.8.
\]

Phần lớn thời gian:

\[
R_A-\delta_A<0
\]

nên:

\[
\boxed{
\lambda_A=0.
}
\]

---

# 5. Hệ quả của \(\lambda=0\)

Nếu:

\[
\lambda_b=0
\]

thì selector:

\[
\alpha_t^b
=
\sigma
\left(
\frac{
e_t^b-\lambda_bu_t^b-\theta_b
}{
\tau_b
}
\right)
\]

trở thành:

\[
\boxed{
\alpha_t^b
\approx
\sigma
\left(
\frac{
e_t^b-\theta_b
}{
\tau_b
}
\right)
}
\]

Do đó pipeline thực tế không còn là:

\[
\text{uncertainty}
\rightarrow
\lambda_b
\rightarrow
\alpha_b
\rightarrow
K_b^{eff}
\]

mà gần như chỉ còn:

\[
\boxed{
e_b
+
\theta_b
+
\tau_b
\rightarrow
K_b^{eff}
}
\]

Vì vậy \(K_C\) và \(K_A\) hiện tại không thực sự được tạo bởi dual-constrained optimization.

---

# 6. Failure mode 2 — Evidence scale mismatch

Hiện tại threshold được đặt:

\[
\theta_C=0.75
\]

và:

\[
\theta_A=0.45.
\]

Dù:

\[
\theta_C>\theta_A
\]

nhưng vẫn quan sát:

\[
K_C^{eff}\gg K_A^{eff}.
\]

Điều này cho thấy distribution của hai evidence khác scale mạnh.

Có thể:

\[
e_t^C
\]

nằm chủ yếu trong vùng:

\[
[0.7,1.0]
\]

trong khi:

\[
e_t^A
\]

có thể tập trung quanh:

\[
[0.2,0.6].
\]

Khi đó cùng một sigmoid gate không thể được so sánh trực tiếp giữa hai branch.

---

# 7. Sửa đổi 1 — Per-video evidence normalization

Thay vì dùng raw evidence:

\[
e_t^b
\]

ta chuẩn hóa trong từng video:

\[
\boxed{
\tilde e_t^b
=
\frac{
e_t^b-\mu_e^b
}{
\sigma_e^b+\epsilon
}
}
\]

trong đó:

\[
\mu_e^b
=
\frac1T
\sum_{t=1}^{T}e_t^b
\]

và:

\[
\sigma_e^b
=
\sqrt{
\frac1T
\sum_t
(e_t^b-\mu_e^b)^2
}.
\]

Selector chuyển thành:

\[
\alpha_t^b
=
\sigma
\left(
\frac{
\tilde e_t^b-\lambda_bu_t^b-\theta_b
}{
\tau_b
}
\right).
\]

Ý nghĩa:

> Selector không còn hỏi score có vượt một absolute threshold hay không, mà hỏi score này mạnh đến đâu so với các snippet khác trong chính video đó.

---

# 8. C-branch evidence

C-branch tạo anomaly confidence:

\[
a_t.
\]

Ta dùng:

\[
e_t^C=a_t
\]

sau đó normalize:

\[
\boxed{
\tilde e_t^C
=
\frac{
a_t-\mu_a
}{
\sigma_a+\epsilon
}
}
\]

với:

\[
\mu_a
=
\frac1T\sum_ta_t.
\]

---

# 9. A-branch evidence

Với class \(y\):

\[
e_t^A=M_{t,y}
\]

trong đó:

\[
M_{t,y}
=
\operatorname{sim}(x_t,p_y).
\]

Normalize:

\[
\boxed{
\tilde e_t^A
=
\frac{
M_{t,y}-\mu_M
}{
\sigma_M+\epsilon
}
}
\]

với:

\[
\mu_M
=
\frac1T
\sum_tM_{t,y}.
\]

---

# 10. Failure mode 3 — Uncertainty scale mismatch

Hiện tại C và A sử dụng hai uncertainty khác bản chất.

Ví dụ C:

\[
u_t^C
=
-a_t\log(a_t)
-(1-a_t)\log(1-a_t)
\]

có maximum:

\[
u_{max}^C=\log2\approx0.693.
\]

Trong khi A có thể dùng:

\[
u_t^A
=
1-
(P_y-P_{2nd})
\]

có range gần:

\[
[0,1].
\]

Do đó:

\[
R_C\approx0.2
\]

và:

\[
R_A\approx0.78
\]

không thể được so sánh trực tiếp.

---

# 11. Sửa đổi 2 — Normalize uncertainty

Ta đưa uncertainty của mỗi branch về cùng miền:

\[
[0,1].
\]

Một cách:

\[
\boxed{
\tilde u_t^b
=
\frac{
u_t^b-u_{min}^b
}{
u_{max}^b-u_{min}^b+\epsilon
}
}
\]

Nếu biết theoretical range thì dùng trực tiếp.

Ví dụ C-branch:

\[
\boxed{
\tilde u_t^C
=
\frac{
H(a_t)
}{
\log2
}
}
\]

nên:

\[
\tilde u_t^C\in[0,1].
\]

A-branch nếu margin uncertainty vốn đã ở \([0,1]\), có thể giữ:

\[
\tilde u_t^A=u_t^A.
\]

---

# 12. C-branch normalized uncertainty

Binary entropy:

\[
H(a_t)
=
-a_t\log(a_t+\epsilon)
-(1-a_t)\log(1-a_t+\epsilon)
\]

Normalize:

\[
\boxed{
\tilde u_t^C
=
\frac{
H(a_t)
}{
\log2
}
}
\]

Interpretation:

\[
a_t\approx0
\quad\text{or}\quad
a_t\approx1
\Rightarrow
\tilde u_t^C\approx0
\]

và:

\[
a_t\approx0.5
\Rightarrow
\tilde u_t^C\approx1.
\]

---

# 13. A-branch normalized uncertainty

Class probability:

\[
P(c|x_t)
=
\operatorname{softmax}(M_{t,:})_c.
\]

Ground-truth class:

\[
y.
\]

Top competitor:

\[
P_{2nd}
=
\max_{c\neq y}P(c|x_t).
\]

Margin:

\[
m_t
=
P_y-P_{2nd}.
\]

Uncertainty:

\[
\boxed{
\tilde u_t^A
=
1-m_t
}
\]

với giá trị được clamp vào:

\[
[0,1].
\]

Một ablation đơn giản hơn:

\[
\boxed{
\tilde u_t^A
=
1-P_y.
}
\]

Nên thử cả hai.

---

# 14. Revised selector

Sau normalization:

\[
\boxed{
z_t^C
=
\tilde e_t^C
-
\lambda_C\tilde u_t^C
-
\theta_C
}
\]

và:

\[
\boxed{
\alpha_t^C
=
\sigma
\left(
\frac{z_t^C}{\tau_C}
\right)
}
\]

A-branch:

\[
\boxed{
z_t^A
=
\tilde e_t^A
-
\lambda_A\tilde u_t^A
-
\theta_A
}
\]

\[
\boxed{
\alpha_t^A
=
\sigma
\left(
\frac{z_t^A}{\tau_A}
\right)
}
\]

---

# 15. Effective support sau sửa đổi

\[
\boxed{
K_C^{eff}
=
\sum_t\alpha_t^C
}
\]

\[
\boxed{
K_A^{eff}
=
\sum_t\alpha_t^A
}
\]

hoặc class-conditioned:

\[
\boxed{
K_{A,c}^{eff}
=
\sum_t\alpha_{t,c}^A.
}
\]

---

# 16. Revised risk

Dùng normalized uncertainty:

\[
\boxed{
R_C
=
\frac{
\sum_t\alpha_t^C\tilde u_t^C
}{
K_C^{eff}+\epsilon
}
}
\]

và:

\[
\boxed{
R_A
=
\frac{
\sum_t\alpha_t^A\tilde u_t^A
}{
K_A^{eff}+\epsilon
}
}
\]

Bây giờ:

\[
R_C,R_A\in[0,1]
\]

nên budget có interpretation nhất quán hơn.

---

# 17. Sửa đổi 3 — Chọn active uncertainty budget

Không nên đặt:

\[
\delta_C,\delta_A
\]

một cách tùy ý.

Sau warm-up / một epoch diagnostic, đo empirical distributions:

\[
R_C^{(1)},\ldots,R_C^{(N)}
\]

và:

\[
R_A^{(1)},\ldots,R_A^{(N)}.
\]

Chọn:

\[
\boxed{
\delta_C
=
Q_q(R_C)
}
\]

và:

\[
\boxed{
\delta_A
=
Q_q(R_A)
}
\]

với:

\[
q\approx0.4\sim0.6.
\]

Ví dụ median:

\[
q=0.5.
\]

Mục tiêu là làm constraint **active một phần đáng kể thời gian**, thay vì luôn inactive.

---

# 18. Dual update sau sửa đổi

\[
\boxed{
\lambda_C
\leftarrow
\left[
\lambda_C
+
\eta_\lambda
(R_C-\delta_C)
\right]_+
}
\]

\[
\boxed{
\lambda_A
\leftarrow
\left[
\lambda_A
+
\eta_\lambda
(R_A-\delta_A)
\right]_+
}
\]

Expected behavior:

Nếu:

\[
R_C>\delta_C
\]

thì:

\[
\lambda_C\uparrow
\]

và C-selector trở nên strict hơn.

Nếu:

\[
R_C<\delta_C
\]

thì:

\[
\lambda_C\downarrow.
\]

Tương tự với A.

---

# 19. Failure mode 4 — Average-risk constraint không kiểm soát support size

Constraint hiện tại:

\[
R_b
=
\frac{
\sum_t\alpha_tu_t
}{
\sum_t\alpha_t
}
\leq
\delta_b
\]

chỉ kiểm soát **average uncertainty**.

Nó không trực tiếp ngăn:

\[
K_b^{eff}
\rightarrow T.
\]

Ví dụ nếu có rất nhiều snippet uncertainty thấp thì model có thể chọn tất cả:

\[
\alpha_t\approx1
\]

nhưng:

\[
R_b
\]

vẫn nhỏ.

Do đó:

\[
\boxed{
R_b\leq\delta_b
}
\]

không đảm bảo:

\[
K_b^{eff}
\]

hợp lý.

Đây có thể là nguyên nhân trực tiếp của:

\[
K_C^{eff}\approx50\sim60.
\]

---

# 20. Sửa đổi 4A — Support range regularization

Trong giai đoạn debug, cách đơn giản nhất là regularize support ratio.

Định nghĩa:

\[
r_b
=
\frac{
K_b^{eff}
}{
T
}.
\]

Ta muốn:

\[
r_{min}^b
\leq
r_b
\leq
r_{max}^b.
\]

Loss:

\[
\boxed{
L_{range}^b
=
\left[
\max(0,r_{min}^b-r_b)
\right]^2
+
\left[
\max(0,r_b-r_{max}^b)
\right]^2
}
\]

Tổng:

\[
\boxed{
L_{range}
=
L_{range}^C
+
L_{range}^A.
}
\]

Ví dụ diagnostic ban đầu:

\[
r_C\in[0.05,0.30]
\]

và:

\[
r_A\in[0.03,0.25].
\]

Các range này chỉ là guardrail để debug, không nên coi là final theoretical choice.

---

# 21. Sửa đổi 4B — Support-budget constraint

Một formulation sạch hơn cho paper là thêm constraint thứ hai:

\[
\boxed{
\frac{
K_b^{eff}
}{
T
}
\leq
\rho_b
}
\]

ngoài uncertainty constraint:

\[
R_b\leq\delta_b.
\]

Primal problem:

\[
\boxed{
\max_{\alpha^b}
J_b(\alpha^b)
}
\]

subject to:

\[
\boxed{
R_b(\alpha^b)\leq\delta_b
}
\]

và:

\[
\boxed{
\frac{
K_b^{eff}
}{
T
}
\leq\rho_b.
}
\]

---

# 22. Two-constraint Lagrangian

Ta có hai dual variables trên mỗi branch:

\[
\lambda_b^u
\]

cho uncertainty constraint,

và:

\[
\lambda_b^k
\]

cho support-size constraint.

Lagrangian:

\[
\boxed{
\mathcal L_b
=
-J_b(\alpha^b)
+
\lambda_b^u
\left(
R_b-\delta_b
\right)
+
\lambda_b^k
\left(
\frac{
K_b^{eff}
}{
T
}
-
\rho_b
\right)
}
\]

với:

\[
\lambda_b^u\geq0
\]

và:

\[
\lambda_b^k\geq0.
\]

---

# 23. Interpretation của hai dual variables

\[
\lambda_b^u
\]

trả lời:

> selected snippets có quá uncertain không?

Trong khi:

\[
\lambda_b^k
\]

trả lời:

> selector có đang chọn quá nhiều snippet không?

Do đó:

\[
\lambda_b^u\uparrow
\Rightarrow
\text{uncertain snippets bị phạt mạnh hơn}
\]

và:

\[
\lambda_b^k\uparrow
\Rightarrow
\text{toàn bộ support bị thu hẹp}.
\]

---

# 24. Dual updates với hai constraints

Uncertainty dual:

\[
\boxed{
\lambda_b^u
\leftarrow
\left[
\lambda_b^u
+
\eta_u
(R_b-\delta_b)
\right]_+
}
\]

Support dual:

\[
\boxed{
\lambda_b^k
\leftarrow
\left[
\lambda_b^k
+
\eta_k
\left(
\frac{K_b^{eff}}{T}-\rho_b
\right)
\right]_+
}
\]

---

# 25. Revised selector với support penalty

Một simple gate có thể viết:

\[
\boxed{
\alpha_t^b
=
\sigma
\left(
\frac{
\tilde e_t^b
-
\lambda_b^u\tilde u_t^b
-
\lambda_b^k
-
\theta_b
}{
\tau_b
}
\right)
}
\]

Trong đó support dual:

\[
\lambda_b^k
\]

hoạt động giống một adaptive global selection cost.

Nếu support quá lớn:

\[
K_b^{eff}/T>\rho_b
\]

thì:

\[
\lambda_b^k\uparrow
\]

làm toàn bộ gate stricter.

---

# 26. Tại sao formulation này phù hợp với C-branch explosion?

Hiện tại C-branch có:

\[
R_C\ll\delta_C
\]

nên uncertainty dual biến mất.

Nhưng:

\[
K_C^{eff}/T
\]

có thể rất lớn.

Với support constraint:

\[
\frac{
K_C^{eff}
}{
T
}
\leq\rho_C
\]

thì dù uncertainty thấp:

\[
R_C<\delta_C
\]

nhưng nếu:

\[
K_C^{eff}/T>\rho_C
\]

ta vẫn có:

\[
\lambda_C^k>0.
\]

Do đó selector không thể mở rộng support vô hạn.

---

# 27. Weighted MIL sau sửa đổi

C-branch:

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

A-branch:

\[
\boxed{
S_A
=
\frac{
\sum_t\alpha_t^A M_{t,y}
}{
K_A^{eff}+\epsilon
}
}
\]

Class-wise:

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

# 28. Total loss — debugging version

Trong phase đầu, dùng range regularization đơn giản:

\[
\boxed{
L
=
L_{VadCLIP}
+
\beta L_{dual}
+
\gamma L_{range}
}
\]

với:

\[
L_{dual}
=
\lambda_C(R_C-\delta_C)
+
\lambda_A(R_A-\delta_A).
\]

---

# 29. Total objective — full constrained version

Nếu dùng two-constraint dual formulation:

\[
\boxed{
\mathcal L
=
L_{VadCLIP}
+
\sum_{b\in\{C,A\}}
\lambda_b^u
(R_b-\delta_b)
+
\sum_{b\in\{C,A\}}
\lambda_b^k
\left(
\frac{
K_b^{eff}
}{
T
}
-\rho_b
\right)
}
\]

---

# 30. Không nên ép \(K_C\approx K_A\)

Mục tiêu không phải:

\[
K_C^{eff}
\approx
K_A^{eff}.
\]

Hai branch có nhiệm vụ khác nhau.

Hoàn toàn có thể:

\[
K_C^{eff}>K_A^{eff}.
\]

Điều cần tránh là:

\[
K_C^{eff}\rightarrow T
\]

hoặc:

\[
K_A^{eff}\rightarrow0.
\]

Do đó ta regularize từng support riêng thay vì dùng:

\[
|K_C-K_A|.
\]

---

# 31. Phân biệt imbalance hợp lý và collapse

Một behavior có thể hợp lý:

\[
\frac{K_C^{eff}}T=0.25
\]

và:

\[
\frac{K_A^{eff}}T=0.08.
\]

Đây có thể phản ánh:

- coarse anomaly region rộng;
- semantic discriminative core hẹp.

Nhưng:

\[
\frac{K_C^{eff}}T\rightarrow0.8
\]

hoặc:

\[
\frac{K_A^{eff}}T\rightarrow0
\]

thì nhiều khả năng là collapse.

---

# 32. Temperature không phải sửa đổi đầu tiên

Selector:

\[
\alpha_t^b
=
\sigma
\left(
\frac{z_t^b}{\tau_b}
\right)
\]

chịu ảnh hưởng lớn bởi:

\[
\tau_b.
\]

Nhưng không nên chỉnh \(\tau\) trước khi sửa scale.

Thứ tự đúng:

1. normalize evidence;
2. normalize uncertainty;
3. làm constraint active;
4. kiểm tra support;
5. cuối cùng mới tune \(\tau_C,\tau_A\).

---

# 33. Sau khi normalize mới tune temperature

Nếu C vẫn quá dense:

\[
K_C^{eff}\uparrow
\]

có thể giảm:

\[
\tau_C.
\]

Nếu A quá sparse:

\[
K_A^{eff}\downarrow
\]

có thể tăng:

\[
\tau_A.
\]

Tuy nhiên đây chỉ nên là fine-tuning.

---

# 34. Diagnostic logging bắt buộc

Mỗi epoch nên log:

## Evidence statistics

\[
\mu(e_C),
\quad
\sigma(e_C),
\quad
\min(e_C),
\quad
\max(e_C)
\]

\[
\mu(e_A),
\quad
\sigma(e_A),
\quad
\min(e_A),
\quad
\max(e_A)
\]

---

## Uncertainty statistics

\[
\mu(u_C),
\quad
\sigma(u_C)
\]

\[
\mu(u_A),
\quad
\sigma(u_A)
\]

---

## Dual statistics

\[
\lambda_C,
\qquad
\lambda_A
\]

hoặc full version:

\[
\lambda_C^u,
\lambda_C^k,
\lambda_A^u,
\lambda_A^k.
\]

---

## Support statistics

\[
K_C^{eff}
\]

\[
K_A^{eff}
\]

và quan trọng hơn:

\[
\boxed{
r_C
=
\frac{K_C^{eff}}T
}
\]

\[
\boxed{
r_A
=
\frac{K_A^{eff}}T.
}
\]

---

# 35. Constraint diagnostics

Log:

\[
R_C-\delta_C
\]

và:

\[
R_A-\delta_A.
\]

Nếu chúng luôn âm:

\[
R_b-\delta_b<0
\]

thì constraint quá lỏng.

Nếu luôn dương:

\[
R_b-\delta_b>0
\]

thì constraint quá chặt.

Desired behavior:

\[
R_b-\delta_b
\]

dao động quanh:

\[
0.
\]

---

# 36. Dual health criterion

Một dual variable khỏe không nên:

\[
\lambda_b=0
\]

trong gần như toàn bộ training.

Cũng không nên:

\[
\lambda_b\rightarrow\infty.
\]

Expected:

\[
\lambda_b
\]

tăng/giảm tùy batch hoặc epoch và đạt một equilibrium khác 0 khi constraint active.

---

# 37. Experiment roadmap

## Experiment A — Diagnose scale only

Không thay objective.

Chỉ thêm logging:

\[
e_C,e_A,u_C,u_A.
\]

Chạy:

\[
1\sim2
\]

epochs.

Mục tiêu:

> xác nhận scale mismatch.

---

## Experiment B — Normalize evidence

Thay:

\[
e_b
\rightarrow
\tilde e_b.
\]

Giữ uncertainty và dual update như cũ.

Theo dõi:

\[
K_C^{eff},
K_A^{eff}.
\]

Mục tiêu:

> xem support gap có chủ yếu đến từ evidence scale không.

---

## Experiment C — Normalize uncertainty

Thay:

\[
u_b
\rightarrow
\tilde u_b.
\]

Sau đó sử dụng budgets cùng interpretation trong:

\[
[0,1].
\]

Mục tiêu:

> làm \(R_C,R_A\) có scale tương thích.

---

## Experiment D — Active dual budgets

Estimate empirical risk distribution.

Chọn:

\[
\delta_C
\]

và:

\[
\delta_A
\]

gần median / middle quantiles.

Mục tiêu:

\[
\lambda_C,\lambda_A
\]

phải thực sự hoạt động.

---

## Experiment E — Support guardrail

Nếu C vẫn có:

\[
K_C^{eff}/T
\]

quá cao, thêm:

\[
L_{range}.
\]

Mục tiêu:

> kiểm chứng performance degradation có đến từ support explosion hay không.

---

## Experiment F — Full support dual

Sau khi E chứng minh support size là vấn đề, thay heuristic range regularizer bằng:

\[
\lambda_b^k
\]

và support constraint.

---

# 38. Ablation table đề xuất

| Variant | Evidence norm | Uncertainty norm | Active uncertainty dual | Support constraint |
|---|---:|---:|---:|---:|
| Current Dual-K | No | No | No | No |
| + Evidence Norm | Yes | No | No/weak | No |
| + Risk Norm | Yes | Yes | No/weak | No |
| + Active Dual | Yes | Yes | Yes | No |
| + Range Guardrail | Yes | Yes | Yes | Regularizer |
| Full Dual-Constraint | Yes | Yes | Yes | Dual |

---

# 39. Expected behavior sau sửa đổi

Ta không cần:

\[
K_C\approx K_A.
\]

Ta mong:

\[
K_C^{eff}
\]

và:

\[
K_A^{eff}
\]

khác nhau nhưng ổn định, không collapse.

Ví dụ:

\[
r_C\approx0.20
\]

\[
r_A\approx0.08
\]

có thể hoàn toàn hợp lý.

Điều quan trọng:

\[
\lambda_C,\lambda_A
\]

phải có phản ứng với risk.

---

# 40. Failure hypothesis được ưu tiên

Dựa trên log hiện tại, thứ tự hypothesis:

## H1 — Dual constraint inactive

Rất mạnh.

\[
\lambda_C,\lambda_A\rightarrow0.
\]

---

## H2 — C-branch support explosion

Rất mạnh.

\[
K_C^{eff}\approx45\sim60.
\]

---

## H3 — Evidence scale mismatch

Mạnh.

Threshold C cao hơn A nhưng C vẫn chọn nhiều hơn đáng kể.

---

## H4 — Uncertainty scale mismatch

Mạnh.

\[
R_C\approx0.2
\]

và:

\[
R_A\approx0.78.
\]

---

## H5 — A-branch starvation

Yếu hơn dự đoán ban đầu.

A-branch vẫn có:

\[
K_A\approx20\sim23
\]

và AUC2 vẫn tăng đáng kể trong training.

Do đó không nên ưu tiên warm-up A trước các vấn đề trên.

---

# 41. Revised minimal formulation

Phiên bản nên thử tiếp theo:

### C-branch

\[
\tilde e_t^C
=
\frac{
a_t-\mu_a
}{
\sigma_a+\epsilon
}
\]

\[
\tilde u_t^C
=
\frac{
H(a_t)
}{
\log2
}
\]

\[
\boxed{
\alpha_t^C
=
\sigma
\left(
\frac{
\tilde e_t^C
-\lambda_C\tilde u_t^C
-\theta_C
}{
\tau_C
}
\right)
}
\]

\[
K_C^{eff}
=
\sum_t\alpha_t^C.
\]

---

### A-branch

\[
\tilde e_t^A
=
\frac{
M_{t,y}-\mu_M
}{
\sigma_M+\epsilon
}
\]

\[
\tilde u_t^A
=
1-
\left(
P_y-P_{2nd}
\right)
\]

\[
\boxed{
\alpha_t^A
=
\sigma
\left(
\frac{
\tilde e_t^A
-\lambda_A\tilde u_t^A
-\theta_A
}{
\tau_A
}
\right)
}
\]

\[
K_A^{eff}
=
\sum_t\alpha_t^A.
\]

---

# 42. Revised MIL

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
S_A
=
\frac{
\sum_t\alpha_t^A M_{t,y}
}{
K_A^{eff}+\epsilon
}
}
\]

---

# 43. Revised dual risk

\[
\boxed{
R_C
=
\frac{
\sum_t\alpha_t^C\tilde u_t^C
}{
K_C^{eff}+\epsilon
}
}
\]

\[
\boxed{
R_A
=
\frac{
\sum_t\alpha_t^A\tilde u_t^A
}{
K_A^{eff}+\epsilon
}
}
\]

---

# 44. Revised dual update

\[
\boxed{
\lambda_C
\leftarrow
[
\lambda_C+\eta_\lambda(R_C-\delta_C)
]_+
}
\]

\[
\boxed{
\lambda_A
\leftarrow
[
\lambda_A+\eta_\lambda(R_A-\delta_A)
]_+
}
\]

với:

\[
\delta_C,\delta_A
\]

được chọn từ empirical normalized-risk distributions thay vì đặt tay ngay từ đầu.

---

# 45. Nếu C vẫn explosion sau revised dual

Thêm support ratio:

\[
r_C
=
\frac{K_C^{eff}}T.
\]

Debug regularizer:

\[
\boxed{
L_{C,support}
=
\max(0,r_C-r_{max}^C)^2.
}
\]

Nếu performance tăng rõ khi dùng term này, đó là bằng chứng:

\[
\boxed{
\text{C support explosion là failure mode thực}
}
\]

và lúc đó mới đáng chuyển sang full support-dual formulation.

---

# 46. Pseudocode — revised minimal version

```python
# --------------------------------------------------
# C BRANCH
# --------------------------------------------------

a = C_branch(X)

# per-video evidence normalization
e_c = (a - a.mean(dim=-1, keepdim=True)) / (
    a.std(dim=-1, keepdim=True) + eps
)

# normalized binary entropy
u_c = binary_entropy(a) / math.log(2.0)

z_c = (
    e_c
    - lambda_c * u_c
    - theta_c
)

alpha_c = torch.sigmoid(z_c / tau_c)

k_c_eff = alpha_c.sum(dim=-1)

score_c = (
    alpha_c * a
).sum(dim=-1) / (k_c_eff + eps)

r_c = (
    alpha_c * u_c
).sum(dim=-1) / (k_c_eff + eps)


# --------------------------------------------------
# A BRANCH
# --------------------------------------------------

M = A_branch(X)

e_a_raw = M[..., y]

e_a = (
    e_a_raw
    - e_a_raw.mean(dim=-1, keepdim=True)
) / (
    e_a_raw.std(dim=-1, keepdim=True)
    + eps
)

P = torch.softmax(M, dim=-1)

p_y = P[..., y]
p_2nd = max_other_class(P, y)

u_a = 1.0 - (p_y - p_2nd)
u_a = torch.clamp(u_a, 0.0, 1.0)

z_a = (
    e_a
    - lambda_a * u_a
    - theta_a
)

alpha_a = torch.sigmoid(z_a / tau_a)

k_a_eff = alpha_a.sum(dim=-1)

score_a = (
    alpha_a * e_a_raw
).sum(dim=-1) / (k_a_eff + eps)

r_a = (
    alpha_a * u_a
).sum(dim=-1) / (k_a_eff + eps)
```

---

# 47. Pseudocode — dual updates

```python
with torch.no_grad():

    lambda_c = torch.clamp(
        lambda_c
        + eta_lambda * (r_c.mean() - delta_c),
        min=0.0
    )

    lambda_a = torch.clamp(
        lambda_a
        + eta_lambda * (r_a.mean() - delta_a),
        min=0.0
    )
```

---

# 48. Logging bắt buộc cho run kế tiếp

```python
log({
    "e_c_mean": e_c_raw.mean(),
    "e_c_std": e_c_raw.std(),
    "e_a_mean": e_a_raw.mean(),
    "e_a_std": e_a_raw.std(),

    "u_c_mean": u_c.mean(),
    "u_a_mean": u_a.mean(),

    "r_c": r_c.mean(),
    "r_a": r_a.mean(),

    "lambda_c": lambda_c,
    "lambda_a": lambda_a,

    "k_c_eff": k_c_eff.mean(),
    "k_a_eff": k_a_eff.mean(),

    "support_ratio_c": (k_c_eff / T).mean(),
    "support_ratio_a": (k_a_eff / T).mean(),
})
```

---

# 49. Tiêu chí để quyết định có tiếp tục hướng Dual-\(K\) hay không

Hướng này đáng tiếp tục nếu sau sửa đổi:

1. \(\lambda_C,\lambda_A\) không collapse về 0 ngay lập tức.
2. \(K_C^{eff}\) không tiến gần toàn bộ sequence.
3. \(K_A^{eff}\) không tiến về 0.
4. Hai support khác nhau có hệ thống:

\[
K_C^{eff}\neq K_A^{eff}
\]

nhưng không collapse.
5. Performance phục hồi ít nhất về baseline VadCLIP trước khi thêm complexity mới.

Nếu sau normalization + active constraints mà:

\[
K_C^{eff}\approx K_A^{eff}
\]

trên hầu hết video, hypothesis về dual support có thể yếu.

Nếu:

\[
K_C^{eff}\neq K_A^{eff}
\]

và variation liên quan tới anomaly category / duration / semantic ambiguity, đó là bằng chứng tốt cho Dual-\(K\).

---

# 50. Kết luận

Failure hiện tại không nên được mô tả đơn giản là:

\[
K_C\text{ quá lớn},
\qquad
K_A\text{ quá nhỏ}.
\]

Root problem rõ hơn là:

\[
\boxed{
\lambda_C,\lambda_A\rightarrow0
}
\]

dẫn đến dual mechanism mất tác dụng.

Sau đó:

\[
\boxed{
\text{different evidence scales}
+
\text{fixed thresholds}
}
\]

trở thành yếu tố chính quyết định \(K_C\) và \(K_A\).

Ngoài ra, uncertainty-only constraint:

\[
R_b\leq\delta_b
\]

không trực tiếp ngăn support explosion.

Do đó roadmap sửa hợp lý là:

\[
\boxed{
\text{Normalize evidence}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Normalize uncertainty}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Choose active dual budgets}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Check whether C support still explodes}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Add support constraint only if needed}
}
\]

Mục tiêu cuối cùng vẫn là:

\[
\boxed{
K_C^{eff}
=
\sum_t\alpha_t^C
}
\]

và:

\[
\boxed{
K_A^{eff}
=
\sum_t\alpha_t^A
}
\]

được tạo bởi **hai branch-specific constrained selectors thực sự hoạt động**, thay vì chỉ là kết quả của hai fixed sigmoid thresholds.
