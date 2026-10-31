# VadCLIP + CNN visual memory — UCF-Crime

Thí nghiệm độc lập, không sửa `src/` và không sửa các thí nghiệm cũ.
Đây là code thử nghiệm; bản sửa đầu vào cached post-LGT chưa có kết quả GPU/AUC mới.

**Dataset đã xuất sau LGT:** thêm `--input-features post_lgt` vào lệnh train.
Feature `.npy` đi trực tiếp vào C/A và CNN; LGT được bỏ qua và đóng băng trong
run này. Loại đầu vào lưu trong checkpoint và được evaluate/predict đọc lại.
Mặc định `clip` vẫn giữ các lệnh/checkpoint cũ. Lệnh đúng cho dataset post-LGT
của người dùng nằm trong [JOINT_MSE.md](JOINT_MSE.md).

**Bản mới học chung `future_mse_joint`:** xem [JOINT_MSE.md](JOINT_MSE.md).
CNN, đầu chấm điểm và đầu dự đoán tương lai được học trong cùng batch với C/A;
không cần pretrain riêng hoặc đóng băng encoder. Dùng notebook và setup cũ,
không tạo notebook mới; evaluator gốc được giữ nguyên.

**Biến thể hai giai đoạn `future_mse`:** dùng notebook Kaggle và các cell setup/CSV cũ,
chạy các lệnh trong [FUTURE_MSE.md](FUTURE_MSE.md). Học CNN nhân quả trước LGT bằng
MSE dự đoán feature CLIP tương lai, bỏ đầu dự đoán, đóng băng encoder rồi học đầu
bổ trợ C/A. Không dùng prototype/cosine trong nhánh mới. Mặc định `prototype`
vẫn chạy thí nghiệm cũ được mô tả bên dưới; lệnh cũ/checkpoint cũ vẫn được hỗ trợ.

## Kiến trúc và cách học

```text
CLIP features -> LGT gốc -> F [B,T,512] -> C/A gốc -> logits C/A
                            |
                            +-> transpose [B,512,T]
                                -> Conv1D(512,256,k=3,p=1) -> GELU
                                -> Conv1D(256,256,k=1)
                                -> transpose + L2 [B,T,256]
                                -> cosine với 8 mẫu Normal và 8 mẫu Abnormal
                                -> hai đầu Linear: delta C [B,T,1], delta A [B,T,14]
                                -> logits C/A + 0.1 * delta C/A
```

- CNN chỉ thay lớp học đặc trưng của **nhánh phụ**. MLP trong C/A và LGT giữ kiến trúc hiện tại.
- Visual Prompt vẫn nhận logits C gốc, trước phần hiệu chỉnh; không tạo vòng phản hồi.
- Các mẫu là tham số thị giác học được, không phải text prototype, không lấy từ dữ liệu test.
- Nhánh phụ so khớp từng vị trí với cùng bộ mẫu. Đây không phải việc lưu tất cả frame của video rồi tìm frame trùng nhau, cũng không dò vị trí vật thể trong ảnh.
- Hai đầu hiệu chỉnh khởi tạo bằng 0. Trong hai epoch đầu nhánh phụ không chạy; từ epoch 3 bật cả residual và loss memory. Trạng thái bật/tắt được lưu trong checkpoint.
- `--aux-branch c/a` chỉ giới hạn nơi cộng residual. Khi train, CNN vẫn có gradient về feature/LGT dùng chung nên cả hai metric có thể thay đổi.

Loss: `CLAS2(C_mới) + CLASM(A_mới) + loss3_gốc + 0.1 * CLAS2(memory_logit)`.
`memory_logit = logsumexp(sim_Abnormal / tau) - logsumexp(sim_Normal / tau)`, `tau=0.1`.
Loss memory dùng cùng hard Top-K và nhãn **mức video**; không giả định mọi vị trí trong video bất thường đều bất thường.
Nó giúp hai nhóm mẫu học ý nghĩa Normal/Abnormal thay vì tăng điểm chỉ vì cảnh xuất hiện nhiều lần.

## Chạy trên Kaggle

Bật GPU và Internet (để tải CLIP lần đầu nếu chưa có cache). Dùng các `.npy` CLIP UCF-Crime đã giải nén; không đưa video thô hoặc file nén feature trực tiếp cho trainer.

**Dùng từ GitHub:** import `kaggle_cnn_memory.ipynb` trong thư mục này vào Kaggle, thêm bộ feature làm Input rồi chạy các cell theo thứ tự. Notebook clone repo nếu chưa có và dùng mã thí nghiệm trực tiếp từ checkout. Nếu đã có repo Kaggle cũ, cập nhật trước:

```bash
!git -C /kaggle/working/VadCLIP pull --ff-only origin main
```

**Cách chuyển mã tùy chọn:** thêm `vadclip_cnn_memory_20261022.zip` làm Input. Notebook cũng hỗ trợ ZIP hoặc thư mục mã đã giải nén, chỉ chép mã vào `experiments/cnn_memory/`.

Cell setup cài `ftfy`, `regex` và kiểm tra import `from pkg_resources import packaging` mà CLIP trong `src/` đang dùng. Nếu import này lỗi, cell cài `setuptools==69.5.1` cho môi trường Kaggle; không sửa `src/`. Lý do tương thích: [setuptools 70 bỏ export packaging từ pkg_resources](https://github.com/pypa/setuptools/issues/4385). Nếu dùng môi trường VadCLIP đang chạy ổn, kiểm tra này giữ nguyên setuptools hiện có.

Nếu bạn đã cập nhật repo và có hai CSV Kaggle từ các lần chạy trước, chạy:

```python
%cd /kaggle/working/VadCLIP
```

Kiểm tra dữ liệu trước, chưa tải CLIP hoặc train:

```bash
!python experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --output-dir outputs/cnn_memory_both_k3_s234 \
  --preflight
```

Chạy CNN k3 bổ trợ cả A/C:

```bash
!python experiments/cnn_memory/train.py \
  --train-list /kaggle/working/ucf_CLIP_rgb_kaggle.csv \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --aux-branch both --cnn-kernel 3 --seed 234 \
  --output-dir outputs/cnn_memory_both_k3_s234
```

Test lại checkpoint tốt nhất; kiến trúc, kernel, nhánh và trạng thái kích hoạt tự đọc từ checkpoint:

```bash
!python experiments/cnn_memory/evaluate.py \
  --test-list /kaggle/working/ucf_CLIP_rgbtest_kaggle.csv \
  --checkpoint outputs/cnn_memory_both_k3_s234/best.pt \
  --log-path outputs/cnn_memory_both_k3_s234/evaluate.log
```

Nếu chưa có CSV Kaggle, script sau tìm file theo đúng tên `.npy`, giữ thứ tự nhãn và crop của CSV gốc:

```bash
!python experiments/cnn_memory/prepare_csv.py --feature-root /kaggle/input
```

Sau đó dùng `--train-list outputs/cnn_memory_data/train.csv --test-list outputs/cnn_memory_data/test.csv` trong các lệnh trên. Nếu có nhiều bản feature cùng tên, chỉ định `--feature-root` hẹp hơn để chọn đúng một bộ. Script không đoán file khi thiếu hoặc trùng tên. Không đổi thứ tự test CSV vì GT đi theo thứ tự này; preflight kiểm tra độ dài nhưng không chứng minh được nội dung GT khớp khi bạn tự đổi thứ tự.

## Các run đối chiếu

Giữ cùng CSV, seed, batch size và lịch train. Thay cặp tùy chọn và thư mục output trong cùng lệnh train:

| Run | Tùy chọn | Output riêng |
| --- | --- | --- |
| Control, tắt nhánh phụ | `--aux-branch none` | `outputs/cnn_memory_control_s234` |
| CNN k1, so khớp từng vị trí | `--aux-branch both --cnn-kernel 1` | `outputs/cnn_memory_both_k1_s234` |
| CNN k3, thêm ngữ cảnh lân cận | `--aux-branch both --cnn-kernel 3` | `outputs/cnn_memory_both_k3_s234` |
| Chỉ cộng vào C | `--aux-branch c --cnn-kernel 3` | `outputs/cnn_memory_c_k3_s234` |
| Chỉ cộng vào A | `--aux-branch a --cnn-kernel 3` | `outputs/cnn_memory_a_k3_s234` |

Run đầu dùng mặc định 10 epoch, seed 234, batch size 64 cho **mỗi** loader Normal/Abnormal (batch ghép 128), AdamW: LR nền `2e-5`, LR nhánh mới `2e-4`, milestones `[4,8]`, gamma `0.1`. Không bật Temporal-A, PCA, ranking, Dual-K, AIS, AMP hoặc tail shift.

## Metric, checkpoint và resume

- Train và test lại checkpoint gọi chung `experiments/topk_variants/test_ucf.py::test`, giữ luồng đánh giá UCF gốc: chia chunk/mask theo `visual_length`, cắt logits về độ dài hợp lệ, `sigmoid(C)`, `1-softmax(A)[Normal]`, repeat 16 và detection mAP tại IoU 0.1–0.5. Evaluator CNN chỉ chuyển logger và nhận đầy đủ metric; không còn vòng inference riêng hoặc tự bỏ chunk đệm. Cần cập nhật đồng bộ code CNN, MSC và `experiments/topk_variants/test_ucf.py` trong cùng checkout.
- Text log dùng cùng cấu trúc các run Top-K trước: `epoch=e/N batch=b/M loss=... loss1=... loss2=... loss3=... lr=... peak_vram=...GB`. `loss1/2/3` tương ứng C, A và text separation. Dòng tiến độ ghi loss **batch hiện tại**; `epoch_summary` ghi trung bình đầy đủ toàn epoch. Loss phụ hiện riêng ở `loss_memory`, `weighted_memory` và các trường trung bình tương ứng.
- AUC1/AP1 và AUC2/AP2 nằm ở hai dòng riêng, mỗi ngưỡng mAP một dòng, kết thúc bằng `average_mAP=...%`. Header ghi seed/device/GPU; marker `checkpoint_resumed` và `model_saved` dùng để đọc lại quá trình chạy. Đổi format text không thay đổi lịch train/eval hoặc dữ liệu checkpoint/JSON.
- `train.log`, `config.json`, `metrics.jsonl`: cấu hình, lịch kích hoạt, loss, AUC1/AP1/AUC2/AP2 và mAP. mAP có đơn vị phần trăm; AUC/AP nằm trong `[0,1]`.
- `best.pt`: checkpoint có **test AUC1 cao nhất từ epoch 3**, cho cả control và run mới. AUC2 được ghi cùng lần đánh giá, không chọn checkpoint riêng theo AUC2. Đây là best-test, không phải kết quả validation độc lập.
- `last.pt`: cuối mỗi epoch, gồm optimizer, scheduler, RNG và trạng thái nhánh phụ. Dùng để resume ở epoch kế tiếp, không tiếp tục giữa epoch.
- Đánh giá mỗi 10 optimizer update và cuối mỗi epoch, dùng raw `sigmoid(C)` và `1-softmax(A)[Normal]`, repeat 16; không thêm hậu xử lý làm mượt.
- Trainer riêng train liên tục qua epoch, **không reload best vào model train sau mỗi epoch như script `src/ucf_train.py`**. Vì vậy control của trainer này mới là đối chiếu trực tiếp; không coi nó là bản chạy nguyên trạng script train gốc hoặc so gain với log cũ khác trainer.
- Module CNN được khởi tạo trong ngữ cảnh bảo toàn RNG để không làm lệch thứ tự DataLoader so với control; kiểm tra eval cũng bảo toàn RNG. Việc này không bảo đảm CUDA hoàn toàn deterministic.

Resume: giữ cả `best.pt` và `last.pt`, chạy lại **đúng các tùy chọn cũ**, cùng `--output-dir`, thêm:

```text
--resume outputs/cnn_memory_both_k3_s234/last.pt
```

Có thể tăng `--max-epoch`; các cài đặt ảnh hưởng thí nghiệm được kiểm tra khi resume. Run mới không ghi đè thư mục output đã có dữ liệu.

## Kiểm tra local

```bash
python -B -m unittest discover -s experiments/cnn_memory/tests -v
```

18 kiểm tra CPU đã qua. Các kiểm tra dùng CLIP giả lập và thay DistanceAdj bằng ma trận đơn vị **chỉ trong fixture**, để kiểm tra forward nền, gradient, padding, phân nhánh và checkpoint. Một fixture khác chạy entrypoint train hai epoch với detector nhỏ trên CPU, xác nhận resume cho cùng weights và metrics như chạy liên tục; log từ fixture được đọc bằng mẫu parser cũ để kiểm tra epoch/batch, AUC và đơn vị mAP. Kiểm thử hồi quy evaluator dùng dữ liệu giả đủ 14 lớp, tính AUC/AP và mAP thật, xác nhận khớp evaluator `src/ucf_test.py` và Top-K về metric, softmax prediction, chunk, length và mask, gồm cả độ dài chia hết `visual_length`. Chưa chạy CLIP pretrained/CUDA hoặc train toàn bộ UCF-Crime ở máy local. LGT gốc dùng CUDA nên các entrypoint train/evaluate thật yêu cầu GPU.
