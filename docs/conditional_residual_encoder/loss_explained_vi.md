# Encoder học quy luật phần dư: kiến trúc, label và loss

Ngày: 21/10/2026. Bản thiết kế nghiên cứu v1, chưa train hoặc benchmark. Những hyperparameter dưới đây là đề xuất pilot, không phải recipe được paper xác nhận.

File chỉnh sửa được: [conditional_residual_encoder_vi.drawio](conditional_residual_encoder_vi.drawio).

## 1. Đọc sơ đồ theo ngôn ngữ đơn giản

Ta giao cho model một bài tập: xem phần dư của một số vùng trong ảnh thật, rồi đoán thống kê phần dư ở những vùng khác.

- Những vùng được xem gọi là context: encoder đọc chúng để tạo một bản mô tả bằng số.
- Những vùng cần đoán gọi là target: phần dư của chúng không được đưa vào encoder hoặc predictor.
- Predictor được biết vài đặc điểm thô của target, như sáng/tối và nhiều/ít cạnh. Nó cần thông tin này vì các vùng khác nhau không nhất thiết có phần dư giống nhau.
- Đáp án là các thống kê do máy tính đo trực tiếp từ target. Không cần người gán nhãn real/fake.
- Loss là điểm phạt cho dự đoán sai và cho embedding bị mất thông tin.

Embedding ở đây chỉ là dãy số model học. Chưa có bằng chứng nó chính là thông số camera hoặc pipeline vật lý.

## 2. File draw.io có bốn trang

1. Kiến trúc tổng thể: luồng ảnh, context, target, label, predictor, loss.
2. Encoder và label: các lớp CNN, tensor shapes, cách đo đáp án.
3. Công thức loss: prediction NLL và ba regularizer, có giải thích từng biến.
4. Train và GMM: một training step, data splits và bước xây detector sau training.

Preview:

- [Kiến trúc](01_architecture.png)
- [Encoder và label](02_encoder_and_labels.png)
- [Công thức loss](03_losses.png)
- [Train và GMM](04_training_and_gmm.png)

Các hình PNG được render từ cùng nội dung và tọa độ với draw.io; chưa xác minh bằng cách mở diagrams.net. XML, ID và liên kết cạnh đã được kiểm tra tự động; preview đã được kiểm tra trực quan.

## 3. Từ điển tất cả biến

| Ký hiệu | Nghĩa | Kích thước hoặc giá trị pilot |
|---|---|---|
| \(x_b\) | Ảnh thật thứ b trong batch | RGB, 3×256×256, giá trị [0,1] |
| \(B\) | Số ảnh trong một batch | Đề xuất64 |
| \(b\) | Chỉ số của một ảnh | 1 đến B |
| \(R\) | Bộ lọc phần dư, giữ nguyên trọng số | FRE pretrained, grayscale1→8 |
| \(r_b=R(x_b)\) | Phần dư của ảnh thứ b | 8×256×256 |
| \(A,B_{\mathrm{ctx}}\) | Tên hai nhóm context của cùng ảnh | Mỗi nhóm3 patch; không phải batch size B |
| \(v\) | Chọn một trong hai nhóm context | A hoặc B |
| \(E_\theta\) | Encoder biến một nhóm context thành embedding | CNN và pooling dùng chung |
| \(\theta\) | Toàn bộ tham số được học của encoder | Conv, GroupNorm, Linear |
| \(z_b^v\) | Bản mô tả bằng số của ảnh b, suy ra từ context v | 64 số |
| \(D\) | Số chiều của embedding z | 64 |
| \(k,l\) | Chỉ số chiều của embedding z | 1 đến64 |
| \(Z_v\) | Ma trận embedding của tất cả ảnh trong batch, từ context v | B×64 |
| \(T\) | Số patch target mỗi ảnh | 2 |
| \(t\) | Chỉ số patch target | 1 hoặc2 |
| \(Q\) | Số ô nhỏ trong một patch target | 16 ô16×16 |
| \(q\) | Chỉ số ô nhỏ | 1 đến16 |
| \(J\) | Số thống kê phần dư mỗi ô | 24 |
| \(j\) | Chỉ số thống kê phần dư | 1 đến24 |
| \(u_{btqj}\) | Giá trị thống kê thô đo từ phần dư target | Một số thực |
| \(m_j,a_j\) | Mean và std của thống kê j trên encoder-train | Cố định trước tối ưu model |
| \(y_{btqj}\) | Label: thống kê target sau chuẩn hóa | Một số thực |
| \(c_{btq}\) | 5 điều kiện thô của ô target | Vector5 |
| \(P_\psi\) | Predictor nhận embedding và điều kiện, xuất mean/log-variance | MLP69→128→128→48 |
| \(\psi\) | Tham số được học của predictor | Các Linear layers |
| \(\mu_{bvtqj}\) | Giá trị trung tâm predictor dự đoán cho label j | Một số thực |
| \(s_{bvtqj}\) | Log của variance predictor dự đoán | Clamp trong [-6,6] |
| \(\sigma^2=\exp(s)\) | Variance dự đoán: mức dao động quanh mean | Dương; khác variance của z theo batch |
| \(\ell_{bvtqj}\) | Điểm phạt Gaussian NLL cho một label | Một số thực, có thể âm |
| \(\lambda\) | Trọng số gói regularizer | Pilot0,1; đối chứng0 |
| \(\epsilon_{\mathrm{std}}\) | Số nhỏ tránh vấn đề sqrt/gradient trong variance loss | 1e-4 |
| \(\epsilon_E,\epsilon_\rho\) | Số nhỏ cho energy/correlation của residual | Pilot1e-8; phải khóa residual scale |
| \(\epsilon_{\mathrm{pool}}\) | Số nhỏ khi lấy std trong pooling | 1e-6 |
| \(h(x)\) | Vector48 thống kê prediction surprises của một ảnh | Dùng fit Gaussian/GMM sau train |
| \(\tau\) | Ngưỡng chấm ảnh bất thường | Percentile95 trên real-calibration |

Chữ “mean” là trung bình cộng; “std” là độ lệch chuẩn; “variance” là bình phương std; “covariance” mô tả hai chiều cùng thay đổi ra sao. “Frozen” nghĩa là trọng số không cập nhật; “stop-gradient” nghĩa là không truyền gradient qua phép tính đó.

## 4. Tạo dữ liệu và label

### 4.1 Chọn patch

Từ ảnh đủ kích thước, lấy native crop256×256; không resize ảnh nhỏ để ép đủ kích thước. Shared rotation90°/flip được áp dụng trước residual extraction.

Tạo9 vị trí có góc trái trên tại các kết hợp x,y thuộc {0,80,160}. Mỗi core64×64, giữa hai core kề nhau có khoảng trống16 pixel. Random chọn8 vị trí và chia A3, B3, target2.

Khoảng trống16 phù hợp giả định FRE có effective receptive field15×15, radius7. Khi đó vùng pixel gốc tác động lên các core liền kề không giao nhau. Nếu residual extractor có receptive field lớn hơn, phải tăng khoảng trống hoặc đổi sampling; không tự coi “không giao nhau về tọa độ output” là không rò rỉ.

Native crop256 có thể làm số ảnh đủ điều kiện giảm; cần lưu eligibility và report số ảnh. Bản pilot học trên vùng crop, chưa sử dụng toàn bộ ảnh.

### 4.2 Ba loại thống kê làm đáp án

Với một ô16×16 và một residual channel:

\[
e=\log\left(\frac1{256}\sum_{i=1}^{16}\sum_{n=1}^{16}r_{in}^2+\epsilon_E\right).
\]

- \(r_{in}\): giá trị phần dư tại hàng i, cột n trong ô, trên một channel.
- Tổng bình phương chia256 đo energy trung bình.
- Log giúp nén khoảng giá trị; \(\epsilon_E\) tránh log0.
- e là energy, không tự đồng nhất với sensor noise variance.

Correlation giữa hai vector số a,b được định nghĩa:

\[
\operatorname{corr}(a,b)=
\frac{\sum_n(a_n-\bar a)(b_n-\bar b)}
{\sqrt{\left[\sum_n(a_n-\bar a)^2\right]\left[\sum_n(b_n-\bar b)^2\right]}+\epsilon_\rho}.
\]

Ở công thức này a,b chỉ là hai vector pixel để tính correlation, không phải context groups hoặc chỉ số batch; \(\bar a,\bar b\) là trung bình từng vector.

- Correlation ngang: flatten ô[:,0:15] và ô[:,1:16], mỗi vector240 số.
- Correlation dọc: flatten ô[0:15,:] và ô[1:16,:], mỗi vector240 số.
- Cả hai lấy trong cùng residual channel. Ô gần hằng số có correlation kém tin cậy; epsilon giúp hữu hạn, không tạo thêm thông tin.

Nối theo thứ tự cố định: energy8, correlation ngang8, correlation dọc8. Tổng cộng24 label/ô. Không phải24 lớp phân loại.

Chuẩn hóa:

\[
y_{btqj}=\frac{u_{btqj}-m_j}{\max(a_j,10^{-3})}.
\]

Thu thập một tập train descriptors cố định với seed/membership đã lưu để ước lượng \(m_j,a_j\), rồi freeze chúng. Không thay mean/std mỗi batch và không dùng real-calibration hoặc final-test để fit scaler.

### 4.3 Điều kiện cục bộ c

Low-pass RGB bằng average filter7×7, stride1, reflect padding3. Gọi ảnh đã lọc là \(\widetilde x\), nằm trong [0,1].

Với mỗi ô target:

\[
c=[\operatorname{mean}(\widetilde R),
\operatorname{mean}(\widetilde G),
\operatorname{mean}(\widetilde B),
\operatorname{std}(\widetilde L),
g].
\]

\[
\widetilde L=0.299\widetilde R+0.587\widetilde G+0.114\widetilde B,
\quad
g=\tfrac12\left[\operatorname{mean}((\Delta_x\widetilde L)^2)
+\operatorname{mean}((\Delta_y\widetilde L)^2)\right].
\]

- Ba mean màu cho biết vùng đó có màu/sáng trung bình thế nào.
- Std luminance cho biết mức biến động sáng trong vùng; dùng population std.
- \(\Delta_x,\Delta_y\): hiệu giữa hai pixel kề ngang/dọc trong ô; g đo năng lượng cạnh.
- Không đưa residual target hoặc label y vào input predictor.
- Có thể standardize năm điều kiện bằng scaler riêng chỉ fit trên encoder-train và lưu cùng artifact.

Low-pass không bảo đảm loại bỏ mọi thông tin có thể suy ra label; c-only control là bắt buộc để đo phần lợi ích của context encoder.

## 5. Encoder và predictor chính xác

Mỗi residual patch đầu vào8×64×64:

| Layer | Output mỗi patch |
|---|---|
| Conv3×3, stride2, padding1, 8→32 + GroupNorm8 + GELU | 32×32×32 |
| Conv3×3, stride2, padding1, 32→64 + GroupNorm8 + GELU | 64×16×16 |
| Conv3×3, stride2, padding1, 64→128 + GroupNorm8 + GELU | 128×8×8 |
| Conv3×3, stride2, padding1, 128→128 + GroupNorm8 + GELU | 128×4×4 |
| Spatial mean128 và std128, nối lại | 256 |
| Linear256→128 | Patch embedding128 |

“GroupNorm8” là8 nhóm, không phải8 channels. Các convolution dùng cùng trọng số cho mọi patch và hai context groups.

Trong spatial/set pooling, dùng population variance và \(\operatorname{std}=\sqrt{\operatorname{Var}+10^{-6}}\). Với3 patch embeddings, lấy mean128 và std128 qua ba patch, nối lại thành256. MLP256→128→64, GELU ở giữa, tạo z.

Không thêm L2 normalization ở cuối z. Set pooling không dùng tọa độ hoặc thứ tự patch làm input.

Với mỗi context embedding z và mỗi ô target, nối z64 với c5. Predictor:

~~~text
Linear69→128 → GELU → Linear128→128 → GELU → Linear128→48
~~~

24 output đầu là \(\mu\), 24 output sau là s. Clamp s trong [-6,6] trước khi tính NLL. Đây là lựa chọn numerical pilot; cần log số output chạm biên vì clamp có thể làm gradient vùng ngoài bằng0.

Tensors:

- z cho hai groups: B×2×64.
- c: B×2target×16ô×5.
- y: B×2target×16ô×24.
- \(\mu,s\): B×2context×2target×16ô×24.

Các thống kê energy/correlation không nhất thiết thực sự Gaussian hoặc độc lập. Diagonal Gaussian head là một scoring approximation ban đầu; nếu sai mô hình nghiêm trọng, có thể thử mixture density hoặc conditional flow trong một ablation riêng.

GroupNorm có thể giảm độ nhạy với biên độ residual. Nếu context embedding bỏ lỡ thông tin energy, cần đối chứng CNN không normalization hoặc thêm fixed energy descriptors từ context; không thay đổi cùng lúc với loss trong run chính.

## 6. Loss chính: conditional Gaussian negative log-likelihood

Cho từng thống kê j:

\[
y_{btqj}\mid c_{btq},z_b^v
\sim
\mathcal N\left(\mu_{bvtqj},\exp(s_{bvtqj})\right).
\]

Nghĩa: predictor đoán trung tâm và độ dao động của đáp án khi biết đặc điểm vùng target và embedding từ context.

Bỏ hằng số không ảnh hưởng gradient:

\[
\ell_{bvtqj}
=\frac12\left[
\exp(-s_{bvtqj})(y_{btqj}-\mu_{bvtqj})^2+s_{bvtqj}
\right].
\]

Loss chính:

\[
\boxed{
\mathcal L_{\mathrm{pred}}
=
\frac{1}{2BTQJ}
\sum_{b=1}^{B}
\sum_{v\in\{A,B\}}
\sum_{t=1}^{T}
\sum_{q=1}^{Q}
\sum_{j=1}^{J}\ell_{bvtqj}
}
\]

Số2 trong mẫu số là hai context groups. Hệ số1/2 trong \(\ell\) là hệ số Gaussian; không được nhầm hai hệ số này. Cả context A và B dự đoán cùng target labels.

Giải thích:

- \((y-\mu)^2\): mean đoán sai bị phạt.
- \(\exp(-s)\): mức phạt tăng nếu model dự đoán variance nhỏ nhưng lại sai.
- \(+s\): ngăn model tăng variance vô hạn để giảm term sai số.
- NLL của phân phối liên tục có thể âm. Không dùng quy tắc “loss âm là sai”.

Ví dụ một chiều:

| Label y | Mean đoán μ | Variance đoán | s | Loss bỏ hằng số |
|---|---|---|---|---|
| 2 | 1 | 1 | 0 | 0,5 |
| 2 | 1 | 4 | log4 | Khoảng0,818 |
| 2 | 1 | 0,25 | log0,25 | Khoảng1,307 |

Đây là số minh họa, không phải kết quả training. Model được thưởng khi độ bất định phù hợp với mức sai số; không phải mọi lần tăng variance đều làm loss tăng.

Gradient trước clamp, cho một phần tử:

\[
\frac{\partial\ell}{\partial\mu}
=\exp(-s)(\mu-y),
\qquad
\frac{\partial\ell}{\partial s}
=\frac12[1-\exp(-s)(y-\mu)^2].
\]

Chúng giải thích cách predictor điều chỉnh mean và variance. Gradient từ predictor tiếp tục truyền vào z và encoder. Label và điều kiện c được giữ cố định.

## 7. Loss phụ thứ nhất: invariance

\[
\mathcal L_{\mathrm{inv}}
=\frac{1}{BD}\sum_{b=1}^{B}\sum_{k=1}^{D}
(z^A_{bk}-z^B_{bk})^2.
\]

Hai nhóm context của cùng ảnh nên tạo embedding tương đối giống nhau vì có thể chia sẻ thông tin acquisition/processing.

Nếu chỉ dùng term này, encoder có thể xuất cùng vector cho mọi ảnh; khi đó loss bằng0 nhưng embedding vô ích. Ngoài ra, các vùng cùng ảnh có thể chịu xử lý cục bộ khác nhau, nên không ép invariance bằng mọi giá.

## 8. Loss phụ thứ hai: variance floor

Với một group v, tính mean embedding qua các ảnh:

\[
\bar z^v_k=\frac1B\sum_b z^v_{bk}.
\]

Sample variance cho chiều k:

\[
V^v_k=\frac1{B-1}\sum_b(z^v_{bk}-\bar z^v_k)^2.
\]

Penalty cho group:

\[
\mathcal V(Z_v)=
\frac1D\sum_{k=1}^{D}
\max(0,1-\sqrt{V^v_k+\epsilon_{\mathrm{std}}}).
\]

\[
\mathcal L_{\mathrm{var}}=
\frac{\mathcal V(Z_A)+\mathcal V(Z_B)}2.
\]

Giải thích:

- Đây là variance của embedding giữa các ảnh, không phải variance mà predictor đoán cho residual label.
- Nếu mọi ảnh cho số gần giống nhau tại chiều k, std thấp và bị phạt.
- Mốc1 quy định scale mong muốn của embedding trong regularizer, không có nghĩa noise ảnh thật có std1.
- Không L2-normalize z vì việc đó xung đột với mục tiêu std1 ở từng chiều.
- B phải ít nhất2; batch quá nhỏ làm ước lượng rất nhiễu.

Variance floor không loại trừ mọi dạng collapse và không đảm bảo thông tin học được là forensic. Cần đo eigen-spectrum/effective rank và đánh giá downstream.

## 9. Loss phụ thứ ba: covariance

Trừ mean mỗi chiều để có ma trận centered:

\[
\widetilde Z_v=Z_v-\mathbf1(\bar z^v)^\top.
\]

Trong đó \(\mathbf1\) là vector B phần tử đều bằng1, để lặp mean cho mọi hàng.

\[
C_v=\frac{\widetilde Z_v^\top\widetilde Z_v}{B-1}.
\]

C có kích thước64×64. Phần tử hàng k, cột l mô tả hai chiều embedding cùng thay đổi thế nào giữa các ảnh.

\[
\mathcal C(Z_v)=\frac1D\sum_{k\ne l}(C_v)_{kl}^2,
\qquad
\mathcal L_{\mathrm{cov}}=\mathcal C(Z_A)+\mathcal C(Z_B).
\]

Không phạt đường chéo vì đó là variance đã được term trước xử lý. Term này giảm các chiều trùng thông tin; không ép phân phối mỗi ảnh thành Gaussian và không chứng minh các chiều độc lập hoàn toàn.

Centered covariance có rank tối đa B−1. Với B64 và D64, không kỳ vọng cùng lúc đạt covariance diagonal đầy đủ rank hoàn hảo trên mỗi batch. Loss không về0 không tự chứng minh training hỏng.

## 10. Loss tổng

\[
\boxed{
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{pred}}
+\lambda\left[
25\mathcal L_{\mathrm{inv}}
+25\mathcal L_{\mathrm{var}}
+\mathcal L_{\mathrm{cov}}
\right]
}
\]

Pilot \(\lambda=0.1\). Ba hệ số25/25/1 dựa trên cấu trúc VICReg, nhưng trọng số0,1 trước gói và cách ghép vào conditional prediction là đề xuất mới, chưa tối ưu.

Ví dụ nếu L_pred=0,7, L_inv=0,02, L_var=0,1 và L_cov=0,4:

\[
L_{\mathrm{total}}=0.7+0.1(25\cdot0.02+25\cdot0.1+0.4)=1.04.
\]

Log cả bốn loss và gradient scale riêng. Chạy ít nhất đối chứng \(\lambda=0\) để biết regularizer có giúp không. Không lấy hệ số này làm điều kiện bất biến nếu các term có scale quá khác.

## 11. Luồng gradient: phần nào thực sự được train?

| Thành phần | Update bằng gradient? | Lý do |
|---|---|---|
| FRE pretrained | Không | Giữ tín hiệu/đáp án phần dư ổn định |
| Energy/correlation label extractor | Không | Đáp án không được tự đổi để giảm loss |
| Label scaler và condition scaler | Không | Fit từ encoder-train, sau đó khóa |
| Low-pass và condition extraction | Không | Tránh nhánh này tự mã hóa target residual |
| Patch CNN, pooling projection, context MLP | Có | Học embedding giúp giải thích target |
| Predictor MLP | Có | Học ánh xạ từ embedding + điều kiện đến phân phối label |
| Gaussian/GMM cuối | Không trong bước SSL | Fit ở giai đoạn riêng sau khi freeze E/P |

Không có nhãn fake trong training encoder, prediction targets, fitting scaler, Gaussian hoặc calibration.

## 12. Training step và cấu hình pilot

~~~python
# Pseudocode; tensor operations/broadcasting cần triển khai theo shape bên trên.
images = next(real_train_loader)  # B ảnh thật

with no_grad():
    residual = frozen_fre(grayscale(images))
    context_a, context_b, target = sample_disjoint_with_gaps(residual)
    target_labels = fixed_label_scaler(residual_statistics(target))
    target_conditions = fixed_condition_scaler(
        coarse_conditions(images, target_coordinates)
    )

embedding_a = encoder(context_a)
embedding_b = encoder(context_b)

mean_a, log_variance_a = predictor(embedding_a, target_conditions)
mean_b, log_variance_b = predictor(embedding_b, target_conditions)

prediction_loss = 0.5 * (
    gaussian_nll_mean(target_labels, mean_a, log_variance_a.clamp(-6, 6))
    + gaussian_nll_mean(target_labels, mean_b, log_variance_b.clamp(-6, 6))
)

invariance, variance, covariance = embedding_regularizers(
    embedding_a, embedding_b
)
loss = prediction_loss + 0.1 * (
    25 * invariance + 25 * variance + covariance
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
~~~

Pilot: B64, AdamW LR1e-4 và weight decay1e-4; không weight decay lên bias/norm parameters. Warmup5% updates, cosine xuống1e-6,20 epochs. Đây chưa phải cấu hình được đo VRAM hoặc throughput. Tính residual statistics, covariance và NLL bằng FP32; nếu dùng mixed precision cần xử lý scaler/overflow đúng.

Validation dùng fixed patch coordinates/group assignments. Chọn checkpoint bằng mean prediction NLL trên encoder-validation theo quy tắc khóa trước; log thêm std, effective rank, clamp fraction và các loss phụ. Không chọn bằng final-test unseen AUROC.

## 13. Sau training: giữ cả encoder và predictor

Khác Forensic Twins bỏ projector, bản này cần predictor để tạo prediction surprises.

Freeze E và P; dùng cùng preprocessing, label scaler và một layout sampling xác định cho real-fit/calibration/test. Có thể khóa A là các vị trí row-major0,1,2; B là3,4,5; target6,7; bỏ vị trí8 trên lưới9 core. Đây chỉ là lựa chọn pilot; phải khóa layout trước đánh giá.

Đối với mỗi ảnh:

\[
a_{vtqj}=\frac12\left[\exp(-s_{vtqj})(y_{tqj}-\mu_{vtqj})^2+s_{vtqj}\right].
\]

Ở đây a là surprise/NLL của một thống kê, khác a_j là std trong label scaler. Để tránh nhầm trong code, nên đặt tên surprise và label_std riêng.

Với mỗi chiều j, tổng hợp64 giá trị từ2 context×2 target×16 ô:

\[
\operatorname{mean}_j=\operatorname{mean}_{v,t,q}a_{vtqj},
\quad
\operatorname{std}_j=\sqrt{\operatorname{Var}_{v,t,q}(a_{vtqj})+\epsilon_{\mathrm{pool}}}.
\]

\[
h(x)=[\operatorname{mean}_1,\ldots,\operatorname{mean}_{24},
\operatorname{std}_1,\ldots,\operatorname{std}_{24}]\in\mathbb R^{48}.
\]

Fit một Gaussian K1 full covariance trên h của real-density-fit, với covariance regularization. Nếu scale các chiều h khác nhau, fit một scaler h chỉ trên real-fit và dùng lại ở mọi giai đoạn. Không cần mặc định thêm PCA khi chỉ có48 chiều.

\[
\operatorname{score}(x)=-\log p_{\mathrm{GMM}}(h(x)),
\qquad
\tau=\operatorname{quantile}_{0.95}\{\operatorname{score}(x):x\in\text{real-calibration}\}.
\]

Score lớn hơn ngưỡng nghĩa là bất thường so với real reference. Ngưỡng này nhắm empirical FPR5% trên calibration, không bảo đảm FPR5% với camera/codec/source mới và không phải xác suất fake.

Gaussian/GMM ở đây mô hình hóa phân phối vector prediction surprises, không trực tiếp mô hình hóa pixel hoặc residual noise.

## 14. Kiểm chứng model có học đúng nhiệm vụ

1. Train một predictor chỉ nhận c. Nếu model đầy đủ không hơn nó trên held-out real NLL, context encoder chưa chứng minh hữu ích.
2. Hoán đổi z giữa các ảnh khi validation. Nếu NLL hầu như không đổi, predictor có thể bỏ qua z. Đây là diagnostic, không thay thế một model c-only được train công bằng.
3. Đánh giá real camera/source/codec chưa thấy. “Học acquisition” mà chỉ nhận ra camera train thì chưa giải quyết generalization.
4. Đánh giá unseen-fake AUROC và TPR ở threshold real-calibration, kèm actual real-test FPR.
5. So \(\lambda=0\) và0,1; đo effective rank của z. Anti-collapse không đồng nghĩa forensic.
6. Chạy Gaussian trực tiếp trên24 mean residual statistics làm baseline đơn giản, để biết predictor/encoder có thêm giá trị không.

## 15. GSD trong thiết kế này

Bản sơ đồ hiện tại cố ý dùng CNN và không có CLIP/GSD. Đây là đúng prototype đã thảo luận, không phải một bản GSD reproduction.

Nếu thay context encoder bằng CLIP+GSD phải thiết kế lại crop size, input channels và tensor contract. Không chiếu CNN128 bằng basis CLIP1024; ngay cả adapter bằng chiều cũng chưa đảm bảo cùng hệ tọa độ. Phần GSD là thí nghiệm mở rộng sau khi conditional objective có bằng chứng hữu ích.

## 16. Nguồn và phạm vi đóng góp

- [Forensic Self-Descriptions, CVPR2025](https://arxiv.org/abs/2503.21003): cơ sở dùng forensic residuals và microstructure modeling.
- [VICReg, ICLR2022](https://arxiv.org/abs/2105.04906) và [official code](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py): invariance, variance floor, covariance regularization.
- [GSD v2](https://arxiv.org/html/2603.09242v2): semantic suppression bằng frozen CLIP anchor; không phải nguồn của kiến trúc CNN hoặc loss conditional trong tài liệu này.

Kiến trúc CNN, label24, conditioning5, sampling8 patch và cách ghép objective/GMM là đề xuất thiết kế trong cuộc trao đổi này. Chưa xác nhận novelty toàn diện, hiệu quả detection hoặc khả năng diễn giải vật lý.
