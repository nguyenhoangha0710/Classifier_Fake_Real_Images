# So sánh mean/median score CAPD, GSD và baseline

**Score trong bảng chính là prob_fake (0–1), không phải accuracy/recall. Real/fake là nhãn thật. CAPD dùng full pooling.**

Mỗi số mean/median tính trực tiếp trên prediction thuộc nhóm. Không lấy trung bình các median case; không thay ngưỡng để thay đổi score. Các model có calibration và preprocessing khác nhau, nên score tự tin hơn không tự chứng minh model tốt hơn. Không convert NLL sang probability bằng sigmoid tùy ý.

## 1. Tiny native: cùng 7.000 ảnh và nhãn

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| CAPD train Tiny | 0.002226 | 0.000001 | 0.999262 | 0.999991 | 0.500744 | 0.664915 |
| GSD train BFree | 0.022424 | 0.003188 | 0.621838 | 0.735618 | 0.322131 | 0.045058 |
| GSD train Tiny | 0.005235 | 0.000658 | 0.998041 | 0.999678 | 0.501638 | 0.776606 |
| CLIP ViT-B/32 probe | 0.293889 | 0.220217 | 0.839656 | 0.920861 | 0.566772 | 0.621139 |
| NPR-ResNet18 | 0.076060 | 0.001102 | 0.930720 | 0.987888 | 0.503390 | 0.541636 |
| AIDE original full | 0.136414 | 0.088939 | 0.770709 | 0.833426 | 0.453562 | 0.389736 |
| AIDE forensic-only ResNet50 | 0.085026 | 0.040408 | 0.805617 | 0.913368 | 0.445322 | 0.265392 |

Mỗi model có3.500 real +3.500 fake. Các baseline được join bằng relative_path sang sample_id của GSD/CAPD; ID và nhãn khớp đủ7.000. CLIP ở đây là OpenCLIP ViT-B/32 probe; không phải CLIP ViT-L/14 probe trong notebook BFree NhatNguyen. AIDE full và AIDE forensic-only ResNet50 là hai model khác nhau; bản forensic-only chỉ có native ở lượt so sánh này.

## 2. Tiny: cùng9 case có đủ CAPD/GSD

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.010258 | 0.001977 | 0.759969 | 0.984298 | 0.385114 | 0.014410 |
| CAPD train Tiny | 0.036539 | 0.000003 | 0.996256 | 0.999990 | 0.516397 | 0.985743 |
| GSD train BFree | 0.033130 | 0.004783 | 0.558016 | 0.626667 | 0.295573 | 0.037750 |
| GSD train Tiny | 0.023718 | 0.001016 | 0.955469 | 0.999395 | 0.489594 | 0.231940 |

Gồm all_jpeg80, all_jpeg90, all_jpeg96, blur_sigma1, fake_jpeg96, native, resize256_jpeg96, resize_square256, resize_square560. Mỗi lớp31.500 score từ9 biến đổi của cùng3.500 ảnh nguồn; không phải31.500 ảnh độc lập. Không tính PNG roundtrip vì GSD không có case này.

## 3. Tiny: cùng3 case có đủ cả ba baseline

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.009595 | 0.001838 | 0.794115 | 0.988408 | 0.401855 | 0.016291 |
| CAPD train Tiny | 0.002236 | 0.000001 | 0.999171 | 0.999991 | 0.500703 | 0.664915 |
| GSD train BFree | 0.022775 | 0.003148 | 0.588660 | 0.675604 | 0.305718 | 0.035628 |
| GSD train Tiny | 0.005321 | 0.000645 | 0.994571 | 0.999552 | 0.499946 | 0.532273 |
| CLIP ViT-B/32 probe | 0.293158 | 0.219588 | 0.793041 | 0.877687 | 0.543100 | 0.565059 |
| NPR-ResNet18 | 0.075745 | 0.001090 | 0.439902 | 0.270408 | 0.257823 | 0.012735 |
| AIDE original full | 0.135583 | 0.088505 | 0.734587 | 0.792946 | 0.435085 | 0.360714 |

Chỉ gộp native, fake_jpeg96, all_jpeg96 cho mọi model:10.500 score/lớp. Không gộp9/10 case CAPD/GSD rồi so với3 case baseline. Tên phép biến đổi chung chưa phải xác minh pixel biến đổi byte-identical; pipeline tiền xử lý model khác nhau.

## 4. Tiny: bảng so sánh từng case

### all_jpeg80

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.018366 | 0.002870 | 0.750864 | 0.982687 | 0.384615 | 0.026999 |
| CAPD train Tiny | 0.001920 | 0.000001 | 0.995166 | 0.999984 | 0.498543 | 0.287962 |
| GSD train BFree | 0.051464 | 0.007541 | 0.570741 | 0.640316 | 0.311103 | 0.062904 |
| GSD train Tiny | 0.006902 | 0.000642 | 0.989760 | 0.999354 | 0.498331 | 0.380651 |

### all_jpeg90

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.012853 | 0.002261 | 0.791573 | 0.989088 | 0.402213 | 0.024500 |
| CAPD train Tiny | 0.001972 | 0.000001 | 0.998456 | 0.999989 | 0.500214 | 0.589332 |
| GSD train BFree | 0.032500 | 0.004486 | 0.577396 | 0.654598 | 0.304948 | 0.043164 |
| GSD train Tiny | 0.006125 | 0.000592 | 0.991705 | 0.999462 | 0.498915 | 0.420358 |

### all_jpeg96

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.009494 | 0.001832 | 0.791811 | 0.988185 | 0.400652 | 0.016034 |
| CAPD train Tiny | 0.002256 | 0.000001 | 0.999125 | 0.999990 | 0.500690 | 0.685063 |
| GSD train BFree | 0.023643 | 0.003124 | 0.572032 | 0.648278 | 0.297838 | 0.033290 |
| GSD train Tiny | 0.005534 | 0.000632 | 0.992836 | 0.999469 | 0.499185 | 0.429109 |
| CLIP ViT-B/32 probe | 0.291698 | 0.217795 | 0.769733 | 0.853547 | 0.530716 | 0.542015 |
| NPR-ResNet18 | 0.075114 | 0.001047 | 0.194492 | 0.004741 | 0.134803 | 0.002263 |
| AIDE original full | 0.133921 | 0.087525 | 0.716526 | 0.769384 | 0.425224 | 0.347426 |

### blur_sigma1

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.014188 | 0.002238 | 0.787617 | 0.985676 | 0.400903 | 0.025731 |
| CAPD train Tiny | 0.003014 | 0.000001 | 0.998848 | 0.999990 | 0.500931 | 0.852176 |
| GSD train BFree | 0.053555 | 0.009431 | 0.687360 | 0.822920 | 0.370457 | 0.130535 |
| GSD train Tiny | 0.007370 | 0.000908 | 0.992646 | 0.999422 | 0.500008 | 0.448220 |

### fake_jpeg96

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.009645 | 0.001841 | 0.791811 | 0.988185 | 0.400728 | 0.016074 |
| CAPD train Tiny | 0.002226 | 0.000001 | 0.999125 | 0.999990 | 0.500675 | 0.681670 |
| GSD train BFree | 0.022258 | 0.003124 | 0.572110 | 0.651320 | 0.297184 | 0.032238 |
| GSD train Tiny | 0.005194 | 0.000641 | 0.992836 | 0.999469 | 0.499015 | 0.386935 |
| CLIP ViT-B/32 probe | 0.293889 | 0.220217 | 0.769733 | 0.853547 | 0.531811 | 0.545286 |
| NPR-ResNet18 | 0.076060 | 0.001102 | 0.194492 | 0.004741 | 0.135276 | 0.002350 |
| AIDE original full | 0.136414 | 0.088939 | 0.716526 | 0.769384 | 0.426470 | 0.350366 |

### native

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| CAPD train Tiny | 0.002226 | 0.000001 | 0.999262 | 0.999991 | 0.500744 | 0.664915 |
| GSD train BFree | 0.022424 | 0.003188 | 0.621838 | 0.735618 | 0.322131 | 0.045058 |
| GSD train Tiny | 0.005235 | 0.000658 | 0.998041 | 0.999678 | 0.501638 | 0.776606 |
| CLIP ViT-B/32 probe | 0.293889 | 0.220217 | 0.839656 | 0.920861 | 0.566772 | 0.621139 |
| NPR-ResNet18 | 0.076060 | 0.001102 | 0.930720 | 0.987888 | 0.503390 | 0.541636 |
| AIDE original full | 0.136414 | 0.088939 | 0.770709 | 0.833426 | 0.453562 | 0.389736 |

### png_roundtrip

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| CAPD train Tiny | 0.002226 | 0.000001 | 0.999262 | 0.999991 | 0.500744 | 0.664915 |
| CLIP ViT-B/32 probe | 0.293889 | 0.220217 | 0.839656 | 0.920861 | 0.566772 | 0.621139 |
| NPR-ResNet18 | 0.076060 | 0.001102 | 0.930720 | 0.987888 | 0.503390 | 0.541636 |
| AIDE original full | 0.136414 | 0.088939 | 0.770709 | 0.833426 | 0.453562 | 0.389736 |

### resize256_jpeg96

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.005296 | 0.002298 | 0.644525 | 0.917831 | 0.324910 | 0.007769 |
| CAPD train Tiny | 0.124890 | 0.002868 | 0.991297 | 0.999995 | 0.558094 | 0.995494 |
| GSD train BFree | 0.034436 | 0.006056 | 0.399292 | 0.249973 | 0.216864 | 0.020964 |
| GSD train Tiny | 0.051803 | 0.005205 | 0.825600 | 0.998251 | 0.438702 | 0.113998 |

### resize_square256

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.004778 | 0.002167 | 0.651738 | 0.927046 | 0.328258 | 0.008051 |
| CAPD train Tiny | 0.179339 | 0.004581 | 0.995464 | 0.999995 | 0.587402 | 0.999345 |
| GSD train BFree | 0.044282 | 0.009486 | 0.435627 | 0.339458 | 0.239954 | 0.034165 |
| GSD train Tiny | 0.107257 | 0.010130 | 0.922607 | 0.999288 | 0.514932 | 0.563312 |

### resize_square560

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.008058 | 0.001003 | 0.831059 | 0.995979 | 0.419559 | 0.012042 |
| CAPD train Tiny | 0.011009 | 0.000326 | 0.989557 | 0.999960 | 0.500283 | 0.444810 |
| GSD train BFree | 0.013609 | 0.002323 | 0.585750 | 0.709975 | 0.299680 | 0.016324 |
| GSD train Tiny | 0.018045 | 0.001657 | 0.893189 | 0.998281 | 0.455617 | 0.083106 |

## 5. BFree: cùng5 nhánh đủ và10.304 ID

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006627 | 0.001360 | 0.874043 | 0.998254 | 0.440335 | 0.009210 |
| CAPD train Tiny | 0.014323 | 0.000654 | 0.662281 | 0.998398 | 0.338302 | 0.004136 |
| GSD train BFree | 0.009255 | 0.002332 | 0.784376 | 0.982048 | 0.396815 | 0.024516 |
| GSD train Tiny | 0.017929 | 0.003416 | 0.644572 | 0.961893 | 0.331250 | 0.015015 |

Mỗi model được tính trên5 cặp real/fake cân bằng:51.520 score/lớp. Real CAPD được dùng lại giữa các nhánh; GSD có score real theo batch context. Giữ nguyên các score context, không tự lấy trung bình theo ID trước khi tính median.

## 6. BFree: theo nhánh, cùng ID giữa4 model

### SD2.1_selfconditioned — 10304 ID/lớp

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006627 | 0.001360 | 0.897182 | 0.991964 | 0.451905 | 0.049193 |
| CAPD train Tiny | 0.014323 | 0.000654 | 0.841209 | 0.998180 | 0.427766 | 0.025051 |
| GSD train BFree | 0.010079 | 0.002521 | 0.611668 | 0.700152 | 0.310874 | 0.029479 |
| GSD train Tiny | 0.017896 | 0.003423 | 0.854500 | 0.984811 | 0.436198 | 0.105948 |

### SD2.1_selfconditioned_origBG — 10304 ID/lớp

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006627 | 0.001360 | 0.485449 | 0.461868 | 0.246038 | 0.003421 |
| CAPD train Tiny | 0.014323 | 0.000654 | 0.115632 | 0.001537 | 0.064977 | 0.000940 |
| GSD train BFree | 0.010675 | 0.002785 | 0.477924 | 0.432588 | 0.244300 | 0.011072 |
| GSD train Tiny | 0.019233 | 0.003773 | 0.149094 | 0.014282 | 0.084164 | 0.006488 |

### SD2.1_inpainted_samecat — 10304 ID/lớp

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006627 | 0.001360 | 0.995738 | 0.999020 | 0.501183 | 0.690001 |
| CAPD train Tiny | 0.014323 | 0.000654 | 0.976488 | 0.999922 | 0.495406 | 0.219428 |
| GSD train BFree | 0.008840 | 0.002166 | 0.915559 | 0.990197 | 0.462200 | 0.129322 |
| GSD train Tiny | 0.017392 | 0.003248 | 0.935405 | 0.997001 | 0.476399 | 0.183411 |

### SD2.1_inpainted_samecat_origBG — 10304 ID/lớp

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006627 | 0.001360 | 0.995589 | 0.998818 | 0.501108 | 0.712675 |
| CAPD train Tiny | 0.014323 | 0.000654 | 0.387658 | 0.029186 | 0.200990 | 0.001868 |
| GSD train BFree | 0.008445 | 0.002247 | 0.994004 | 0.999137 | 0.501225 | 0.448066 |
| GSD train Tiny | 0.017769 | 0.003504 | 0.320269 | 0.066146 | 0.169019 | 0.009139 |

### SD2.1_inpainted_diffcat — 10304 ID/lớp

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006627 | 0.001360 | 0.996256 | 0.999048 | 0.501442 | 0.724734 |
| CAPD train Tiny | 0.014323 | 0.000654 | 0.990419 | 0.999966 | 0.502371 | 0.514771 |
| GSD train BFree | 0.008233 | 0.002027 | 0.922724 | 0.989389 | 0.465479 | 0.140322 |
| GSD train Tiny | 0.017355 | 0.003191 | 0.963591 | 0.998439 | 0.490473 | 0.268696 |

### SD2.1_inpainted_diffcat_origBG — 7024 ID/lớp

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.006948 | 0.001372 | 0.991843 | 0.998811 | 0.499395 | 0.426123 |
| CAPD train Tiny | 0.014277 | 0.000654 | 0.621835 | 0.992806 | 0.318056 | 0.004009 |
| GSD train BFree | 0.008253 | 0.002083 | 0.969230 | 0.996919 | 0.488742 | 0.190273 |
| GSD train Tiny | 0.017247 | 0.003344 | 0.530060 | 0.595198 | 0.273654 | 0.013829 |

Nhánh diffcat_origBG chỉ so trên7.024 ID đã có của GSD train BFree, không gọi là nhánh hoàn tất và không đưa vào bảng gộp5 nhánh.

## 7. Baseline Twins: cùng cohort clean1.557 ID ×5 nhánh

### Probability của CAPD/GSD

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 0.007113 | 0.001386 | 0.869732 | 0.998197 | 0.438422 | 0.009399 |
| CAPD train Tiny | 0.016621 | 0.000632 | 0.663587 | 0.998518 | 0.340104 | 0.003917 |
| GSD train BFree | 0.009318 | 0.002497 | 0.780256 | 0.981311 | 0.394787 | 0.024238 |
| GSD train Tiny | 0.022639 | 0.003573 | 0.646162 | 0.966537 | 0.334401 | 0.016689 |

### NLL nguyên gốc của Twins/GMM — KHÁC THANG với bảng trên

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| Twins native GMM | -259.691694 | -369.306358 | -328.342235 | -384.606649 | -294.016964 | -377.686127 |
| Twins codec GMM | -163.788289 | -364.521497 | -339.815612 | -423.291076 | -251.801951 | -401.992305 |

NLL có thể âm; theo policy lưu trong pipeline, NLL cao hơn dự đoán fake. Không so khoảng cách mean NLL với khoảng cách probability. Các model trên được lọc về đúng cùng1.557 ID của baseline; đây không phải cohort10.304 ID.

## 8. Spectral-GMM trên Tiny: score NLL riêng

| Model | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| Spectral/MFM-GMM | -167.938631 | -257.753600 | 225.405655 | 55.726970 | 28.733512 | -73.753761 |

Đối chiếu đủ7.000 relative paths/labels như Tiny native, nhưng pipeline Spectral nguồn có áp dụng JPEG (jpeg_applied) và fixed-grid tile. Đây không phải probability hay đúng pixel pipeline native của CAPD/GSD; giữ bảng NLL riêng, không gộp cùng các phép biến đổi Tiny ở mục2/3.

## 9. Baseline chưa đủ prediction để báo mean/median BFree

- CLIP ViT-L/14 probe và AIDE full trong notebook NhatNguyen có metrics/mean score một số nhóm, nhưng chưa có prediction CSV hoặc median đầy đủ để tính thống kê BFree cùng cohort ở đây. Không lấy score Tiny thay cho BFree hoặc suy median từ AUC/accuracy.
- NPR có prediction Tiny đã đối chiếu, chưa có prediction BFree được xác minh. Không điền số giả cho baseline còn thiếu.

## File local

- [compare_scores.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/compare_scores.py) — tải nguồn CAPD train Tiny ở bước chuẩn bị, script này chỉ đọc CSV local và lập bảng.
- [comparison.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/comparison.json) — full-precision statistics, counts, AUC tham khảo và SHA256 nguồn.
