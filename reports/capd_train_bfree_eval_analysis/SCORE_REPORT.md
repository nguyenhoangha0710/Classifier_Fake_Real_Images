# CAPD train BFree: mean và median score dự đoán

Score chính là **prob_fake** trên thang0–1, không phải accuracy/recall/F1. Real/fake ở đây là **nhãn thật** của ảnh. Score càng gần1 càng nghiêng về fake; ảnh real tốt nên có score thấp.

Tính trực tiếp từ prediction CSV; kiểm tra SHA256 nguồn khớp lần audit trước. Thay ngưỡng chỉ thay nhãn dự đoán, **không làm mean/median score thay đổi**. Số trong bảng được làm tròn6 chữ số thập phân, JSON giữ độ chính xác số thực.

## 1. Tiny, full pooling

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| png_roundtrip | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| fake_jpeg96 | 0.009645 | 0.001841 | 0.791811 | 0.988185 | 0.400728 | 0.016074 |
| all_jpeg96 | 0.009494 | 0.001832 | 0.791811 | 0.988185 | 0.400652 | 0.016034 |
| all_jpeg90 | 0.012853 | 0.002261 | 0.791573 | 0.989088 | 0.402213 | 0.024500 |
| all_jpeg80 | 0.018366 | 0.002870 | 0.750864 | 0.982687 | 0.384615 | 0.026999 |
| resize_square256 | 0.004778 | 0.002167 | 0.651738 | 0.927046 | 0.328258 | 0.008051 |
| resize_square560 | 0.008058 | 0.001003 | 0.831059 | 0.995979 | 0.419559 | 0.012042 |
| blur_sigma1 | 0.014188 | 0.002238 | 0.787617 | 0.985676 | 0.400903 | 0.025731 |
| resize256_jpeg96 | 0.005296 | 0.002298 | 0.644525 | 0.917831 | 0.324910 | 0.007769 |

Mỗi case có3.500 real +3.500 fake; All là gộp7.000 score của case đó. Các case là biến đổi của cùng ảnh nguồn.

## 2. BFree, full pooling

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 0.006627 | 0.001360 | 0.897182 | 0.991964 | 0.451905 | 0.049193 |
| SD2.1_selfconditioned_origBG | 0.006627 | 0.001360 | 0.485449 | 0.461868 | 0.246038 | 0.003421 |
| SD2.1_inpainted_samecat | 0.006627 | 0.001360 | 0.995738 | 0.999020 | 0.501183 | 0.690001 |
| SD2.1_inpainted_samecat_origBG | 0.006627 | 0.001360 | 0.995589 | 0.998818 | 0.501108 | 0.712675 |
| SD2.1_inpainted_diffcat | 0.006627 | 0.001360 | 0.996256 | 0.999048 | 0.501442 | 0.724734 |
| SD2.1_inpainted_diffcat_origBG | 0.006627 | 0.001360 | 0.991912 | 0.998800 | 0.499269 | 0.422601 |

Mỗi nhánh so10.304 real +10.304 fake. Score real full pooling là cùng một tập được dùng lại cho6 nhánh.

## 3. Score gộp theo dataset

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| tiny_10_case_views | 0.010197 | 0.001959 | 0.763844 | 0.984816 | 0.387021 | 0.014624 |
| bfree_unique_images | 0.006627 | 0.001360 | 0.893688 | 0.998426 | 0.766965 | 0.997782 |
| bfree_six_balanced_pairs | 0.006627 | 0.001360 | 0.893688 | 0.998426 | 0.450158 | 0.011137 |

- tiny_10_case_views:70.000 score,35.000 real/35.000 fake; cùng7.000 ảnh nguồn qua10 biến đổi, không phải70.000 ảnh độc lập.
- bfree_unique_images:72.128 score,10.304 real/61.824 fake; real chỉ đếm một lần. All mean/median chịu ảnh hưởng tỷ lệ real:fake1:6.
- bfree_six_balanced_pairs:123.648 score theo6 cặp đánh giá cân bằng; score real được đếm lại6 lần,61.824 real/61.824 fake. Không xem là123.648 ảnh độc lập.
- Median tính trực tiếp trên tất cả score thuộc nhóm, không lấy trung bình hoặc trung vị của các median case.

## 4. Các pooling và raw logit z

Full là kết quả chính. Window/inner/nopad là các cách gộp score khác nhau; không trộn các pooling. Logit z là score trước sigmoid, có thể âm/dương và không cùng thang với prob_fake.

### Tiny — lse_full — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| png_roundtrip | 0.009645 | 0.001841 | 0.798723 | 0.988672 | 0.404184 | 0.016570 |
| fake_jpeg96 | 0.009645 | 0.001841 | 0.791811 | 0.988185 | 0.400728 | 0.016074 |
| all_jpeg96 | 0.009494 | 0.001832 | 0.791811 | 0.988185 | 0.400652 | 0.016034 |
| all_jpeg90 | 0.012853 | 0.002261 | 0.791573 | 0.989088 | 0.402213 | 0.024500 |
| all_jpeg80 | 0.018366 | 0.002870 | 0.750864 | 0.982687 | 0.384615 | 0.026999 |
| resize_square256 | 0.004778 | 0.002167 | 0.651738 | 0.927046 | 0.328258 | 0.008051 |
| resize_square560 | 0.008058 | 0.001003 | 0.831059 | 0.995979 | 0.419559 | 0.012042 |
| blur_sigma1 | 0.014188 | 0.002238 | 0.787617 | 0.985676 | 0.400903 | 0.025731 |
| resize256_jpeg96 | 0.005296 | 0.002298 | 0.644525 | 0.917831 | 0.324910 | 0.007769 |

### Tiny — lse_full — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | -6.057362 | -6.295667 | 3.380176 | 4.469061 | -1.338593 | -4.083441 |
| png_roundtrip | -6.057362 | -6.295667 | 3.380176 | 4.469061 | -1.338593 | -4.083441 |
| fake_jpeg96 | -6.057362 | -6.295667 | 3.332420 | 4.426511 | -1.362471 | -4.114326 |
| all_jpeg96 | -6.061557 | -6.300387 | 3.332420 | 4.426511 | -1.364569 | -4.116884 |
| all_jpeg90 | -5.809556 | -6.089606 | 3.396337 | 4.506964 | -1.206610 | -3.684292 |
| all_jpeg80 | -5.507718 | -5.850417 | 2.930908 | 4.038849 | -1.288405 | -3.584575 |
| resize_square256 | -6.067246 | -6.132283 | 1.686898 | 2.542177 | -2.190174 | -4.813892 |
| resize_square560 | -6.714595 | -6.903882 | 3.956119 | 5.512230 | -1.379238 | -4.407267 |
| blur_sigma1 | -5.792295 | -6.099795 | 3.221127 | 4.231362 | -1.285584 | -3.633989 |
| resize256_jpeg96 | -6.011695 | -6.073431 | 1.640492 | 2.413239 | -2.185602 | -4.849791 |

### Tiny — lse_inner — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | 0.000435 | 0.000262 | 0.013087 | 0.009886 | 0.006761 | 0.000985 |
| png_roundtrip | 0.000435 | 0.000262 | 0.013087 | 0.009886 | 0.006761 | 0.000985 |
| fake_jpeg96 | 0.000435 | 0.000262 | 0.013244 | 0.009978 | 0.006839 | 0.000982 |
| all_jpeg96 | 0.000441 | 0.000261 | 0.013244 | 0.009978 | 0.006842 | 0.000976 |
| all_jpeg90 | 0.000493 | 0.000294 | 0.013597 | 0.010131 | 0.007045 | 0.001121 |
| all_jpeg80 | 0.000591 | 0.000336 | 0.012654 | 0.009370 | 0.006623 | 0.001231 |
| resize_square256 | 0.000512 | 0.000377 | 0.011442 | 0.008026 | 0.005977 | 0.000872 |
| resize_square560 | 0.000309 | 0.000200 | 0.007909 | 0.007536 | 0.004109 | 0.000636 |
| blur_sigma1 | 0.000516 | 0.000304 | 0.013290 | 0.009860 | 0.006903 | 0.001172 |
| resize256_jpeg96 | 0.000535 | 0.000390 | 0.011795 | 0.008305 | 0.006165 | 0.000894 |

### Tiny — lse_inner — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | -8.215947 | -8.248039 | -4.807681 | -4.606717 | -6.511814 | -6.921515 |
| png_roundtrip | -8.215947 | -8.248039 | -4.807681 | -4.606717 | -6.511814 | -6.921515 |
| fake_jpeg96 | -8.215947 | -8.248039 | -4.803970 | -4.597349 | -6.509958 | -6.925210 |
| all_jpeg96 | -8.217854 | -8.249531 | -4.803970 | -4.597349 | -6.510912 | -6.931454 |
| all_jpeg90 | -8.095132 | -8.131236 | -4.783231 | -4.581969 | -6.439181 | -6.792193 |
| all_jpeg80 | -7.949494 | -7.998724 | -4.882848 | -4.660876 | -6.416171 | -6.698846 |
| resize_square256 | -7.913279 | -7.882648 | -5.171516 | -4.816973 | -6.542398 | -7.044398 |
| resize_square560 | -8.490064 | -8.514786 | -5.194039 | -4.880439 | -6.842052 | -7.359472 |
| blur_sigma1 | -8.058089 | -8.097531 | -4.806341 | -4.609344 | -6.432215 | -6.748097 |
| resize256_jpeg96 | -7.871892 | -7.848653 | -5.156467 | -4.782557 | -6.514180 | -7.018675 |

### Tiny — lse_nopad — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | 0.003267 | 0.000295 | 0.798723 | 0.988672 | 0.400995 | 0.004810 |
| png_roundtrip | 0.003267 | 0.000295 | 0.798723 | 0.988672 | 0.400995 | 0.004810 |
| fake_jpeg96 | 0.003267 | 0.000295 | 0.791811 | 0.988185 | 0.397539 | 0.004871 |
| all_jpeg96 | 0.003289 | 0.000297 | 0.791811 | 0.988185 | 0.397550 | 0.004927 |
| all_jpeg90 | 0.003079 | 0.000336 | 0.791573 | 0.989088 | 0.397326 | 0.005508 |
| all_jpeg80 | 0.003421 | 0.000387 | 0.750864 | 0.982687 | 0.377142 | 0.005534 |
| resize_square256 | 0.004778 | 0.002167 | 0.651738 | 0.927046 | 0.328258 | 0.008051 |
| resize_square560 | 0.008058 | 0.001003 | 0.831059 | 0.995979 | 0.419559 | 0.012042 |
| blur_sigma1 | 0.003511 | 0.000345 | 0.787617 | 0.985676 | 0.395564 | 0.005394 |
| resize256_jpeg96 | 0.005296 | 0.002298 | 0.644525 | 0.917831 | 0.324910 | 0.007769 |

### Tiny — lse_nopad — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| native | -7.986928 | -8.126610 | 3.380176 | 4.469061 | -2.303376 | -5.332320 |
| png_roundtrip | -7.986928 | -8.126610 | 3.380176 | 4.469061 | -2.303376 | -5.332320 |
| fake_jpeg96 | -7.986928 | -8.126610 | 3.332420 | 4.426511 | -2.327254 | -5.319482 |
| all_jpeg96 | -7.988331 | -8.121309 | 3.332420 | 4.426511 | -2.327956 | -5.308024 |
| all_jpeg90 | -7.865546 | -7.997340 | 3.396337 | 4.506964 | -2.234604 | -5.196097 |
| all_jpeg80 | -7.715934 | -7.855745 | 2.930908 | 4.038849 | -2.392513 | -5.191342 |
| resize_square256 | -6.067246 | -6.132283 | 1.686898 | 2.542177 | -2.190174 | -4.813892 |
| resize_square560 | -6.714595 | -6.903882 | 3.956119 | 5.512230 | -1.379238 | -4.407267 |
| blur_sigma1 | -7.825931 | -7.972302 | 3.221127 | 4.231362 | -2.302402 | -5.217052 |
| resize256_jpeg96 | -6.011695 | -6.073431 | 1.640492 | 2.413239 | -2.185602 | -4.849791 |

### BFree — lse_full — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 0.006627 | 0.001360 | 0.897182 | 0.991964 | 0.451905 | 0.049193 |
| SD2.1_selfconditioned_origBG | 0.006627 | 0.001360 | 0.485449 | 0.461868 | 0.246038 | 0.003421 |
| SD2.1_inpainted_samecat | 0.006627 | 0.001360 | 0.995738 | 0.999020 | 0.501183 | 0.690001 |
| SD2.1_inpainted_samecat_origBG | 0.006627 | 0.001360 | 0.995589 | 0.998818 | 0.501108 | 0.712675 |
| SD2.1_inpainted_diffcat | 0.006627 | 0.001360 | 0.996256 | 0.999048 | 0.501442 | 0.724734 |
| SD2.1_inpainted_diffcat_origBG | 0.006627 | 0.001360 | 0.991912 | 0.998800 | 0.499269 | 0.422601 |

### BFree — lse_full — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | -6.450558 | -6.598647 | 4.158162 | 4.815721 | -1.146198 | -2.961571 |
| SD2.1_selfconditioned_origBG | -6.450558 | -6.598647 | -0.335515 | -0.152823 | -3.393036 | -5.674417 |
| SD2.1_inpainted_samecat | -6.450558 | -6.598647 | 6.640311 | 6.927247 | 0.094877 | 0.801528 |
| SD2.1_inpainted_samecat_origBG | -6.450558 | -6.598647 | 6.517617 | 6.739249 | 0.033529 | 0.908443 |
| SD2.1_inpainted_diffcat | -6.450558 | -6.598647 | 6.677140 | 6.955515 | 0.113291 | 0.968098 |
| SD2.1_inpainted_diffcat_origBG | -6.450558 | -6.598647 | 6.408924 | 6.724306 | -0.020817 | -0.312106 |

### BFree — lse_window — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 0.008051 | 0.001810 | 0.908991 | 0.993953 | 0.458521 | 0.064163 |
| SD2.1_selfconditioned_origBG | 0.008051 | 0.001810 | 0.506943 | 0.535721 | 0.257497 | 0.004634 |
| SD2.1_inpainted_samecat | 0.008051 | 0.001810 | 0.996574 | 0.999258 | 0.502312 | 0.739095 |
| SD2.1_inpainted_samecat_origBG | 0.008051 | 0.001810 | 0.996464 | 0.999121 | 0.502258 | 0.764482 |
| SD2.1_inpainted_diffcat | 0.008051 | 0.001810 | 0.997013 | 0.999275 | 0.502532 | 0.784497 |
| SD2.1_inpainted_diffcat_origBG | 0.008051 | 0.001810 | 0.993176 | 0.999099 | 0.500614 | 0.494235 |

### BFree — lse_window — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | -6.162368 | -6.312660 | 4.444185 | 5.102147 | -0.859091 | -2.680012 |
| SD2.1_selfconditioned_origBG | -6.162368 | -6.312660 | -0.037813 | 0.143126 | -3.100090 | -5.369755 |
| SD2.1_inpainted_samecat | -6.162368 | -6.312660 | 6.921913 | 7.205081 | 0.379773 | 1.041896 |
| SD2.1_inpainted_samecat_origBG | -6.162368 | -6.312660 | 6.813557 | 7.035777 | 0.325595 | 1.177505 |
| SD2.1_inpainted_diffcat | -6.162368 | -6.312660 | 6.955163 | 7.228515 | 0.396398 | 1.292092 |
| SD2.1_inpainted_diffcat_origBG | -6.162368 | -6.312660 | 6.695789 | 7.010765 | 0.266711 | -0.023061 |

### BFree — lse_inner — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 0.000367 | 0.000244 | 0.013518 | 0.009261 | 0.006942 | 0.001197 |
| SD2.1_selfconditioned_origBG | 0.000367 | 0.000244 | 0.005236 | 0.002733 | 0.002801 | 0.000463 |
| SD2.1_inpainted_samecat | 0.000367 | 0.000244 | 0.013979 | 0.011180 | 0.007173 | 0.003546 |
| SD2.1_inpainted_samecat_origBG | 0.000367 | 0.000244 | 0.012928 | 0.010747 | 0.006647 | 0.003312 |
| SD2.1_inpainted_diffcat | 0.000367 | 0.000244 | 0.013972 | 0.011134 | 0.007169 | 0.003756 |
| SD2.1_inpainted_diffcat_origBG | 0.000367 | 0.000244 | 0.012980 | 0.010683 | 0.006673 | 0.003013 |

### BFree — lse_inner — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | -8.315627 | -8.318286 | -4.718804 | -4.672603 | -6.517215 | -6.726980 |
| SD2.1_selfconditioned_origBG | -8.315627 | -8.318286 | -6.120983 | -5.899508 | -7.218305 | -7.678087 |
| SD2.1_inpainted_samecat | -8.315627 | -8.318286 | -4.426405 | -4.482374 | -6.371016 | -5.638454 |
| SD2.1_inpainted_samecat_origBG | -8.315627 | -8.318286 | -4.491237 | -4.522345 | -6.403432 | -5.706839 |
| SD2.1_inpainted_diffcat | -8.315627 | -8.318286 | -4.426589 | -4.486588 | -6.371108 | -5.580583 |
| SD2.1_inpainted_diffcat_origBG | -8.315627 | -8.318286 | -4.492413 | -4.528397 | -6.404020 | -5.801953 |

### BFree — lse_nopad — prob_fake

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 0.006627 | 0.001360 | 0.897182 | 0.991964 | 0.451905 | 0.049193 |
| SD2.1_selfconditioned_origBG | 0.006627 | 0.001360 | 0.485449 | 0.461868 | 0.246038 | 0.003421 |
| SD2.1_inpainted_samecat | 0.006627 | 0.001360 | 0.995738 | 0.999020 | 0.501183 | 0.690001 |
| SD2.1_inpainted_samecat_origBG | 0.006627 | 0.001360 | 0.995589 | 0.998818 | 0.501108 | 0.712675 |
| SD2.1_inpainted_diffcat | 0.006627 | 0.001360 | 0.996256 | 0.999048 | 0.501442 | 0.724734 |
| SD2.1_inpainted_diffcat_origBG | 0.006627 | 0.001360 | 0.991912 | 0.998800 | 0.499269 | 0.422601 |

### BFree — lse_nopad — logit

| Case/nhánh | Real mean | Real median | Fake mean | Fake median | All mean | All median |
| --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | -6.450558 | -6.598647 | 4.158162 | 4.815721 | -1.146198 | -2.961571 |
| SD2.1_selfconditioned_origBG | -6.450558 | -6.598647 | -0.335515 | -0.152823 | -3.393036 | -5.674417 |
| SD2.1_inpainted_samecat | -6.450558 | -6.598647 | 6.640311 | 6.927247 | 0.094877 | 0.801528 |
| SD2.1_inpainted_samecat_origBG | -6.450558 | -6.598647 | 6.517617 | 6.739249 | 0.033529 | 0.908443 |
| SD2.1_inpainted_diffcat | -6.450558 | -6.598647 | 6.677140 | 6.955515 | 0.113291 | 0.968098 |
| SD2.1_inpainted_diffcat_origBG | -6.450558 | -6.598647 | 6.408924 | 6.724306 | -0.020817 | -0.312106 |

## File chạy

- [score_statistics.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/score_statistics.py).
- [score_statistics.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/score_statistics.json) chứa count, mean, median, p05/p95 cho score và logit, từng pooling/nhãn.
