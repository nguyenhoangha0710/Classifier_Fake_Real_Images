# Phân bố score GSD trên 10 case

**Mọi score trong báo cáo đều là P(fake), kể cả score của ảnh thật.** Không dùng calibration. 3.500 ảnh thật và 3.500 ảnh giả mỗi case. P(real) của cùng ảnh = 1 − P(fake). Hai trung vị lấy từ hai nhóm ảnh khác nhau không cần cộng bằng 1.

Nguồn: D:\LLM\LVLM\MLLM_detect_fake_image\HoangHa_Code\reports\gsd_all7_variants_analysis\source\predictions.csv

SHA-256: `674a43dc8e740cfa96e3c95732e512d737a6461af42548b1f0baa7e40469b621`

## Trung vị và lỗi ở ngưỡng 0,5

| Case | Trung vị P(fake) trên ảnh thật | Trung vị P(fake) trên ảnh giả | Ảnh thật bị báo giả (%) | Ảnh giả bị bỏ sót (%) |
|---|---:|---:|---:|---:|
| Native | 0.00003944 | 0.99994051 | 0.06 | 0.00 |
| PNG lossless | 0.00003944 | 0.99994051 | 0.06 | 0.00 |
| Chỉ fake JPEG96 | 0.00003883 | 0.00759934 | 0.06 | 70.14 |
| Real + fake JPEG96 | 0.00003868 | 0.00765105 | 0.06 | 70.11 |
| Real + fake JPEG90 | 0.00003952 | 0.00008699 | 0.00 | 98.31 |
| Real + fake JPEG80 | 0.00003929 | 0.00005956 | 0.00 | 99.86 |
| Resize256 | 0.00030179 | 0.99990237 | 8.94 | 5.74 |
| Resize560 | 0.00005225 | 0.99989688 | 0.80 | 2.20 |
| Blur σ=1 | 0.00088391 | 0.99993718 | 14.74 | 0.00 |
| Resize256 + JPEG96 | 0.00006254 | 0.00032663 | 0.34 | 84.43 |

## Khoảng phân bố, không phải khoảng tin cậy
Q25–Q75 chứa khoảng 50% ảnh ở giữa phân bố; Q05–Q95 chứa khoảng 90% ảnh ở giữa. Các vùng này không bao phủ toàn bộ ảnh hoặc toàn bộ ảnh bị phân loại sai.
| Case | Nhóm ảnh theo nhãn thật | Q05 | Q25 | Trung vị | Q75 | Q95 | Trung bình |
|---|---|---:|---:|---:|---:|---:|---:|
| Native | Ảnh thật | 1.84831514e-05 | 2.81886641e-05 | 3.94425879e-05 | 5.89783449e-05 | 0.000133942129 | 0.00066517625 |
| Native | Ảnh giả | 0.999751699 | 0.99990499 | 0.999940515 | 0.999959111 | 0.999973893 | 0.999561412 |
| PNG lossless | Ảnh thật | 1.84831514e-05 | 2.81886641e-05 | 3.94425879e-05 | 5.89783449e-05 | 0.000133942129 | 0.00066517625 |
| PNG lossless | Ảnh giả | 0.999751699 | 0.99990499 | 0.999940515 | 0.999959111 | 0.999973893 | 0.999561412 |
| Chỉ fake JPEG96 | Ảnh thật | 1.81291962e-05 | 2.75356924e-05 | 3.88311091e-05 | 5.80640226e-05 | 0.00012982171 | 0.000664657624 |
| Chỉ fake JPEG96 | Ảnh giả | 4.26308719e-05 | 0.000254023034 | 0.00759933516 | 0.825008616 | 0.999721378 | 0.304131227 |
| Real + fake JPEG96 | Ảnh thật | 1.81966036e-05 | 2.77516501e-05 | 3.86797292e-05 | 5.78376712e-05 | 0.000122434445 | 0.000480857546 |
| Real + fake JPEG96 | Ảnh giả | 4.198631e-05 | 0.000258779815 | 0.00765104545 | 0.823557347 | 0.999723232 | 0.304336201 |
| Real + fake JPEG90 | Ảnh thật | 1.89251477e-05 | 2.82989877e-05 | 3.95197712e-05 | 5.76121929e-05 | 0.00011637299 | 0.000132273127 |
| Real + fake JPEG90 | Ảnh giả | 2.45818852e-05 | 4.55755471e-05 | 8.69920004e-05 | 0.00024453253 | 0.0154257727 | 0.0194027856 |
| Real + fake JPEG80 | Ảnh thật | 1.91444943e-05 | 2.87446201e-05 | 3.92888178e-05 | 5.73875986e-05 | 0.000111479043 | 5.38674517e-05 |
| Real + fake JPEG80 | Ảnh giả | 2.25620515e-05 | 3.77469232e-05 | 5.95572001e-05 | 0.000112792972 | 0.000560769378 | 0.0019988427 |
| Resize256 | Ảnh thật | 3.8528935e-05 | 9.66676635e-05 | 0.000301794862 | 0.00471836934 | 0.931343955 | 0.0927682725 |
| Resize256 | Ảnh giả | 0.288268678 | 0.999375522 | 0.999902368 | 0.999951482 | 0.999971992 | 0.938035082 |
| Resize560 | Ảnh thật | 2.16131411e-05 | 3.50811206e-05 | 5.22522714e-05 | 8.71618904e-05 | 0.000387781118 | 0.00815153525 |
| Resize560 | Ảnh giả | 0.979204005 | 0.999747813 | 0.999896884 | 0.999941945 | 0.999966502 | 0.976200315 |
| Blur σ=1 | Ảnh thật | 4.36502698e-05 | 0.000143697514 | 0.000883908302 | 0.0474369945 | 0.989449918 | 0.15344538 |
| Blur σ=1 | Ảnh giả | 0.99976722 | 0.999902368 | 0.999937177 | 0.999956846 | 0.999972349 | 0.999898585 |
| Resize256 + JPEG96 | Ảnh thật | 2.42055066e-05 | 4.03779013e-05 | 6.25371322e-05 | 0.000109858282 | 0.000442487304 | 0.00421931808 |
| Resize256 + JPEG96 | Ảnh giả | 3.33441858e-05 | 8.12442813e-05 | 0.000326626454 | 0.0199568113 | 0.998931497 | 0.157509571 |

## Cách đọc biểu đồ
Màu xanh: ảnh thật theo nhãn gốc; màu cam: ảnh giả theo nhãn gốc. Hộp là Q25–Q75, vạch/điểm là trung vị, râu là Q05–Q95; không vẽ 10% ảnh ở hai đuôi. Trục ngang dùng thang logit để nhìn rõ score rất nhỏ và rất gần 1; nhãn trục vẫn là P(fake). Vạch đứng ở 0,5: bên phải bị model nhận là fake, bên trái bị nhận là real.
## Diễn giải
Native/PNG: ảnh thật tập trung gần 0, ảnh giả gần 1. PNG lossless giữ nguyên prediction. JPEG96: score ảnh giả trải rất rộng và dịch mạnh về 0; score ảnh thật gần như giữ nguyên. JPEG90/80: hai phân bố chồng lấn nhiều hơn. Resize256/blur: một phần score ảnh thật tăng mạnh, dù trung vị vẫn dưới 0,5; trung vị một mình không mô tả các đuôi gây báo nhầm. Resize256+JPEG96 làm score fake thấp và có thêm chồng lấn với real.
