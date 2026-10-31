# GSD augmented: tổng hợp9 case

Đã kiểm tra đủ9 case, mỗi case7.000 ảnh (3.500 real +3.500 fake), ngưỡng0,5.
Đây là9 biến thể của cùng một tập ảnh, không phải63.000 ảnh độc lập.

|Case|Accuracy / BA|Real recall|Fake recall|Fake F1|ROC-AUC|AP|FP|FN|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|native|99.8571%|99.7429%|99.9714%|99.8573%|99.9986%|99.9986%|9|1|
|fake_jpeg96|99.6714%|99.7429%|99.6000%|99.6712%|99.9932%|99.9931%|9|14|
|all_jpeg96|99.6286%|99.6571%|99.6000%|99.6285%|99.9918%|99.9917%|12|14|
|all_jpeg90|99.4857%|99.5714%|99.4000%|99.4853%|99.9906%|99.9907%|15|21|
|all_jpeg80|99.4714%|99.6000%|99.3429%|99.4707%|99.9843%|99.9846%|14|23|
|resize_square256|92.2000%|91.7143%|92.6857%|92.2377%|97.5754%|98.0290%|290|256|
|resize_square560|94.7571%|99.0286%|90.4857%|94.5232%|99.4563%|99.4872%|34|333|
|blur_sigma1|99.6714%|99.7143%|99.6286%|99.6713%|99.9947%|99.9947%|10|13|
|resize256_jpeg96|90.0857%|96.9429%|83.2286%|89.3558%|96.0380%|96.8907%|107|587|

## So sánh checkpoint raw-only trước

|Case|Raw-only accuracy|Augmented accuracy|Chênh lệch điểm %|
|---|---:|---:|---:|
|native|99.9714%|99.8571%|-0.1143|
|fake_jpeg96|64.9000%|99.6714%|+34.7714|
|all_jpeg96|64.9143%|99.6286%|+34.7143|
|all_jpeg90|50.8429%|99.4857%|+48.6429|
|all_jpeg80|50.0714%|99.4714%|+49.4000|
|resize_square256|92.6571%|92.2000%|-0.4571|
|resize_square560|98.5000%|94.7571%|-3.7429|
|blur_sigma1|92.6286%|99.6714%|+7.0429|
|resize256_jpeg96|57.6143%|90.0857%|+32.4714|

## Giới hạn diễn giải

JPEG và blur được cải thiện rõ trên cohort này. Resize vẫn là điểm yếu, đặc biệt Midjourney.
Checkpoint trained all7; đây không phải unseen-generator hoặc cross-dataset evaluation.
Không tối ưu ngưỡng hay fit calibration trên test. Phân tích không sửa các CSV nguồn.
