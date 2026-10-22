# Test checkpoint All7 crop560 trên Kaggle

File để import: **test_gsd_all7_crop560_variants_kaggle.ipynb**.
Notebook tự chứa runtime; không cần upload các file Python, không train lại.

## Input cần gắn

- `yangsangtai/tiny-genimage`.
- `hadinho0710/checkpoint-gsd-sdv5-train`, version 2: thư mục `gsd_raw_all7_crop560_v1`.
- GPU, Internet. Internet dùng tải CLIP pretrained gốc cho frozen anchor; không dùng model fine-tune làm anchor.

Evaluator tự tìm `seed42/clip_gsd/best.pt` và kiểm tra signature/epoch đã chọn.
Checkpoint cũ SD15 hoặc native không crop560 sẽ bị từ chối.
Giữ nguyên các file `config.json`, `seed42/test_manifest.csv`, `seed42/split_summary.json`,
`seed42/clip_gsd/selected_checkpoint.json` đi kèm checkpoint. Thêm predictions cũ để đối chiếu native.
Nếu auto-discovery báo nhiều đường dẫn, điền `checkpoint_path` và `dataset_root` trong config.
`dataset_root` là thư mục chứa các thư mục generator, không phải thư mục `val` hay `ai`.
Nếu offline: gắn bản **pretrained gốc** Hugging Face của `openai/clip-vit-large-patch14` và điền `pretrained_model_path`.

## Các case

| Tên | Biến đổi trên RGB gốc, trước crop |
|---|---|
| `native` | Không thêm biến đổi; vẫn crop/pad như train |
| `png_roundtrip` | PNG encode/decode lossless, đối chứng |
| `fake_jpeg96` | Chỉ fake JPEG Q96, real giữ nguyên |
| `all_jpeg96`, `all_jpeg90`, `all_jpeg80` | Cả hai lớp JPEG cùng Q |
| `resize_square256`, `resize_square560` | BICUBIC hình vuông, có upscale ảnh nhỏ |
| `blur_sigma1` | Pillow GaussianBlur radius 1 pixel |
| `resize256_jpeg96` | Resize vuông 256 rồi JPEG Q96 |

10 case trên là mặc định. Tùy chọn thêm: `all_jpeg1`…`all_jpeg100`,
`fake_jpeg1`…`fake_jpeg100`, `resize_square224/336/512`, `blur_sigma2`,
`resize_long560` (giữ tỷ lệ, chỉ thu nhỏ cạnh dài >560, không upscale).
JPEG subsampling=0/4:4:4, optimize=False, progressive=False; real vốn JPEG sẽ bị nén lại trong case all-JPEG.
PNG roundtrip không khôi phục được ảnh trước nén JPEG: chỉ bảo toàn RGB hiện đang giải mã.

Sau biến đổi: center crop riêng từng chiều tối đa 560, edge-pad phải/dưới 0–13 pixel tới bội 14,
normalize CLIP. Không resize/crop tự động bổ sung để tránh OOM.
Ví dụ resize256 → pad266 → 362 token; resize560 → 1601 token.
Square resize có thể làm méo tỷ lệ, thay đổi token và dấu vết nội suy: không diễn giải nó chỉ là phép thử nén.
Blur thực hiện ở độ phân giải gốc; radius 1 không có cùng mức tương đối trên mọi kích thước ảnh.

## Cố định phép so sánh

- Đúng 7.000 test IDs và thứ tự manifest đã lưu, không chia split mới. Đọc đúng test files;
  chỉ kiểm tra đường dẫn/kích thước file trước chạy, không scan/decode/hash lại train dataset.
- Việc kiểm tra kích thước file và kích thước ảnh **không xác minh nội dung bằng hash**;
  cần gắn đúng bản Tiny-GenImage đã train. Hash gốc giữ trong manifest để truy vết.
- Logical batch=8 như checkpoint; microbatch=1 và anchor_micro_batch=1. Không gom batch theo size hoặc nhãn.
- Frozen anchor pretrained gốc; detector load strict checkpoint; model.eval(), no_grad(), không optimizer.
- Mỗi case tái tính basis cả batch. Real score ở fake-only JPEG có thể đổi do fake cùng batch thay đổi.
  Không tái sử dụng real scores native để ghép với fake scores mới.
- Threshold lấy nguyên từ config train (0.5), không tune trên test.
- `native_reference_check.json`: đối chiếu xác suất/nhãn/batch với native đã lưu.
  Sai khác tối đa >0.001 hoặc có đổi quyết định → `review_required`.
  Đây là cảnh báo để điều tra, evaluator vẫn lưu stress tests; không tự sửa mô hình.
- Test All7 gồm generator đã thấy khi train; không phải cross-generator hay CommFor/B-Free evaluation.
- Fake-only case phụ thuộc nhãn, chỉ dùng chẩn đoán; không dùng làm preprocessing triển khai.

## Chạy và tiếp tục

Run All để full test. Mỗi case 875 logical batches. Mặc định 10 case =70.000 image forwards qua
detector, cộng anchor. Không có training/backprop; vẫn cần thời gian GPU đáng kể.
Muốn smoke: bật config mẫu `max_batches=2` và dùng output riêng; smoke không đại diện 7.000 ảnh.
Smoke chỉ chọn prefix các batch nguyên vẹn, không xáo trộn/tạo batch mới.

Log batch đầu và mỗi 10 batch; lưu progress mỗi 25 batch (200 ảnh) và cuối case.
CSV shard ghi tạm rồi atomic rename. Chạy lại cùng output/config để tiếp tục từ batch đã lưu;
batch chưa lưu sẽ chạy lại. Session Kaggle mới không tự giữ `/kaggle/working`:
lưu toàn bộ output thành dataset/download, rồi chép lại vào thư mục ghi được trong session mới.
Không đặt output trực tiếp vào `/kaggle/input` (read-only).
Fingerprint chặn trộn kết quả khác checkpoint, manifest, crop, precision, microbatch, pretrained-model
reference hoặc phiên bản runtime/GPU. Đổi các yếu tố này cần output mới.
Case mới có thể thêm vào danh sách và dùng lại tiến độ các case cũ với cùng giao thức.
Chỉ bật một GPU cho inference; T4x2 không tự làm nhanh gấp đôi vì batch basis không được phân tán.
OOM sẽ dừng, không lặng lẽ bỏ ảnh/giảm độ phân giải. Không hứa VRAM trước khi smoke trên GPU thực.

## Output

Trong `/kaggle/working/gsd_all7_crop560_variants/`:

- `metrics.csv`: pooled và từng generator, N, Accuracy, Balanced Accuracy, real/fake recall,
  fake precision/F1, ROC-AUC, AP, TN/FP/FN/TP.
- `macro_all.csv`: trung bình đều bảy generator, không phải macro unseen.
- `delta_from_native.csv`: thay đổi so với native; đơn vị 0–1, nhân 100 thành điểm phần trăm.
- `predictions_<case>.csv`, `predictions.csv`: score từng ảnh và batch/token count để kiểm tra.
- `paired_<case>.csv`, `paired_summary.csv`: so khớp cùng ID, độ dịch score, flip rate,
  số đúng→sai và sai→đúng theo generator/nhãn.
- `provenance.json`, `evaluation_config.json`, `native_reference_check.json`, `completion.json`.
- `progress/`: các shard cho resume. `variant_comparison.png`: biểu đồ tổng hợp.

Cell cuối gói kết quả/progress thành ZIP; không chứa checkpoint hay ảnh đầu vào.
Không so sánh các con số test JPEG đã có nếu khác cohort/checkpoint/subsampling/resize/thứ tự biến đổi.

## Kiểm thử local

```powershell
python -m unittest discover -s baselines/clip_gsd_tiny_kaggle -p "test_*.py"
python baselines/clip_gsd_tiny_kaggle/build_evaluation_notebook.py
```

Tests dùng CLIP ngẫu nhiên rất nhỏ trên CPU, xác minh preprocessing, native prediction tương đương runtime train,
batch membership, resume sau gián đoạn, reference check và metric; không thay thế chạy checkpoint thật trên Kaggle GPU.
