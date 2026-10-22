"""Build a new standalone Kaggle notebook; never overwrite raw-only experiments."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = 'train_gsd_all7_crop560_blur_jpeg_kaggle.ipynb'


def build_notebook():
    source = '\n\n'.join((HERE / name).read_text(encoding='utf-8') for name in
                           ('runtime.py', 'runtime_raw_all7.py', 'manifest_reuse.py', 'runtime_aug_all7.py'))
    ast.parse(source)

    def md(text):
        return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))

    def code(text):
        return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                    source=text.splitlines(keepends=True))

    cells = [md('''# GSD — All7 crop560 + Gaussian blur/JPEG, fixed validation

Gắn **yangsangtai/tiny-genimage** VÀ dataset/output chứa manifest của lần train All7 trước.
Bật GPU và Internet. Notebook tự chứa toàn bộ code.
Đây là thí nghiệm MỚI; không thay thế notebook/checkpoint raw-only.

## Input và chế độ bỏ scan
Mặc định `data_mode='reuse_manifests'`: không decode/hash toàn bộ ảnh, không chia lại split.
Input thứ hai cần thư mục sau (không cần best.pt/last.pt):

```text
<manifest_root>/
  config.json                    # khuyến nghị, nếu có
  seed42/
    train_manifest.csv           # bắt buộc
    validation_manifest.csv      # bắt buộc
    test_manifest.csv            # bắt buộc
    split_summary.json           # khuyến nghị để đối chiếu thứ tự/count
```

Có thể dùng `hadinho0710/checkpoint-gsd-sdv5-train` phiên bản chứa
`gsd_raw_all7_crop560_v1/seed42/` hoặc upload gói manifest-only đi kèm.
Để `manifest_root=None` và `dataset_root=None` cho tự tìm nếu chỉ có một ứng viên;
nếu có nhiều, điền chính xác đường dẫn thư mục tương ứng dưới /kaggle/input.
Đường dẫn tuyệt đối cũ trong CSV được bỏ qua; code map lại bằng relative_path.
Chỉ kiểm tra metadata/ảnh trùng theo hash ĐÃ LƯU và file tồn tại/đúng byte length.
Tin tưởng audit trước và đúng phiên bản dataset: không phát hiện thay đổi nội dung cùng kích thước file,
không phát hiện ảnh mới ngoài manifest. Thiếu/sai input thì dừng, KHÔNG tự fallback sang full scan.
Muốn audit lại phải chủ động đặt `data_mode='full_scan'`, dùng output_root mới.

## Train
Giữ All7, center crop tối đa 560 mỗi chiều, edge-pad14, token động, logical batch8,
microbatch1, CLIP/GSD và optimizer cũ. Không resize, flip, consistency loss hay model phụ.
JPEG và blur thực hiện trên RGB gốc TRƯỚC crop/padding/normalization:

- Blur: xác suất 0.5, sigma lấy đều trong [0, 2].
- JPEG: xác suất 0.5 độc lập, quality nguyên lấy đều 75–100, subsampling0 (4:4:4).
- Nếu cùng được chọn: blur → JPEG. Trung bình bốn nhóm native/blur/JPEG/cả hai: 25% mỗi nhóm.
- Cùng phân phối cho real/fake và mọi generator; lấy lại tham số mỗi epoch.
- Anchor đóng băng và detector nhận CÙNG tensor đã xử lý; không tăng số forward train.

Paper GSD có JPEG/blur nhưng không công bố đầy đủ recipe. Các xác suất/khoảng trên là lựa
chọn triển khai, KHÔNG phải tham số chính thức. All7/crop560/batch8 cũng khác protocol paper SD14.
Nguồn: https://arxiv.org/html/2603.09242v2#S5.SS1

## Validation và early stopping
Chỉ dùng validation tách từ TRAIN folders theo pixel-hash groups như cũ.
Official test không dùng chọn checkpoint. Mỗi epoch đánh giá hai view cùng ID/order/batch:

1. Native: không blur/JPEG thêm.
2. Processed: cùng phân phối augmentation train, nhưng tham số theo ảnh được cố định bằng seed.

`val_loss = (val_native_loss + val_processed_loss) / 2`, tính CE trực tiếp từ logits.
Early stopping patience3/min_delta0.001, tối đa50 epoch; khôi phục best.pt theo loss này.
`val_native` và `val_processed` ghi riêng accuracy/AUC/AP/recall/confusion matrix tại threshold0.5.
Validation mất thêm một lượt inference; không random lại processing giữa các epoch.
Ảnh validation không lưu thêm: bảng tham số CSV đủ tái tạo bằng code/seed/config.

## Chạy an toàn
Khởi tạo từ pretrained CLIP. Không gắn best.pt raw-only làm điểm khởi tạo.
`resume=True` CHỈ khôi phục last.pt của chính giao thức mới, cùng config/manifests.
Chạy smoke trong output_root RIÊNG trước; sau đó full run ở thư mục mặc định khác.
Không tự giảm ảnh/batch, không bỏ ảnh khi OOM. Chưa kiểm chứng full-size CUDA trên Kaggle.
Chế độ mặc định chỉ kiểm tra manifest/path/size khi chạy lại; resume cuối epoch, không giữa epoch.
Manifest có thể lấy từ run raw-only nhưng KHÔNG lấy trọng số detector raw-only để tiếp tục train.

Output: `/kaggle/working/gsd_all7_crop560_blur_jpeg_manifest_v2`.
Sau mỗi epoch: best.pt/last.pt, history.json, validation_metrics.csv,
validation_predictions_epochNNN.csv. Mỗi seed có validation_processing_plan.csv và validation_protocol.json.
Lưu/download output của Kaggle trước khi reset session; /kaggle/working không phải lưu trữ vĩnh viễn.
'''), code('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" "scikit-learn<1.9" "pillow<12" matplotlib tqdm\n'),
        md('## Runtime\nKhông cần upload file Python riêng.'), code(source),
        code('''CONFIG = dict(DEFAULT_CONFIG)
CONFIG.update(
    output_root='/kaggle/working/gsd_all7_crop560_blur_jpeg_manifest_v2',
    dataset_root=None,
    data_mode='reuse_manifests', manifest_root=None,
    models=['clip_gsd'], seeds=[42],
    input_mode='center_crop_max_edge_pad14', crop_max_side=560,
    epochs=50, early_stopping=True, patience=3, min_delta=0.001,
    train_batch=8, test_batch=8, micro_batch=1, anchor_micro_batch=1,
    num_workers=0, precision='fp16', gradient_checkpointing=True,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0.0, 2.0], horizontal_flip_probability=0.0,
    validation_processing_seed=42,
    validation_monitor='mean_native_fixed_processed_cross_entropy',
    eval_cases=['native'],
    smoke=False, resume=True,
)
# Smoke: dùng cùng crop/augmentation/validation, ít ảnh và 1 epoch; output phải riêng.
# CONFIG.update(smoke=True, output_root='/kaggle/working/gsd_all7_crop560_blur_jpeg_manifest_smoke')
# Nếu tự tìm bị nhiều ứng viên: trỏ tới thư mục chứa seed42 (không trỏ tới best.pt).
# CONFIG['manifest_root'] = '/kaggle/input/<attached-output>/gsd_raw_all7_crop560_v1'
# CONFIG['dataset_root'] = '/kaggle/input/datasets/yangsangtai/tiny-genimage'
# Audit lại từ đầu chỉ khi muốn; không tự fallback:
# CONFIG.update(data_mode='full_scan', output_root='/kaggle/working/gsd_all7_blur_jpeg_fullscan_v2')
# Test variants chỉ chạy SAU training, không dùng để chọn checkpoint:
# CONFIG['eval_cases'] = ['native', 'jpeg96', 'jpeg90', 'jpeg80']
print(json.dumps(CONFIG, indent=2))
'''), code('metrics = run(CONFIG)\n'),
        code('''out = Path(CONFIG['output_root'])
display(pd.read_csv(out / 'macro_all.csv'))
display(pd.read_csv(out / 'metrics.csv'))
for seed in CONFIG['seeds']:
    for name in CONFIG['models']:
        folder = out / f'seed{seed}' / name
        print(folder)
        display(pd.read_csv(folder / 'validation_metrics.csv'))
        print(json.loads((folder / 'selected_checkpoint.json').read_text()))
print('Checkpoints:', list(out.glob('seed*/clip_gsd/*.pt')))
''')]
    for index, cell in enumerate(cells):
        cell['id'] = f'aug-all7-{index:02d}'
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), kaggle=dict(accelerator='gpu', isInternetEnabled=True)))


def main():
    path = HERE / NOTEBOOK
    path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
