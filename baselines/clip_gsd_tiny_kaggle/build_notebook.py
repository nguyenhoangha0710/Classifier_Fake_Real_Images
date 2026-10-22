"""Generate readable, self-contained Kaggle ipynb; no repository input required."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True))


def main():
    source = (HERE / 'runtime.py').read_text(encoding='utf-8')
    ast.parse(source)
    cells = [
        md('''# Tiny-GenImage: CLIP + GSD, early stopping (paper v2)

**Mục tiêu:** train real/SD v1.5, test trên 7 generator; báo riêng 6 unseen.
Dataset yangsangtai/tiny-genimage loại bỏ SD14. Đây là adaptation dùng SD15,
không phải giao thức train SD14 của full GenImage.
Nguồn: [GSD v2](https://arxiv.org/html/2603.09242v2), Figure 3, Eq. 9–12, Table IV/VIII.

### Kaggle setup
- Add Input **yangsangtai/tiny-genimage** (folder generator/train|val/nature|ai).
- Chọn GPU T4/P100, bật Internet để tải CLIP. Chỉ GPU 0 được sử dụng, kể cả 2×T4.
- Notebook tự chứa toàn bộ runtime, không cần upload repo.
- Có thể đặt `model_id` thành folder Hugging Face checkpoint offline rồi `local_files_only=True`.
- Full fine-tuning ViT-L/14 rất tốn thời gian. Chạy smoke trước trong output_root riêng.

### Paper / assumptions
Giữ CLIP ViT-L/14, full fine-tune, AdamW backbone LR 1e-6, batch train 128/test 512,
GSD trước 4 block cuối (0-based 20–23), uncentered Batch-SVD k=64,
λmax=1, init a=2,b=1, two-class cross-entropy, average pooling.

Paper không công bố epochs/schedule, optimizer details, head LR, augmentation severity,
checkpoint selection, input preprocessing/pooling token details. **Giả định được log:**
Tối đa 50 epochs, early stopping val CE (patience=3, min_delta=.001), constant LR;
head/modulation LR=1e-6; WD=.01; betas=(.9,.999); eps=1e-8;
OpenAI CLIP 224 preprocessing; mean non-CLS tokens sau final LayerNorm;
flip=.5, blur=.5/sigma[0,2], JPEG=.5/Q[75,100]/subsampling0.
Tiny-GenImage là adaptation, không phải full GenImage reproduction hay official code.

### Run configuration
Mặc định chỉ chạy `clip_gsd`; dùng CLIP fine-tune đã có làm baseline ngoài notebook.
Muốn so sánh trực tiếp cần kiểm tra dữ liệu/preprocessing/training budget của run cũ.
Early stopping dùng validation tách từ SD15 train; không dùng test.
Mỗi lần val loss giảm hơn min_delta so với mốc cải thiện gần nhất thì reset bộ đếm.
Sau patience epoch không cải thiện đủ, dừng và restore checkpoint có val loss thấp nhất tuyệt đối.
Bộ đếm và trạng thái stopped được lưu trong last.pt để resume không train tiếp ngoài ý muốn.
GSD dùng frozen anchor copy đúng pretrained weights. Không LoRA, không freeze forensic blocks.
SVD tính trên **toàn logical batch**, rồi micro-batch backward; không SVD riêng từng micro-batch.
Gradient checkpointing + FP16 giúp tiết kiệm VRAM; không tự giảm logical batch hay k khi OOM.
FP16 AMP skipped steps được log; cần kiểm tra khi diễn giải fairness.

### Evaluation
Một test cohort duy nhất, cân bằng real/fake từng generator, loại decoded-pixel overlaps
với train và validation. Validation tách từ SD15 **train folder**, chọn best bằng val CE;
không chọn threshold/checkpoint trên test. Threshold cố định .5.
Test batch order được random theo ID không dùng nhãn; giữ nguyên batch membership qua mọi case/model.
GSD Batch-Inference có phụ thuộc đồng batch: batch ID được lưu để kiểm toán.

Các case: native; all-JPEG96; resize-square256; resize-square256 rồi all-JPEG96.
Biến đổi trước CLIP preprocessing, áp dụng giống nhau cho cả nhãn. Có thể thêm Q90/Q80.
Resize square có thay đổi aspect ratio/FOV; gọi đây là **processing stress-test**, không phải
chứng minh đã loại size bias. Cùng JPEG Q cũng không xóa prior/double compression.
Không dùng test labels để tạo semantic basis.

Output: audit format/size/quantization hashes, manifests, excluded overlaps, config/versions,
best/last checkpoints, history, per-image predictions, per-generator metrics, macro unseen/all,
paired probability shifts + flip rates, GSD−FT deltas và chart.
Pooled metric có thể lặp real ảnh giữa các generator; ưu tiên macro/per-generator.
Một seed chỉ là pilot; đổi seeds=[42,43,44] cho kết luận nghiên cứu.
'''),
        code('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" "scikit-learn<1.9" "pillow<12" matplotlib tqdm\n'),
        md('## Runtime (có thể thu gọn cell dài này)\nSửa config ở cell sau, không cần sửa runtime.'),
        code(source),
        code('''CONFIG = dict(DEFAULT_CONFIG)
CONFIG.update(
    dataset_root=None,  # auto-discover; set explicit path if multiple datasets match
    output_root='/kaggle/working/clip_gsd_tiny_sd15',
    source_generator='sd15',
    expected_generators=['sd15', 'adm', 'biggan', 'glide', 'midjourney', 'wukong', 'vqdm'],
    strict_eight_generators=False,
    model_id='openai/clip-vit-large-patch14',
    seeds=[42],
    models=['clip_gsd'],
    epochs=50,  # maximum; provisional: paper does NOT disclose epochs
    early_stopping=True, patience=3, min_delta=0.001,
    train_batch=128, test_batch=512,
    micro_batch=2, anchor_micro_batch=8,
    num_workers=2,
    precision='fp16',
    jpeg_eval_qualities=[96],  # optionally [96,90,80]
    size_control=256,
    smoke=False,
    resume=True,
)
# Smoke: CONFIG.update(smoke=True, output_root='/kaggle/working/clip_gsd_smoke')
# If OOM: reduce micro_batch and/or anchor_micro_batch, NOT train_batch/test_batch/k.
# Resume: keep config unchanged and keep last.pt under output_root (same Kaggle session).
# After a session reset, copy a previous output folder to /kaggle/working first.
print(json.dumps(CONFIG, indent=2))
'''),
        md('## Chạy train GSD và paired evaluation\nTải pretrained CLIP, train GSD với early stopping, rồi test best checkpoint qua từng case.'),
        code('metrics = run(CONFIG)\n'),
        code('''out = Path(CONFIG['output_root'])
display(pd.read_csv(out / 'macro_unseen.csv'))
display(pd.read_csv(out / 'model_comparison.csv', header=[0,1], index_col=[0,1,2]))
display(pd.read_csv(out / 'paired_summary.csv').head(20))
from IPython.display import Image as DisplayImage, display
display(DisplayImage(filename=str(out / 'unseen_balanced_accuracy.png')))
'''),
        md('''## Download
Checkpoint full ViT-L/14 và optimizer lớn. Cell dưới đóng gói **báo cáo nhẹ**, không checkpoints.
Checkpoint `seed42/<model>/best.pt` dùng cho inference; `last.pt` có optimizer để resume.
`completion.json` phải ghi complete; smoke không phải kết quả nghiên cứu.
'''),
        code('''import zipfile
archive = out.parent / (out.name + '_reports.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in out.rglob('*'):
        if file.is_file() and file.suffix != '.pt':
            z.write(file, file.relative_to(out))
print(archive)
'''),
    ]
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11'),
        kaggle=dict(accelerator='gpu', isInternetEnabled=True)))
    for i, cell in enumerate(cells):
        cell['id'] = f'clip-gsd-{i:02d}'
    target = HERE / 'train_clip_vs_gsd_tiny_kaggle.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
