"""Standalone notebook for the user's recovered blur/JPEG best checkpoint."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = 'test_gsd_uploaded_best_variants_kaggle.ipynb'


def build_notebook():
    files = ('runtime.py', 'runtime_raw_all7.py', 'manifest_reuse.py',
             'evaluate_all7_variants.py', 'evaluate_uploaded_best.py')
    source = '\n\n'.join((HERE / name).read_text(encoding='utf-8') for name in files)
    ast.parse(source)

    def md(text):
        return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))

    def code(text):
        return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True))

    cells = [md('''# GSD uploaded best — đủ 9 case, không train, có resume

Bản sửa full-suite: cell chạy CHỐT LẠI đủ9 case ngay trước evaluator. Cấu hình cũ
cases=['native'] / max_new_cases_per_run=1 / max_batches=2 không thể âm thầm giới hạn full test.
Giữ SMOKE_ONLY=False để chạy toàn bộ. Chỉ đổi True khi CỐ Ý chạy kiểm tra16 ảnh/case.

## Input cần gắn

1. **Checkpoint mới tải về:** upload `best.pt` thành Kaggle Dataset và gắn vào notebook.
   File local `best.zip` của bạn thực chất là một checkpoint PyTorch ZIP-format, KHÔNG phải
   ZIP chứa best.pt. Bản `best.pt` đi kèm chỉ là bản sao đổi tên, không đổi trọng số.
   Ưu tiên upload best.pt, không tự giải nén nó. Code vẫn hỗ trợ best.zip gốc, ZIP bọc best.pt,
   hoặc folder best/ với data.pkl + data/ + version nếu Kaggle đã giải nén tự động.
2. **Ảnh:** `yangsangtai/tiny-genimage`, đúng phiên bản đã dùng khi train.
3. **Manifest:** `hadinho0710/mainfest-train-genimage` đã dùng trước, hoặc gói
   `gsd_all7_manifests_seed42.zip`. Chỉ cần seed42/test_manifest.csv; split_summary.json khuyến nghị.
4. **Chỉ khi tiếp tục test:** gắn OUTPUT của lượt test trước làm Input; resume_from='auto'.

Bật **GPU + Internet** để tải frozen anchor là CLIP pretrained gốc. Không dùng trọng số detector
đã fine-tune làm anchor. Không cần last.pt, optimizer, config.json ngoài hoặc selected_checkpoint.json:
config và epoch đã nằm TRONG checkpoint mới này. Không nạp checkpoint không rõ nguồn.
Code dùng torch.load(weights_only=True), không fallback sang pickle không giới hạn.

Checkpoint đã kiểm tra: epoch1, signature
`4595045b4c6c76c1ad4aa303a7d2f1690cfac8150add4548d24609a6c980dce6`.
Đây là best đã lưu trước timeout, không chứng minh training đã hoàn tất early stopping.

## Giao thức

Đúng 7.000 test IDs/thứ tự/batch8 cũ; microbatch1; crop tối đa560 mỗi chiều, edge-pad14,
CLIP normalization, Gram-eigh/SDPA như lúc train. Biến đổi trên RGB gốc TRƯỚC crop.
Không scan/hash RGB toàn bộ dataset. Kiểm tra metadata manifest và đường dẫn/file byte length;
decode và kiểm tra dimensions khi từng ảnh thực sự được inference.
Không random augmentation ở test, không train lại, không tuning threshold trên test; threshold0.5.
Mỗi case ước lượng lại GSD basis cho TOÀN BỘ batch, kể cả real ở fake-only JPEG.

## Case và tiết kiệm thời gian

| Case | Xử lý trước crop |
|---|---|
| native | Không processing thêm |
| fake_jpeg96 | Fake JPEG96, real native; cần nhãn, chỉ là chẩn đoán bias |
| all_jpeg96 / 90 / 80 | Cả hai lớp JPEG cùng quality |
| resize_square256 / 560 | Resize BICUBIC vuông, kể cả upscale ảnh nhỏ |
| blur_sigma1 | Gaussian blur radius1 trên RGB gốc |
| resize256_jpeg96 | Resize vuông256 rồi JPEG96 |

PNG roundtrip bị bỏ mặc định vì không đổi RGB pixel. Nếu thêm lại, code ghi alias native,
KHÔNG giả vờ đã đo một case PNG độc lập. Request trùng hoặc tên khác nhưng cùng transform cũng
được gộp. Fake-JPEG96 KHÔNG trùng all-JPEG96; không dùng lại score real giữa hai case vì GSD phụ thuộc batch.
JPEG giữ subsampling0 (4:4:4), optimize=False, progressive=False như train/evaluator cũ.

Mặc định yêu cầu **đủ9 case**, KHÔNG giới hạn số case/lượt, ngân sách **9.5 giờ tính cả chuẩn bị**. Hết ngân sách:
lưu các batch hoàn chỉnh và KẾT THÚC BÌNH THƯỜNG để Kaggle có thể lưu output trước 12h.
Native/case hoàn tất được dùng lại, zero forward. Không có bảo đảm nếu GPU/I/O treo hoặc nền tảng
kill phiên trước ngân sách. Đủ9 case mới có nghĩa hoàn tất full suite. Nếu hết giờ trước thì status=paused,
remaining_cases ghi rõ các case còn lại; gắn output chạy tiếp. Không hứa9 case sẽ luôn xong trong một phiên.

## Chạy tiếp sau một lượt

Nếu status='paused': lưu output/ZIP kết quả; gắn output đó vào lượt chạy tiếp theo cùng checkpoint,
ảnh, manifest, phiên bản thư viện và loại GPU. `resume_from='auto'` tìm progress có fingerprint khớp.
Nếu đổi environment hoặc checkpoint thì dùng output_root mới; không trộn score giữa hai mô hình.
`resume_from` cụ thể phải trỏ tới thư mục chứa evaluation_state.json, KHÔNG trỏ tới best.pt.
Không gắn output cũ thì lượt mới không thể biết những case đã chạy trước đó.

Lưu shard mỗi10 batch (80 ảnh), không lưu giữa logical batch. Hard kill có thể mất các batch chưa flush.
Metrics chính chỉ gồm case ĐÃ ĐỦ ảnh; case chưa đủ ghi riêng partial_metrics.csv.
Sau mỗi case đủ ảnh, in kết quả ngay và ghi metrics.csv / predictions_CASE.csv.
CSV có prob_fake=softmax[:,1], logit_real, logit_fake, logit_margin để phân tích/calibration sau.
Model mới nên KHÔNG so sánh native với predictions native của checkpoint raw-only cũ.
'''), code('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" "scikit-learn<1.9" "pillow<12" matplotlib tqdm\n'),
        md('## Runtime\nSelf-contained. Cell dưới chỉ gọi evaluator, không gọi train.'), code(source),
        code('''SMOKE_ONLY = False          # False=đủ9 case; True=kiểm tra16 ảnh/case, output riêng
EVAL_CONFIG = dict(UPLOAD_DEFAULTS)
EVAL_CONFIG.update(
    checkpoint_path=None,       # auto đúng signature mới; hoặc đường dẫn best.pt / best.zip
    test_manifest_path=None,    # ưu tiên manifest_root đã ghi trong checkpoint nếu còn tồn tại
    dataset_root=None,
    output_root='/kaggle/working/gsd_uploaded_best_variants_v1',
    pretrained_model_path=None,
    resume_from='auto',         # tự tìm OUTPUT TEST TRƯỚC có fingerprint khớp
    cases=list(UPLOADED_FULL_SUITE_CASES),
    max_new_cases_per_run=None,  # không dừng chỉ vì đã chạy đủ1/3 case
    time_budget_hours=9.5,
    save_every_batches=10, log_every_batches=10,
    micro_batch=1, anchor_micro_batch=1, num_workers=0,
    max_batches=None,           # None=7.000 ảnh/case; không đổi test batch8
)
# Nếu có nhiều nguồn khác nhau, điền đường dẫn THỰC TẾ trên Kaggle:
# EVAL_CONFIG['checkpoint_path'] = '/kaggle/input/<checkpoint-dataset>/best.pt'
# EVAL_CONFIG['test_manifest_path'] = '/kaggle/input/datasets/hadinho0710/mainfest-train-genimage/seed42/test_manifest.csv'
# Nếu auto không tìm được progress trước:
# EVAL_CONFIG['resume_from'] = '/kaggle/input/<saved-test-output>/gsd_uploaded_best_variants_v1'
# Nếu muốn dùng lại native đã chạy: Add Input dataset output trước và giữ cùng môi trường.
# Auto chỉ dùng lại fingerprint KHỚP. Khác Pillow/torch/GPU/checkpoint thì không trộn score.
# Không thêm cell native-only. Cell chạy dưới sẽ chốt lại cả9 case và bỏ max_batches/case limit.
# Smoke: đổi SMOKE_ONLY=True; tự chọn output riêng, không trộn vào full test.
print(json.dumps(EVAL_CONFIG, indent=2))
'''), code('''# Chốt cấu hình tại đúng thời điểm gọi evaluator, kể cả có cell cũ ghi đè trước đó.
EVAL_CONFIG = finalize_uploaded_notebook_config(EVAL_CONFIG, smoke_only=SMOKE_ONLY)
metrics = run_uploaded_evaluation(EVAL_CONFIG)
'''),
        code('''out = Path(EVAL_CONFIG['output_root'])
state = json.loads((out / 'evaluation_state.json').read_text())
print('STATUS:', state['status'], '| reason:', state['reason'])
print('Completed:', state['completed_cases'])
print('Remaining:', state['remaining_cases'])
print('MODE:', 'SMOKE — không phải full test' if SMOKE_ONLY else 'FULL 9-CASE SUITE')
if not SMOKE_ONLY and state['planned_cases'] != list(UPLOADED_FULL_SUITE_CASES):
    raise RuntimeError('Wrong case plan: this is not the full nine-case evaluation.')
if not SMOKE_ONLY and state['status'] == 'complete':
    print('ĐÃ HOÀN TẤT ĐỦ9 CASE, mỗi case7.000 ảnh.')
display(metrics[metrics.generator == 'pooled'])
if (out / 'partial_metrics.csv').is_file():
    print('PARTIAL: chưa đủ ảnh, không trộn vào bảng kết quả chính.')
    display(pd.read_csv(out / 'partial_metrics.csv'))
print('Checkpoint:', (out / 'checkpoint_inspection.json').read_text())
'''), code('''# Tạo gói nhỏ để tải / upload thành dataset input cho lượt test tiếp theo.
# Chỉ chứa metrics/predictions/progress/metadata; KHÔNG chứa model hoặc ảnh.
archive_path = out.parent / (out.name + '_results.zip')
temp = archive_path.with_suffix('.zip.tmp')
with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.csv', '.json', '.png'):
            archive.write(path, (Path(out.name) / path.relative_to(out)).as_posix())
temp.replace(archive_path)
from IPython.display import FileLink
display(FileLink(str(archive_path)))
print('Nếu paused: lưu output này và Add Input nó ở lượt tiếp theo; không cần chạy lại native.')
''')]
    for index, cell in enumerate(cells):
        cell['id'] = f'uploaded-best-{index:02d}'
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and not text.startswith('%pip'):
            ast.parse(text)
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), kaggle=dict(accelerator='gpu', isInternetEnabled=True)))


if __name__ == '__main__':
    path = HERE / NOTEBOOK
    path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding='utf-8')
    print(path)
