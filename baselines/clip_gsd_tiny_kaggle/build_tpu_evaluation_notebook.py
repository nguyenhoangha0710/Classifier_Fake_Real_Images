"""Generate a self-contained Kaggle TPU v5e-8 inference notebook."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = 'test_gsd_uploaded_best_variants_tpu_kaggle.ipynb'


def build_notebook():
    files = ('runtime.py', 'runtime_raw_all7.py', 'manifest_reuse.py',
             'evaluate_all7_variants.py', 'evaluate_uploaded_best.py', 'evaluate_uploaded_best_tpu.py')
    source = '\n\n'.join((HERE / f).read_text(encoding='utf-8') for f in files)
    ast.parse(source)
    def md(text):
        return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))
    def code(text):
        ast.parse(text)
        return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True))
    cells = [md('''# GSD uploaded best — Kaggle TPU v5e-8, inference only

## Input / Settings

Chọn **TPU v5e-8**, bật Internet. Đợi phiên hết Queued trước khi chạy.
Gắn ba input giống bản GPU:

1. Checkpoint augmented của bạn: best.pt, best.zip, hoặc **folder best/** chứa data.pkl + data/ + version.
   Không chọn riêng data.pkl. Tự tìm theo signature checkpoint đã kiểm tra, không theo tên dataset.
2. `yangsangtai/tiny-genimage`, đúng phiên bản dữ liệu gốc.
3. `hadinho0710/mainfest-train-genimage`, có seed42/test_manifest.csv.
4. Lượt tiếp theo: thêm OUTPUT TEST **TPU** của lượt trước để resume; không dùng progress GPU.

Checkpoint không cần train lại. Frozen anchor luôn là CLIP pretrained gốc, không phải detector fine-tuned.
Không cần last.pt/optimizer. Không scan/hash RGB toàn bộ ảnh trước khi test.

## Giao thức và giới hạn

- Giữ đúng cohort/thứ tự 7.000 ảnh, logical batch8, microbatch1; toàn bộ token của cả batch cùng tính basis.
- RGB transform trước crop tối đa560 mỗi chiều, edge-pad14, CLIP normalization, nội suy position ở cả hai nhánh.
- **Một process, một thiết bị TPU** của slice v5e-8. Bản đầu CHƯA tận dụng cả8 chip, không hứa nhanh hơn GPU.
- CLIP chạy TPU mixed BF16, eager attention; Gram/GSD projection giữ FP32, **eigh chạy CPU FP32**.
  CPU không lưu toàn bộ ảnh: chỉ nhận bốn Gram1024x1024/batch để giải eigenvectors.
- Nội suy positional embedding bằng đúng công thức HF bicubic trên CPU, cache tối đa8 shape/nhánh.
- Đây là backend số học KHÁC CUDA FP16/SDPA. Không trộn score/metrics hai backend hoặc khẳng định bit-exact.
- Không resize thêm, không thêm token padding để cố định shape, không đổi batch theo kích thước ảnh.
  Với XLA, shape mới có thể compile lại; lượt đầu có thể chậm. Theo dõi log anchor/detector và thời gian batch.
- Không tự pip thay torch/torch_xla/torch_tpu: dùng backend có sẵn trong ảnh TPU Kaggle.
  Auto ưu tiên TorchTPU nếu đã cài, nếu không dùng PyTorch/XLA hiện có (major/minor phải khớp torch).

Nguồn API: [Google TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu),
[XLA runtime release2.8](https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/runtime.py).
Google đã chuyển hướng workload mới sang TorchTPU; nhánh XLA ở đây để tương thích image Kaggle hiện có.

## Chạy lần đầu

Mặc định **RUN_FULL_TEST=False**: chạy toy numerical preflight TPU/CPU, rồi 2 batch native + fake_jpeg96
(16 ảnh/case). Đây là SMOKE, KHÔNG phải kết quả test7.000 ảnh. Không dùng AUC/accuracy smoke để kết luận.
Kiểm tra tpu_preflight.json, dự đoán finite, tốc độ, log lỗi/fallback. Toy preflight KHÔNG chứng minh
checkpoint thực tương đương GPU. Muốn kiểm tra parity cần cùng checkpoint + cùng8 IDs/batch ở hai backend.

Sau khi smoke chạy ổn: đổi RUN_FULL_TEST=True và Save & Run All. Output full khác output smoke,
không dùng lại smoke vào full. Chín case giống bản GPU, bỏ PNG roundtrip vì trùng RGB native.

## Lưu và tiếp tục

Full mặc định tối đa3 case mới hoặc9h/lượt, dừng bình thường để xuất ZIP trước giới hạn phiên.
Lưu shard sau MỖI batch hoàn chỉnh (8ảnh), metrics in khi đủ7.000 ảnh/case.
Nếu bị hard kill giữa batch, batch chưa ghi phải chạy lại. Không bảo đảm giữ working nếu Kaggle không lưu Output.
Planned pause: tải ZIP / lưu Output, Add Input nó ở lượt sau và giữ backend/precision/phiên bản/hardware như cũ.
resume_from='auto' chỉ khôi phục fingerprint khớp. Không gắn output cũ thì phiên mới không thể resume.
CSV gồm prob_fake=softmax(logits.float())[:,1], logit_real/logit_fake/logit_margin, batch_id, input_tokens.
JPEG fake-only là phép chẩn đoán dùng nhãn; không phải preprocessing triển khai ngoài thực tế.
'''), code('''# Giữ nguyên TPU runtime của Kaggle. Không install/upgrade torch ở cell này.
import os, sys, subprocess, importlib.util, importlib.metadata
os.environ.setdefault('PJRT_DEVICE', 'TPU')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
before_torch = importlib.metadata.version('torch')
backends = {name: importlib.util.find_spec(name) is not None for name in ('torch_tpu', 'torch_xla')}
print('torch:', before_torch, '| installed TPU backends:', backends, flush=True)
if not any(backends.values()):
    raise RuntimeError('Chưa có TPU runtime. Chọn TPU v5e-8 và khởi động lại phiên; gửi log này nếu vẫn lỗi.')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.46.3', 'huggingface_hub>=0.23.2,<1.0',
    'pandas<3', 'scikit-learn<1.9', 'pillow<12', 'matplotlib', 'tqdm'])
if importlib.metadata.version('torch') != before_torch:
    raise RuntimeError('torch unexpectedly changed; restart with matching TPU runtime before proceeding')
'''), md('## Runtime — tự chứa toàn bộ code, không gọi train'), code(source), code('''# LẦN ĐẦU giữ False. Sau khi kiểm tra smoke ổn mới chuyển True.
RUN_FULL_TEST = False
EVAL_CONFIG = dict(TPU_DEFAULTS)
EVAL_CONFIG.update(
    checkpoint_path=None, test_manifest_path=None, dataset_root=None,
    tpu_backend='auto',           # auto / xla / torch_tpu (phải là backend có sẵn)
    tpu_precision='bf16',         # fp32 để chẩn đoán; đổi precision => output/fingerprint mới
    cpu_threads=4,
    output_root='/kaggle/working/gsd_uploaded_best_tpu_v1',
    resume_from='auto',
    max_batches=None, max_new_cases_per_run=3, time_budget_hours=9.0,
    save_every_batches=1, log_every_batches=1,
    micro_batch=1, anchor_micro_batch=1, num_workers=0,
    cases=['native', 'fake_jpeg96', 'all_jpeg96', 'all_jpeg90', 'all_jpeg80',
           'resize_square256', 'resize_square560', 'blur_sigma1', 'resize256_jpeg96'],
)
if not RUN_FULL_TEST:
    EVAL_CONFIG.update(max_batches=2, cases=['native', 'fake_jpeg96'],
        output_root='/kaggle/working/gsd_uploaded_best_tpu_smoke_v1',
        resume_from=None, max_new_cases_per_run=2, time_budget_hours=1.5)
# Đường dẫn được auto tìm trong /kaggle/input, kể cả folder best/ đã giải nén.
# Nếu cần explicit: checkpoint_path phải trỏ cả folder best/, không trỏ data.pkl.
# EVAL_CONFIG['checkpoint_path'] = '/kaggle/input/<dataset>/best'
# EVAL_CONFIG['test_manifest_path'] = '/kaggle/input/<manifest>/seed42/test_manifest.csv'
print('MODE:', 'FULL TEST' if RUN_FULL_TEST else 'SMOKE ONLY — NOT final metrics')
print(json.dumps(EVAL_CONFIG, indent=2))
'''), code('''metrics = run_uploaded_tpu_evaluation(EVAL_CONFIG)
'''), code('''out = Path(EVAL_CONFIG['output_root'])
state = json.loads((out / 'evaluation_state.json').read_text())
print('MODE:', 'FULL' if RUN_FULL_TEST else 'SMOKE — KHÔNG dùng để báo cáo accuracy/AUC')
print('STATUS:', state['status'], '| reason:', state['reason'])
print('Completed:', state['completed_cases'], '| Remaining:', state['remaining_cases'])
display(metrics[metrics.generator == 'pooled'])
print('TPU numerical sanity:', (out / 'tpu_preflight.json').read_text())
print('Backend fingerprint:', (out / 'run_fingerprint.json').read_text())
if (out / 'partial_metrics.csv').is_file():
    print('PARTIAL — chưa đủ ảnh/case:')
    display(pd.read_csv(out / 'partial_metrics.csv'))
'''), code('''# ZIP nhỏ chứa dự đoán + progress + metadata, không có ảnh hoặc trọng số.
archive_path = out.parent / (out.name + '_results.zip')
temp = archive_path.with_suffix('.zip.tmp')
with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.csv', '.json'):
            archive.write(path, (Path(out.name) / path.relative_to(out)).as_posix())
temp.replace(archive_path)
from IPython.display import FileLink
display(FileLink(str(archive_path)))
print('Nếu full paused: lưu output, Add Input vào phiên TPU tiếp theo để resume.')
''')]
    for i, cell in enumerate(cells):
        cell['id'] = f'uploaded-tpu-{i:02d}'
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), kaggle=dict(accelerator='tpu', isInternetEnabled=True)))


if __name__ == '__main__':
    target = HERE / NOTEBOOK
    target.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding='utf-8')
    print(target)
