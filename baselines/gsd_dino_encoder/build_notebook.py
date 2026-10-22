"""Build a standalone Kaggle notebook with embedded project modules."""
import base64
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(HERE.glob('*.py')):
        if path.name != 'build_notebook.py':
            archive.write(path, path.relative_to(ROOT).as_posix())
    archive.write(HERE / 'config.json', 'baselines/gsd_dino_encoder/config.json')
    for name in ('data.py', 'progress.py', 'download_clip.py'):
        path = ROOT / 'baselines/gsd_forensic_twins' / name
        archive.write(path, path.relative_to(ROOT).as_posix())
    for path in sorted((HERE / 'tests').glob('*.py')):
        archive.write(path, path.relative_to(ROOT).as_posix())
encoded = base64.b64encode(buffer.getvalue()).decode('ascii')


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


cells = [cell('markdown', '''# GSD + DINO: real-only encoder
CLIP ViT-L/14 pretrained, fixed reference basis k64, DINO + iBOT + 0.1 KoLeo.
Enable GPU and Internet in Kaggle settings. CLIP downloads directly from Hugging Face;
no CLIP dataset input is needed. Project code is embedded in this notebook.
Attach `hoangha071005/twins-gmm-200k-checkpoint` and the COCO/ImageNet pixel datasets
used by its manifest. The checkpoint dataset supplies the CSV, not necessarily the pixels.
Outputs are written under `/kaggle/working`. Run cells in order; main training is explicit.
'''), cell('code', f'''import base64, io, zipfile, pathlib, sys, subprocess, os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
CODE = pathlib.Path('/kaggle/working/gsd_dino_code')
CODE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('{encoded}'))) as bundle:
    bundle.extractall(CODE)
sys.path.insert(0, str(CODE))
def run(*args):
    print('[launch]', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=CODE, check=True)
run(sys.executable, '-m', 'pip', 'install', 'transformers==4.46.3', 'pillow>=9.1')
'''), cell('code', '''# Validate the SAME Python interpreter used by reference/training subprocesses.
# Stop before downloads/data preparation if this session cannot run CUDA.
run(sys.executable, '-c', """
import torch
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)
if torch.version.cuda is None:
    raise RuntimeError(
        'This Python has CPU-only PyTorch. Enable a GPU accelerator for the Kaggle '
        'session and use its CUDA-enabled PyTorch environment; restart the session '
        'after changing hardware. If GPU is already enabled, check the installed '
        'torch build. Installing transformers does not add CUDA to torch.')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA PyTorch is installed but no GPU is accessible. '
                       'Check the Kaggle session accelerator and restart the session.')
for i in range(torch.cuda.device_count()):
    print('GPU', i, torch.cuda.get_device_name(i), flush=True)
""")
'''), cell('code', '''PIXELS = pathlib.Path('/kaggle/input')
RELATIVE = pathlib.Path('forensic_twins_v2_200k/manifests/train_real_200k.csv')
ROOTS = [
    PIXELS/'datasets/hoangha071005/twins-gmm-200k-checkpoint',
    PIXELS/'twins-gmm-200k-checkpoint',
]
PARENT = next((root/RELATIVE for root in ROOTS if (root/RELATIVE).is_file()), None)
if PARENT is None:
    raise FileNotFoundError('Attach hoangha071005/twins-gmm-200k-checkpoint; expected ' + str(RELATIVE))
DATA = pathlib.Path('/kaggle/working/gsd_dino_data')
DATA.mkdir(exist_ok=True)
ELIGIBLE = DATA/'train_real_eligible.csv'
BASIS = DATA/'fixed_basis.pt'
SMOKE = pathlib.Path('/kaggle/working/gsd_dino_smoke')
MAIN = pathlib.Path('/kaggle/working/gsd_dino_main')
BATCH_PER_GPU = 2  # minimum for per-GPU KoLeo; measure VRAM in smoke
GPUS = 1
WORKERS = 0
print('Parent manifest:', PARENT)
'''), cell('code', '''# Download once before launching training workers; save resolved revision.
from baselines.gsd_forensic_twins.download_clip import download_clip
CLIP = download_clip('/kaggle/working/clip_cache', revision='main')
print('Local CLIP snapshot:', CLIP)
'''), cell('code', '''run(sys.executable, '-m', 'unittest', 'discover',
    '-s', CODE/'baselines/gsd_dino_encoder/tests', '-v')
'''), cell('code', '''# Filter using saved width/height; only read headers if dimensions are missing.
run(sys.executable, '-m', 'baselines.gsd_dino_encoder.data',
    '--parent', PARENT, '--pixel-root', PIXELS, '--output', ELIGIBLE,
    '--min-side', 224, '--audit', 'metadata')
'''), cell('code', '''# Offline frozen CLIP reference; generated from eligible training images only.
run(sys.executable, '-m', 'baselines.gsd_dino_encoder.reference',
    '--clip', CLIP, '--manifest', ELIGIBLE, '--pixel-root', PIXELS,
    '--output', BASIS, '--reference-size', 4096, '--tokens-per-image', 64,
    '--batch-size', 8, '--device', 'cuda')
'''), cell('code', '''def train_command(output, smoke=0, resume=None):
    assert BATCH_PER_GPU >= 2, 'KoLeo requires >=2 images on EACH GPU'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={GPUS}', '-m', 'baselines.gsd_dino_encoder.train',
               '--clip', CLIP, '--basis', BASIS, '--manifest', ELIGIBLE,
               '--pixel-root', PIXELS, '--output', output,
               '--batch-per-gpu', BATCH_PER_GPU, '--workers', WORKERS]
    if smoke:
        command += ['--smoke-steps', smoke]
    if resume is not None:
        command += ['--resume', resume]
    return command
run(*train_command(SMOKE, smoke=20))
'''), cell('code', '''# Review smoke losses/VRAM first. Full ViT-L CUDA memory fit is not yet measured.
RUN_MAIN = False
RESUME = None  # e.g. MAIN/'last.ckpt', same output/config as original main run
if RUN_MAIN:
    run(*train_command(MAIN, resume=RESUME))
else:
    print('Set RUN_MAIN=True after reviewing smoke to start full training.')
'''), cell('code', '''# Run after completing main training. Smoke checkpoints cannot be exported.
EXPORT = False
if EXPORT:
    run(sys.executable, '-m', 'baselines.gsd_dino_encoder.export', 'export',
        '--checkpoint', MAIN/'last.ckpt', '--output', MAIN/'encoder.pt')
''')]
notebook = dict(cells=cells, metadata={
    'kernelspec': dict(display_name='Python 3', language='python', name='python3'),
    'language_info': dict(name='python')}, nbformat=4, nbformat_minor=5)
destination = HERE / 'gsd_dino_kaggle.ipynb'
destination.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(destination)
