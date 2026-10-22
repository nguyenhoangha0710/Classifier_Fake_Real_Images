"""Bundle sources into a standalone Kaggle notebook; no external write or launch."""
import base64
import io
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
files=['model.py','data.py','train.py','prepare.py','density.py','download_clip.py','progress.py','test_model.py','test_ddp.py','config.json']
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for name in files: z.write(HERE/name,name)
encoded=base64.b64encode(buffer.getvalue()).decode()


def cell(kind,source):
    result={'cell_type':kind,'metadata':{},'source':source.splitlines(True)}
    if kind=='code': result.update(execution_count=None,outputs=[])
    return result


cells=[cell('markdown','''# Native Twins–GSD: ViT-L/14, GSD blocks 21–24
Encoder-only training notebook. Real-only pairs, Batch-SVD K64, adaptive suppression.
Choose GPU hardware and measure full-gradient smoke before a 20-epoch run.
Enable Kaggle Settings > Internet. CLIP weights download from Hugging Face into Working.
Mount the real image manifests and COCO/ImageNet pixels; no CLIP Kaggle Input is needed.
Keep smoke and main output directories separate. Save Version preserves Working artifacts.
'''),cell('code',f'''import base64, io, zipfile, pathlib, sys, subprocess, json, os
os.environ['PYTHONUNBUFFERED']='1'
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING','1')
code_root=pathlib.Path('/kaggle/working/gsd_twins_code')
code_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('{encoded}'))) as bundle:
    bundle.extractall(code_root)
print(code_root)
'''),cell('code','''# Install only if the Kaggle environment does not already provide these versions.
subprocess.run([sys.executable,'-m','pip','install','transformers==4.46.3','pillow>=9.1','scikit-learn>=1.2'],check=True)
'''),cell('code','''# Explicit portable paths: image_path or relative_path must resolve under PIXELS.
CLIP_CACHE=pathlib.Path('/kaggle/working/clip_cache')
CLIP_REVISION='main'  # resolved commit is saved; use that commit for repeat experiments
PIXELS=pathlib.Path('/kaggle/input')
# Exact dataset-relative path recorded in CHECKPOINTS.md / kaggle_inputs.md.
# Two direct file checks handle Kaggle's modern and legacy mount conventions;
# no glob/rglob, recursive scan, or image-directory traversal.
CHECKPOINT_ROOT=pathlib.Path('/kaggle/input/datasets/hoangha071005/twins-gmm-200k-checkpoint')
LEGACY_CHECKPOINT_ROOT=pathlib.Path('/kaggle/input/twins-gmm-200k-checkpoint')
MANIFEST_RELATIVE=pathlib.Path('forensic_twins_v2_200k/manifests/train_real_200k.csv')
PARENT=CHECKPOINT_ROOT/MANIFEST_RELATIVE
if not PARENT.is_file():
    PARENT=LEGACY_CHECKPOINT_ROOT/MANIFEST_RELATIVE
    if not PARENT.is_file():
        raise FileNotFoundError(
            'Saved manifest is missing at both explicit mount paths: '
            f'{CHECKPOINT_ROOT/MANIFEST_RELATIVE} ; '
            f'{LEGACY_CHECKPOINT_ROOT/MANIFEST_RELATIVE}. '
            'Attach hoangha071005/twins-gmm-200k-checkpoint as Kaggle Input.'
        )
ELIGIBLE=pathlib.Path('/kaggle/working/ssl_train_eligible.csv')
SMOKE=pathlib.Path('/kaggle/working/gsd_native_smoke_amp_v2')
MAIN=pathlib.Path('/kaggle/working/gsd_native_main')
BATCH_PER_GPU=2  # preflight starting point; lock actual batch before B0/C0
GPUS=1          # explicitly choose; full model has not been validated on T4
WORKERS=2       # train.py uses spawn after CUDA initialization
LOADER_TIMEOUT=120
LOG_EVERY=10
PRECISION='auto'  # BF16 on supported GPUs; otherwise FP16 + dynamic loss scaling

def run(*args):
    print('[launch]', ' '.join(str(v) for v in args),flush=True)
    subprocess.run([str(v) for v in args],cwd=code_root,check=True)

print('SSL parent manifest:',PARENT)
'''),cell('code','''# Download once before torchrun; all workers then load the same pinned local snapshot.
sys.path.insert(0,str(code_root))
from download_clip import download_clip
CLIP=download_clip(CLIP_CACHE,revision=CLIP_REVISION)
assert (CLIP/'config.json').is_file()
'''),cell('code','''print('[stage] CPU contract tests (not training)',flush=True)
run(sys.executable,'-u','-m','unittest','discover','-s',code_root,'-p','test_*.py','-v')
'''),cell('code','''print('[stage] PREPARE: fast saved-metadata filtering; loss begins only in the next cell',flush=True)
run(sys.executable,'-u',code_root/'prepare.py','--parent',PARENT,'--pixel-root',PIXELS,
    '--output',ELIGIBLE,'--purpose','ssl','--audit','metadata','--log-every',1000,'--reuse')
print('[stage] PREPARE COMPLETE. Next: GPU smoke, then optional main training.',flush=True)
'''),cell('code','''# Global batch must be >=2 for Barlow. Batch-per-GPU1 needs >=2 GPUs.
assert BATCH_PER_GPU*GPUS>=2
def train_command(output, smoke=0):
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={GPUS}',
             code_root/'train.py','--clip',CLIP,'--manifest',ELIGIBLE,'--pixel-root',PIXELS,
             '--output',output,'--batch-per-gpu',BATCH_PER_GPU,'--workers',WORKERS,
             '--loader-timeout',LOADER_TIMEOUT,'--log-every',LOG_EVERY,'--heartbeat',30,
             '--precision',PRECISION]
    if smoke: command+=['--smoke-steps',smoke]
    return command
run(*train_command(SMOKE,200))
'''),cell('code','''# Run only after reviewing smoke logs/VRAM/runtime and locking the paired-run config.
RUN_MAIN=False
print('[stage] MAIN20 epochs:', 'ENABLED' if RUN_MAIN else 'DISABLED (RUN_MAIN=False)',flush=True)
if RUN_MAIN:
    run(*train_command(MAIN))
'''),cell('markdown','''## Detector after training
Use density.py extract for saved real-fit, calibration and test manifests.
Extraction uses per-sample SVD and FP32 at all four layers. density.py fit accepts only real fit/calibration
features, checks disjoint sample identities and fits PCA128 + full Gaussian with a p95 calibration threshold.
density.py score writes per-image NLL and fixed-threshold predictions. See the repository README for commands.
''')]
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                  'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
(HERE/'gsd_twins_native_kaggle.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
print('Built notebook with',len(files),'embedded source files')
