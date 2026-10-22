"""Build the standalone Kaggle encoder-training notebook from checked local sources."""
import base64
import io
import json
from pathlib import Path
import zipfile

BASE=Path(__file__).resolve().parent
def main():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(BASE/'train_codec.py','train_codec.py')
        for name in ['__init__.py','core.py','data.py','checkpointing.py']:
            archive.write(BASE.parent/'forensic_twins_original/src'/name,'native/src/'+name)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    cells=[]
    def add(kind,text):
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code': cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# Forensic Twins Codec — encoder only, Kaggle 2×T4

Fine-tune native encoder/projector for 20 epochs; FRE frozen. View1 native, view2 50% JPEG Q80–100 / 35% down-up→JPEG / 15% identity. No GMM fit or B-Free evaluation.

**Add Input:**
- Competition `imagenet-object-localization-challenge` (accept rules).
- Dataset `awsaf49/coco-2017-dataset`.
- Dataset `hoangha071005/twins-gmm-200k-checkpoint`.

Training uses the exact existing `train_real_200k.csv` (100K per source), no manifest rebuilding. B-Free and old fitted GMM are not required.

Outputs remain in `/kaggle/working/forensic_twins_codec`, captured by Save Version. Each epoch uploads latest checkpoint, manifest, FRE, config, CSV logs and PNG plots to `hoangha071005/forensic-twins-codec-checkpoint-encoder`. Enable Internet and Kaggle account permissions for publishing. Upload failure warns but preserves Working. A hard session stop before upload can lose unpublished steps.
''')
    add('code','''from pathlib import Path
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working/forensic_twins_codec')
CODE_ROOT = Path('/kaggle/working/forensic_twins_codec_code')
NATIVE_ROOT = ''  # optional exact mount override; default resolves two known Kaggle layouts
RESUME_CHECKPOINT = ''  # optional codec last.ckpt from previous Output/Dataset
MODE = 'native_random'  # native_native is matched continued-training control; use separate Output
PUBLISH = True
''')
    add('code',f'''import base64, io, zipfile, subprocess, sys, json
CODE_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_ARCHIVE_B64 = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    archive.extractall(CODE_ROOT)
import torch, PIL, matplotlib
print('torch:', torch.__version__, 'Pillow:', PIL.__version__, 'GPUs:', torch.cuda.device_count())
assert torch.cuda.device_count() == 2, 'Select GPU T4 ×2'
subprocess.run(['nvidia-smi'], check=False)
''')
    add('markdown','''## Preflight — CPU/I/O, before distributed training
Validates manifest SHA256, 200K real/source counts, every rebased pixel path, native checkpoint/FRE identity. Errors stop execution; no fallback sampling, dataset creation or silent missing-file removal. Native bundle must contain `forensic_twins_v2_200k/{manifests/train_real_200k.csv,cache/fre.pt,artifacts/checkpoints/last.ckpt}`.
''')
    add('code','''common = ['--input-root', str(INPUT_ROOT), '--output-root', str(OUTPUT_ROOT), '--mode', MODE]
if NATIVE_ROOT: common += ['--native-root', NATIVE_ROOT]
if RESUME_CHECKPOINT: common += ['--resume-checkpoint', RESUME_CHECKPOINT]
if not PUBLISH: common += ['--no-publish']
subprocess.run([sys.executable, str(CODE_ROOT/'train_codec.py'), *common, '--preflight-only'], check=True)
''')
    add('markdown','''## Train / resume
Batch128=64×2; LR6e-5, warmup2 epochs, cosine→1e-6, clip1, FP16. Native warm-start loads only encoder/projector; codec resume also restores optimizer/scheduler/scaler and position. Codec identity rejects changes in data/config/native checkpoint. Save last.ckpt every250 optimizer steps and each epoch; print every100 updates plus epoch summaries. Epoch CSV after mid-epoch resume describes the resumed segment. Per-policy counts and feature std in CSV are rank0 diagnostics.

To resume a new session: add your output checkpoint Dataset as Input and set RESUME_CHECKPOINT to its last.ckpt (flat publish layout). Existing local last.ckpt is auto-resumed. For continuing loss curves, copy previous logs/ (or unpack logs.zip) into OUTPUT_ROOT before training; checkpoint state does not require logs.
''')
    add('code','''subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(CODE_ROOT/'train_codec.py'), *common], check=True)
''')
    add('code','''from IPython.display import display, Image, FileLink
curve = OUTPUT_ROOT/'figures/training_curves.png'
if curve.exists(): display(Image(filename=str(curve)))
for name in ['last.ckpt','encoder_final.pt','logs/train_steps.csv','logs/epoch_metrics.csv','figures/training_curves.png']:
    path = OUTPUT_ROOT/name
    if path.exists(): display(FileLink(str(path)))
print('Encoder training only. Fit a NEW GMM in a separate notebook.')
''')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    path=BASE/'forensic_twins_codec_encoder_kaggle.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Built {path}; cells={len(cells)}, source_bytes={len(buffer.getvalue())}')
if __name__=='__main__': main()
