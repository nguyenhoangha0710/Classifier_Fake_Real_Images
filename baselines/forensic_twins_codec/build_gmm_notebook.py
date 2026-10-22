"""Package codec evaluation plus native helpers into a standalone Kaggle notebook."""
import base64
import io
import json
from pathlib import Path
import zipfile
BASE=Path(__file__).resolve().parent

def main():
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(BASE/'evaluate_codec_gmm.py','evaluate_codec_gmm.py')
        native=BASE.parent/'forensic_twins_original'
        for name in ['fit_gmm_and_evaluate.py','test_bfree_gmm_kaggle.py','src/__init__.py','src/core.py','src/data.py','src/checkpointing.py']:
            z.write(native/name,'native/'+name)
    encoded=base64.b64encode(data.getvalue()).decode()
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code': c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown','''# Codec encoder200K → new GMM → Tiny / B-Free

Native K1/full Gaussian. Use saved160K real-fit and40K calibration. Main threshold Q95 is estimated ONLY on calibration-real; Q90/Q99 are supplementary. Test labels never choose thresholds. Encoder/FRE frozen; no projector or retraining.

## Kaggle Inputs (six)
1. `hoangha071005/checkpoint-encoder-twins-codec-200k` — final codec encoder, FRE, train200K manifest.
2. `hoangha071005/evaluation-twins-bfree` — full-run V2 saved fit/calibration and B-Free clean manifests.
3. Competition `imagenet-object-localization-challenge` — accept rules, real pixels.
4. `awsaf49/coco-2017-dataset` — real pixels.
5. `yangsangtai/tiny-genimage` — Tiny test.
6. `baopv051/b-free-training-data` — B-Free test pixels.

No native checkpoint or fitted GMM input is needed. Select GPU T4×2. Source code is embedded; Internet not required if all Inputs are mounted. This notebook saves Working/Output only; it does not upload to an unspecified Dataset.

Calibration40K is held out from GMM fitting, but was seen during encoder SSL. B-Free clean has1557 source IDs,1557 real+9342 fake. It is a previously inspected regression benchmark.
''')
    add('code','''from pathlib import Path
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working/forensic_twins_codec_gmm')
CODE_ROOT = Path('/kaggle/working/forensic_twins_codec_gmm_code')
# Optional exact directory overrides; leave empty for verified default paths.
CODEC_ROOT = ''  # directory containing encoder_final.pt, fre.pt, train_real_200k.csv
SAVED_MANIFESTS = ''  # directory containing augmented_gmm_fit_base.csv, augmented_gmm_calibration_base.csv, bfree_valid_clean_eval.csv
BFREE_ROOT = ''
TINY_ROOT = ''
BATCH_PER_GPU = 32
NUM_WORKERS = 4
BOOTSTRAP_SAMPLES = 500
# To continue another session, copy/extract your previous complete Output tree
# into OUTPUT_ROOT before running. run_identity.json rejects incompatible caches.
''')
    add('code',f'''import base64, io, zipfile, subprocess, sys
CODE_ROOT.mkdir(parents=True,exist_ok=True)
SOURCE_ARCHIVE_B64 = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    archive.extractall(CODE_ROOT)
import torch, numpy, matplotlib
assert torch.cuda.is_available(), 'Enable Kaggle GPU'
print('torch:',torch.__version__,'GPUs:',torch.cuda.device_count())
common=['--input-root',str(INPUT_ROOT),'--output-root',str(OUTPUT_ROOT),'--batch-per-gpu',str(BATCH_PER_GPU),'--workers',str(NUM_WORKERS),'--bootstrap-samples',str(BOOTSTRAP_SAMPLES)]
for flag,value in [('--codec-root',CODEC_ROOT),('--saved-manifests',SAVED_MANIFESTS),('--bfree-root',BFREE_ROOT),('--tiny-root',TINY_ROOT)]:
    if value: common += [flag,value]
def run(stage):
    subprocess.run([sys.executable,str(CODE_ROOT/'evaluate_codec_gmm.py'),'--stage',stage,*common],check=True)
''')
    for stage,description in [('prepare','Verify input hashes, exact split160K/40K, full200K membership,1557 B-Free groups, source non-overlap and pixel paths. Missing/changed manifests stop; no rebuilding or silent fallback.'),('fit','Extract native center256 embeddings, fit Gaussian K1/full, calibrate Q90/Q95/Q99. Embedding memmaps save every1000 images. Saves Gaussian, thresholds and calibration histogram.'),('tiny','Evaluate native, fake-codec-matched, common JPEG95/80/65/50 and resize0.75. Native remains codec-confounded; report protocols separately.'),('bfree','Reuse clean manifest; evaluate native, JPEG95/80/65/50 and resize0.75. Per-variant balanced metrics, macro scores, group bootstrap CIs, predictions and figures. No threshold retuning.')]:
        add('markdown',f'## {stage}\n\n{description}')
        add('code',f"run('{stage}')\n")
    add('code','''from IPython.display import display, Image, FileLink
import json
thresholds=OUTPUT_ROOT/'artifacts/gmm/thresholds.json'
print('Locked thresholds:',json.loads(thresholds.read_text())['percentiles'])
plot=OUTPUT_ROOT/'artifacts/gmm/calibration_thresholds.png'
if plot.exists(): display(Image(filename=str(plot)))
for relative in ['tiny/protocol_comparison.csv','bfree/artifacts/metrics/protocol_comparison.csv','artifacts/gmm/real_k1_full.npz','artifacts/gmm/thresholds.json']:
    path=OUTPUT_ROOT/relative
    if path.exists(): display(FileLink(str(path)))
print('Save Version preserves all Working artifacts. Keep encoder+FRE+Gaussian+thresholds+run_identity together.')
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    path=BASE/'forensic_twins_codec_gmm_eval_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Built',path,'cells=',len(cells))
if __name__=='__main__': main()
