"""Build the self-contained CAPD Kaggle notebooks.

Embeds the GSD runtimes (unchanged) + capd_runtime.py into one file written with %%writefile,
so data loading / splits / augmentation / validation / test transforms are identical to GSD-aug.

Usage (from repo root):  python baselines/capd/build_capd_notebooks.py
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GSD = HERE.parent / 'clip_gsd_tiny_kaggle'
GSD_FILES = ('runtime.py', 'runtime_raw_all7.py', 'manifest_reuse.py', 'runtime_aug_all7.py')
PIP = ('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" '
       '"scikit-learn<1.9" "pillow<12" tqdm\n')


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.strip('\n').splitlines(keepends=True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=text.strip('\n').splitlines(keepends=True))


def notebook(cells):
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                           language_info=dict(name='python')), nbformat=4, nbformat_minor=5)


def runtime_source(bfree=False):
    names = GSD_FILES + (('bfree_data.py', 'bfree_train.py') if bfree else ())
    parts = [(GSD / name).read_text(encoding='utf-8') for name in names]
    parts.append((HERE / 'capd_runtime.py').read_text(encoding='utf-8'))
    if bfree:
        parts.append((HERE / 'capd_bfree.py').read_text(encoding='utf-8'))
    source = '\n\n'.join(parts)
    ast.parse(source)
    return source


LAUNCHER = r'''
import json, os, subprocess, sys, time
from pathlib import Path

def launch(runs, script='/kaggle/working/capd_all.py'):
    procs = []
    for run in runs:
        cfg = dict(run); gpu = cfg.pop('gpu'); job = cfg.pop('job', cfg['run_name'])
        path = Path(f"/kaggle/working/cfg_{job}.json")
        path.write_text(json.dumps(cfg, indent=2))
        logf = open(f"/kaggle/working/log_{job}.txt", 'a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        p = subprocess.Popen([sys.executable, script, str(path)],
                             env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((job, p))
        print(f"started {job} on GPU {gpu} (pid {p.pid}); log: /kaggle/working/log_{job}.txt")
    while any(p.poll() is None for _, p in procs):
        time.sleep(300)
        for name, p in procs:
            lines = Path(f'/kaggle/working/log_{name}.txt').read_text(errors='ignore').splitlines()
            print(f"--- {name} ({'running' if p.poll() is None else 'exit ' + str(p.returncode)}) ---")
            print('\n'.join(lines[-4:]), flush=True)
    for name, p in procs:
        lines = Path(f'/kaggle/working/log_{name}.txt').read_text(errors='ignore').splitlines()
        print(f"=== {name}: exit code {p.returncode} ===")
        print('\n'.join(lines[-25:]))
'''


PREFLIGHT = r"""
# Kiem tra input bang DUONG DAN CO DINH (nhanh). Chi quet /kaggle/input (rat cham khi co B-Free) neu duong dan mac dinh khong ton tai.
import torch
from pathlib import Path
IN = Path('/kaggle/input')
D = IN / 'datasets'
ok = True
def say(flag, msg):
    global ok
    ok &= bool(flag)
    print(('OK   ' if flag else 'LOI  ') + msg, flush=True)

def locate(default, pattern, test):
    if default is not None and test(Path(default)):
        return [Path(default)]
    print(f'NOTE khong thay {default} -> quet /kaggle/input tim {pattern} (co the mat vai phut)', flush=True)
    return sorted({h for p in IN.rglob(pattern) for h in [test(p)] if isinstance(h, Path)})

# 1) GPU
say(torch.cuda.device_count() >= len(RUNS), f'GPU: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"} (can {len(RUNS)})')

# 2) Manifest GSD (thu muc chua seed42/*_manifest.csv, khong phai output CAPD)
def is_manifest(p):
    root = p if p.is_dir() and (p / 'seed42').is_dir() else p.parent.parent
    sd = root / 'seed42'
    good = all((sd / f'{k}_manifest.csv').is_file() for k in ('train', 'validation', 'test')) and not (sd / 'architecture.json').is_file()
    return root if good else None
m = locate(MANIFEST_ROOT or D / 'hadinho0710/mainfest-train-genimage', 'train_manifest.csv', is_manifest)
MANIFEST_ROOT = str(m[0]) if len(m) == 1 else None
say(MANIFEST_ROOT, f'Manifest: {MANIFEST_ROOT}  ({[str(x) for x in m]})')

# 3) Tiny-GenImage (thu muc chua imagenet_*)
tiny = locate(D / 'yangsangtai/tiny-genimage', 'imagenet_ai_0424_sdv5',
              lambda p: (p if (p / 'imagenet_ai_0424_sdv5').is_dir() else p.parent if p.name == 'imagenet_ai_0424_sdv5' and p.is_dir() else None))
say(len(tiny) == 1, f'Tiny-GenImage: {[str(t) for t in tiny]}')

# 4) DINOv3
if any(r['backbone'] == 'dinov3' for r in RUNS):
    W = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    exact = locate(D / 'hadinho0710/dinov3-weight/dinov3' / W, W, lambda p: p if p.is_file() and p.name == W else None)
    say(len(exact) == 1, f'DINOv3 weight: {[str(p) for p in exact]}')
    if exact:
        say(1.0e9 < exact[0].stat().st_size < 1.4e9, f'DINOv3 size: {exact[0].stat().st_size / 1e9:.2f} GB (mong doi ~1.2)')
    repo = locate(D / 'hadinho0710/dinov3-weight/dinov3/dinov3_repo', 'backbones.py',
                  lambda p: (p if (p / 'dinov3/hub/backbones.py').is_file() else p.parent.parent.parent
                             if p.name == 'backbones.py' and p.parent.name == 'hub' and p.parent.parent.name == 'dinov3' else None))
    print(('OK   ' if repo else 'NOTE ') + f'DINOv3 repo: {[str(r) for r in repo]}' + ('' if repo else ' -> se git clone (can Internet ON)'))
    for r in RUNS:
        if r['backbone'] == 'dinov3':
            r['dinov3_weights'] = str(exact[0]) if exact else None
            r['dinov3_repo'] = str(repo[0]) if repo else None

# 5) B-Free (chi khi co job eval_bfree)
if any('eval_bfree' in r.get('modes', []) for r in RUNS):
    bf = locate(D / 'baopv051/b-free-training-data', 'valid_list.csv',
                lambda p: (p if (p / 'valid_list.csv').is_file() and (p / 'train_list.csv').is_file() else
                           p.parent if p.name == 'valid_list.csv' and (p.parent / 'train_list.csv').is_file() else None))
    say(len(bf) == 1, f'B-Free: {[str(b) for b in bf]}')
    for r in RUNS:
        if 'eval_bfree' in r.get('modes', []) and len(bf) == 1:
            r['bfree_root'] = str(bf[0])

# 6) Resume / checkpoint
if RESUME_FROM:
    say(Path(RESUME_FROM).exists(), f'RESUME_FROM: {RESUME_FROM}')
    for r in RUNS:
        if 'train' not in r.get('modes', []):
            b = Path(RESUME_FROM) / r['run_name'] / 'seed42' / 'best.pt'
            say(b.is_file(), f"best.pt cho {r.get('job', r['run_name'])}: {b}")
elif any('train' not in r.get('modes', []) for r in RUNS):
    say(False, 'Chua tim thay checkpoint: Add Input dataset checkpoint')

for r in RUNS:   # duong dan co dinh -> runtime khong quet /kaggle/input
    r['manifest_root'] = MANIFEST_ROOT
    r['dataset_root'] = str(tiny[0]) if len(tiny) == 1 else None
print('\nTAT CA OK -> chay cell tiep theo' if ok else '\nCO LOI -> sua cau hinh/Input truoc khi chay tiep')
assert ok, 'Preflight failed'
"""

SUMMARY = r'''
import json, pandas as pd
from pathlib import Path
for run in RUNS:
    folder = Path(run['output_root']) / run['run_name'] / 'seed42'
    print('=' * 20, run.get('job', run['run_name']), '=' * 20)
    for name in ('status.json', 'selected_checkpoint.json', 'completion.json'):
        if (folder / name).exists():
            print(name, json.loads((folder / name).read_text()))
    if (folder / 'validation_metrics.csv').exists():
        v = pd.read_csv(folder / 'validation_metrics.csv')
        print(v[['epoch', 'view', 'monitor_loss', 'auc', 'balanced_accuracy', 'fake_recall', 'real_recall']].to_string(index=False))
    for f in (folder / 'tiny_eval' / 'metrics.csv', folder / 'bfree_eval' / 'metrics_bfree.csv'):
        if f.exists():
            m = pd.read_csv(f)
            key = 'generator' if 'generator' in m else 'branch'
            sel = m[m[key].isin(['pooled', 'macro'])] if key == 'generator' else m
            cols = [c for c in ('case', 'branch', 'pooling', 'auc', 'balanced_accuracy', 'fake_recall', 'real_recall', 'tpr_at_fpr5') if c in m]
            print(f.name); print(sel[cols].to_string(index=False))
'''


NOTEBOOK_VERSION = '2026-10-31 17:50 (TN2 B-Free DINOv3 only)'
BANNER = f'print("=" * 60)\nprint("Nguyen Hoang Ha")\nprint("CAPD notebook version: {NOTEBOOK_VERSION}")\nprint("=" * 60)\n'


def train_notebook(source):
    return notebook([
        md('''
# CAPD — TN1: train trên Tiny-GenImage, giao thức GIỐNG HỆT GSD-aug

Notebook chạy **2 model song song trên 2 GPU T4**:
- **GPU 0 → R1 `r1_clip`**: CLIP ViT-L/14 + LoRA + PatchHead + LSE (cùng backbone với GSD → so sánh công bằng nhất).
- **GPU 1 → R2 `r2_dinov3`**: DINOv3 ViT-L/16 + LoRA + PatchHead + LSE (model đề xuất chính).

Dữ liệu, split, augmentation (blur/JPEG p=0.5), crop ≤560 + pad, validation 2 view, early stopping
(patience 3) và các case test **dùng nguyên code của GSD-aug** (được nhúng bên dưới).

## Input cần Add (cột phải → Add Input)
1. `yangsangtai/tiny-genimage`
2. Dataset manifest GSD (`hadinho0710/mainfest-train-genimage`, có `seed42/train|validation|test_manifest.csv`)
3. Dataset DINOv3 private của bạn (file `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, nếu có thì cả thư mục `dinov3_repo`)
4. (Chỉ khi chạy tiếp phiên trước) Output của phiên trước → điền `RESUME_FROM`

## Settings
Accelerator **GPU T4 x2**, **Internet ON** (tải CLIP từ Hugging Face; DINOv3 repo nếu chưa có).

## Cách chạy
1. Lần đầu: để `SMOKE = True` → **Run All** (~10–20 phút). Kiểm tra log không lỗi, có dòng `[exit] complete`.
2. Sau đó: `SMOKE = False` → **Save Version → Save & Run All (Commit)**.
3. Nếu hết 12h (log ghi `paused_time_budget`): tạo phiên mới, Add Input = output phiên trước,
   điền `RESUME_FROM`, chạy lại. Train tiếp đúng bước đã dừng.
'''),
        code(BANNER),
        code(PIP),
        md('## Runtime (GSD runtimes + CAPD) — không cần sửa'),
        code('%%writefile /kaggle/working/capd_all.py\n' + source),
        md('## Cấu hình — CHỈ SỬA Ở ĐÂY'),
        code('''
SMOKE = True                      # True: chạy thử nhỏ; False: train thật
RESUME_FROM = None                # vd '/kaggle/input/<output-phien-truoc>/capd_tn1' khi chạy tiếp
MANIFEST_ROOT = '/kaggle/input/datasets/hadinho0710/mainfest-train-genimage'   # None = tự tìm
OUTPUT_ROOT = '/kaggle/working/capd_tn1_smoke' if SMOKE else '/kaggle/working/capd_tn1'

COMMON = dict(output_root=OUTPUT_ROOT, resume_from=RESUME_FROM, manifest_root=MANIFEST_ROOT,
              smoke=SMOKE, modes=['train'], eval_after_train=['native'], time_budget_hours=11.0)
RUNS = [
    dict(COMMON, gpu=0, run_name='r1_clip', backbone='clip'),
    dict(COMMON, gpu=1, run_name='r2_dinov3', backbone='dinov3'),
]
# Muốn chạy 1 model: xóa dòng tương ứng trong RUNS.
'''),
        md('## Kiểm tra đường dẫn & input (preflight) — phải in TAT CA OK'),
        code(PREFLIGHT),
        code(LAUNCHER + '\nlaunch(RUNS)\n'),
        md('## Tóm tắt kết quả'),
        code(SUMMARY),
    ])


def eval_notebook(source):
    return notebook([
        md('''
# CAPD — TN1: đánh giá Tiny 10 case + B-Free valid (6 nhánh)

Dùng `best.pt` của 1 model (`EVAL_RUN`); Tiny và B-Free chạy song song trên 2 GPU.
- Tiny: native, png_roundtrip, fake_jpeg96, all_jpeg96/90/80, resize_square256/560, blur_sigma1, resize256_jpeg96
  (đúng các case của evaluator GSD).
- B-Free valid: 10.304 ID × (1 real + 6 nhánh fake), ảnh nguyên 512, không crop/resize;
  điểm `lse_full` (giống train) và `lse_window` (cửa sổ 224 px) đều được lưu, kèm bản đồ điểm patch.

## Input cần Add
1. `yangsangtai/tiny-genimage`, dataset manifest GSD
2. `baopv051/b-free-training-data`
3. Dataset DINOv3 private
4. **Dataset checkpoint** (vd `hadinho0710/checkpoin-capd-train-tiny`, chứa `capd_tn1/r2_dinov3/seed42/best.pt`). Notebook tự tìm.

Settings: GPU T4 x2, Internet ON. Chọn model bằng `EVAL_RUN`. GPU 0 test Tiny, GPU 1 test B-Free cùng lúc.
Chỉ copy `best.pt` (bỏ qua epoch_*.pt / last_state.pt); checkpoint cũ 1,2 GB được tự cắt còn LoRA + head.
'''),
        code(BANNER),
        code(PIP),
        md('## Runtime — không cần sửa'),
        code('%%writefile /kaggle/working/capd_all.py\n' + source),
        md('## Cấu hình — CHỈ SỬA Ở ĐÂY'),
        code('''
SMOKE = False
EVAL_RUN = 'r2_dinov3'            # 'r2_dinov3' / 'r1_clip' (TN1) hoặc 'b2_dinov3' / 'b1_clip' (TN2, điền TRAIN_OUTPUT = .../capd_tn2)
BACKBONE = {'r1_clip': 'clip', 'r2_dinov3': 'dinov3', 'b1_clip': 'clip', 'b2_dinov3': 'dinov3'}[EVAL_RUN]  # r*: TN1 (Tiny), b*: TN2 (B-Free)
TRAIN_OUTPUT = None               # None = tự tìm thư mục chứa <EVAL_RUN>/seed42/best.pt trong /kaggle/input
MANIFEST_ROOT = '/kaggle/input/datasets/hadinho0710/mainfest-train-genimage'   # None = tự tìm

from pathlib import Path
_ck = Path('/kaggle/input/datasets/hadinho0710/checkpoin-capd-train-tiny/capd_tn1')
if TRAIN_OUTPUT is None and (_ck / EVAL_RUN / 'seed42' / 'best.pt').is_file():
    TRAIN_OUTPUT = str(_ck)
if TRAIN_OUTPUT is None:
    hits = sorted({p.parent.parent.parent for p in Path('/kaggle/input').rglob('best.pt')
                   if p.parent.name == 'seed42' and p.parent.parent.name == EVAL_RUN})
    TRAIN_OUTPUT = str(hits[0]) if len(hits) == 1 else None
    print('best.pt tìm thấy:', [str(h) for h in hits])
RESUME_FROM = TRAIN_OUTPUT

# 1 model, 2 GPU: GPU 0 test Tiny 10 case, GPU 1 test B-Free valid (thư mục output riêng, không ghi đè nhau)
COMMON = dict(resume_from=TRAIN_OUTPUT, manifest_root=MANIFEST_ROOT, smoke=SMOKE,
              run_name=EVAL_RUN, backbone=BACKBONE, time_budget_hours=11.0)
RUNS = [
    dict(COMMON, gpu=0, job=f'{EVAL_RUN}_tiny',  modes=['eval_tiny'],  output_root='/kaggle/working/capd_eval_tiny'),
    dict(COMMON, gpu=1, job=f'{EVAL_RUN}_bfree', modes=['eval_bfree'], output_root='/kaggle/working/capd_eval_bfree'),
]
# Chạy tiếp khi hết giờ: Add output phiên này làm input, đặt resume_from của từng job =
#   '/kaggle/input/<output>/capd_eval_tiny' và '/kaggle/input/<output>/capd_eval_bfree'.
'''),
        md('## Kiểm tra đường dẫn & input (preflight) — phải in TAT CA OK'),
        code(PREFLIGHT),
        code(LAUNCHER + '\nlaunch(RUNS)\n'),
        md('## Tóm tắt kết quả'),
        code(SUMMARY),
    ])


def dinov3_notebook():
    return notebook([
        md('''
# Tải DINOv3 ViT-L/16 → tạo Kaggle Dataset PRIVATE

1. Settings: **Internet ON** (không cần GPU).
2. Dán link file **`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`** từ email Meta vào `URL` (giữ trong dấu nháy).
3. Run All. Kiểm tra: dung lượng ~1.2 GB, model load được, in ra `patch tokens (1, 196, 1024)`.
4. Save Version → mở tab **Output** → **New Dataset** → đặt tên `dinov3-vitl16-lvd1689m` → **Private** → Create.

Không chia sẻ link hay dataset công khai (DINOv3 License). Link trong email hết hạn sau vài ngày.
'''),
        code('''
URL = "DAN_LINK_VAO_DAY"   # link của file dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
'''),
        code('''
import os, hashlib, subprocess
from pathlib import Path
out = Path('/kaggle/working/dinov3'); out.mkdir(parents=True, exist_ok=True)
pth = out / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
if not pth.exists():
    subprocess.run(['wget', '-q', '-O', str(pth), URL], check=True)
print('size GB:', round(pth.stat().st_size / 1e9, 3))
sha = hashlib.sha256(pth.read_bytes()).hexdigest()
print('sha256:', sha, '| prefix 8aa4cbdd:', sha.startswith('8aa4cbdd'))
repo = out / 'dinov3_repo'
if not repo.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/dinov3', str(repo)], check=True)
print(sorted(os.listdir(out)))
'''),
        code('''
import sys, torch
sys.path.insert(0, str(repo))
from dinov3.hub.backbones import dinov3_vitl16
m = dinov3_vitl16(pretrained=False)
m.load_state_dict(torch.load(pth, map_location='cpu', weights_only=True), strict=True)
m.eval()
with torch.no_grad():
    f = m.forward_features(torch.randn(1, 3, 224, 224))
print({k: tuple(v.shape) for k, v in f.items() if hasattr(v, 'shape')})
print('patch tokens', tuple(f['x_norm_patchtokens'].shape), '| storage tokens', m.n_storage_tokens)
print('OK -> Save Version, rồi tạo Dataset PRIVATE từ Output')
'''),
    ])


TN2_PREFLIGHT = r"""
# Kiem tra input bang DUONG DAN CO DINH (khong quet /kaggle/input).
import torch, json
from pathlib import Path
D = Path('/kaggle/input/datasets')
ok = True
def say(flag, msg):
    global ok
    ok &= bool(flag)
    print(('OK   ' if flag else 'LOI  ') + msg, flush=True)

say(torch.cuda.device_count() >= len(RUNS), f'GPU: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"} (can {len(RUNS)})')
bf = Path(BFREE_ROOT)
say((bf / 'train_list.csv').is_file() and (bf / 'valid_list.csv').is_file() and (bf / 'COCO_real_512').is_dir(), f'B-Free: {bf}')
say(any((bf / n / n).is_dir() or (bf / n).is_dir() for n in ['SD2.1_inpainted_samecat_origBG']), 'B-Free co nhanh SD2.1_inpainted_samecat_origBG')
if any(r['backbone'] == 'dinov3' for r in RUNS):
    w, repo = Path(DINOV3_WEIGHTS), Path(DINOV3_REPO)
    say(w.is_file() and 1.0e9 < w.stat().st_size < 1.4e9, f'DINOv3 weight: {w}')
    say((repo / 'dinov3/hub/backbones.py').is_file(), f'DINOv3 repo: {repo}')
g = Path(GSD_OUTPUT) if GSD_OUTPUT else None
if g and (g / 'audit_train.csv').is_file() and (g / 'split_summary.json').is_file():
    s = json.loads((g / 'split_summary.json').read_text())
    print(f"OK   GSD-P2 audit: {g}  (eligible={s['eligible']}, val={s['validation']}, VE={s['virtual_epochs']}) -> bo qua audit ~40 phut, split trung GSD")
else:
    print(f'NOTE Khong thay output GSD-P2 ({g}) -> se tu audit (~40 phut); split van giong GSD (cung ham, cung seed)')
    for r in RUNS:
        r['gsd_output_dir'] = None
if RESUME_FROM:
    for r in RUNS:
        st = Path(RESUME_FROM) / r['run_name'] / 'seed42' / 'last_state.pt'
        say(st.is_file(), f"resume {r['run_name']}: {st}")
print('\nTAT CA OK -> chay cell tiep theo' if ok else '\nCO LOI -> sua cau hinh/Input truoc khi chay tiep')
assert ok, 'Preflight failed'
"""

TN2_SUMMARY = r"""
import json, pandas as pd
from pathlib import Path
for run in RUNS:
    folder = Path(run['output_root'] + ('_smoke' if run.get('smoke') else '')) / run['run_name'] / 'seed42'
    print('=' * 20, run['run_name'], folder, '=' * 20)
    for name in ('status.json', 'split_vs_gsd.json', 'selected_checkpoint.json', 'completion.json'):
        if (folder / name).exists():
            print(name, json.loads((folder / name).read_text()))
    if (folder / 'validation_metrics.csv').exists():
        v = pd.read_csv(folder / 'validation_metrics.csv')
        cols = [c for c in ['ve', 'view', 'monitor_loss', 'auc', 'balanced_accuracy', 'fake_recall', 'real_recall',
                            'area_small_fake_recall', 'area_medium_fake_recall', 'area_large_fake_recall',
                            'inner_balanced_accuracy'] if c in v]
        print(v[cols].round(4).to_string(index=False))
"""


def tn2_notebook(source):
    return notebook([
        md('''
# CAPD — TN2: train trên B-Free `SD2.1_inpainted_samecat_origBG` (giao thức GIỐNG HỆT GSD-P2)

Run **`b2_dinov3`** (CAPD-DINOv3) trên GPU 0. Ảnh B-Free PNG 512×512 đưa nguyên vào model:
không resize, không crop, không pad (512 = 32×16 patch). Train có blur/JPEG p=0,5 như GSD-P2.

Dữ liệu / audit mask / split 2.000 ID val / virtual epoch 12.600 ID / batch 8 = 4 cặp / blur-JPEG chung cặp /
val_loss / early stopping (patience 2) = **đúng code `bfree_data.py` + `bfree_train.py` của GSD-P2** (nhúng bên dưới).
Khác: model CAPD + loss L_img + 0,5·L_neg (+ L_gap warm-up). Không dùng L_pix.

## Input cần Add
1. `baopv051/b-free-training-data`
2. Dataset DINOv3 private (`hadinho0710/dinov3-weight`)
3. Output GSD-P2 `hoangha071005/check-point-gsd-train-bfee-samecat-ori` (để dùng lại audit → split trùng GSD, tiết kiệm ~40 phút)

Settings: **GPU T4** (x1 hoặc x2 đều được), **Internet ON**.
Chạy: `SMOKE=True` → Run All (~15 phút, kiểm tra log có `[exit] complete`) → `SMOKE=False` → Save & Run All (Commit).
Hết giờ (`paused_time_budget`): Add output phiên này làm Input, điền `RESUME_FROM`, chạy lại.
'''),
        code(BANNER),
        code(PIP),
        md('## Runtime (GSD runtimes + GSD-P2 bfree_data/bfree_train + CAPD) — không cần sửa'),
        code('%%writefile /kaggle/working/capd_bfree_all.py\n' + source),
        md('## Cấu hình — CHỈ SỬA Ở ĐÂY'),
        code('''
SMOKE = True                      # True: chạy thử 64 ID; False: train thật
RESUME_FROM = None                # vd '/kaggle/input/<output-phien-truoc>/capd_tn2' khi chạy tiếp
BFREE_ROOT = '/kaggle/input/datasets/baopv051/b-free-training-data'
DINOV3_WEIGHTS = '/kaggle/input/datasets/hadinho0710/dinov3-weight/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3_REPO = '/kaggle/input/datasets/hadinho0710/dinov3-weight/dinov3/dinov3_repo'
GSD_OUTPUT = '/kaggle/input/datasets/hoangha071005/check-point-gsd-train-bfee-samecat-ori/gsd_bfree_samecat_origbg_v1'

COMMON = dict(task='bfree_train', output_root='/kaggle/working/capd_tn2', resume_from=RESUME_FROM, smoke=SMOKE,
              bfree_root=BFREE_ROOT, gsd_output_dir=GSD_OUTPUT, dinov3_weights=DINOV3_WEIGHTS, dinov3_repo=DINOV3_REPO,
              time_budget_hours=10.8)
RUNS = [
    dict(COMMON, gpu=0, run_name='b2_dinov3', backbone='dinov3'),
]
# (Tùy chọn sau này) CAPD-CLIP trên GPU 1: dict(COMMON, gpu=1, run_name='b1_clip', backbone='clip')
'''),
        md('## Kiểm tra input (preflight) — phải in TAT CA OK'),
        code(TN2_PREFLIGHT),
        code(LAUNCHER + "\nlaunch(RUNS, '/kaggle/working/capd_bfree_all.py')\n"),
        md('## Tóm tắt kết quả validation'),
        code(TN2_SUMMARY),
    ])


def main():
    source = runtime_source()
    for name, nb in [('train_capd_tn1_kaggle.ipynb', train_notebook(source)),
                     ('eval_capd_tn1_kaggle.ipynb', eval_notebook(source)),
                     ('dinov3_download_kaggle.ipynb', dinov3_notebook()),
                     ('train_capd_bfree_kaggle.ipynb', tn2_notebook(runtime_source(bfree=True)))]:
        (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
        print('wrote', HERE / name)


if __name__ == '__main__':
    main()
