"""Build the B-Free samecat_origBG training notebook + two evaluation notebooks.

The evaluation notebooks are COPIES of the existing, already-run evaluators
(test_gsd_uploaded_best_variants_kaggle.ipynb, test_gsd_uploaded_best_bfree_valid_kaggle.ipynb)
with one inserted cell that lets them accept the B-Free-trained checkpoint, and a new output folder.
Inference code, cohorts, batch protocol and metrics stay byte-identical -> directly comparable.
"""
import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_NB = 'train_gsd_bfree_samecat_origbg_kaggle.ipynb'
TINY_NB = 'test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb'
BFREE_NB = 'test_gsd_bfree_samecat_origbg_bfree_valid_kaggle.ipynb'
RUNTIME_FILES = ('runtime.py', 'runtime_raw_all7.py', 'manifest_reuse.py', 'runtime_aug_all7.py',
                 'bfree_data.py', 'bfree_train.py')
PIP = ('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" '
       '"scikit-learn<1.9" "pillow<12" matplotlib tqdm\n')


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.strip('\n').splitlines(keepends=True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=text.strip('\n').splitlines(keepends=True))


def notebook(cells):
    return dict(cells=cells, nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'),
        kaggle=dict(accelerator='gpu', isInternetEnabled=True, isGpuEnabled=True)))


TRAIN_INTRO = '''
# GSD — train trên B-Free `SD2.1_inpainted_samecat_origBG` (kế hoạch P2)

**Input duy nhất:** [baopv051/b-free-training-data](https://www.kaggle.com/datasets/baopv051/b-free-training-data).
**Bật:** GPU (T4 hoặc P100, chỉ dùng 1 GPU) và **Internet** (tải CLIP ViT-L/14 pretrained).
Rồi bấm **Save Version → Save & Run All**. Notebook tự chứa toàn bộ code.

## Notebook làm gì
1. **Audit** (~10–15 phút, CPU): đọc `train_list.csv`/`valid_list.csv` (kiểm MD5 chính thức), lấy
   `train_list − valid_list` = 41.213 COCO ID. Giải mã từng cặp `COCO_real_512` / `samecat_origBG`,
   tính mask vùng inpaint = `max kênh |fake − real| > 8`. Loại cặp lỗi/thiếu, mask < 0,1% hoặc > 95%.
   Lưu `audit_train.csv`, `audit_summary.json`; mask của 10.304 ID valid lưu riêng để phân tích sau.
2. **Chia:** 2.000 ID val (phân tầng theo 3 nhóm diện tích mask), phần còn lại xáo 1 lần cố định.
   **Virtual epoch (VE) = 12.600 ID = 25.200 ảnh** (đúng cỡ 1 epoch GSD-aug-Tiny), tối đa 3 VE, không lặp ID.
   **Không** dùng `valid_list` và **không** dùng 5 nhánh B-Free còn lại để chọn checkpoint.
3. **Train:** batch 8 = 4 COCO ID × (real, fake) → basis GSD tính trên cặp cùng nội dung.
   Blur (p0.5, σ∈[0,2]) → JPEG (p0.5, Q75–100, 4:4:4) **dùng chung tham số cho cả cặp**, đổi mỗi VE.
   Không flip/crop/resize. Model, GSD (4 block cuối, k=64), AdamW lr 1e-6, fp16, crop560/pad14:
   **giống hệt GSD-aug-Tiny**, khởi tạo từ CLIP pretrained.
4. **Sau mỗi VE:** validate 2 view (native + processed cố định), `val_loss = 0.5·CE_native + 0.5·CE_processed`,
   early stopping patience 2. Lưu `checkpoint_veN.pt`; `best.pt` = VE có val_loss thấp nhất.
   Mỗi checkpoint được nạp lại bằng `weights_only=True` và kiểm tra đúng định dạng evaluator cũ.

## Thời gian (ước lượng từ tốc độ GSD-aug-Tiny/B-Free đã đo, chưa đo trực tiếp)
- 1 VE ≈ 4 giờ train + ≈ 0,5 giờ validate trên T4. Một phiên Kaggle 12 giờ ≈ audit + 2 VE.
- `time_budget_hours=11`: gần hết giờ, notebook **tự lưu `last.pt` và dừng sạch** (status `paused_time_budget`).

## Nếu bị dừng vì hết giờ (resume)
1. Mở notebook → **Add Input → Your Work/Notebook Output** → chọn output của lần chạy trước.
2. Giữ nguyên config, **Save & Run All** lại. Code tự tìm `run_state.json` cùng cấu hình, chép sang
   `/kaggle/working`, tiếp tục **đúng bước** đã dừng (kể cả giữa VE), không audit lại.
3. Khi `status = complete`, `last.pt` (trạng thái optimizer ~3,6 GB) được xóa cho Output gọn; giữ `best.pt`.

## Output (`/kaggle/working/gsd_bfree_samecat_origbg_v1/`)
`best.pt` · `checkpoint_ve1.pt` … · `history.json` · `train_log.csv` · `validation_predictions_veN.csv`
(xác suất real/fake + nhóm diện tích) · `audit_*.csv/json` · `split_summary.json` · `run_state.json` ·
`selected_checkpoint.json` · `completion.json` (khi xong).

## So sánh với GSD-aug-Tiny
Dùng 2 notebook đi kèm (cùng code evaluator cũ, chỉ thêm một cell nhận checkpoint B-Free):
`test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb` (Tiny 9 case, 7.000 ảnh/case) và
`test_gsd_bfree_samecat_origbg_bfree_valid_kaggle.ipynb` (B-Free valid 10.304 ID × 6 nhánh).
`checkpoint_ve1.pt` (25.200 ảnh) = so sánh cùng lượng dữ liệu với Tiny (câu hỏi P1).
'''

TRAIN_CONFIG = '''
SMOKE = False   # True = thử nhanh 64 ID, 2 VE × 2 bước, output riêng *_smoke (KHÔNG phải kết quả thật)
CONFIG = dict(BFREE_TRAIN_DEFAULTS)
CONFIG.update(
    input_root='/kaggle/input',
    bfree_root=None,          # auto: folder có train_list.csv + valid_list.csv + COCO_real_512
    output_root='/kaggle/working/gsd_bfree_samecat_origbg_v1',
    resume_from='auto',       # auto tìm Output lần trước (cùng cấu hình) đã Add Input
    train_branch='SD2.1_inpainted_samecat_origBG',
    time_budget_hours=11.0,
    smoke=SMOKE,
)
# Chỉ điền khi auto báo lỗi không tìm thấy / nhiều nguồn:
# CONFIG['bfree_root'] = '/kaggle/input/datasets/baopv051/b-free-training-data'
# CONFIG['resume_from'] = '/kaggle/input/<output-lần-trước>/gsd_bfree_samecat_origbg_v1'
# KHÔNG đổi các tham số model/augmentation: chúng phải giống GSD-aug-Tiny để so sánh được.
print(json.dumps(CONFIG, indent=2))
'''

TRAIN_SUMMARY = '''
out = Path(CONFIG['output_root'] + ('_smoke' if SMOKE else ''))
print('STATUS:', json.loads((out / 'run_state.json').read_text())['status'])
print(json.dumps(json.loads((out / 'split_summary.json').read_text()), indent=1))
history = json.loads((out / 'history.json').read_text()) if (out / 'history.json').is_file() else []
rows = []
for e in history:
    for view in ('native', 'processed'):
        v = e['val_' + view]
        rows.append(dict(VE=e['virtual_epoch'], images_seen=e['images_seen'], view=view, val_loss=e['val_loss'],
                         loss=v['loss'], BA=v['balanced_accuracy'], AUC=v['auc'], fake_recall=v['fake_recall'],
                         real_recall=v['real_recall'],
                         **{f'fake_recall_{g}': v['by_area'].get(g, {}).get('fake_recall') for g in ('small', 'medium', 'large')}))
if rows:
    display(pd.DataFrame(rows))
if (out / 'train_log.csv').is_file():
    log = pd.read_csv(out / 'train_log.csv')
    display(log.tail(10))
print(json.dumps(json.loads((out / 'selected_checkpoint.json').read_text()), indent=1))
print('Files:', sorted(p.name for p in out.iterdir()))
'''

COMPAT_CELL = '''
# === Nhận checkpoint GSD train trên B-Free (gsd_bfree_single_branch_paired_v1) ===============
# Evaluator bên trên là bản GỐC đã dùng cho GSD-aug-Tiny. Cell này chỉ đổi bước kiểm tra checkpoint:
# đọc config['eval_compat_config'] (do notebook train ghi), kiểm tra mọi khóa inference trùng với
# config train, rồi gọi lại đúng validate_uploaded_payload gốc. Không đổi preprocessing, batch, metric.
# train_generators trong compat config chỉ mô tả layout manifest TEST Tiny, không phải dữ liệu train.
BFREE_TRAINED_PROTOCOL = 'gsd_bfree_single_branch_paired_v1'
BFREE_INFERENCE_KEYS = ('model_id', 'input_mode', 'crop_max_side', 'test_batch', 'micro_batch',
                        'anchor_micro_batch', 'precision', 'basis_method', 'attention', 'pooling', 'k',
                        'lambda_max', 'a_init', 'b_init', 'jpeg_subsampling', 'threshold')
if '_tiny_validate_uploaded_payload' not in globals():
    _tiny_validate_uploaded_payload = validate_uploaded_payload


def validate_uploaded_payload(payload, options):
    if not isinstance(payload, dict) or not isinstance(payload.get('config'), dict):
        raise ValueError('Checkpoint must contain model/config/epoch/signature')
    cfg = payload['config']
    if cfg.get('training_protocol') != BFREE_TRAINED_PROTOCOL:
        raise ValueError('Notebook này chỉ chấm checkpoint GSD train trên B-Free (best.pt / checkpoint_veN.pt)')
    compat = cfg.get('eval_compat_config')
    if not isinstance(compat, dict):
        raise ValueError('Checkpoint thiếu eval_compat_config')
    for key in BFREE_INFERENCE_KEYS:
        if compat.get(key) != cfg.get(key):
            raise ValueError(f'eval_compat_config khác config train ở khóa {key}')
    print(f"[compat] B-Free-trained checkpoint: branch={cfg['train_branch']}, "
          f"VE{payload['epoch'] + 1}, images_seen={payload.get('images_seen')}", flush=True)
    return _tiny_validate_uploaded_payload(dict(payload, config=compat), options)
'''

TINY_INTRO = '''
# GSD train trên B-Free `samecat_origBG` → test **Tiny-GenImage 9 case** (so với GSD-aug-Tiny)

Bản sao của `test_gsd_uploaded_best_variants_kaggle.ipynb` (evaluator đã dùng cho GSD-aug-Tiny):
**cùng 7.000 ảnh test, cùng thứ tự/batch8, cùng 9 case, threshold 0.5** → số liệu so trực tiếp được.
Khác biệt duy nhất: một cell nhận checkpoint B-Free và output riêng `gsd_bfree_samecat_origbg_tiny9_v1`.

**Input:** (1) Output notebook train B-Free (chứa `best.pt`; muốn chấm VE1 thì đặt `checkpoint_path` tới
`checkpoint_ve1.pt`); (2) `yangsangtai/tiny-genimage`; (3) manifest `hadinho0710/mainfest-train-genimage`
(hoặc gói `gsd_all7_manifests_seed42.zip`). **Không** gắn kèm checkpoint GSD-aug-Tiny cũ. GPU + Internet.
Hướng dẫn chi tiết của evaluator gốc ở cell markdown kế tiếp (các ý về signature epoch1 không áp dụng ở đây).
'''

BFREE_INTRO = '''
# GSD train trên B-Free `samecat_origBG` → test **B-Free valid 10.304 ID × 6 nhánh**

Bản sao của `test_gsd_uploaded_best_bfree_valid_kaggle.ipynb` (đã cho GSD-aug-Tiny macro BA 81,1):
**cùng 10.304 ID valid, cùng thứ tự, batch8 = 4 cặp, tính lại real mỗi nhánh, threshold 0.5.**
Khác biệt: một cell nhận checkpoint B-Free, output riêng `gsd_bfree_samecat_origbg_bfree_valid_v1`.

Diễn giải: `SD2.1_inpainted_samecat_origBG` giờ là **in-domain** (cùng nhánh đã train, ID valid chưa từng train);
5 nhánh còn lại là **held-out theo cách tạo ảnh**. Không báo B-Free valid là cross-dataset nữa.

**Input:** (1) Output notebook train B-Free; (2) `baopv051/b-free-training-data`. Nếu không tự tìm được
checkpoint (output gắn ở thư mục sâu), đặt `checkpoint_path` tới `.../gsd_bfree_samecat_origbg_v1/best.pt`.
Mask diện tích cho 10.304 ID valid nằm ở `audit_valid.csv` của output train → ghép để báo recall theo diện tích.
'''


def build_train():
    source = '\n\n'.join((HERE / name).read_text(encoding='utf-8') for name in RUNTIME_FILES)
    ast.parse(source)
    cells = [md(TRAIN_INTRO), code(PIP), md('## Runtime (tự chứa: runtime GSD-aug-Tiny + lớp dữ liệu/train B-Free)'),
             code(source), md('## Cấu hình'), code(TRAIN_CONFIG),
             md('## Train (tự resume; tự dừng sạch trước giới hạn 12 giờ)'),
             code('report = run_bfree_training(CONFIG)'), md('## Tóm tắt'), code(TRAIN_SUMMARY)]
    return notebook(cells)


def derive_eval(template, intro, output_root, config_var):
    nb = json.loads((HERE / template).read_text(encoding='utf-8'))
    cells = nb['cells']
    runtime_index = max(range(len(cells)), key=lambda i: len(''.join(cells[i]['source'])))
    config_index = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code'
                        and f'{config_var}.update(' in ''.join(c['source']))
    if config_index != runtime_index + 1:
        raise ValueError(f'Unexpected layout in {template}')
    text = ''.join(cells[config_index]['source'])
    old = f'{config_var}.update(\n'
    if text.count(old) != 1:
        raise ValueError('config cell layout changed')
    text = text.replace(old, old + "    expected_signature=None,   # checkpoint B-Free: signature mới, kiểm tra qua compat cell\n"
                                   "    expected_epoch=None,       # best.pt có thể là VE1/2/3\n", 1)
    import re
    text, n = re.subn(r"output_root='[^']*'", f"output_root='{output_root}'", text, count=1)
    if n != 1:
        raise ValueError('output_root not found in config cell')
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    new_cells = ([md(intro)] + cells[:runtime_index + 1] +
                 [md('## Nhận checkpoint train trên B-Free'), code(COMPAT_CELL), code(text)] + cells[config_index + 1:])
    ast.parse(''.join(new_cells[runtime_index + 3]['source']))
    nb = copy.deepcopy(nb)
    nb['cells'] = new_cells
    return nb


def build_all():
    targets = {
        TRAIN_NB: build_train(),
        TINY_NB: derive_eval('test_gsd_uploaded_best_variants_kaggle.ipynb', TINY_INTRO,
                             '/kaggle/working/gsd_bfree_samecat_origbg_tiny9_v1', 'EVAL_CONFIG'),
        BFREE_NB: derive_eval('test_gsd_uploaded_best_bfree_valid_kaggle.ipynb', BFREE_INTRO,
                              '/kaggle/working/gsd_bfree_samecat_origbg_bfree_valid_v1', 'BFREE_CONFIG'),
    }
    for name, nb in targets.items():
        (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print('wrote', name, sum(len(''.join(c['source'])) for c in nb['cells']), 'chars')
    return targets


if __name__ == '__main__':
    build_all()


# ============================================================================ combined test notebook
COMBINED_NB = 'test_gsd_bfree_samecat_origbg_tiny9_and_bfree_kaggle.ipynb'

COMBINED_INTRO = '''
# Test checkpoint GSD train trên B-Free `samecat_origBG`: **B-Free valid 6 nhánh + Tiny-GenImage 9 case**

Một notebook chạy **cả hai bộ test bằng đúng evaluator đã dùng cho GSD-aug-Tiny** (runtime giữ nguyên
byte-for-byte, chỉ thêm cell nhận checkpoint B-Free), cuối notebook tự in **bảng so sánh với GSD-aug-Tiny**.

## Input cần Add (4 dataset) + GPU T4 + Internet
1. `hoangha071005/check-point-gsd-train-bfee-samecat-ori` (checkpoint; mặc định đọc `best.pt` = VE2).
2. `baopv051/b-free-training-data`.
3. `yangsangtai/tiny-genimage` (đúng phiên bản đã dùng).
4. `hadinho0710/mainfest-train-genimage` (cần `seed42/test_manifest.csv`, 7.000 ảnh test).

## Giao thức (giống hệt baseline)
- **B-Free:** 10.304 ID `valid_list`, 6 nhánh, mỗi nhánh 10.304 real + 10.304 fake, batch8 = 4 cặp cùng ID,
  native PNG 512 → pad 518, threshold 0.5. `samecat_origBG` = **in-domain** (cùng nhánh train, ID chưa thấy);
  5 nhánh còn lại = **held-out theo cách tạo ảnh**.
- **Tiny:** 7.000 ảnh test, 9 case (native, fake_jpeg96, all_jpeg96/90/80, resize256/560, blur1, resize256_jpeg96).
  Với checkpoint này Tiny là **cross-dataset** (GSD-aug-Tiny thì train trên Tiny) → so sánh để đo mức tổng quát.

## Thời gian và resume
- Ước lượng: B-Free ~7–8 giờ, Tiny ~2,5–3 giờ trên T4 → có thể cần **2 phiên**.
- Chạy B-Free trước, Tiny sau, chung ngân sách `TOTAL_BUDGET_HOURS=11.3`. Hết giờ thì mỗi evaluator tự lưu
  tiến độ theo batch và dừng sạch.
- Phiên sau: **Add Input → Output của phiên này**, giữ nguyên config, Save & Run All → tự tiếp tục
  (cùng GPU loại, cùng thư viện; khác môi trường thì evaluator chạy lại từ đầu thay vì trộn score).
- Muốn chấm `checkpoint_ve1.pt` (25.200 ảnh = cùng lượng dữ liệu Tiny): đổi `CHECKPOINT_NAME`, output tách riêng.
'''

LOCATOR_CELL = '''
# === Tìm input có giới hạn độ sâu (không quét ~430k ảnh B-Free bằng rglob) + resume Tiny có giới hạn ===
import os, hashlib
COMBINED_T0 = time.time()
SKIP_DIR_NAMES = {'COCO_real_512', 'SD2.1_selfconditioned', 'SD2.1_selfconditioned_origBG',
                  'SD2.1_inpainted_samecat', 'SD2.1_inpainted_samecat_origBG', 'SD2.1_inpainted_diffcat',
                  'SD2.1_inpainted_diffcat_origBG', 'progress', 'data', '.data', 'ai', 'nature'}


def bounded_dirs(root='/kaggle/input', max_depth=6):
    pending = [(Path(root), 0)]
    while pending:
        folder, depth = pending.pop()
        yield folder
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in folder.iterdir() if p.is_dir() and not p.is_symlink())
        except OSError:
            continue
        pending.extend((p, depth + 1) for p in reversed(children) if p.name not in SKIP_DIR_NAMES)


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def locate_checkpoint(name, input_root='/kaggle/input'):
    hits = [d / name for d in bounded_dirs(input_root) if (d / name).is_file() and (d / 'run_state.json').is_file()]
    if not hits:
        raise FileNotFoundError(f'Không thấy {name} cạnh run_state.json trong /kaggle/input. Add dataset checkpoint '
                                'hoặc đặt CHECKPOINT_PATH thủ công.')
    if len({p.stat().st_size for p in hits}) != 1:
        raise ValueError(f'Nhiều {name} khác nhau: {hits}. Đặt CHECKPOINT_PATH thủ công.')
    return hits[0]


def locate_test_manifest(input_root='/kaggle/input'):
    hits = [d / 'test_manifest.csv' for d in bounded_dirs(input_root)
            if d.name == 'seed42' and (d / 'test_manifest.csv').is_file()]
    if not hits:
        raise FileNotFoundError('Không thấy seed42/test_manifest.csv: Add dataset hadinho0710/mainfest-train-genimage.')
    if len({file_digest(p) for p in hits}) != 1:
        raise ValueError(f'Nhiều test_manifest.csv khác nhau: {hits}. Đặt TINY_TEST_MANIFEST thủ công.')
    return hits[0]


def locate_tiny_root(manifest_path, input_root='/kaggle/input'):
    first = pd.read_csv(manifest_path, nrows=1, dtype=str).relative_path.iloc[0]
    hits = sorted({d.resolve() for d in bounded_dirs(input_root, max_depth=5) if (d / first).is_file()})
    if len(hits) != 1:
        raise ValueError(f'Cần đúng một thư mục Tiny-GenImage chứa {first}; tìm thấy {hits}. Đặt TINY_ROOT thủ công.')
    return hits[0]


if '_original_restore_uploaded_progress' not in globals():
    _original_restore_uploaded_progress = restore_uploaded_progress


def restore_uploaded_progress(out, options, fingerprint):
    """Same semantics as the original 'auto' resume, but searches prior outputs with a bounded walk."""
    if options.get('resume_from') != 'auto' or any((out / 'progress').glob('*/part_*.csv')):
        return _original_restore_uploaded_progress(out, options, fingerprint)
    found = []
    for d in bounded_dirs(options['input_root']):
        path = d / 'evaluation_state.json'
        if path.is_file():
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
            if state.get('evaluator') == UPLOAD_EVALUATOR and state.get('fingerprint') == fingerprint:
                found.append((sum(state.get('saved_rows', {}).values()), str(d), d))
    if not found:
        print('[resume] Tiny: không có tiến độ cũ khớp fingerprint; chạy từ đầu.', flush=True)
        return None
    return _original_restore_uploaded_progress(out, dict(options, resume_from=str(sorted(found)[-1][2])), fingerprint)
'''

PARALLEL_MANIFEST_CELL = '''
# === Đọc header PNG của 72.128 file B-Free SONG SONG (bản gốc đọc tuần tự ~11 file/s ≈ 1 h 50) ===
# Logic, kiểm tra, thứ tự bản ghi và file ghi ra GIỐNG HỆT prepare_bfree_manifest gốc; chỉ phần đọc
# header chạy bằng nhiều luồng. Thứ tự giữ nguyên nên records_hash/fingerprint trùng bản tuần tự.
import concurrent.futures
MANIFEST_HEADER_THREADS = 32
if '_original_prepare_bfree_manifest' not in globals():
    _original_prepare_bfree_manifest = prepare_bfree_manifest


def prepare_bfree_manifest(root, opts, out):
    all_ids,train_md5 = bfree_list(root/'train_list.csv','train_list',BFREE_ALL_COUNT)
    valid,valid_md5 = bfree_list(root/'valid_list.csv','valid_list',BFREE_VALID_COUNT)
    if not set(valid.id) <= set(all_ids.id):
        raise ValueError('Valid IDs must be contained in the released train_list master list')
    common = all_ids.set_index('id').loc[valid.id]
    if any(common[k].tolist() != valid[k].tolist() for k in ('same_category','diff_category')):
        raise ValueError('Train/valid category metadata disagree for the same IDs')
    ids = sorted(valid.id, key=lambda image_id: (digest(['bfree_valid',42,image_id]),image_id))
    if opts['smoke_only']:
        ids = ids[:4]
    directories = {name:bfree_folder(root,name,ids[0]) for name in (BFREE_REAL,*BFREE_BRANCHES)}
    jobs = [(image_id, variant, directories[variant]/(image_id+'.png'))
            for image_id in ids for variant in (BFREE_REAL,*BFREE_BRANCHES)]
    print(f'[manifest] {len(ids):,} valid IDs x7 folders; PNG headers/path/size ONLY, no RGB/hash scan; '
          f'{MANIFEST_HEADER_THREADS} threads.',flush=True)

    def read(job):
        image_id, variant, path = job
        if not path.is_file():
            raise FileNotFoundError(f'Missing B-Free valid member: {path}; no IDs are silently dropped')
        width,height,size = bfree_png_header(path)
        relative = path.relative_to(root).as_posix()
        return dict(coco_id=image_id,variant=variant,label=int(variant!=BFREE_REAL),
            relative_path=relative,path=str(path),width=width,height=height,file_bytes=size,
            sample_id=hashlib.sha256(relative.encode()).hexdigest())

    records, started = [], time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(MANIFEST_HEADER_THREADS) as pool:
        for record in pool.map(read, jobs, chunksize=16):   # map() keeps the original order
            records.append(record)
            if len(records)%opts['header_log_every']==0:
                print(f'[manifest] {len(records):,}/{len(jobs):,} headers checked '
                      f'({len(records)/(time.monotonic()-started):.0f}/s)',flush=True)
    manifest = pd.DataFrame(records)
    selected = valid.set_index('id').loc[ids].reset_index()
    metadata = dict(dataset='baopv051/b-free-training-data',split='official_validation_ids',
        source_train_list_count=len(all_ids),source_valid_list_count=len(valid),
        valid_contained_in_train_list=True,train_list_md5=train_md5,valid_list_md5=valid_md5,
        selected_id_count=len(ids),id_order_hash=digest(ids),smoke=opts['smoke_only'],
        unique_images=len(manifest),predictions_per_branch=2*len(ids),total_branch_predictions=12*len(ids),
        branches=list(BFREE_BRANCHES),
        audit='PNG headers + file sizes; RGB decode checked at inference. No full image/pixel hash audit; '
              'resume assumes identical dataset contents/version, same-size content edits are not detected by provenance.')
    fields = ['coco_id','variant','label','relative_path','width','height','file_bytes','sample_id']
    identity = dict(metadata=metadata,records_hash=digest(manifest[fields].to_dict('records')))
    bind_upload_metadata(out/'manifest_identity.json',identity)
    save_csv(out/'bfree_valid_manifest.csv',manifest)
    save_csv(out/'bfree_valid_ids.csv',selected)
    atomic_json(out/'bfree_manifest_summary.json',metadata)
    return manifest,ids,identity
'''

COMBINED_CONFIG = '''
SMOKE_ONLY = False              # True = thử nhanh (B-Free 4 ID, Tiny 16 ảnh/case), output smoke riêng
CHECKPOINT_NAME = 'best.pt'     # 'best.pt' (VE2) hoặc 'checkpoint_ve1.pt' / 'checkpoint_ve2.pt'
RUN_BFREE, RUN_TINY = True, True
TOTAL_BUDGET_HOURS = 11.3       # chung cho cả hai evaluator trong phiên này (Kaggle giới hạn 12 h)

# Để None = tự tìm (giới hạn độ sâu). Điền tay nếu báo lỗi:
CHECKPOINT_PATH = None          # '/kaggle/input/datasets/hoangha071005/check-point-gsd-train-bfee-samecat-ori/gsd_bfree_samecat_origbg_v1/best.pt'
TINY_TEST_MANIFEST = None       # '/kaggle/input/datasets/hadinho0710/mainfest-train-genimage/seed42/test_manifest.csv'
TINY_ROOT = None                # '/kaggle/input/datasets/yangsangtai/tiny-genimage'
BFREE_ROOT = None               # '/kaggle/input/datasets/baopv051/b-free-training-data'

CHECKPOINT_PATH = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else locate_checkpoint(CHECKPOINT_NAME)
TAG = Path(CHECKPOINT_NAME).stem.replace('checkpoint_', '')
BFREE_OUT = f'/kaggle/working/gsd_bfree_samecat_origbg_{TAG}_bfree_valid_v1'
TINY_OUT = f'/kaggle/working/gsd_bfree_samecat_origbg_{TAG}_tiny9_v1'
if RUN_TINY:
    TINY_TEST_MANIFEST = Path(TINY_TEST_MANIFEST) if TINY_TEST_MANIFEST else locate_test_manifest()
    TINY_ROOT = Path(TINY_ROOT) if TINY_ROOT else locate_tiny_root(TINY_TEST_MANIFEST)
print('checkpoint :', CHECKPOINT_PATH)
print('tiny       :', TINY_TEST_MANIFEST, '|', TINY_ROOT)
print('outputs    :', BFREE_OUT, '|', TINY_OUT)
'''

COMBINED_RUN = '''
def remaining_hours():
    return TOTAL_BUDGET_HOURS - (time.time() - COMBINED_T0) / 3600 - 0.15

if RUN_BFREE:
    budget = min(10.5, remaining_hours())
    if budget < 0.3:
        print('[B-Free] Không đủ thời gian trong phiên này; chạy lại với Output làm Input.')
    else:
        BFREE_CONFIG = dict(BFREE_DEFAULTS)
        BFREE_CONFIG.update(input_root='/kaggle/input', bfree_root=BFREE_ROOT, checkpoint_path=str(CHECKPOINT_PATH),
                            output_root=BFREE_OUT, resume_from='auto', expected_signature=None, expected_epoch=None,
                            time_budget_hours=round(budget, 3), micro_batch=1, anchor_micro_batch=1, num_workers=0)
        BFREE_CONFIG = bfree_options(dict(BFREE_CONFIG, smoke_only=SMOKE_ONLY))
        print(f'[B-Free] budget {BFREE_CONFIG["time_budget_hours"]:.2f} h', flush=True)
        run_bfree_valid_evaluation(BFREE_CONFIG)
        gc.collect(); torch.cuda.empty_cache()

if RUN_TINY:
    budget = min(10.5, remaining_hours())
    if budget < 0.3:
        print('[Tiny] Không đủ thời gian trong phiên này; chạy lại với Output làm Input (B-Free đã xong sẽ được bỏ qua).')
    else:
        EVAL_CONFIG = dict(UPLOAD_DEFAULTS)
        EVAL_CONFIG.update(checkpoint_path=str(CHECKPOINT_PATH), test_manifest_path=str(TINY_TEST_MANIFEST),
                           dataset_root=str(TINY_ROOT), output_root=TINY_OUT, resume_from='auto',
                           expected_signature=None, expected_epoch=None, time_budget_hours=round(budget, 3),
                           micro_batch=1, anchor_micro_batch=1, num_workers=0)
        EVAL_CONFIG = finalize_uploaded_notebook_config(EVAL_CONFIG, smoke_only=SMOKE_ONLY)
        print(f'[Tiny] budget {EVAL_CONFIG["time_budget_hours"]:.2f} h', flush=True)
        run_uploaded_evaluation(EVAL_CONFIG)
'''

COMBINED_SUMMARY = '''
# GSD-aug-Tiny (checkpoint epoch1, signature 4595045b…) trên CÙNG evaluator — số đã chạy trước đây.
BASE_BFREE = {  # branch: (BA, AUC, real_recall, fake_recall)
    'SD2.1_selfconditioned': (0.9371, 0.9949, 0.9951, 0.8790),
    'SD2.1_selfconditioned_origBG': (0.5612, 0.7189, 0.9948, 0.1276),
    'SD2.1_inpainted_samecat': (0.9734, 0.9984, 0.9956, 0.9512),
    'SD2.1_inpainted_samecat_origBG': (0.6484, 0.8319, 0.9953, 0.3015),
    'SD2.1_inpainted_diffcat': (0.9852, 0.9991, 0.9948, 0.9757),
    'SD2.1_inpainted_diffcat_origBG': (0.7599, 0.9095, 0.9949, 0.5249)}
BASE_TINY = {  # case (pooled 7.000 ảnh): (BA, AUC, real_recall, fake_recall)
    'native': (0.998571, 0.999986, 0.997429, 0.999714), 'fake_jpeg96': (0.996714, 0.999932, 0.997429, 0.996000),
    'all_jpeg96': (0.996286, 0.999918, 0.996571, 0.996000), 'all_jpeg90': (0.994857, 0.999906, 0.995714, 0.994000),
    'all_jpeg80': (0.994714, 0.999843, 0.996000, 0.993429), 'resize_square256': (0.922000, 0.975754, 0.917143, 0.926857),
    'resize_square560': (0.947571, 0.994563, 0.990286, 0.904857), 'blur_sigma1': (0.996714, 0.999947, 0.997143, 0.996286),
    'resize256_jpeg96': (0.900857, 0.960380, 0.969429, 0.832286)}
TRAIN_BRANCH = 'SD2.1_inpainted_samecat_origBG'


def compare(table, key, base):
    rows = []
    for _, r in table.iterrows():
        b = base.get(r[key])
        if b is None:
            continue
        rows.append({key: r[key], 'BA_new': r.balanced_accuracy, 'BA_gsdaug_tiny': b[0], 'ΔBA': r.balanced_accuracy - b[0],
                     'AUC_new': r.auc, 'AUC_gsdaug_tiny': b[1], 'real_recall_new': r.real_recall,
                     'real_recall_base': b[2], 'fake_recall_new': r.fake_recall, 'fake_recall_base': b[3]})
    return pd.DataFrame(rows)

for name, out in (('B-FREE', Path(BFREE_CONFIG['output_root']) if 'BFREE_CONFIG' in globals() else Path(BFREE_OUT)),
                  ('TINY', Path(EVAL_CONFIG['output_root']) if 'EVAL_CONFIG' in globals() else Path(TINY_OUT))):
    state_path = out / 'evaluation_state.json'
    if not state_path.is_file():
        print(f'{name}: chưa chạy'); continue
    state = json.loads(state_path.read_text())
    print(f"\\n===== {name}: status={state['status']} ({out})")
    if not (out / 'metrics.csv').is_file():
        continue
    metrics = pd.read_csv(out / 'metrics.csv')
    if name == 'B-FREE':
        table = compare(metrics, 'branch', BASE_BFREE)
        table['role'] = np.where(table.branch == TRAIN_BRANCH, 'in-domain', 'held-out')
        display(table)
        held = table[table.role == 'held-out']
        if len(table) == 6:
            print(f"macro BA 6 nhánh: new={table.BA_new.mean():.4f} vs GSD-aug-Tiny={table.BA_gsdaug_tiny.mean():.4f}")
        if len(held):
            print(f"macro BA held-out ({len(held)} nhánh): new={held.BA_new.mean():.4f} vs base={held.BA_gsdaug_tiny.mean():.4f}")
    else:
        pooled = metrics[metrics.generator == 'pooled']
        display(compare(pooled, 'case', BASE_TINY))
        display(metrics[(metrics.case == 'native') & (metrics.generator != 'pooled')]
                [['generator', 'balanced_accuracy', 'auc', 'real_recall', 'fake_recall']])
    if (out / 'partial_metrics.csv').is_file():
        print('PARTIAL (chưa đủ ảnh, không dùng làm kết quả):'); display(pd.read_csv(out / 'partial_metrics.csv'))
'''

COMBINED_ZIP = '''
# Gói nhỏ (csv/json, không có model/ảnh) để tải về phân tích.
archive_path = Path('/kaggle/working/gsd_bfree_samecat_origbg_test_results.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    outs = {Path(BFREE_CONFIG['output_root']) if 'BFREE_CONFIG' in globals() else Path(BFREE_OUT),
            Path(EVAL_CONFIG['output_root']) if 'EVAL_CONFIG' in globals() else Path(TINY_OUT)}
    for out in outs:
        for path in sorted(out.rglob('*')) if out.is_dir() else []:
            if path.is_file() and path.suffix in ('.csv', '.json'):
                archive.write(path, (Path(out.name) / path.relative_to(out)).as_posix())
from IPython.display import FileLink
display(FileLink(str(archive_path)))
print('Nếu còn evaluator paused: Save Version, Add Output này làm Input, chạy lại notebook.')
'''


def build_combined():
    bfree = json.loads((HERE / 'test_gsd_uploaded_best_bfree_valid_kaggle.ipynb').read_text(encoding='utf-8'))
    runtime = max((''.join(c['source']) for c in bfree['cells'] if c['cell_type'] == 'code'), key=len)
    for needed in ('def run_uploaded_evaluation', 'def run_bfree_valid_evaluation', 'def finalize_uploaded_notebook_config',
                   'def restore_uploaded_progress', 'UPLOAD_EVALUATOR'):
        if needed not in runtime:
            raise ValueError(f'runtime lacks {needed}')
    cells = [md(COMBINED_INTRO), code(PIP),
             md('## Runtime evaluator (byte-identical với notebook test B-Free/Tiny đã dùng cho GSD-aug-Tiny)'),
             code(runtime), md('## Nhận checkpoint train trên B-Free'), code(COMPAT_CELL),
             md('## Tìm input (giới hạn độ sâu) và resume'), code(LOCATOR_CELL),
             md('## Đọc header B-Free song song'), code(PARALLEL_MANIFEST_CELL),
             md('## Cấu hình'), code(COMBINED_CONFIG),
             md('## Chạy: B-Free valid 6 nhánh → Tiny 9 case'), code(COMBINED_RUN),
             md('## So sánh với GSD-aug-Tiny'), code(COMBINED_SUMMARY), code(COMBINED_ZIP)]
    for c in cells:
        if c['cell_type'] == 'code' and not ''.join(c['source']).startswith('%'):
            ast.parse(''.join(c['source']))
    return notebook(cells)


if __name__ == '__main__':
    nb = build_combined()
    (HERE / COMBINED_NB).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('wrote', COMBINED_NB)
