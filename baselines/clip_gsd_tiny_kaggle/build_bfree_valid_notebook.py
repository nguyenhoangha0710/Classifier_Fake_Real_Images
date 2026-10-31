"""Build the standalone GPU notebook for the current checkpoint on B-Free valid."""
import ast
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
NOTEBOOK='test_gsd_uploaded_best_bfree_valid_kaggle.ipynb'


def build_notebook():
    source='\n\n'.join((HERE/name).read_text(encoding='utf-8') for name in (
        'runtime.py','runtime_raw_all7.py','manifest_reuse.py','evaluate_all7_variants.py',
        'evaluate_uploaded_best.py','evaluate_bfree_valid.py'))
    ast.parse(source)
    def md(text):
        return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))
    def code(text):
        if not text.startswith('%pip'):
            ast.parse(text)
        return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True))
    cells=[md('''# GSD checkpoint hiện tại — B-Free official validation IDs

**GPU + Internet. Không train lại, không fit calibration, không chọn ngưỡng trên B-Free.**
Notebook tự chứa toàn bộ code; gửi một file notebook là đủ, không cần gửi các file.py.

## Chỉ cần hai input

1. Dataset checkpoint augmented đã dùng test GenImage9 case: best.pt, best.zip, hoặc folder best/
   đã được Kaggle giải nén (data.pkl + data/ + version). Không trỏ riêng data.pkl.
2. [baopv051/b-free-training-data](https://www.kaggle.com/datasets/baopv051/b-free-training-data),
   chứa train_list.csv, valid_list.csv, COCO_real_512 và sáu thư mục SD2.1.
3. Chỉ khi tiếp tục: thêm OUTPUT B-FREE của lượt trước làm Input để resume.

KHÔNG cần Tiny-GenImage, manifest GenImage, mask hay bbox. Không tải178GB về working hoặc giải nén lại.
Internet dùng tải ORIGINAL pretrained CLIP ViT-L/14 làm frozen anchor; không dùng detector fine-tuned làm anchor.
Checkpoint đúng signature augmented epoch1: 4595045b4c6c76c1ad4aa303a7d2f1690cfac8150add4548d24609a6c980dce6.
Chỉ load checkpoint bạn tin cậy; weights_only=True, strict state_dict. Không có optimizer/training.

## Cohort và sáu nhánh

Lấy đúng **10.304 ID của valid_list.csv**. Mỗi ID có1 real +6 fake; sáu fake là sáu biến thể SD2.1,
KHÔNG phải sáu generator độc lập. Tên gọi báo cáo: cross-dataset evaluation on B-Free validation split,
không phải official B-Free test set. Không dùng valid này học ngưỡng rồi báo lại test trên cùng ảnh.

| Nhánh fake | So sánh với cùng tập COCO real |
|---|---|
| SD2.1_selfconditioned |10.304 real +10.304 fake|
| SD2.1_selfconditioned_origBG |10.304 real +10.304 fake|
| SD2.1_inpainted_samecat |10.304 real +10.304 fake|
| SD2.1_inpainted_samecat_origBG |10.304 real +10.304 fake|
| SD2.1_inpainted_diffcat |10.304 real +10.304 fake|
| SD2.1_inpainted_diffcat_origBG |10.304 real +10.304 fake|

Hai list CSV được kiểm tra MD5 khớp tác giả. train_list có51.517 ID và chứa toàn bộ valid;
code KHÔNG dùng nó để train. Dùng để xác minh valid và category metadata, không trộn train vào eval.
[Tài liệu/checksum tác giả](https://github.com/grip-unina/B-Free/tree/main/training_data).

## Native processing và giao thức batch GSD

- Chỉ native PNG của B-Free; không tự thêm JPEG/resize/blur hay dùng preprocessing B-Free504 của tác giả.
- Giữ pipeline checkpoint: RGB → center crop tối đa560 mỗi chiều → edge-pad14 → CLIP normalization.
  Ảnh512x512 được giữ nguyên512x512, pad phải/dưới thành518x518; 1.370 token gồm CLS.
- Giữ FP16/SDPA, Gram-eigh k64, semantic suppression trước20–23, frozen anchor nguyên bản.
- Một GPU, logical batch8, microbatch1, anchor_microbatch1. Không tự đổi batch khi OOM.
- Thứ tự ID deterministic seed42, giống nhau ở sáu nhánh. Mỗi batch có4 ID, xen kẽ real/fake cùng ID.
  Đây là giao thức đánh giá của notebook, không phải tuyên bố giao thức gốc của paper B-Free.
- **Tính basis từ toàn bộ8 ảnh; tính lại score real ở MỖI nhánh**, không dùng lại score real giữa các nhánh.
  Có72.128 file ảnh khác nhau nhưng123.648 lượt dự đoán (6 x20.608). Không coi các ảnh cùng ID là độc lập.
- Chỉ đọc PNG header/path/file size trước inference, KHÔNG full RGB/hash scan. Decode đầy đủ khi thực sự test.
  Fail nếu thiếu/hỏng ảnh; không tự bỏ ID hoặc sửa resize. Metadata không phát hiện được thay đổi nội dung
  giữ nguyên kích thước byte; chỉ resume với đúng phiên bản dữ liệu gốc.

## Chạy và lưu kết quả

Mặc định **SMOKE_ONLY=False: đủ valid, đủ sáu nhánh**. Có thể đổi True để thử4 ID ở output riêng.
Không có giới hạn1/3 nhánh mỗi lượt. Ngân sách9,5h/lượt tính cả chuẩn bị, sau đó planned pause bình thường.
Đã lưu CSV dự đoán và metrics ngay khi xong mỗi nhánh; progress shard sau mỗi10 batch hoàn chỉnh.
Hard kill có thể mất batch chưa flush. Việc code ghi working không bảo đảm nền tảng luôn publish Output sau crash.

Output: /kaggle/working/gsd_bfree_valid_v1. CSV gồm COCO ID, variant, real/fake, branch,
prob_fake=softmax(logits.float())[:,1], hai logits, margin, batch_id và input_tokens.
metrics.csv: riêng từng nhánh (accuracy, BA, precision/recall/F1 fake, AUC/AP, confusion matrix).
macro_metrics.csv: trung bình các nhánh ĐÃ HOÀN TẤT, kèm cờ all_branches_complete; không gộp lệch1:6.
Partial metrics được tách riêng, không báo là full test. Không tối ưu ngưỡng trên test.

Nếu paused, lưu/download Output, thêm nó làm Input ở phiên GPU tiếp theo. resume_from='auto' chỉ dùng
progress đúng fingerprint (checkpoint/IDs/batch/processing/precision/torch/transformers/Pillow/GPU).
Không dùng progress GenImage/TPU hoặc smoke để ghép full B-Free. Không gắn Output thì phiên mới chạy lại.
Cell cuối tạo ZIP nhỏ chỉ chứa reports/predictions/manifests/progress, không có ảnh/model.
'''),code('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" "scikit-learn<1.9" "pillow<12" matplotlib tqdm\n'),
        md('## Runtime — tự chứa, không gọi train'),code(source),code('''SMOKE_ONLY = False     # False=10.304 valid ID và đủ6 nhánh; True=4 ID, output riêng
BFREE_CONFIG = dict(BFREE_DEFAULTS)
BFREE_CONFIG.update(
    input_root='/kaggle/input',
    bfree_root=None,            # auto; folder có valid_list.csv + COCO_real_512
    checkpoint_path=None,       # auto; best.pt / best.zip / CẢ folder best/
    output_root='/kaggle/working/gsd_bfree_valid_v1',
    resume_from='auto',
    pretrained_model_path=None,
    time_budget_hours=9.5,
    save_every_batches=10, log_every_batches=10, header_log_every=2000,
    micro_batch=1, anchor_micro_batch=1, num_workers=0,
)
# Chỉ điền đường dẫn nếu auto báo nhiều nguồn hoặc không tìm thấy:
# BFREE_CONFIG['bfree_root'] = '/kaggle/input/datasets/baopv051/b-free-training-data'
# BFREE_CONFIG['checkpoint_path'] = '/kaggle/input/<checkpoint-dataset>/best'
# BFREE_CONFIG['resume_from'] = '/kaggle/input/<previous-output>/gsd_bfree_valid_v1'
# Không có cases=['native'] / max_ids / threshold: đây là full B-Free-valid six-branch evaluation.
'''),code('''# Chốt mode ngay trước khi chạy. Không lấy nhầm smoke/native-only từ notebook GenImage cũ.
BFREE_CONFIG = bfree_options(dict(BFREE_CONFIG, smoke_only=SMOKE_ONLY))
print(json.dumps(BFREE_CONFIG, indent=2))
metrics = run_bfree_valid_evaluation(BFREE_CONFIG)
'''),code('''out = Path(BFREE_CONFIG['output_root'])
state = json.loads((out / 'evaluation_state.json').read_text())
print('MODE:', 'SMOKE ONLY — không dùng báo cáo full valid' if SMOKE_ONLY else 'FULL B-FREE VALID')
print('STATUS:', state['status'], '| completed:', len(state['completed_branches']), '/6')
print('Remaining:', state['remaining_branches'])
display(metrics)
if (out / 'macro_metrics.csv').is_file():
    print('Macro: chỉ các nhánh hoàn tất; kiểm tra all_branches_complete.')
    display(pd.read_csv(out / 'macro_metrics.csv'))
if (out / 'partial_metrics.csv').is_file():
    print('PARTIAL — chưa đủ ảnh, không trộn vào kết quả full:')
    display(pd.read_csv(out / 'partial_metrics.csv'))
if state['status'] == 'complete' and not SMOKE_ONLY:
    assert len(state['completed_branches']) == 6 and state['selected_ids'] == 10304
    print('Hoàn tất6 nhánh x20.608 ảnh. Có72.128 file khác nhau,123.648 lượt dự đoán.')
'''),code('''archive_path = out.parent / (out.name + '_results.zip')
temp = archive_path.with_suffix('.zip.tmp')
with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.csv', '.json'):
            archive.write(path, (Path(out.name) / path.relative_to(out)).as_posix())
temp.replace(archive_path)
from IPython.display import FileLink
display(FileLink(str(archive_path)))
print('Nếu paused: lưu Output này, Add Input vào phiên sau để resume.')
''')]
    for i,cell in enumerate(cells):
        cell['id']=f'gsd-bfree-valid-{i:02d}'
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='gpu',isInternetEnabled=True)))


if __name__=='__main__':
    target=HERE/NOTEBOOK
    target.write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1),encoding='utf-8')
    print(target)
