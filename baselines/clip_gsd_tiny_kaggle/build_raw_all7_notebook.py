"""Build standalone native-resolution All7 notebook without changing the legacy SD15 notebook."""
import ast
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main(center_crop=False):
    source=(HERE/'runtime.py').read_text(encoding='utf-8')+'\n\n'+(HERE/'runtime_raw_all7.py').read_text(encoding='utf-8')
    ast.parse(source)
    def md(s):
        return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
    def code(s):
        return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
    cells=[md('''# CLIP + GSD — All7, raw images, variable native token grids

Attach **yangsangtai/tiny-genimage** and enable GPU + Internet.

Train on all seven generator train folders. No train JPEG, blur, flip, resize or crop.
Decode RGB and CLIP-normalize; right/bottom edge padding to a multiple of 14 preserves every source pixel.
Both detector and frozen anchor interpolate positional embeddings for the actual token grid.

This is a NEW experimental protocol, not the paper's SD14 protocol. Default: GSD only, one GPU.
Native resolution can be very expensive, including images with tens of thousands of tokens.
SDPA reduces attention memory; the Gram eigensolver streams ALL patch tokens to construct the
same uncentered principal subspace as SVD in exact arithmetic (numerical conditioning differs).
No hidden image-size cap, token subsampling, skipped large images or automatic downscaling.
An image may still exceed GPU memory even with micro_batch=1; that run stops explicitly.

Logical batch=8, micro_batch=1; these differ from the earlier 224-pixel experiment.
Mixed image shapes stay in the same logical batch. Computation chunks only combine equal shapes.
Partial final batches are included, so size buckets do not drop rare image shapes or classes.
The basis weights every patch token equally, so larger images contribute more tokens.

Unique decoded-pixel groups are assigned train/validation once, preventing cross-generator duplicates
from leaking across the split. Then balance each generator and label to the common minimum count.
Validation is raw, from TRAIN folders; the generator val folders remain TEST only.
Early stop: min validation CE, patience=3, min_delta=.001, maximum 50 epochs.
Save best.pt and last.pt atomically after each epoch; last.pt includes optimizer/scaler/stopper/config.
New output directory: do not resume an old SD15/224 checkpoint. Keep exact config for resume.

Default evaluation: native only. Optional stress cases run AFTER training on the selected checkpoint;
they never change training. Native + JPEG96/90/80 are shown in the config comment.
Fake-only JPEG is label-dependent diagnostic processing, not deployable preprocessing.
All seven test generators are seen; no unseen-generator claim or macro_unseen report.
Save the original pretrained model reference, config, token census, manifests and best.pt
for later CommFor/B-Free evaluation. Those external evaluation runners are separate.
'''),code('%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" "scikit-learn<1.9" "pillow<12" matplotlib tqdm\n'),
        md('## Runtime\nSelf-contained. Edit only the configuration cell below.'),code(source),
        code('''CONFIG = dict(DEFAULT_CONFIG)
CONFIG.update(
    output_root='/kaggle/working/gsd_raw_all7_native_v1',
    dataset_root=None,
    models=['clip_gsd'],  # optional paired control: ['clip_ft', 'clip_gsd']
    epochs=50, early_stopping=True, patience=3, min_delta=0.001,
    train_batch=8, test_batch=8, micro_batch=1, anchor_micro_batch=1,
    num_workers=0, precision='fp16', gradient_checkpointing=True,
    eval_cases=['native'],
    smoke=False, resume=True,
)
# Optional test variants ONLY (training remains raw):
# CONFIG['eval_cases'] = ['native','fake_jpeg96','jpeg96','jpeg90','jpeg80','size','size_jpeg96']
# Smoke still uses full image resolution; it does not resize images:
# CONFIG.update(smoke=True, output_root='/kaggle/working/gsd_raw_all7_native_smoke')
print(json.dumps(CONFIG, indent=2))
'''),code('metrics = run(CONFIG)\n'),code('''out=Path(CONFIG['output_root'])
display(pd.read_csv(out/'macro_all.csv'))
display(pd.read_csv(out/'metrics.csv'))
print('Checkpoints:', list(out.glob('seed*/clip_gsd/*.pt')))
''')]
    if center_crop:
        cells[0]=md('''# CLIP + GSD — All7, center crop tối đa 560, không resize

Gắn dataset **yangsangtai/tiny-genimage**, bật GPU và Internet.

Train cả 7 generator. Ảnh lớn: center crop từng chiều xuống tối đa 560 pixel.
Chiều nhỏ hơn hoặc bằng 560 được giữ nguyên, không phóng ảnh nhỏ lên 560.
Sau crop, padding cạnh phải/dưới 0–13 pixel tới bội số 14; normalize CLIP.
Không resize, không JPEG/blur/flip augmentation trong train. Giữ nguyên giá trị pixel trong crop.

Ví dụ: 1024×1024 → 560×560 (1601 tokens); 512×512 → 518×518 (1370 tokens);
224×224 giữ nguyên (257 tokens); 1024×448 → 560×448 (1281 tokens).
Positional embeddings được nội suy cho lưới thực tế ở cả detector lẫn frozen anchor.
Đây là crop ở tỷ lệ pixel gốc: mất vùng ngoài tâm ảnh lớn, không phải giữ toàn bộ ảnh raw.

Logical batch=8, microbatch=1; SDPA, Gram-eigh dùng toàn bộ patch tokens trong batch.
Train/validation/test native dùng CÙNG chính sách crop. All7 hash-group split ngăn ảnh trùng
đi vào cả train và validation. Early stopping patience=3/min_delta=.001, tối đa 50 epoch.
Khôi phục best.pt theo validation CE. best.pt/last.pt lưu sau mỗi epoch hoàn thành.
Không resume checkpoint native/SD15 vào giao thức mới. 560 chưa được kiểm chứng VRAM trên T4;
chạy smoke trong thư mục riêng trước. Không tự thu nhỏ/bỏ ảnh khi OOM.

Mặc định test native chỉ có crop/padding nói trên. Có thể bật thêm test JPEG ở config:
biến đổi JPEG trên ảnh gốc TRƯỚC crop, không thay đổi train. Resize stress-test là case riêng.
token_census.csv ghi cả kích thước gốc, tọa độ crop và kích thước/token đưa vào model.
Notebook hiện vẫn scan đầy đủ khi chạy lại; resume chỉ khôi phục training checkpoint cuối epoch.
''')
        config=''.join(cells[4]['source'])
        config=config.replace('gsd_raw_all7_native_v1','gsd_raw_all7_crop560_v1')
        config=config.replace('    dataset_root=None,',"    dataset_root=None,\n    input_mode='center_crop_max_edge_pad14', crop_max_side=560,")
        config=config.replace('Smoke still uses full image resolution; it does not resize images:',
                              'Smoke uses the SAME max-560 crop policy; no resize:')
        config=config.replace('gsd_raw_all7_native_smoke','gsd_raw_all7_crop560_smoke')
        cells[4]=code(config)
    for i,c in enumerate(cells):
        c['id']=f'raw-all7-{i:02d}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='gpu',isInternetEnabled=True)))
    path=HERE/('train_gsd_raw_all7_crop560_kaggle.ipynb' if center_crop else 'train_gsd_raw_all7_native_kaggle.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print(path)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--center-crop',action='store_true')
    main(parser.parse_args().center_crop)
