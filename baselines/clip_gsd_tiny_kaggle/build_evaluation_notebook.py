"""Build a self-contained inference-only Kaggle notebook; no training invocation."""
import ast
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    source='\n\n'.join((HERE/name).read_text(encoding='utf-8') for name in
                         ('runtime.py','runtime_raw_all7.py','evaluate_all7_variants.py'))
    ast.parse(source)
    def md(s):
        return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
    def code(s):
        return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
    cells=[md('''# GSD All7 crop560 — test checkpoint trên các biến thể ảnh

**Chỉ inference, không train lại.** Dùng `best.pt` của lần train All7 crop560,
đọc lại đúng 7.000 ảnh / thứ tự / batch từ `test_manifest.csv` đã lưu.
Không scan RGB/hash lại toàn bộ dataset, không đọc các ảnh train.

## Chuẩn bị Kaggle

1. Import notebook này vào một Kaggle Notebook mới; bật **GPU** và **Internet**.
2. Add Input: [Tiny-GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
3. Add Input: [checkpoint của bạn](https://www.kaggle.com/datasets/hadinho0710/checkpoint-gsd-sdv5-train),
   chọn **version 2 / All7 crop560**, có `gsd_raw_all7_crop560_v1/`.
4. Run All. Code tự tìm checkpoint và dataset; nếu có nhiều bản trùng, điền đường dẫn ở config.

Frozen anchor cần **CLIP pretrained gốc** `openai/clip-vit-large-patch14`, tải qua Internet.
Không dùng detector đã fine-tune làm anchor. Có thể dùng bản pretrained gốc offline qua
`pretrained_model_path` (thư mục Hugging Face có config + weights).
Không tải checkpoint không rõ nguồn: PyTorch checkpoint chứa dữ liệu pickle.

## Giao thức cố định

Biến đổi trên RGB ảnh gốc → center crop từng chiều tối đa 560 như train →
edge-pad phải/dưới tới bội 14 → CLIP normalize. Không tự đổi crop, batch hay bỏ ảnh khi OOM.
`native` nghĩa là **không thêm biến đổi**, nhưng vẫn có crop/padding của model; không phải full-resolution.
Batch logic giữ nguyên 8 ảnh; microbatch=1 chỉ chia nhỏ tính toán, không thay nhóm tính basis.
Mỗi biến thể tính lại basis và prediction cho **toàn bộ** nhóm ảnh.

Chạy native trước và đối chiếu kết quả cũ trong `native_reference_check.json`.
Nếu `review_required`, chưa nên kết luận độ bền từ các biến thể trước khi kiểm tra sai lệch native.
Mọi case giữ threshold 0.5, không chọn lại threshold/best checkpoint bằng test.

| Case mặc định | Xử lý trước crop |
|---|---|
| native | Giữ RGB giải mã gốc |
| png_roundtrip | Encode/decode PNG lossless: kiểm tra đối chứng |
| fake_jpeg96 | Chỉ fake encode/decode JPEG Q96; real giữ nguyên |
| all_jpeg96 / 90 / 80 | Cả real và fake encode/decode JPEG ở cùng quality |
| resize_square256 / 560 | Cả hai lớp resize BICUBIC về hình vuông, kể cả upscale ảnh nhỏ |
| blur_sigma1 | Gaussian blur radius 1 pixel ở độ phân giải gốc |
| resize256_jpeg96 | Resize vuông 256, sau đó JPEG Q96 |

JPEG mặc định subsampling=0 (4:4:4), optimize=False, progressive=False.
Fake-only JPEG cần nhãn, chỉ là phép chẩn đoán bias, **không phải pipeline triển khai**.
Do basis phụ thuộc batch, score real có thể đổi ở case fake-only dù pixel real không đổi.
Resize vuông thay cả kích thước, tỷ lệ ảnh và số token: đây là stress test, không cô lập riêng JPEG bias.
Test trên 7 generator đã thấy trong train, không chứng minh khả năng generalize sang generator mới.

Lưu tiến độ mỗi 25 batch hoàn chỉnh và log mỗi 10 batch. Chạy lại cùng output sẽ tiếp tục;
batch chưa lưu phải chạy lại. Muốn tiếp tục ở session khác: lưu/download toàn bộ thư mục output,
chép nó trở lại thư mục ghi được trong `/kaggle/working` rồi dùng cùng config/environment.
Đổi checkpoint, manifest, môi trường hoặc giao thức: dùng thư mục output mới.
10 case là 70.000 lượt inference, nên chạy smoke trước nếu cần kiểm tra setup.
'''),code('''%pip install -q "transformers==4.46.3" "huggingface_hub>=0.23.2,<1.0" "pandas<3" "scikit-learn<1.9" matplotlib tqdm
# Giữ torch/Pillow của Kaggle; phiên bản thực tế được ghi trong provenance.json.
'''),md('## Runtime\nCác định nghĩa model dùng lại đúng implementation lúc train. Cell chạy phía dưới chỉ gọi evaluator.'),
        code(source),code('''EVAL_CONFIG = dict(EVAL_DEFAULTS)
EVAL_CONFIG.update(
    checkpoint_path=None,  # auto best.pt, KHÔNG dùng last.pt
    dataset_root=None,     # thư mục chứa imagenet_glide/, imagenet_ai_0419_biggan/, ...
    output_root='/kaggle/working/gsd_all7_crop560_variants',
    pretrained_model_path=None,
    cases=[
        'native', 'png_roundtrip',
        'fake_jpeg96', 'all_jpeg96', 'all_jpeg90', 'all_jpeg80',
        'resize_square256', 'resize_square560', 'blur_sigma1', 'resize256_jpeg96',
    ],
    micro_batch=1, anchor_micro_batch=1, num_workers=0,
    save_every_batches=25, log_every_batches=10,
    max_batches=None,  # None = toàn bộ 7.000 ảnh/case
)
# Case bổ sung hợp lệ:
# 'all_jpeg70', 'fake_jpeg90', 'resize_square224', 'resize_square336',
# 'resize_square512', 'resize_long560' (giữ tỷ lệ, chỉ thu nhỏ), 'blur_sigma2'.
# Smoke giữ nguyên thành viên batch, KHÔNG coi đây là kết quả test đầy đủ:
# EVAL_CONFIG.update(max_batches=2, cases=['native','fake_jpeg96'],
#                    output_root='/kaggle/working/gsd_variants_smoke')
print(json.dumps(EVAL_CONFIG, indent=2))
'''),code('metrics = run_evaluation(EVAL_CONFIG)\n'),code('''out=Path(EVAL_CONFIG['output_root'])
pooled=metrics[metrics.generator=='pooled'].copy()
display(pooled)
delta=pd.read_csv(out/'delta_from_native.csv')
display(delta[delta.generator=='pooled'][['case','delta_accuracy','delta_balanced_accuracy',
    'delta_real_recall','delta_fake_recall','delta_auc','delta_ap']])
print('Delta ở đơn vị 0–1; nhân 100 để đọc điểm phần trăm.')
print('Native reference:', (out/'native_reference_check.json').read_text())
print('Toàn bộ kết quả:', out)
'''),code('''# Biểu đồ so sánh; không thay đổi mô hình/threshold theo kết quả test.
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,5))
pooled.set_index('case')[['balanced_accuracy','auc']].mul(100).plot.bar(ax=ax)
ax.set(ylabel='%', title='GSD All7 crop560 — image-variant stress tests', ylim=(0,101))
ax.legend(['Balanced accuracy','ROC-AUC'])
fig.tight_layout(); fig.savefig(out/'variant_comparison.png',dpi=160)
plt.show()
'''),code('''# Gói CSV/JSON/progress/plot để tải xuống. Không chứa model weights hay ảnh đầu vào.
import zipfile
archive=out.parent/(out.name+'_results.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(out.rglob('*')):
        if path.is_file():
            bundle.write(path, str(Path(out.name)/path.relative_to(out)))
from IPython.display import FileLink
display(FileLink(str(archive)))
''')]
    for i,cell in enumerate(cells):
        cell['id']=f'gsd-variants-{i:02d}'
        if cell['cell_type']=='code' and not ''.join(cell['source']).startswith('%pip'):
            ast.parse(''.join(cell['source']))
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='gpu',isInternetEnabled=True)))
    target=HERE/'test_gsd_all7_crop560_variants_kaggle.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
