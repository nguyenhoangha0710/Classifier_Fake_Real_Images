import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source'

def read(relative):
    with (SRC / relative).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))

tiny = read('capd_eval_tiny/r2_dinov3/seed42/tiny_eval/metrics.csv')
bfree = read('capd_eval_bfree/r2_dinov3/seed42/bfree_eval/metrics_bfree.csv')
checks = 0
for row in tiny + bfree:
    if row.get('branch') == 'macro':
        continue
    n, tn, fp, fn, tp = [float(row[k]) for k in ('n', 'tn', 'fp', 'fn', 'tp')]
    assert tn + fp + fn + tp == n
    assert abs((tn + tp) / n - float(row['accuracy'])) < 1e-12
    assert abs((tp / (tp + fn) + tn / (tn + fp)) / 2 - float(row['balanced_accuracy'])) < 1e-12
    checks += 1
pooled = [r for r in tiny if r['generator'] == 'pooled']
macros = [r for r in bfree if r['branch'] == 'macro']
for macro in macros:
    rows = [r for r in bfree if r['pooling'] == macro['pooling'] and r['branch'] != 'macro']
    for key in ('auc', 'ap', 'balanced_accuracy', 'fake_recall', 'real_recall'):
        assert abs(sum(float(r[key]) for r in rows) / len(rows) - float(macro[key])) < 1e-12

def pct(row, key):
    return f"{100 * float(row[key]):.2f}"

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

lines = [
    '# Tổng hợp CAPD: Tiny-GenImage và BFree',
    '',
    'Nguồn: [Kaggle dataset, version 1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree).',
    '',
    '## Mô hình và cách đánh giá',
    '',
    '- Run `r2_dinov3`, backbone DINOv3 ViT-L/16, LoRA rank 8, seed 42, checkpoint epoch 3; validation monitor CE = 0.006938915.',
    '- 3.421.698 tham số trainable / 306.575.874 tổng tham số. Geometry gốc: center crop tối đa 560, replicate-pad tới bội patch 16, không resize.',
    '- Accuracy, balanced accuracy (BA) và recall bên dưới dùng ngưỡng cố định 0,5. AUC/AP được đổi sang phần trăm.',
    '- Tiny: 7.000 ảnh test, gồm 3.500 real và 3.500 fake từ 7 generator đã xuất hiện trong training. Mỗi case là biến đổi của cùng tập test, không phải tập ảnh độc lập.',
    '- BFree: code đọc valid_list.csv và kiểm tra MD5; 10.304 COCO ID, 10.304 ảnh real dùng chung và 6 × 10.304 ảnh fake, tổng cộng 72.128 ảnh riêng biệt. Mỗi nhánh so sánh 20.608 ảnh (real + fake). Đây là BFree validation, dù tên dataset có chữ test.',
    '- BFree lse_full gộp toàn bộ patch; lse_window lấy điểm cao nhất qua các cửa sổ 14×14 patch (224×224 pixel, stride 2 patch).',
    '',
    '## Tiny: kết quả tổng thể',
    '',
    table(['Case', 'Acc./BA (%)', 'AUC (%)', 'AP (%)', 'Recall fake (%)', 'Recall real (%)', 'FP / FN'],
          [[r['case'], pct(r,'accuracy'), pct(r,'auc'), pct(r,'ap'), pct(r,'fake_recall'), pct(r,'real_recall'), f"{r['fp']} / {r['fn']}"] for r in pooled]),
    '',
    'Native: 6 ảnh real bị gọi nhầm fake, 1 ảnh fake bị bỏ sót. PNG roundtrip có toàn bộ metrics trùng native. JPEG Q80–96 và blur sigma 1 giữ accuracy ≥99,79%.',
    '',
    'Resize 256×256 là điểm yếu lớn nhất: accuracy 91,20%, với 601 false positive / 3.500 ảnh real (17,17%) và 15 false negative. AUC vẫn 99,50%, cho thấy khả năng xếp hạng còn tốt nhưng quyết định tại ngưỡng 0,5 bị lệch. ECE tăng từ 0,00105 (native) lên 0,08740. Resize 256 + JPEG96 đạt 93,74%; resize 560 đạt 99,21%.',
    '',
    '## Tiny: theo generator',
    '',
    table(['Generator', 'Native acc. (%)', 'Resize256 acc. (%)', 'Resize560 acc. (%)'],
          [[g] + [pct(next(r for r in tiny if r['generator']==g and r['case']==case),'accuracy') for case in ('native','resize_square256','resize_square560')]
           for g in ('adm','biggan','glide','midjourney','sd15','vqdm','wukong')]),
    '',
    'Mỗi dòng gồm 500 real và 500 fake. Midjourney khó hơn các nguồn khác sau resize: ở 560 có 21/500 fake bị bỏ sót; ở 256 AUC=97,63% và TPR tại FPR 1%=61,60%, thấp hơn các generator còn lại.',
    '',
    '## BFree: theo nhánh',
    '',
    table(['Nhánh', 'AUC full (%)', 'BA full (%)', 'BA window (%)', 'Recall fake full (%)', 'Recall fake window (%)'],
          [[r['branch'].replace('SD2.1_',''), pct(r,'auc'), pct(r,'balanced_accuracy'),
            pct(next(w for w in bfree if w['branch']==r['branch'] and w['pooling']=='lse_window'),'balanced_accuracy'),
            pct(r,'fake_recall'), pct(next(w for w in bfree if w['branch']==r['branch'] and w['pooling']=='lse_window'),'fake_recall')]
           for r in bfree if r['pooling']=='lse_full']),
    '',
    table(['Pooling', 'Macro AUC (%)', 'Macro AP (%)', 'Macro BA (%)', 'Macro recall fake (%)', 'Recall real (%)'],
          [[r['pooling'],pct(r,'auc'),pct(r,'ap'),pct(r,'balanced_accuracy'),pct(r,'fake_recall'),pct(r,'real_recall')] for r in macros]),
    '',
    'Window tăng macro BA 0,24 điểm phần trăm và macro fake recall 0,60 điểm; AUC gần như không đổi. False positive trên tập real chung tăng từ 80 lên 93 ảnh.',
    '',
    'Các nhánh origBG giảm mạnh: BA full chỉ 55,00–81,09%, trong khi các nhánh tương ứng không origBG đạt 92,28–99,23%. Selfconditioned_origBG yếu nhất: bỏ sót 9.193/10.304 fake với full, hoặc 9.104 với window. Kết quả cho thấy khó phát hiện các mẫu thuộc nhánh này; chưa đủ bằng chứng từ bảng metrics để kết luận nguyên nhân nhân quả.',
    '',
    '## Cách diễn giải và giới hạn',
    '',
    '- Mô hình rất tốt trên Tiny và bền với các mức JPEG/blur đã thử, nhưng chưa bền với resize 256.',
    '- Khả năng chuyển sang BFree khác biệt mạnh theo nhánh; macro BA 82,54% (full) hoặc 82,78% (window) không có nghĩa mọi nhánh đều đạt mức đó.',
    '- Không gộp các nhánh BFree thành accuracy trên 123.648 ảnh độc lập: tập real được sử dụng lại 6 lần. Các dòng macro là trung bình 6 nhánh.',
    '- Cột best_ba/best_ba_threshold trong nguồn được tìm trực tiếp bằng nhãn tập đánh giá, chỉ là kết quả tham khảo trên tập đó. Không dùng làm kết quả ngưỡng triển khai đã xác thực; muốn đổi ngưỡng cần tập calibration độc lập.',
    '- Chỉ có một seed. Chưa có độ lệch chuẩn hoặc bằng chứng tổng quát cho các mức JPEG/blur/resize ngoài những case liệt kê.',
    '- Đã kiểm tra n=TN+FP+FN+TP và tính lại accuracy/BA từ confusion counts cho toàn bộ 92 dòng không phải macro, đồng thời đối chiếu các macro BFree. Không tính lại AUC/AP từ prediction từng ảnh.',
    '',
    'Các file nguồn, inventory đủ 372 file và SHA-256 của 31 file đã tải nằm trong thư mục source/. Không tải checkpoint hoặc bản đồ patch dung lượng lớn.',
]
(ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
(ROOT / 'analysis.json').write_text(json.dumps({'confusion_rows_checked': checks, 'tiny_pooled': pooled, 'bfree_metrics': bfree}, indent=2), encoding='utf8')
print(f'Checked {checks} confusion matrices and both BFree macro rows. Wrote REPORT.md and analysis.json.')
