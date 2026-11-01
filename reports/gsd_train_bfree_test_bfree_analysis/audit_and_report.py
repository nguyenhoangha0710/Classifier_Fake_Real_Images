"""Validate evaluation IDs/status and render a summary without treating partials as complete."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'source/gsd_bfree_samecat_origbg_best_bfree_valid_v1'


def read(path):
    with path.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def pct(v):
    return f'{100*v:.2f}'


def main():
    spec = importlib.util.spec_from_file_location('run_analysis', ROOT/'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        module.main()  # Recompute ROC AUC/AP and metrics directly from source predictions.
    a = json.loads((ROOT/'analysis.json').read_text(encoding='utf8'))
    state = json.loads((RUN/'completion.json').read_text(encoding='utf8'))
    checkpoint = json.loads((RUN/'checkpoint_inspection.json').read_text(encoding='utf8'))
    full, partial = read(RUN/'predictions.csv'), read(RUN/'predictions_SD2.1_inpainted_diffcat_origBG.partial.csv')
    split = ROOT/'training_split_source'
    train_ids = {r['coco_id'].zfill(12) for r in read(split/'train_order.csv')}
    val_ids = {r['coco_id'].zfill(12) for r in read(split/'validation_ids.csv')}
    previous_ids = {r['id'].zfill(12) for r in read(ROOT.parent/'gsd_bfree_valid_analysis/source/gsd_bfree_valid_v1/bfree_valid_ids.csv')}
    assert len(train_ids)==38661 and len(val_ids)==2000 and len(previous_ids)==10304
    all_rows = full+partial
    eval_ids = {r['coco_id'].zfill(12) for r in all_rows}
    assert eval_ids==previous_ids
    assert not train_ids & eval_ids and not val_ids & eval_ids and not train_ids & val_ids
    assert len({(r['case'],r['coco_id'],r['label']) for r in all_rows})==len(all_rows)
    assert all(r['width']=='512' and r['height']=='512' and r['input_tokens']=='1370' for r in all_rows)
    assert all(r['case']==r['generator'] and r['variant']==(r['case'] if r['label']=='1' else 'COCO_real_512') for r in all_rows)
    assert all(np.isfinite(float(r['prob_fake'])) and 0<=float(r['prob_fake'])<=1 for r in all_rows)
    for branch in state['planned_branches']:
        rows = [r for r in all_rows if r['case']==branch]
        assert len(rows)==state['saved_rows'][branch]
        real = {r['coco_id'].zfill(12) for r in rows if r['label']=='0'}
        fake = {r['coco_id'].zfill(12) for r in rows if r['label']=='1'}
        assert real==fake
        if branch in state['completed_branches']:
            assert real==previous_ids
    published = read(RUN/'metrics.csv')+read(RUN/'partial_metrics.csv')
    for r in published:
        metrics = a['computed_metrics'].get(r['branch'],a['partial_metrics'])
        for key in ['n','tn','fp','fn','tp','accuracy','balanced_accuracy','auc','ap']:
            assert abs(metrics[key]-float(r[key]))<1e-12,(r['branch'],key)
    macro = {key:float(np.mean([r[key] for r in a['computed_metrics'].values()]))
             for key in ['accuracy','balanced_accuracy','real_recall','fake_recall','auc','ap']}
    reported_macro = read(RUN/'macro_metrics.csv')[0]
    for key,v in macro.items():
        assert abs(v-float(reported_macro[key]))<1e-12
    files=[RUN/'predictions.csv',RUN/'predictions_SD2.1_inpainted_diffcat_origBG.partial.csv',split/'train_order.csv',split/'validation_ids.csv']
    checks = dict(full_branches=5,partial_branches=1,rows_saved=len(all_rows),expected_rows=state['expected_prediction_rows'],
                  unique_images_saved=len({(r['variant'],r['coco_id']) for r in all_rows}),
                  eval_unique_coco_ids=len(eval_ids),train_pool_unique_coco_ids=len(train_ids),internal_validation_ids=len(val_ids),
                  train_pool_eval_overlap=0,internal_validation_eval_overlap=0,
                  all_complete_branch_ids_match_official_valid=True,partial_pairs_match=True,
                  confusion_and_auc_ap_match_sources=True,macro_completed_five=macro,
                  sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (ROOT/'audit_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    last=a['partial_metrics']
    rows=[]
    for b,r in a['computed_metrics'].items():
        rows.append([b.replace('SD2.1_',''),'Đủ',r['n'],pct(r['balanced_accuracy']),pct(r['auc']),pct(r['fake_recall']),pct(r['real_recall']),r['fp'],r['fn']])
    rows.append(['inpainted_diffcat_origBG','CHƯA ĐỦ',last['n'],pct(last['balanced_accuracy']),pct(last['auc']),pct(last['fake_recall']),pct(last['real_recall']),last['fp'],last['fn']])
    lines=['# GSD train BFree → đánh giá BFree validation', '',
           'Nguồn: [hoangha071005/test-gsd-train-bfree-test-bfree, version 1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-train-bfree-test-bfree). Metadata, completion.json và metrics.csv được đối chiếu lại trực tiếp với API Kaggle; hai file kết quả khớp từng byte bản local.', '',
           '## Trạng thái chạy', '',
           '- status=paused, reason=time_budget; thời gian đã chạy khoảng 10,50 giờ. Tên completion.json không có nghĩa toàn bộ evaluation đã complete.',
           '- Hoàn tất 5/6 nhánh, mỗi nhánh 20.608 dự đoán. Nhánh inpainted_diffcat_origBG còn dang dở: 14.048/20.608 dòng = 7.024/10.304 cặp real/fake (68,17%). Còn thiếu 6.560 dòng, tương ứng 3.280 cặp.',
           '- Tổng đã lưu: 117.088/123.648 dự đoán theo context, tương ứng 68.848/72.128 ảnh riêng biệt. Không dùng macro 5 nhánh làm kết quả đủ 6 nhánh, không trộn partial vào macro đủ.', '',
           '## Model và dữ liệu', '',
           f"- Checkpoint: `{checkpoint['source']}`; virtual epoch {checkpoint['epoch']}, signature `{checkpoint['signature']}`.",
           '- CLIP ViT-L/14 + GSD, seed42; train riêng nhánh SD2.1_inpainted_samecat_origBG và real COCO_real_512. Đây là train BFree một nhánh, không phải recipe phối hợp sáu nhánh đầy đủ của paper B-Free.',
           '- Evaluation: official valid_list.csv, 10.304 COCO IDs; ảnh PNG 512×512, 1.370 token, ngưỡng fake probability >=0,5. Trên mỗi nhánh, số real và fake bằng nhau nên accuracy = balanced accuracy (BA).',
           '- valid_list.csv là tập con của train_list.csv trong bản phát hành. Đã tải train_order.csv và validation_ids.csv từ đúng checkpoint dataset v1: 38.661 train-pool IDs và 2.000 internal-validation IDs; giao với 10.304 IDs evaluation đều bằng 0. Kiểm tra theo COCO ID, không phải audit near-duplicate/pixel hash.',
           '- Đối chiếu từng nhánh đủ: đúng 10.304 real IDs và 10.304 fake IDs theo danh sách official valid đã lưu; partial có cùng 7.024 IDs giữa hai lớp. Không có trùng dòng (case, coco_id, label).', '',
           '## Kết quả tại ngưỡng 0,5', '',
           table(['Nhánh','Trạng thái','n','BA/Acc (%)','AUC (%)','Recall fake (%)','Recall real (%)','FP','FN'],rows), '',
           f"**Macro chỉ trên 5 nhánh đã hoàn tất:** BA/accuracy {pct(macro['balanced_accuracy'])}%; AUC {pct(macro['auc'])}%; AP {pct(macro['ap'])}%; recall fake {pct(macro['fake_recall'])}%; recall real {pct(macro['real_recall'])}%.", '',
           'FP = real bị báo nhầm fake; FN = fake bị bỏ sót. Với GSD, real được suy luận lại theo từng branch/batch context, nên không cộng FP của các nhánh rồi gọi là số ảnh real riêng biệt bị báo nhầm.', '',
           '## Nhận xét', '',
           '- Nhánh cùng kiểu với training, inpainted_samecat_origBG: BA 99,72%, AUC 99,9933%; chỉ bỏ sót 35/10.304 fake và báo nhầm 23/10.304 real. ID evaluation không nằm trong train pool/internal validation đã đối chiếu.',
           '- Chuyển sang inpainted_samecat và inpainted_diffcat vẫn tốt: BA 97,00% và 97,51%. Nhánh diffcat_origBG mới có kết quả tạm thời 98,92%; chưa suy ra kết quả toàn bộ nhánh.',
           '- Hai nhánh selfconditioned yếu hơn: BA 80,73% và 73,34%, fake recall 61,75% và 47,00%. Real recall vẫn xấp xỉ 99,7%, cho thấy ngưỡng 0,5 bỏ sót nhiều fake. AUC 99,12% và 95,47% cho thấy khả năng xếp hạng tốt hơn kết quả quyết định ở ngưỡng cố định.', '',
           '## So sánh GSD train Tiny trước đó trên các nhánh hoàn tất', '',
           table(['Nhánh','BA train Tiny (%)','BA train BFree (%)','Thay đổi (điểm %)'],
                 [[r['branch'].replace('SD2.1_',''),pct(r['old_tiny_trained_accuracy']),pct(r['new_bfree_trained_accuracy']),f"{r['accuracy_change_pp']:+.2f}"] for r in a['comparison_to_tiny_trained_gsd']]), '',
           'So sánh cùng ngưỡng 0,5 và các nhánh đủ. Không so macro 5 nhánh mới với macro 6 nhánh cũ. Train BFree cải thiện mạnh samecat_origBG (+34,87 điểm) và selfconditioned_origBG (+17,22 điểm), nhưng selfconditioned giảm 12,98 điểm; chưa thể nói tốt hơn đồng đều mọi nhánh.', '',
           '## Việc còn lại', '',
           '- Resume cùng checkpoint và fingerprint để hoàn thành 6.560 dự đoán còn lại của diffcat_origBG, rồi xuất lại macro đủ 6 nhánh.',
           '- Có thể phân tích ngưỡng cho selfconditioned sau đó; các ngưỡng diagnostic trong analysis.json dùng trực tiếp nhãn evaluation, không phải ngưỡng được xác thực trên tập độc lập.',
           '- Không audit lại toàn bộ pixel/ảnh gần trùng, không tải hoặc thực thi model weights trong lần tổng hợp này.', '',
           '## File local', '',
           f"- [analyze.py]({(ROOT/'analyze.py').as_posix()}): tính lại AUC/AP, confusion và phân bố score từ prediction.",
           f"- [audit_and_report.py]({(ROOT/'audit_and_report.py').as_posix()}): kiểm tra cohort/overlap, trạng thái partial và tạo báo cáo này.",
           f"- [audit_checks.json]({(ROOT/'audit_checks.json').as_posix()}): kết quả kiểm tra và SHA-256 nguồn.",
           f"- [analysis.json]({(ROOT/'analysis.json').as_posix()}): metrics chi tiết, phân bố điểm, so sánh từng nhánh."]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in checks.items() if k!='sha256'},indent=2))


if __name__=='__main__':
    main()
