"""Audit the nine-case evaluation without changing source files or tuning thresholds."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = HERE/'source/gsd_uploaded_best_variants_v1'
spec = importlib.util.spec_from_file_location('native_audit', HERE.parent/'gsd_augmented_native_analysis/analyze.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    predictions = pd.read_csv(SOURCE/'predictions.csv')
    reported = pd.read_csv(SOURCE/'metrics.csv')
    state = audit.read_json(SOURCE/'evaluation_state.json')
    fp = audit.read_json(SOURCE/'run_fingerprint.json')
    native = predictions[predictions.case == 'native'].reset_index(drop=True)
    expected_manifest = pd.read_csv(PROJECT/'reports/gsd_all7_crop560_analysis/source/seed42/test_manifest.csv')
    old = pd.read_csv(PROJECT/'reports/gsd_all7_variants_analysis/source/metrics.csv')
    old_predictions = pd.read_csv(PROJECT/'reports/gsd_all7_variants_analysis/source/predictions.csv')
    checks = dict(total_prediction_rows=len(predictions), cases=predictions.case.unique().tolist(),
        duplicate_case_ids=int(predictions.duplicated(['case','sample_id']).sum()),
        native_order_matches_saved_test_manifest=native.sample_id.tolist()==expected_manifest.sample_id.tolist(),
        order_hash_matches=audit.digest(native.sample_id.tolist())==fp['sample_order_hash'],
        finite_scores_logits=bool(np.isfinite(predictions[['prob_fake','logit_real','logit_fake','logit_margin']]).all().all()),
        max_metric_difference=0., per_case={})
    margin = predictions.logit_fake - predictions.logit_real
    checks['max_softmax_difference'] = float(np.max(np.abs(1/(1+np.exp(-margin))-predictions.prob_fake)))
    checks['max_margin_difference'] = float(np.max(np.abs(margin-predictions.logit_margin)))
    rows, distributions, pairs, comparison = [], [], [], []
    for case in state['planned_cases']:
        frame = predictions[predictions.case == case].reset_index(drop=True)
        checks['per_case'][case] = dict(n=len(frame), unique_ids=int(frame.sample_id.nunique()),
            same_order=frame.sample_id.tolist()==native.sample_id.tolist(),
            same_metadata=all(frame[k].tolist()==native[k].tolist() for k in
                ('relative_path','label','generator','pixel_hash','width','height')),
            labels=frame.groupby('label').size().to_dict(),
            generator_class_counts=frame.groupby(['generator','label']).size().tolist(),
            exact_batch_membership=bool(np.array_equal(frame.batch_id,np.arange(len(frame))//8)))
        old_frame = old_predictions[old_predictions.case == case].reset_index(drop=True)
        checks['per_case'][case]['old_raw_cohort_matches'] = (
            frame.sample_id.tolist()==old_frame.sample_id.tolist() and
            frame.label.tolist()==old_frame.label.tolist() and
            frame.batch_id.tolist()==old_frame.batch_id.tolist())
        for gen, group in [('pooled',frame),*frame.groupby('generator')]:
            metrics = audit.metrics(group)
            rows.append(dict(case=case,generator=gen,**metrics))
            saved = reported[(reported.case==case)&(reported.generator==gen)].iloc[0]
            for key in ('n','accuracy','balanced_accuracy','real_recall','fake_recall','fake_precision','fake_f1','auc','ap','tn','fp','fn','tp'):
                checks['max_metric_difference'] = max(checks['max_metric_difference'],abs(float(saved[key])-metrics[key]))
        for label, group in frame.groupby('label'):
            distributions.append(dict(case=case,label=int(label),n=len(group),
                median=float(group.prob_fake.median()),p05=float(group.prob_fake.quantile(.05)),
                p95=float(group.prob_fake.quantile(.95)),mean=float(group.prob_fake.mean()),
                median_logit_margin=float(group.logit_margin.median())))
        original_correct = (native.prob_fake>=.5).astype(int)==native.label
        changed_correct = (frame.prob_fake>=.5).astype(int)==frame.label
        pairs.append(dict(case=case,correct_to_wrong=int((original_correct&~changed_correct).sum()),
            wrong_to_correct=int((~original_correct&changed_correct).sum()),
            mean_abs_score_delta=float(np.abs(frame.prob_fake-native.prob_fake).mean()),
            real_mean_score_delta=float((frame.prob_fake-native.prob_fake)[frame.label==0].mean()),
            fake_mean_score_delta=float((frame.prob_fake-native.prob_fake)[frame.label==1].mean())))
        old_row = old[(old.case==case)&(old.generator=='pooled')].iloc[0]
        new_row = rows[-8]
        comparison.append(dict(case=case,old_accuracy=float(old_row.accuracy),new_accuracy=new_row['accuracy'],
            accuracy_delta_pp=100*(new_row['accuracy']-old_row.accuracy),old_auc=float(old_row.auc),
            new_auc=new_row['auc'],old_fake_recall=float(old_row.fake_recall),
            new_fake_recall=new_row['fake_recall']))
    result = dict(dataset='https://www.kaggle.com/datasets/hoangha071005/gsd-test-nine-cases',
        version=audit.read_json(HERE/'source/dataset_metadata.json')['currentVersionNumber'],
        state=state,fingerprint=fp,checks=checks,metrics=rows,score_distributions=distributions,
        paired_vs_native=pairs,comparison_with_old_raw=comparison,
        note='Fixed threshold0.5. Descriptive checkpoint comparison, not a controlled augmentation ablation. '
             '63,000 predictions are nine views of the same7,000 images, not63,000 independent images.')
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    table = pd.DataFrame(rows)
    pooled = table[table.generator=='pooled']
    lines = ['# GSD augmented: tổng hợp9 case', '',
        'Đã kiểm tra đủ9 case, mỗi case7.000 ảnh (3.500 real +3.500 fake), ngưỡng0,5.',
        'Đây là9 biến thể của cùng một tập ảnh, không phải63.000 ảnh độc lập.', '',
        '|Case|Accuracy / BA|Real recall|Fake recall|Fake F1|ROC-AUC|AP|FP|FN|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in pooled.to_dict('records'):
        lines.append('|'+row['case']+'|'+ '|'.join(f"{row[k]*100:.4f}%" for k in
            ('accuracy','real_recall','fake_recall','fake_f1','auc','ap'))+'|'+str(row['fp'])+'|'+str(row['fn'])+'|')
    lines += ['', '## So sánh checkpoint raw-only trước', '',
        '|Case|Raw-only accuracy|Augmented accuracy|Chênh lệch điểm %|', '|---|---:|---:|---:|']
    for row in comparison:
        lines.append(f"|{row['case']}|{row['old_accuracy']*100:.4f}%|{row['new_accuracy']*100:.4f}%|{row['accuracy_delta_pp']:+.4f}|")
    lines += ['', '## Giới hạn diễn giải', '',
        'JPEG và blur được cải thiện rõ trên cohort này. Resize vẫn là điểm yếu, đặc biệt Midjourney.',
        'Checkpoint trained all7; đây không phải unseen-generator hoặc cross-dataset evaluation.',
        'Không tối ưu ngưỡng hay fit calibration trên test. Phân tích không sửa các CSV nguồn.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('CHECKS',json.dumps(checks,indent=2,default=str))
    print('\nPOOLED\n'+pooled.to_string(index=False))
    print('\nDISTRIBUTIONS\n'+pd.DataFrame(distributions).to_string(index=False))
    print('\nCOMPARE RAW\n'+pd.DataFrame(comparison).to_string(index=False))
    print('\nMIDJOURNEY\n'+table[table.generator=='midjourney'].to_string(index=False))
    print('\nBIGGAN\n'+table[table.generator=='biggan'].to_string(index=False))


if __name__=='__main__':
    main()
