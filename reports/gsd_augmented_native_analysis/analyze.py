"""Read-only audit of public evaluation CSVs using bundled pandas/numpy.

Writes only derived JSON/Markdown. Source tables, models and notebooks stay intact.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = HERE/'source'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def metrics(frame):
    y = frame.label.to_numpy(dtype=int)
    p = frame.prob_fake.to_numpy(dtype=float)
    predicted = p >= .5
    tn, fp = int(((y == 0) & ~predicted).sum()), int(((y == 0) & predicted).sum())
    fn, tp = int(((y == 1) & ~predicted).sum()), int(((y == 1) & predicted).sum())
    positives, negatives = int(y.sum()), int((y == 0).sum())
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = float((ranks[y == 1].sum() - positives*(positives+1)/2) / (positives*negatives))
    order = np.argsort(-p, kind='stable')
    cum_tp = np.cumsum(y[order])
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p)-1]
    ap = float(np.sum((cum_tp[ends] / (ends+1)) * np.diff(np.r_[0, cum_tp[ends]])) / positives)
    real_recall, fake_recall = tn/negatives, tp/positives
    margin = frame.logit_fake.to_numpy() - frame.logit_real.to_numpy()
    ce = float(np.mean(np.logaddexp(0., margin) - y*margin))
    return dict(n=len(frame), accuracy=(tn+tp)/len(frame), balanced_accuracy=(real_recall+fake_recall)/2,
        real_recall=real_recall, fake_recall=fake_recall, fake_precision=tp/(tp+fp),
        fake_f1=2*tp/(2*tp+fp+fn), auc=auc, ap=ap, tn=tn, fp=fp, fn=fn, tp=tp,
        ce_from_logits=ce, brier=float(np.mean((p-y)**2)))


def main():
    native = SOURCE/'gsd_native'
    predictions_path = native/'predictions_native.csv'
    frame = pd.read_csv(predictions_path)
    reported = pd.read_csv(native/'metrics.csv')
    fingerprint = read_json(native/'run_fingerprint.json')
    checkpoint = read_json(native/'checkpoint_inspection.json')
    state = read_json(native/'evaluation_state.json')
    inventory = read_json(SOURCE/'dataset_file_inventory.json')
    duplicate = SOURCE/'gsd_native_results/gsd_native/predictions_native.csv'
    base_manifest = PROJECT/'reports/gsd_all7_crop560_analysis/source/seed42'
    expected = pd.read_csv(base_manifest/'test_manifest.csv')
    checks = dict(
        rows=len(frame), unique_ids=int(frame.sample_id.nunique()),
        duplicate_ids=int(frame.sample_id.duplicated().sum()), cases=frame.case.unique().tolist(),
        seeds=frame.seed.unique().tolist(), models=frame.model.unique().tolist(),
        labels=frame.groupby('label').size().to_dict(), generators=frame.groupby('generator').size().to_dict(),
        metadata_order_match=frame.sample_id.tolist() == expected.sample_id.tolist(),
        expected_order_hash=digest(frame.sample_id.tolist()),
        order_hash_matches_fingerprint=digest(frame.sample_id.tolist()) == fingerprint['sample_order_hash'],
        cohort_fields_match=all(frame[k].tolist() == expected[k].tolist()
            for k in ('relative_path', 'generator', 'label', 'pixel_hash', 'width', 'height')),
        batch_ids_exact=bool(np.array_equal(frame.batch_id, np.arange(len(frame))//8)),
        batch_count=int(frame.batch_id.nunique()),
        every_batch_size8=bool(frame.groupby('batch_id').size().eq(8).all()),
        score_range_valid=bool(frame.prob_fake.between(0, 1).all()),
        logits_and_scores_finite=bool(np.isfinite(frame[['prob_fake','logit_real','logit_fake','logit_margin']]).all().all()),
        tokens_match_crop560_pad14=bool(np.array_equal(frame.input_tokens,
            ((np.minimum(frame.width,560)+13)//14)*((np.minimum(frame.height,560)+13)//14)+1)),
        duplicate_export_identical=hashlib.sha256(predictions_path.read_bytes()).digest() == hashlib.sha256(duplicate.read_bytes()).digest(),
    )
    margin = frame.logit_fake.to_numpy() - frame.logit_real.to_numpy()
    probability = 1 / (1 + np.exp(-margin))
    checks['max_softmax_discrepancy'] = float(np.max(np.abs(frame.prob_fake-probability)))
    checks['max_logit_margin_discrepancy'] = float(np.max(np.abs(frame.logit_margin-margin)))
    checks['max_source_metric_discrepancy'] = 0.
    groups = [('pooled', frame), *frame.groupby('generator')]
    rows = []
    for gen, group in groups:
        row = dict(generator=gen, **metrics(group))
        old = reported[(reported.generator == gen) & (reported.case == 'native')].iloc[0]
        for key in ('n','accuracy','balanced_accuracy','real_recall','fake_recall','fake_precision','fake_f1','auc','ap','tn','fp','fn','tp'):
            error = abs(float(row[key]) - float(old[key]))
            checks['max_source_metric_discrepancy'] = max(checks['max_source_metric_discrepancy'], error)
        rows.append(row)
    checks['unique_decoded_pixel_hashes'] = int(frame.pixel_hash.nunique())
    checks['repeated_pixel_rows'] = int(frame.pixel_hash.duplicated().sum())
    overlap = {}
    for split in ('train','validation'):
        split_frame = pd.read_csv(base_manifest/f'{split}_manifest.csv', usecols=['pixel_hash'])
        overlap[split] = len(set(frame.pixel_hash) & set(split_frame.pixel_hash))
    checks['pixel_overlap_with_saved_manifests'] = overlap
    errors = frame[(frame.prob_fake >= .5).astype(int) != frame.label].copy()
    errors['error_type'] = np.where(errors.label == 0, 'false_positive_real_as_fake', 'false_negative_fake_as_real')
    distribution = {}
    for label, group in frame.groupby('label'):
        p = group.prob_fake
        distribution[str(label)] = dict(n=len(p), min=float(p.min()), p01=float(p.quantile(.01)),
            p05=float(p.quantile(.05)), median=float(p.median()), p95=float(p.quantile(.95)),
            p99=float(p.quantile(.99)), max=float(p.max()), mean=float(p.mean()),
            median_logit_margin=float(group.logit_margin.median()))
    # Independent score bins quantify ambiguity without selecting thresholds on test.
    ambiguity = {name:int(mask.sum()) for name,mask in {
        'probability_0.1_to_0.9':frame.prob_fake.between(.1,.9),
        'probability_0.4_to_0.6':frame.prob_fake.between(.4,.6),
        'confident_errors_p_wrong_ge_0.9': ((frame.label == 0)&(frame.prob_fake >= .9)) |
                                         ((frame.label == 1)&(frame.prob_fake <= .1)),
    }.items()}
    old_path = PROJECT/'reports/gsd_all7_variants_analysis/source/predictions.csv'
    old = pd.read_csv(old_path)
    old = old[old.case == 'native'].set_index('sample_id').loc[frame.sample_id].reset_index()
    assert old.label.tolist() == frame.label.tolist()
    assert old.batch_id.tolist() == frame.batch_id.tolist()
    was_correct = ((old.prob_fake >= .5).astype(int).to_numpy() == frame.label.to_numpy())
    now_correct = ((frame.prob_fake >= .5).astype(int).to_numpy() == frame.label.to_numpy())
    old_summary = pd.read_csv(PROJECT/'reports/gsd_all7_variants_analysis/source/metrics.csv')
    old_summary = old_summary[(old_summary.case == 'native')&(old_summary.generator == 'pooled')].iloc[0].to_dict()
    comparison = dict(old_native_metrics=old_summary,
        new_minus_old_accuracy_pp=(rows[0]['accuracy']-old_summary['accuracy'])*100,
        correct_to_wrong=int((was_correct & ~now_correct).sum()),
        wrong_to_correct=int((~was_correct & now_correct).sum()),
        both_wrong=int((~was_correct & ~now_correct).sum()),
        warning='Different training checkpoints; descriptive comparison, NOT controlled augmentation ablation.')
    full_cases = sorted(set(frame.case))
    missing = ['fake_jpeg96','all_jpeg96','all_jpeg90','all_jpeg80','resize_square256','resize_square560','blur_sigma1','resize256_jpeg96']
    result = dict(dataset='https://www.kaggle.com/datasets/hoangha071005/evaluate-augmented-tiny-gsd-new',
        version=read_json(SOURCE/'dataset_metadata.json')['currentVersionNumber'],
        full_cases=full_cases, missing_variant_cases=missing,
        smoke_state=read_json(SOURCE/'gsd_smoke/evaluation_state.json'),
        state=state, fingerprint=fingerprint, checkpoint=checkpoint, checks=checks,
        metrics=rows, score_distribution=distribution, ambiguity=ambiguity,
        errors=errors.to_dict('records'), old_raw_native_comparison=comparison,
        source_inventory_count=len(inventory))
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    report = ['# Augmented GSD: native evaluation audit', '',
        'Dataset v1. GPU Tesla T4, FP16 + SDPA. Checkpoint epoch1, seed42.',
        'Only native is complete (7,000 images). The 16-image smoke and duplicated native export are not independent tests.',
        'No JPEG/resize/blur results are present. status=complete refers only to the requested native case.', '',
        '|Generator|N|Accuracy|Real recall|Fake recall|AUC|AP|TN|FP|FN|TP|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append('|'+str(row['generator'])+'|'+str(row['n'])+'|'+
            '|'.join(f"{100*row[k]:.4f}%" for k in ('accuracy','real_recall','fake_recall','auc','ap'))+'|'+
            '|'.join(str(row[k]) for k in ('tn','fp','fn','tp'))+'|')
    report += ['', '## Interpretation', '',
        'Strong in-distribution discrimination at the fixed 0.5 threshold; 9 false positives and 1 false negative.',
        'This does not establish robustness to JPEG/resize/blur or unseen-generator/dataset transfer.',
        'Saved native/processed validation is from the selected checkpoint, not a full training history or proof of convergence.',
        'Old raw-model native accuracy is slightly higher; this alone neither proves overfitting nor establishes augmentation harm.',
        'No threshold was optimized on this test set. Detailed integrity checks, score quantiles and error IDs are in analysis.json.']
    (HERE/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(dict(checks=checks, metrics=rows, distribution=distribution, ambiguity=ambiguity,
        errors=errors[['generator','label','prob_fake','logit_margin','relative_path']].to_dict('records'),
        comparison=comparison), indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
