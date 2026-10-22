"""Post-hoc threshold diagnostics from saved scores. Never changes model/test source files.

Threshold selection here uses TEST labels: descriptive in-sample optima, NOT validated
deployment thresholds. Equivalent score values are always moved as a complete group.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCOPE = 'posthoc_selected_and_evaluated_on_test_not_validation'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rates(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    real = tn / (tn + fp)
    fake = tp / (tp + fn)
    return dict(n=int(n), accuracy=float((tn+tp)/n), balanced_accuracy=float((real+fake)/2),
                real_recall=float(real), fake_recall=float(fake), fpr=float(1-real),
                fake_precision=float(tp/(tp+fp)) if tp+fp else 0.,
                fake_f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def at_threshold(labels, scores, threshold):
    y = np.asarray(labels); z = np.asarray(scores) >= threshold
    return dict(threshold=float(threshold), **rates(
        int(((y == 0) & ~z).sum()), int(((y == 0) & z).sum()),
        int(((y == 1) & ~z).sum()), int(((y == 1) & z).sum())))


def sweep(labels, scores):
    y = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64)
    if not len(y) or len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Both real and fake labels required')
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Expected finite probabilities in [0, 1]')
    order = np.argsort(-scores, kind='stable'); s = scores[order]; y = y[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    positives = int(y.sum()); negatives = len(y)-positives
    tp = np.r_[0, y.cumsum()[ends]]
    fp = np.r_[0, ends+1-tp[1:]]
    fn = positives-tp; tn = negatives-fp
    thresholds = np.r_[np.nextafter(s[0], np.inf), s[ends]]
    precision = np.divide(tp, tp+fp, out=np.zeros(len(tp), dtype=float), where=(tp+fp)>0)
    return pd.DataFrame(dict(threshold=thresholds, n=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
        accuracy=(tp+tn)/len(y), balanced_accuracy=.5*(tp/positives+tn/negatives),
        real_recall=tn/negatives, fake_recall=tp/positives, fpr=fp/negatives,
        fake_precision=precision, fake_f1=2*tp/(2*tp+fp+fn)))


def choose(curve, objective, max_fpr=None):
    eligible = curve if max_fpr is None else curve[curve.fpr <= max_fpr]
    # Descending thresholds: exact objective ties select the highest threshold.
    if objective == 'balanced_accuracy':
        positives = int(curve.iloc[0].fn); negatives = int(curve.iloc[0].tn)
        score = eligible.tp.to_numpy()*negatives + eligible.tn.to_numpy()*positives
    elif objective == 'fake_f1':
        score = eligible.fake_f1.to_numpy()
    elif objective == 'fake_recall':
        score = eligible.tp.to_numpy()
    else:
        raise ValueError(objective)
    row = eligible.iloc[int(np.argmax(score))]
    result = dict(threshold=float(row.threshold),
        **rates(int(row.tn), int(row.fp), int(row.fn), int(row.tp)))
    original_index = int(row.name)
    result['equivalent_threshold_interval'] = dict(
        lower_exclusive=float(curve.iloc[original_index+1].threshold) if original_index+1 < len(curve) else None,
        upper_inclusive=float(row.threshold), decision='prob_fake >= threshold')
    return result


def analyze():
    source = HERE/'source/predictions.csv'; original_sha = sha256(source)
    predictions = pd.read_csv(source, float_precision='round_trip')
    saved = pd.read_csv(HERE/'source/metrics.csv')
    base = predictions[predictions['case'] == 'native'].reset_index(drop=True)
    assert len(predictions) == 70000 and len(base) == 7000 and base.sample_id.is_unique
    all_cases = {}; quantiles = {}; baseline_error = 0.
    fixed = [.5, .1, .01, .001, .0005, .0002, .0001, .00005, .00001, 0.]
    for name, frame in predictions.groupby('case', sort=False):
        frame = frame.reset_index(drop=True)
        assert len(frame) == 7000
        for column in ('sample_id', 'label', 'generator', 'batch_id'):
            assert frame[column].tolist() == base[column].tolist(), (name, column)
        assert frame.groupby(['generator', 'label']).size().eq(500).all()
        y = frame.label.to_numpy(); s = frame.prob_fake.to_numpy()
        curve = sweep(y, s)
        original = at_threshold(y, s, .5)
        expected = saved[(saved['case'] == name) & (saved.generator == 'pooled')].iloc[0]
        for field in original:
            if field != 'threshold':
                reference = (1-expected.real_recall) if field == 'fpr' else expected[field]
                baseline_error = max(baseline_error, abs(original[field]-reference))
                assert abs(original[field]-reference) < 1e-12, (name, field)
        all_cases[name] = dict(n=len(frame), distinct_threshold_states=len(curve),
            baseline_0_5=original, best_balanced_accuracy=choose(curve, 'balanced_accuracy'),
            best_fake_f1=choose(curve, 'fake_f1'),
            recall_at_fpr_le_1pct=choose(curve, 'fake_recall', .01),
            recall_at_fpr_le_5pct=choose(curve, 'fake_recall', .05),
            auc=float(expected.auc), ap=float(expected.ap),
            fixed_thresholds=[at_threshold(y, s, t) for t in fixed])
        for selection in ('best_balanced_accuracy', 'best_fake_f1', 'recall_at_fpr_le_1pct', 'recall_at_fpr_le_5pct'):
            result = all_cases[name][selection]
            independent = at_threshold(y, s, result['threshold'])
            for field in independent:
                assert independent[field] == result[field], (name, selection, field)
        quantiles[name] = {str(label): {
            str(q): float(np.quantile(s[y == label], q)) for q in (0, .1, .25, .5, .75, .9, 1)
        } for label in (0, 1)}

    jpeg_threshold = all_cases['fake_jpeg96']['best_balanced_accuracy']['threshold']
    shared = {}; generators = {}
    for name, frame in predictions.groupby('case', sort=False):
        shared[name] = at_threshold(frame.label, frame.prob_fake, jpeg_threshold)
    for gen, frame in predictions[predictions['case'] == 'fake_jpeg96'].groupby('generator'):
        generators[gen] = dict(baseline_0_5=at_threshold(frame.label, frame.prob_fake, .5),
            common_jpeg96_threshold=at_threshold(frame.label, frame.prob_fake, jpeg_threshold),
            generator_specific_posthoc_optimum=choose(sweep(frame.label, frame.prob_fake), 'balanced_accuracy'))
    assert sha256(source) == original_sha, 'Input CSV changed unexpectedly'
    result = dict(analysis_scope=SCOPE,
        warning='Every optimized threshold uses test labels and is evaluated in-sample; not a fair independently tuned benchmark or deployment recommendation.',
        source=str(source), source_sha256=original_sha, source_preserved=True,
        score='prob_fake, final class-1 softmax probability; classification is score >= threshold',
        tie_handling='Move all equal scores together; choose highest threshold if objective ties',
        precision_when_no_predicted_fake=0., baseline_max_abs_error=baseline_error,
        cases=all_cases, score_quantiles=quantiles,
        shared_fake_jpeg96_threshold=jpeg_threshold,
        shared_fake_jpeg96_threshold_across_cases=shared,
        fake_jpeg96_per_generator=generators)
    target = HERE/'threshold_analysis.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


def report(result):
    def line(name, r):
        return (f'{name}: t={r["threshold"]:.17g}; BA={r["balanced_accuracy"]*100:.4f}; '
                f'real={r["real_recall"]*100:.4f}; fake={r["fake_recall"]*100:.4f}; '
                f'precision={r["fake_precision"]*100:.4f}; F1={r["fake_f1"]*100:.4f}; '
                f'CM=[{r["tn"]},{r["fp"]};{r["fn"]},{r["tp"]}]')
    print(result['warning'])
    for name, case in result['cases'].items():
        print(line(name+' BEST BA', case['best_balanced_accuracy']))
    print('FAKE JPEG96 THRESHOLD COMPARISON')
    case = result['cases']['fake_jpeg96']
    for strategy in ('baseline_0_5', 'best_balanced_accuracy', 'best_fake_f1', 'recall_at_fpr_le_1pct', 'recall_at_fpr_le_5pct'):
        print(line(strategy, case[strategy]))
    for row in case['fixed_thresholds']:
        print(line('fixed', row))
    print('FAKE JPEG96 COMMON THRESHOLD PER GENERATOR')
    for gen, rows in result['fake_jpeg96_per_generator'].items():
        print(line(gen, rows['common_jpeg96_threshold']))
    print('APPLY THAT SAME JPEG96 THRESHOLD TO ALL CASES')
    for name, row in result['shared_fake_jpeg96_threshold_across_cases'].items():
        print(line(name, row))
    print('Source unchanged:', result['source_preserved'], 'baseline error:', result['baseline_max_abs_error'])


def markdown_report(result):
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |',
                          '| '+' | '.join(['---']*len(headers))+' |',
                          *['| '+' | '.join(map(str, row))+' |' for row in rows]])
    def percent(value):
        return f'{value*100:.2f}'
    def metric_row(label, row):
        return [label, format(row['threshold'], '.17g'), *[percent(row[k]) for k in
                ('accuracy', 'real_recall', 'fake_recall', 'fake_precision', 'fake_f1')],
                f"[{row['tn']} {row['fp']}; {row['fn']} {row['tp']}] (real/fake)"]
    headers = ['Strategy', 'Threshold', 'Acc=BA (%)', 'Real recall (%)', 'Fake recall (%)',
               'Fake precision (%)', 'Fake F1 (%)', 'Confusion matrix']
    jpeg = result['cases']['fake_jpeg96']
    strategies = [('Original threshold', jpeg['baseline_0_5']),
                  ('Best BA / Youden on TEST', jpeg['best_balanced_accuracy']),
                  ('Best fake F1 on TEST', jpeg['best_fake_f1']),
                  ('Highest recall with FPR <=1% on TEST', jpeg['recall_at_fpr_le_1pct']),
                  ('Highest recall with FPR <=5% on TEST', jpeg['recall_at_fpr_le_5pct'])]
    sections = ['# Post-hoc threshold sweep: GSD All7 image variants',
        '**TEST-LABEL SELECTION. These are in-sample descriptive optima, not independently validated deployment thresholds.**',
        'No model inference or retraining. Original predictions/config/model were not modified. '
        'Fake is label 1. Final score is `prob_fake` (class-1 softmax probability). '
        'Predict fake iff `prob_fake >= threshold`. Equal scores move as a complete group. '
        'For objective ties choose the highest threshold. Precision is recorded as zero if no fake is predicted.',
        'Source: '+result['source'], 'SHA-256: `'+result['source_sha256']+'`',
        'Validated 70,000 rows, ten cases, identical ordered 7,000 IDs and logical batch membership. '
        'Each case has 3,500 real and 3,500 fake; each generator has 500+500 images. '
        'Baseline at 0.5 agrees with saved metrics; max absolute difference '+str(result['baseline_max_abs_error'])+'.',
        '## Fake-only JPEG96: threshold objectives',
        table(headers, [metric_row(name, row) for name, row in strategies]),
        'ROC-AUC stays '+percent(jpeg['auc'])+'%; AP stays '+percent(jpeg['ap'])+
        '%. Changing the decision threshold does not change the underlying ranking/scores.',
        '## Fake-only JPEG96: fixed thresholds',
        table(headers, [metric_row('fixed', row) for row in jpeg['fixed_thresholds']]),
        '## Fake-only JPEG96: ONE shared threshold, per generator',
        'Shared threshold selected on the pooled JPEG96 TEST set: `'+format(result['shared_fake_jpeg96_threshold'], '.17g')+'`. '
        'The following table does NOT select different thresholds per generator.',
        table(headers, [metric_row(gen, rows['common_jpeg96_threshold'])
                        for gen, rows in result['fake_jpeg96_per_generator'].items()]),
        '## Independently optimized threshold for each case (TEST oracle)',
        'Each row selects a different threshold using labels of that same case. '
        'These results are not a single globally calibrated model.',
        table(headers, [metric_row(name, rows['best_balanced_accuracy'])
                        for name, rows in result['cases'].items()]),
        '## Apply the SAME JPEG96 threshold to every case',
        'This diagnoses transfer only. Do not replace the original global threshold based on the JPEG96 optimum.',
        table(headers, [metric_row(name, row) for name, row in result['shared_fake_jpeg96_threshold_across_cases'].items()]),
        '## Interpretation',
        'The large JPEG96 gain confirms a poor operating point at 0.5, but discrimination has still degraded '
        '(native AUC nearly 100%, JPEG96 AUC 94.17%). JPEG90/80 remain substantially impaired even at their '
        'in-sample best thresholds. The JPEG96 threshold transfers poorly to blur/resize. '
        'Use an independent validation/calibration set matching intended conditions, freeze its threshold, '
        'and evaluate on a fresh untouched test set before claiming a robustness improvement or comparing '
        'against CLIP/AIDE with thresholds selected under a different protocol.',
        'The JSON companion retains all thresholds at full floating-point precision, score quantiles, '
        'and explicitly labeled generator-specific oracle optima for diagnostics only.']
    return '\n\n'.join(sections)+'\n'


if __name__ == '__main__':
    result = analyze()
    report(result)
    (HERE/'threshold_report.md').write_text(markdown_report(result), encoding='utf-8')
