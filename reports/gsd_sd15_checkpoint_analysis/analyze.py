"""Read-only audit of downloaded Kaggle reports; write derived diagnostics separately."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / 'source'
def readj(name):
    return json.loads((SRC / name).read_text(encoding='utf-8-sig'))

history = pd.DataFrame(readj('seed42/clip_gsd/history.json'))
metrics = pd.read_csv(SRC / 'metrics.csv')
pred = pd.read_csv(SRC / 'predictions.csv')
assert not pred.duplicated(['seed', 'model', 'case', 'sample_id']).any()
def calc(g):
    y = g.label.to_numpy()
    p = g.prob_fake.to_numpy()
    pos, neg = y == 1, y == 0
    rank = pd.Series(p).rank(method='average').to_numpy()
    auc = (rank[pos].sum() - pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())
    tn, fp = int((p[neg] < .5).sum()), int((p[neg] >= .5).sum())
    fn, tp = int((p[pos] < .5).sum()), int((p[pos] >= .5).sum())
    return dict(n=len(g), auc=auc, balanced_accuracy=(tp/pos.sum()+tn/neg.sum())/2,
                fake_recall=tp/pos.sum(), real_recall=tn/neg.sum(), tn=tn, fp=fp, fn=fn, tp=tp,
                brier=float(np.mean((p-y)**2)),
                real_median=float(np.median(p[neg])), fake_median=float(np.median(p[pos])),
                confident_false_real=int((p[pos]<.01).sum()))
checks = []
diagnostics = []
for case, frame in pred.groupby('case'):
    for gen, g in [('pooled', frame), *frame.groupby('generator')]:
        values = calc(g)
        reported = metrics[(metrics.case == case)&(metrics.generator == gen)].iloc[0]
        for field in ('n','auc','balanced_accuracy','fake_recall','real_recall','tn','fp','fn','tp'):
            assert np.isclose(values[field], reported[field], atol=1e-12), (case, gen, field)
        diagnostics.append(dict(case=case, generator=gen, **values))
    checks.append(dict(case=case, sample_ids=sorted(frame.sample_id), batches=frame.set_index('sample_id').batch_id.sort_index().tolist()))
assert all(x['sample_ids']==checks[0]['sample_ids'] and x['batches']==checks[0]['batches'] for x in checks)
splits = {s:pd.read_csv(SRC / f'seed42/{s}_manifest.csv') for s in ('train','validation','test')}
overlaps = {}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    overlaps[f'{a}_{b}_pixels'] = len(set(splits[a].pixel_hash)&set(splits[b].pixel_hash))
audit = pd.read_csv(SRC / 'image_audit.csv')
base = pred[pred.case == 'native'].set_index('sample_id')
pair_rows = []
for case in ('jpeg96','size','size_jpeg96'):
    changed = pred[pred.case==case].set_index('sample_id').loc[base.index]
    for gen in sorted(base.generator.unique()):
        for label in (0,1):
            mask = (base.generator==gen)&(base.label==label)
            a,b = base.loc[mask], changed.loc[mask]
            right_a = (a.prob_fake >= .5).to_numpy()==label
            right_b = (b.prob_fake >= .5).to_numpy()==label
            pair_rows.append(dict(case=case,generator=gen,label=label,
                correct_to_wrong=int((right_a&~right_b).sum()), wrong_to_correct=int((~right_a&right_b).sum()),
                flip_rate=float((right_a!=right_b).mean()), mean_delta=float((b.prob_fake-a.prob_fake).mean())))
paired = pd.DataFrame(pair_rows)
reported_pairs = pd.read_csv(SRC/'paired_summary.csv').merge(paired,on=['case','generator','label'],suffixes=('_reported','_checked'))
assert np.allclose(reported_pairs.flip_rate_reported, reported_pairs.flip_rate_checked)
assert np.allclose(reported_pairs.mean_delta_reported, reported_pairs.mean_delta_checked)
diag = pd.DataFrame(diagnostics)
native = diag[(diag.case=='native')&(diag.generator!='pooled')]
for case, group in diag[diag.generator!='pooled'].groupby('case'):
    unseen = group[group.generator!='sd15']
    reported = pd.read_csv(SRC/'macro_unseen.csv').set_index('case').loc[case]
    assert np.isclose(unseen.auc.mean(), reported.auc)
    assert np.isclose(unseen.balanced_accuracy.mean(), reported.balanced_accuracy)
modulation = []
for row in history.to_dict('records'):
    modulation.append(dict(epoch=row['epoch'], max_a_change=max(abs(v['a']-2) for v in row['modulation'].values()),
                           max_b_change=max(abs(v['b']-1) for v in row['modulation'].values())))
result = dict(history=history.to_dict('records'), diagnostics=diagnostics, paired=pair_rows,
    verified_metrics_rows=len(metrics), prediction_rows=len(pred),
    overlaps=overlaps, split_counts={s:len(f) for s,f in splits.items()},
    test_unique_pixels=int(splits['test'].pixel_hash.nunique()),
    modulation=modulation, selected=readj('seed42/clip_gsd/selected_checkpoint.json'),
    status=readj('seed42/clip_gsd/training_status.json'),
    audit_sizes=audit.groupby(['split','generator','label'])[['width','height']].median().reset_index().to_dict('records'))
(HERE/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('TRAIN\n'+history[['epoch','train_loss','val_loss','auc','balanced_accuracy','fake_recall','real_recall','amp_skipped_steps']].to_string(index=False))
print('NATIVE\n'+native.to_string(index=False))
print('VALIDATION/COHORT AUDIT', json.dumps({k:result[k] for k in ('verified_metrics_rows','prediction_rows','overlaps','split_counts','test_unique_pixels','selected','status','modulation')}))
print('PAIRED\n'+paired.to_string(index=False))
