"""Reconcile All7 v2 saved reports; no loading or executing .pt checkpoints."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SRC=HERE/'source'
def js(name):
    return json.loads((SRC/name).read_text(encoding='utf-8-sig'))

cfg=js('config.json'); history=pd.DataFrame(js('seed42/clip_gsd/history.json'))
pred=pd.read_csv(SRC/'predictions.csv'); reported=pd.read_csv(SRC/'metrics.csv')
assert set(pred.case)=={'native'} and set(pred.model)=={'clip_gsd'}
assert not pred.duplicated(['seed','model','case','sample_id']).any()
def metrics(g):
    y=g.label.to_numpy(); p=g.prob_fake.to_numpy(); pos=y==1; neg=y==0
    ranks=pd.Series(p).rank(method='average').to_numpy()
    auc=(ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())
    thresholds=pd.DataFrame({'p':p,'positive':pos.astype(int)}).groupby('p').agg(positive=('positive','sum'),n=('positive','size')).sort_index(ascending=False)
    ap=float((thresholds.positive/pos.sum()*(thresholds.positive.cumsum()/thresholds.n.cumsum())).sum())
    tp=int((p[pos]>=.5).sum()); fn=int(pos.sum()-tp); tn=int((p[neg]<.5).sum()); fp=int(neg.sum()-tn)
    return dict(n=len(g),auc=float(auc),ap=ap,accuracy=(tp+tn)/len(g),
        balanced_accuracy=(tp/pos.sum()+tn/neg.sum())/2,real_recall=tn/neg.sum(),fake_recall=tp/pos.sum(),
        fake_precision=tp/(tp+fp),fake_f1=2*tp/(2*tp+fp+fn),tn=tn,fp=fp,fn=fn,tp=tp,
        brier=float(np.mean((p-y)**2)),real_score_median=float(np.median(p[neg])),fake_score_median=float(np.median(p[pos])))
rows=[]
for gen,g in [('pooled',pred),*pred.groupby('generator')]:
    actual=metrics(g); saved=reported[reported.generator==gen].iloc[0]
    for field in ('n','auc','ap','accuracy','balanced_accuracy','real_recall','fake_recall','tn','fp','fn','tp'):
        assert np.isclose(actual[field],saved[field],atol=1e-12), (gen,field)
    rows.append(dict(generator=gen,**actual))
frames={s:pd.read_csv(SRC/f'seed42/{s}_manifest.csv') for s in ('train','validation','test')}
overlaps={}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    for key in ('pixel_hash','file_hash','sample_id'):
        overlaps[f'{a}_{b}_{key}']=len(set(frames[a][key])&set(frames[b][key]))
assert all(v==0 for v in overlaps.values()),overlaps
test=frames['test']; assert set(test.sample_id)==set(pred.sample_id)
joined=pred.merge(test[['sample_id','label','generator','relative_path','pixel_hash']],on='sample_id',validate='one_to_one',suffixes=('','_manifest'))
assert (joined.label==joined.label_manifest).all()
assert (joined.generator==joined.generator_manifest).all()
assert (joined.pixel_hash==joined.pixel_hash_manifest).all()
errors=joined[(joined.prob_fake>=.5).astype(int)!=joined.label]
selected=js('seed42/clip_gsd/selected_checkpoint.json'); status=js('seed42/clip_gsd/training_status.json')
assert selected['epoch']==int(history.loc[history.val_loss.idxmin(),'epoch'])
assert len(history)==status['epoch']
stopped=history.early_stopping.iloc[-1]
if isinstance(stopped,str):
    stopped=json.loads(stopped)
census=pd.read_csv(SRC/'token_census.csv')
assert census.tokens.max()<=1601
assert ((census.input_width//14)*(census.input_height//14)+1==census.tokens).all()
assert (census.crop_width<=560).all() and (census.crop_height<=560).all()
token_summary=census.groupby(['split','generator','label']).tokens.agg(['min','median','max']).reset_index()
shape_checks=[]
for features in [['tokens'],['input_width','input_height']]:
    training=frames['train'][['sample_id','label']].merge(census[['sample_id',*features]],on='sample_id',validate='one_to_one')
    # Diagnostic classifier: majority label for a shape seen in TRAIN, real for unseen/tied shapes.
    # No pixel content, generator identifier, or test-label tuning is used.
    means=training.groupby(features).label.mean().rename('train_fake_fraction').reset_index()
    for split in ('validation','test'):
        probe=frames[split][['sample_id','label']].merge(census[['sample_id',*features]],on='sample_id',validate='one_to_one')
        probe=probe.merge(means,on=features,how='left',validate='many_to_one')
        probe['prob_fake']=(probe.train_fake_fraction.fillna(0.)>.5).astype(float)
        vals=metrics(probe)
        shape_checks.append(dict(features=features,split=split,accuracy=vals['accuracy'],
            real_recall=vals['real_recall'],fake_recall=vals['fake_recall'],
            unseen_shapes=int(probe.train_fake_fraction.isna().sum())))
crop_summary=dict(images=len(census),cropped=int(((census.width>560)|(census.height>560)).sum()),
    min_tokens=int(census.tokens.min()),median_tokens=float(census.tokens.median()),max_tokens=int(census.tokens.max()))
old_manifest=HERE.parent/'gsd_sd15_checkpoint_analysis/source/seed42/test_manifest.csv'
old_overlap=None
if old_manifest.exists():
    old=pd.read_csv(old_manifest)
    old_overlap=len(set(old.relative_path)&set(test.relative_path))
result=dict(dataset_version=js('dataset_metadata.json')['currentVersionNumber'],config=cfg,
    history=history.to_dict('records'),selected=selected,status=status,metrics=rows,
    split_counts={s:len(f) for s,f in frames.items()},
    group_counts={s:f.groupby(['generator','label']).size().to_dict() for s,f in frames.items()},
    overlaps=overlaps,old_sd15_test_path_overlap=old_overlap,
    errors=errors[['generator','label','prob_fake','relative_path','width','height']].to_dict('records'),
    token_summary=token_summary.to_dict('records'),crop_summary=crop_summary,shape_only_diagnostics=shape_checks,
    train_validation_seconds=float(history.seconds.sum()),amp_skipped_steps=int(history.amp_skipped_steps.sum()),
    source_sha256={str(f.relative_to(SRC)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in SRC.rglob('*') if f.is_file()})
# JSON keys must be strings; group labels remain explicit in human-readable tables.
result['group_counts']={s:[dict(generator=g,label=int(y),n=int(n)) for (g,y),n in counts.items()]
                        for s,counts in result['group_counts'].items()}
(HERE/'diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('HISTORY\n'+history[['epoch','train_loss','val_loss','seconds','amp_skipped_steps','accuracy','auc','real_recall','fake_recall']].to_string(index=False))
print('CHECKPOINT',selected,status)
print('METRICS\n'+pd.DataFrame(rows).to_string(index=False))
print('AUDIT',json.dumps({k:result[k] for k in ('split_counts','overlaps','old_sd15_test_path_overlap','errors','crop_summary','train_validation_seconds','amp_skipped_steps')},ensure_ascii=False))
print('TOKEN SUMMARY\n'+token_summary.to_string(index=False))
print('SHAPE-ONLY DIAGNOSTICS',json.dumps(shape_checks))
