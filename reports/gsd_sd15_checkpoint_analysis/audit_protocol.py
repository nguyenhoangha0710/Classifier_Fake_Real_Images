"""Freeze evidence and compare existing experimental protocols, without changing a model."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT/'benchmarks/tiny_jpeg_bias_three_models/tiny_jpeg_bias_three_models_results'
def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

old = csvrows(BENCH/'dataset/full_test_manifest.csv')
new = csvrows(HERE/'source/seed42/test_manifest.csv')
keys = lambda rows: {(r['relative_path'],int(r['label'])) for r in rows}
cohort = dict(benchmark_n=len(old), gsd_n=len(new), intersection=len(keys(old)&keys(new)),
              benchmark_only=len(keys(old)-keys(new)),gsd_only=len(keys(new)-keys(old)))
overall = csvrows(BENCH/'metrics/overall_by_model_case.csv')
paired = csvrows(BENCH/'metrics/paired_probability_shift.csv')
summaries=[]
for model in ('clip_linear_probe','npr_resnet18','aide_original_full'):
    a=next(r for r in overall if r['model']==model and r['case']=='raw')
    b=next(r for r in overall if r['model']==model and r['case']=='fake_jpeg96')
    for row in (a,b):
        tn,fp,fn,tp=[int(row[x]) for x in ('tn','fp','fn','tp')]
        assert tn+fp+fn+tp==int(row['num_samples'])
        assert abs((tn+tp)/(tn+fp+fn+tp)-float(row['accuracy']))<1e-12
    relevant=[r for r in paired if r['model']==model and r['case']=='fake_jpeg96']
    real=[r for r in relevant if int(r['label'])==0]
    assert all(float(r['fake_probability_delta'])==0 for r in real)
    transitions={}
    for label in (0,1):
        group=[r for r in relevant if int(r['label'])==label]
        transitions[str(label)]=dict(
            correct_to_wrong=sum(int(r['base_predicted_label'])==label and int(r['case_predicted_label'])!=label for r in group),
            wrong_to_correct=sum(int(r['base_predicted_label'])!=label and int(r['case_predicted_label'])==label for r in group))
    summaries.append(dict(model=model,raw_accuracy=float(a['accuracy']),jpeg_accuracy=float(b['accuracy']),
        accuracy_drop_pp=(float(a['accuracy'])-float(b['accuracy']))*100,
        raw_auc=float(a['roc_auc']),jpeg_auc=float(b['roc_auc']),
        raw_fake_recall=float(a['fake_recall']),jpeg_fake_recall=float(b['fake_recall']),
        raw_real_recall=float(a['real_recall']),jpeg_real_recall=float(b['real_recall']),
        paired_transitions=transitions))
files=[HERE/'source'/p for p in (
    'config.json','seed42/train_manifest.csv','seed42/validation_manifest.csv',
    'seed42/test_manifest.csv','seed42/clip_gsd/selected_checkpoint.json')]
files += [BENCH/p for p in ('config.json','provenance/checkpoints.json','provenance/codec_transform.json',
                           'metrics/overall_by_model_case.csv','dataset/full_test_manifest.csv')]
files += [ROOT/'baselines/clip_gsd_tiny_kaggle/runtime.py',
          ROOT/'benchmarks/tiny_jpeg_bias_three_models/test_tiny_jpeg_bias_three_models_kaggle.py']
lock=dict(protocol_id='sd15_gsd_ablation_v1',date='2026-10-20',
    purpose='Isolate GSD contribution, separately from historical all-generator baselines',
    source='sd15', models=['clip_ft','clip_gsd'], backbone='openai/clip-vit-large-patch14',
    trainable='full vision backbone and identical 1024-to-2 classifier',pooling='patch_mean_post_ln',
    seed=42,manifest_policy='Use exact saved train/validation/test files, do not resample',
    threshold=.5,checkpoint_selection='minimum source-validation cross entropy',
    early_stopping=dict(patience=3,min_delta=.001,max_epochs=50),
    train_batch=128,test_batch=512,micro_batch=2,anchor_micro_batch=8,
    optimizer_policy='Same optimizer and augmentation config as saved SD15 run in both arms',
    evaluation_cases=dict(native='No codec transformation',
        fake_jpeg96='Label-dependent diagnostic only: real unchanged, fake reencoded JPEG Q96',
        all_jpeg96='Both labels reencoded JPEG Q96'),
    codec=dict(quality=96,subsampling=0,optimize=False,progressive=False,order='before model preprocessing'),
    batch_policy='Same label-blind ordered sample IDs and logical batch IDs for every case/model',
    headline_metrics=['macro_unseen_AUC','macro_unseen_balanced_accuracy','per_generator_fake_recall','per_generator_real_recall'],
    test_policy='Existing test already inspected; exploratory, no test-based tuning; confirm on untouched cohort',
    historical_baselines='ViT-B/32 linear probe, NPR and AIDE are contextual, not causal GSD controls',
    implementation_gaps=['fake_jpeg96 is not in current GSD runtime; requires actual inference, not mixing existing scores',
                         'No occupancy/lambda/gradient telemetry in historical run',
                         'No author-code parity confirmation'],
    evidence_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files},
    cohort_comparison=cohort,paired_jpeg_results=summaries)
(HERE/'protocol_lock_v1.json').write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(cohort=cohort,results=summaries),indent=2))
