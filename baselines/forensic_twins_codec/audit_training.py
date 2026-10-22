"""Offline descriptive audit; no changes to training/model/data."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'artifacts/training_audit_20261020'
def read(name):
    with (ROOT/name).open() as f: return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
epochs=read('epoch_metrics.csv'); steps=read('train_steps.csv')
expected=(ROOT/'encoder_final.pt.sha256').read_text().split()[0]
actual=hashlib.sha256((ROOT/'encoder_final.pt').read_bytes()).hexdigest()
assert actual==expected, 'Bad final encoder checksum'
state=torch.load(ROOT/'encoder_final.pt',map_location='cpu',weights_only=True)
assert all(torch.isfinite(v).all() for v in state['encoder'].values())
summary={
 'epochs':len(epochs),'logged_steps':len(steps),'optimizer_step':state['optimizer_step'],
 'successful_updates':sum(r['steps_in_segment'] for r in epochs),
 'attempted_updates_if_no_resume':20*1562,
 'checksum_verified':True,'finite_encoder_weights':True,
 'loss_reduction_epoch1_to20_pct':100*(1-epochs[-1]['loss']/epochs[0]['loss']),
 'loss_reduction_epoch3_to20_pct':100*(1-epochs[-1]['loss']/epochs[2]['loss']),
 'loss_reduction_epoch10_to20_pct':100*(1-epochs[-1]['loss']/epochs[9]['loss']),
 'best_training_epoch':int(min(epochs,key=lambda r:r['loss'])['epoch']),
 'late_loss_mean':np.mean([r['loss'] for r in epochs[11:]]),
 'late_loss_std_across_epochs':np.std([r['loss'] for r in epochs[11:]]),
 'logged_grad_norm_min':min(r['grad_norm'] for r in steps),
 'logged_grad_norm_max':max(r['grad_norm'] for r in steps),
 'logged_fraction_norm_above_clip1':np.mean([r['grad_norm']>1 for r in steps]),
 'epoch20_weighted_redundancy':epochs[-1]['redundancy']*5e-5,
 'policy_counts_rank0':{k:sum(r[k] for r in epochs) for k in ['local_rank0_identity','local_rank0_jpeg','local_rank0_resize_jpeg']},
 'epoch20_feature_std':epochs[-1]['feature_std'],
 'metadata_manifest_matches':state['manifest_hash']=='56023995de848527cecad2298c2ddfa074bcd06dacd7f314b6f6bf9554066f90',
}
print(json.dumps(summary,indent=2)); (ROOT/'audit_summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,2,figsize=(11,7))
x=[r['epoch'] for r in epochs]
axes[0,0].plot(x,[r['loss'] for r in epochs],label='Total')
axes[0,0].plot(x,[r['invariance'] for r in epochs],label='Invariance')
axes[0,0].plot(x,[r['redundancy']*5e-5 for r in epochs],label='Weighted redundancy')
axes[0,0].legend(); axes[0,0].set_ylabel('Epoch mean loss')
for ax,key in [(axes[0,1],'diag_mean'),(axes[1,0],'offdiag_abs_mean'),(axes[1,1],'feature_std')]:
    ax.plot(x,[r[key] for r in epochs],marker='.'); ax.set_ylabel(key)
for ax in axes.flat: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT/'audit_curves.png',dpi=160)
