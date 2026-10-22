import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'diagnostics.json').read_text(encoding='utf-8'))
h=d['history']; epochs=[r['epoch'] for r in h]
fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
ax=axes[0,0]
ax.plot(epochs,[r['train_loss'] for r in h],'o-',label='Train')
ax.plot(epochs,[r['val_loss'] for r in h],'o-',label='Validation')
ax.axvline(1,color='#777',linestyle='--',label='Selected epoch 1')
ax.set(yscale='log',xlabel='Epoch',ylabel='Cross-entropy',title='Validation best at epoch 1; early stop at 4',xticks=epochs)
ax.legend(fontsize=8)
ax=axes[0,1]
ax.bar(epochs,[int(r['fp']+r['fn']) for r in h],color='#2563eb')
for r in h:
    ax.text(r['epoch'],r['fp']+r['fn']+.08,str(r['fp']+r['fn']),ha='center')
ax.set(xlabel='Epoch',ylabel='Incorrect validation images / 2,800',title='Validation errors remain low but do not improve',xticks=epochs,ylim=(0,5))
ax=axes[1,0]
gens=[r for r in d['metrics'] if r['generator']!='pooled']
x=np.arange(len(gens)); ax.bar(x,[r['fp'] for r in gens],color='#ea9800',label='Real classified fake')
ax.set_xticks(x,[r['generator'] for r in gens],rotation=20)
ax.set(ylabel='Test errors / 1,000 per generator',title='Native test: 2 false positives, no false negatives',ylim=(0,2))
ax.legend(fontsize=8)
ax=axes[1,1]
c=pd.read_csv(HERE/'source/token_census.csv'); c=c[c.split=='test']
bins=np.geomspace(10,1700,35)
for label,name,color in [(0,'Real','#2563eb'),(1,'Fake','#ea9800')]:
    v=c.loc[c.label==label,'tokens']
    ax.hist(v,bins=bins,weights=np.ones(len(v))*100/len(v),histtype='step',linewidth=1.8,label=name,color=color)
ax.set(xscale='log',xlabel='Tokens per image (log scale)',ylabel='Percent within class',title='Image geometry is strongly associated with the label')
ax.legend(fontsize=8)
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('CLIP + GSD | All7 center crop max 560 | seed 42 | dataset v2',fontsize=14)
fig.savefig(HERE/'training_and_test.png',dpi=170)
fig.savefig(HERE/'training_and_test.svg')
print(HERE/'training_and_test.png')
