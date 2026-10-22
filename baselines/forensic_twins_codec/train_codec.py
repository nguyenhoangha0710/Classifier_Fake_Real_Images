"""Encoder-only codec fine-tuning; native modules are bundled by the notebook builder."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

HERE = Path(__file__).resolve().parent
NATIVE = HERE / 'native' if (HERE / 'native').exists() else HERE.parent / 'forensic_twins_original'
sys.path.insert(0, str(NATIVE))
from src.core import ForensicResidualExtractor, ForensicTwins, barlow_twins_loss
from src.data import read_manifest, discover_imagenet_train, discover_coco_roots, _transpose, _sample_disjoint_boxes, _extract_halo
from src.checkpointing import atomic_torch_save, sha256_file, load_checkpoint_cpu

SLUG = 'hoangha071005/forensic-twins-codec-checkpoint-encoder'
MANIFEST_HASH = '56023995de848527cecad2298c2ddfa074bcd06dacd7f314b6f6bf9554066f90'

def dataset_root(root, owner, slug):
    candidates = [root / slug, root / 'datasets' / owner / slug]
    found = [p for p in candidates if p.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one mount for {owner}/{slug}: {candidates}; use explicit paths')
    return found[0]

def codec(image, rng, mode='native_random'):
    """Full-RGB intervention before crop; return original-size image and audit record."""
    u = rng.random() if mode == 'native_random' else 1.0
    record = {'policy': 'identity', 'quality': None, 'scale': None, 'subsampling': None}
    if u >= .85:
        return image, record
    if u >= .5:
        scale = rng.uniform(.75, 1.)
        method = rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
        w, h = image.size
        image = image.resize((max(1, round(w*scale)), max(1, round(h*scale))), method).resize((w,h), method)
        record.update(policy='resize_jpeg', scale=scale, interpolation=method.name)
    else:
        record['policy'] = 'jpeg'
    q, sub = rng.randint(80,100), rng.choice([0,1,2])
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=q, subsampling=sub)
    buf.seek(0)
    with Image.open(buf) as decoded:
        result = decoded.convert('RGB')
    record.update(quality=q, subsampling=sub)
    return result, record

class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, key):
        index, epoch, step = key
        return self.rows[index], epoch, step

class Batches:
    """Step-keyed RNG makes augmentation reproducible across worker counts and resume."""
    def __init__(self, sampler, batch, epoch, start=0):
        self.sampler, self.batch, self.epoch, self.start = sampler, batch, epoch, start
    def __len__(self): return len(self.sampler)//self.batch - self.start
    def __iter__(self):
        items = list(self.sampler)
        for step in range(self.start, len(items)//self.batch):
            yield [(i,self.epoch,step) for i in items[step*self.batch:(step+1)*self.batch]]

class Views:
    def __init__(self, seed=42, mode='native_random'): self.seed, self.mode = seed, mode
    def __call__(self, entries):
        epoch, step = entries[0][1:]
        size_rng = random.Random(self.seed + epoch*1000000 + step)
        a, b = size_rng.choice([64,96,128]), size_rng.choice([64,96,128])
        # Interior of min-side256 is242 after excluding the FRE radius on both sides.
        if a+b > 242: a = 96
        first, second, records = [], [], []
        for row, _, _ in entries:
            seed = int(hashlib.sha256(f'{self.seed}:{epoch}:{step}:{row["sample_id"]}'.encode()).hexdigest()[:16],16)
            rng = random.Random(seed)
            with Image.open(row['image_path']) as src: image = src.convert('RGB')
            rotation = rng.randrange(4)
            if rotation: image = _transpose(image, f'rot{rotation*90}')
            for op in ['flip_h','flip_v']:
                if rng.random()<.5: image = _transpose(image,op)
            altered, record = codec(image,rng,self.mode)
            x,y = _sample_disjoint_boxes(rng,image.width-14,image.height-14,(a,a),(b,b))
            x,y = (x[0]+7,x[1]+7,x[2],x[3]), (y[0]+7,y[1]+7,y[2],y[3])
            for im,box,dest in [(image,x,first),(altered,y,second)]:
                gray = torch.from_numpy(np.array(im.convert('L'),dtype=np.float32,copy=True))[None]/255
                dest.append(_extract_halo(gray,box,7))
            records.append({'sample_id':row['sample_id'],'box1':x,'box2':y,**record})
        return torch.stack(first),torch.stack(second),records

def plot_logs(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = output/'logs/train_steps.csv'
    if not p.exists(): return
    with p.open() as f: rows=list(csv.DictReader(f))
    if not rows: return
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,key in zip(axes.flat,['loss','invariance','redundancy','lr']):
        ax.plot([int(r['step']) for r in rows],[float(r[key]) for r in rows],linewidth=1)
        ax.set(xlabel='Optimizer step',ylabel=key); ax.grid(alpha=.2)
    fig.tight_layout(); (output/'figures').mkdir(exist_ok=True)
    fig.savefig(output/'figures/training_curves.png',dpi=150); plt.close(fig)
    epoch_file=output/'logs/epoch_metrics.csv'
    if epoch_file.exists():
        with epoch_file.open() as f: epochs=list(csv.DictReader(f))
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot([int(r['epoch']) for r in epochs],[float(r['loss']) for r in epochs],marker='o')
        ax.set(xlabel='Epoch',ylabel='Mean training loss',title='Epoch mean (resumed epochs: remaining segment)'); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output/'figures/epoch_loss.png',dpi=150); plt.close(fig)

def append(path,row):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists: w.writeheader()
        w.writerow(row)

def publish(output):
    """Version only this run's explicitly authorized destination. Never erase Working."""
    stage=output/'publish'; stage.mkdir(exist_ok=True)
    for name in ['last.ckpt','last.ckpt.sha256','encoder_final.pt','encoder_final.pt.sha256','fre.pt','train_real_200k.csv','config.json','run_metadata.json']:
        if (output/name).exists(): shutil.copy2(output/name,stage/name)
    for name in ['logs','figures']:
        shutil.make_archive(str(stage/name),'zip',output,name)
    (stage/'dataset-metadata.json').write_text(json.dumps({'id':SLUG,'title':'Forensic Twins Codec Checkpoint Encoder','licenses':[{'name':'CC-BY-NC-SA-4.0'}]}))
    command=['kaggle','datasets','version','-p',str(stage),'-m','Codec encoder training checkpoint','-r','zip']
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=180)
        # Creation is allowed only if Kaggle explicitly reports a missing dataset.
        if result.returncode and ('404' in result.stderr or 'not found' in result.stderr.lower()):
            result=subprocess.run(['kaggle','datasets','create','-p',str(stage),'-r','zip'],capture_output=True,text=True,timeout=180)
        print('[publish] '+('OK' if result.returncode==0 else 'WARNING; Working retained: '+result.stderr[-600:]),flush=True)
    except (OSError,subprocess.TimeoutExpired) as exc: print(f'[publish] WARNING; Working retained: {exc}',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--output-root',type=Path,default=Path('/kaggle/working/forensic_twins_codec'))
    p.add_argument('--native-root',type=Path)
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--mode',choices=['native_random','native_native'],default='native_random')
    p.add_argument('--no-publish',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(); out=args.output_root; out.mkdir(parents=True,exist_ok=True)
    native=args.native_root or dataset_root(args.input_root,'hoangha071005','twins-gmm-200k-checkpoint')
    bundle=native/'forensic_twins_v2_200k'
    manifest=bundle/'manifests/train_real_200k.csv'; fre_path=bundle/'cache/fre.pt'
    initial=bundle/'artifacts/checkpoints/last.ckpt'
    if sha256_file(manifest)!=MANIFEST_HASH: raise ValueError('Unexpected train200K manifest hash')
    rows=read_manifest(manifest)
    if len(rows)!=200000 or Counter(r['source'] for r in rows)!=Counter(coco2017=100000,imagenet1k=100000): raise ValueError('Wrong cohort')
    if {int(r['label']) for r in rows}!={0} or len({r['sample_id'] for r in rows})!=200000: raise ValueError('Invalid labels/duplicate IDs')
    imroot=discover_imagenet_train(args.input_root); coroot=discover_coco_roots(args.input_root)
    for r in rows:
        old=Path(r['image_path'])
        if r['source']=='imagenet1k': target=imroot/(r.get('class_id') or old.parent.name)/old.name
        else: target=coroot[r.get('original_split') or 'train2017']/old.name
        if not target.is_file(): raise FileNotFoundError(target)
        if min(int(r['width']),int(r['height']))<256: raise ValueError('Manifest min-side below256')
        r['image_path']=str(target)
    payload=load_checkpoint_cpu(initial)
    if payload['manifest_hash']!=MANIFEST_HASH or payload['fre_hash']!=sha256_file(fre_path): raise ValueError('Native manifest/FRE identity mismatch')
    print(f'[preflight] 200000 real paths OK; native epoch={payload["epoch"]}, step={payload["optimizer_step"]}',flush=True)
    if args.preflight_only: return
    rank=int(os.environ.get('RANK',0)); local=int(os.environ.get('LOCAL_RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    if world!=2 or not torch.cuda.is_available(): raise RuntimeError('Use torchrun --standalone --nproc_per_node=2 on Kaggle 2xT4')
    torch.cuda.set_device(local); dist.init_process_group('nccl'); device=torch.device('cuda',local)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    cfg={'profile':'codec_encoder_v1','mode':args.mode,'seed':42,'epochs':20,'batch_per_gpu':64,'global_batch':128,'lr':6e-5,'min_lr':1e-6,'warmup_epochs':2,'weight_decay':.0025,'lambda_bt':5e-5,'eps_bt':1e-4,'native_sha256':sha256_file(initial),'manifest_hash':MANIFEST_HASH,'fre_hash':sha256_file(fre_path),'policy_version':1}
    identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    model=ForensicTwins(ForensicResidualExtractor.from_pretrained(fre_path)).to(device)
    model.encoder.load_state_dict(payload['encoder'],strict=True)
    model.projector.load_state_dict(payload['projector'],strict=True)
    del payload
    optim=torch.optim.AdamW((q for q in model.parameters() if q.requires_grad),lr=6e-5,weight_decay=.0025)
    steps_per_epoch=(len(rows)//world)//64; total=20*steps_per_epoch; warm=2*steps_per_epoch
    def schedule(s):
        if s<warm: return (s+1)/warm
        progress=min(1.,(s-warm)/max(1,total-warm))
        return 1/60+(1-1/60)*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,schedule)
    scaler=torch.amp.GradScaler('cuda'); epoch0=start=step=0
    resume=args.resume_checkpoint or (out/'last.ckpt' if (out/'last.ckpt').exists() else None)
    if resume:
        state=load_checkpoint_cpu(resume)
        if state['config_hash']!=identity: raise ValueError('Codec resume identity mismatch; refusing native/other run resume')
        model.encoder.load_state_dict(state['encoder']); model.projector.load_state_dict(state['projector'])
        optim.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        epoch0,start,step=state['epoch'],state['next_batch_index'],state['optimizer_step']
        print(f'[resume rank{rank}] epoch={epoch0} batch={start} step={step}',flush=True)
    model=DDP(model,device_ids=[local]); model.train()
    if rank==0:
        for d in ['logs','figures']: (out/d).mkdir(exist_ok=True)
        shutil.copy2(manifest,out/manifest.name); shutil.copy2(fre_path,out/'fre.pt')
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
        (out/'run_metadata.json').write_text(json.dumps({'config_hash':identity,'initial_checkpoint':str(initial),'torch':torch.__version__,'pillow':Image.__version__,'manifest':str(manifest),'task':'encoder_only'},indent=2))
    dist.barrier()
    sampler=DistributedSampler(Rows(rows),num_replicas=world,rank=rank,shuffle=True,seed=42,drop_last=True)
    def save(epoch,next_batch):
        if rank==0:
            atomic_torch_save({'schema_version':1,'encoder':model.module.encoder.state_dict(),'projector':model.module.projector.state_dict(),'optimizer':optim.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'config':cfg,'config_hash':identity,'manifest_hash':MANIFEST_HASH,'fre_hash':cfg['fre_hash'],'epoch':epoch,'next_batch_index':next_batch,'optimizer_step':step},out/'last.ckpt')
        dist.barrier()
    for epoch in range(epoch0,20):
        sampler.set_epoch(epoch)
        loader=DataLoader(Rows(rows),batch_sampler=Batches(sampler,64,epoch,start if epoch==epoch0 else 0),num_workers=4,pin_memory=True,collate_fn=Views(42,args.mode))
        sums=Counter(); policies=Counter(); count=0; began=time.monotonic()
        for batch_index,(v1,v2,records) in enumerate(loader,start=start if epoch==epoch0 else 0):
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): z1,z2,e1,e2=model(v1.to(device),v2.to(device))
            loss,metrics=barlow_twins_loss(e1,e2,5e-5,1e-4)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optim)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scale=scaler.get_scale(); scaler.step(optim); scaler.update()
            if scaler.get_scale()<scale:
                if rank==0: print('[AMP] overflow; optimizer step skipped',flush=True)
                continue
            scheduler.step(); step+=1; count+=1
            policies.update(r['policy'] for r in records)
            values={k:float(v) for k,v in metrics.items()}; values['lr']=optim.param_groups[0]['lr']; values['grad_norm']=float(norm)
            values['feature_std']=float(torch.cat([z1,z2]).float().std(0,unbiased=False).mean())
            sums.update(values)
            if rank==0 and (step%100==0 or batch_index==0):
                append(out/'logs/train_steps.csv',{'epoch':epoch+1,'step':step,**values})
                print(f'[train] epoch={epoch+1}/20 batch={batch_index+1}/{steps_per_epoch} step={step} loss={values["loss"]:.3f} diag={values["diag_mean"]:.3f} lr={values["lr"]:.2e}',flush=True)
            if step%250==0: save(epoch,batch_index+1)
        start=0
        if rank==0:
            append(out/'logs/epoch_metrics.csv',{'epoch':epoch+1,'steps_in_segment':count,'seconds':time.monotonic()-began,**{k:v/max(count,1) for k,v in sums.items()},**{f'local_rank0_{k}':policies[k] for k in ['identity','jpeg','resize_jpeg']}})
            plot_logs(out)
            print(f'[epoch] {epoch+1}/20 mean_loss={sums["loss"]/max(count,1):.3f} time={time.monotonic()-began:.0f}s',flush=True)
        save(epoch+1,0)
        if rank==0:
            if (epoch+1)%5==0: shutil.copy2(out/'last.ckpt',out/f'epoch_{epoch+1:02d}.ckpt')
            if epoch==19: atomic_torch_save({'encoder':model.module.encoder.state_dict(),'config':cfg,'manifest_hash':MANIFEST_HASH,'fre_hash':cfg['fre_hash'],'optimizer_step':step},out/'encoder_final.pt')
            if not args.no_publish: publish(out)
        dist.barrier()
    dist.destroy_process_group()

if __name__=='__main__': main()
