"""torchrun trainer. Batch size is explicit and must be measured by GPU preflight."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data import NativePairs, read_rows, sha256
from model import TwinsGSD, barlow_loss
from progress import Progress


MODEL_KEYS = ('layers','k','hidden','output','bypass','activation_checkpointing')


def finish_update(model, optimizer, scaler, clip):
    """Let GradScaler skip overflow updates; never clip contaminated gradients."""
    scaler.unscale_(optimizer)
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients: raise RuntimeError('No gradients after backward')
    # Inspect without modifying gradients. clip_grad_norm_(..., inf) would still
    # contaminate finite tensors when the computed coefficient is NaN.
    grad=torch.stack([g.detach().norm(2) for g in gradients]).double().norm(2)
    if not torch.isfinite(grad):
        bad=[name for name,p in model.named_parameters()
             if p.grad is not None and not torch.isfinite(p.grad).all()]
        if not bad:
            grad=torch.stack([g.detach().double().norm(2) for g in gradients]).norm(2)
        elif not scaler.is_enabled():
            raise FloatingPointError('Nonfinite unscaled gradients: '+', '.join(bad[:12]))
        else:
            previous=scaler.get_scale()
            scaler.step(optimizer)  # unscale_ found_inf causes the optimizer step to be skipped
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return False, float('nan'), {'old_scale':previous,'new_scale':scaler.get_scale(),'bad_parameters':bad[:20]}
    coefficient=(clip/(grad+1e-6)).clamp(max=1).to(gradients[0].dtype)
    torch._foreach_mul_(gradients,coefficient)
    scaler.step(optimizer);scaler.update()
    return True,float(grad),{}


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def weights_hash(model):
    h = hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--clip',required=True,help='Local Hugging Face CLIP ViT-L/14 directory')
    p.add_argument('--manifest',type=Path,required=True,help='Prepared eligible real-only manifest')
    p.add_argument('--pixel-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-per-gpu',type=int,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--loader-timeout',type=int,default=120)
    p.add_argument('--dist-timeout',type=int,default=600)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--heartbeat',type=int,default=30)
    p.add_argument('--precision',choices=['auto','fp16','bf16','fp32'],default='auto')
    p.add_argument('--max-consecutive-overflows',type=int,default=8)
    p.add_argument('--resume',type=Path)
    p.add_argument('--smoke-steps',type=int,default=0)
    args = p.parse_args()
    if args.log_every<1 or args.heartbeat<1 or args.loader_timeout<1 or args.dist_timeout<1 or args.workers<0:
        raise ValueError('Invalid logging/timeout/worker options')
    rank=int(os.environ.get('RANK','0'))
    with Progress(f'train rank{rank}',interval=args.heartbeat) as progress:
        try:
            run(args,progress)
        finally:
            if dist.is_initialized(): dist.destroy_process_group()


def run(args,progress):
    progress.stage('START: checking CUDA and distributed environment')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full ViT-L/14 training; run test_model.py on CPU')
    if args.batch_per_gpu<1 or args.smoke_steps<0:
        raise ValueError('Invalid batch/smoke steps')
    world = int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    free,total_memory=torch.cuda.mem_get_info(device)
    progress.stage(f'GPU={torch.cuda.get_device_name(local)} free={free/1024**3:.2f}GiB '
                   f'total={total_memory/1024**3:.2f}GiB; world={world}')
    if world>1:
        progress.stage(f'DDP/NCCL init: world={world}, timeout={args.dist_timeout}s')
        dist.init_process_group('nccl',timeout=timedelta(seconds=args.dist_timeout))
    config=json.loads(args.config.read_text())
    precision=args.precision
    if precision=='auto': precision='bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 not supported on this GPU; select fp16 or fp32')
    amp_dtype=torch.bfloat16 if precision=='bf16' else torch.float16
    progress.stage(f'precision={precision}; projector/GSD/Barlow=FP32; FP16 initial loss scale=128')
    args.output.mkdir(parents=True,exist_ok=True)
    existing=args.output/'run_identity.json'
    if existing.exists() and not args.resume:
        raise FileExistsError('Output already has run identity; choose new directory or --resume')
    progress.stage(f'Loading local CLIP on CPU: {args.clip}')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    model=TwinsGSD.from_pretrained(args.clip,**{k:config[k] for k in MODEL_KEYS})
    progress.stage('Hashing frozen initialization on CPU (before GPU transfer)')
    frozen_hash=weights_hash(model.es)
    progress.stage(f'Moving Es/Ef/projector to {device}: {torch.cuda.get_device_name(local)}')
    model.to(device)
    progress.stage(f'Reading eligible manifest: {args.manifest}')
    rows=read_rows(args.manifest)
    # A prepared manifest is required; reject labeled fake rows before constructing loader.
    if any(r.get('label','real').lower() not in ('0','0.0','real') for r in rows):
        raise ValueError('Non-real training rows')
    dataset=NativePairs(rows,args.pixel_root,config['seed'])
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=config['seed'],drop_last=True)
    # CUDA has already been initialized: fork workers can deadlock. Use spawn explicitly.
    worker_options={'multiprocessing_context':'spawn','timeout':args.loader_timeout,'prefetch_factor':2} if args.workers else {}
    loader=DataLoader(dataset,batch_size=args.batch_per_gpu,sampler=sampler,num_workers=args.workers,drop_last=True,pin_memory=True,persistent_workers=False,**worker_options)
    if len(loader)==0 or args.batch_per_gpu*world<2:
        raise ValueError('Insufficient images/global batch for Barlow')
    identity={'config':config,'manifest_sha256':sha256(args.manifest),'clip_weights_sha256':frozen_hash,
              'global_batch':args.batch_per_gpu*world,'world_size':world,'steps_per_epoch':len(loader),
              'torch':torch.__version__,'smoke_steps':args.smoke_steps}
    identity['precision_policy']={'backbone':precision,'projector':'fp32','fp16_initial_scale':128.}
    groups=[{'params':list(model.ef.parameters()),'lr':config['lr_encoder'],'weight_decay':config['weight_decay']},
            {'params':list(model.projector.parameters()),'lr':config['lr_projector'],'weight_decay':config['weight_decay']}]
    if not config['bypass']:
        groups.append({'params':list(model.gsd.parameters()),'lr':config['lr_modulation'],'weight_decay':0.})
    else:
        model.gsd.requires_grad_(False)
    assert not any(x.requires_grad for x in model.es.parameters())
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    total=config['epochs']*len(loader); warm=max(1,int(total*config['warmup_fraction']))
    def factor(step):
        if step<warm: return (step+1)/warm
        progress=min(1.,(step-warm)/max(1,total-warm))
        return config['lr_end_fraction']+(1-config['lr_end_fraction'])*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16',init_scale=128.)
    start=0
    if args.resume:
        progress.stage(f'Loading resume checkpoint: {args.resume}')
        payload=torch.load(args.resume,map_location='cpu',weights_only=False)
        if payload['identity']!=identity: raise ValueError('Resume identity mismatch')
        model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])
        start=payload['next_epoch']
        rng=payload['rng'][rank]
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'],device)
    if rank==0:
        existing.write_text(json.dumps(identity,indent=2))
    progress.stage(f'Constructing DDP wrapper; rows={len(rows):,}; batches/epoch={len(loader):,}; '
                   f'global_batch={args.batch_per_gpu*world}; workers={args.workers}; smoke_steps={args.smoke_steps}')
    wrapper=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    updates=0
    consecutive_overflows=0
    for epoch in range(start,config['epochs']):
        dataset.epoch=epoch;sampler.set_epoch(epoch);wrapper.train();torch.cuda.reset_peak_memory_stats(device)
        began=time.perf_counter()
        progress.stage(f'epoch{epoch+1}/{config["epochs"]}: starting DataLoader workers (spawn)')
        iterator=iter(loader)
        for step in range(len(loader)):
            traced = step==0 or step%args.log_every==0
            model.trace = progress.stage if step==0 else None
            progress.stage(f'epoch{epoch+1} step{step+1}/{len(loader)}: waiting for CPU batch',emit=traced)
            data_began=time.perf_counter()
            try:
                x1,x2,ids,rectangles=next(iterator)
            except Exception:
                (args.output/f'failure_rank{rank}.json').write_text(json.dumps({'epoch':epoch,'step':step,'stage':'dataloader'}))
                raise
            data_seconds=time.perf_counter()-data_began
            progress.stage(f'epoch{epoch+1} step{step+1}: forward; data_wait={data_seconds:.2f}s',emit=traced)
            step_began=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast('cuda',dtype=amp_dtype,enabled=precision!='fp32'):
                    z1,z2,e1,e2=wrapper(x1.to(device,non_blocking=True),x2.to(device,non_blocking=True))
                    progress.stage(f'epoch{epoch+1} step{step+1}: Barlow correlation',emit=traced)
                    loss,metrics=barlow_loss(e1,e2,config['barlow_lambda'],config['barlow_eps'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                progress.stage(f'epoch{epoch+1} step{step+1}: backward; loss={loss.item():.6f}',emit=traced)
                scaler.scale(loss).backward()
                if any(v.grad is not None for v in model.es.parameters()): raise RuntimeError('Es gradient contract failed')
                if step==0:
                    for name,param in model.ef.named_parameters():
                        if param.grad is None: raise RuntimeError('Missing Ef gradient: '+name)
                    if not config['bypass'] and any(p.grad is None for p in model.gsd.parameters()):
                        raise RuntimeError('Missing modulation gradients')
                progress.stage(f'epoch{epoch+1} step{step+1}: optimizer update',emit=traced)
                updated,grad,overflow=finish_update(model,optimizer,scaler,config['gradient_clip'])
                if not updated:
                    consecutive_overflows+=1
                    event={'epoch':epoch,'step':step,'event':'amp_overflow_skip','ids':list(ids),**overflow}
                    with (args.output/f'overflow_rank{rank}.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
                    progress.stage(f'AMP overflow: skipped update; scale {overflow["old_scale"]} -> {overflow["new_scale"]}; '
                                   f'consecutive={consecutive_overflows}; bad={overflow["bad_parameters"][:3]}')
                    if consecutive_overflows>=args.max_consecutive_overflows:
                        raise FloatingPointError('Repeated AMP overflow; see overflow_rank*.jsonl; run --precision fp32 for diagnosis')
                    continue
                consecutive_overflows=0
                scheduler.step();updates+=1
            except Exception:
                (args.output/f'failure_rank{rank}.json').write_text(json.dumps({'epoch':epoch,'step':step,'stage':progress.current,'ids':list(ids)}))
                raise
            # Flush a completion marker on every step; detailed GPU diagnostics every log interval.
            progress.stage(f'epoch{epoch+1}: completed step{step+1}/{len(loader)}',emit=False)
            if rank==0:
                # Rank diagnostics are local; GSD and Barlow statistics are global.
                record={'epoch':epoch,'step':step,'loss':loss.item(),'grad_norm':float(grad),
                        'z_variance':z1.detach().float().var(0,unbiased=False).mean().item(),
                        'z_norm':z1.detach().float().norm(dim=1).mean().item(),
                        'peak_vram_bytes':torch.cuda.max_memory_allocated(device),
                        'seconds_per_step':(time.perf_counter()-began)/(step+1),
                        'data_wait_seconds':data_seconds,
                        'compute_seconds':time.perf_counter()-step_began,
                        **{k:float(v) for k,v in metrics.items()},**{k:float(v) for k,v in model.stats.items()}}
                if step%50==0:
                    with torch.no_grad(), torch.autocast('cuda',enabled=False):
                        singular=torch.linalg.svdvals(z1.detach().float()-z1.detach().float().mean(0))
                        prob=singular/singular.sum().clamp_min(1e-12)
                        record['z_effective_rank_local']=float(torch.exp(-(prob*prob.clamp_min(1e-12).log()).sum()))
                with (args.output/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                if step%args.log_every==0:
                    print(f'[loss] epoch={epoch+1}/{config["epochs"]} step={step+1}/{len(loader)} '
                          f'loss={record["loss"]:.5f} diag={record["diagonal_loss"]:.5f} '
                          f'offdiag={record["offdiagonal_loss"]:.5f} data={data_seconds:.2f}s '
                          f'compute={record["compute_seconds"]:.2f}s '
                          f'VRAM={record["peak_vram_bytes"]/1024**3:.2f}GiB',flush=True)
            if args.smoke_steps and updates>=args.smoke_steps:
                if rank==0:
                    progress.stage('Saving smoke checkpoint')
                    atomic_save({'identity':identity,'model':model.state_dict(),'smoke_only':True},args.output/'smoke.ckpt')
                    progress.stage('SMOKE COMPLETE. Main20-epoch training is a separate notebook cell.')
                if world>1: dist.destroy_process_group()
                return
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state(device)}
        all_rng=[None]*world
        if world>1: dist.all_gather_object(all_rng,rng)
        else: all_rng=[rng]
        if rank==0:
            progress.stage(f'epoch{epoch+1}: saving model/optimizer checkpoint')
            payload={'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':all_rng,'next_epoch':epoch+1}
            atomic_save(payload,args.output/f'epoch{epoch+1:02d}.ckpt')
            atomic_save(payload,args.output/'last.ckpt')
        if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()
    progress.stage('TRAINING COMPLETE')


if __name__=='__main__': main()
