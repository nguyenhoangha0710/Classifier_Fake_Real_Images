"""CUDA single-GPU/torchrun trainer; no download and no automatic full training."""
import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from baselines.gsd_forensic_twins.progress import Progress
from .data import CollateViews, RealViews, read_manifest
from .losses import distributed, ssl_loss
from .model import GSDEncoder, StudentNetwork, load_clip, make_teacher, update_teacher
from .reference import load_basis
from .utils import atomic_save, load_config, restore_rng, rng_state, seed_all, sha256, state_hash, write_json


def cosine(start, end, progress):
    progress = min(1.0, max(0.0, progress))
    return end + (start-end) * (1 + math.cos(math.pi * progress)) * 0.5


def schedules(config, step, total):
    warm = int(total * config['warmup_fraction'])
    if warm and step < warm:
        lr_factor = (step + 1) / warm
    else:
        lr_factor = cosine(1, config['lr_end_fraction'], (step-warm)/max(1,total-warm-1))
    fraction = step / max(1, total-1)
    temperature_warm = int(total * config['teacher_temperature_warmup_fraction'])
    alpha = min(1.0, step / max(1, temperature_warm)) if temperature_warm else 1.0
    return {'lr_factor': lr_factor,
            'weight_decay': cosine(config['weight_decay'], config['weight_decay_end'], fraction),
            'momentum': cosine(config['teacher_momentum'], config['teacher_momentum_end'], fraction),
            'temperature': config['teacher_temperature_start'] + alpha *
                           (config['teacher_temperature']-config['teacher_temperature_start'])}


def optimizer_groups(student, config):
    groups = {}
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(('dino_head.', 'ibot_head.')):
            role = 'heads'
        elif '.gsd.' in name:
            role = 'gsd'
        elif name.endswith('mask_token'):
            role = 'mask'
        else:
            role = 'encoder'
        decay = param.ndim > 1 and not name.endswith('.bias') and role != 'gsd'
        key = (role, decay)
        if key not in groups:
            lr = config['lr_' + role]
            groups[key] = {'params': [], 'lr': lr, 'base_lr': lr, 'decay': decay,
                           'weight_decay': config['weight_decay'] if decay else 0.0}
        groups[key]['params'].append(param)
    return list(groups.values())


def finish_update(student, optimizer, scaler, clip):
    """Overflow skips student, EMA and schedules together on all ranks."""
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    if not gradients:
        raise RuntimeError('No student gradients')
    finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all().to(torch.int32)
    if distributed():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite gradients in BF16/FP32 training')
        scaler.update(new_scale=scaler.get_scale() * 0.5)
        optimizer.zero_grad(set_to_none=True)
        return False, float('nan')
    norm = torch.stack([g.detach().double().norm() for g in gradients]).norm()
    coefficient = min(1.0, clip / (norm.item()+1e-6))
    for grad in gradients:
        grad.mul_(coefficient)
    scaler.step(optimizer)
    scaler.update()
    return True, norm.item()


def save_checkpoint(path, student, teacher, optimizer, scaler, config, identity,
                    epoch, updates, attempts, smoke, completed):
    states = [None] * (dist.get_world_size() if distributed() else 1)
    state = rng_state()
    if distributed():
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if not distributed() or dist.get_rank() == 0:
        atomic_save({'format': 'gsd-dino-training-v1', 'student': student.state_dict(),
                     'teacher': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'config': config, 'identity': identity,
                     'vision_config': student.encoder.vision.config.to_dict(),
                     'epoch': epoch, 'epoch_completed': completed, 'updates': updates,
                     'attempts': attempts, 'smoke_only': smoke, 'rng_per_rank': states}, path)
    if distributed():
        dist.barrier()


def run(args, progress):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full CLIP ViT-L/14; use CPU unit tests for contracts')
    if args.batch_per_gpu < 2:
        raise ValueError('KoLeo needs batch-per-gpu >= 2; gradient accumulation cannot replace this')
    if args.workers < 0 or args.smoke_steps < 0 or args.log_every < 1 or args.loader_timeout < 1:
        raise ValueError('Invalid loader/log/smoke options')
    world, rank, local = (int(os.environ.get(key, default))
                          for key,default in [('WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=args.dist_timeout))
    config = load_config(args.config)
    seed_all(config['seed'])
    precision = args.precision
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is unsupported on this device')
    amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Output is not empty; choose a fresh directory or --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.resolve().parent != args.output.resolve():
        raise ValueError('Resume checkpoint must belong to the selected output directory')
    progress.stage('Reading fixed basis and real-only training manifest')
    basis = load_basis(args.basis)
    manifest_hash = sha256(args.manifest)
    if basis['training_manifest_sha256'] != manifest_hash:
        raise ValueError('Basis was not fitted from this training manifest')
    if basis['layers'] != config['layers'] or basis['k'] != config['k']:
        raise ValueError('Basis/config mismatch')
    rows = read_manifest(args.manifest)
    if not set(basis['reference_ids']).issubset({r['sample_id'] for r in rows}):
        raise ValueError('Reference IDs are not a subset of training')
    progress.stage('Loading local pretrained CLIP on CPU and checking anchor hash')
    vision = load_clip(args.clip)
    initial_hash = state_hash(vision)
    if initial_hash != basis['clip_state_sha256']:
        raise ValueError('Pretrained CLIP differs from the frozen reference anchor')
    encoder = GSDEncoder(vision, basis['bases'], config['lambda_max'], config['activation_checkpointing'])
    student = StudentNetwork(encoder, config)
    teacher = make_teacher(student)
    progress.stage('Moving student + EMA teacher to GPU; no online semantic anchor')
    student.to(device); teacher.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(student, config), betas=(0.9,0.999), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16', init_scale=128.0)
    dataset = RealViews(rows, args.pixel_root, config)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                  seed=config['seed'], drop_last=True)
    worker_options = {'multiprocessing_context': 'spawn', 'timeout': args.loader_timeout,
                      'prefetch_factor': 2} if args.workers else {}
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_per_gpu, sampler=sampler, drop_last=True,
                        collate_fn=CollateViews(config), num_workers=args.workers, pin_memory=True,
                        generator=generator, **worker_options)
    if not len(loader):
        raise ValueError('No complete training batches')
    identity = {'config': config, 'training_manifest_sha256': manifest_hash,
                'basis_sha256': sha256(args.basis), 'clip_state_sha256': initial_hash,
                'pixel_root': str(args.pixel_root.resolve()), 'world_size': world,
                'batch_per_gpu': args.batch_per_gpu, 'steps_per_epoch': len(loader),
                'precision': precision, 'torch': torch.__version__, 'smoke_steps': args.smoke_steps}
    start_epoch, updates, attempts = 0, 0, 0
    if args.resume:
        checkpoint_state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint_state.get('format') != 'gsd-dino-training-v1' or checkpoint_state['identity'] != identity:
            raise ValueError('Checkpoint identity does not match this run')
        if checkpoint_state['smoke_only'] or not checkpoint_state['epoch_completed']:
            raise ValueError('Only completed-epoch MAIN checkpoints can resume; smoke is separate')
        student.load_state_dict(checkpoint_state['student'])
        teacher.load_state_dict(checkpoint_state['teacher'])
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        scaler.load_state_dict(checkpoint_state['scaler'])
        start_epoch = checkpoint_state['epoch'] + 1
        updates, attempts = checkpoint_state['updates'], checkpoint_state['attempts']
        restore_rng(checkpoint_state['rng_per_rank'][rank])
        del checkpoint_state
    if start_epoch >= config['epochs']:
        raise ValueError('Run already finished all configured epochs')
    if rank == 0:
        write_json(identity, args.output / 'run_identity.json')
    model = DDP(student, device_ids=[local], broadcast_buffers=False) if world > 1 else student
    total_steps = config['epochs'] * len(loader)
    if args.smoke_steps > total_steps:
        raise ValueError('smoke-steps exceeds the configured number of attempted training batches')
    consecutive_overflows = 0
    progress.stage(f'Training ready: real={len(rows)} steps/epoch={len(loader)} precision={precision}')
    for epoch in range(start_epoch, config['epochs']):
        dataset.epoch = epoch; sampler.set_epoch(epoch)
        generator.manual_seed(config['seed'] + epoch * world + rank)
        student.train(); teacher.eval()
        for iteration, batch in enumerate(loader):
            progress.stage(f'epoch={epoch+1} batch={iteration+1}/{len(loader)} teacher/student forward', emit=False)
            global_images = batch['global'].to(device, non_blocking=True)
            local_images = batch['local'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)
            schedule = schedules(config, updates, total_steps)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * schedule['lr_factor']
                group['weight_decay'] = schedule['weight_decay'] if group['decay'] else 0.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=precision != 'fp32'):
                targets = teacher.teacher_forward(global_images, masks)
                predictions = model(global_images, local_images, masks)
                loss, parts = ssl_loss(predictions, targets, masks, config, schedule['temperature'])
            finite_loss = torch.isfinite(loss).to(torch.int32)
            if distributed():
                dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
            if not finite_loss.item():
                raise FloatingPointError('Nonfinite loss on at least one rank')
            progress.stage(f'epoch={epoch+1} batch={iteration+1} backward / optimizer', emit=False)
            scaler.scale(loss).backward()
            if epoch < config['freeze_last_layer_epochs']:
                for head in (student.dino_head, student.ibot_head):
                    for param in head.last_layer.parameters():
                        param.grad = None
            updated, grad_norm = finish_update(student, optimizer, scaler, config['gradient_clip'])
            attempts += 1
            if updated:
                update_teacher(student, teacher, schedule['momentum'])
                updates += 1; consecutive_overflows = 0
            else:
                consecutive_overflows += 1
                if consecutive_overflows >= 8:
                    raise FloatingPointError('Eight consecutive FP16 overflows; inspect precision/LR')
            values = torch.tensor([loss.item()] + [parts[key].item() for key in ('dino','ibot','koleo')], device=device)
            if distributed():
                dist.all_reduce(values); values /= world
            if rank == 0:
                record = {'epoch':epoch+1,'batch':iteration+1,'updates':updates,'attempts':attempts,
                          'updated':updated,'loss':values[0].item(),'dino':values[1].item(),
                          'ibot':values[2].item(),'koleo':values[3].item(),
                          'gradient_norm':grad_norm if math.isfinite(grad_norm) else None,
                          'mask_fraction':masks.float().mean().item(),'scaler':scaler.get_scale(),
                          'memory_allocated_gib':torch.cuda.memory_allocated()/1024**3, **schedule,
                          'gsd_parameters': {layer: {'a': module.a.item(), 'b': module.b.item()}
                                             for layer,module in student.encoder.gsd.items()}}
                with (args.output/'train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record) + '\n')
                if iteration % args.log_every == 0:
                    progress.stage(f"epoch={epoch+1} batch={iteration+1} loss={record['loss']:.4f} "
                                   f"D={record['dino']:.4f} P={record['ibot']:.4f} K={record['koleo']:.4f} update={updated}")
            del targets, predictions, loss, parts
            if args.smoke_steps and attempts >= args.smoke_steps:
                save_checkpoint(args.output/'smoke.ckpt', student, teacher, optimizer, scaler,
                                config, identity, epoch, updates, attempts, True, False)
                progress.stage('Smoke complete; checkpoint is smoke-only, not a trained encoder')
                return
        progress.stage(f'epoch {epoch+1} complete; saving checkpoint')
        # A smoke run can span an epoch: intermediate snapshots remain smoke-only.
        checkpoint_name = 'smoke.ckpt' if args.smoke_steps else 'last.ckpt'
        save_checkpoint(args.output/checkpoint_name, student, teacher, optimizer, scaler,
                        config, identity, epoch, updates, attempts, bool(args.smoke_steps), True)
    progress.stage(f'Finished: successful updates={updates}; attempted={attempts}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--clip', required=True, help='Local pretrained CLIP directory')
    p.add_argument('--basis', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--pixel-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-per-gpu', type=int, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--loader-timeout', type=int, default=120)
    p.add_argument('--dist-timeout', type=int, default=600)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--precision', choices=['auto','fp16','bf16','fp32'], default='auto')
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke-steps', type=int, default=0)
    args = p.parse_args()
    with Progress('gsd-dino rank'+os.environ.get('RANK','0'), interval=30) as progress:
        try:
            run(args, progress)
        finally:
            if distributed():
                dist.destroy_process_group()


if __name__ == '__main__':
    main()
