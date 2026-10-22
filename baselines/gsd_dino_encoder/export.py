"""Export an EMA backbone without SSL heads, or extract frozen image embeddings."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageRows, read_manifest
from .model import GSDEncoder
from .utils import atomic_save, sha256, write_json


def export_checkpoint(checkpoint, output):
    checkpoint = Path(checkpoint); output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('format') != 'gsd-dino-training-v1':
        raise ValueError('Not a GSD-DINO training checkpoint')
    if state['smoke_only'] or not state['epoch_completed'] or state['updates'] < 1:
        raise ValueError('Cannot export a smoke/incomplete/untrained checkpoint as the main encoder')
    encoder_state = {key[len('encoder.'):]:value for key,value in state['teacher'].items()
                     if key.startswith('encoder.')}
    artifact = {'format':'gsd-dino-encoder-v1', 'encoder':encoder_state,
                'vision_config':state['vision_config'],'config':state['config'],
                'identity':state['identity'],'training_epoch':state['epoch']+1,
                'successful_updates':state['updates'],'checkpoint_sha256':sha256(checkpoint),
                'preprocessing': {'size':state['config']['global_size'],
                                  'policy':'CLIP_resize_short_side_center_crop_EXIF_RGB'}}
    atomic_save(artifact, output)
    return artifact


def load_encoder(path, device='cpu'):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    artifact = torch.load(path, map_location='cpu', weights_only=False)
    if artifact.get('format') != 'gsd-dino-encoder-v1':
        raise ValueError('Unsupported encoder artifact')
    bases = {key.split('.')[1]:value for key,value in artifact['encoder'].items()
             if key.startswith('gsd.') and key.endswith('.basis')}
    vision = CLIPVisionModel(CLIPVisionConfig.from_dict(artifact['vision_config']))
    encoder = GSDEncoder(vision, bases, artifact['config']['lambda_max'])
    encoder.load_state_dict(artifact['encoder'], strict=True)
    return encoder.to(device).requires_grad_(False).eval(), artifact


def extract(args):
    if args.output.suffix != '.npy':
        raise ValueError('Embedding output must end in .npy')
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid loader settings')
    meta_path = args.output.with_suffix('.json')
    if args.output.exists() or meta_path.exists():
        raise FileExistsError('Choose fresh extraction outputs')
    encoder, artifact = load_encoder(args.encoder, args.device)
    rows = read_manifest(args.manifest, real_only=False)
    worker_options = {'multiprocessing_context':'spawn'} if args.workers else {}
    loader = DataLoader(ImageRows(rows,args.pixel_root,artifact['preprocessing']['size']),
                        batch_size=args.batch_size,num_workers=args.workers,shuffle=False,**worker_options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.tmp')
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(rows),encoder.dim))
    cursor = 0
    with torch.inference_mode():
        for images, ids in loader:
            features = encoder(images.to(args.device))['cls'].float().cpu().numpy()
            if not np.isfinite(features).all():
                raise FloatingPointError('Nonfinite encoder output')
            values[cursor:cursor+len(features)] = features
            cursor += len(features)
            print(f'[extract] {cursor}/{len(rows)}', flush=True)
    values.flush(); del values
    os.replace(temporary,args.output)
    write_json({'encoder_sha256':sha256(args.encoder),'manifest_sha256':sha256(args.manifest),
                'features_sha256':sha256(args.output),'count':len(rows),'dimension':encoder.dim,
                'readout':'final_layernorm_CLS','preprocessing':artifact['preprocessing'],
                'sample_ids':[row['sample_id'] for row in rows]},meta_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    save = commands.add_parser('export')
    save.add_argument('--checkpoint', type=Path, required=True)
    save.add_argument('--output', type=Path, required=True)
    read = commands.add_parser('extract')
    read.add_argument('--encoder',type=Path,required=True)
    read.add_argument('--manifest',type=Path,required=True)
    read.add_argument('--pixel-root',type=Path,required=True)
    read.add_argument('--output',type=Path,required=True)
    read.add_argument('--device',default='cuda')
    read.add_argument('--batch-size',type=int,default=16)
    read.add_argument('--workers',type=int,default=0)
    args = p.parse_args()
    if args.command == 'export':
        export_checkpoint(args.checkpoint,args.output)
        print('Exported EMA encoder without SSL heads:',args.output)
    else:
        extract(args)


if __name__ == '__main__':
    main()
