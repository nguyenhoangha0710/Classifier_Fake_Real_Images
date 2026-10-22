"""Fit native K1/full Gaussian and evaluate the locked codec encoder. No training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
NATIVE=HERE/'native' if (HERE/'native').is_dir() else HERE.parent/'forensic_twins_original'
sys.path.insert(0,str(NATIVE))
import fit_gmm_and_evaluate as g
import test_bfree_gmm_kaggle as b
from src.core import ForensicViT, ForensicResidualExtractor
from src.data import read_manifest, discover_imagenet_train, discover_coco_roots
from src.checkpointing import sha256_file

HASHES={'train_real_200k.csv':'56023995de848527cecad2298c2ddfa074bcd06dacd7f314b6f6bf9554066f90','augmented_gmm_fit_base.csv':'47d6adc1f82d4a25778893eb6a7c0565f25d758cbc397ab2351689d887712591','augmented_gmm_calibration_base.csv':'246cfac49ba4c86f29cada43263ffd48a0b4f46c5d0e8da444b89e34ec11d949','bfree_valid_clean_eval.csv':'30d0dd7c2eb3063549141df8512c44e33d7fc7719703c8a1473c316390e9ab29'}

def root(base,owner,slug):
    found=[p for p in [base/slug,base/'datasets'/owner/slug] if p.is_dir()]
    if len(found)!=1: raise FileNotFoundError(f'Mount {owner}/{slug} or specify exact root override; found={found}')
    return found[0]

def verify(path):
    actual=sha256_file(path)
    if actual!=HASHES[path.name]: raise ValueError(f'Manifest differs from locked full200K run: {path}')
    return actual

def rebase(rows,inputs):
    im=discover_imagenet_train(inputs); co=discover_coco_roots(inputs)
    output=[]
    for original in rows:
        row=dict(original); old=Path(row['image_path'])
        if int(row['label'])!=0: raise ValueError('Non-real density training row')
        if row['source']=='imagenet1k': path=im/(row.get('class_id') or old.parent.name)/old.name
        elif row['source']=='coco2017': path=co[row.get('original_split') or 'train2017']/old.name
        else: raise ValueError('Unknown source')
        if not path.is_file(): raise FileNotFoundError(path)
        if min(int(row['width']),int(row['height']))<256: raise ValueError(f'Undersized image: {path}')
        row.update(image_path=str(path),relative_path=f'{row["source"]}/{old.parent.name}/{old.name}')
        output.append(row)
    return output

class ProtocolDataset(b.BFreeProtocolDataset):
    def __getitem__(self,index):
        if self.rows[index].get('jpeg_reencode',False):
            # Tiny fake-codec-matched stores per-image qtables/subsampling in rows.
            value=g.ProtocolImageDataset([self.rows[index]],self.crop_size)[0]['gray_halo']
            return value,index
        return super().__getitem__(index)

def cached_extract(rows,spec,extractor,device,batch,workers,folder):
    directory=folder/'artifacts/embeddings'; directory.mkdir(parents=True,exist_ok=True)
    file=directory/f'{spec.name}_cls192.npy'; progress=directory/f'{spec.name}_progress.json'
    key={'protocol':spec.__dict__,'rows':hashlib.sha256(json.dumps([{k:v for k,v in r.items() if k!='image_path'} for r in rows],sort_keys=True).encode()).hexdigest(),'count':len(rows)}
    start=0
    if progress.exists():
        previous=json.loads(progress.read_text())
        if previous['identity']!=key: raise ValueError('Embedding cache identity mismatch')
        start=int(previous['next_index'])
        matrix=np.lib.format.open_memmap(file,mode='r+')
        if matrix.shape!=(len(rows),192) or not 0<=start<=len(rows): raise ValueError('Invalid cache shape/progress')
    else:
        matrix=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=(len(rows),192))
        b.atomic_write_json(progress,{'identity':key,'next_index':0})
    loader=torch.utils.data.DataLoader(ProtocolDataset(rows[start:],spec,256),batch_size=batch,num_workers=workers,shuffle=False,pin_memory=True)
    cursor=last=start
    with torch.inference_mode():
        for gray,_ in loader:
            with torch.autocast('cuda',dtype=torch.float16): values=extractor(gray.to(device)).float().cpu().numpy()
            if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite embedding')
            matrix[cursor:cursor+len(values)]=values; cursor+=len(values)
            if cursor-last>=1000 or cursor==len(rows):
                matrix.flush(); b.atomic_write_json(progress,{'identity':key,'next_index':cursor})
                print(f'[embedding] {spec.name}: {cursor}/{len(rows)}',flush=True); last=cursor
    del matrix
    return np.load(file,mmap_mode='r')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare','fit','tiny','bfree'],required=True)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--output-root',type=Path,default=Path('/kaggle/working/forensic_twins_codec_gmm'))
    for name in ['codec-root','saved-manifests','bfree-root','tiny-root']: p.add_argument('--'+name,type=Path)
    p.add_argument('--batch-per-gpu',type=int,default=32); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--bootstrap-samples',type=int,default=500)
    args=p.parse_args(); out=args.output_root; out.mkdir(parents=True,exist_ok=True)
    codec=args.codec_root or root(args.input_root,'hoangha071005','checkpoint-encoder-twins-codec-200k')/'forensic_twins_codec'
    saved=args.saved_manifests or root(args.input_root,'hoangha071005','evaluation-twins-bfree')/'forensic_twins_augmented_gmm_full/manifests'
    # Explicit overrides target directories containing these exact files; no wildcard scan or resampling.
    parent=codec/'train_real_200k.csv'; fre=codec/'fre.pt'; checkpoint=codec/'encoder_final.pt'
    fitfile=saved/'augmented_gmm_fit_base.csv'; calfile=saved/'augmented_gmm_calibration_base.csv'; bf=saved/'bfree_valid_clean_eval.csv'
    for path in [parent,fitfile,calfile,bf]: verify(path)
    sidecar=checkpoint.with_suffix('.pt.sha256')
    if not sidecar.exists() or sidecar.read_text().split()[0]!=sha256_file(checkpoint): raise ValueError('Final encoder checksum missing/mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['manifest_hash']!=HASHES[parent.name] or state['fre_hash']!=sha256_file(fre): raise ValueError('Encoder data/FRE mismatch')
    if state['config']['mode']!='native_random' or state['config']['epochs']!=20: raise ValueError('Expected locked codec20 native_random encoder')
    identity={'encoder_sha256':sha256_file(checkpoint),'fre_sha256':sha256_file(fre),'manifest_hashes':HASHES,'crop':256,'reg_covar':1e-6,'percentiles':[90,95,99],'primary':95,'protocols':['native','jpeg_q95','jpeg_q80','jpeg_q65','jpeg_q50','resize_075'],'embedding_dim':192,'preprocess':'native_legacy_center_crop_then_zero_halo7_v1'}
    lock=out/'run_identity.json'
    if lock.exists() and json.loads(lock.read_text())!=identity: raise ValueError('Output resume identity mismatch; use separate output')
    b.atomic_write_json(lock,identity)
    parent_rows=read_manifest(parent); fit_rows=read_manifest(fitfile); cal_rows=read_manifest(calfile)
    ids=lambda rows:{r['sample_id'] for r in rows}
    if (len(parent_rows),len(fit_rows),len(cal_rows))!=(200000,160000,40000): raise ValueError('Wrong split sizes')
    if len(ids(parent_rows))!=200000 or len(ids(fit_rows))!=160000 or len(ids(cal_rows))!=40000 or ids(fit_rows)&ids(cal_rows) or ids(fit_rows)|ids(cal_rows)!=ids(parent_rows): raise ValueError('Bad split membership')
    bfroot=args.bfree_root or root(args.input_root,'baopv051','b-free-training-data')
    bfrows=b.load_reusable_manifest(bf,bfroot)
    if len(bfrows)!=10899 or len({r['bfree_id'] for r in bfrows})!=1557: raise ValueError('Wrong B-Free cohort')
    if {str(r['bfree_id']).zfill(12) for r in bfrows}&b.coco_ids_from_training_manifest(parent): raise ValueError('B-Free source overlap')
    if any(not Path(r['image_path']).is_file() for r in bfrows): raise FileNotFoundError('B-Free manifest pixels missing')
    manifests=out/'manifests'; manifests.mkdir(exist_ok=True)
    if args.stage=='prepare':
        for name,rows in [('gmm_fit_real.csv',rebase(fit_rows,args.input_root)),('gmm_calibration_real.csv',rebase(cal_rows,args.input_root)),('bfree_valid_clean_eval.csv',bfrows)]: b.write_csv(rows,manifests/name)
        b.atomic_write_json(out/'preflight.json',{'fit':160000,'calibration':40000,'bfree_ids':1557,'bfree_images':10899,'checkpoint_optimizer_step':state['optimizer_step'],'calibration_encoder_seen':True})
        print('[prepare] verified saved split160K/40K and clean B-Free1557 IDs; no data resampling',flush=True); return
    if not (out/'preflight.json').exists(): raise RuntimeError('Run prepare first')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for extraction')
    encoder=ForensicViT(in_channels=8,patch_size=8,embed_dim=192,depth=6,num_heads=3,mlp_ratio=4.)
    encoder.load_state_dict(state['encoder'],strict=True)
    extractor=g.FrozenFeatureExtractor(ForensicResidualExtractor.from_pretrained(fre),encoder).eval().cuda()
    del state
    devices=torch.cuda.device_count()
    if devices>1: extractor=torch.nn.DataParallel(extractor)
    device=torch.device('cuda:0'); batch=args.batch_per_gpu*devices
    def extract(rows,spec,folder):
        rows=[dict(r,relative_path=r.get('relative_path') or str(r['image_path'])) for r in rows]
        return cached_extract(rows,spec,extractor,device,batch,args.workers,folder)
    density=out/'artifacts/gmm'; density.mkdir(parents=True,exist_ok=True)
    if args.stage=='fit':
        fit_rows=rebase(read_manifest(manifests/'gmm_fit_real.csv'),args.input_root)
        cal_rows=rebase(read_manifest(manifests/'gmm_calibration_real.csv'),args.input_root)
        z=extract(fit_rows,b.ProtocolSpec('real_fit_native'),out)
        model=g.fit_single_gaussian(np.asarray(z),1e-6)
        if not all(np.isfinite(v).all() for v in model.values()): raise FloatingPointError('Invalid Gaussian')
        zc=extract(cal_rows,b.ProtocolSpec('real_calibration_native'),out)
        scores=g.gaussian_nll(np.asarray(zc),model)
        thresholds=g.calibrate_thresholds(scores,[90,95,99],95)
        thresholds.update({'num_real_fit':160000,'num_real_calibration':40000,'calibration_encoder_seen':True,'identity':identity})
        g.save_gmm(model,density/'real_k1_full.npz',identity)
        b.atomic_write_json(density/'thresholds.json',thresholds)
        np.save(density/'calibration_nll.npy',scores)
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4)); ax.hist(scores,bins=100)
        for q,t in thresholds['percentiles'].items(): ax.axvline(t,label=f'Q{q}={t:.2f}')
        ax.set(xlabel='Real calibration NLL',ylabel='Count'); ax.legend(); fig.tight_layout(); fig.savefig(density/'calibration_thresholds.png',dpi=150); plt.close(fig)
        print('[fit] new codec Gaussian saved. Locked real-only thresholds:',thresholds['percentiles'],flush=True); return
    model=g.load_gmm(density/'real_k1_full.npz'); thresholds=json.loads((density/'thresholds.json').read_text())
    if thresholds['identity']!=identity: raise ValueError('Density provenance mismatch')
    if args.stage=='bfree':
        summaries=[]
        for name in ['native_png','jpeg_q95','jpeg_q80','jpeg_q65','jpeg_q50','resize_075']:
            spec=b.parse_protocol(name); folder=out/'bfree'; values=extract(bfrows,spec,folder)
            summaries.append(b.evaluate_protocol(spec,bfrows,values,model,thresholds,folder,args.bootstrap_samples,[90,95,99],300,42))
        b.save_cross_protocol_summary(summaries,out/'bfree'); return
    tiny=g.discover_tiny_root(args.tiny_root,args.input_root)
    rows=g.build_native_eval_rows(tiny,None,42,256,out/'tiny/eligibility_audit.csv')
    protocols=[('native_raw',rows,b.ProtocolSpec('native_raw'))]
    matched=g.build_codec_matched_rows(rows,out/'tiny',42)
    protocols.append(('fake_codec_matched',matched,b.ProtocolSpec('fake_codec_matched')))
    protocols += [(name,rows,b.parse_protocol(name)) for name in ['jpeg_q95','jpeg_q80','jpeg_q65','jpeg_q50','resize_075']]
    summaries=[]
    for name,items,spec in protocols:
        folder=out/'tiny'; values=extract(items,spec,folder); scores=g.gaussian_nll(np.asarray(values),model)
        labels=np.array([int(r['label']) for r in items]); generators=np.array([r['generator'] for r in items])
        metrics=folder/'artifacts/metrics'; metrics.mkdir(parents=True,exist_ok=True)
        ranking=[]
        for group in ['pooled',*sorted(set(generators))]:
            mask=np.ones(len(items),dtype=bool) if group=='pooled' else generators==group
            ranking.append({'protocol':name,'scope':group,'auroc':g.auroc(labels[mask],scores[mask]),'average_precision':g.average_precision(labels[mask],scores[mask])})
        g.write_dict_rows(ranking,metrics/f'{name}_ranking.csv')
        g.write_dict_rows(g.threshold_metric_rows(labels,scores,generators,name,thresholds),metrics/f'{name}_threshold_metrics.csv')
        g.write_dict_rows([{'sample_id':r['sample_id'],'generator':r['generator'],'label':int(r['label']),'protocol':name,'nll':float(s),**{f'pred_q{q}':int(s>t) for q,t in thresholds['percentiles'].items()}} for r,s in zip(items,scores)],metrics/f'{name}_predictions.csv')
        g.save_visualizations(np.asarray(values),labels,scores,generators,name,model,thresholds,folder,300,42)
        summary={'protocol':name,'images':len(items),'pooled_auroc':ranking[0]['auroc'],'macro_auroc':float(np.nanmean([r['auroc'] for r in ranking[1:]]))}
        summaries.append(summary); print('[tiny]',summary,flush=True)
    g.write_dict_rows(summaries,out/'tiny/protocol_comparison.csv')

if __name__=='__main__': main()
