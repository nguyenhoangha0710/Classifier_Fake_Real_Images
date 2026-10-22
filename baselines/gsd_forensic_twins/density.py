"""Per-sample extraction and fit-only-real PCA/Gaussian artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from data import read_rows, pixel_path, tensor, sha256
from model import TwinsGSD
from train import MODEL_KEYS


class Centers(Dataset):
    def __init__(self, rows, root): self.rows,self.root=rows,root
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(pixel_path(self.rows[i],self.root)) as im:
            im=im.convert('RGB');w,h=im.size
            if min(w,h)<224: raise ValueError('Prepare/filter invalid center crop first')
            return tensor(im.crop(((w-224)//2,(h-224)//2,(w-224)//2+224,(h-224)//2+224)))


def fit_gaussian(features, components=128, reg=1e-6):
    from sklearn.decomposition import PCA
    if len(features)<=components: raise ValueError('Insufficient fit rows for PCA dimension')
    pca=PCA(n_components=components,whiten=False,svd_solver='full')
    x=pca.fit_transform(features.astype(np.float64))
    mean=x.mean(0);delta=x-mean
    cov=delta.T@delta/len(x)+np.eye(components)*reg
    chol=np.linalg.cholesky(cov)
    return {'pca_mean':pca.mean_,'pca_components':pca.components_,
            'mean':mean,'chol':chol,'reg_covar':np.array(reg)}


def score(features, artifact):
    x=(features-artifact['pca_mean'])@artifact['pca_components'].T-artifact['mean']
    solved=np.linalg.solve(artifact['chol'],x.T).T
    return .5*(x.shape[1]*np.log(2*np.pi)+2*np.log(np.diag(artifact['chol'])).sum()+np.square(solved).sum(1))


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='stage',required=True)
    ext=sub.add_parser('extract')
    for name in ('checkpoint','clip','manifest','pixel-root','output'): ext.add_argument('--'+name,required=True,type=Path)
    ext.add_argument('--batch',type=int,default=4);ext.add_argument('--workers',type=int,default=2)
    fit=sub.add_parser('fit')
    for name in ('fit','calibration','output'): fit.add_argument('--'+name,required=True,type=Path)
    fit.add_argument('--components',type=int,default=128)
    ev=sub.add_parser('score')
    for name in ('features','artifact','output'): ev.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.stage=='extract':
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload.get('smoke_only'): raise ValueError('Smoke checkpoint cannot be main detector')
        c=payload['identity']['config']
        model=TwinsGSD.from_pretrained(str(args.clip),**{k:c[k] for k in MODEL_KEYS})
        model.load_state_dict(payload['model']);model.eval().requires_grad_(False)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
        rows=read_rows(args.manifest)
        loader=DataLoader(Centers(rows,args.pixel_root),batch_size=args.batch,num_workers=args.workers)
        file=args.output.with_suffix('.npy')
        if file.exists(): raise FileExistsError(file)
        features=np.lib.format.open_memmap(file,mode='w+',dtype='float32',shape=(len(rows),1024))
        offset=0
        with torch.inference_mode():
            for x in loader:
                # No autocast: evaluation feature definition is explicitly FP32.
                z=model.encode(x.to(device),'sample').cpu().numpy()
                if not np.isfinite(z).all(): raise FloatingPointError('Nonfinite extracted features')
                features[offset:offset+len(z)]=z;offset+=len(z);features.flush()
        meta={'checkpoint_sha256':sha256(args.checkpoint),'inference':'sample_fp32',
              'manifest_sha256':sha256(args.manifest),'rows':rows,'features_sha256':sha256(file)}
        args.output.write_text(json.dumps(meta,indent=2))
        return
    def load(meta_path):
        meta=json.loads(meta_path.read_text());file=meta_path.with_suffix('.npy')
        if sha256(file)!=meta['features_sha256']: raise ValueError('Feature identity mismatch')
        x=np.load(file,mmap_mode='r')
        if len(x)!=len(meta['rows']) or not np.isfinite(x).all(): raise ValueError('Invalid feature matrix')
        return x,meta
    if args.stage=='fit':
        x,fm=load(args.fit);cal,cm=load(args.calibration)
        for key in ('checkpoint_sha256','inference'):
            if fm[key]!=cm[key]: raise ValueError('Fit/calibration representation mismatch')
        def identity(r): return r.get('sample_id') or r.get('image_id') or r['image_path']
        if {identity(r) for r in fm['rows']}&{identity(r) for r in cm['rows']}:
            raise ValueError('Fit/calibration image overlap')
        for meta in (fm,cm):
            if any(r.get('label','real').lower() not in ('real','0','0.0') for r in meta['rows']):
                raise ValueError('Non-real density/calibration row')
        artifact=fit_gaussian(x,args.components)
        artifact['threshold']=np.percentile(score(cal,artifact),95)
        artifact['checkpoint_sha256']=np.array(fm['checkpoint_sha256'])
        artifact['inference']=np.array(fm['inference'])
        artifact['fit_manifest_sha256']=np.array(fm['manifest_sha256'])
        artifact['calibration_manifest_sha256']=np.array(cm['manifest_sha256'])
        with args.output.open('wb') as f: np.savez(f,**artifact)
    else:
        x,meta=load(args.features)
        with np.load(args.artifact,allow_pickle=False) as source: artifact=dict(source)
        for key in ('checkpoint_sha256','inference'):
            if meta[key]!=str(artifact[key]): raise ValueError('Detector/feature representation mismatch')
        nll=score(x,artifact)
        from csv import DictWriter
        with args.output.open('w',newline='',encoding='utf-8') as f:
            writer=DictWriter(f,fieldnames=['sample_id','label','generator','nll','threshold','prediction'])
            writer.writeheader()
            for r,s in zip(meta['rows'],nll):
                writer.writerow({'sample_id':r.get('sample_id',r.get('image_path')),'label':r.get('label',''),
                                 'generator':r.get('generator',''),'nll':float(s),'threshold':float(artifact['threshold']),
                                 'prediction':int(s>artifact['threshold'])})


if __name__=='__main__': main()
