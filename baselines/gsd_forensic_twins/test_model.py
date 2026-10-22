"""CPU contract tests using tiny random CLIP; never download model weights."""
import copy
import random
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import CLIPVisionModel, CLIPVisionConfig

from model import AdaptiveGSD, TwinsGSD, semantic_basis, barlow_loss
from data import boxes, prepare_manifest, NativePairs, read_rows, pixel_path
from PIL import Image
from density import fit_gaussian, score
from torch.utils.data import DataLoader
from train import finish_update

torch.set_num_threads(1)


def tiny(bypass=False):
    config=CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=4,
                           num_attention_heads=2,image_size=8,patch_size=4,attention_dropout=0.)
    return TwinsGSD(CLIPVisionModel(config),layers=(1,2,3,4),k=2,hidden=24,output=12,bypass=bypass)


class Contracts(unittest.TestCase):
    def test_amp_overflow_skips_update_and_reduces_scale(self):
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        before=[p.detach().clone() for p in model.parameters()]
        scaler.scale(model(torch.randn(2,3)).sum()).backward()
        next(model.parameters()).grad[0,0]=float('inf')
        updated,_,info=finish_update(model,optimizer,scaler,1.)
        self.assertFalse(updated);self.assertEqual(info['new_scale'],64.)
        for p,q in zip(model.parameters(),before): torch.testing.assert_close(p,q)
        scaler.scale(model(torch.randn(2,3)).sum()).backward()
        updated,norm,_=finish_update(model,optimizer,scaler,1.)
        self.assertTrue(updated);self.assertTrue(np.isfinite(norm))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(model.parameters(),before)))

    def test_spawn_dataloader_returns_batch(self):
        with tempfile.TemporaryDirectory(prefix='gsd-worker-') as folder:
            root=Path(folder)
            Image.new('RGB',(448,224),color=(80,90,100)).save(root/'a.png')
            ds=NativePairs([{'sample_id':'a','image_path':str(root/'a.png')}],root)
            loader=DataLoader(ds,batch_size=1,num_workers=1,multiprocessing_context='spawn',timeout=30)
            iterator=iter(loader)
            x,y,ids,rectangles=next(iterator)
            self.assertEqual(x.shape,(1,3,224,224))
            self.assertEqual(ids[0],'a')
            with self.assertRaises(StopIteration): next(iterator)

    def test_prepare_reuse_checks_identity(self):
        import prepare
        with tempfile.TemporaryDirectory(prefix='gsd-reuse-') as folder:
            root=Path(folder);(root/'a.jpg').touch()
            parent=root/'parent.csv'
            parent.write_text('sample_id,image_path,width,height,label\na,a.jpg,448,224,0\n')
            output=root/'eligible.csv'
            argv=['prepare.py','--parent',str(parent),'--pixel-root',str(root),
                  '--output',str(output),'--purpose','ssl','--reuse']
            with patch('sys.argv',argv): prepare.main();prepare.main()
            parent.write_text(parent.read_text()+'b,a.jpg,448,224,0\n')
            with patch('sys.argv',argv),self.assertRaises(ValueError): prepare.main()

    def test_metadata_prepare_avoids_pixel_reads(self):
        with tempfile.TemporaryDirectory(prefix='gsd-metadata-') as folder:
            root=Path(folder)
            # Existing file paths only: metadata-mode must not try to decode them.
            (root/'large.jpg').touch();(root/'small.jpg').touch()
            parent=root/'parent.csv'
            with parent.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','image_path','label','width','height'])
                w.writeheader();w.writerows([
                    {'sample_id':'a','image_path':'large.jpg','label':'0','width':448,'height':224},
                    {'sample_id':'b','image_path':'small.jpg','label':'0','width':224,'height':224}])
            with patch('data.Image.open',side_effect=AssertionError('Unexpected pixel audit')):
                result=prepare_manifest(parent,root,root/'eligible.csv',True,audit='metadata')
            self.assertEqual(result['eligible_count'],1)
            self.assertEqual(result['header_reads'],0)
            with self.assertRaises(Exception):
                prepare_manifest(parent,root,root/'audit.csv',True,audit='full')

    def test_known_kaggle_mount_rebase_no_scan(self):
        with patch('data.mounted_root',return_value=Path('/mounted/coco')) as mount:
            row={'source':'coco2017','image_path':'/kaggle/input/old/coco2017/train2017/000000001.jpg'}
            self.assertEqual(pixel_path(row,'/kaggle/input'),Path('/mounted/coco/000000001.jpg'))
            mount.assert_called_once_with('/kaggle/input','coco','train2017')
        with patch('data.mounted_root',return_value=Path('/mounted/train')):
            row={'source':'imagenet1k','image_path':'/kaggle/input/old/ILSVRC/Data/CLS-LOC/train/n1/n1_1.JPEG'}
            self.assertEqual(pixel_path(row,'/kaggle/input'),Path('/mounted/train/n1/n1_1.JPEG'))

    def test_preparation_and_reproducible_native_pairs(self):
        with tempfile.TemporaryDirectory(prefix='gsd-data-') as folder:
            root=Path(folder)
            Image.fromarray(np.random.default_rng(1).integers(0,256,(224,448,3),dtype=np.uint8)).save(root/'large.png')
            Image.new('RGB',(224,224)).save(root/'small.png')
            parent=root/'parent.csv'
            with parent.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','relative_path','label'])
                w.writeheader();w.writerows([{'sample_id':'a','relative_path':'large.png','label':'real'},
                                             {'sample_id':'b','relative_path':'small.png','label':'real'}])
            result=prepare_manifest(parent,root,root/'eligible.csv',True)
            self.assertEqual(result['eligible_count'],1)
            ds=NativePairs(read_rows(root/'eligible.csv'),root)
            first=ds[0];second=ds[0]
            self.assertEqual(first[0].shape,(3,224,224))
            torch.testing.assert_close(first[0],second[0]);torch.testing.assert_close(first[3],second[3])
            self.assertEqual(prepare_manifest(parent,root,root/'center.csv',False)['eligible_count'],2)

    def test_geometry(self):
        for w,h in [(448,224),(224,448),(640,480)]:
            for seed in range(20):
                a,b=boxes(w,h,224,random.Random(seed))
                self.assertTrue(a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1])
                for x,y,right,bottom in (a,b):
                    self.assertTrue(0<=x<right<=w and 0<=y<bottom<=h)
        with self.assertRaises(ValueError): boxes(300,300,224,random.Random(0))

    def test_basis_and_extremes(self):
        p=torch.randn(3,5,8);v,_=semantic_basis(p,2,'sample')
        torch.testing.assert_close(v.transpose(-1,-2)@v,torch.eye(2).expand(3,2,2))
        f=torch.randn(3,6,8)
        m=AdaptiveGSD(lambda_max=0)
        clean,_=m(f,v);torch.testing.assert_close(clean,f)
        residual=f-(f@v)@v.transpose(-1,-2)
        torch.testing.assert_close(residual@v,torch.zeros(3,6,2),atol=2e-6,rtol=0)
        _,stats=AdaptiveGSD()(f,v)
        self.assertGreater(stats['lambda_min'].item(),.5)

    def test_full_forward_gradients_and_frozen(self):
        torch.manual_seed(12);m=tiny().train()
        self.assertFalse(m.es.training)
        seen=[]
        hooks=[m.gsd[str(i)].register_forward_hook(lambda module,args,out,i=i: seen.append((i,args[0].shape))) for i in (1,2,3,4)]
        x,y=torch.randn(4,3,8,8),torch.randn(4,3,8,8)
        z1,z2,e1,e2=m(x,y);loss,_=barlow_loss(e1,e2);loss.backward()
        self.assertEqual(z1.shape,(4,16));self.assertEqual(len(seen),8)
        self.assertTrue(all(shape==(4,5,16) for _,shape in seen))
        self.assertTrue(all(p.grad is None for p in m.es.parameters()))
        for group in (m.ef,m.projector,m.gsd):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters()))
        self.assertTrue(any(p.grad.abs().sum()>0 for p in m.gsd.parameters()))
        for h in hooks: h.remove()

    def test_sample_mode_independence_and_propagation(self):
        torch.manual_seed(8);m=tiny().eval();x=torch.randn(3,3,8,8)
        with torch.no_grad():
            expected=m.encode(x,'sample');single=torch.cat([m.encode(t[None],'sample') for t in x])
            torch.testing.assert_close(expected,single,atol=3e-6,rtol=3e-5)
            torch.testing.assert_close(m.encode(x.flip(0),'sample').flip(0),expected)
            # First GSD must affect the input of block2, not just the final readout.
            seen=[]
            h=m.ef.vision_model.encoder.layers[1].register_forward_pre_hook(lambda module,args: seen.append(args[0].clone()))
            m.encode(x,'sample');m.gsd['1'].lambda_max=0;m.encode(x,'sample');h.remove()
            self.assertGreater((seen[0]-seen[1]).abs().max().item(),1e-4)

    def test_checkpointing(self):
        a=tiny().train();b=copy.deepcopy(a);b.activation_checkpointing=True
        x=torch.randn(3,3,8,8);y=torch.randn_like(x)
        for m in (a,b):
            _,_,e1,e2=m(x,y);loss,_=barlow_loss(e1,e2);loss.backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad)

    def test_gaussian(self):
        from sklearn.mixture import GaussianMixture
        rng=np.random.default_rng(11);x=rng.normal(size=(80,8));y=rng.normal(size=(12,8))
        artifact=fit_gaussian(x,4)
        px=(x-artifact['pca_mean'])@artifact['pca_components'].T
        py=(y-artifact['pca_mean'])@artifact['pca_components'].T
        g=GaussianMixture(n_components=1,covariance_type='full',reg_covar=1e-6).fit(px)
        np.testing.assert_allclose(score(y,artifact),-g.score_samples(py),rtol=1e-8,atol=1e-8)


if __name__=='__main__': unittest.main()
