import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
import runtime_raw_all7 as r


class NativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(5)
        self.cfg=dict(r.DEFAULT_CONFIG,precision='fp32',k=3,micro_batch=2,anchor_micro_batch=2,
            train_batch=2,test_batch=2,num_workers=0,gradient_checkpointing=True,epochs=3,patience=1)
        c=r.CLIPVisionConfig(image_size=28,patch_size=14,hidden_size=16,intermediate_size=32,
            num_hidden_layers=6,num_attention_heads=2,attention_dropout=0.)
        c._attn_implementation='sdpa'
        self.vision=r.CLIPVisionModel(c).vision_model

    def test_native_pixel_preservation_and_edge_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=np.random.default_rng(1).integers(0,256,(31,43,3),dtype=np.uint8)
            p=Path(tmp)/'a.png'; Image.fromarray(a).save(p)
            f=pd.DataFrame([dict(path=str(p),sample_id='x',label=1)])
            x,_,_=r.Images(f,self.cfg,42,train=True)[0]
            self.assertEqual(tuple(x.shape),(3,42,56))
            un=(x*torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
                +torch.tensor([.48145466,.4578275,.40821073])[:,None,None])
            recovered=(un*255).round().byte().permute(1,2,0).numpy()
            np.testing.assert_array_equal(recovered[:31,:43],a)
            np.testing.assert_array_equal(recovered[-1,-1],a[-1,-1])
            y,_,_=r.Images(f,self.cfg,99,train=True,epoch=7)[0]
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_variable_tokens_both_branches_and_gram_equivalence(self):
        anchor=copy.deepcopy(self.vision).requires_grad_(False).eval()
        model=r.NativeDetector(copy.deepcopy(self.vision),self.cfg,True)
        images=[torch.randn(3,28,42),torch.randn(3,42,56),torch.randn(3,28,28)]
        bases=r.estimate_bases(anchor,images,self.cfg,torch.device('cpu'),model.selected)
        chunks={l:[] for l in model.selected}
        with torch.no_grad():
            for x in images:
                h=anchor.pre_layrnorm(anchor.embeddings(x[None],interpolate_pos_encoding=True))
                for l,layer in enumerate(anchor.encoder.layers):
                    if l in chunks:
                        chunks[l].append(h[:,1:].flatten(0,1))
                    h=layer(h,None,None)[0]
        for l in model.selected:
            v=r.top_basis(torch.cat(chunks[l]),3)
            torch.testing.assert_close(v@v.T,bases[l]@bases[l].T,rtol=2e-4,atol=2e-4)
        loss=sum(model(x[None],bases).square().mean() for x in images)
        loss.backward()
        self.assertIsNotNone(model.vision.embeddings.position_embedding.weight.grad)
        for g in model.gsd.values():
            self.assertTrue(torch.isfinite(g.a.grad)); self.assertTrue(torch.isfinite(g.b.grad))

    def test_all7_group_split_and_loader_complete_coverage(self):
        rows=[]
        for gen in r.GENS7:
            for split,n in [('train',16),('test',6)]:
                for label in (0,1):
                    for i in range(n):
                        rows.append(dict(sample_id=f'{gen}{split}{label}{i}',generator=gen,split=split,label=label,
                            pixel_hash=f'{gen}{split}{label}{i}'))
        # Same real image appears in multiple generators: it must never cross train/validation.
        rows.append(dict(rows[0],sample_id='shared-other-generator',generator='biggan'))
        with tempfile.TemporaryDirectory() as tmp:
            tr,va,te=r.make_splits(pd.DataFrame(rows),dict(self.cfg,validation_fraction=.25),42,Path(tmp))
            self.assertEqual(set(tr.generator),set(r.GENS7)); self.assertEqual(set(va.generator),set(r.GENS7))
            self.assertFalse(set(tr.pixel_hash)&set(va.pixel_hash))
            self.assertEqual(tr.groupby(['generator','label']).size().nunique(),1)
            self.assertEqual(va.groupby(['generator','label']).size().nunique(),1)
            p=Path(tmp)/'im.png'; Image.new('RGB',(29,43)).save(p)
            frame=pd.DataFrame([dict(path=str(p),label=i%2,sample_id=str(i)) for i in range(5)])
            seen=[]
            for images,labels,indices in r.loader(frame,self.cfg,42,train=True):
                seen.extend(indices.tolist())
            self.assertEqual(sorted(seen),list(range(5)))

    def test_early_stop_checkpoint_and_resume_native_mixed_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); rows=[]
            for i,size in enumerate([(28,28),(42,28),(28,42),(42,42)]):
                p=folder/f'{i}.png'; Image.new('RGB',size,(i*30,20,40)).save(p)
                rows.append(dict(path=str(p),label=i%2,sample_id=str(i)))
            f=pd.DataFrame(rows); model=r.NativeDetector(copy.deepcopy(self.vision),self.cfg,True)
            anchor=copy.deepcopy(self.vision).requires_grad_(False).eval()
            with patch.object(r,'predict',return_value=(np.array([.1,.9,.1,.9]),np.zeros(4))):
                r.train_model(model,anchor,f,f,self.cfg,42,torch.device('cpu'),folder,'native-test')
                before=copy.deepcopy(model.state_dict())
                r.train_model(model,anchor,f,f,self.cfg,42,torch.device('cpu'),folder,'native-test')
            hist=json.loads((folder/'history.json').read_text())
            self.assertEqual(len(hist),2); self.assertTrue(hist[-1]['early_stopping']['stopped'])
            self.assertEqual(hist[0]['train_images'],4)
            self.assertEqual(json.loads((folder/'selected_checkpoint.json').read_text())['epoch'],1)
            for k,v in before.items():
                torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)

    def test_reject_augmented_raw_protocol(self):
        r.validate_raw_config(self.cfg)
        with self.assertRaisesRegex(ValueError,'forbids'):
            r.validate_raw_config(dict(self.cfg,jpeg_probability=.5))

    def test_crop560_geometry_and_token_bound(self):
        cfg=dict(self.cfg,input_mode='center_crop_max_edge_pad14',crop_max_side=560)
        r.validate_raw_config(cfg)
        for w,h,pw,ph,tokens in [(1024,1024,560,560,1601),(512,512,518,518,1370),
                                  (224,224,224,224,257),(1024,448,560,448,1281),
                                  (1025,180,560,182,521)]:
            g=r.input_geometry(w,h,cfg)
            self.assertEqual((g['input_width'],g['input_height'],g['tokens']),(pw,ph,tokens))
            self.assertLessEqual(g['tokens'],1601)
            self.assertEqual(g['crop_left'],(w-min(w,560))//2)
            self.assertEqual(g['crop_top'],(h-min(h,560))//2)
        with self.assertRaisesRegex(ValueError,'multiple of 14'):
            r.validate_raw_config(dict(cfg,crop_max_side=556))

    def test_crop560_preserves_center_pixels_and_small_images(self):
        cfg=dict(self.cfg,input_mode='center_crop_max_edge_pad14',crop_max_side=560)
        def decode(x):
            mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
            std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
            return ((x*std+mean)*255).round().byte().permute(1,2,0).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            for i,(h,w) in enumerate([(1024,1024),(224,224),(512,512),(448,1024),(181,1025)]):
                a=np.random.default_rng(i).integers(0,256,(h,w,3),dtype=np.uint8)
                path=Path(tmp)/f'{i}.png'; Image.fromarray(a).save(path)
                frame=pd.DataFrame([dict(path=str(path),label=1,sample_id=str(i))])
                x,_,_=r.Images(frame,cfg,42,train=True)[0]
                g=r.input_geometry(w,h,cfg)
                expected=a[g['crop_top']:g['crop_top']+g['crop_height'],g['crop_left']:g['crop_left']+g['crop_width']]
                restored=decode(x)
                np.testing.assert_array_equal(restored[:g['crop_height'],:g['crop_width']],expected)
                self.assertEqual(restored.shape[:2],(g['input_height'],g['input_width']))
                test_x,_,_=r.Images(frame,cfg,42,case='native')[0]
                torch.testing.assert_close(x,test_x,rtol=0,atol=0)

    def test_jpeg_is_applied_before_center_crop(self):
        cfg=dict(self.cfg,input_mode='center_crop_max_edge_pad14',crop_max_side=28)
        with tempfile.TemporaryDirectory() as tmp:
            a=np.random.default_rng(8).integers(0,256,(51,65,3),dtype=np.uint8)
            path=Path(tmp)/'a.png'; Image.fromarray(a).save(path)
            # JPEG original first, then crop, must match the evaluation pipeline exactly.
            jpg=r.jpeg(Image.fromarray(a),96,cfg['jpeg_subsampling'])
            target=Path(tmp)/'decoded.png'; jpg.save(target)
            original=pd.DataFrame([dict(path=str(path),label=1,sample_id='a')])
            decoded=original.assign(path=str(target))
            got=r.Images(original,cfg,42,case='jpeg96')[0][0]
            expected=r.Images(decoded,cfg,42,case='native')[0][0]
            torch.testing.assert_close(got,expected,rtol=0,atol=0)

    def test_native_prediction_order_and_fake_only_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for i,size in enumerate([(28,42),(42,28),(28,28)]):
                a=np.random.default_rng(i).integers(0,256,(*size,3),dtype=np.uint8)
                p=Path(tmp)/f'{i}.png'; Image.fromarray(a).save(p)
                rows.append(dict(path=str(p),sample_id=str(i),label=i%2))
            frame=pd.DataFrame(rows)
            model=r.NativeDetector(copy.deepcopy(self.vision),self.cfg,True).eval()
            anchor=copy.deepcopy(self.vision).requires_grad_(False).eval()
            probs,bids=r.predict(model,anchor,frame,self.cfg,42,torch.device('cpu'),'native')
            self.assertEqual(bids.tolist(),[0,0,1]); self.assertTrue(np.isfinite(probs).all())
            # Vary compute chunk sizes without changing the logical basis or sample order.
            other,_=r.predict(model,anchor,frame,dict(self.cfg,micro_batch=1,anchor_micro_batch=1),42,torch.device('cpu'),'native')
            np.testing.assert_allclose(probs,other,atol=2e-5)
            raw=r.Images(frame,self.cfg,42,case='native')
            changed=r.Images(frame,self.cfg,42,case='fake_jpeg96')
            torch.testing.assert_close(raw[0][0],changed[0][0],rtol=0,atol=0)
            self.assertFalse(torch.equal(raw[1][0],changed[1][0]))

if __name__=='__main__':
    unittest.main()
