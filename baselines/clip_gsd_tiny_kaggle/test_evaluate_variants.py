"""CPU regression tests; no Kaggle credentials, pretrained download or real checkpoint needed."""
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

import evaluate_all7_variants as r


class VariantTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.cfg=dict(r.DEFAULT_CONFIG,precision='fp32',k=3,micro_batch=1,anchor_micro_batch=1,
            test_batch=2,num_workers=0,input_mode='center_crop_max_edge_pad14',crop_max_side=560)
        self.options=dict(r.EVAL_DEFAULTS,save_every_batches=1,log_every_batches=1)

    def make_images(self,folder,n=6):
        rows=[]
        for i in range(n):
            relative=f'imagenet_glide/val/{"ai" if i%2 else "nature"}/{i}.png'
            path=folder/relative; path.parent.mkdir(parents=True,exist_ok=True)
            array=np.random.default_rng(i).integers(0,256,(28+(i%2)*14,28+(i%3)*14,3),dtype=np.uint8)
            Image.fromarray(array).save(path)
            rows.append(dict(sample_id=f'id{i}',path=str(path),relative_path=relative,
                generator='glide',label=i%2,width=array.shape[1],height=array.shape[0],
                file_hash=r.file_sha(path),pixel_hash=str(i),file_bytes=path.stat().st_size))
        return pd.DataFrame(rows)

    def tiny_model(self):
        cfg=r.CLIPVisionConfig(image_size=28,patch_size=14,hidden_size=16,intermediate_size=32,
            num_hidden_layers=6,num_attention_heads=2,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        vision=r.CLIPVisionModel(cfg).vision_model
        anchor=copy.deepcopy(vision).requires_grad_(False).eval()
        model=r.NativeDetector(vision,self.cfg,True).requires_grad_(False).eval()
        return model,anchor

    def test_native_and_png_exactly_match_training_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=self.make_images(Path(tmp),2)
            # Include large/asymmetric image to exercise the actual crop560 path.
            p=Path(rows.loc[0,'path']); Image.new('RGB',(1025,180),(80,90,100)).save(p)
            rows.loc[0,['width','height']]=[1025,180]
            for index in range(len(rows)):
                old,_,_=r.Images(rows,self.cfg,42,case='native')[index]
                native,_,_=r.VariantImages(rows,self.cfg,r.case_spec('native'))[index]
                png,_,_=r.VariantImages(rows,self.cfg,r.case_spec('png_roundtrip'))[index]
                torch.testing.assert_close(native,old,rtol=0,atol=0)
                torch.testing.assert_close(native,png,rtol=0,atol=0)

    def test_jpeg_is_label_dependent_only_for_fake_case(self):
        a=np.random.default_rng(4).integers(0,256,(35,47,3),dtype=np.uint8); im=Image.fromarray(a)
        real=r.transform_variant(im,0,r.case_spec('fake_jpeg96'),self.cfg)
        fake=r.transform_variant(im,1,r.case_spec('fake_jpeg96'),self.cfg)
        both=r.transform_variant(im,0,r.case_spec('all_jpeg96'),self.cfg)
        np.testing.assert_array_equal(np.asarray(real),a)
        self.assertFalse(np.array_equal(np.asarray(fake),a))
        np.testing.assert_array_equal(np.asarray(fake),np.asarray(both))
        self.assertEqual(fake.size,im.size)

    def test_resize_shapes_and_transformation_order(self):
        im=Image.new('RGB',(1024,448),(40,80,100))
        for side in (224,256,336,512,560):
            result=r.transform_variant(im,0,r.case_spec(f'resize_square{side}'),self.cfg)
            self.assertEqual(result.size,(side,side))
        long=r.transform_variant(im,0,r.case_spec('resize_long560'),self.cfg)
        self.assertEqual(long.size,(560,245))
        small=r.transform_variant(Image.new('RGB',(100,200)),0,r.case_spec('resize_long560'),self.cfg)
        self.assertEqual(small.size,(100,200))
        x=r.transform_variant(im,1,r.case_spec('resize256_jpeg96'),self.cfg)
        expected=r.jpeg(im.resize((256,256),Image.Resampling.BICUBIC),96,0)
        np.testing.assert_array_equal(np.asarray(x),np.asarray(expected))
        self.assertEqual(r.input_geometry(256,256,self.cfg)['tokens'],362)
        self.assertEqual(r.input_geometry(560,560,self.cfg)['tokens'],1601)

    def test_blur_and_case_validation(self):
        a=np.zeros((31,31,3),dtype=np.uint8); a[15,15]=255
        blurred=np.asarray(r.transform_variant(Image.fromarray(a),0,r.case_spec('blur_sigma1'),self.cfg))
        self.assertGreater(blurred[14,15,0],0); self.assertLess(blurred[15,15,0],255)
        for name in ('jpeg96','all_jpeg0','all_jpeg101','fake_jpeg-1','resize_square1024','unknown'):
            with self.assertRaises(ValueError):
                r.case_spec(name)

    def test_remap_saved_manifest_without_decoding_or_scanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=self.make_images(root/'tiny-genimage',2)
            rows['path']='/old/kaggle/path/does/not/exist'
            opts=dict(self.options,input_root=str(root))
            with patch.object(r.Image,'open',side_effect=AssertionError('must not decode')):
                result,base=r.resolve_test_files(rows,opts)
            self.assertEqual(base,(root/'tiny-genimage').resolve())
            self.assertTrue(all(Path(p).is_file() for p in result.path))
            changed=rows.copy(); changed.loc[0,'file_bytes']+=1
            with self.assertRaisesRegex(ValueError,'changed input'):
                r.resolve_test_files(changed,opts)
            changed=rows.copy(); changed.loc[0,'relative_path']='../outside.png'
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                r.resolve_test_files(changed,opts)

    def test_checkpoint_discovery_selects_crop560_not_old_sd15(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode,crop,name in [('center_crop_max_edge_pad14',560,'new'),('native_rgb_edge_pad14',None,'old')]:
                folder=root/name; (folder/'seed42/clip_gsd').mkdir(parents=True)
                (folder/'seed42/clip_gsd/best.pt').touch(); (folder/'seed42/test_manifest.csv').touch()
                (folder/'config.json').write_text(json.dumps(dict(input_mode=mode,crop_max_side=crop)))
            found=r.find_checkpoint(dict(self.options,input_root=str(root)))
            self.assertEqual(found,root/'new/seed42/clip_gsd/best.pt')
            with self.assertRaises(FileNotFoundError):
                r.find_checkpoint(dict(self.options,checkpoint_path=str(root/'missing.pt')))

    def test_full_evaluation_resume_preserves_order_and_batch_bases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=self.make_images(root/'input'); model,anchor=self.tiny_model()
            out=root/'out'; out.mkdir()
            # Crash AFTER one complete, persisted logical batch. No partial-batch reuse.
            original=r.estimate_bases; count=[0]
            def crash_after_one(*args,**kwargs):
                count[0]+=1
                if count[0]==2:
                    raise RuntimeError('simulated interruption')
                return original(*args,**kwargs)
            with patch.object(r,'estimate_bases',side_effect=crash_after_one):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    r.evaluate_case(model,anchor,rows,self.cfg,self.options,'native',out,'same',torch.device('cpu'))
            self.assertEqual(len(pd.read_csv(next((out/'progress/native').glob('part_*.csv')))),2)
            resumed=r.evaluate_case(model,anchor,rows,self.cfg,self.options,'native',out,'same',torch.device('cpu'))
            reference,_=r.predict(model,anchor,rows,self.cfg,42,torch.device('cpu'),'native')
            np.testing.assert_allclose(resumed.prob_fake,reference,rtol=0,atol=1e-15)
            self.assertEqual(resumed.sample_id.tolist(),rows.sample_id.tolist())
            self.assertEqual(resumed.batch_id.tolist(),[0,0,1,1,2,2])
            with patch.object(r,'estimate_bases',side_effect=AssertionError('complete case must not run')):
                cached=r.evaluate_case(model,anchor,rows,self.cfg,self.options,'native',out,'same',torch.device('cpu'))
            np.testing.assert_allclose(cached.prob_fake,resumed.prob_fake,rtol=0,atol=1e-15)
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                r.evaluate_case(model,anchor,rows,self.cfg,self.options,'native',out,'different',torch.device('cpu'))
            # PNG changes neither pixels nor GSD basis/results.
            png=r.evaluate_case(model,anchor,rows,self.cfg,self.options,'png_roundtrip',out,'same',torch.device('cpu'))
            np.testing.assert_allclose(png.prob_fake,resumed.prob_fake,rtol=0,atol=1e-15)

    def test_tail_batch_and_partial_shard_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=self.make_images(root/'input',5); model,anchor=self.tiny_model()
            out=root/'out'; out.mkdir()
            result=r.evaluate_case(model,anchor,rows,self.cfg,self.options,'native',out,'x',torch.device('cpu'))
            self.assertEqual(result.batch_id.tolist(),[0,0,1,1,2])
            folder=root/'bad'; folder.mkdir()
            result.iloc[:1].to_csv(folder/'part_0000000.csv',index=False)
            with self.assertRaisesRegex(ValueError,'complete logical batch'):
                r.load_case_parts(folder,rows,2)
            result.iloc[:2].assign(sample_id=['bad','bad']).to_csv(folder/'part_0000000.csv',index=False)
            with self.assertRaisesRegex(ValueError,'ordered manifest'):
                r.load_case_parts(folder,rows,2)

    def test_metrics_paired_deltas_and_single_class_smoke(self):
        base=pd.DataFrame(dict(sample_id=['a','b','c','d'],generator=['g']*4,label=[0,0,1,1],
            prob_fake=[.1,.8,.9,.2],batch_id=[0,0,1,1],case=['native']*4,seed=[42]*4))
        changed=base.assign(case='all_jpeg96',prob_fake=[.1,.2,.9,.8])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); table=r.write_eval_reports(pd.concat([base,changed]),self.cfg,out)
            self.assertEqual(table[(table.case=='native')&(table.generator=='pooled')].accuracy.iloc[0],.5)
            after=table[(table.case=='all_jpeg96')&(table.generator=='pooled')].iloc[0]
            self.assertEqual((after.tn,after.fp,after.fn,after.tp),(2,0,0,2))
            summary=pd.read_csv(out/'paired_summary.csv')
            self.assertEqual(summary.wrong_to_correct.sum(),2)
            self.assertEqual(summary.correct_to_wrong.sum(),0)
            delta=pd.read_csv(out/'delta_from_native.csv')
            self.assertEqual(delta[(delta.case=='all_jpeg96')&(delta.generator=='pooled')].delta_accuracy.iloc[0],.5)
            self.assertTrue(np.isnan(r.eval_metrics(base.iloc[:2])['auc']))

    def test_reference_check_and_json_resume_normalization(self):
        base=pd.DataFrame(dict(sample_id=['a','b'],label=[0,1],prob_fake=[.1,.9],batch_id=[0,0],
            case=['native']*2,model=['clip_gsd']*2,seed=[42]*2))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ref=root/'predictions.csv'; base.to_csv(ref,index=False)
            result=r.compare_native_reference(base,ref,self.options,root)
            self.assertEqual(result['status'],'match')
            result=r.compare_native_reference(base.assign(prob_fake=[.8,.9]),ref,self.options,root)
            self.assertEqual(result['status'],'review_required'); self.assertEqual(result['prediction_flips'],1)
            meta=dict(config=dict(betas=(.9,.999)))
            r.bind_metadata(root/'metadata.json',meta)
            r.bind_metadata(root/'metadata.json',meta)  # JSON tuples become lists; repeated binds must work.


if __name__=='__main__':
    unittest.main()
