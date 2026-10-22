"""CPU-only numerical/information-flow tests; never download pretrained weights."""
import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel

from baselines.gsd_dino_encoder.data import CollateViews, RealViews, prepare, read_manifest
from baselines.gsd_dino_encoder.export import export_checkpoint, extract, load_encoder
from baselines.gsd_dino_encoder.losses import dino_loss, ibot_loss, koleo_loss, sinkhorn_targets, ssl_loss
from baselines.gsd_dino_encoder.model import GSDEncoder, StudentNetwork, clip_tokens, make_teacher, update_teacher
from baselines.gsd_dino_encoder.reference import GramAccumulator, reference_rows
from baselines.gsd_dino_encoder.train import finish_update, optimizer_groups, save_checkpoint, schedules
from baselines.gsd_dino_encoder.utils import atomic_save, load_config

torch.set_num_threads(1)


def tiny():
    torch.manual_seed(123)
    vision_config = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                     num_attention_heads=2, image_size=8, patch_size=2, attention_dropout=0.)
    bases = {str(layer):torch.linalg.qr(torch.randn(16,8)).Q for layer in (3,4)}
    config = dict(dino_prototypes=7, ibot_prototypes=9, head_hidden=24, dino_bottleneck=8,
                  ibot_bottleneck=8, local_crops=2, sinkhorn_iterations=3,
                  student_temperature=.1, koleo_weight=.1, lambda_max=1., global_size=8)
    student = StudentNetwork(GSDEncoder(CLIPVisionModel(vision_config),bases),config)
    return student, make_teacher(student), config


def views():
    global_images = torch.randn(4,3,8,8)
    local_images = torch.randn(4,3,4,4)
    masks = torch.zeros(4,16,dtype=torch.bool)
    masks[0,:4] = True; masks[2,4:7] = True
    return global_images,local_images,masks


class ModelContracts(unittest.TestCase):
    def test_mask_removes_pixel_information_before_attention(self):
        student,_,_ = tiny()
        student.eval()
        image = torch.randn(2,3,8,8)
        masks = torch.zeros(2,16,dtype=torch.bool); masks[:,0] = True
        changed = image.clone(); changed[:,:,:2,:2] += 1000
        original = student.encoder(image,masks)
        masked = student.encoder(changed,masks)
        torch.testing.assert_close(original['cls'],masked['cls'],rtol=0,atol=0)
        torch.testing.assert_close(original['patches'],masked['patches'],rtol=0,atol=0)
        self.assertFalse(torch.allclose(student.encoder(image)['patches'],student.encoder(changed)['patches']))

    def test_token_sizes_and_reference_k_exceeds_local_patch_count(self):
        student,_,_ = tiny()
        for shape,n in [((2,3,8,8),16),((2,3,4,4),4),((2,3,8,4),8)]:
            out = student.encoder(torch.randn(*shape))
            self.assertEqual(out['cls'].shape,(2,16))
            self.assertEqual(out['patches'].shape,(2,n,16))
        self.assertEqual(student.encoder.gsd['3'].basis.shape,(16,8))

    def test_original_clip_embedding_path_matches_at_native_resolution(self):
        student,_,_ = tiny()
        vision = student.encoder.vision
        images = torch.randn(2,3,8,8)
        expected = vision.vision_model.pre_layrnorm(vision.vision_model.embeddings(images))
        torch.testing.assert_close(clip_tokens(vision,images),expected)

    def test_teacher_copy_backward_and_ema_do_not_mutate_basis(self):
        student,teacher,config = tiny()
        for p,q in zip(student.parameters(),teacher.parameters()):
            torch.testing.assert_close(p,q)
        g,l,m = views()
        fixed = {key:value.clone() for key,value in student.named_buffers() if key.endswith('.basis')}
        before_teacher = {key:value.clone() for key,value in teacher.named_parameters()}
        optimizer = torch.optim.AdamW(student.parameters(),lr=1e-3)
        loss,parts = ssl_loss(student(g,l,m),teacher.teacher_forward(g,m),m,config,.07)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(student.encoder.mask_token.grad.norm().item(),0)
        self.assertTrue(any(p.grad is not None for p in student.ibot_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.dino_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.encoder.gsd.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        optimizer.step()
        update_teacher(student,teacher,.9)
        for key,param in teacher.named_parameters():
            expected = before_teacher[key]*.9 + dict(student.named_parameters())[key].detach()*.1
            torch.testing.assert_close(param,expected)
        for key,value in fixed.items():
            torch.testing.assert_close(dict(student.named_buffers())[key],value,rtol=0,atol=0)
            torch.testing.assert_close(dict(teacher.named_buffers())[key],value,rtol=0,atol=0)

    def test_teacher_is_unmasked_even_when_selecting_mask_positions(self):
        _,teacher,_ = tiny()
        g,_,m = views()
        result = teacher.teacher_forward(g,m)
        full = teacher.encoder(g)
        torch.testing.assert_close(result['ibot'],teacher.ibot_head(full['patches'][m]))

    def test_activation_checkpointing_preserves_gradients(self):
        a,_,config = tiny(); b = copy.deepcopy(a)
        a.train(); b.train(); b.encoder.activation_checkpointing = True
        g,l,m = views()
        out_a,out_b = a(g,l,m),b(g,l,m)
        (out_a['dino'].sum()+out_a['ibot'].sum()).backward()
        (out_b['dino'].sum()+out_b['ibot'].sum()).backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad)

    def test_encoder_batch_invariance(self):
        student,_,_ = tiny(); student.eval()
        a,b = torch.randn(1,3,8,8),torch.randn(3,3,8,8)
        with torch.no_grad():
            alone = student.encoder(a)['cls']
            together = student.encoder(torch.cat((a,b)))['cls'][:1]
        torch.testing.assert_close(alone,together,atol=2e-6,rtol=2e-6)


class LossContracts(unittest.TestCase):
    def test_sinkhorn_matches_probability_domain_official_balancing(self):
        torch.manual_seed(3)
        logits = torch.randn(12,5)*.1
        q = (logits/.07).exp().T
        q /= q.sum()
        for _ in range(3):
            q /= q.sum(1,keepdim=True); q /= 5
            q /= q.sum(0,keepdim=True); q /= 12
        expected = (q*12).T
        actual = sinkhorn_targets(logits,.07)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(actual.sum(1),torch.ones(12))

    def test_sinkhorn_handles_large_logits_and_detaches_targets(self):
        logits = (torch.randn(10,7)*1000).requires_grad_()
        q = sinkhorn_targets(logits,.04)
        self.assertTrue(torch.isfinite(q).all())
        self.assertFalse(q.requires_grad)
        torch.testing.assert_close(q.sum(1),torch.ones(10),atol=.01,rtol=.01)

    def test_dino_excludes_same_view(self):
        # B=1, no locals isolates the two cross-global pairs.
        targets = torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
        same = torch.tensor([[10.,-10.],[-10.,10.]],requires_grad=True)
        crossed = same.flip(0).detach().requires_grad_()
        high = dino_loss(same,targets,1,0,temperature=1)
        low = dino_loss(crossed,targets,1,0,temperature=1)
        self.assertGreater(high.item(),19)
        self.assertLess(low.item(),1e-5)
        low.backward(); self.assertIsNone(targets.grad)

    def test_dino_local_views_are_supervised_by_both_globals(self):
        targets = torch.tensor([[1.,0.],[0.,1.]])
        logits = torch.tensor([[3.,-2.],[-1.,2.],[.2,.8]],requires_grad=True)
        expected = (F.cross_entropy(logits[1:2],torch.tensor([0])) +
                    F.cross_entropy(logits[0:1],torch.tensor([1])) +
                    F.cross_entropy(logits[2:3],torch.tensor([0])) +
                    F.cross_entropy(logits[2:3],torch.tensor([1]))) / 4
        torch.testing.assert_close(dino_loss(logits,targets,1,1,temperature=1),expected)

    def test_ibot_normalizes_per_crop_not_per_masked_token(self):
        masks = torch.tensor([[1,0,0],[1,1,1],[0,0,0],[0,0,0]],dtype=torch.bool)
        logits = torch.tensor([[0.,2.],[3.,0.],[1.,0.],[2.,0.]],requires_grad=True)
        target = torch.tensor([[1.,0.]]*4,requires_grad=True)
        ce = F.cross_entropy(logits,torch.zeros(4,dtype=torch.long),reduction='none')
        expected = (ce[0]+ce[1:].mean())/4
        actual = ibot_loss(logits,target,masks,temperature=1)
        torch.testing.assert_close(actual,expected)
        actual.backward();self.assertIsNone(target.grad)

    def test_koleo_penalizes_crowded_embeddings(self):
        crowded = torch.tensor([[1.,0.],[1.,.00001],[1.,-.00001]],requires_grad=True)
        spread = torch.tensor([[1.,0.],[-.5,.866],[-.5,-.866]],requires_grad=True)
        self.assertGreater(koleo_loss(crowded).item(),koleo_loss(spread).item()+5)
        koleo_loss(spread).backward();self.assertTrue(torch.isfinite(spread.grad).all())
        with self.assertRaises(ValueError):koleo_loss(torch.ones(1,3))


class ReferenceAndDataContracts(unittest.TestCase):
    def test_streaming_basis_matches_direct_uncentered_svd_subspace(self):
        torch.manual_seed(7)
        tokens = torch.randn(200,16,dtype=torch.float64)
        accumulator = GramAccumulator(16)
        accumulator.add(tokens[:80]);accumulator.add(tokens[80:])
        basis,_ = accumulator.basis(8)
        direct = torch.linalg.svd(tokens,full_matrices=False).Vh[:8].T
        torch.testing.assert_close((basis@basis.T).double(),direct@direct.T,atol=1e-6,rtol=1e-6)

    def test_reference_is_repeatable_subset_with_source_coverage(self):
        rows = [{'sample_id':str(i),'source':'a' if i<9 else 'b'} for i in range(10)]
        subset = reference_rows(rows,4,42)
        self.assertEqual(subset,reference_rows(rows,4,42))
        self.assertEqual(len({r['sample_id'] for r in subset}),4)
        self.assertEqual({r['source'] for r in subset},{'a','b'})

    def test_prepare_rejects_fake_duplicate_and_filters_small(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            Image.new('RGB',(240,224)).save(root/'a.png')
            Image.new('RGB',(100,100)).save(root/'b.png')
            parent=root/'parent.csv'
            parent.write_text('sample_id,image_path,label\na,a.png,real\nb,b.png,real\n')
            result=prepare(parent,root,root/'eligible.csv')
            self.assertEqual(result['eligible_count'],1)
            self.assertEqual(read_manifest(root/'eligible.csv')[0]['sample_id'],'a')
            parent.write_text('sample_id,image_path,label\na,a.png,fake\n')
            with self.assertRaises(ValueError):read_manifest(parent)
            parent.write_text('sample_id,image_path,label\na,a.png,real\nb,a.png,real\n')
            with self.assertRaises(ValueError):read_manifest(parent)

    def test_prepare_metadata_avoids_pixel_io_and_falls_back_only_when_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root/'parent.csv'
            parent.write_text('sample_id,image_path,width,height\na,absent.png,300,240\nb,small.png,100,100\n')
            with patch('baselines.gsd_dino_encoder.data.Image.open', side_effect=AssertionError('unexpected image I/O')):
                result = prepare(parent, root, root/'eligible.csv')
            self.assertEqual(result['metadata_count'], 2)
            self.assertEqual(result['header_count'], 0)
            self.assertEqual(result['eligible_count'], 1)
            Image.new('RGB', (240, 224)).save(root/'fallback.png')
            parent.write_text('sample_id,image_path,width,height\na,absent.png,300,240\nb,fallback.png,,\n')
            original_open = Image.open
            with patch('baselines.gsd_dino_encoder.data.Image.open', wraps=original_open) as opened:
                result = prepare(parent, root, root/'eligible.csv')
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(result['header_count'], 1)
            self.assertEqual(result['metadata_count'], 1)

    def test_multicrop_is_repeatable_and_does_not_change_caller_rng(self):
        config = json.loads(Path(__file__).parents[1].joinpath('config.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            Image.fromarray(np.random.default_rng(2).integers(0,256,(300,400,3),dtype=np.uint8)).save(root/'a.png')
            Image.fromarray(np.random.default_rng(3).integers(0,256,(300,400,3),dtype=np.uint8)).save(root/'b.png')
            rows=[{'image_path':str(root/(name+'.png')),'sample_id':name} for name in ('a','b')]
            ds=RealViews(rows,root,config)
            before=torch.get_rng_state().clone();py_before=random.getstate()
            sample=ds[0]
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(py_before,random.getstate())
            again=ds[0]
            for a,b in zip(sample['global']+sample['local'],again['global']+again['local']):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            batch=CollateViews(config)([sample,ds[1]])
            self.assertEqual(batch['global'].shape,(4,3,224,224))
            self.assertEqual(batch['local'].shape,(16,3,98,98))
            self.assertEqual(batch['masks'].shape,(4,256))
            self.assertTrue(batch['masks'].any());self.assertTrue((~batch['masks']).any())
            ds.epoch=1
            self.assertFalse(torch.equal(sample['global'][0],ds[0]['global'][0]))


class TrainingArtifactContracts(unittest.TestCase):
    def test_fp16_style_overflow_skips_update(self):
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        before=[p.detach().clone() for p in model.parameters()]
        scaler.scale(model(torch.ones(2,2)).sum()).backward()
        next(model.parameters()).grad[0,0]=float('inf')
        updated,_=finish_update(model,optimizer,scaler,3.)
        self.assertFalse(updated);self.assertEqual(scaler.get_scale(),64.)
        for p,q in zip(model.parameters(),before):torch.testing.assert_close(p,q)
        scaler.scale(model(torch.ones(2,2)).sum()).backward()
        self.assertTrue(finish_update(model,optimizer,scaler,3.)[0])

    def test_schedule_endpoints_and_parameter_groups(self):
        config=json.loads(Path(__file__).parents[1].joinpath('config.json').read_text())
        start,end=schedules(config,0,100),schedules(config,99,100)
        self.assertAlmostEqual(start['momentum'],.994)
        self.assertAlmostEqual(end['momentum'],1.)
        self.assertAlmostEqual(end['lr_factor'],config['lr_end_fraction'])
        student,_,_=tiny()
        groups=optimizer_groups(student,config)
        params=[p for group in groups for p in group['params']]
        self.assertEqual(len(params),len(list(student.parameters())))
        self.assertEqual(len(params),len({id(p) for p in params}))

    def test_export_roundtrip_removes_heads_and_preserves_cls(self):
        student,teacher,config=tiny()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); checkpoint=root/'last.ckpt'; output=root/'encoder.pt'
            state={'format':'gsd-dino-training-v1','teacher':teacher.state_dict(),
                   'vision_config':teacher.encoder.vision.config.to_dict(),'config':config,
                   'identity':{},'epoch':0,'updates':1,'smoke_only':False,'epoch_completed':True}
            atomic_save(state,checkpoint)
            artifact=export_checkpoint(checkpoint,output)
            self.assertFalse(any('head' in key for key in artifact['encoder']))
            encoder,_=load_encoder(output)
            image=torch.randn(2,3,8,8)
            torch.testing.assert_close(encoder(image)['cls'],teacher.encoder(image)['cls'])
            state['smoke_only']=True;atomic_save(state,checkpoint)
            with self.assertRaises(ValueError):export_checkpoint(checkpoint,root/'bad.pt')

    def test_production_config_blocks_architecture_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json';path.write_text('{"k":32}')
            with self.assertRaises(ValueError):load_config(path)

    def test_checkpoint_restores_optimizer_and_ema_for_next_update(self):
        student,teacher,config=tiny()
        g,l,m=views()
        optimizer=torch.optim.AdamW(student.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        def step(network,target,opt):
            opt.zero_grad(set_to_none=True)
            loss,_=ssl_loss(network(g,l,m),target.teacher_forward(g,m),m,config,.07)
            loss.backward();opt.step();update_teacher(network,target,.9)
        step(student,teacher,optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.ckpt'
            save_checkpoint(path,student,teacher,optimizer,scaler,config,{},0,1,1,False,True)
            state=torch.load(path,weights_only=False)
            resumed,resumed_teacher,_=tiny()
            resumed.load_state_dict(state['student']);resumed_teacher.load_state_dict(state['teacher'])
            resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-4)
            resumed_optimizer.load_state_dict(state['optimizer'])
            step(student,teacher,optimizer)
            step(resumed,resumed_teacher,resumed_optimizer)
            for p,q in zip(student.parameters(),resumed.parameters()):torch.testing.assert_close(p,q)
            for p,q in zip(teacher.parameters(),resumed_teacher.parameters()):torch.testing.assert_close(p,q)
            self.assertTrue(state['epoch_completed']);self.assertFalse(state['smoke_only'])

    def test_extraction_preserves_sample_order_and_feature_file(self):
        _,teacher,config=tiny()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,color in [('b',(10,20,30)),('a',(80,40,50))]:
                Image.new('RGB',(12,10),color=color).save(root/(name+'.png'))
            manifest=root/'eval.csv'
            manifest.write_text('sample_id,image_path,label\nb,b.png,fake\na,a.png,real\n')
            checkpoint=root/'last.ckpt';encoder=root/'encoder.pt'
            atomic_save({'format':'gsd-dino-training-v1','teacher':teacher.state_dict(),
                         'vision_config':teacher.encoder.vision.config.to_dict(),'config':config,
                         'identity':{},'epoch':0,'updates':1,'smoke_only':False,'epoch_completed':True},checkpoint)
            export_checkpoint(checkpoint,encoder)
            args=SimpleNamespace(output=root/'features.npy',batch_size=2,workers=0,encoder=encoder,
                                 device='cpu',manifest=manifest,pixel_root=root)
            extract(args)
            features=np.load(args.output)
            meta=json.loads(args.output.with_suffix('.json').read_text())
            self.assertEqual(features.shape,(2,16))
            self.assertTrue(np.isfinite(features).all())
            self.assertEqual(meta['sample_ids'],['b','a'])


if __name__ == '__main__':
    unittest.main()
