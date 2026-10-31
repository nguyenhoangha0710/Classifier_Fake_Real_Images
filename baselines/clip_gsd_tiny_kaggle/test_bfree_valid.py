"""CPU unit/integration tests. These are not a pretrained Kaggle GPU benchmark."""
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

import evaluate_bfree_valid as r
import test_uploaded_evaluation as fixture
from build_bfree_valid_notebook import build_notebook,NOTEBOOK


class BFreeValidTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=self.root/'input/b-free-training-data'
        self.data.mkdir(parents=True)
        self.fixture=fixture.UploadedEvaluationTests()
        self.fixture.setUp()
        self.cfg=dict(self.fixture.cfg,test_batch=8)
        self.opts=dict(r.BFREE_DEFAULTS,input_root=str(self.root/'input'),output_root=str(self.root/'out'),
            checkpoint_cache=str(self.root/'cache'),expected_signature='unit-signature',
            save_every_batches=1,log_every_batches=100,header_log_every=1000,resume_from=None)
        rows=[dict(id=f'{i:012d}',same_category='dog',diff_category='cat') for i in range(12)]
        pd.DataFrame(rows).to_csv(self.data/'train_list.csv',index=False)
        pd.DataFrame(rows[:8]).to_csv(self.data/'valid_list.csv',index=False)
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(r,'BFREE_ALL_COUNT',12))
        self.stack.enter_context(patch.object(r,'BFREE_VALID_COUNT',8))
        self.stack.enter_context(patch.object(r,'BFREE_IMAGE_SIDE',28))
        self.stack.enter_context(patch.dict(r.BFREE_LIST_MD5,{
            key:hashlib.md5((self.data/(key+'.csv')).read_bytes()).hexdigest() for key in ('train_list','valid_list')}))
        for v,variant in enumerate((r.BFREE_REAL,*r.BFREE_BRANCHES)):
            folder=self.data/variant/variant
            folder.mkdir(parents=True)
            for i in range(8):
                pixels=np.random.default_rng(20*v+i).integers(0,256,(28,28,3),dtype=np.uint8)
                Image.fromarray(pixels).save(folder/f'{i:012d}.png')
        self.out=Path(self.opts['output_root']); self.out.mkdir()

    def manifest(self,opts=None,out=None):
        return r.prepare_bfree_manifest(self.data,opts or self.opts,out or self.out)

    def models(self,cfg=None):
        return self.fixture.model(cfg or self.cfg)

    def test_exact_valid_ids_and_six_balanced_branch_cohorts(self):
        frame,ids,identity=self.manifest()
        self.assertEqual(len(ids),8)
        self.assertEqual(set(ids),{f'{i:012d}' for i in range(8)})
        self.assertEqual(len(frame),56)
        self.assertEqual(identity['metadata']['total_branch_predictions'],96)
        first_batches=[]
        for branch in r.BFREE_BRANCHES:
            cohort=r.bfree_branch_frame(frame,ids,branch)
            self.assertEqual(len(cohort),16)
            self.assertEqual(cohort.label.tolist(),[0,1]*8)
            self.assertEqual(cohort.coco_id.tolist(),[image_id for image_id in ids for _ in range(2)])
            self.assertEqual(cohort.iloc[:8].coco_id.nunique(),4)
            first_batches.append(cohort.iloc[:8].coco_id.tolist())
        self.assertTrue(all(order==first_batches[0] for order in first_batches))
        # Master-list IDs8..11 must never reach evaluation.
        self.assertFalse(set(frame.coco_id)&{f'{i:012d}' for i in range(8,12)})

    def test_official_list_checksum_missing_images_and_wrong_headers_fail(self):
        with (self.data/'valid_list.csv').open('ab') as handle:
            handle.write(b'\n')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.manifest()
        with (self.data/'valid_list.csv').open('rb+') as handle:
            handle.truncate((self.data/'valid_list.csv').stat().st_size-1)
        missing=self.data/r.BFREE_BRANCHES[0]/r.BFREE_BRANCHES[0]/'000000000003.png'
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            self.manifest()
        Image.new('RGB',(14,28)).save(missing)
        with self.assertRaisesRegex(ValueError,'Expected 28x28'):
            self.manifest()

    def test_single_nested_mount_and_checkpoint_search_exclude_image_tree(self):
        flat=self.data/r.BFREE_BRANCHES[0]
        nested=flat/r.BFREE_BRANCHES[0]
        for path in nested.iterdir():
            shutil.copyfile(path,flat/path.name)
        # Both populated layouts are ambiguous until the original nested example is removed.
        with self.assertRaises(ValueError):
            r.bfree_folder(self.data,r.BFREE_BRANCHES[0],'000000000000')
        for path in nested.iterdir():
            path.unlink()
        self.assertEqual(r.bfree_folder(self.data,r.BFREE_BRANCHES[0],'000000000000'),flat)
        self.assertEqual(r.find_bfree_root(self.opts),self.data)
        checkpoint=self.root/'input/uploaded/best.pt'
        checkpoint.parent.mkdir()
        torch.save(self.fixture.payload(),checkpoint)
        source,path,payload=r.find_bfree_checkpoint(self.opts,self.data)
        self.assertEqual(source,checkpoint)
        self.assertEqual(payload['signature'],'unit-signature')

    def test_smoke_is_explicit_separate_and_all_branches_remain_selected(self):
        self.assertFalse(r.BFREE_DEFAULTS['smoke_only'])
        self.assertEqual(len(r.BFREE_BRANCHES),6)
        options=r.bfree_options(dict(self.opts,smoke_only=True))
        self.assertNotEqual(options['output_root'],self.opts['output_root'])
        self.assertIsNone(options['resume_from'])
        small,ids,identity=self.manifest(options)
        self.assertEqual(len(ids),4)
        self.assertEqual(len(small),28)
        self.assertTrue(identity['metadata']['smoke'])
        for change in (dict(cases=['native']),dict(max_ids=1),dict(threshold=.1),dict(smoke_only=1),dict(micro_batch=2)):
            with self.assertRaises(ValueError):
                r.bfree_options(dict(self.opts,**change))

    def test_native_preprocessing_matches_existing_pipeline_and512_padding(self):
        path=self.root/'large.png'
        Image.fromarray(np.random.default_rng(21).integers(0,256,(512,512,3),dtype=np.uint8)).save(path)
        frame=pd.DataFrame([dict(path=str(path),width=512,height=512,label=0)])
        x,_,_=r.BFreeImages(frame,self.cfg)[0]
        original,_,_=r.Images(frame,self.cfg,42,False,0,'native')[0]
        torch.testing.assert_close(x,original,rtol=0,atol=0)
        self.assertEqual(tuple(x.shape),(3,518,518))
        self.assertEqual(r.input_geometry(512,512,self.cfg)['tokens'],1370)

    def test_cache_preserves_zero_padded_ids_and_recomputes_real_each_branch(self):
        manifest,ids,_=self.manifest()
        model,anchor=self.models()
        for branch in r.BFREE_BRANCHES[:2]:
            frame=r.bfree_branch_frame(manifest,ids,branch)
            with patch.object(r,'estimate_bases',wraps=r.estimate_bases) as basis:
                result,complete=r.evaluate_bfree_branch(model,anchor,frame,self.cfg,self.opts,branch,
                    self.out,'unit-fp',torch.device('cpu'),float('inf'),lambda *args:None)
                self.assertEqual(basis.call_count,2)  # no shared-real score reuse
            self.assertTrue(complete)
            cached,n=r.read_bfree_parts(self.out/'progress'/branch,frame,branch,'unit-fp')
            self.assertEqual(n,16)
            self.assertEqual(cached.coco_id.tolist(),frame.coco_id.tolist())
            with patch.object(r,'estimate_bases',side_effect=AssertionError('completed branch must not rerun')):
                reused,complete=r.evaluate_bfree_branch(model,anchor,frame,self.cfg,self.opts,branch,
                    self.out,'unit-fp',torch.device('cpu'),float('inf'),lambda *args:None)
                self.assertTrue(complete)
        summary=r.bfree_reports({branch:pd.read_csv(self.out/f'predictions_{branch}.csv') for branch in r.BFREE_BRANCHES[:2]},self.out)
        self.assertEqual(len(summary),2)
        macro=pd.read_csv(self.out/'macro_metrics.csv').iloc[0]
        self.assertFalse(macro.all_branches_complete)
        self.assertEqual(macro.completed_branches,2)

    def test_planned_pause_flushes_complete_batch_then_resume_matches_fresh(self):
        manifest,ids,_=self.manifest(); branch=r.BFREE_BRANCHES[0]
        frame=r.bfree_branch_frame(manifest,ids,branch); model,anchor=self.models()
        expired=[False]; original=r.estimate_bases
        def one_batch(*args,**kwargs):
            bases=original(*args,**kwargs); expired[0]=True; return bases
        with patch.object(r,'estimate_bases',side_effect=one_batch), \
                patch.object(r.time,'monotonic',side_effect=lambda:11. if expired[0] else 0.):
            part,complete=r.evaluate_bfree_branch(model,anchor,frame,self.cfg,
                dict(self.opts,save_every_batches=10),branch,self.out,'unit-fp',torch.device('cpu'),10.,lambda *args:None)
        self.assertFalse(complete); self.assertEqual(len(part),8)
        resumed,complete=r.evaluate_bfree_branch(model,anchor,frame,self.cfg,self.opts,branch,
            self.out,'unit-fp',torch.device('cpu'),float('inf'),lambda *args:None)
        fresh=self.root/'fresh'; fresh.mkdir()
        reference,_=r.evaluate_bfree_branch(model,anchor,frame,self.cfg,self.opts,branch,
            fresh,'unit-fp',torch.device('cpu'),float('inf'),lambda *args:None)
        self.assertTrue(complete)
        np.testing.assert_allclose(resumed.prob_fake,reference.prob_fake,rtol=1e-6,atol=1e-7)
        self.assertEqual(resumed.batch_id.tolist(),[0]*8+[1]*8)

    def test_changed_fingerprint_or_corrupt_predictions_rejected(self):
        manifest,ids,_=self.manifest(); branch=r.BFREE_BRANCHES[0]
        frame=r.bfree_branch_frame(manifest,ids,branch); model,anchor=self.models()
        r.evaluate_bfree_branch(model,anchor,frame,self.cfg,self.opts,branch,self.out,'unit-fp',
            torch.device('cpu'),float('inf'),lambda *args:None)
        folder=self.out/'progress'/branch
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            r.read_bfree_parts(folder,frame,branch,'different')
        part=folder/'part_0000000.csv'
        corrupt=pd.read_csv(part,dtype={'coco_id':str}); corrupt['prob_fake']=.1234
        corrupt.to_csv(part,index=False)
        with self.assertRaisesRegex(ValueError,'probability'):
            r.read_bfree_parts(folder,frame,branch,'unit-fp')

    def test_all_six_driver_and_cross_session_restore_without_repeated_forwards(self):
        checkpoint=self.root/'input/uploaded/best.pt'; checkpoint.parent.mkdir()
        torch.save(self.fixture.payload(),checkpoint)
        opts=dict(self.opts,checkpoint_path=str(checkpoint),resume_from='auto')
        original=r.evaluate_bfree_branch
        def cpu_branch(*args,**kwargs):
            values=list(args); values[8]=torch.device('cpu')
            return original(*values,**kwargs)
        with patch.object(r,'build_models',side_effect=lambda cfg,device,gsd:self.models(cfg)), \
                patch.object(r,'evaluate_bfree_branch',side_effect=cpu_branch), \
                patch.object(r.torch.cuda,'is_available',return_value=True), \
                patch.object(r.torch.cuda,'get_device_name',return_value='mock-GPU'), \
                patch.object(r.torch.cuda,'empty_cache'),patch.object(r,'seed_all'):
            with patch.object(r,'estimate_bases',wraps=r.estimate_bases) as basis:
                table=r.run_bfree_valid_evaluation(opts)
                self.assertEqual(basis.call_count,12)
            self.assertEqual(len(table),6)
            state=json.loads((self.out/'evaluation_state.json').read_text())
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['expected_prediction_rows'],96)
            shutil.copytree(self.out,self.root/'input/previous')
            resumed=self.root/'resumed'
            with patch.object(r,'estimate_bases',side_effect=AssertionError('cached branches must not rerun')):
                second=r.run_bfree_valid_evaluation(dict(opts,output_root=str(resumed)))
            self.assertEqual(len(second),6)
            self.assertTrue(pd.read_csv(resumed/'macro_metrics.csv').iloc[0].all_branches_complete)

    def test_notebook_embeds_runtime_defaults_to_full_and_no_genimage_input(self):
        notebook=build_notebook()
        for cell in notebook['cells']:
            if cell['cell_type']=='code' and not ''.join(cell['source']).startswith('%pip'):
                ast.parse(''.join(cell['source']))
        namespace={'__name__':'bfree_notebook_test'}
        exec(''.join(notebook['cells'][3]['source']),namespace)
        exec(''.join(notebook['cells'][4]['source']),namespace)
        self.assertFalse(namespace['SMOKE_ONLY'])
        self.assertNotIn('test_manifest_path',namespace['BFREE_CONFIG'])
        captured=[]
        namespace['run_bfree_valid_evaluation']=lambda config:captured.append(config)
        exec(''.join(notebook['cells'][5]['source']),namespace)
        self.assertEqual(len(captured),1)
        self.assertFalse(captured[0]['smoke_only'])
        self.assertEqual(len(namespace['BFREE_BRANCHES']),6)
        self.assertEqual(json.loads((Path(__file__).parent/NOTEBOOK).read_text(encoding='utf-8')),notebook)


if __name__=='__main__':
    unittest.main()
