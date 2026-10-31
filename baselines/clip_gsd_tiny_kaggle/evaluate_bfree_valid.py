"""GPU-only zero-shot GSD on B-Free's official validation IDs.

The six fake folders are six evaluation branches of SD2.1, NOT six generators.
Every logical batch contains four real/fake ID pairs; all eight images determine
the GSD basis. Real scores must therefore be recomputed for each branch.
"""
if 'find_uploaded_checkpoint' not in globals():
    from pathlib import Path
    _bfree_here = Path(__file__).resolve().parent
    exec(compile((_bfree_here/'evaluate_uploaded_best.py').read_text(encoding='utf-8'),
                 str(_bfree_here/'evaluate_uploaded_best.py'), 'exec'), globals())

import os
import struct

BFREE_EVALUATOR = 'gsd_bfree_official_valid_native_pairs4_batch8_v1'
BFREE_REAL = 'COCO_real_512'
BFREE_BRANCHES = (
    'SD2.1_selfconditioned', 'SD2.1_selfconditioned_origBG',
    'SD2.1_inpainted_samecat', 'SD2.1_inpainted_samecat_origBG',
    'SD2.1_inpainted_diffcat', 'SD2.1_inpainted_diffcat_origBG',
)
BFREE_IMAGE_SIDE = 512
BFREE_ALL_COUNT, BFREE_VALID_COUNT = 51517, 10304
BFREE_LIST_MD5 = dict(train_list='06581da53bf81741501396d71eb7ec5f',
                      valid_list='72e7b4be2d049029dbedca7e75af91b0')
BFREE_DEFAULTS = dict(
    input_root='/kaggle/input', bfree_root=None, checkpoint_path=None,
    output_root='/kaggle/working/gsd_bfree_valid_v1',
    checkpoint_cache='/kaggle/working/gsd_bfree_checkpoint_cache',
    expected_signature=UPLOAD_DEFAULTS['expected_signature'], expected_epoch=1,
    pretrained_model_path=None, resume_from='auto', smoke_only=False,
    micro_batch=1, anchor_micro_batch=1, num_workers=0,
    save_every_batches=10, log_every_batches=10, header_log_every=2000,
    time_budget_hours=9.5,
)


def bfree_options(options=None):
    opts = dict(BFREE_DEFAULTS, **(options or {}))
    if not isinstance(opts['smoke_only'], bool):
        raise ValueError('smoke_only must be True/False')
    if opts['smoke_only']:
        opts.update(output_root='/kaggle/working/gsd_bfree_valid_smoke_v1', resume_from=None)
    if opts['micro_batch'] != 1 or opts['anchor_micro_batch'] != 1 or opts['num_workers'] != 0:
        raise ValueError('B-Free GPU protocol fixes microbatch1/anchor1/workers0, logical batch8')
    for key in ('save_every_batches','log_every_batches','header_log_every'):
        if isinstance(opts[key], bool) or not isinstance(opts[key], int) or opts[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < opts['time_budget_hours'] <= 10.5:
        raise ValueError('time_budget_hours must be in (0,10.5] for a GPU session')
    # Cases, max_ids and threshold cannot silently reduce or retune this experiment.
    if set(opts) - set(BFREE_DEFAULTS):
        raise ValueError('Unknown B-Free options: '+str(sorted(set(opts)-set(BFREE_DEFAULTS))))
    return opts


def bfree_mount_roots(input_root):
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f'Missing input_root: {root}')
    mounts = [p for p in root.iterdir() if p.is_dir() and p.name != 'datasets']
    if (root/'datasets').is_dir():
        for owner in (root/'datasets').iterdir():
            if owner.is_dir():
                mounts.extend(p for p in owner.iterdir() if p.is_dir())
    return sorted(mounts)


def find_bfree_root(opts):
    candidates = [Path(opts['bfree_root'])] if opts['bfree_root'] else bfree_mount_roots(opts['input_root'])
    matches = [p for p in candidates if (p/'valid_list.csv').is_file() and (p/'train_list.csv').is_file()
               and (p/BFREE_REAL).is_dir()]
    if len(matches) != 1:
        raise ValueError('Set bfree_root to the dataset directory containing valid_list.csv, train_list.csv '
                         f'and COCO_real_512. Found {list(map(str,matches))}')
    return matches[0]


def find_bfree_checkpoint(opts, bfree_root):
    if opts['checkpoint_path']:
        return find_uploaded_checkpoint(opts)
    # Do NOT recursively walk the hundreds of thousands of B-Free images to find weights.
    candidates = []
    for root in bfree_mount_roots(opts['input_root']):
        if os.path.normcase(os.path.abspath(root)) == os.path.normcase(os.path.abspath(bfree_root)):
            continue
        pending = [(root,0)]
        while pending:
            folder, depth = pending.pop()
            if (folder/'data.pkl').is_file() and (folder/'version').is_file() and (folder/'data').is_dir():
                candidates.append(folder)
                continue
            for item in folder.iterdir():
                if item.is_file() and item.name in ('best.pt','best.pth','best.zip'):
                    candidates.append(item)
                elif item.is_dir() and not item.is_symlink() and depth < 3 and item.name not in ('data','.data'):
                    pending.append((item,depth+1))
    matches, errors = [], []
    for candidate in sorted(set(candidates)):
        try:
            matches.append(find_uploaded_checkpoint(dict(opts,checkpoint_path=str(candidate))))
        except (ValueError,RuntimeError,OSError) as exc:
            errors.append(f'{candidate}: {exc}')
    if not matches:
        raise ValueError('No matching augmented best checkpoint; set checkpoint_path explicitly. '+ '; '.join(errors))
    if len({file_sha(path) for _,path,_ in matches}) != 1:
        raise ValueError('Different matching checkpoints found; set checkpoint_path explicitly')
    return matches[0]


def bfree_list(path, key, count):
    raw = path.read_bytes()
    checksum = hashlib.md5(raw).hexdigest()
    if checksum != BFREE_LIST_MD5[key]:
        raise ValueError(f'{path.name}: checksum differs from official B-Free list ({checksum})')
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    if set(frame.columns) != {'id','same_category','diff_category'} or len(frame) != count:
        raise ValueError(f'Wrong schema/count in {path}')
    if frame.id.duplicated().any() or not frame.id.str.fullmatch(r'\d{12}').all():
        raise ValueError(f'Invalid or duplicated COCO IDs in {path}')
    return frame,checksum


def bfree_folder(root, name, example_id):
    choices = [p for p in (root/name/name,root/name) if (p/(example_id+'.png')).is_file()]
    if not choices:
        raise FileNotFoundError(f'Missing {name}/{example_id}.png in either single/double nested layout')
    if len(choices) != 1:
        raise ValueError(f'Cannot uniquely locate {name}; expected single/double nested folder, found {choices}')
    return choices[0]


def bfree_png_header(path):
    with path.open('rb') as handle:
        header = handle.read(33)
    if len(header) != 33 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Invalid PNG header: {path}')
    width,height = struct.unpack('>II',header[16:24])
    if (width,height) != (BFREE_IMAGE_SIDE,BFREE_IMAGE_SIDE):
        raise ValueError(f'Expected {BFREE_IMAGE_SIDE}x{BFREE_IMAGE_SIDE}: {path} is {width}x{height}')
    return width,height,path.stat().st_size


def prepare_bfree_manifest(root, opts, out):
    all_ids,train_md5 = bfree_list(root/'train_list.csv','train_list',BFREE_ALL_COUNT)
    valid,valid_md5 = bfree_list(root/'valid_list.csv','valid_list',BFREE_VALID_COUNT)
    if not set(valid.id) <= set(all_ids.id):
        raise ValueError('Valid IDs must be contained in the released train_list master list')
    common = all_ids.set_index('id').loc[valid.id]
    if any(common[k].tolist() != valid[k].tolist() for k in ('same_category','diff_category')):
        raise ValueError('Train/valid category metadata disagree for the same IDs')
    # Deterministic ID-only order. Every branch uses exactly the same four IDs per batch.
    ids = sorted(valid.id, key=lambda image_id: (digest(['bfree_valid',42,image_id]),image_id))
    if opts['smoke_only']:
        ids = ids[:4]
    directories = {name:bfree_folder(root,name,ids[0]) for name in (BFREE_REAL,*BFREE_BRANCHES)}
    records = []
    print(f'[manifest] {len(ids):,} valid IDs x7 folders; PNG headers/path/size ONLY, no RGB/hash scan.',flush=True)
    for image_id in ids:
        for variant in (BFREE_REAL,*BFREE_BRANCHES):
            path = directories[variant]/(image_id+'.png')
            if not path.is_file():
                raise FileNotFoundError(f'Missing B-Free valid member: {path}; no IDs are silently dropped')
            width,height,size = bfree_png_header(path)
            relative = path.relative_to(root).as_posix()
            records.append(dict(coco_id=image_id,variant=variant,label=int(variant!=BFREE_REAL),
                relative_path=relative,path=str(path),width=width,height=height,file_bytes=size,
                sample_id=hashlib.sha256(relative.encode()).hexdigest()))
            if len(records)%opts['header_log_every']==0:
                print(f'[manifest] {len(records):,}/{7*len(ids):,} headers checked',flush=True)
    manifest = pd.DataFrame(records)
    selected = valid.set_index('id').loc[ids].reset_index()
    metadata = dict(dataset='baopv051/b-free-training-data',split='official_validation_ids',
        source_train_list_count=len(all_ids),source_valid_list_count=len(valid),
        valid_contained_in_train_list=True,train_list_md5=train_md5,valid_list_md5=valid_md5,
        selected_id_count=len(ids),id_order_hash=digest(ids),smoke=opts['smoke_only'],
        unique_images=len(manifest),predictions_per_branch=2*len(ids),total_branch_predictions=12*len(ids),
        branches=list(BFREE_BRANCHES),
        audit='PNG headers + file sizes; RGB decode checked at inference. No full image/pixel hash audit; '
              'resume assumes identical dataset contents/version, same-size content edits are not detected by provenance.')
    fields = ['coco_id','variant','label','relative_path','width','height','file_bytes','sample_id']
    identity = dict(metadata=metadata,records_hash=digest(manifest[fields].to_dict('records')))
    bind_upload_metadata(out/'manifest_identity.json',identity)
    save_csv(out/'bfree_valid_manifest.csv',manifest)
    save_csv(out/'bfree_valid_ids.csv',selected)
    atomic_json(out/'bfree_manifest_summary.json',metadata)
    return manifest,ids,identity


def bfree_branch_frame(manifest, ids, branch):
    if branch not in BFREE_BRANCHES:
        raise ValueError(f'Unknown B-Free branch: {branch}')
    lookup = manifest.set_index(['coco_id','variant'])
    records = []
    for image_id in ids:
        for variant in (BFREE_REAL,branch):
            record = lookup.loc[(image_id,variant)].to_dict()
            record.update(coco_id=image_id,variant=variant,generator=branch)
            records.append(record)
    frame = pd.DataFrame(records)
    if not frame.label.tolist() == [0,1]*len(ids):
        raise ValueError('Expected interleaved real/fake ID pairs')
    return frame


class BFreeImages(Dataset):
    def __init__(self, frame, cfg):
        self.rows,self.cfg = frame.to_dict('records'),cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        with Image.open(row['path']) as source:
            if source.format != 'PNG' or source.size != (row['width'],row['height']):
                raise ValueError(f'Image changed since manifest: {row["path"]}')
            image = source.convert('RGB')
        geom = input_geometry(*image.size,self.cfg)
        image = image.crop((geom['crop_left'],geom['crop_top'],
                            geom['crop_left']+geom['crop_width'],geom['crop_top']+geom['crop_height']))
        x = torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255.
        h,w = x.shape[-2:]
        x = F.pad(x.unsqueeze(0),(0,(-w)%14,0,(-h)%14),mode='replicate').squeeze(0)
        mean = x.new_tensor([.48145466,.4578275,.40821073])[:,None,None]
        std = x.new_tensor([.26862954,.26130258,.27577711])[:,None,None]
        return (x-mean)/std,int(row['label']),index


def read_bfree_parts(folder,frame,branch,fingerprint):
    shards = sorted(folder.glob('part_*.csv'))
    if not shards:
        return pd.DataFrame(),0
    if not (folder/'metadata.json').is_file():
        raise ValueError('Progress has no B-Free fingerprint metadata')
    bind_upload_metadata(folder/'metadata.json',dict(evaluator=BFREE_EVALUATOR,run=fingerprint,branch=branch))
    parts,done = [],0
    for path in shards:
        part = pd.read_csv(path,dtype={'coco_id':str,'sample_id':str})
        if path.name != f'part_{done:07d}.csv' or not len(part):
            raise ValueError('Non-contiguous/empty B-Free progress shard')
        parts.append(part); done += len(part)
    prior = pd.concat(parts,ignore_index=True)
    if done>len(frame) or (done<len(frame) and done%8):
        raise ValueError('Resume must preserve complete logical batch8')
    expected = frame.iloc[:done]
    for key in ('sample_id','coco_id','variant','label','relative_path','generator'):
        if prior[key].tolist() != expected[key].tolist():
            raise ValueError(f'B-Free cache/manifest mismatch in {key}')
    if set(prior.case)!={branch} or set(prior.model)!={'clip_gsd'} or set(prior.seed)!={42}:
        raise ValueError('Wrong branch/model/seed in saved predictions')
    if prior.batch_id.tolist()!=[i//8 for i in range(done)]:
        raise ValueError('Saved B-Free batch membership changed')
    numeric = prior[['prob_fake','logit_real','logit_fake','logit_margin','input_tokens']].to_numpy()
    if not np.isfinite(numeric).all() or not prior.prob_fake.between(0,1).all():
        raise ValueError('Nonfinite/out-of-range cached predictions')
    logits = torch.from_numpy(prior[['logit_real','logit_fake']].to_numpy())
    if not np.allclose(prior.prob_fake,logits.softmax(-1)[:,1].numpy(),rtol=1e-6,atol=2e-7):
        raise ValueError('Cached probability does not match logits')
    if not np.allclose(prior.logit_margin,prior.logit_fake-prior.logit_real,rtol=1e-5,atol=1e-5):
        raise ValueError('Cached logit margin is corrupt')
    tokens = [input_geometry(w,h,dict(input_mode='center_crop_max_edge_pad14',crop_max_side=560))['tokens']
              for w,h in zip(expected.width,expected.height)]
    if prior.input_tokens.tolist()!=tokens:
        raise ValueError('Cached token geometry differs')
    return prior,done


@torch.no_grad()
def evaluate_bfree_branch(model,anchor,frame,cfg,opts,branch,out,fingerprint,device,deadline,progress):
    folder = out/'progress'/branch
    folder.mkdir(parents=True,exist_ok=True)
    prior,start = read_bfree_parts(folder,frame,branch,fingerprint)
    bind_upload_metadata(folder/'metadata.json',dict(evaluator=BFREE_EVALUATOR,run=fingerprint,branch=branch))
    if start == len(frame):
        return prior,True
    data = DataLoader(Subset(BFreeImages(frame,cfg),range(start,len(frame))),batch_size=8,
        shuffle=False,num_workers=0,collate_fn=native_collate,pin_memory=device.type=='cuda',drop_last=False)
    pending,parts,done,first = [],[prior] if start else [],start,start
    launched = time.monotonic()
    def flush():
        nonlocal pending
        if pending:
            part = pd.concat(pending,ignore_index=True)
            save_csv(folder/f'part_{first:07d}.csv',part)
            parts.append(part); pending=[]
            progress(branch,done)
    try:
        for local_bid,(images,labels,indices) in enumerate(data):
            if time.monotonic()>=deadline:
                break
            batch_started = time.monotonic()
            bid = start//8+local_bid
            # Same four IDs per logical batch in each branch; no cached real score reuse.
            bases = estimate_bases(anchor,images,cfg,device,model.selected)
            scores,probs = np.empty((len(images),2),np.float32),np.empty(len(images),np.float32)
            for offset,x in image_chunks(images,1):
                with amp(cfg,device):
                    logits = model(x.to(device),bases)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f'Nonfinite logits in {branch}, batch{bid}')
                scores[offset:offset+len(x)] = logits.float().cpu().numpy()
                probs[offset:offset+len(x)] = logits.float().softmax(-1)[:,1].cpu().numpy()
            item = frame.iloc[indices.tolist()].drop(columns=['path']).copy()
            item = item.assign(case=branch,model='clip_gsd',seed=42,prob_fake=probs,
                logit_real=scores[:,0],logit_fake=scores[:,1],logit_margin=scores[:,1]-scores[:,0],
                batch_id=bid,input_tokens=[(x.shape[-2]//14)*(x.shape[-1]//14)+1 for x in images])
            if not pending:
                first = int(indices[0])
            pending.append(item); done=int(indices[-1])+1
            if local_bid==0 or (bid+1)%opts['log_every_batches']==0:
                print(f'[{branch}] {done:,}/{len(frame):,} images; batch{bid+1}/{math.ceil(len(frame)/8)}; '
                      f'{time.monotonic()-batch_started:.1f}s/batch, {time.monotonic()-launched:.1f}s this branch invocation',flush=True)
            if len(pending)>=opts['save_every_batches'] or done==len(frame):
                flush()
    finally:
        flush()  # planned pause/soft exception only; a hard platform kill cannot execute this
    result = pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    complete = len(result)==len(frame)
    if len(result):
        save_csv(out/f'predictions_{branch}{"" if complete else ".partial"}.csv',result)
    return result,complete


def restore_bfree_progress(out,opts,fingerprint):
    if any((out/'progress').glob('*/part_*.csv')):
        return None
    request = opts['resume_from']
    if not request:
        return None
    if request=='auto':
        matches=[]
        # Search prior OUTPUT only, not the very large image dataset.
        for mount in bfree_mount_roots(opts['input_root']):
            if (mount/'valid_list.csv').is_file():
                continue
            pending=[(mount,0)]
            while pending:
                folder,depth=pending.pop()
                path=folder/'evaluation_state.json'
                if path.is_file():
                    state=json.loads(path.read_text(encoding='utf-8'))
                    if state.get('evaluator')==BFREE_EVALUATOR and state.get('fingerprint')==fingerprint:
                        matches.append((sum(state.get('saved_rows',{}).values()),str(folder),folder))
                if depth<3:
                    pending.extend((p,depth+1) for p in folder.iterdir() if p.is_dir() and
                                   not p.is_symlink() and p.name not in ('data','.data','progress'))
        if not matches:
            print('[resume] No matching prior B-Free output found; starting fresh.',flush=True)
            return None
        source=sorted(matches)[-1][2]
    else:
        source=Path(request)
    state=json.loads((source/'evaluation_state.json').read_text(encoding='utf-8'))
    if state.get('evaluator')!=BFREE_EVALUATOR or state.get('fingerprint')!=fingerprint:
        raise ValueError('B-Free resume checkpoint/cohort/protocol/environment mismatch')
    for path in sorted((source/'progress').rglob('*')):
        if path.is_file() and path.suffix in ('.csv','.json'):
            if path.is_symlink():
                raise ValueError('Symlink in saved B-Free progress')
            target=out/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    print(f'[resume] Restored B-Free progress from {source}',flush=True)
    return source


def bfree_reports(completed,out):
    rows=[]
    for branch,result in completed.items():
        row=dict(branch=branch,complete=True,**eval_metrics(result,.5))
        rows.append(row)
    table=pd.DataFrame(rows)
    if len(table):
        save_csv(out/'metrics.csv',table)
        fields=['accuracy','balanced_accuracy','real_recall','fake_recall','fake_precision','fake_f1','auc','ap']
        macro=dict(completed_branches=len(table),expected_branches=6,all_branches_complete=len(table)==6,
                   **{key:float(table[key].mean()) for key in fields})
        save_csv(out/'macro_metrics.csv',pd.DataFrame([macro]))
        save_csv(out/'predictions.csv',pd.concat(list(completed.values()),ignore_index=True))
    return table


def run_bfree_valid_evaluation(options=None):
    started=time.monotonic(); opts=bfree_options(options)
    deadline=started+opts['time_budget_hours']*3600
    out=Path(opts['output_root']); out.mkdir(parents=True,exist_ok=True)
    root=find_bfree_root(opts)
    source,checkpoint_path,payload=find_bfree_checkpoint(opts,root)
    cfg=validate_uploaded_payload(payload,opts)
    if cfg['seeds']!=[42] or cfg['threshold']!=.5:
        raise ValueError('This B-Free protocol fixes checkpoint seed42 and threshold0.5')
    cfg.update(micro_batch=1,anchor_micro_batch=1,num_workers=0,gradient_checkpointing=False)
    manifest,ids,manifest_identity=prepare_bfree_manifest(root,opts,out)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('Select Kaggle GPU. This notebook is not a TPU notebook.')
    if cfg['precision']=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Checkpoint BF16 unsupported on this GPU')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    identity=dict(evaluator=BFREE_EVALUATOR,checkpoint_sha256=file_sha(checkpoint_path),
        training_signature=payload['signature'],epoch=payload['epoch']+1,
        manifest=manifest_identity,id_order_hash=digest(ids),selected_ids=len(ids),
        branches=list(BFREE_BRANCHES),smoke=opts['smoke_only'],
        batch_protocol='four_id_pairs_interleaved_real_fake_batch8_no_real_score_reuse',
        inference={key:cfg[key] for key in ('input_mode','crop_max_side','test_batch','micro_batch',
            'anchor_micro_batch','precision','basis_method','attention','pooling','k','lambda_max','threshold')},
        processing='native_png_rgb_no_resize_no_jpeg_no_blur_edge_pad14',
        anchor=opts['pretrained_model_path'] or cfg['model_id'],
        environment=dict(torch=str(torch.__version__),transformers=transformers.__version__,
                         pillow=PIL.__version__,gpu=torch.cuda.get_device_name(0)))
    fingerprint=digest(identity)
    bind_upload_metadata(out/'run_fingerprint.json',identity)
    restored=restore_bfree_progress(out,opts,fingerprint)
    atomic_json(out/'evaluation_config.json',opts)
    atomic_json(out/'checkpoint_inspection.json',dict(source=str(source),epoch=payload['epoch']+1,
        signature=payload['signature'],protocol=payload.get('protocol'),training_config=payload['config'],
        inference_only=True,threshold=.5,calibration=False,training=False))
    atomic_json(out/'source_paths.json',dict(bfree=str(root),checkpoint=str(source),
        restored_from=str(restored) if restored else None))
    state=dict(evaluator=BFREE_EVALUATOR,fingerprint=fingerprint,status='running',
        selected_ids=len(ids),n_per_branch=2*len(ids),expected_unique_images=7*len(ids),
        expected_prediction_rows=12*len(ids),planned_branches=list(BFREE_BRANCHES),
        completed_branches=[],saved_rows={},smoke=opts['smoke_only'],training=False,threshold=.5)
    def progress(branch,done):
        state['saved_rows'][branch]=done
        atomic_json(out/'evaluation_state.json',state)
    atomic_json(out/'evaluation_state.json',state)
    print(f'[protocol] {"SMOKE ONLY" if opts["smoke_only"] else "FULL VALID"}: '
          f'{len(ids):,} IDs, SIX branches, {2*len(ids):,} balanced images/branch; threshold0.5; INFERENCE ONLY.',flush=True)
    print('[protocol] Real scores recomputed for each branch; 4 paired IDs per logical batch8. '
          'All six requested; no branch-count limit. No JPEG/resize/blur variants added.',flush=True)
    completed={}; model=anchor=None; reason='complete'
    for branch in BFREE_BRANCHES:
        frame=bfree_branch_frame(manifest,ids,branch)
        result,done=read_bfree_parts(out/'progress'/branch,frame,branch,fingerprint)
        if done==len(frame):
            print(f'[{branch}] reuse complete branch, zero forward.',flush=True)
            complete=True
        else:
            if time.monotonic()>=deadline:
                reason='time_budget'; break
            if model is None:
                if opts['pretrained_model_path']:
                    cfg.update(model_id=opts['pretrained_model_path'],local_files_only=True)
                seed_all(42)
                print('[model] Loading original pretrained anchor and strict augmented detector checkpoint.',flush=True)
                model,anchor=build_models(cfg,device,True)
                model.load_state_dict(payload['model'],strict=True)
                del payload
                model.requires_grad_(False).eval(); anchor.requires_grad_(False).eval()
            result,complete=evaluate_bfree_branch(model,anchor,frame,cfg,opts,branch,out,
                                                   fingerprint,device,deadline,progress)
        progress(branch,len(result))
        if not complete:
            if len(result):
                save_csv(out/'partial_metrics.csv',pd.DataFrame([dict(branch=branch,complete=False,
                    **eval_metrics(result,.5))]))
            reason='time_budget'; break
        save_csv(out/f'predictions_{branch}.csv',result)
        completed[branch]=result; state['completed_branches'].append(branch)
        table=bfree_reports(completed,out)
        print(table[table.branch==branch].to_string(index=False),flush=True)
        torch.cuda.empty_cache(); gc.collect()
    table=bfree_reports(completed,out) if completed else pd.DataFrame(columns=['branch','n','accuracy','auc','ap'])
    finished=len(completed)==len(BFREE_BRANCHES)
    state.update(status='complete' if finished else 'paused',reason='complete' if finished else reason,
        remaining_branches=[branch for branch in BFREE_BRANCHES if branch not in completed],
        elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'evaluation_state.json',state); atomic_json(out/'completion.json',state)
    print(f'[session] {state["status"]}; {len(completed)}/6 branches complete; '
          f'remaining={state["remaining_branches"]}',flush=True)
    if not finished:
        print('[next] Save this output and attach it next run with the same GPU/environment; '
              'resume_from=auto restores complete-batch progress.',flush=True)
    return table
