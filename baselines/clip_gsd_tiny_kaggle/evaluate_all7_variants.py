"""Inference-only evaluation for All7 crop560 checkpoints; resumable complete logical batches."""
if 'NativeDetector' not in globals():
    from pathlib import Path
    _here=Path(__file__).resolve().parent
    exec(compile((_here/'runtime.py').read_text(encoding='utf-8'),str(_here/'runtime.py'),'exec'),globals())
    exec(compile((_here/'runtime_raw_all7.py').read_text(encoding='utf-8'),str(_here/'runtime_raw_all7.py'),'exec'),globals())

from torch.utils.data import Subset
import transformers

EVAL_DEFAULTS=dict(
    input_root='/kaggle/input', checkpoint_path=None, dataset_root=None,
    output_root='/kaggle/working/gsd_all7_crop560_variants', pretrained_model_path=None,
    expected_training_signature='18ea6abe15bbb769490788a8ec5516457b7538ebb3dad1018d4a94948a2a79cc',
    cases=['native','png_roundtrip','fake_jpeg96','all_jpeg96','all_jpeg90','all_jpeg80',
           'resize_square256','resize_square560','blur_sigma1','resize256_jpeg96'],
    micro_batch=1,anchor_micro_batch=1,num_workers=0,save_every_batches=25,log_every_batches=10,
    max_batches=None, native_tolerance=1e-3,verify_file_sizes=True)


def case_spec(name):
    fixed={'native':dict(kind='native'), 'png_roundtrip':dict(kind='png'),
           'blur_sigma1':dict(kind='blur',sigma=1.),'blur_sigma2':dict(kind='blur',sigma=2.),
           'resize256_jpeg96':dict(kind='resize_jpeg',side=256,quality=96),
           'resize_long560':dict(kind='resize_long',side=560)}
    if name in fixed:
        return fixed[name]
    for prefix,kind in [('fake_jpeg','fake_jpeg'),('all_jpeg','all_jpeg'),('resize_square','resize_square')]:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            value=int(name[len(prefix):])
            if kind in ('fake_jpeg','all_jpeg') and 1<=value<=100:
                return dict(kind=kind,quality=value)
            if kind=='resize_square' and value in (224,256,336,512,560):
                return dict(kind=kind,side=value)
    raise ValueError(f'Unsupported case: {name}')


def transform_variant(image,label,spec,cfg):
    kind=spec['kind']
    if kind=='png':
        buf=io.BytesIO(); image.save(buf,format='PNG'); buf.seek(0)
        with Image.open(buf) as decoded:
            image=decoded.convert('RGB').copy()
    elif kind in ('all_jpeg','fake_jpeg'):
        if kind=='all_jpeg' or label==1:
            image=jpeg(image,spec['quality'],cfg['jpeg_subsampling'])
    elif kind=='blur':
        image=image.filter(ImageFilter.GaussianBlur(spec['sigma']))
    elif kind in ('resize_square','resize_jpeg'):
        image=image.resize((spec['side'],spec['side']),Image.Resampling.BICUBIC)
        if kind=='resize_jpeg':
            image=jpeg(image,spec['quality'],cfg['jpeg_subsampling'])
    elif kind=='resize_long':
        w,h=image.size
        ratio=min(1.,spec['side']/max(w,h))
        image=image.resize((max(1,round(w*ratio)),max(1,round(h*ratio))),Image.Resampling.BICUBIC)
    elif kind!='native':
        raise ValueError(kind)
    return image


class VariantImages(Dataset):
    def __init__(self,frame,cfg,spec):
        self.rows=frame.to_dict('records'); self.cfg=cfg; self.spec=spec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['path']) as im:
            original=im.convert('RGB')
        if original.size!=(int(row['width']),int(row['height'])):
            raise ValueError(f"Image dimensions changed: {row['relative_path']}")
        image=transform_variant(original,int(row['label']),self.spec,self.cfg)
        geom=input_geometry(*image.size,self.cfg)
        image=image.crop((geom['crop_left'],geom['crop_top'],geom['crop_left']+geom['crop_width'],geom['crop_top']+geom['crop_height']))
        x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255.
        h,w=x.shape[-2:]; x=F.pad(x[None],(0,(-w)%14,0,(-h)%14),mode='replicate')[0]
        x=(x-x.new_tensor([.48145466,.4578275,.40821073])[:,None,None])/x.new_tensor([.26862954,.26130258,.27577711])[:,None,None]
        return x,int(row['label']),index


def file_sha(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_checkpoint(options):
    if options['checkpoint_path']:
        path=Path(options['checkpoint_path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates=[]
    for p in Path(options['input_root']).rglob('best.pt'):
        root=p.parents[2]
        if (root/'seed42/test_manifest.csv').is_file() and (root/'config.json').is_file():
            config=json.loads((root/'config.json').read_text(encoding='utf-8'))
            if config.get('input_mode')=='center_crop_max_edge_pad14' and config.get('crop_max_side')==560:
                candidates.append(p)
    if len(candidates)!=1:
        raise ValueError(f'Set checkpoint_path explicitly. Crop560 best.pt candidates: {[str(p) for p in candidates]}')
    return candidates[0]


def resolve_test_files(frame,options):
    required={'sample_id','relative_path','label','generator','width','height','file_hash','pixel_hash','file_bytes'}
    if not required<=set(frame.columns) or frame.sample_id.duplicated().any():
        raise ValueError('Invalid saved test manifest')
    for relative in frame.relative_path:
        if Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
            raise ValueError('Unsafe relative image path in manifest')
    first=frame.relative_path.iloc[0]; root=Path(options['input_root'])
    if options['dataset_root']:
        candidates=[Path(options['dataset_root'])]
    else:
        candidates=[root/'tiny-genimage',root/'datasets/yangsangtai/tiny-genimage']
        generator=Path(first).parts[0]
        for depth in range(5):
            candidates.extend(p.parent for p in root.glob('/'.join(['*']*depth+[generator])) if p.is_dir())
    matches=sorted({p.resolve() for p in candidates if (p/first).is_file()})
    if len(matches)!=1:
        raise ValueError(f'Set dataset_root explicitly. Candidates: {[str(p) for p in matches]}')
    base=matches[0]; resolved=frame.copy(); resolved['path']=[str(base/p) for p in frame.relative_path]
    for row in resolved.itertuples():
        p=Path(row.path)
        if not p.is_file() or (options['verify_file_sizes'] and p.stat().st_size!=int(row.file_bytes)):
            raise ValueError(f'Missing or changed input file: {p}')
    print(f'[manifest] {len(resolved):,} files validated by path/size, no RGB/hash scan: {base}',flush=True)
    return resolved,base


def save_csv(path,frame):
    path=Path(path); temp=path.with_suffix('.tmp')
    frame.to_csv(temp,index=False); temp.replace(path)


def bind_metadata(path,metadata):
    metadata=json.loads(json.dumps(metadata,default=str))
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous!=metadata:
            raise ValueError(f'Resume fingerprint changed: {path}. Use a new output directory.')
    else:
        dump(path,metadata)


def load_case_parts(folder,frame,batch_size):
    parts=[pd.read_csv(p) for p in sorted(folder.glob('part_*.csv'))]
    if not parts:
        return pd.DataFrame(),0
    prior=pd.concat(parts,ignore_index=True)
    n=len(prior)
    if n>len(frame) or (n<len(frame) and n%batch_size):
        raise ValueError('Resume must begin after a complete logical batch')
    expected=frame.iloc[:n]
    if prior.sample_id.tolist()!=expected.sample_id.tolist() or prior.label.tolist()!=expected.label.tolist() or prior.generator.tolist()!=expected.generator.tolist():
        raise ValueError('Saved progress does not match the ordered manifest')
    if prior.batch_id.tolist()!=[i//batch_size for i in range(n)] or not np.isfinite(prior.prob_fake).all():
        raise ValueError('Corrupt probability/batch records in progress')
    return prior,n


@torch.no_grad()
def evaluate_case(model,anchor,frame,cfg,options,name,out,fingerprint,device):
    spec=case_spec(name); folder=out/'progress'/name; folder.mkdir(parents=True,exist_ok=True)
    bind_metadata(folder/'metadata.json',dict(run=fingerprint,case=spec))
    prior,start=load_case_parts(folder,frame,cfg['test_batch'])
    if start==len(frame):
        print(f'[{name}] using completed progress',flush=True)
        return prior
    model.eval(); dataset=VariantImages(frame,cfg,spec)
    data=DataLoader(Subset(dataset,range(start,len(frame))),batch_size=cfg['test_batch'],shuffle=False,
        num_workers=options['num_workers'],collate_fn=native_collate,pin_memory=device.type=='cuda')
    pending=[]; parts=[prior] if len(prior) else []; batch0=start//cfg['test_batch']; tick=time.monotonic()
    try:
        for local_bid,(images,labels,indices) in enumerate(data):
            bid=batch0+local_bid
            bases=estimate_bases(anchor,images,cfg,device,model.selected)
            probabilities=np.empty(len(images))
            for offset,x in image_chunks(images,cfg['micro_batch']):
                with amp(cfg,device):
                    logits=model(x.to(device),bases)
                probabilities[offset:offset+len(x)]=logits.float().softmax(-1)[:,1].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError(f'Nonfinite predictions: {name}, batch {bid}')
            item=frame.iloc[indices.tolist()][['sample_id','relative_path','generator','label','pixel_hash','width','height']].copy()
            item=item.assign(case=name,model='clip_gsd',seed=cfg['seeds'][0],prob_fake=probabilities,batch_id=bid,
                input_tokens=[(x.shape[-2]//14)*(x.shape[-1]//14)+1 for x in images])
            pending.append(item)
            done=int(indices[-1])+1
            if local_bid==0 or (bid+1)%options['log_every_batches']==0:
                print(f'[{name}] batch {bid+1}/{math.ceil(len(frame)/cfg["test_batch"])}; {done:,} images',flush=True)
            if len(pending)>=options['save_every_batches'] or done==len(frame):
                chunk=pd.concat(pending,ignore_index=True)
                save_csv(folder/f'part_{int(indices[0])-sum(len(p) for p in pending[:-1]):07d}.csv',chunk)
                parts.append(chunk); pending=[]
                print(f'[{name}] {done:,}/{len(frame):,} images; batch {bid+1}; {time.monotonic()-tick:.1f}s this invocation',flush=True)
                dump(folder/'status.json',dict(completed_images=done,total=len(frame),last_batch=bid))
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(f'CUDA OOM in {name}; saved progress can be resumed, unsaved batches will rerun. No automatic crop/batch/resize changes.') from exc
    result=pd.concat(parts,ignore_index=True)
    if result.sample_id.tolist()!=frame.sample_id.tolist():
        raise ValueError('Incomplete/incorrect case order')
    save_csv(out/f'predictions_{name}.csv',result)
    return result


def eval_metrics(group,threshold=.5):
    y=group.label.to_numpy(); p=group.prob_fake.to_numpy(); z=p>=threshold
    tn=int(((y==0)&~z).sum()); fp=int(((y==0)&z).sum()); fn=int(((y==1)&~z).sum()); tp=int(((y==1)&z).sum())
    real=tn/(tn+fp) if tn+fp else float('nan'); fake=tp/(tp+fn) if tp+fn else float('nan')
    both=len(np.unique(y))==2
    return dict(n=len(y),accuracy=float((y==z).mean()),balanced_accuracy=(real+fake)/2,
        real_recall=real,fake_recall=fake,fake_precision=tp/(tp+fp) if tp+fp else 0.,
        fake_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        auc=float(roc_auc_score(y,p)) if both else float('nan'),
        ap=float(average_precision_score(y,p)) if both else float('nan'),tn=tn,fp=fp,fn=fn,tp=tp)


def write_eval_reports(all_predictions,cfg,out):
    rows=[]
    for case,f in all_predictions.groupby('case',sort=False):
        for gen,g in [('pooled',f),*f.groupby('generator')]:
            rows.append(dict(case=case,generator=gen,**eval_metrics(g,cfg['threshold'])))
    table=pd.DataFrame(rows); save_csv(out/'metrics.csv',table)
    per=table[table.generator!='pooled']
    per.groupby('case')[['accuracy','balanced_accuracy','real_recall','fake_recall','auc','ap']].mean().to_csv(out/'macro_all.csv')
    base=all_predictions[all_predictions.case=='native'].set_index('sample_id'); pairs=[]
    for name,f in all_predictions[all_predictions.case!='native'].groupby('case'):
        changed=f.set_index('sample_id').loc[base.index]
        if not np.array_equal(base.batch_id,changed.batch_id):
            raise ValueError('Batch membership changed across variants')
        pair=base[['generator','label','batch_id']].copy()
        pair['native_prob']=base.prob_fake; pair['changed_prob']=changed.prob_fake
        pair['delta']=pair.changed_prob-pair.native_prob
        pair['native_correct']=(pair.native_prob>=cfg['threshold']).astype(int)==pair.label
        pair['changed_correct']=(pair.changed_prob>=cfg['threshold']).astype(int)==pair.label
        save_csv(out/f'paired_{name}.csv',pair.reset_index())
        for (gen,label),g in pair.groupby(['generator','label']):
            pairs.append(dict(case=name,generator=gen,label=label,n=len(g),mean_delta=g.delta.mean(),
                mean_abs_delta=g.delta.abs().mean(),flip_rate=(g.native_correct!=g.changed_correct).mean(),
                correct_to_wrong=int((g.native_correct&~g.changed_correct).sum()),
                wrong_to_correct=int((~g.native_correct&g.changed_correct).sum())))
    save_csv(out/'paired_summary.csv',pd.DataFrame(pairs))
    delta=table.merge(table[table.case=='native'].drop(columns='case'),on='generator',suffixes=('','_native'))
    for field in ('accuracy','balanced_accuracy','real_recall','fake_recall','auc','ap'):
        delta['delta_'+field]=delta[field]-delta[field+'_native']
    save_csv(out/'delta_from_native.csv',delta)
    return table


def compare_native_reference(native,reference_path,options,out):
    if not reference_path.is_file():
        result=dict(status='unavailable',reason='No historical native predictions')
    else:
        old=pd.read_csv(reference_path)
        if 'case' in old:
            old=old[old.case=='native']
        if 'model' in old:
            old=old[old.model=='clip_gsd']
        if 'seed' in old:
            old=old[old.seed==int(native.seed.iloc[0])]
        if old.sample_id.duplicated().any() or not set(native.sample_id)<=set(old.sample_id):
            raise ValueError('Historical native reference cohort mismatch')
        old=old.set_index('sample_id').loc[native.sample_id]
        if not np.array_equal(old.label.to_numpy(),native.label.to_numpy()) or not np.array_equal(old.batch_id.to_numpy(),native.batch_id.to_numpy()):
            raise ValueError('Historical native labels/batch membership mismatch')
        delta=np.abs(native.prob_fake.to_numpy()-old.prob_fake.to_numpy())
        flips=int(((native.prob_fake.to_numpy()>=.5)!=(old.prob_fake.to_numpy()>=.5)).sum())
        result=dict(status='match' if delta.max()<=options['native_tolerance'] and flips==0 else 'review_required',
            n=len(native),max_abs_probability_delta=float(delta.max()),mean_abs_probability_delta=float(delta.mean()),prediction_flips=flips,
            note='GPU eigensolver/precision can differ slightly; investigate material native mismatch before interpreting variants')
    dump(out/'native_reference_check.json',result); print('[native reference]',result,flush=True)
    return result


def run_evaluation(options=None):
    opts={**EVAL_DEFAULTS,**(options or {})}
    if not opts['cases'] or opts['cases'][0]!='native' or len(set(opts['cases']))!=len(opts['cases']):
        raise ValueError('Cases must be unique and start with native')
    for name in opts['cases']:
        case_spec(name)
    if min(opts[k] for k in ('micro_batch','anchor_micro_batch','save_every_batches','log_every_batches'))<1 or opts['num_workers']<0:
        raise ValueError('Positive compute/checkpoint intervals required')
    if opts['max_batches'] is not None and (not isinstance(opts['max_batches'],int) or opts['max_batches']<1):
        raise ValueError('max_batches must be None or a positive integer (smoke only)')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('Full pretrained evaluation requires GPU; unit tests use tiny CPU models')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    checkpoint_path=find_checkpoint(opts); saved_root=checkpoint_path.parents[2]
    print(f'[checkpoint] {checkpoint_path}; loading CPU weights; NEVER training',flush=True)
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    cfg=payload.get('config')
    if not cfg:
        raise ValueError('Checkpoint must include training config')
    validate_raw_config(cfg)
    if cfg['input_mode']!='center_crop_max_edge_pad14' or cfg.get('crop_max_side')!=560:
        raise ValueError('Expected All7 center-crop560 checkpoint')
    if opts['expected_training_signature'] and payload.get('signature')!=opts['expected_training_signature']:
        raise ValueError('Wrong training checkpoint signature')
    selected=json.loads((checkpoint_path.parent/'selected_checkpoint.json').read_text(encoding='utf-8'))
    if selected['epoch']!=payload['epoch']+1 or selected['signature']!=payload['signature']:
        raise ValueError('Checkpoint does not match selected epoch/signature')
    manifest_path=saved_root/f"seed{cfg['seeds'][0]}/test_manifest.csv"
    frame=pd.read_csv(manifest_path)
    if len(frame)!=7000 or set(frame.generator)!=set(GENS7) or set(frame.label)!={0,1}:
        raise ValueError('Expected complete original 7,000-image All7 manifest')
    split=json.loads((manifest_path.parent/'split_summary.json').read_text(encoding='utf-8'))
    if digest(frame.sample_id.tolist())!=split['test_hash']:
        raise ValueError('Manifest ID order/hash differs from saved training split')
    if len(cfg['seeds'])!=1:
        raise ValueError('Select a single-seed checkpoint for this evaluator')
    if opts['max_batches'] is not None:
        frame=frame.iloc[:opts['max_batches']*cfg['test_batch']].copy()
    frame,base=resolve_test_files(frame,opts)
    cfg={**cfg,'micro_batch':opts['micro_batch'],'anchor_micro_batch':opts['anchor_micro_batch']}
    if cfg['precision']=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Checkpoint inference precision bf16 unsupported on this GPU')
    out=Path(opts['output_root']); out.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=file_sha(checkpoint_path)
    inference_cfg={k:v for k,v in cfg.items() if k not in ('dataset_root','input_root','output_root','resume','scan_log_every','eval_cases','model_id')}
    provenance=dict(checkpoint_sha256=checkpoint_sha,training_signature=payload['signature'],selected_epoch=payload['epoch']+1,
        manifest_sha256=file_sha(manifest_path),sample_order_hash=digest(frame.sample_id.tolist()),n=len(frame),
        inference_config=inference_cfg,transform_version='original_before_crop560_v1',max_batches=opts['max_batches'],
        anchor_model=opts['pretrained_model_path'] or cfg['model_id'],
        environment=dict(torch=str(torch.__version__),transformers=transformers.__version__,
            pillow=PIL.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0)))
    fingerprint=digest(provenance); bind_metadata(out/'provenance.json',provenance)
    dump(out/'evaluation_config.json',opts)
    dump(out/'source_paths.json',dict(checkpoint=str(checkpoint_path),dataset_root=str(base),manifest=str(manifest_path)))
    print(f"[model] recreate pretrained frozen anchor; batch={cfg['test_batch']}, native crop max={cfg['crop_max_side']}",flush=True)
    if opts['pretrained_model_path']:
        cfg.update(model_id=opts['pretrained_model_path'],local_files_only=True)
    seed_all(cfg['seeds'][0]); model,anchor=build_models(cfg,device,True)
    model.load_state_dict(payload['model'],strict=True); del payload
    model.requires_grad_(False).eval(); anchor.requires_grad_(False).eval()
    predictions=[]; ref=None
    for name in opts['cases']:
        result=evaluate_case(model,anchor,frame,cfg,opts,name,out,fingerprint,device)
        save_csv(out/f'predictions_{name}.csv',result)
        predictions.append(result)
        write_eval_reports(pd.concat(predictions,ignore_index=True),cfg,out)
        if name=='native':
            reference=checkpoint_path.parent/'predictions_native.csv'
            if not reference.is_file():
                reference=saved_root/'predictions.csv'
            ref=compare_native_reference(result,reference,opts,out)
        gc.collect(); torch.cuda.empty_cache()
    all_predictions=pd.concat(predictions,ignore_index=True); save_csv(out/'predictions.csv',all_predictions)
    table=write_eval_reports(all_predictions,cfg,out)
    dump(out/'completion.json',dict(status='complete',training=False,smoke=opts['max_batches'] is not None,
        evaluated_cases=opts['cases'],n_per_case=len(frame),fingerprint=fingerprint,native_reference=ref))
    print(table[table.generator=='pooled'].to_string(index=False),flush=True)
    return table
