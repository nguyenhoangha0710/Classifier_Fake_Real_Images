"""Native-resolution, raw-only All7 training. Also appended to the base runtime in the notebook."""
if 'DEFAULT_CONFIG' not in globals():
    from pathlib import Path
    _base_file = Path(__file__).with_name('runtime.py')
    exec(compile(_base_file.read_text(encoding='utf-8'), str(_base_file), 'exec'), globals())

import time

GENS7 = ['sd15', 'adm', 'biggan', 'glide', 'midjourney', 'wukong', 'vqdm']
DEFAULT_CONFIG = dict(DEFAULT_CONFIG, output_root='/kaggle/working/gsd_raw_all7_native_v1',
    source_generator='all7', train_generators=GENS7, expected_generators=GENS7,
    train_batch=8, test_batch=8, micro_batch=1, anchor_micro_batch=1, num_workers=0,
    jpeg_probability=0., blur_probability=0., horizontal_flip_probability=0.,
    input_mode='native_rgb_edge_pad14', basis_method='gram_eigh', attention='sdpa',
    eval_cases=['native'], scan_log_every=500)

_base_scan = scan_dataset

def scan_dataset(cfg, out):
    print('[scan] Reading all seven train/val folders; RGB decoding and hashing. No resize.', flush=True)
    start = time.monotonic()
    frame = _base_scan(cfg, out)
    frame.groupby(['split','generator','label']).size().rename('n').to_csv(out/'dataset_counts.csv')
    print(f'[scan] {len(frame):,} images audited in {time.monotonic()-start:.1f}s', flush=True)
    return frame


def make_splits(df, cfg, seed, out):
    source = df[(df.split=='train') & df.generator.isin(cfg['train_generators'])].copy()
    # Assign each decoded-pixel group ONCE, even when it occurs in several generators.
    unique = source.drop_duplicates('pixel_hash')
    strata = unique.generator + ':' + unique.label.astype(str)
    a, b = train_test_split(unique, test_size=cfg['validation_fraction'],
                          stratify=strata, random_state=seed)
    def select(frame, limit):
        groups = [g.drop_duplicates('pixel_hash') for _,g in frame.groupby('generator',sort=True)]
        if set(frame.generator)!=set(cfg['train_generators']):
            raise ValueError('Every configured train generator must have a nonempty cohort')
        n = min(int(g.label.value_counts().reindex([0,1],fill_value=0).min()) for g in groups)
        if limit is not None:
            n = min(n, limit)
        if n < 1:
            raise ValueError('A generator lacks real/fake images after hash-group splitting')
        return pd.concat([balanced(g, seed, n) for g in groups],ignore_index=True)
    tr = select(source[source.pixel_hash.isin(a.pixel_hash)], cfg['smoke_train_per_class'] if cfg['smoke'] else None)
    va = select(source[source.pixel_hash.isin(b.pixel_hash)], cfg['smoke_eval_per_class'] if cfg['smoke'] else None)
    tests = df[df.split=='test'].copy()
    removed = tests[tests.pixel_hash.isin(source.pixel_hash)]
    removed.to_csv(out/'excluded_test_overlap.csv',index=False)
    tests = tests[~tests.pixel_hash.isin(source.pixel_hash)].drop_duplicates(['generator','pixel_hash'])
    tests = pd.concat([balanced(g,seed,cfg['smoke_eval_per_class'] if cfg['smoke'] else None)
                       for _,g in tests.groupby('generator',sort=True)],ignore_index=True)
    def ordered(f):
        return f.assign(_order=f.sample_id.map(lambda s:digest([seed,s]))).sort_values('_order').drop(columns='_order').reset_index(drop=True)
    tr,va,tests=map(ordered,(tr,va,tests))
    assert not set(tr.pixel_hash)&set(va.pixel_hash)
    assert not (set(tr.pixel_hash)|set(va.pixel_hash))&set(tests.pixel_hash)
    for name,frame in [('train',tr),('validation',va),('test',tests)]:
        frame.to_csv(out/f'{name}_manifest.csv',index=False)
        frame.groupby(['generator','label']).size().rename('n').to_csv(out/f'{name}_counts.csv')
    dump(out/'split_summary.json',dict(train=len(tr),validation=len(va),test=len(tests),
        test_overlap_removed=len(removed),selection='All7 raw; unique pixel groups split before per-generator/class balancing',
        train_hash=digest(tr.sample_id.tolist()),validation_hash=digest(va.sample_id.tolist()),test_hash=digest(tests.sample_id.tolist())))
    print(f'[split] train={len(tr):,}, validation={len(va):,}, test={len(tests):,}; all 7 seen',flush=True)
    return tr,va,tests


def processing_cases(cfg):
    return cfg['eval_cases']


def input_geometry(width, height, cfg):
    """Source-pixel crop coordinates and exact model dimensions, without resampling."""
    cap = cfg['crop_max_side'] if cfg['input_mode']=='center_crop_max_edge_pad14' else None
    cw = min(width, cap) if cap else width
    ch = min(height, cap) if cap else height
    left, top = (width-cw)//2, (height-ch)//2
    pw, ph = ((cw+13)//14)*14, ((ch+13)//14)*14
    return dict(crop_left=left,crop_top=top,crop_width=cw,crop_height=ch,
                input_width=pw,input_height=ph,grid_w=pw//14,grid_h=ph//14,
                tokens=(pw//14)*(ph//14)+1)


def protocol_id(cfg):
    if cfg['input_mode']=='center_crop_max_edge_pad14':
        return f"raw_all7_center_crop{cfg['crop_max_side']}_v1"
    return 'raw_all7_native_v1'


class Images(Dataset):
    def __init__(self,frame,cfg,seed,train=False,epoch=0,case='native'):
        self.rows=frame.to_dict('records'); self.cfg=cfg; self.case=case; self.train=train
        if train and case!='native':
            raise ValueError('Training must use native raw images')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(row['path']) as im:
            image=im.convert('RGB')
        if not self.train:
            if self.case.startswith('fake_jpeg'):
                if int(row['label'])==1:
                    image=jpeg(image,int(self.case[len('fake_jpeg'):]),self.cfg['jpeg_subsampling'])
            elif self.case!='native':
                image=process(image,self.case,self.cfg)
        geom=input_geometry(*image.size,self.cfg)
        image=image.crop((geom['crop_left'],geom['crop_top'],
                          geom['crop_left']+geom['crop_width'],geom['crop_top']+geom['crop_height']))
        x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255.
        # Preserve pixels inside the selected crop. No resize or upscaling of smaller images.
        h,w=x.shape[-2:]
        x=F.pad(x.unsqueeze(0),(0,(-w)%14,0,(-h)%14),mode='replicate').squeeze(0)
        mean=x.new_tensor([.48145466,.4578275,.40821073])[:,None,None]
        std=x.new_tensor([.26862954,.26130258,.27577711])[:,None,None]
        return (x-mean)/std,int(row['label']),i


def native_collate(items):
    x,y,i=zip(*items)
    return list(x),torch.tensor(y,dtype=torch.long),torch.tensor(i,dtype=torch.long)


def loader(frame,cfg,seed,train=False,epoch=0,case='native'):
    return DataLoader(Images(frame,cfg,seed,train,epoch,case),
        batch_size=cfg['train_batch'] if train else cfg['test_batch'], shuffle=train,
        generator=torch.Generator().manual_seed(seed+epoch),num_workers=cfg['num_workers'],
        collate_fn=native_collate,pin_memory=torch.cuda.is_available(),drop_last=False)


def image_chunks(images,limit):
    # Keep membership and order of the mixed logical batch, regardless of shape or label.
    start=0
    while start<len(images):
        end=start+1
        while end<min(start+limit,len(images)) and images[end].shape==images[start].shape:
            end+=1
        yield start,torch.stack(images[start:end])
        start=end


class NativeDetector(Detector):
    def forward(self,x,bases=None):
        h=self.vision.pre_layrnorm(self.vision.embeddings(x,interpolate_pos_encoding=True))
        for l,block in enumerate(self.vision.encoder.layers):
            if str(l) in self.gsd:
                h=self.gsd[str(l)](h,bases[l])
            def apply(t,layer=block):
                return layer(t,attention_mask=None,causal_attention_mask=None)[0]
            h=checkpoint(apply,h,use_reentrant=False) if self.training and self.cfg['gradient_checkpointing'] else apply(h)
        return self.head(self.vision.post_layernorm(h)[:,1:].mean(1))


@torch.no_grad()
def estimate_bases(anchor,images,cfg,device,selected):
    # M.T @ M has the same principal right-singular subspace as uncentered M.
    # Stream ALL tokens: no token sampling and no BN x d concatenation on GPU.
    grams={l:torch.zeros(anchor.config.hidden_size,anchor.config.hidden_size,device=device,dtype=torch.float32) for l in selected}
    tokens=0
    for _,x in image_chunks(images,cfg['anchor_micro_batch']):
        with amp(cfg,device):
            h=anchor.pre_layrnorm(anchor.embeddings(x.to(device),interpolate_pos_encoding=True))
            tokens+=h.shape[0]*(h.shape[1]-1)
            for l,layer in enumerate(anchor.encoder.layers):
                if l in grams:
                    with torch.autocast(device.type,enabled=False):
                        m=h[:,1:].flatten(0,1).float()
                        grams[l].add_(m.T@m)
                h=layer(h,attention_mask=None,causal_attention_mask=None)[0]
    if cfg['k']>min(tokens,anchor.config.hidden_size):
        raise ValueError(f"k={cfg['k']} exceeds token/feature dimensions of this logical batch ({tokens})")
    bases={}
    with torch.autocast(device.type,enabled=False):
        for l,g in grams.items():
            _,v=torch.linalg.eigh((g+g.T)*(0.5/tokens))
            bases[l]=v[:,-cfg['k']:].contiguous().detach()
    return bases


def build_models(cfg,device,gsd):
    pretrained=CLIPVisionModel.from_pretrained(cfg['model_id'],local_files_only=cfg['local_files_only'],attn_implementation=cfg['attention'])
    v=pretrained.vision_model
    if (v.config.hidden_size,v.config.patch_size)!=(1024,14):
        raise ValueError('Native All7 protocol requires CLIP ViT-L/14')
    model=NativeDetector(v,cfg,gsd).to(device)
    anchor=copy.deepcopy(v).requires_grad_(False).eval().to(device) if gsd else None
    if anchor is not None and device.type=='cuda' and cfg['precision']!='fp32':
        anchor.to(dtype=torch.float16 if cfg['precision']=='fp16' else torch.bfloat16)
    return model,anchor


@torch.no_grad()
def predict(model,anchor,frame,cfg,seed,device,case):
    model.eval(); result=np.full(len(frame),np.nan); batches=np.full(len(frame),-1)
    for bid,(images,_,indices) in enumerate(loader(frame,cfg,seed,case=case)):
        bases=estimate_bases(anchor,images,cfg,device,model.selected) if anchor is not None else None
        for start,x in image_chunks(images,cfg['micro_batch']):
            with amp(cfg,device):
                logits=model(x.to(device),bases)
            ix=indices[start:start+len(x)].numpy()
            result[ix]=logits.float().softmax(-1)[:,1].cpu().numpy(); batches[ix]=bid
        if bid%25==0:
            print(f'[{case}] batch {bid+1}/{math.ceil(len(frame)/cfg["test_batch"])}',flush=True)
    if not np.isfinite(result).all():
        raise ValueError('Missing/nonfinite prediction')
    return result,batches


def save_checkpoint(path,state):
    temp=path.with_suffix('.tmp')
    torch.save(state,temp)
    temp.replace(path)


def train_model(model,anchor,train,val,cfg,seed,device,folder,signature):
    groups=[dict(params=model.vision.parameters(),lr=cfg['lr']),dict(params=model.head.parameters(),lr=cfg['head_lr'])]
    if len(model.gsd):
        groups.append(dict(params=model.gsd.parameters(),lr=cfg['modulation_lr']))
    optimizer=torch.optim.AdamW(groups,betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['precision']=='fp16')
    stopper=EarlyStopping(cfg['early_stopping'],cfg['patience'],cfg['min_delta'])
    latest=folder/'last.pt'; start=0; best=float('inf'); history=[]
    if cfg['resume'] and latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['signature']!=signature:
            raise ValueError('Resume config/manifests changed; use a new output_root')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        stopper.state.update(ck['early_stopping']); start=ck['epoch']+1; best=ck['best']; history=ck['history']
        del ck
        print(f'[resume] next epoch {start+1}, stopped={stopper.state["stopped"]}',flush=True)
    for epoch in range(start,cfg['epochs']):
        if stopper.state['stopped']:
            break
        model.train(); running=0.; count=0; skipped=0; tick=time.monotonic()
        for bid,(images,labels,_) in enumerate(loader(train,cfg,seed,True,epoch)):
            bases=estimate_bases(anchor,images,cfg,device,model.selected) if anchor is not None else None
            optimizer.zero_grad(set_to_none=True)
            for a,x in image_chunks(images,cfg['micro_batch']):
                with amp(cfg,device):
                    logits=model(x.to(device),bases)
                    loss=F.cross_entropy(logits.float(),labels[a:a+len(x)].to(device),reduction='sum')/len(images)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward(); running+=float(loss.detach())*len(images)
            old=scaler.get_scale(); scaler.step(optimizer); scaler.update(); skipped+=int(scaler.get_scale()<old); count+=len(images)
            if bid%10==0:
                print(f'[epoch {epoch+1}] batch {bid+1}/{math.ceil(len(train)/cfg["train_batch"])}; loss={running/count:.5f}',flush=True)
        p,_=predict(model,anchor,val,cfg,seed,device,'native')
        vloss=float(F.binary_cross_entropy(torch.from_numpy(p).double().clamp(1e-7,1-1e-7),torch.tensor(val.label.to_numpy(),dtype=torch.float64)))
        stop=stopper.update(vloss)
        entry=dict(epoch=epoch+1,train_loss=running/count,val_loss=vloss,seconds=time.monotonic()-tick,
            amp_skipped_steps=skipped,train_images=count,**metric(val.label.to_numpy(),p,cfg['threshold']),
            modulation={l:dict(a=float(g.a.detach()),b=float(g.b.detach())) for l,g in model.gsd.items()},early_stopping=dict(stopper.state))
        history.append(entry); print(entry,flush=True)
        if vloss<best:
            best=vloss; save_checkpoint(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch,signature=signature,config=cfg))
        save_checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            epoch=epoch,best=best,history=history,signature=signature,config=cfg,early_stopping=dict(stopper.state)))
        dump(folder/'history.json',history)
        dump(folder/'training_status.json',dict(epoch=epoch+1,max_epochs=cfg['epochs'],early_stopping=dict(stopper.state)))
        if stop:
            print('[early stopping] Restore best checkpoint for evaluation.',flush=True)
            break
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model'])
    dump(folder/'selected_checkpoint.json',dict(epoch=ck['epoch']+1,criterion='minimum All7 raw validation cross entropy',signature=signature))


def summarize(predictions,cfg,out):
    rows=[]
    for (seed,model,case),f in predictions.groupby(['seed','model','case']):
        for gen,g in [('pooled',f),*f.groupby('generator')]:
            rows.append(dict(seed=seed,model=model,case=case,generator=gen,
                seen=gen in cfg['train_generators'],**metric(g.label.to_numpy(),g.prob_fake.to_numpy(),cfg['threshold'])))
    result=pd.DataFrame(rows); result.to_csv(out/'metrics.csv',index=False)
    result[result.generator!='pooled'].groupby(['seed','model','case'])[['auc','balanced_accuracy','fake_recall','real_recall','accuracy']].mean().to_csv(out/'macro_all.csv')
    return result


def validate_raw_config(cfg):
    if set(cfg['train_generators'])!=set(GENS7) or set(cfg['expected_generators'])!=set(GENS7):
        raise ValueError('This protocol requires exactly all seven Tiny generators')
    if any(cfg[k]!=0 for k in ('jpeg_probability','blur_probability','horizontal_flip_probability')):
        raise ValueError('Raw-only training forbids JPEG/blur/flip augmentation')
    if cfg['input_mode'] not in ('native_rgb_edge_pad14','center_crop_max_edge_pad14') or cfg['basis_method']!='gram_eigh':
        raise ValueError('Unsupported native-input protocol')
    if cfg['input_mode']=='center_crop_max_edge_pad14':
        cap=cfg.get('crop_max_side')
        if isinstance(cap,bool) or not isinstance(cap,int) or cap<14 or cap%14:
            raise ValueError('crop_max_side must be a positive integer multiple of 14')
    if cfg['attention']!='sdpa' or cfg['pooling']!='patch_mean_post_ln':
        raise ValueError('Native protocol requires SDPA and patch_mean_post_ln')
    if min(cfg[k] for k in ('train_batch','test_batch','micro_batch','anchor_micro_batch','epochs','k'))<1:
        raise ValueError('Positive batches, epochs and k required')
    if cfg['precision'] not in ('fp16','bf16','fp32'):
        raise ValueError('precision must be fp16, bf16 or fp32')
    if not cfg['eval_cases'] or 'native' not in cfg['eval_cases']:
        raise ValueError('Evaluation must include native')
    for case in cfg['eval_cases']:
        if case not in ('native','size') and not any(case==prefix+str(q) for prefix in ('jpeg','fake_jpeg','size_jpeg') for q in range(1,101)):
            raise ValueError(f'Unknown evaluation case: {case}')


def run(config=None):
    cfg={**DEFAULT_CONFIG,**(config or {})}; validate_raw_config(cfg)
    if cfg['smoke']:
        cfg.update(epochs=1,train_batch=2,test_batch=2,k=4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('Use Kaggle GPU; CPU unit tests cover native input separately')
    if cfg['precision']=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16; use fp16 on T4')
    out=Path(cfg['output_root']); out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        previous=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if previous!=cfg:
            raise ValueError('Existing output config differs; choose a new output_root before running')
    dump(out/'config.json',cfg)
    dump(out/'environment.json',dict(torch=torch.__version__,pillow=PIL.__version__,python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),use_devices=1,protocol=protocol_id(cfg)))
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    df=scan_dataset(cfg,out)
    census=df[['sample_id','split','generator','label','width','height']].copy()
    geometry=pd.DataFrame([input_geometry(int(w),int(h),cfg) for w,h in zip(census.width,census.height)])
    census=pd.concat([census.reset_index(drop=True),geometry],axis=1)
    census.to_csv(out/'token_census.csv',index=False)
    print(f'[{protocol_id(cfg)}] tokens min/median/max: {census.tokens.min()}/{census.tokens.median()}/{census.tokens.max()}. No pixel resampling.',flush=True)
    results=[]
    for seed in cfg['seeds']:
        split_out=out/f'seed{seed}'; split_out.mkdir(exist_ok=True)
        train,val,tests=make_splits(df,cfg,seed,split_out)
        signature=digest(dict(config=cfg,train=train.file_hash.tolist(),val=val.file_hash.tolist(),test=tests.file_hash.tolist(),protocol=protocol_id(cfg)))
        for name in cfg['models']:
            if name not in ('clip_ft','clip_gsd'):
                raise ValueError(name)
            seed_all(seed); folder=split_out/name; folder.mkdir(exist_ok=True)
            print(f'[model] Loading {name}; native position interpolation in BOTH branches.',flush=True)
            model,anchor=build_models(cfg,device,name=='clip_gsd')
            dump(folder/'architecture.json',dict(backbone=str(model.vision.config),input_mode=cfg['input_mode'],
                basis=cfg['basis_method'],attention=cfg['attention'],all_generators_seen=True,patch_size=14,
                protocol=protocol_id(cfg),crop_max_side=cfg.get('crop_max_side'),
                padding='right/bottom edge replication to multiple of 14; no resize',
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            try:
                train_model(model,anchor,train,val,cfg,seed,device,folder,signature)
                for case in processing_cases(cfg):
                    p,batches=predict(model,anchor,tests,cfg,seed,device,case)
                    f=tests[['sample_id','generator','label','pixel_hash','width','height']].assign(seed=seed,model=name,case=case,prob_fake=p,batch_id=batches)
                    f.to_csv(folder/f'predictions_{case}.csv',index=False); results.append(f)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError('CUDA OOM under the configured input protocol. No automatic resize, skip or batch reduction. Use more VRAM or explicitly start a smaller-input/batch protocol; last.pt retains completed epochs.') from exc
            del model,anchor; gc.collect(); torch.cuda.empty_cache()
    predictions=pd.concat(results,ignore_index=True); predictions.to_csv(out/'predictions.csv',index=False)
    metrics=summarize(predictions,cfg,out)
    dump(out/'completion.json',dict(status='complete',smoke=cfg['smoke'],protocol=protocol_id(cfg),config_hash=digest(cfg)))
    print(metrics[metrics.generator=='pooled'].to_string(index=False),flush=True)
    return metrics
