"""Render a ZIP of NPY patch logits or an NPZ, without source images or CSV.

python baselines/capd/heatmap/render_bfree_archive.py
"""
import argparse
import csv
import html
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'smaps_00004.zip')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--skip-existing',action='store_true',help='Reuse individual PNGs already rendered for this archive')
    args=parser.parse_args()
    archive_name=args.input.stem
    part=archive_name.removeprefix('smaps_') if hasattr(str,'removeprefix') else archive_name.replace('smaps_','',1)
    if args.output is None:args.output=HERE/f'rendered_bfree_part{part}'
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    individual=out/'individual';individual.mkdir(exist_ok=True)
    arrays=[]
    with zipfile.ZipFile(args.input) as archive:
        for member in sorted(archive.infolist(),key=lambda x:x.filename):
            if member.is_dir() or not member.filename.lower().endswith('.npy'):continue
            # Read in memory; never extract untrusted archive paths.
            a=np.load(io.BytesIO(archive.read(member)),allow_pickle=False).astype(np.float64)
            if a.ndim!=2 or not a.size or not np.isfinite(a).all():raise ValueError(member.filename)
            name=Path(member.filename).stem
            if any(x['id']==name for x in arrays):raise ValueError('Duplicate sample ID')
            if not all(c.isalnum() or c in '_-.' for c in name):raise ValueError('Unsafe sample ID')
            arrays.append({'id':name,'raw':a,'sigmoid':1/(1+np.exp(-np.clip(a,-60,60)))})
    if not arrays:raise ValueError('Archive contains no .npy arrays')
    vmin=min(x['raw'].min() for x in arrays);vmax=max(x['raw'].max() for x in arrays)
    if vmin==vmax:vmax=vmin+1
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),18) if font_path.is_file() else ImageFont.load_default()
    small=ImageFont.truetype(str(font_path),15) if font_path.is_file() else ImageFont.load_default()
    cmap=cm.get_cmap('inferno')
    nearest=Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST

    def color_array(a,lo,hi):
        return Image.fromarray(np.uint8(cmap(np.clip((a-lo)/(hi-lo),0,1))*255)[:,:,:3])

    def legend(draw,canvas,left,top,lo,hi):
        gradient=np.linspace(lo,hi,380)[None,:]
        canvas.paste(color_array(gradient,lo,hi).resize((380,16),nearest),(left,top))
        for x,t in [(left,f'{lo:.3f}'),(left+335,f'{hi:.3f}')]:draw.text((x,top+20),t,font=small,fill='#263445')

    for x in arrays:
        if args.skip_existing and (individual/f"{x['id']}.png").is_file():continue
        canvas=Image.new('RGB',(940,600),'white');draw=ImageDraw.Draw(canvas)
        draw.text((22,10),x['id'],font=font,fill='#263445')
        draw.text((22,37),f"Patch grid: {x['raw'].shape} | min={x['raw'].min():.3f}, max={x['raw'].max():.3f}",font=small,fill='#263445')
        for left,key,lo,hi,label in [(25,'raw',vmin,vmax,'Raw patch logit (shared archive scale)'),(495,'sigmoid',0,1,'Sigmoid(patch logit), fixed 0-1 scale')]:
            draw.text((left,70),label,font=small,fill='#263445')
            im=color_array(x[key],lo,hi)
            ratio=min(420/im.width,420/im.height)
            im=im.resize((int(im.width*ratio),int(im.height*ratio)),nearest)
            canvas.paste(im,(left+(420-im.width)//2,98+(420-im.height)//2))
            legend(draw,canvas,left,535,lo,hi)
        canvas.save(individual/f"{x['id']}.png")

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    ordered=sorted(arrays,key=lambda x:x['raw'].max())
    selected=[ordered[i] for i in np.unique(np.linspace(0,len(ordered)-1,min(6,len(ordered)),dtype=int))]
    def mosaic(items,path,mode,title,cols=3):
        rows=int(np.ceil(len(items)/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(cols*3.4,rows*3.4),squeeze=False,layout='constrained')
        for ax in axes.flat:ax.set_visible(False)
        for ax,x in zip(axes.flat,items):
            ax.set_visible(True)
            lo,hi=(vmin,vmax) if mode=='raw' else (0,1)
            im=ax.imshow(x[mode],cmap='inferno',vmin=lo,vmax=hi,interpolation='nearest')
            name=x['id'].replace('bfree__','').replace('__',' · ')
            ax.set_title(f"{name}\nlogit min={x['raw'].min():.2f}, max={x['raw'].max():.2f}",fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=list(axes.flat),shrink=.7,label='Logit patch gốc' if mode=='raw' else 'Sigmoid(logit patch)')
        fig.suptitle(title,fontsize=14,fontweight='bold',y=1.04)
        fig.supxlabel('Cùng thang màu cho mọi mẫu · Sáng: điểm patch cao hơn\nKhông phải xác suất định vị đã hiệu chỉnh; chưa chồng lên ảnh gốc.',fontsize=9)
        fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
    mosaic(selected,out/'preview_logits.png','raw',f'B-Free part {part}: bản đồ logit patch')
    mosaic(selected,out/'preview_sigmoid.png','sigmoid',f'B-Free part {part}: sigmoid của điểm patch')
    for start in range(0,len(arrays),24):
        mosaic(arrays[start:start+24],out/f'contact_sheet_{start//24+1:02d}.png','raw',f'B-Free part {part} · mẫu {start+1}–{min(start+24,len(arrays))}',cols=4)
    rows=[dict(id=x['id'],grid_h=x['raw'].shape[0],grid_w=x['raw'].shape[1],
               logit_min=float(x['raw'].min()),logit_max=float(x['raw'].max()),logit_mean=float(x['raw'].mean())) for x in arrays]
    with (out/'index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for x in arrays:
        group=x['id'].rsplit('__',1)[0];groups[group]=groups.get(group,0)+1
    (out/'summary.json').write_text(json.dumps(dict(input=str(args.input.resolve()),n=len(arrays),groups=groups,
        logit_range=[float(vmin),float(vmax)],selection=[x['id'] for x in selected],
        notes='Label inferred only from stored sample ID. No CSV: no image probability, prediction or correctness assigned. This part is not the whole evaluation.'),ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for x in arrays:
        sid=html.escape(x['id'])
        cards.append(f'<figure><a href="individual/{sid}.png"><img loading="lazy" src="individual/{sid}.png"></a><figcaption>{sid}</figcaption></figure>')
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>B-Free heatmaps</title><style>body{font:16px system-ui;margin:24px;background:#edf1f6;color:#263445}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:16px}figure{margin:0;background:white;padding:12px}img{width:100%}figcaption{overflow-wrap:anywhere;font-size:12px}</style><h1>B-Free · smaps_00004</h1><p>Mỗi ảnh: logit gốc bên trái, sigmoid bên phải. Cùng thang màu giữa các mẫu. Chưa có ảnh gốc/CSV dự đoán.</p><div class="grid">'+''.join(cards)+'</div>',encoding='utf-8')
    gallery=out/'gallery.html'
    gallery.write_text(gallery.read_text(encoding='utf-8').replace('B-Free · smaps_00004',f'B-Free · {html.escape(archive_name)}'),encoding='utf-8')
    print(json.dumps({'rendered':len(arrays),'groups':groups,'logit_range':[float(vmin),float(vmax)],'output':str(out.resolve())},ensure_ascii=False))

if __name__=='__main__':main()
