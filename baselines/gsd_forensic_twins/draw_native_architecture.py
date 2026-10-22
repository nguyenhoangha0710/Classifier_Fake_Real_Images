"""Generate matching editable draw.io architecture and PNG preview (native C0)."""
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1740, 1510
im = Image.new('RGB', (W, H), '#f8fafc')
d = ImageDraw.Draw(im)
font_path = 'C:/Windows/Fonts/arial.ttf'
fonts = {s: ImageFont.truetype(font_path, s) for s in (17, 19, 23, 30)}
mx = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17')
diagram = ET.SubElement(mx, 'diagram', name='Native RGB - C0', id='native-c0')
model = ET.SubElement(diagram, 'mxGraphModel', dx='1740', dy='1510', grid='1', gridSize='10', page='1', pageWidth=str(W), pageHeight=str(H))
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')
nodes = {}

def box(key, x, y, w, h, text, fill='#ffffff', size=19, border='#94a3b8'):
    nodes[key] = (x, y, w, h)
    c = ET.SubElement(root, 'mxCell', id=key, value=text, vertex='1', parent='1',
        style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={border};fontColor=#0f172a;fontSize={size};fontFamily=Arial;spacing=10;')
    ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
    d.rounded_rectangle((x,y,x+w,y+h),radius=12,fill=fill,outline=border,width=2)
    text_box = d.multiline_textbbox((0,0), text, font=fonts[size], spacing=7, align='center')
    tw, th = text_box[2]-text_box[0], text_box[3]-text_box[1]
    d.multiline_text((x+(w-tw)/2,y+(h-th)/2-text_box[1]),text,font=fonts[size],fill='#0f172a',spacing=7,align='center')

def label(key,x,y,w,h,text,size=23):
    c=ET.SubElement(root,'mxCell',id=key,value=text,vertex='1',parent='1',style=f'text;html=0;whiteSpace=wrap;align=left;verticalAlign=middle;fontSize={size};fontColor=#0f172a;fontFamily=Arial;')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    d.multiline_text((x,y),text,font=fonts[size],fill='#0f172a',spacing=7)

edge_count=0
def edge(a,b,points=None,color='#64748b',dash=False):
    global edge_count
    edge_count+=1
    ax,ay,aw,ah=nodes[a]; bx,by,bw,bh=nodes[b]
    coords=points or [(ax+aw/2,ay+ah),(bx+bw/2,by)]
    # Explicit waypoints preserve exactly the same layout in diagrams.net.
    c=ET.SubElement(root,'mxCell',id=f'e{edge_count}',source=a,target=b,edge='1',parent='1',style=f'edgeStyle=none;rounded=0;html=0;endArrow=block;endFill=1;strokeWidth=2;strokeColor={color};dashed={int(dash)};exitX={(coords[0][0]-ax)/aw};exitY={(coords[0][1]-ay)/ah};entryX={(coords[-1][0]-bx)/bw};entryY={(coords[-1][1]-by)/bh};')
    g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    arr=ET.SubElement(g,'Array',**{'as':'points'})
    for px,py in coords[1:-1]: ET.SubElement(arr,'mxPoint',x=str(px),y=str(py))
    if dash:
        import math
        for start,end in zip(coords,coords[1:]):
            dist=math.hypot(end[0]-start[0],end[1]-start[1])
            for t in range(0,int(dist),14):
                u,v=t/max(dist,1),min(t+8,dist)/max(dist,1)
                d.line([(start[0]+(end[0]-start[0])*u,start[1]+(end[1]-start[1])*u),(start[0]+(end[0]-start[0])*v,start[1]+(end[1]-start[1])*v)],fill=color,width=3)
    else: d.line(coords,fill=color,width=3)
    import math
    end,prev=coords[-1],coords[-2]; angle=math.atan2(end[1]-prev[1],end[0]-prev[0])
    d.polygon([end,(end[0]-13*math.cos(angle-.45),end[1]-13*math.sin(angle-.45)),(end[0]-13*math.cos(angle+.45),end[1]-13*math.sin(angle+.45))],fill=color)

label('title',45,25,1600,45,'TWINS + GSD  |  NATIVE RGB TRAINING (C0)',30)
label('subtitle',45,80,1600,45,'Native = decoded source pixels; no added JPEG / resize processing. Not camera RAW.',19)
label('train_title',45,145,660,40,'A. REAL-ONLY SELF-SUPERVISED TRAINING')
label('detail_title',800,145,880,40,'B. SHARED GSD ENCODER — ONE CROP')

box('real',175,205,400,80,'Real RGB images\nCOCO + ImageNet', '#e0f2fe')
box('geo',175,320,400,80,'Shared rotation / flips\nNative resolution retained')
box('crop',120,435,510,90,'Disjoint RGB crops A and B\n224 x 224 each; A intersection B = empty\nCLIP mean / std normalization')
box('x1',70,565,270,65,'x1: B x 3 x 224 x 224','#e0f2fe',17)
box('x2',410,565,270,65,'x2: B x 3 x 224 x 224','#e0f2fe',17)
box('g1',70,675,270,90,'GSD encoder\nEs + Ef + per-crop V1','#ede9fe')
box('g2',410,675,270,90,'GSD encoder\nEs + Ef + per-crop V2','#ede9fe')
box('p1',70,810,270,85,'z_clean1: B x 768\nShared projector\n768 > 2048 > 8192','#fef3c7',17)
box('p2',410,810,270,85,'z_clean2: B x 768\nShared projector\n768 > 2048 > 8192','#fef3c7',17)
box('loss',120,955,510,100,'BARLOW TWINS LOSS\nGlobal-batch normalization / correlation\nlambda_BT = 5e-5; statistics in FP32','#fee2e2')
label('sharing',65,1090,660,75,'Both views share Es, Ef and projector weights.\nV1 and V2 are computed independently per crop.',17)
for a,b in [('real','geo'),('geo','crop'),('x1','g1'),('x2','g2'),('g1','p1'),('g2','p2')]:edge(a,b)
edge('crop','x1',[(255,525),(255,545),(205,545),(205,565)])
edge('crop','x2',[(495,525),(495,545),(545,545),(545,565)])
edge('p1','loss',[(205,895),(205,925),(280,925),(280,955)])
edge('p2','loss',[(545,895),(545,925),(470,925),(470,955)])

box('xv',1055,205,390,65,'Crop xv: B x 3 x 224 x 224','#e0f2fe')
box('es',805,320,400,100,'Semantic encoder Es\nCLIP ViT-B/16\nFrozen / eval / no_grad','#e2e8f0')
box('ef',1280,320,400,100,'Forensic encoder Ef\nSame CLIP pretrained initialization\nTrain last 2 blocks + final LN','#ede9fe')
box('tokens',805,460,400,85,'Final-LN patch tokens (no CLS)\nS_v: B x 196 x 768')
box('cls',1280,460,400,85,'Final-LN CLS before visual projection\nz_v: B x 768')
box('svd',805,585,400,100,'Per-crop SVD in FP32\nS_v = A Sigma Vh; no centering\nV_v = top-8 right singular vectors','#fef3c7')
box('basis',805,725,400,75,'V_v: B x 768 x 8\nDetached / stop-gradient','#e2e8f0')
box('projection',1280,725,400,75,'p_v = (z_v V_v) V_v^T\np_v: B x 768','#fef3c7')
box('supp',1280,855,400,85,'Partial semantic suppression\nz_clean_v = z_v - 0.25 p_v','#dcfce7')
box('out',1280,985,400,70,'Output z_clean_v: B x 768','#dcfce7')
edge('xv','es',[(1140,270),(1140,295),(1005,295),(1005,320)])
edge('xv','ef',[(1360,270),(1360,295),(1480,295),(1480,320)])
for a,b in [('es','tokens'),('tokens','svd'),('svd','basis'),('ef','cls'),('projection','supp'),('supp','out')]:edge(a,b)
edge('cls','projection')
edge('basis','projection',[(1205,762),(1280,762)])
edge('cls','supp',[(1680,502),(1705,502),(1705,897),(1680,897)])
label('grad',820,1085,860,65,'Gradient: loss > projector > z_clean > Ef (last 2 blocks + LN).\nEs and SVD basis receive no gradients.',17)

label('infer_title',45,1185,1600,40,'C. AFTER SSL — REAL-ONLY DENSITY FIT AND INFERENCE')
box('inputeval',45,1250,290,100,'Native center crop 224\nCLIP normalization\nReal-fit / calibration / test',size=17)
box('enc_eval',385,1250,300,100,'Freeze Es + Ef\nPer-crop GSD retained\nProjector removed','#ede9fe',17)
box('pca',735,1250,250,100,'PCA 768 > 128\nFit on real-fit only\nwhiten = False',size=17)
box('gauss',1035,1250,270,100,'Gaussian: 1 component\nFull covariance\nFit on real-fit only',size=17)
box('score',1355,1250,330,100,'Score = -log p(z | real)\nThreshold: real-calibration p95\nFake if score > threshold','#dcfce7',17)
for a,b in [('inputeval','enc_eval'),('enc_eval','pca'),('pca','gauss'),('gauss','score')]:
    ax,ay,aw,ah=nodes[a];bx,by,bw,bh=nodes[b];edge(a,b,[(ax+aw,ay+ah/2),(bx,by+bh/2)])
label('foot',45,1390,1610,80,'C0 configuration: K = 8, alpha = 0.25. B0 control: alpha = 0 with the same training protocol.\nEligible SSL pool: parent 200K filtered for two native crops. Density-fit / calibration: saved 160K / 40K splits.\nCalibration is disjoint from density-fit, but has been seen during SSL.',17)
ET.ElementTree(mx).write(OUT/'gsd_twins_native_architecture.drawio', encoding='utf-8', xml_declaration=True)
im.save(OUT/'gsd_twins_native_architecture.png')
print('Created editable draw.io + PNG:', OUT)
