"""Build editable, uncompressed draw.io pages and matching PNG previews."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1800, 1250
FONT = 'C:/Windows/Fonts/arial.ttf'
BOLD = 'C:/Windows/Fonts/arialbd.ttf'
COLORS = {'blue': '#e4efff', 'green': '#def5ea', 'orange': '#fff0d6',
          'purple': '#eee6ff', 'red': '#ffe5e5', 'white': '#ffffff'}
mx = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17')
pages = []

class Page:
    def __init__(self, name, title, subtitle):
        self.name, self.nodes, self.edges = name, {}, []
        diagram = ET.SubElement(mx, 'diagram', name=name, id=name.split(' ')[0])
        model = ET.SubElement(diagram, 'mxGraphModel', grid='1', gridSize='10',
                              page='1', pageWidth=str(W), pageHeight=str(H))
        self.root = ET.SubElement(model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.box('title', 40, 25, 1720, 60, title, 'white', 31, True)
        self.box('sub', 40, 95, 1720, 60, subtitle, 'white', 20)
        pages.append(self)

    def box(self, key, x, y, w, h, text, color='white', size=21, bold=False):
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        scratch = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        lines = []
        for para in text.split('\n'):
            line = ''
            for word in para.split(' '):
                test = (line + ' ' + word).strip()
                if scratch.textbbox((0, 0), test, font=font)[2] > w - 32 and line:
                    lines.append(line)
                    line = word
                else:
                    line = test
            lines.append(line)
        wrapped = '\n'.join(lines)
        bbox = scratch.multiline_textbbox((0, 0), wrapped, font=font, spacing=7)
        assert bbox[3] - bbox[1] <= h - 20, (self.name, key, wrapped)
        self.nodes[key] = (x, y, w, h, wrapped, COLORS[color], size, bold)
        cell = ET.SubElement(self.root, 'mxCell', id=key, value=wrapped, vertex='1', parent='1',
            style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={COLORS[color]};strokeColor=#a4b3c5;'
                  f'fontColor=#17243a;fontSize={size};fontFamily=Arial;fontStyle={int(bold)};spacing=14;')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as': 'geometry'})

    def edge(self, a, b, points=None, color='#466480', dash=False):
        ax, ay, aw, ah = self.nodes[a][:4]
        bx, by, bw, bh = self.nodes[b][:4]
        pts = points or [(ax + aw, ay + ah / 2), (bx, by + bh / 2)]
        self.edges.append((pts, color, dash))
        cell = ET.SubElement(self.root, 'mxCell', id='edge' + str(len(self.edges)),
            source=a, target=b, edge='1', parent='1',
            style=f'edgeStyle=none;rounded=0;endArrow=block;endFill=1;strokeWidth=2;strokeColor={color};'
                  f'dashed={int(dash)};exitX={(pts[0][0]-ax)/aw};exitY={(pts[0][1]-ay)/ah};'
                  f'entryX={(pts[-1][0]-bx)/bw};entryY={(pts[-1][1]-by)/bh};')
        geo = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as':'geometry'})
        arr = ET.SubElement(geo, 'Array', **{'as':'points'})
        for x, y in pts[1:-1]:
            ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))

    def footer(self, text):
        self.box('footer', 40, 1175, 1720, 50, text, 'white', 16)

    def save_preview(self):
        im = Image.new('RGB', (W, H), '#f7f9fc')
        draw = ImageDraw.Draw(im)
        for pts, color, dash in self.edges:
            for a, b in zip(pts, pts[1:]):
                if not dash:
                    draw.line([a, b], fill=color, width=3)
                else:
                    dx, dy = b[0]-a[0], b[1]-a[1]
                    length = math.hypot(dx, dy)
                    for start in range(0, int(length), 17):
                        end = min(start + 10, length)
                        draw.line([(a[0]+dx*start/length, a[1]+dy*start/length),
                                   (a[0]+dx*end/length, a[1]+dy*end/length)], fill=color, width=3)
            end, prev = pts[-1], pts[-2]
            angle = math.atan2(end[1]-prev[1], end[0]-prev[0])
            draw.polygon([end, (end[0]-14*math.cos(angle-.45), end[1]-14*math.sin(angle-.45)),
                          (end[0]-14*math.cos(angle+.45), end[1]-14*math.sin(angle+.45))], fill=color)
        for x,y,w,h,text,fill,size,bold in self.nodes.values():
            draw.rounded_rectangle((x,y,x+w,y+h), radius=12, fill=fill, outline='#a4b3c5', width=2)
            font = ImageFont.truetype(BOLD if bold else FONT, size)
            bb = draw.multiline_textbbox((0,0), text, font=font, spacing=7, align='center')
            draw.multiline_text((x+(w-(bb[2]-bb[0]))/2-bb[0], y+(h-(bb[3]-bb[1]))/2-bb[1]),
                                text, font=font, spacing=7, align='center', fill='#17243a')
        path = OUT / (self.name.split(' ')[0] + '.png')
        im.save(path)

p = Page('01 Workflow', 'DINOv2 | Workflow trong paper gốc',
         'Nguồn chính: arXiv:2304.07193v2 (02/02/2024), TMLR 01/2024. Không có nhánh forensic/GSD bổ sung.')
p.box('web',40,210,300,140,'Ảnh web\n~1,2 tỷ ảnh sau xử lý', 'blue')
p.box('dedup',410,210,360,140,'Loại ảnh gần trùng\nLoại trùng benchmark val/test', 'blue')
p.box('retrieve',840,210,430,140,'Retrieval bằng đặc trưng tự giám sát\n+ các tập ảnh tuyển chọn', 'blue')
p.box('data',1340,210,420,140,'LVD-142M\n142.109.386 ảnh', 'blue',23,True)
for a,b in [('web','dedup'),('dedup','retrieve'),('retrieve','data')]: p.edge(a,b)
p.box('pre',1340,430,420,180,'A. Pretrain ViT-g/14\nStudent học bằng gradient\nTeacher theo EMA', 'orange',23)
p.edge('data','pre',[(1550,350),(1550,430)])
p.box('hr',840,430,430,180,'Thích nghi độ phân giải\n518 × 518\n10k iterations bổ sung', 'orange',23)
p.edge('pre','hr',[(1340,520),(1270,520)])
p.box('large',410,430,360,180,'Encoder lớn đã học\nViT-g/14', 'green',23,True)
p.edge('hr','large',[(840,520),(770,520)])
p.box('distill',410,740,600,170,'B. Distill sang S / B / L\nTeacher lớn đóng băng\nEMA student nhỏ là model cuối', 'purple',23)
p.edge('large','distill',[(590,610),(590,740)])
p.box('use',1160,740,600,170,'C. Dùng encoder\nCLS embedding + patch embeddings\nBỏ các head pretrain', 'green',23)
p.edge('distill','use')
p.edge('large','use',[(590,610),(590,665),(1460,665),(1460,740)])
p.box('eval',410,990,1350,120,'Đánh giá riêng: k-NN / linear probe / segmentation / depth / retrieval\nLinear probe dùng nhãn downstream; pretraining không dùng class label hoặc caption.', 'white',22)
p.edge('use','eval',[(1460,910),(1460,990)])
p.footer('Paper §3, §4, §5 Model distillation, Appendix B.1–B.3 | Mỗi trang sau mở rộng một phần của workflow này.')

p = Page('02 Architecture', 'DINOv2 | Teacher–student trong pretraining',
         'ViT-g/14: D=1536, 40 blocks, 24 attention heads, SwiGLU. Hai mạng cùng kiến trúc; hai head DINO/iBOT tách riêng.')
p.box('tinput',40,220,300,150,'TEACHER INPUT\n2 global crops\n224 × 224, không mask', 'green')
p.box('tvit',410,220,350,150,'Teacher ViT\nPatch embed 14 × 14\n[CLS] + positional embed\nTransformer + LayerNorm', 'green')
p.box('tcls',840,190,310,100,'CLS token\nz_t: [2B, 1536]', 'green')
p.box('tpatch',840,340,310,100,'Patch tokens\nP_t: [2B, 256, 1536]', 'green')
p.box('th1',1220,190,260,100,'DINO head_t\nprototype scores', 'green')
p.box('th2',1220,340,260,100,'iBOT head_t\nprototype scores', 'green')
p.box('targets',1550,210,210,210,'Temperature\nSinkhorn–Knopp\n3 iterations\nSTOP GRADIENT', 'green',20)
p.edge('tinput','tvit')
p.edge('tvit','tcls',[(760,265),(800,265),(800,240),(840,240)])
p.edge('tvit','tpatch',[(760,325),(800,325),(800,390),(840,390)])
p.edge('tcls','th1'); p.edge('tpatch','th2')
p.edge('th1','targets',[(1480,240),(1550,240)])
p.edge('th2','targets',[(1480,390),(1550,390)])
p.box('sinput',40,660,300,180,'STUDENT INPUT\n2 global crops: có mask\n8 local crops: không mask\nLocal: 98 × 98', 'blue')
p.box('svit',410,660,350,180,'Student ViT\nCùng global crops với teacher\nThay patch bị che bằng mask token\nBackprop qua backbone', 'blue')
p.box('scls',840,640,310,100,'CLS của global + local\nTổng 10B vector D chiều', 'blue')
p.box('spatch',840,810,310,110,'Chọn masked patch tokens\nChỉ từ global crops', 'blue')
p.box('sh1',1220,640,260,100,'DINO head_s\nsoftmax(scores / T_s)', 'blue')
p.box('sh2',1220,810,260,110,'iBOT head_s\nsoftmax(scores / T_s)', 'blue')
p.box('loss',1550,650,210,270,'DINO loss\n+ iBOT loss\n+ 0,1 KoLeo\n↓\nGradient student', 'orange',21)
p.edge('sinput','svit')
p.edge('svit','scls',[(760,705),(800,705),(800,690),(840,690)])
p.edge('svit','spatch',[(760,795),(800,795),(800,865),(840,865)])
p.edge('scls','sh1'); p.edge('spatch','sh2')
p.edge('sh1','loss',[(1480,690),(1550,690)])
p.edge('sh2','loss',[(1480,865),(1550,865)])
p.edge('targets','loss',[(1655,420),(1655,650)],'#269267',True)
p.box('kinput',1200,755,280,45,'KoLeo: CLS global', 'purple',17)
p.edge('scls','kinput',[(995,740),(995,777),(1200,777)],'#8b5cc6')
p.edge('kinput','loss',[(1480,777),(1550,777)],'#8b5cc6')
p.edge('loss','svit',[(1655,920),(1655,970),(585,970),(585,840)],'#c94f4f',True)
p.box('ema',410,490,740,95,'EMA sau optimizer step: θ_t ← m θ_t + (1 − m) θ_s\nÁp dụng cho backbone và các head; teacher không nhận gradient.', 'purple',20)
p.edge('svit','ema',[(585,660),(585,585)],'#8b5cc6',True)
p.edge('ema','tvit',[(585,490),(585,370)],'#8b5cc6',True)
p.box('embed',40,1010,1720,110,'EMBEDDING = token sau backbone, trước head. Prototype scores phục vụ loss, không phải class label.\nB = số ảnh trong batch; D = embedding dimension. Registers không thuộc paper gốc này.', 'white',22)
p.footer('Paper §4–§5 | Code: ssl_meta_arch.py, augmentations.py, configs/train/vitg14.yaml. Xanh lá: teacher; xanh dương: student.')

p = Page('03 Losses', 'DINOv2 | Loss và cách ghép mục tiêu',
         'g1, g2: hai global crops. V = {g1, g2, l1, …, l8}. M_g: vị trí mask trên global crop g. H(q,p) = −Σ q log p.')
p.box('dino',40,210,540,290,'1. IMAGE LEVEL — DINO\nL_DINO = mean H(q_t(g), p_s(v))\ng ∈ {g1,g2}; v ∈ V; v ≠ g\n\nTeacher g1 → student g2 + local\nTeacher g2 → student g1 + local\nKhông ghép global cùng view.', 'blue',22)
p.box('ibot',630,210,540,290,'2. PATCH LEVEL — iBOT\nL_iBOT = mean H(q_t(g,i), p_s(g,i))\ng ∈ {g1,g2}; i ∈ M_g\n\nCùng crop, cùng vị trí patch\nTeacher nhìn patch; student bị che\nDự đoán phân phối, không tái tạo RGB.', 'green',22)
p.box('koleo',1220,210,540,290,'3. EMBEDDING — KoLeo\nu_i = z_i / ||z_i||₂\nd_i = min_{j ≠ i} ||u_i − u_j||₂\nL_KoLeo = −(1/n) Σ log d_i\n\nÁp dụng trên student CLS tokens\nKhuyến khích đặc trưng trải rộng.', 'purple',22)
p.box('total',400,610,1000,130,'L_total = L_DINO + L_iBOT + 0,1 × L_KoLeo\nDạng tổng hợp; mean/weight theo crop và masked token được chuẩn hóa trong code.', 'orange',25,True)
p.edge('dino','total',[(310,500),(310,560),(650,560),(650,610)])
p.edge('ibot','total',[(900,500),(900,610)])
p.edge('koleo','total',[(1490,500),(1490,560),(1150,560),(1150,610)])
p.box('target',40,840,820,240,'TẠO TARGET\nStudent: softmax với temperature T_s\nTeacher: temperature T_t + Sinkhorn–Knopp\nTarget teacher được detach\nHead DINO và head iBOT không chia sẻ trọng số.', 'white',22)
p.box('meaning',940,840,820,240,'GIỚI HẠN CỦA LOSS\nKhớp teacher ≠ có ground-truth embedding\nLoss thấp ≠ phân loại downstream tốt\nĐặc trưng được đánh giá riêng bằng probe/k-NN\nKoLeo: paper và code có khác biệt phạm vi crop (README).', 'white',22)
p.footer('Paper §4, Appendix B.1 | Code ssl_meta_arch.py: cross-view global pairing, same-position masking, per-crop loss normalization.')

p = Page('04 Training', 'DINOv2 | Một training step và lịch huấn luyện',
         'Đây là pretraining model lớn từ đầu. Các con số bên phải lấy từ paper Table 16 / Appendix B, không lấy YAML mặc định.')
steps = [
 ('a','1. Lấy batch ảnh LVD-142M\nKhông dùng class label / caption.'),
 ('b','2. Tạo 2 global + 8 local crops\nRandom resized crop, flip; color jitter, grayscale, blur / solarize.'),
 ('c','3. Mask một phần global student\nTeacher nhận global không mask; local student không mask.'),
 ('d','4. Teacher forward: no_grad\nStudent forward: có autograd; tính DINO + iBOT + KoLeo.'),
 ('e','5. Backward → AdamW student\nCập nhật backbone và hai head student.'),
 ('f','6. EMA teacher sau optimizer step\nCập nhật schedule; tiếp tục batch kế tiếp.')]
for i,(key,text) in enumerate(steps):
    p.box(key,60,190+i*145,850,110,text,'blue' if i%2==0 else 'green',21)
    if i: p.edge(steps[i-1][0],key,[(485,190+(i-1)*145+110),(485,190+i*145)])
p.box('config',1010,190,730,385,'PRETRAIN ViT-g/14 — PAPER\nAdamW; batch size 3072\n625k iterations; LR 3,5 × 10⁻⁴\nWarmup 100k iterations\nWeight decay cosine: 0,04 → 0,2\nEMA momentum cosine: 0,994 → 1\nStochastic depth: 0,4; LayerScale: 10⁻⁵\nFP16; head gradient reduction FP32', 'orange',23)
p.box('high',1010,635,730,165,'HIGH-RES ADAPTATION\n518 × 518; 10k iterations bổ sung\nNén các schedule; giảm base LR\nKhông nhầm với 625k iterations ban đầu.', 'orange',22)
p.box('infra',1010,860,730,165,'HẠ TẦNG / HIỆU SUẤT\nMemory-efficient attention; sequence packing\nEfficient stochastic depth; FSDP\nPacking dùng block-diagonal attention mask.', 'white',22)
p.footer('Paper §5, Table 16, Appendix B.1–B.2 | 224/98 và 8 local crops: đối chiếu cấu hình ViT-g + code augmentation phát hành.')

p = Page('05 Distill and inference', 'DINOv2 | Distillation khác pretraining; embedding dùng ở đâu?',
         'Teacher lớn đóng băng khi distill; EMA của student nhỏ là bản dùng cuối. Không dùng vòng EMA để cập nhật teacher lớn.')
p.box('img',40,220,300,130,'Ảnh / multi-crops\nKhông masking', 'white',23)
p.box('big',440,190,490,150,'Teacher lớn: ViT-g/14 đã học\nFROZEN — không gradient\nKhông cập nhật bằng EMA student nhỏ', 'green',23)
p.box('small',440,440,490,150,'Student nhỏ: ViT-S / B / L\nCó gradient\nKhông stochastic depth', 'blue',23)
p.edge('img','big',[(340,255),(390,255),(390,265),(440,265)])
p.edge('img','small',[(190,350),(190,515),(440,515)])
p.box('loss',1090,315,620,150,'Giữ các loss distillation\niBOT trên hai global crops dù không masking\nBackprop chỉ qua student nhỏ', 'orange',23)
p.edge('big','loss',[(930,265),(1010,265),(1010,350),(1090,350)])
p.edge('small','loss',[(930,515),(1010,515),(1010,430),(1090,430)])
p.box('ema',440,690,490,125,'Bản EMA riêng của student nhỏ\nĐược giữ làm model cuối', 'purple',23)
p.edge('small','ema',[(685,590),(685,690)],'#8b5cc6',True)
p.box('enc',1090,690,620,125,'Inference: ảnh → backbone → LayerNorm\nBỏ DINO/iBOT heads; không mask', 'green',23)
p.edge('ema','enc')
p.box('cls',1090,900,280,150,'CLS embedding\n[B, D]\nImage-level tasks', 'green',23)
p.box('patch',1430,900,280,150,'Patch embeddings\n[B, N, D]\nDense tasks', 'green',23)
p.edge('enc','cls',[(1230,815),(1230,900)])
p.edge('enc','patch',[(1570,815),(1570,900)])
p.box('dims',40,900,900,150,'Embedding dimension: S=384; B=768; L=1024; g=1536\nN=(H/14) × (W/14), với H,W chia hết cho 14\nLinear probe: đóng băng backbone, chỉ train classifier bằng nhãn.', 'white',22)
p.footer('Paper §5 Model distillation; Appendix B.3 | Kích thước và đầu ra: code vision_transformer.py. Chi tiết paper/code khác nhau: README.')

path = OUT / 'dinov2_original_workflow_architecture.drawio'
ET.ElementTree(mx).write(path, encoding='utf-8', xml_declaration=True)
for page in pages:
    page.save_preview()

# Structural checks for editable draw.io cells and attached connectors.
tree = ET.parse(path)
for diagram in tree.findall('diagram'):
    cells = diagram.findall('./mxGraphModel/root/mxCell')
    ids = [c.get('id') for c in cells]
    assert len(ids) == len(set(ids))
    for c in cells:
        if c.get('edge') == '1':
            assert c.get('source') in ids and c.get('target') in ids
print('Created 5 editable pages and 5 PNG previews; XML, edge references and text bounds passed.')
