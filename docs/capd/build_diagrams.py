"""Build CAPD architecture + workflow diagrams as SVG/HTML (for PNG) and .drawio from one spec."""
import html, json, sys
from xml.sax.saxutils import escape

PAL = {
 'input':  ('#EEF2F7', '#5B6B82'),
 'frozen': ('#E3ECFA', '#3A6BC4'),
 'train':  ('#FDEBD8', '#D9822B'),
 'out':    ('#E2F4EA', '#2F8F5B'),
 'loss':   ('#FBE3E6', '#C2414F'),
 'data':   ('#EFE8FB', '#7A57C9'),
 'proc':   ('#E6F3F7', '#2C8AA6'),
 'dec':    ('#FFF6D6', '#B88A00'),
 'note':   ('#F7F7F5', '#8A8A85'),
}
LH = 17

class D:
    def __init__(s, W, H, title):
        s.W, s.H, s.title = W, H, title
        s.nodes, s.edges, s.groups, s.texts = {}, [], [], []
    def node(s, id, x, y, w, title, lines=(), kind='proc', h=None, shape='rect', dashed=False):
        if h is None: h = 26 + LH*len(lines) + (8 if lines else 0)
        s.nodes[id] = dict(x=x, y=y, w=w, h=h, title=title, lines=list(lines), kind=kind, shape=shape, dashed=dashed)
    def group(s, x, y, w, h, title, kind='note', dashed=False):
        s.groups.append(dict(x=x, y=y, w=w, h=h, title=title, kind=kind, dashed=dashed))
    def edge(s, a, b, sa='r', sb='l', label='', dashed=False, via=None):
        s.edges.append(dict(a=a, b=b, sa=sa, sb=sb, label=label, dashed=dashed, via=via or []))
    def text(s, x, y, t, size=12, color='#555', anchor='start', bold=False):
        s.texts.append(dict(x=x, y=y, t=t, size=size, color=color, anchor=anchor, bold=bold))

    # ---------- geometry ----------
    def port(s, id, side, frac=0.5):
        n = s.nodes[id]
        x, y, w, h = n['x'], n['y'], n['w'], n['h']
        return {'r': (x+w, y+h*frac), 'l': (x, y+h*frac), 't': (x+w*frac, y), 'b': (x+w*frac, y+h)}[side]
    def route(s, e):
        (sx, sy), (tx, ty) = s.port(e['a'], e['sa']), s.port(e['b'], e['sb'])
        pts = [(sx, sy)]
        if e['via']:
            pts += e['via']
        elif e['sa'] in 'rl' and e['sb'] in 'rl':
            if abs(sy-ty) > 1:
                mx = (sx+tx)/2; pts += [(mx, sy), (mx, ty)]
        elif e['sa'] in 'tb' and e['sb'] in 'tb':
            if abs(sx-tx) > 1:
                my = (sy+ty)/2; pts += [(sx, my), (tx, my)]
        elif e['sa'] in 'rl':
            pts += [(tx, sy)]
        else:
            pts += [(sx, ty)]
        pts.append((tx, ty))
        return pts

    # ---------- SVG ----------
    def svg(s):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.W}" height="{s.H}" viewBox="0 0 {s.W} {s.H}" font-family="Inter, DejaVu Sans, sans-serif">',
             '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4A4A48"/></marker></defs>',
             f'<rect width="{s.W}" height="{s.H}" fill="#FFFFFF"/>',
             f'<text x="24" y="36" font-size="22" font-weight="700" fill="#1F2430">{escape(s.title)}</text>']
        for g in s.groups:
            f, st = PAL[g['kind']]
            da = ' stroke-dasharray="7 5"' if g['dashed'] else ''
            o.append(f'<rect x="{g["x"]}" y="{g["y"]}" width="{g["w"]}" height="{g["h"]}" rx="12" fill="{f}" fill-opacity="0.45" stroke="{st}" stroke-width="1.4"{da}/>')
            o.append(f'<text x="{g["x"]+14}" y="{g["y"]+22}" font-size="14" font-weight="700" fill="{st}">{escape(g["title"])}</text>')
        for e in s.edges:
            p = s.route(e)
            d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in p)
            da = ' stroke-dasharray="6 4"' if e['dashed'] else ''
            o.append(f'<path d="{d}" fill="none" stroke="#4A4A48" stroke-width="1.6"{da} marker-end="url(#ar)"/>')
            if e['label']:
                # label at midpoint of longest segment
                segs = list(zip(p, p[1:]))
                (x1, y1), (x2, y2) = max(segs, key=lambda q: abs(q[0][0]-q[1][0])+abs(q[0][1]-q[1][1]))
                lx, ly = (x1+x2)/2, (y1+y2)/2
                tw = 6.6*len(e['label'])+10
                o.append(f'<rect x="{lx-tw/2:.1f}" y="{ly-10:.1f}" width="{tw:.1f}" height="19" rx="4" fill="#FFFFFF" stroke="#D0D0CC"/>')
                o.append(f'<text x="{lx:.1f}" y="{ly+4:.1f}" font-size="11.5" text-anchor="middle" fill="#333">{escape(e["label"])}</text>')
        for id, n in s.nodes.items():
            f, st = PAL[n['kind']]
            x, y, w, h = n['x'], n['y'], n['w'], n['h']
            da = ' stroke-dasharray="6 4"' if n['dashed'] else ''
            if n['shape'] == 'diamond':
                cx, cy = x+w/2, y+h/2
                o.append(f'<polygon points="{cx},{y} {x+w},{cy} {cx},{y+h} {x},{cy}" fill="{f}" stroke="{st}" stroke-width="1.6"{da}/>')
            else:
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{f}" stroke="{st}" stroke-width="1.6"{da}/>')
            total = 18 + LH*len(n['lines'])
            ty = y + (h-total)/2 + 14
            o.append(f'<text x="{x+w/2}" y="{ty:.1f}" font-size="13.5" font-weight="700" text-anchor="middle" fill="#1F2430">{escape(n["title"])}</text>')
            for i, l in enumerate(n['lines']):
                o.append(f'<text x="{x+w/2}" y="{ty+19+LH*i:.1f}" font-size="12" text-anchor="middle" fill="#3A3A38">{escape(l)}</text>')
        for t in s.texts:
            fw = ' font-weight="700"' if t['bold'] else ''
            o.append(f'<text x="{t["x"]}" y="{t["y"]}" font-size="{t["size"]}" text-anchor="{t["anchor"]}" fill="{t["color"]}"{fw}>{escape(t["t"])}</text>')
        o.append('</svg>')
        return '\n'.join(o)

    # ---------- draw.io ----------
    def drawio(s, page):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        k = 0
        def nid():
            nonlocal k; k += 1; return f'c{k}'
        cells.append(f'<mxCell id="{nid()}" value="{escape(s.title)}" style="text;html=1;fontSize=22;fontStyle=1;fontFamily=Inter;" vertex="1" parent="1"><mxGeometry x="20" y="12" width="1000" height="34" as="geometry"/></mxCell>')
        for g in s.groups:
            f, st = PAL[g['kind']]
            style = f'rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={f};strokeColor={st};verticalAlign=top;align=left;spacingLeft=12;spacingTop=4;fontStyle=1;fontSize=14;fontColor={st};fontFamily=Inter;opacity=60;' + ('dashed=1;' if g['dashed'] else '')
            cells.append(f'<mxCell id="{nid()}" value="{escape(g["title"])}" style="{style}" vertex="1" parent="1"><mxGeometry x="{g["x"]}" y="{g["y"]}" width="{g["w"]}" height="{g["h"]}" as="geometry"/></mxCell>')
        ids = {}
        for id, n in s.nodes.items():
            f, st = PAL[n['kind']]
            val = f'<b>{html.escape(n["title"])}</b>' + ''.join(f'<br><font style="font-size:12px">{html.escape(l)}</font>' for l in n['lines'])
            shape = 'rhombus;' if n['shape'] == 'diamond' else 'rounded=1;'
            style = f'{shape}whiteSpace=wrap;html=1;fillColor={f};strokeColor={st};fontFamily=Inter;fontSize=13;strokeWidth=1.5;' + ('dashed=1;' if n['dashed'] else '')
            cid = nid(); ids[id] = cid
            cells.append(f'<mxCell id="{cid}" value="{escape(val)}" style="{style}" vertex="1" parent="1"><mxGeometry x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" as="geometry"/></mxCell>')
        sidexy = {'r': (1, .5), 'l': (0, .5), 't': (.5, 0), 'b': (.5, 1)}
        for e in s.edges:
            ex, ey = sidexy[e['sa']]; nx, ny = sidexy[e['sb']]
            style = f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#4A4A48;strokeWidth=1.5;fontFamily=Inter;fontSize=11;exitX={ex};exitY={ey};entryX={nx};entryY={ny};' + ('dashed=1;' if e['dashed'] else '')
            pts = s.route(e)[1:-1]
            arr = ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts)
            geo = f'<mxGeometry relative="1" as="geometry"><Array as="points">{arr}</Array></mxGeometry>' if pts else '<mxGeometry relative="1" as="geometry"/>'
            cells.append(f'<mxCell id="{nid()}" value="{escape(e["label"])}" style="{style}" edge="1" parent="1" source="{ids[e["a"]]}" target="{ids[e["b"]]}">{geo}</mxCell>')
        for t in s.texts:
            al = {'start': 'left', 'middle': 'center', 'end': 'right'}[t['anchor']]
            cells.append(f'<mxCell id="{nid()}" value="{escape(t["t"])}" style="text;html=1;align={al};fontSize={t["size"]};fontColor={t["color"]};fontFamily=Inter;{"fontStyle=1;" if t["bold"] else ""}" vertex="1" parent="1"><mxGeometry x="{t["x"] - (0 if al=="left" else 200)}" y="{t["y"]-14}" width="{400 if al=="center" else 380}" height="20" as="geometry"/></mxCell>')
        body = ''.join(cells)
        return f'<diagram id="{page}" name="{escape(page)}"><mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" page="1" pageWidth="{s.W}" pageHeight="{s.H}"><root>{body}</root></mxGraphModel></diagram>'

def legend(d, x, y, items):
    for i, (kind, label, dashed) in enumerate(items):
        f, st = PAL[kind]
        yy = y + i*24
        d.groups  # noqa
        d.node(f'lg{i}', x, yy, 22, '', [], kind, h=16, dashed=dashed)
        d.text(x+30, yy+13, label, 12, '#333')

# =========================== FIGURE 1: ARCHITECTURE ===========================
A = D(1560, 900, 'CAPD — Codec-Aligned Patch Detector: kiến trúc')
A.node('in', 24, 205, 170, 'Ảnh RGB', ['độ phân giải gốc', '(không resize)'], 'input')
A.node('tr', 236, 140, 196, 'Train', ['random crop 224×224', 'codec aug ghép cặp real–fake'], 'proc')
A.node('te', 236, 268, 196, 'Test', ['tiling 224×224 toàn ảnh', '(tối đa 16 tile)'], 'proc')
A.edge('in', 'tr', 'r', 'l'); A.edge('in', 'te', 'r', 'l')

A.group(466, 92, 278, 300, 'Backbone: DINOv3 ViT-L/16', 'frozen')
A.node('pe', 486, 132, 238, 'Patch embedding 16×16', ['224² → 196 token + CLS'], 'frozen')
A.node('bb', 486, 212, 238, '24 block Transformer', ['trọng số gốc đóng băng'], 'frozen')
A.node('lo', 486, 300, 238, 'LoRA r = 8 (train)', ['chèn vào qkv, proj, fc1, fc2 (~6M)'], 'train')
A.edge('tr', 'pe', 'r', 'l'); A.edge('te', 'pe', 'r', 'l')
A.edge('pe', 'bb', 'b', 't'); A.edge('lo', 'bb', 't', 'b', 'ΔW = B·A', dashed=True)

A.node('tok', 486, 420, 238, 'Patch tokens', ['14 × 14 × 1024 (bỏ CLS)'], 'frozen')
A.edge('bb', 'tok', 'l', 'l', via=[(476, A.port('bb','l')[1]), (476, A.port('tok','l')[1])])

A.group(782, 92, 262, 300, 'PatchHead (train)', 'train')
A.node('h1', 800, 130, 226, 'LayerNorm → reshape', ['lưới 14×14×1024'], 'train')
A.node('h2', 800, 196, 226, 'DWConv 3×3 → GELU', ['ngữ cảnh patch lân cận'], 'train')
A.node('h3', 800, 262, 226, 'Conv 1×1: 1024→256 → GELU', [], 'train', h=40)
A.node('h4', 800, 320, 226, 'Conv 1×1: 256→1', [], 'train', h=40)
ty_ = A.port('tok','r')[1]
A.edge('tok', 'h1', 'r', 'l', via=[(752, ty_), (752, A.port('h1','l')[1])])
A.edge('h1', 'h2', 'b', 't'); A.edge('h2', 'h3', 'b', 't'); A.edge('h3', 'h4', 'b', 't')

A.node('map', 1080, 120, 210, 'Bản đồ logit patch s', ['s ∈ R^(14×14) mỗi tile'], 'out')
A.node('mil', 1080, 250, 210, 'MIL pooling (LSE, r = 5)', ['z = (1/r)·log mean exp(r·s)', 'gộp mọi patch của mọi tile'], 'train')
A.edge('h4', 'map', 'r', 'l', via=[(1056, A.port('h4','r')[1]), (1056, A.port('map','l')[1])])
A.edge('map', 'mil', 'b', 't')
A.node('pf', 1330, 258, 200, 'P(fake) = σ(z − α)', ['α = 0 mặc định (ngưỡng 0.5)'], 'out')
A.node('hm', 1330, 120, 200, 'Heatmap vùng nghi sinh', ['upsample s về kích thước ảnh'], 'out')
A.edge('mil', 'pf', 'r', 'l'); A.edge('map', 'hm', 'r', 'l')

A.node('gap', 800, 470, 226, 'Nhánh phụ: GAP → Linear', ['logit z_gap (chỉ dùng khi train)'], 'train', dashed=True)
A.edge('tok', 'gap', 'r', 'l', dashed=True, via=[(752, ty_), (752, A.port('gap','l')[1])])

A.group(1080, 412, 450, 158, 'Hàm loss (chỉ khi train)', 'loss')
A.node('ls', 1098, 446, 414, 'L = L_img + 0.5·L_gap + 0.5·L_neg + 0.2·L_pix',
       ['L_img: BCE(z, y)      L_gap: BCE(z_gap, y)',
        'L_neg: mọi patch của ảnh real → 0 (không cần mask)',
        'L_pix: BCE patch với mask (chỉ fake có mask, tùy chọn)'], 'loss')
A.edge('mil', 'ls', 'b', 't', dashed=True, via=[(1185, 380), (1305, 380), (1305, 446)])
A.edge('gap', 'ls', 'r', 'l', dashed=True)

# Extension B
A.group(24, 620, 1506, 255, 'Mở rộng B — Gated-Experts (chỉ làm khi CAPD đã ổn định; oracle gain > 3 điểm)', 'note', dashed=True)
A.node('low', 48, 668, 330, 'Expert low-level', ['SRM/NPR residual → ResNet18 (scratch)', 'FiLM điều kiện: bảng lượng tử JPEG kênh Y (64)', '+ cờ "không nén" → logit z_low'], 'train', dashed=True)
A.node('cz', 48, 790, 330, 'Expert patch = CAPD (logit z)', [], 'out', h=44, dashed=True)
A.node('meta', 420, 700, 330, 'Meta-features (không nhãn)', ['chất lượng/QT, kích thước ảnh,', 'bất đồng giữa expert, % patch nghi'], 'proc', dashed=True)
A.node('gate', 800, 700, 300, 'Gate MLP → trọng số softmax', ['train bằng stacking trên val', 'khác generator (không end-to-end)'], 'train', dashed=True)
A.node('fuse', 1150, 700, 350, 'Logit hợp nhất → hiệu chỉnh', ['temperature trên val + α không nhãn', '(bỏ qua nếu logit không hai đỉnh)'], 'out', dashed=True)
A.edge('low', 'meta', 'r', 'l', dashed=True)
A.edge('meta', 'gate', 'r', 'l', dashed=True)
A.edge('cz', 'gate', 'r', 'b', dashed=True, via=[(950, 812)])
A.edge('gate', 'fuse', 'r', 'l', dashed=True)

A.text(24, 600, 'Chú thích:', 12.5, '#333', bold=True)
for i, (kind, label) in enumerate([('frozen', 'đóng băng'), ('train', 'được train'), ('out', 'đầu ra'), ('loss', 'loss'), ('input', 'đầu vào / xử lý')]):
    f, st = PAL[kind]
    A.node(f'lg{i}', 100 + i*150, 588, 22, '', [], kind, h=16)
    A.text(130 + i*150, 601, label, 12, '#333')
A.text(860, 601, 'Nét đứt: chỉ dùng khi train hoặc thuộc phần mở rộng', 12, '#555')

# =========================== FIGURE 2: WORKFLOW ===========================
W = D(1600, 1040, 'CAPD — workflow: dữ liệu → train → đánh giá')
W.group(20, 60, 360, 640, '① Dữ liệu (đã có trên Kaggle + tự sinh)', 'data')
W.node('r1', 40, 100, 320, 'Real', ['ImageNet + COCO (manifest 200K)', 'real COCO của B-Free'], 'data')
W.node('r2', 40, 200, 320, 'Real chất lượng cao', ['RAISE + FFHQ chưa nén', '(một phần train, phần còn lại test)'], 'data')
W.node('f1', 40, 316, 320, 'Fake B-Free (SD2.1)', ['self-conditioned, inpainted,', 'inpainted + origBG (cùng COCO ID)'], 'data')
W.node('f2', 40, 432, 320, 'Fake tái tạo VAE (~20k)', ['SD2.1 / SDXL / FLUX: encode → decode'], 'data')
W.node('f3', 40, 530, 320, 'Inpaint-exchange tự sinh (~10k)', ['mask 2–40%, dán lại pixel gốc', 'có mask → dùng cho L_pix'], 'data')
W.text(40, 660, 'KHÔNG train trên Tiny, CommFor, INP-X', 12.5, '#C2414F', bold=True)
W.text(40, 680, '→ các tập này chỉ để test unseen', 12, '#555')

W.group(410, 60, 360, 640, '② Ghép cặp, codec, kiểm tra shortcut', 'proc')
W.node('pair', 430, 100, 320, 'Ghép cặp real–fake', ['theo source ID (COCO ID)'], 'proc')
W.node('codec', 430, 200, 320, 'Codec aug DÙNG CHUNG cho cả cặp', ['20% giữ nguyên · 50% JPEG Q60–100', '10% WebP · 20% resize 0.5–1.5 + JPEG'], 'proc')
W.node('samp', 430, 330, 320, 'Lấy mẫu cân bằng', ['50% real / 50% fake; fake: 35% self-cond,', '35% inpaint (½ origBG), 20% VAE, 10% exch.'], 'proc')
W.node('aud', 470, 470, 240, 'Luật shortcut ≤ 60%?', ['(chỉ số token / chỉ QT)'], 'dec', h=120, shape='diamond')
for s_ in ['r1', 'r2', 'f1', 'f2', 'f3']:
    W.edge(s_, 'pair', 'r', 'l', via=[(390, W.port(s_, 'r')[1]), (390, 130)])
W.edge('pair', 'codec', 'b', 't'); W.edge('codec', 'samp', 'b', 't'); W.edge('samp', 'aud', 'b', 't')
W.edge('aud', 'codec', 'l', 'l', dashed=True, via=[(420, 530), (420, W.port('codec','l')[1])])
W.text(428, 612, 'Không → sửa dữ liệu, chạy lại', 12, '#B88A00', bold=True)

W.group(800, 60, 360, 640, '③ Train (Kaggle 2×T4)', 'train')
W.node('smoke', 820, 100, 320, 'Smoke run', ['2k ảnh, 1 epoch: đo VRAM, tốc độ,', 'kiểm tra loss giảm'], 'train')
W.node('full', 820, 216, 320, 'Train đầy đủ', ['fp16 · DDP · grad checkpointing', 'AdamW (LoRA 1e-4, head 1e-3), cosine', '3–5 epoch · seed 42 / 43 / 44'], 'train')
W.node('val', 820, 356, 320, 'Validation mỗi epoch', ['B-Free valid + RAISE held-out'], 'train')
W.node('sel', 820, 456, 320, 'Chọn checkpoint', ['AUC trung bình + FPR real không nén', '(không chọn theo Tiny / test)'], 'train')
W.edge('aud', 'smoke', 'r', 'l', 'Có', via=[(785, 530), (785, 133)])
W.edge('smoke', 'full', 'b', 't'); W.edge('full', 'val', 'b', 't'); W.edge('val', 'sel', 'b', 't')

W.group(1190, 60, 390, 640, '④ Đánh giá (một model, không tune đích)', 'out')
W.node('tests', 1210, 100, 350, 'Cohort test', ['Tiny 7 gen: raw / fake-JPEG96 / resize256+JPEG96', 'CommFor cân bằng 2.000 ảnh', 'B-Free valid 6 nhánh (gồm origBG)', 'INP-X: INP và INP-X', 'Synthbuster / Chameleon (nếu có)', 'RAISE / FFHQ held-out (FPR)'], 'out')
W.node('met', 1210, 290, 350, 'Chỉ số', ['AUC · AP · BA@0.5 · Real/Fake-acc', 'TPR@FPR 1% / 5% · worst-case', 'bootstrap CI qua 3 seed'], 'out')
W.node('go', 1255, 430, 260, 'Đạt mục tiêu?', ['CommFor ≥ AIDE Full', 'origBG recall ≥ GSD + 20'], 'dec', h=140, shape='diamond')
W.edge('sel', 'tests', 'r', 'l', via=[(1182, W.port('sel','r')[1]), (1182, W.port('tests','l')[1])])
W.edge('tests', 'met', 'b', 't'); W.edge('met', 'go', 'b', 't')

W.group(20, 730, 1560, 290, '⑤ Ablation, so sánh baseline, mở rộng', 'note')
W.node('base', 40, 780, 360, 'Train lại baseline trên CÙNG dữ liệu', ['CLIP-LP · NPR-ResNet18 · GSD', '→ kiểm chứng "dữ liệu > kiến trúc"'], 'proc')
W.node('abl', 440, 780, 420, 'Ablation CAPD', ['head: CLS / GAP / PatchHead / +MIL', 'có/không origBG · λ_pix = 0 / 0.2', 'codec ghép cặp / độc lập · real không nén', 'DINOv3 / DINOv2 / CLIP-L · tiling / resize'], 'proc')
W.node('final', 900, 800, 250, 'Bảng kết quả cuối', ['CAPD vs CLIP, NPR, AIDE, GSD', 'trung bình + worst-case'], 'out')
W.node('ext', 1240, 800, 320, 'Mở rộng B: Gated-Experts', ['chỉ khi oracle gain > 3 điểm'], 'note', dashed=True)
W.node('err', 1240, 930, 320, 'Phân tích lỗi', ['xem heatmap, sửa dữ liệu / head'], 'loss', h=64)
W.edge('go', 'abl', 'b', 't', 'Có', via=[(1385, 600), (1385, 750), (650, 750)])
W.edge('base', 'abl', 'r', 'l')
W.edge('abl', 'final', 'r', 'l')
W.edge('final', 'ext', 'r', 'l', dashed=True)
W.edge('go', 'err', 'r', 'r', 'Không', dashed=True, via=[(1575, 500), (1575, 962)])
W.edge('err', 'full', 'l', 'r', 'train lại', dashed=True, via=[(1170, W.port('err','l')[1]), (1170, W.port('full','r')[1])])

if __name__ == '__main__':
    out = sys.argv[1]
    for name, d in [('capd_architecture', A), ('capd_workflow', W)]:
        open(f'{out}/{name}.svg', 'w').write(d.svg())
    xml = f'<mxfile host="app.diagrams.net">{A.drawio("Kiến trúc CAPD")}{W.drawio("Workflow CAPD")}</mxfile>'
    open(f'{out}/capd_architecture_workflow.drawio', 'w').write(xml)
    print('ok')
