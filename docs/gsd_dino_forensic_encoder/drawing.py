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
        # Arial has incomplete Unicode subscript/superscript coverage on Windows.
        # Use portable plain-text math labels in both the XML and PNG previews.
        text = text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')).replace('ᵀ', '^T')
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

