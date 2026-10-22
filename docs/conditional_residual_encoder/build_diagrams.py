"""Generate editable draw.io pages and matching previews; no model training."""
from pathlib import Path
import math
import textwrap
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONT = 'C:/Windows/Fonts/arial.ttf'
BOLD = 'C:/Windows/Fonts/arialbd.ttf'
COLORS = {'input': '#dbeafe', 'fixed': '#e2e8f0', 'learn': '#ede9fe',
          'label': '#dcfce7', 'loss': '#ffedd5', 'note': '#fef3c7'}
mx = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17')
validation = []


class Page:
    def __init__(self, slug, name, title, subtitle, height=1500):
        self.slug, self.w, self.h = slug, 1800, height
        diagram = ET.SubElement(mx, 'diagram', id=slug, name=name)
        model = ET.SubElement(diagram, 'mxGraphModel', grid='1', gridSize='10',
                              page='1', pageWidth=str(self.w), pageHeight=str(height),
                              math='0', background='#f8fafc')
        self.root = ET.SubElement(model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.nodes, self.count = {}, 0
        self.im = Image.new('RGB', (self.w, height), '#f8fafc')
        self.draw = ImageDraw.Draw(self.im)
        self.text('title', 45, 25, 1710, 60, title, 32, True)
        self.text('subtitle', 45, 90, 1710, 70, subtitle, 21)

    def wrap(self, text, width, font):
        result = []
        for line in text.split('\n'):
            current = ''
            for word in line.split(' '):
                trial = (current + ' ' + word).strip()
                if current and self.draw.textlength(trial, font=font) > width:
                    result.append(current)
                    current = word
                else:
                    current = trial
            result.append(current)
        return '\n'.join(result)

    def node(self, key, x, y, w, h, text, kind='fixed', size=22, bold=False, align='center', plain=False):
        if key in self.nodes:
            raise ValueError('Duplicate node: '+key)
        if x < 0 or y < 0 or x+w > self.w or y+h > self.h:
            raise ValueError('Out of canvas: '+key)
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        # Arial on this host lacks the set-membership glyph; use readable Vietnamese.
        wrapped = self.wrap(text.replace('∈', 'thuộc'), w-30, font)
        box = self.draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=7, align=align)
        th = box[3]-box[1]
        if th > h-12:
            raise ValueError(f'Text overflow {self.slug}/{key}: {th} > {h-12}')
        fill = COLORS.get(kind, kind)
        style = (f'whiteSpace=wrap;html=0;fontFamily=Arial;fontSize={size};fontColor=#0f172a;'
                 f'align={align};verticalAlign=middle;spacing=15;fontStyle={int(bold)};')
        style += 'text;strokeColor=none;fillColor=none;' if plain else f'rounded=1;fillColor={fill};strokeColor=#94a3b8;strokeWidth=1.5;'
        cell = ET.SubElement(self.root, 'mxCell', id=key, value=wrapped, vertex='1', parent='1', style=style)
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
        self.nodes[key] = (x, y, w, h)
        if not plain:
            self.draw.rounded_rectangle((x, y, x+w, y+h), radius=14, fill=fill, outline='#94a3b8', width=2)
        tx = x+15 if align == 'left' else x+(w-(box[2]-box[0]))/2
        ty = y+(h-th)/2-box[1]
        self.draw.multiline_text((tx, ty), wrapped, font=font, fill='#0f172a', spacing=7, align=align)

    def text(self, key, x, y, w, h, text, size=22, bold=False):
        self.node(key,x,y,w,h,text,size=size,bold=bold,align='left',plain=True)

    def edge(self, source, target, points=None, color='#64748b', dashed=False):
        self.count += 1
        ax,ay,aw,ah = self.nodes[source]
        bx,by,bw,bh = self.nodes[target]
        coords = points or [(ax+aw/2,ay+ah), (bx+bw/2,by)]
        style = (f'edgeStyle=none;rounded=0;endArrow=block;endFill=1;strokeWidth=2;'
                 f'strokeColor={color};dashed={int(dashed)};'
                 f'exitX={(coords[0][0]-ax)/aw};exitY={(coords[0][1]-ay)/ah};'
                 f'entryX={(coords[-1][0]-bx)/bw};entryY={(coords[-1][1]-by)/bh};')
        cell = ET.SubElement(self.root,'mxCell',id=f'edge{self.count}',source=source,target=target,edge='1',parent='1',style=style)
        geo = ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
        arr = ET.SubElement(geo,'Array',**{'as':'points'})
        for x,y in coords[1:-1]:
            ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        for start,end in zip(coords,coords[1:]):
            length=math.dist(start,end)
            if dashed:
                for step in range(0,int(length),15):
                    t0,t1=step/max(length,1),min(step+9,length)/max(length,1)
                    self.draw.line([(start[0]+(end[0]-start[0])*t0,start[1]+(end[1]-start[1])*t0),
                                    (start[0]+(end[0]-start[0])*t1,start[1]+(end[1]-start[1])*t1)],fill=color,width=3)
            else:
                self.draw.line([start,end],fill=color,width=3)
        end,prev=coords[-1],coords[-2]
        a=math.atan2(end[1]-prev[1],end[0]-prev[0])
        self.draw.polygon([end,(end[0]-14*math.cos(a-.45),end[1]-14*math.sin(a-.45)),
                          (end[0]-14*math.cos(a+.45),end[1]-14*math.sin(a+.45))],fill=color)

    def save(self):
        path=OUT/(self.slug+'.png')
        self.im.save(path)
        validation.append({'page':self.slug,'nodes':len(self.nodes),'edges':self.count,'preview':path.name})


p = Page('01_architecture','01 - Kiến trúc tổng thể',
         'ENCODER HỌC QUY LUẬT PHẦN DƯ CỦA ẢNH THẬT',
         'Bản thiết kế nghiên cứu v1 • Chỉ dùng ảnh thật để train • Chưa được thực nghiệm xác nhận', 1630)
p.node('image',650,170,500,100,'Ảnh thật RGB\nB × 3 × 256 × 256','input')
p.node('fre',90,320,680,100,'BỘ LỌC PHẦN DƯ R — FROZEN\nGrayscale → FRE 1→8 channels\nOutput: B × 8 × 256 × 256','fixed')
p.node('condition',1050,320,660,130,'ĐIỀU KIỆN CỤC BỘ c — KHÔNG HỌC\nLow-pass RGB; mỗi ô target lấy 5 số:\nmean R/G/B, độ biến động sáng, năng lượng cạnh','fixed')
p.node('sample',90,490,680,100,'CHỌN 8 PATCH 64×64 KHÔNG GIAO NHAU\n3 context A + 3 context B + 2 target\nGiữ khoảng trống ≥16 px với FRE kernel15','fixed')
p.node('a',45,660,340,100,'CONTEXT A\nB × 3 × 8 × 64 × 64\nNhững miếng được xem','input',20)
p.node('b',425,660,340,100,'CONTEXT B\nB × 3 × 8 × 64 × 64\nNhóm xem thứ hai','input',20)
p.node('target',825,660,340,100,'TARGET RESIDUAL\nB × 2 × 8 × 64 × 64\nKhông vào encoder','label',20)
p.node('ct',1250,660,480,100,'c: B × 2 × 16 × 5\n2 target; 16 ô/target\nChỉ đặc điểm ảnh thô','fixed',20)
p.node('encoder',45,820,720,110,'SHARED ENCODER E — ĐƯỢC TRAIN\nPatch CNN → pooling từng patch → pooling cả nhóm\nHai nhóm A/B dùng chung trọng số','learn')
p.node('label',825,820,340,145,'LABEL y — TỰ TÍNH\n16 ô 16×16/target\n8 energy + 8 corr ngang\n+ 8 corr dọc = 24 số\nB × 2 × 16 × 24','label',20)
p.node('za',45,1000,340,80,'z_A: B × 64\nBản mô tả từ context A','learn',20)
p.node('zb',425,1000,340,80,'z_B: B × 64\nBản mô tả từ context B','learn',20)
p.node('pred',1230,840,500,140,'BỘ DỰ ĐOÁN P — ĐƯỢC TRAIN\nNối z (64 số) với c (5 số)\nMLP 69→128→128→48\nDùng chung cho A và B','learn',21)
p.node('output',1230,1040,500,110,'μ: 24 giá trị trung tâm dự đoán\ns: 24 log-variance dự đoán\nMỗi context dự đoán cùng 2 target','learn',21)
p.node('ssl',45,1160,720,105,'LOSS PHỤ TRÊN z_A VÀ z_B\nCùng ảnh → tương đồng; giữa các ảnh → giữ thông tin\nInvariance + variance floor + covariance','loss',21)
p.node('nll',825,1210,905,100,'LOSS CHÍNH: CONDITIONAL GAUSSIAN NLL\nSo μ, s dự đoán với label y đo từ target\nDự đoán cả giá trị và độ dao động phù hợp','loss',21)
p.node('total',300,1390,1200,90,'LOSS TỔNG = L_pred + 0,1 × (25 L_inv + 25 L_var + L_cov)\nBackprop chỉ cập nhật encoder E và predictor P','loss',23,True)
p.text('legend',45,1510,1710,75,'Xám: phép tính cố định / không nhận gradient • Tím: tham số được học • Xanh lá: đáp án tự tạo • Cam: loss\nB = số ảnh trong một batch. z là embedding học được, chưa chứng minh là thông số camera. Không có CLIP/GSD trong bản CNN này.',20)
p.edge('image','fre',[(760,270),(760,295),(430,295),(430,320)])
p.edge('image','condition',[(1040,270),(1040,295),(1380,295),(1380,320)])
p.edge('fre','sample')
for key,x in [('a',215),('b',595),('target',995)]:
    p.edge('sample',key,[(430,590),(430,620),(x,620),(x,660)])
p.edge('condition','ct')
p.edge('a','encoder',[(215,760),(215,820)])
p.edge('b','encoder',[(595,760),(595,820)])
p.edge('target','label')
p.edge('encoder','za',[(215,930),(215,1000)])
p.edge('encoder','zb',[(595,930),(595,1000)])
p.edge('ct','pred',[(1490,760),(1490,840)])
p.edge('za','pred',[(215,1080),(215,1110),(790,1110),(790,800),(1470,800),(1470,840)])
p.edge('zb','pred',[(765,1040),(1195,1040),(1195,910),(1230,910)])
p.edge('pred','output')
p.edge('za','ssl',[(215,1080),(215,1160)])
p.edge('zb','ssl',[(595,1080),(595,1160)])
p.edge('label','nll',[(995,965),(995,1210)],color='#15803d')
p.edge('output','nll',[(1480,1150),(1480,1210)])
p.edge('ssl','total',[(405,1265),(405,1350),(650,1350),(650,1390)])
p.edge('nll','total',[(1275,1310),(1275,1350),(1150,1350),(1150,1390)])
p.save()

p = Page('02_encoder_and_labels','02 - Encoder và label',
         'BÊN TRONG ENCODER VÀ CÁCH TẠO ĐÁP ÁN',
         'Encoder đọc context; đáp án lấy từ target khác vị trí. Mỗi ký hiệu đều gắn với một đối tượng cụ thể.',1630)
p.text('left',45,165,810,55,'A. ENCODER — TỪ 3 PATCH ĐẾN 64 SỐ',25,True)
p.text('right',945,165,810,55,'B. LABEL — ĐO TRỰC TIẾP TỪ TARGET',25,True)
p.node('input',75,245,740,85,'Một nhóm context: B × 3 × 8 × 64 × 64\nGộp B và 3 patch để chạy chung CNN','input')
p.node('cnn',75,390,740,210,'PATCH CNN — DÙNG CHUNG CHO TẤT CẢ PATCH\nConv3×3, stride2, padding1 + GroupNorm8 + GELU\n8→32: 32×32; 32→64: 16×16\n64→128: 8×8; 128→128: 4×4\nOutput mỗi patch: 128 × 4 × 4','learn',23)
p.node('pool',75,660,740,110,'SPATIAL POOLING TRONG MỖI PATCH\nMean128 + std128 → 256 số\nLinear 256→128 → patch embedding128','learn')
p.node('set',75,830,740,115,'SET POOLING TRÊN 3 PATCH\nMean128 + std128 → 256 số\nKhông dùng thứ tự hoặc tọa độ patch làm input','learn')
p.node('mlp',75,1005,740,100,'MLP 256→128→64, GELU ở lớp giữa\nz: B × 64 — không L2-normalize trước loss variance','learn')
p.node('pred',75,1165,740,125,'PREDICTOR CHO TỪNG Ô TARGET\n[z:64 ; c:5] → 69→128→128→48\n24 số μ + 24 số s = log(variance)\ns được clamp [-6, 6] trước NLL','learn',21)
p.node('r',975,245,740,85,'Một target residual: 8 × 64 × 64\nChia 4×4 ô; mỗi ô có 8 × 16 × 16 giá trị','label')
p.node('stat',975,390,740,220,'MỖI Ô, MỖI RESIDUAL CHANNEL:\n1. e = log(mean(r²) + ε_E): độ lớn phần dư\n2. ρ_h = corr(pixel bên trái, pixel bên phải)\n3. ρ_v = corr(pixel phía trên, pixel phía dưới)\n8 channels × 3 thống kê = 24 số\nε_E = 1e-8 với output FRE đã khóa scale','label',22)
p.node('normalize',975,670,740,130,'CHUẨN HÓA LABEL BẰNG TRAIN STATISTICS\ny_j = (u_j − m_j) / max(a_j, 1e-3)\nu_j: thống kê thô; m_j: mean; a_j: std\nChỉ ước lượng m, a từ ảnh thuộc encoder-train','label',22)
p.node('y',975,860,740,85,'Đáp án cuối y: B × 2 × 16 × 24\nB ảnh; 2 target; 16 ô; 24 thống kê','label')
p.node('c',975,1005,740,150,'ĐIỀU KIỆN c: B × 2 × 16 × 5\nTừ RGB low-pass bằng average filter7×7\nMean R/G/B; std luminance; gradient energy\nLuminance = 0,299R + 0,587G + 0,114B\nKhông đưa residual target vào predictor','fixed',21)
p.node('sample',975,1215,740,160,'SAMPLING TRÊN NATIVE CROP256\n9 vị trí: x,y ∈ {0,80,160}; mỗi core64\nRandom chọn 8 core: A3, B3, target2\nGap16 hạn chế chia sẻ receptive field của FRE15\nGiữ target/cùng tọa độ khi lấy RGB conditions','note',21)
p.node('note',75,1415,1640,150,'TẠI SAO LABEL KHÔNG ĐƯỢC HỌC?\nNếu encoder và bộ tạo đáp án cùng tự do, chúng có thể cùng đổi để bài tập trở nên tầm thường.\nFRE, thống kê label, chuẩn hóa label và phép lấy conditions đều cố định / stop-gradient.\nResidual vẫn có thể chứa texture; 24 thống kê là thiết kế pilot, chưa phải mô tả đầy đủ quá trình tạo ảnh.','note',22)
for a,b in [('input','cnn'),('cnn','pool'),('pool','set'),('set','mlp'),('mlp','pred'),('r','stat'),('stat','normalize'),('normalize','y')]: p.edge(a,b)
p.save()

p = Page('03_losses','03 - Công thức loss',
         'LOSS: CHẤM ĐIỂM DỰ ĐOÁN VÀ GIỮ EMBEDDING CÓ THÔNG TIN',
         'Công thức trong file này dùng text chỉnh sửa được. File loss_explained_vi.md có công thức LaTeX và giải thích đầy đủ.',2060)
p.node('vars',45,175,1710,180,'TỪ ĐIỂN BIẾN\nB: số ảnh/batch; v ∈ {A,B}: nhóm context; t=1..2: target patch; q=1..16: ô/target; j=1..24: thống kê.\nz_A,z_B: bản mô tả 64 số từ hai nhóm context; c: 5 đặc điểm thô của ô target; y: đáp án 24 thống kê đã chuẩn hóa.\nμ: giá trị trung tâm predictor đoán; s: log-variance predictor đoán; θ: encoder weights; ψ: predictor weights.\nD=64: chiều embedding; ε=1e-4: số nhỏ trong variance regularizer; các mean/covariance bên dưới tính theo batch ảnh.','fixed',22,align='left')
p.node('pred',45,405,1710,240,'1. LOSS CHÍNH — DỰ ĐOÁN PHÂN PHỐI CỦA LABEL\nℓ = 0,5 × [ exp(−s) × (y − μ)² + s ]\nL_pred = mean ℓ trên mọi ảnh B, cả hai context A/B, 2 target, 16 ô và 24 thống kê.\nμ và s = P_ψ(z_v, c). Dùng s sau clamp [-6,6]; label y không có gradient.\n(y−μ)²: đoán sai bị phạt. exp(−s): đoán quá tự tin mà sai bị phạt mạnh. +s: ngăn tăng variance vô hạn.\nCó thể bỏ hằng số 0,5 log(2π). NLL không nhất thiết không âm; loss âm không tự có nghĩa training bị lỗi.','loss',23,align='left')
p.node('inv',45,700,1710,170,'2. INVARIANCE — HAI NHÓM CÙNG ẢNH CẦN TƯƠNG ĐỒNG\nL_inv = mean_(b,k) [ z_A[b,k] − z_B[b,k] ]²\nb là chỉ số ảnh; k=1..64 là chiều embedding. Mean = tổng chia B×64.\nNếu chỉ dùng loss này, model có thể xuất một vector giống hệt cho mọi ảnh → mất thông tin.','loss',23,align='left')
p.node('var',45,925,1710,225,'3. VARIANCE FLOOR — GIỮ SỰ ĐA DẠNG GIỮA CÁC ẢNH\nv(Z) = (1/D) Σ_k max(0, 1 − sqrt(Var_batch(Z[:,k]) + ε))\nL_var = [ v(Z_A) + v(Z_B) ] / 2\nZ_A và Z_B là ma trận B×64; mỗi hàng là embedding của một ảnh.\nNếu một chiều gần như giống nhau trên mọi ảnh, std nhỏ hơn1 → bị phạt. Dùng sample variance, mẫu số B−1.\nKhông L2-normalize z trước term này; B≥2, pilot khuyên B64. Chống collapse không chứng minh học đúng forensic.','loss',22,align='left')
p.node('cov',45,1205,1710,240,'4. COVARIANCE — GIẢM CÁC CHIỀU TRÙNG THÔNG TIN\nZ_centered = Z − mean_batch(Z)\nC(Z) = Z_centeredᵀ Z_centered / (B−1)\ncov_penalty(Z) = (1/D) Σ_(k≠l) C(Z)[k,l]²\nL_cov = cov_penalty(Z_A) + cov_penalty(Z_B)\nC(Z) kích thước64×64; bỏ đường chéo. Các chiều cùng tăng/giảm theo nhau bị phạt. Không ép từng ảnh thành Gaussian.','loss',23,align='left')
p.node('total',45,1500,1710,155,'5. LOSS TỔNG — RECIPE THỬ NGHIỆM\nL_total = L_pred + λ × (25 L_inv + 25 L_var + L_cov)\nBắt đầu λ=0,1; chạy đối chứng λ=0. Không coi các hệ số này là đã tối ưu cho forensic.\nBackprop qua predictor → z → encoder. Không update FRE, label statistics hoặc RGB conditions.','loss',24,True,align='left')
p.node('example',45,1710,1710,170,'VÍ DỤ NLL MỘT CHIỀU — CHỈ MINH HỌA, KHÔNG PHẢI KẾT QUẢ TRAIN\nLabel y=2. Model đoán μ=1 và variance=1 → s=0 → ℓ=0,5.\nGiữ μ=1 nhưng đoán variance=4 → s=log4 → ℓ≈0,818: tăng độ bất định không miễn phí.\nMột chiều không đại diện toàn loss; phải lấy mean đúng các trục đã ghi.','note',23,align='left')
p.text('warning',45,1920,1710,95,'LOSS GIẢM CHỈ CHỨNG MINH MODEL LÀM BÀI TẬP TỐT HƠN.\nCần kiểm tra c-only, hoán đổi z, unseen camera/source và unseen generators để chứng minh ích lợi cho detection.',23,True)
p.save()

p = Page('04_training_and_gmm','04 - Train và GMM',
         'TỪ MỘT BƯỚC TRAIN ĐẾN ĐÁNH GIÁ ẢNH MỚI',
         'Phân biệt: học encoder/predictor trước; fit density sau. Không dùng final-test để lựa model hoặc threshold.',1630)
p.text('tr',45,165,800,55,'A. HỌC BẰNG ẢNH THẬT',25,True)
p.text('inf',945,165,800,55,'B. FREEZE VÀ XÂY DETECTOR',25,True)
p.node('d',75,245,740,105,'REAL ENCODER-TRAIN\nNative crop256; không resize để ép đủ kích thước\nSampling A3 + B3 + target2; label chuẩn hóa cố định','input',21)
p.node('f',75,410,740,115,'FORWARD\nR frozen → context encoder E → z_A,z_B\nP(z_A,c), P(z_B,c) → μ,s\nTự tính đáp án y từ residual target','learn',21)
p.node('l',75,585,740,100,'LOSS TỔNG\nPrediction NLL + invariance + variance + covariance\nTính label statistics / loss nhạy số học bằng FP32','loss',21)
p.node('update',75,745,740,100,'BACKWARD VÀ ADAMW UPDATE\nUpdate θ (encoder) và ψ (predictor)\nKhông có BCE real/fake hoặc generator label','learn',21)
p.node('val',75,905,740,140,'REAL VALIDATION — FIXED PATCH LAYOUT\nSo prediction NLL; std/effective rank của z\nSo predictor c-only và predictor với z bị hoán đổi\nLoss tốt không đảm bảo AUROC unseen tốt','note',21)
p.node('recipe',75,1105,740,170,'PILOT ĐỀ XUẤT — CHƯA BENCHMARK\nBatch64 ảnh; AdamW LR1e-4, WD1e-4\n20 epochs; warmup5% rồi cosine đến LR1e-6\nShared flip/rotation90°; chưa thêm JPEG/blur\nKhóa seed, data membership và cách chọn checkpoint','fixed',21)
p.node('freeze',975,245,740,100,'FREEZE ENCODER E VÀ PREDICTOR P\nGiữ cả predictor để đo surprise\nKhóa sampling / preprocessing / label scaler','fixed',21)
p.node('surprise',975,405,740,140,'ĐO SAI SỐ PHÂN PHỐI TRÊN REAL-FIT\na[v,t,q,j] = 0,5 × [exp(−s) × (y−μ)² + s]\nVới mỗi chiều j: lấy mean và std qua v,t,q\nh = [mean24 ; std24] → embedding48','label',21)
p.node('gauss',975,605,740,105,'FIT GAUSSIAN/GMM TRÊN h CỦA REAL-FIT\nBắt đầu K=1, full covariance + regularization\nFit bằng real-fit, không dùng calibration/test','learn',21)
p.node('cal',975,770,740,130,'REAL-CALIBRATION RIÊNG\nscore = −log p_GMM(h)\nNgưỡng τ = percentile95 của calibration scores\nNhắm empirical FPR5% trên calibration','fixed',21)
p.node('test',975,940,740,120,'ẢNH MỚI → CÙNG E/P → h → score\nscore > τ: bất thường so với real reference\nKhông tự đồng nhất score với P(fake)\nReport real FPR và unseen-fake TPR/AUROC','input',21)
p.node('audit',975,1120,740,155,'THÍ NGHIỆM QUYẾT ĐỊNH\n1. Predictor chỉ dùng c\n2. Predictor dùng c + đúng z\n3. Predictor dùng c + z từ ảnh khác\nĐúng z phải có ích trên validation và detection.','note',21)
p.node('split',75,1340,1640,105,'TÁCH DỮ LIỆU THEO ẢNH GỐC\nEncoder-train / encoder-validation / real-density-fit / real-calibration / final-test.\nMọi crop hoặc biến thể của một ảnh phải cùng split. Final-test gồm real mới + fake unseen generators.','note',22)
p.text('gsd',75,1490,1640,100,'GSD KHÔNG CÓ TRONG BẢN CNN V1 NÀY. Nếu thay context CNN bằng CLIP+GSD, phải thiết kế lại input size và tensor contract.\nKhông chiếu trực tiếp CNN feature128 bằng semantic basis CLIP1024. Bản GSD là một thí nghiệm mở rộng riêng.',21)
for a,b in [('d','f'),('f','l'),('l','update'),('update','val'),('freeze','surprise'),('surprise','gauss'),('gauss','cal'),('cal','test')]: p.edge(a,b)
p.save()

destination=OUT/'conditional_residual_encoder_vi.drawio'
ET.indent(mx, space='  ')
ET.ElementTree(mx).write(destination,encoding='utf-8',xml_declaration=True)
# Structural validation: IDs and edge references are valid within each editable page.
parsed=ET.parse(destination)
for diagram in parsed.getroot().findall('diagram'):
    cells=diagram.findall('.//mxCell')
    ids=[c.attrib['id'] for c in cells]
    assert len(ids)==len(set(ids)), diagram.attrib['name']
    for cell in cells:
        if cell.get('edge')=='1':
            assert cell.get('source') in ids and cell.get('target') in ids
import json
(OUT/'diagram_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'drawio':str(destination),'pages':validation},ensure_ascii=False,indent=2))
