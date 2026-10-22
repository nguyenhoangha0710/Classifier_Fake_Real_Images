"""Proposed real-only CLIP+GSD encoder: editable draw.io and shared-layout previews."""
import json
import xml.etree.ElementTree as ET
from PIL import Image
from drawing import Page, OUT, mx, pages, W, H

p = Page('01 Workflow', 'Model đề xuất | Workflow tổng thể',
         'Real-only 200K • CLIP ViT-L/14 pretrained • GSD last 4, k=64 • Fixed reference basis • DINOv2-style SSL')
p.box('raw',40,210,360,150,'POOL 200K ẢNH REAL\nẢnh raw từ dataset\nKhông thêm fake vào pretraining', 'blue',23,True)
p.box('manifest',490,210,520,150,'Audit / lọc ảnh nhỏ, lỗi và trùng\nKhóa manifest train và nguồn dữ liệu\nGhi số ảnh hợp lệ thực tế sau lọc', 'blue',22)
p.box('ref',1130,210,630,150,'Chọn reference từ TRAIN\nKhông dùng validation/test\nSố ảnh và policy reference: CHƯA CHỐT', 'purple',22)
p.edge('raw','manifest'); p.edge('manifest','ref')
p.box('basis',1130,460,630,155,'OFFLINE: frozen CLIP → SVD\n4 basis V₂₁,V₂₂,V₂₃,V₂₄\nMỗi basis [1024,64], giữ cố định', 'purple',23)
p.edge('ref','basis',[(1445,360),(1445,460)])
p.box('init',40,460,970,155,'Khởi tạo student = CLIP pretrained + GSD + DINO/iBOT heads\nTeacher = copy toàn bộ student ban đầu\nTrainable student; teacher cập nhật EMA', 'green',23)
p.box('ssl',490,740,1270,175,'ONLINE SSL\nẢnh raw → preprocessing tạo global/local crops → teacher/student\nDINO + iBOT + 0,1 KoLeo → gradient student → EMA teacher\nKhông chạy semantic anchor/SVD lại trong từng training step', 'orange',23)
p.edge('manifest','ssl',[(750,360),(1055,360),(1055,690),(800,690),(800,740)])
p.edge('init','ssl',[(525,615),(525,740)])
p.edge('basis','ssl',[(1445,615),(1445,740)],'#8b5cc6',True)
p.box('export',490,1005,1270,110,'Export encoder teacher EMA + basis + preprocessing contract\nBỏ DINO/iBOT heads → CLS / patch embeddings → đánh giá downstream riêng', 'green',23)
p.edge('ssl','export',[(1125,915),(1125,1005)])
p.footer('Thiết kế đã thống nhất trong chat. Đây là biến thể nghiên cứu CLIP+GSD kết hợp DINOv2; chưa có kết quả training xác nhận.')

p = Page('02 Reference basis', 'Model đề xuất | Tạo basis GSD một lần, trước khi train',
         'Cách B đã chọn: basis độc lập với ảnh/crop đang xử lý. Basis cố định; không đồng nghĩa mức suppression λ cố định.')
p.box('ref',40,220,340,160,'Reference ảnh REAL\nLấy trong manifest train\nPolicy crop/resize và số ảnh\nCHƯA CHỐT', 'blue',22)
p.box('anchor',470,220,400,160,'CLIP ViT-L/14 pretrained\nSemantic anchor FROZEN\nEval mode, no_grad\nKhông mask trong bước offline', 'green',22)
p.box('features',960,220,370,160,'Lấy PATCH tokens\nSau các block 21,22,23,24\nKhông lấy CLS để fit basis\nMỗi token: 1024 chiều', 'green',22)
p.box('svd',1420,220,340,160,'Gộp token theo từng layer\nP_l = U_l Σ_l V_lᵀ\nLấy 64 right singular vectors', 'purple',22)
for a,b in [('ref','anchor'),('anchor','features'),('features','svd')]: p.edge(a,b)
p.box('v21',40,560,360,130,'V₂₁\n[1024,64]', 'purple',25,True)
p.box('v22',490,560,360,130,'V₂₂\n[1024,64]', 'purple',25,True)
p.box('v23',940,560,360,130,'V₂₃\n[1024,64]', 'purple',25,True)
p.box('v24',1390,560,370,130,'V₂₄\n[1024,64]', 'purple',25,True)
for key,x in [('v21',220),('v22',670),('v23',1120),('v24',1575)]:
    p.edge('svd',key,[(1590,380),(1590,470),(x,470),(x,560)],'#8b5cc6')
p.box('save',40,820,810,210,'LƯU BASIS ARTIFACT\n4 ma trận FP32; V_lᵀ V_l ≈ I₆₄\nKhóa CLIP checkpoint, reference IDs/hash\nLưu layer, k, preprocessing và phương pháp fit\nSVD không center để khớp phép GSD hiện tại', 'purple',22)
p.box('reuse',940,820,820,210,'DÙNG LẠI TRONG TRAIN VÀ INFERENCE\nStudent + teacher dùng cùng V_l ở từng layer\nKhông fit lại theo batch, không EMA basis\nLocal crop 49 patch vẫn nhân được với V_l [1024,64]\nTập reference không tham gia làm target riêng', 'green',22)
p.edge('save','reuse')
p.footer('GSD paper: reference-based inference. Dùng fixed reference basis cả lúc SSL là quyết định của model đề xuất, không phải recipe đã được paper kiểm chứng.')

p = Page('03 Teacher student', 'Model đề xuất | Kiến trúc teacher–student',
         'Cả hai encoder đều khởi tạo từ CLIP pretrained. Transformer và attention đã nằm bên trong CLIP; GSD là phần chèn thêm.')
p.box('basis',560,190,680,100,'SHARED FIXED BASIS\nV₂₁,V₂₂,V₂₃,V₂₄ — [1024,64] mỗi layer', 'purple',23)
p.box('tin',40,355,760,110,'TEACHER INPUT\nHai global crops không mask, từ ảnh REAL', 'green',23)
p.box('sin',1000,355,760,110,'STUDENT INPUT\nCùng global crops + local crops; mask global tokens', 'blue',23)
p.box('te',40,560,760,185,'TEACHER ENCODER — NO GRAD\nCLIP ViT-L/14 pretrained: patch embedding + 24 blocks\nGSD sau block 21,22,23,24 → final LayerNorm\nTeacher backbone và GSD đi theo student bằng EMA', 'green',23)
p.box('se',1000,560,760,185,'STUDENT ENCODER — TRAINABLE\nCLIP ViT-L/14 pretrained: patch embedding + 24 blocks\nGSD sau block 21,22,23,24 → final LayerNorm\nMask token ở đầu vào; CLS không bị mask', 'blue',23)
p.edge('tin','te',[(420,465),(420,560)])
p.edge('sin','se',[(1380,465),(1380,560)])
p.edge('basis','te',[(880,290),(880,515),(420,515),(420,560)],'#8b5cc6',True)
p.edge('basis','se',[(920,290),(920,515),(1380,515),(1380,560)],'#8b5cc6',True)
p.box('tout',40,820,760,95,'CLS: [2B,1024] + patches: [2B,256,1024]\nGlobal crop 224×224; mọi target đều detach', 'green',22)
p.box('sout',1000,820,760,95,'CLS của global + local; patch tokens giữ thứ tự\nGlobal: 256 patches; local 98×98: 49 patches', 'blue',22)
p.edge('te','tout',[(420,745),(420,820)])
p.edge('se','sout',[(1380,745),(1380,820)])
p.box('th',40,990,760,120,'DINO head_t ← CLS; iBOT head_t ← patch tokens\nTemperature + Sinkhorn–Knopp tạo targets\nHai head riêng; được cập nhật bằng EMA', 'green',22)
p.box('sh',1000,990,760,120,'DINO head_s ← CLS; iBOT head_s ← masked patches\nStudent softmax / log-softmax\nKoLeo lấy CLS trực tiếp, trước head', 'blue',22)
p.edge('tout','th',[(420,915),(420,990)])
p.edge('sout','sh',[(1380,915),(1380,990)])
p.footer('Luồng head/loss theo DINOv2. Luồng token GSD theo code hiện tại, đổi sang fixed basis. Chi tiết gradient/EMA ở trang 07.')

p = Page('04 Inside encoder', 'Model đề xuất | Bên trong một encoder CLIP + GSD',
         'Cùng sơ đồ cho teacher/student. Khác biệt: student có masking ở đầu vào; teacher dùng trọng số EMA và không mask.')
p.box('image',40,210,440,135,'Crop RGB đã preprocessing\n[B,3,H,W]\nTeacher/student nhận ảnh tensor', 'white',23)
p.box('patch',610,210,510,135,'CLIP patch embedding 14×14\n[B,N,1024]\nN = (H/14) × (W/14)', 'blue',23)
p.box('mask',1250,210,510,135,'Student: thay masked patch tokens\nThêm CLS + positional embedding\nCLIP pre-LayerNorm', 'blue',22)
p.edge('image','patch'); p.edge('patch','mask')
p.box('front',40,455,440,140,'CLIP Transformer blocks 1–20\nAttention + FFN có sẵn trong CLIP\n[B,N+1,1024]', 'blue',23)
p.edge('mask','front',[(1505,345),(1505,400),(260,400),(260,455)])
p.box('b21',610,455,510,140,'CLIP block 21 → GSD₂₁\nDùng V₂₁ [1024,64]\nGiữ nguyên [B,N+1,1024]', 'purple',23)
p.box('b22',1250,455,510,140,'CLIP block 22 → GSD₂₂\nDùng V₂₂ [1024,64]\nGiữ nguyên [B,N+1,1024]', 'purple',23)
p.edge('front','b21'); p.edge('b21','b22')
p.box('b23',1250,730,510,140,'CLIP block 23 → GSD₂₃\nDùng V₂₃ [1024,64]\nCLS và patch positions vẫn giữ nguyên', 'purple',23)
p.box('b24',610,730,510,140,'CLIP block 24 → GSD₂₄\nDùng V₂₄ [1024,64]\nGSD áp dụng lên toàn chuỗi token', 'purple',23)
p.box('out',40,730,440,140,'Final CLIP LayerNorm\nCLS: [B,1024]\nPatches: [B,N,1024]\nKHÔNG patch-mean trước iBOT', 'green',22)
p.edge('b22','b23',[(1505,595),(1505,730)])
p.edge('b23','b24',[(1250,800),(1120,800)])
p.edge('b24','out',[(610,800),(480,800)])
p.box('formula',40,990,1720,130,'GSD_l: F_parallel = (F_l V_l) V_lᵀ ; r = ||F_parallel||²_F / (||F_l||²_F + ε)\nλ_l = λ_max sigmoid(a_l r + b_l) ; F_hat = F_l − λ_l F_parallel\nV_l cố định; a_l,b_l học ở student / EMA ở teacher. Không dùng CLIP visual projection hoặc text encoder.', 'white',22)
p.footer('GSD xử lý chiều feature, không xóa/đổi thứ tự token. Embedding cuối dùng hidden dimension 1024 của CLIP ViT-L/14.')

p = Page('05 Crops and mask', 'Model đề xuất | Crop ở ngoài encoder; mask ở bên trong student',
         'Crop = chọn một vùng ảnh raw. Patch embedding = chuyển crop thành token. Mask = thay nội dung token, không xóa vị trí.')
p.box('raw',40,210,360,140,'Ảnh raw REAL\nĐọc/decode từ dataset\nKhông có nhánh raw riêng\nngoài teacher/student', 'white',22)
p.box('prep',490,210,580,140,'PREPROCESSING / AUGMENTATION\nDINOv2-style multi-crop\n2 global 224×224 + 8 local 98×98\nAugment → tensor → normalize', 'blue',22)
p.box('views',1160,210,600,140,'Teacher nhận global views\nStudent nhận global + local views\nMỗi encoder tự thực hiện patch embedding\nCùng global view có cùng lưới vị trí', 'green',22)
p.edge('raw','prep'); p.edge('prep','views')
p.box('tseq',40,480,800,150,'TEACHER — GLOBAL VIEW g1\n[CLS] [p1] [p2] [p3] [p4] …\nKhông mask; đi qua CLIP + GSD\nĐầu ra có token tại mọi vị trí', 'green',23)
p.box('sseq',960,480,800,150,'STUDENT — CÙNG GLOBAL VIEW g1\n[CLS] [p1] [MASK] [p3] [MASK] …\nMask token học được, thêm vị trí trước Transformer\nĐầu ra vẫn có token số 2, số 4, …', 'blue',23)
p.box('dino',40,770,800,165,'DINO — KHÁC VIEW\nTeacher CLS(g1) → student CLS(g2), CLS(local)\nTeacher CLS(g2) → student CLS(g1), CLS(local)\nKhông ghép teacher/student global cùng view', 'orange',22)
p.box('ibot',960,770,800,165,'iBOT — CÙNG VIEW, CÙNG VỊ TRÍ\nTeacher patch(g1,2) → student patch(g1,2) bị che\nTeacher patch(g1,4) → student patch(g1,4) bị che\nLoss chỉ chọn masked indices, không ghép chéo crop', 'orange',22)
p.edge('tseq','ibot',[(440,630),(440,685),(1360,685),(1360,770)])
p.edge('sseq','ibot',[(1530,630),(1530,770)])
p.box('note',40,1010,1720,110,'Global student: một phần patch được mask; local student: không mask; CLS không mask.\nKhông bôi đen ảnh RGB để thay thế token masking. Fixed reference basis không xem crop hiện tại.', 'white',22)
p.footer('2 global / 8 local, 224/98 theo cấu hình ViT-g DINOv2 phát hành. Augmentation giữ recipe gốc; khả năng phù hợp forensic cần được đánh giá.')

p = Page('06 Losses', 'Model đề xuất | Ba loss và đúng vị trí lấy đặc trưng',
         'K_D / K_P: số prototype của hai head, CHƯA CHỐT. Khác hoàn toàn k=64 của GSD. H(q,p) = −Σ_j q_j log p_j.')
p.box('dino',40,210,540,340,'1. DINO — CLS → DINO HEAD\np_s = softmax(head_s(CLS_s) / τ_s)\nq_t = teacher targets sau Sinkhorn–Knopp\nL_D = mean H(q_t(g), p_s(v))\n\nChỉ cặp view khác nhau, cùng ảnh\n2 global × (1 global khác + 8 local)\n= 18 cặp mỗi ảnh; teacher target detach', 'blue',21)
p.box('ibot',630,210,540,340,'2. iBOT — PATCH → iBOT HEAD\np_s^P = softmax(head_s(P_s) / τ_s)\nq_t^P = target patch teacher, đã detach\nL_P = mean_crop mean_mask H(q_t^P, p_s^P)\n\nCùng global crop, cùng patch index\nNormalize theo số mask của từng crop\nKhông-mask crop đóng góp 0', 'green',21)
p.box('koleo',1220,210,540,340,'3. KoLeo — CLS TRƯỚC HEAD\nu_b = CLS_b / ||CLS_b||₂\nd_b = min_{c ≠ b} ||u_b − u_c||₂\nL_K^(g) = −mean_b log(d_b + ε)\n\nChỉ student global CLS\nTính riêng g1 và g2 rồi cộng\nKhông lấy hai views cùng ảnh làm NN', 'purple',21)
p.box('total',390,660,1020,130,'L_total = L_DINO + L_iBOT + 0,1 × L_KoLeo\nGiữ loss normalization và teacher-target balancing của DINOv2 code.', 'orange',25,True)
p.edge('dino','total',[(310,550),(310,610),(650,610),(650,660)])
p.edge('ibot','total',[(900,550),(900,660)])
p.edge('koleo','total',[(1490,550),(1490,610),(1150,610),(1150,660)])
p.box('grad',40,890,820,210,'GRADIENT ĐI QUA\nStudent heads → token embeddings → GSD → CLIP\nKoLeo đi trực tiếp từ CLS trước head\nMask token nhận gradient qua student\nBasis V và semantic anchor không nhận gradient', 'blue',22)
p.box('limits',940,890,820,210,'Ý NGHĨA / GIỚI HẠN\nDINO: nhất quán toàn ảnh giữa views\niBOT: dự đoán phân phối đặc trưng vùng bị che\nKoLeo: khuyến khích độ phân tán embedding\nKhông có BCE/CE real–fake; chưa bảo đảm forensic', 'white',22)
p.footer('DINOv2 losses: dino_clstoken_loss.py, ibot_patch_loss.py, koleo_loss.py và ssl_meta_arch.py. Không thay bằng Barlow Twins/VICReg.')

p = Page('07 Training and export', 'Model đề xuất | Vòng train, EMA và export encoder',
         'Giữ cơ chế DINOv2; không áp nguyên lịch 625k iterations / batch / LR của ViT-g train từ đầu lên CLIP pretrained.')
steps = [
 ('a','1. Nạp batch REAL → tạo global/local crops\nDùng manifest hợp lệ từ pool 200K; không lấy fake.'),
 ('b','2. Student global: thay patch bằng mask token\nTeacher global và student local không mask.'),
 ('c','3. Teacher forward no_grad + student forward\nCả hai dùng cùng các fixed reference basis.'),
 ('d','4. Tính DINO + iBOT + 0,1 KoLeo\nZero-grad → backward → AdamW cập nhật student.'),
 ('e','5. EMA sau optimizer step\nθ_t ← m θ_t + (1−m) θ_s; không EMA basis.'),
 ('f','6. Log, checkpoint và validation protocol riêng\nChỉ export trọng số train hợp lệ; không dùng test để chọn model.')]
for i,(key,text) in enumerate(steps):
    p.box(key,40,190+i*145,900,110,text,'blue' if i%2==0 else 'green',21)
    if i: p.edge(steps[i-1][0],key,[(490,190+(i-1)*145+110),(490,190+i*145)])
p.box('state',1040,190,720,260,'QUY TẮC TRẠNG THÁI\nStudent = trainable CLIP + GSD + hai heads\nTeacher = copy student lúc đầu, rồi EMA\nFrozen CLIP anchor chỉ dùng tạo basis offline\nBasis lưu dạng buffer/artifact cố định\nEMA không đồng nghĩa λ_student = λ_teacher', 'purple',22)
p.box('export',1040,510,720,230,'EXPORT / INFERENCE\nGiữ teacher EMA encoder + 4 basis\nBỏ DINO/iBOT heads; không masking\nẢnh → preprocessing → CLIP+GSD → LayerNorm\nCLS [B,1024] hoặc patches [B,N,1024]\nClassifier / density model là bước downstream riêng', 'green',22)
p.box('todo',1040,800,720,260,'THÔNG SỐ CHƯA CHỐT\nReference size và sampling / preprocessing policy\nSố prototypes và cấu hình head\nLR từng nhóm, batch, số bước / epochs\nWarmup, momentum / temperature schedules\nNgưỡng lọc ảnh nhỏ; policy validation\nChưa tự thêm high-res stage hoặc model distillation', 'orange',21)
p.footer('Artifact này mô tả thiết kế; không thay code training đang có và không khởi chạy train. Protocol đánh giá forensic cần khóa trước khi chạy.')

path = OUT / 'gsd_dino_forensic_encoder.drawio'
ET.ElementTree(mx).write(path, encoding='utf-8', xml_declaration=True)
for page in pages:
    page.save_preview()

overview = Image.new('RGB', (1800, 2500), '#dce3ec')
for i,page in enumerate(pages):
    im = Image.open(OUT / (page.name.split(' ')[0] + '.png'))
    overview.paste(im.resize((900,625)), ((i % 2)*900, (i // 2)*625))
overview.save(OUT / 'overview.png')

report = {'pages': [], 'checks': ['unique cell IDs', 'valid edge references', 'nodes in page bounds',
                                  'text fits boxes', 'PNG dimensions']}
for diagram,page in zip(ET.parse(path).findall('diagram'),pages):
    cells = diagram.findall('./mxGraphModel/root/mxCell')
    ids = [c.get('id') for c in cells]
    assert len(ids) == len(set(ids))
    for c in cells:
        if c.get('edge') == '1':
            assert c.get('source') in ids and c.get('target') in ids
    for key,node in page.nodes.items():
        x,y,w,h = node[:4]
        assert 0 <= x and 0 <= y and x+w <= W and y+h <= H, key
    png = OUT / (page.name.split(' ')[0] + '.png')
    assert Image.open(png).size == (W,H)
    report['pages'].append({'name':page.name, 'nodes':len(page.nodes), 'edges':len(page.edges)})
report['status'] = 'passed'
(OUT / 'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Created 7 editable draw.io pages, 7 PNG previews, overview.png and validation.json.')
