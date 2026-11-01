"""Focused train-BFree/test-BFree comparison with explicit matched-case coverage."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
KEYS=['balanced_accuracy','auc','ap','fake_recall','real_recall','fake_precision','real_precision','fake_f1','real_f1']
HEAD=['Case','Acc./BA','AUC','AP','Recall fake','Recall real','Precision fake','Precision real','F1 fake','F1 real']
MODELS=['CAPD train BFree','GSD train BFree']

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    d=json.loads((ROOT/'metrics_comparison.json').read_text(encoding='utf8'))
    cases=list(d['bfree_branch_metrics'][MODELS[0]])
    last='SD2.1_inpainted_diffcat_origBG'
    lines=['# TRAIN BFree → TEST BFree: so sánh rõ theo từng case', '',
           '- CAPD: DINOv3+LoRA+patch head, full pooling. GSD: CLIP ViT-L/14+GSD. Cả hai train real COCO + fake **SD2.1_inpainted_samecat_origBG (case4)**, checkpoint tốt nhất virtual epoch2.',
           '- Probability threshold0,5 cho cả hai; mọi chỉ số trong bảng là %. Accuracy=BA vì mỗi case cân bằng hai lớp.',
           '- Case1–5:10.304 ID/case =10.304 real +10.304 fake, cùng ID/nhãn giữa model.',
           '- Case6: GSD chỉ chạy7.024/10.304 ID. Lọc CAPD về đúng7.024 ID này:14.048 ảnh/model. Không dùng điểm CAPD full10.304 ID để so với GSD partial.',
           '- Mean5 là trung bình số học metrics case1–5; không gộp case6 partial vào macro và không gọi kết quả là đủ6 case.', '']
    rows_by_model={}
    for model in MODELS:
        rows=[]
        for i,case in enumerate(cases,1):
            r=d['bfree_branch_metrics'][model][case];assert r['n']==20608
            label=f"{i}. {case.replace('SD2.1_','')}"+(' **(nhánh train)**' if 'samecat_origBG' in case else '')
            rows.append([label]+[f'{100*r[k]:.2f}' for k in KEYS])
        r=d['bfree_partial7024'][model];assert r['n']==14048
        rows.append(['6. inpainted_diffcat_origBG **(partial matched)**']+[f'{100*r[k]:.2f}' for k in KEYS])
        rows.append(['**Mean case1–5**']+[f"{100*d['bfree_common_five'][model]['mean'][k]:.2f}" for k in KEYS])
        rows_by_model[model]=rows
        lines+=['## '+model,'',table(HEAD,rows),'']
    delta=[]
    for i,case in enumerate(cases+[last],1):
        a=d['bfree_branch_metrics'][MODELS[0]][case] if case!=last else d['bfree_partial7024'][MODELS[0]]
        b=d['bfree_branch_metrics'][MODELS[1]][case] if case!=last else d['bfree_partial7024'][MODELS[1]]
        delta.append([f'{i}. '+case.replace('SD2.1_','')]+[f'{100*(a[k]-b[k]):+.2f}' for k in ['balanced_accuracy','auc','fake_recall','real_recall']])
    lines+=['## Chênh lệch CAPD − GSD, đơn vị điểm phần trăm','',
            table(['Case','Δ BA','Δ AUC','Δ recall fake','Δ recall real'],delta),'',
            '## Nhận xét','',
            '- Case4 (cùng kiểu fake với training) cả hai gần99,7% BA. Không dùng riêng case4 để kết luận khả năng tổng quát.',
            '- CAPD hơn rõ ở case1 selfconditioned: BA +15,18 điểm, recall fake +30,39 điểm. Hai case inpainting không origBG (3/5) cũng có recall fake cao hơn GSD.',
            '- Case2 selfconditioned_origBG yếu ở cả hai: recall fake CAPD48,93%, GSD47,00%, đều bỏ sót hơn nửa fake. AUC GSD95,47% cao hơn CAPD93,22%, nên không nói CAPD thắng mọi chỉ số.',
            '- Case6 partial: CAPD99,59% BA, GSD98,92% trên cùng7.024 ID; chưa phải kết quả cuối của toàn nhánh.',
            '- Cùng5 case đủ: CAPD93,91% BA và88,14% recall fake; GSD89,66% BA và79,58% recall fake. Đây là kết quả ngưỡng0,5; chưa trộn kết quả tối ưu bằng nhãn evaluation.', '',
            '## File chạy','',f"- [bfree_case_report.py]({(ROOT/'bfree_case_report.py').as_posix()}) đọc JSON kết quả đã audit và lập bảng.",
            f"- [metrics_comparison.json]({(ROOT/'metrics_comparison.json').as_posix()}) chứa TN/FP/FN/TP và số liệu đầy đủ.", '']
    (ROOT/'TRAIN_BFREE_TEST_BFREE_CASES.md').write_text('\n'.join(lines),encoding='utf8')
    print('Wrote matched case tables for both models: 5 complete + 1 partial; macro excludes partial.')

if __name__=='__main__':main()
