"""Render matched BFree metrics with metric/model two-level column headers."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
GROUPS=[('Kết quả tổng thể',[('balanced_accuracy','Acc./BA'),('auc','AUC'),('ap','AP')]),
        ('Recall',[('fake_recall','Recall fake'),('real_recall','Recall real')]),
        ('Precision và F1',[('fake_precision','Precision fake'),('real_precision','Precision real'),('fake_f1','F1 fake'),('real_f1','F1 real')])]
GROUPS.insert(0,('Bảng gộp đủ metrics',[metric for _,metrics in GROUPS for metric in metrics]))
MODELS=['CAPD train BFree','GSD train BFree']

def main():
    d=json.loads((ROOT/'metrics_comparison.json').read_text(encoding='utf8'))
    cases=list(d['bfree_branch_metrics'][MODELS[0]])
    rows=[]
    for i,c in enumerate(cases,1):rows.append((f'{i}. '+c.replace('SD2.1_',''),{m:d['bfree_branch_metrics'][m][c] for m in MODELS},'train' if i==4 else ''))
    rows.append(('6. inpainted_diffcat_origBG *',{m:d['bfree_partial7024'][m] for m in MODELS},'partial'))
    rows.append(('Mean case 1–5',{m:d['bfree_common_five'][m]['mean'] for m in MODELS},'mean'))
    note='Train BFree → Test BFree · ngưỡng 0,5 · CAPD full pooling · đơn vị %. Case 1–5: 10.304 ID/lớp. Case 6*: cùng 7.024 ID/lớp, còn partial; không đưa vào Mean. Cả hai train nhánh case 4.'
    md=['# So sánh TRAIN BFree → TEST BFree', '',note,'']
    tables=[]
    for title,metrics in GROUPS:
        md+=['## '+title,'',
             '| Case | '+' | '.join(f'**{label}** | ' for _,label in metrics)+' |',
             '|---|'+'---:|'*(2*len(metrics)),
             '| | '+' | '.join('CAPD | GSD' for _ in metrics)+' |']
        th='<tr><th rowspan="2" class="case">Case</th>'+''.join(f'<th colspan="2">{html.escape(label)}</th>' for _,label in metrics)+'</tr>'
        th+='<tr>'+''.join('<th>CAPD</th><th>GSD</th>' for _ in metrics)+'</tr>'
        body=''
        for label,values,kind in rows:
            vals=[f'{100*values[m][key]:.2f}' for key,_ in metrics for m in MODELS]
            md.append('| '+label+' | '+' | '.join(vals)+' |')
            cells=[]
            for key,_ in metrics:
                a,b=[values[m][key] for m in MODELS]
                for m in MODELS:
                    v=values[m][key]
                    cls=' winner' if v>min(a,b)+1e-12 else ''
                    cells.append(f'<td class="{cls.strip()}">{100*v:.2f}</td>')
            body+=f'<tr class="{kind}"><th class="case">{html.escape(label)}</th>'+''.join(cells)+'</tr>'
        md+=['']
        tables.append(f'<section><h2>{html.escape(title)}</h2><div class="scroll"><table><thead>{th}</thead><tbody>{body}</tbody></table></div></section>')
    md+=['Case 4 là nhánh cùng kiểu với training. Màu nhấn trong bản HTML chỉ giá trị cao hơn trong cặp, không phải kết luận model thắng mọi mặt. AUC/AP 100,00 có thể do làm tròn.','']
    (ROOT/'BFREE_GROUPED_TABLES.md').write_text('\n'.join(md),encoding='utf8')
    css='''body{font:15px/1.5 system-ui,sans-serif;margin:28px;color:#182435;background:#f5f7fa}main{max-width:1500px;margin:auto}h1{font-size:25px}h2{font-size:19px}section{background:white;padding:18px;border-radius:12px;margin:20px 0;border:1px solid #dce3ec}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px 13px;border:1px solid #dce3ec;text-align:right;white-space:nowrap}thead th{background:#183553;color:white;text-align:center}thead tr:nth-child(2) th{background:#31577c}.case{text-align:left;position:sticky;left:0;background:white;min-width:280px}thead .case{background:#183553}tbody tr:nth-child(even) td,tbody tr:nth-child(even) .case{background:#f4f7fb}.winner{font-weight:700;color:#075b40}.train .case{border-left:5px solid #3076c6}.partial .case{border-left:5px solid #d08b00}.mean th,.mean td{border-top:3px solid #183553;font-weight:bold}p{max-width:1100px}.legend{color:#526379;font-size:13px}'''
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAPD và GSD — BFree</title><style>'+css+'</style><main><h1>TRAIN BFree → TEST BFree</h1><p>'+html.escape(note)+'</p>'+''.join(tables)+'<p class="legend">Chữ xanh đậm: giá trị cao hơn trong từng cặp. Viền xanh: case cùng kiểu với training. Viền vàng: partial. AUC/AP 100,00 có thể do làm tròn. Không thay ngưỡng hoặc pipeline giữa các dòng.</p></main></html>'
    (ROOT/'BFREE_GROUPED_TABLES.html').write_text(page,encoding='utf8')
    print('Wrote BFREE_GROUPED_TABLES.html and BFREE_GROUPED_TABLES.md.')

if __name__=='__main__':main()
