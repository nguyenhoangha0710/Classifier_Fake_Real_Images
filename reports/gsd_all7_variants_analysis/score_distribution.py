"""Read-only score statistics; no calibration, threshold fitting or image inference.

Compute with the bundled scientific runtime. Plot from the JSON statistics with
Matplotlib (a plotting-only fallback when that library is absent from the bundle).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {
    'native': 'Native', 'png_roundtrip': 'PNG lossless',
    'fake_jpeg96': 'Chỉ fake JPEG96', 'all_jpeg96': 'Real + fake JPEG96',
    'all_jpeg90': 'Real + fake JPEG90', 'all_jpeg80': 'Real + fake JPEG80',
    'resize_square256': 'Resize256', 'resize_square560': 'Resize560',
    'blur_sigma1': 'Blur σ=1', 'resize256_jpeg96': 'Resize256 + JPEG96',
}


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compute():
    import numpy as np
    import pandas as pd
    source = HERE/'source/predictions.csv'; original_sha = file_sha(source)
    frame = pd.read_csv(source, float_precision='round_trip')
    saved = pd.read_csv(HERE/'source/metrics.csv')
    native = frame[frame['case'] == 'native'].reset_index(drop=True)
    assert len(frame) == 70000 and len(native) == 7000
    assert set(frame['case']) == set(LABELS)
    groups = {}
    quantiles = dict(q01=.01, q05=.05, q25=.25, median=.5, q75=.75, q95=.95, q99=.99)
    for case, g in frame.groupby('case', sort=False):
        g = g.reset_index(drop=True)
        assert len(g) == 7000 and g.sample_id.is_unique
        for column in ('sample_id', 'label', 'generator', 'batch_id'):
            assert g[column].tolist() == native[column].tolist()
        reference = saved[(saved['case'] == case) & (saved.generator == 'pooled')].iloc[0]
        groups[case] = {}
        for label in (0, 1):
            scores = g.loc[g.label == label, 'prob_fake']
            assert len(scores) == 3500 and np.isfinite(scores).all() and scores.between(0, 1).all()
            positive = int((scores >= .5).sum())
            expected = int(reference.fp if label == 0 else reference.tp)
            assert positive == expected
            stats = {name: float(scores.quantile(q)) for name, q in quantiles.items()}
            stats.update(n=len(scores), min=float(scores.min()), max=float(scores.max()),
                mean=float(scores.mean()), predicted_fake_at_0_5=positive,
                predicted_real_at_0_5=len(scores)-positive,
                misclassified_at_0_5=positive if label == 0 else len(scores)-positive)
            assert all(stats[a] <= stats[b] for a, b in zip(quantiles, list(quantiles)[1:]))
            groups[case][str(label)] = stats
    assert file_sha(source) == original_sha
    result = dict(score_definition='P(fake) after two-class softmax for EVERY image, not P(real) for real images',
        label_definition={'0': 'ground-truth real', '1': 'ground-truth fake'},
        source=str(source), source_sha256=original_sha, source_unchanged=True,
        calibration_applied=False, decision_threshold=.5, n_predictions=len(frame),
        quantiles='Empirical quantiles, pandas linear interpolation. Ranges are data intervals, not confidence intervals.',
        cases=groups)
    (HERE/'score_distribution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Phân bố score GSD trên 10 case',
        '**Mọi score trong báo cáo đều là P(fake), kể cả score của ảnh thật.** '
        'Không dùng calibration. 3.500 ảnh thật và 3.500 ảnh giả mỗi case. '
        'P(real) của cùng ảnh = 1 − P(fake). Hai trung vị lấy từ hai nhóm ảnh khác nhau không cần cộng bằng 1.',
        'Nguồn: '+str(source), 'SHA-256: `'+original_sha+'`',
        '## Trung vị và lỗi ở ngưỡng 0,5',
        '| Case | Trung vị P(fake) trên ảnh thật | Trung vị P(fake) trên ảnh giả | Ảnh thật bị báo giả (%) | Ảnh giả bị bỏ sót (%) |',
        '|---|---:|---:|---:|---:|']
    for case, group in groups.items():
        real, fake = group['0'], group['1']
        row = (f"| {LABELS[case]} | {real['median']:.8f} | {fake['median']:.8f} | "
               f"{100*real['misclassified_at_0_5']/real['n']:.2f} | "
               f"{100*fake['misclassified_at_0_5']/fake['n']:.2f} |")
        lines.append(row)
        print(row)
    lines.extend(['', '## Khoảng phân bố, không phải khoảng tin cậy',
        'Q25–Q75 chứa khoảng 50% ảnh ở giữa phân bố; Q05–Q95 chứa khoảng 90% ảnh ở giữa. '
        'Các vùng này không bao phủ toàn bộ ảnh hoặc toàn bộ ảnh bị phân loại sai.',
        '| Case | Nhóm ảnh theo nhãn thật | Q05 | Q25 | Trung vị | Q75 | Q95 | Trung bình |',
        '|---|---|---:|---:|---:|---:|---:|---:|'])
    for case, group in groups.items():
        for label in ('0', '1'):
            r = group[label]
            lines.append('| '+LABELS[case]+' | '+('Ảnh thật' if label == '0' else 'Ảnh giả')+' | '+
                         ' | '.join(f'{r[k]:.9g}' for k in ('q05','q25','median','q75','q95','mean'))+' |')
    lines.extend(['', '## Cách đọc biểu đồ',
        'Màu xanh: ảnh thật theo nhãn gốc; màu cam: ảnh giả theo nhãn gốc. '
        'Hộp là Q25–Q75, vạch/điểm là trung vị, râu là Q05–Q95; không vẽ 10% ảnh ở hai đuôi. '
        'Trục ngang dùng thang logit để nhìn rõ score rất nhỏ và rất gần 1; nhãn trục vẫn là P(fake). '
        'Vạch đứng ở 0,5: bên phải bị model nhận là fake, bên trái bị nhận là real.',
        '## Diễn giải',
        'Native/PNG: ảnh thật tập trung gần 0, ảnh giả gần 1. PNG lossless giữ nguyên prediction. '
        'JPEG96: score ảnh giả trải rất rộng và dịch mạnh về 0; score ảnh thật gần như giữ nguyên. '
        'JPEG90/80: hai phân bố chồng lấn nhiều hơn. Resize256/blur: một phần score ảnh thật tăng mạnh, '
        'dù trung vị vẫn dưới 0,5; trung vị một mình không mô tả các đuôi gây báo nhầm. '
        'Resize256+JPEG96 làm score fake thấp và có thêm chồng lấn với real.'])
    (HERE/'score_distribution_report.md').write_text('\n\n'.join(lines[:6])+'\n'+
        '\n'.join(lines[6:])+'\n', encoding='utf-8')
    return result


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    result = json.loads((HERE/'score_distribution.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    colors = {'0': '#2476ad', '1': '#d77a25'}
    fig, ax = plt.subplots(figsize=(9, 7.8))
    for i, (case, groups) in enumerate(result['cases'].items()):
        for label, offset in [('0', -.17), ('1', .17)]:
            s = groups[label]
            statistics = dict(label='', whislo=s['q05'], q1=s['q25'], med=s['median'],
                q3=s['q75'], whishi=s['q95'], fliers=[])
            c = colors[label]
            ax.bxp([statistics], positions=[i+offset], vert=False, widths=.24,
                showfliers=False, manage_ticks=False, patch_artist=True,
                boxprops=dict(facecolor=c, alpha=.27, edgecolor=c, linewidth=1),
                medianprops=dict(color=c, linewidth=2.3), whiskerprops=dict(color=c, linewidth=1.2),
                capprops=dict(color=c, linewidth=1.2))
            ax.scatter(s['median'], i+offset, s=15, color=c, zorder=4)
        if i < len(result['cases'])-1:
            ax.axhline(i+.5, color='#e8e8e8', lw=.6, zorder=0)
    ax.set_xscale('logit')
    ax.set_xlim(.000005, .999995)
    ticks = [.00001, .001, .1, .5, .9, .999, .99999]
    ax.set_xticks(ticks, labels=['0,00001', '0,001', '0,1', '0,5', '0,9', '0,999', '0,99999'])
    ax.minorticks_off()
    ax.axvline(.5, color='#585858', lw=1.3, ls='--', zorder=0)
    ax.set_yticks(range(len(result['cases'])), labels=[LABELS[c] for c in result['cases']])
    ax.set_ylim(len(result['cases'])-.45, -.55)
    ax.set_xlabel('Score P(fake) — thang logit, không phải trục tuyến tính', labelpad=12)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#bbbbbb')
    legend = [Patch(facecolor=colors['0'], alpha=.4, label='Ảnh thật (nhãn 0)'),
              Patch(facecolor=colors['1'], alpha=.4, label='Ảnh giả (nhãn 1)'),
              Line2D([0],[0], color='#585858', ls='--', label='Ngưỡng 0,5')]
    ax.legend(handles=legend, loc='lower left', bbox_to_anchor=(-.01, 1.015),
              ncol=3, frameon=False, fontsize=10, columnspacing=1.5)
    fig.suptitle('Phân bố P(fake) của ảnh thật và ảnh giả', x=.035, ha='left', y=.98, fontsize=15)
    fig.text(.035,.025,'Hộp: 50% ở giữa · Vạch/điểm: trung vị · Râu: 90% ở giữa\n'
             'Không vẽ 10% ảnh ở hai đuôi; số liệu lỗi trong bảng dùng toàn bộ 7.000 ảnh/case.',
             ha='left', fontsize=10, color='#444444')
    fig.subplots_adjust(left=.25, right=.97, top=.88, bottom=.15)
    target = HERE/'score_distribution.png'
    fig.savefig(target, dpi=170, facecolor='white')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.plot:
        plot()
    else:
        compute()
