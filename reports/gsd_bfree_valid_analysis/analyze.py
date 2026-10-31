"""Recompute B-Free metrics and cohort checks, leaving downloaded sources intact."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source' / 'gsd_bfree_valid_v1'
spec = importlib.util.spec_from_file_location('metric_helpers', HERE.parent / 'gsd_augmented_native_analysis' / 'analyze.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    frame = pd.read_csv(SOURCE / 'predictions.csv', dtype={'coco_id': str})
    reported = pd.read_csv(SOURCE / 'metrics.csv')
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv').iloc[0]
    state = helpers.read_json(SOURCE / 'completion.json')
    fingerprint = helpers.read_json(SOURCE / 'run_fingerprint.json')
    ids_table = pd.read_csv(SOURCE / 'bfree_valid_ids.csv', dtype=str)
    ids = ids_table['id'].tolist()
    checks = dict(rows=len(frame), unique_images=int(frame[['coco_id', 'variant']].drop_duplicates().shape[0]),
                  unique_coco_ids=int(frame.coco_id.nunique()), unique_context_rows=int(frame[['case', 'sample_id']].drop_duplicates().shape[0]),
                  status=state['status'], smoke=state['smoke'], threshold=state['threshold'],
                  finite=bool(np.isfinite(frame[['prob_fake', 'logit_real', 'logit_fake', 'logit_margin']]).all().all()),
                  valid_probabilities=bool(frame.prob_fake.between(0, 1).all()),
                  native512=bool(frame.width.eq(512).all() and frame.height.eq(512).all()),
                  tokens1370=bool(frame.input_tokens.eq(1370).all()),
                  id_order_hash_matches=helpers.digest(ids) == fingerprint['id_order_hash'],
                  fingerprint_matches=helpers.digest(fingerprint) == state['fingerprint'],
                  every_branch_full=True, max_reported_metric_difference=0., branches={})
    margin = frame.logit_fake.to_numpy() - frame.logit_real.to_numpy()
    checks['max_softmax_difference'] = float(np.max(np.abs(frame.prob_fake - 1/(1+np.exp(-margin)))))
    checks['max_margin_difference'] = float(np.max(np.abs(frame.logit_margin-margin)))
    rows, distributions = [], {}
    for branch in fingerprint['branches']:
        group = frame[frame.case == branch]
        real, fake = group[group.label == 0], group[group.label == 1]
        branch_checks = dict(n=len(group), real=len(real), fake=len(fake),
            ids_real_match=real.coco_id.tolist() == ids, ids_fake_match=fake.coco_id.tolist() == ids,
            alternates_real_fake=group.label.tolist() == [0, 1]*len(ids),
            batch_ids_match=bool(np.array_equal(group.batch_id.to_numpy(), np.arange(len(group))//8)),
            fake_variant_matches=bool(fake.variant.eq(branch).all()), real_variant_matches=bool(real.variant.eq('COCO_real_512').all()),
            generator_context_matches=bool(group.generator.eq(branch).all()))
        checks['branches'][branch] = branch_checks
        checks['every_branch_full'] &= all([len(group) == 20608, len(real) == 10304, len(fake) == 10304,
            *[branch_checks[k] for k in branch_checks if k.endswith('_match') or k.endswith('_matches')],
            branch_checks['alternates_real_fake']])
        row = dict(branch=branch, **helpers.metrics(group))
        old = reported[reported.branch == branch].iloc[0]
        for key in reported.columns:
            if key in row and key not in ('branch', 'complete'):
                checks['max_reported_metric_difference'] = max(checks['max_reported_metric_difference'], abs(float(old[key])-row[key]))
        rows.append(row)
        distributions[branch] = {label: dict(n=len(g), mean=float(g.prob_fake.mean()),
            p05=float(g.prob_fake.quantile(.05)), median=float(g.prob_fake.median()), p95=float(g.prob_fake.quantile(.95)),
            median_logit_margin=float(g.logit_margin.median())) for label, g in [('real', real), ('fake', fake)]}
    results = pd.DataFrame(rows)
    fields = ['accuracy', 'balanced_accuracy', 'real_recall', 'fake_recall', 'fake_precision', 'fake_f1', 'auc', 'ap']
    computed_macro = {key: float(results[key].mean()) for key in fields}
    checks['max_macro_difference'] = max(abs(computed_macro[key]-float(macro[key])) for key in fields)
    subgroup = {name: {key:float(results[results.branch.str.endswith('_origBG') == orig][key].mean()) for key in fields}
                for name, orig in [('without_origBG', False), ('origBG', True)]}
    real_scores = frame[frame.label == 0].pivot(index='coco_id', columns='case', values='prob_fake')
    real_flips = real_scores.ge(.5).nunique(axis=1).gt(1)
    checks['real_ids_with_branch_dependent_classification'] = int(real_flips.sum())
    checks['real_score_range_median'] = float((real_scores.max(axis=1)-real_scores.min(axis=1)).median())
    result = dict(checks=checks, metrics=rows, macro=computed_macro, subgroup_macro=subgroup,
        confusion_sum={key:int(results[key].sum()) for key in ['tn','fp','fn','tp']},
        score_distribution=distributions, elapsed_hours=state['elapsed_seconds']/3600)
    (HERE / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
