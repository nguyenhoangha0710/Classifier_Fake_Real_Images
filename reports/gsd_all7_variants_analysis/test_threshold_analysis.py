import unittest
import numpy as np
from threshold_analysis import at_threshold, sweep, choose


class ThresholdTests(unittest.TestCase):
    def test_complete_tied_score_sweep_matches_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            y = np.r_[0, 1, rng.integers(0, 2, 18)]
            s = rng.choice([0., .0001, .1, .5, .9, 1.], len(y))
            curve = sweep(y, s)
            for row in curve.itertuples():
                independent = at_threshold(y, s, row.threshold)
                for field in independent:
                    self.assertAlmostEqual(independent[field], getattr(row, field), places=14)
            self.assertEqual(curve.iloc[0].tp+curve.iloc[0].fp, 0)
            self.assertEqual(curve.iloc[-1].tn+curve.iloc[-1].fn, 0)
            for objective in ('balanced_accuracy', 'fake_f1', 'fake_recall'):
                for fpr in (None, .01, .25):
                    chosen = choose(curve, objective, fpr)
                    eligible = curve if fpr is None else curve[curve.fpr <= fpr]
                    self.assertAlmostEqual(chosen[objective], eligible[objective].max(), places=14)
                    if fpr is not None:
                        self.assertLessEqual(chosen['fpr'], fpr+1e-14)

    def test_perfect_ranking_but_zero_recall_at_half(self):
        y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([.01, .02, .03, .1, .2, .3])
        self.assertEqual(at_threshold(y, s, .5)['fake_recall'], 0)
        self.assertEqual(choose(sweep(y, s), 'balanced_accuracy')['balanced_accuracy'], 1)

    def test_ties_are_indivisible_and_conservative_selection(self):
        curve = sweep([0, 1], [.5, .5])
        self.assertEqual(len(curve), 2)
        selected = choose(curve, 'balanced_accuracy')
        self.assertGreater(selected['threshold'], .5)
        self.assertEqual(selected['fp'], 0)
        self.assertEqual(choose(curve, 'fake_f1')['fake_f1'], 2/3)

    def test_bad_scores_and_missing_class_rejected(self):
        for y, s in [([0, 1], [0, np.nan]), ([0, 1], [-1, .2]), ([1, 1], [.2, .4])]:
            with self.assertRaises(ValueError):
                sweep(y, s)


if __name__ == '__main__':
    unittest.main()
