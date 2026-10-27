"""Meaningful offline identities, reachability and target-feasibility checks."""
import importlib.util
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('attention_gap_model', HERE/'gap_model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class GapArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = model.make_model(HERE.parents[3]/'attention-public-leaderboard-20261025.local.json')
        cls.times = {(r['load'], r['layer']):r['current_us'] for r in cls.result['path_inputs']}
        cls.reference = {(r['load'], r['layer']):r['fixed_reference_us'] for r in cls.result['path_inputs']}

    def test_measured_score_reconstruction_and_homogeneous_scaling(self):
        self.assertAlmostEqual(model.score(self.times, self.reference), 1.2336330588627955, places=12)
        for keep in (.25, .5, .75):
            score = model.score(model.scaled(self.times, model.PATHS, 1-keep), self.reference)
            self.assertAlmostEqual(score*keep, self.result['measured_current']['score'], places=12)
        cut = self.result['all_paths_uniform_cut_needed_percent']/100
        self.assertAlmostEqual(model.score(model.scaled(self.times, model.PATHS, cut), self.reference),
                               self.result['public_summary']['leader_speedup_public_rounded'], places=12)

    def test_group_root_and_frozen_other_path_unattainability(self):
        target = self.result['public_summary']['leader_speedup_public_rounded']
        for name in ['three_large_FA3_full', 'two_full_verify']:
            group = self.result['group_models'][name]
            self.assertFalse(group['target_reachable_without_floor_constraint'])
            self.assertIsNone(group['uniform_cut_needed_percent'])
            self.assertLess(group['zero_selected_time_conditional_limit'], target)
        for name in ['four_large_full', 'all_six_full']:
            group = self.result['group_models'][name]
            cut = group['uniform_cut_needed_percent']/100
            self.assertAlmostEqual(model.score(model.scaled(self.times, model.GROUPS[name], cut), self.reference), target, places=12)
            self.assertEqual({row['load'] for row in group['required_cut_task_floor_violations']},
                             {'decode 16','decode 32','verify 16','verify 32'})

    def test_broad_scenario_and_closed_form_required_B1(self):
        current = self.result['measured_current']['score']
        broad = self.result['broad_engineering_scenario']
        self.assertAlmostEqual(broad['score'], current*(2**4*(1/.3)**2)**(1/6), places=12)
        self.assertGreater(broad['score'], self.result['public_summary']['leader_speedup_public_rounded'])
        self.assertEqual(broad['task_floor_violations'], [])
        required = self.result['required_B1_after_large_half']
        ratio = self.result['leader_over_current']
        self.assertAlmostEqual(required['cut_percent']/100, 1-4/ratio**3, places=12)
        self.assertAlmostEqual(required['score'], self.result['public_summary']['leader_speedup_public_rounded'], places=12)
        self.assertEqual(required['task_floor_violations'], [])

    def test_full_floor_scenario_still_needs_sliding_work(self):
        needed = self.result['required_sliding_after_full_TASK_floors']
        target = self.result['public_summary']['leader_speedup_public_rounded']
        self.assertLess(needed['full_floor_only_score'], target)
        self.assertGreater(needed['sliding_uniform_cut_percent'], 0)
        self.assertAlmostEqual(needed['score'], target, places=12)
        self.assertEqual(needed['task_floor_violations'], [])

    def test_public_summary_excludes_competitor_jobs_and_rank_inference(self):
        self.assertEqual(set(self.result['public_summary']), {'observed_utc','leader_team',
            'leader_speedup_public_rounded','ranked_count','our_rank'})
        self.assertIsNone(self.result['public_summary']['our_rank'])
        self.assertFalse(self.result['measured_current']['ranked_entry'])
        self.assertEqual(self.result['measured_current']['mode'], 'check')
        self.assertNotIn('rows', self.result)
        self.assertEqual(len(self.result['path_inputs']), 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
