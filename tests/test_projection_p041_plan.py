"""Frozen receipt/traffic-model validation; never executes CUDA kernels."""
import math
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import projection_p041_plan as plan


class ProjectionEvidence(unittest.TestCase):
    def setUp(self):
        # This historical P039 analysis must not depend on today's P046 selection.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name in (plan.SOURCE, plan.RECEIPT, plan.CSV, "official/reference.py", "TASK.md"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(plan.ROOT / name, target)
        (root / "experiments").mkdir()
        self.ledger = root / "experiments/projection-cycle2-budget.json"
        self.ledger.write_text(json.dumps({"selection": {"sha256": plan.SOURCE_SHA256}}))
        scope = patch.object(plan, "ROOT", root)
        scope.start()
        self.addCleanup(scope.stop)

    def test_changed_selection_still_blocks_historical_plan(self):
        self.ledger.write_text(json.dumps({"selection": {"sha256": "different-selection"}}))
        with self.assertRaisesRegex(ValueError, "Selection changed"):
            plan.build()

    def test_frozen_score_and_parent(self):
        report = plan.build()
        self.assertAlmostEqual(report['score'], 1.173238998969285, places=12)
        self.assertEqual(report['parent']['job'], 'd5636ac8d1404abd')
        self.assertEqual(len(report['six_weighted_loads']), 6)
        self.assertEqual(report['prerequisites'], {'passed': 9, 'total': 9})
        self.assertEqual(report['shapes'], {'passed': 30, 'total': 30})
        self.assertFalse(report['gpu_launched'])

    def test_family_fractions_are_per_load(self):
        fractions = plan.build()['mean_within_load_family_fraction']
        self.assertAlmostEqual(sum(fractions.values()), 1., places=12)
        self.assertAlmostEqual(fractions['pooler'], .3261557, places=6)

    def test_weight_scaling_cancels_in_ratio(self):
        rows = plan.read_rows()
        one = [{**r, 'weight': 1} for r in rows]
        self.assertAlmostEqual(plan.score(rows)[0], plan.score(one)[0], places=12)
        self.assertEqual({r['weight'] for r in rows}, {62})

    def test_multi_load_value_not_single_shape(self):
        rows = plan.read_rows()
        broad = plan.sensitivity(rows, 'pooler', .1, 2)
        narrow = plan.sensitivity(rows, 'pooler', .1, 2, 256)
        self.assertGreater(broad, 4*narrow)
        self.assertAlmostEqual(broad, .026638, places=5)
        self.assertEqual(plan.sensitivity(rows, 'pooler', 0), 0)
        with self.assertRaises(ValueError):
            plan.sensitivity(rows, 'pooler', 1)

    def test_compulsory_bytes_and_unpool_boundary(self):
        pool = plan.compulsory_model('pooler', 128)
        self.assertEqual(pool['weights_bytes'], 64*1024**2)
        self.assertEqual(pool['flops'], 2*128*4096*8192)
        self.assertEqual(plan.compulsory_model('qkv_gate', 128)['weights_bytes'], 36*1024**2)
        self.assertEqual(plan.compulsory_model('unpooler', 8)['n'], 1024)
        self.assertEqual(plan.compulsory_model('unpooler', 9)['n'], 8192)
        self.assertEqual(plan.compulsory_model('router', 1)['weights_bytes'], 8*1024**2)

    def test_source_change_blocks_diagnostic_identity(self):
        with patch.object(plan, 'SOURCE_SHA256', 'wrong'), self.assertRaisesRegex(ValueError, 'source changed'):
            plan.build()


if __name__ == '__main__':
    unittest.main()
