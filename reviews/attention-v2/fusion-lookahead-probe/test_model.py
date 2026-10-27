"""Offline arithmetic/provenance checks; no CUDA or remote work."""
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import model


class CompositionModel(unittest.TestCase):
    def test_frozen_evidence_and_original_goal_remain_distinct(self):
        output = model.build()
        self.assertAlmostEqual(output['reconstructed_speedup'],1.2255554363764571)
        self.assertAlmostEqual(output['old_a215']['target_cut_percent'],5.189108000112796)
        self.assertFalse(output['old_a215']['original_goal_met'])
        self.assertAlmostEqual(output['transfer_old_relative_cut']['aggregate_gain_percent'],.7520171384544749)
        self.assertEqual(output['proposed_new_composition_gate']['median_paired_target_cut_percent'],4)
        self.assertFalse(output['achieved_composition_score'])

    def test_zero_signed_regression_and_monotonic_gain(self):
        share = .8470980929400972
        self.assertEqual(model.score_factor(share,0),1)
        self.assertLess(model.score_factor(share,-.05),1)
        self.assertLess(model.score_factor(share,.03),model.score_factor(share,.04))
        self.assertAlmostEqual(100*(model.score_factor(share,.04)-1),.5761750433444401)

    def test_inverse_and_invalid_cost(self):
        share = .8470980929400972
        for gain in [.005,.01,.015]:
            cut = model.required_cut(share,gain)
            self.assertAlmostEqual(model.score_factor(share,cut)-1,gain)
        for share,cut in [(0,.1),(1,.1),(.8,math.inf),(.8,1.25)]:
            with self.assertRaises(ValueError): model.score_factor(share,cut)

    def test_changed_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in model.EVIDENCE_HASHES:
                file=root/name;file.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(model.V2/name,file)
            file=root/'mixed_fa3_page_lookahead.py'
            file.write_bytes(file.read_bytes()+b'\n')
            with self.assertRaises(ValueError): model.evidence(root)


if __name__ == '__main__': unittest.main(verbosity=2)
