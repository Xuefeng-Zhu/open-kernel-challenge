"""Local arithmetic/provenance checks only; no CUDA or FA3 execution."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unsplit_score_model', HERE/'model.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ScoreModel(unittest.TestCase):
    def test_zero_saving_and_weighted_known_two_microseconds(self):
        costs = {16: 4218.297426832483, 32: 7382.8180127379965}
        self.assertEqual(model.score_factor(costs, {}), 1.)
        self.assertAlmostEqual(model.score_factor(costs, {16: 2., 32: 2.}), 1.0057817762412133, places=14)

    def test_gain_is_monotonic_and_signed_regression_is_not_clipped(self):
        costs = {16: 4218.297426832483, 32: 7382.8180127379965}
        values = [model.score_factor(costs, {16: x, 32: x}) for x in [-2., 0., 2., 4.]]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], 1.)
        self.assertGreater(model.score_factor(costs, {16: 2.}), model.score_factor(costs, {32: 2.}))

    def test_threshold_inversion_and_invalid_cost(self):
        costs = {16: 4218.297426832483, 32: 7382.8180127379965}
        saving = model.equal_saving_for_gain(costs, .005)
        self.assertTrue(math.isclose(model.score_factor(costs, {16: saving, 32: saving}), 1.005, rel_tol=1e-13))
        for value in [1000., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                model.score_factor(costs, {16: value})

    def test_provenance_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'changed.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                model.checked_bytes(path, model.OFFICIAL_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
