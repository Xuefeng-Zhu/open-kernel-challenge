import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tiny_capability_worker', HERE/'worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class Helper:
    @staticmethod
    def metric_values(text):
        return [123.] if 'ONE_METRIC' in text else []


class Tests(unittest.TestCase):
    def command(self, **overrides):
        text = '\n'.join('TINY_PROGRESS_JSON='+json.dumps({'phase': p, 'passed': True})
            for p in ('pre', 'measured', 'post'))+'\nONE_METRIC'
        return {'stdout': text, 'stderr': '', 'state': 'complete', 'returncode': 0, **overrides}

    def test_only_middle_matched_eager_kernel_profiled(self):
        args = w.profile_argv('/ncu', Path('/payload'), 'sm__cycles_elapsed.avg')
        self.assertEqual(args[args.index('--launch-skip')+1], '1')
        self.assertEqual(args[args.index('--launch-count')+1], '1')
        self.assertEqual(args[args.index('--replay-mode')+1], 'kernel')
        self.assertNotIn('--nvtx', args)
        self.assertNotIn('selected', ' '.join(args))
        self.assertEqual(args[-1], '--tiny-child')

    def test_complete_gate(self):
        self.assertTrue(w.classify(self.command(), Helper)['passed'])

    def test_timeout_not_permission_denial(self):
        d = w.classify(self.command(state='timeout', returncode=-9), Helper)
        self.assertFalse(d['passed'])
        self.assertFalse(d['permission_denial_observed'])

    def test_actual_permission_error_preserved(self):
        d = w.classify(self.command(state='nonzero_exit', returncode=1,
            stderr='ERR_NVGPUCTRPERM'), Helper)
        self.assertTrue(d['permission_denial_observed'])
        self.assertFalse(d['passed'])

    def test_missing_counter_or_postcheck_not_success(self):
        self.assertFalse(w.classify(self.command(stdout=''), Helper)['passed'])
        text = self.command()['stdout'].replace('"phase": "post"', '"phase": "missing"')
        self.assertFalse(w.classify(self.command(stdout=text), Helper)['passed'])


if __name__ == '__main__':
    unittest.main()
