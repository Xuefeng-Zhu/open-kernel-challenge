"""Admission safety tests with a fake CLI and temporary files; no network/GPU."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_cycle as runner


class CycleLimits(unittest.TestCase):
    def exercise(self, n, maximum, final=False, active=False, uncertain=False, reject=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            slot = root/'slot.json'
            slot.write_text(json.dumps(dict(state='running' if active else 'terminal',
                                           problem='muse_attention_decode', experiment='older')))
            cycle2 = maximum == 10
            ledger = root/('attention-cycle2-budget.json' if cycle2 else 'attention-cycle-budget.json')
            requests = [dict(experiment=f'old{i}', state='terminal', status='done') for i in range(n)]
            if uncertain:
                requests[0].update(state='uncertain', status='unknown')
            raw = b'# okc: problem=muse_attention_decode\ndef kernel_fn(data):\n    return data\n'
            (root/'candidate.py').write_bytes(raw)
            import hashlib
            data = dict(maximum_new_checks=maximum, reserved_final_checks=1, requests=requests,
                        max_active_jobs=1, rules_version='test', plans=dict(X=dict(candidate='candidate.py',
                        expected_sha256=hashlib.sha256(raw).hexdigest(), final=final, hypothesis='fake test')))
            ledger.write_text(json.dumps(data))
            cli_result = SimpleNamespace(returncode=0, stdout='{"job_id":"fake-accepted"}', stderr='')
            quota = dict(checks=dict(left=19), scored=dict(used=0))
            args = ['check_cycle.py', 'send', 'X']+(['--cycle2'] if cycle2 else [])
            with patch.multiple(runner, HERE=root, PROJECT=root, LEDGER=ledger, GPU_SLOT=slot), \
                 patch('sys.argv', args), \
                 patch.object(runner, 'quota', return_value=quota), \
                 patch.object(runner, 'cli', return_value=(cli_result, dict(job_id='fake-accepted'))) as cli, \
                 patch.object(runner.subprocess, 'check_output', side_effect=['fake-commit', raw]), \
                 patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='{"version":"test"}')), \
                 redirect_stdout(io.StringIO()):
                if reject:
                    with self.assertRaisesRegex(RuntimeError, reject):
                        runner.main()
                    cli.assert_not_called()
                    self.assertEqual(len(json.loads(ledger.read_text())['requests']), n)
                else:
                    runner.main()
                    self.assertEqual(cli.call_count, 1)
                    saved = json.loads(ledger.read_text())
                    self.assertEqual(saved['remaining'], maximum-n-1)
                    self.assertEqual(saved['requests'][-1]['job_id'], 'fake-accepted')

    def test_old_six_ceiling(self):
        self.exercise(6, 6, final=True, reject='exhausted')

    def test_new_ten_ceiling(self):
        self.exercise(10, 10, final=True, reject='exhausted')

    def test_new_budget_allows_seventh(self):
        self.exercise(6, 10)

    def test_last_new_check_reserved(self):
        self.exercise(9, 10, reject='reserved')

    def test_last_complete_candidate_allowed(self):
        self.exercise(9, 10, final=True)

    def test_cross_cycle_active_gpu_guard(self):
        self.exercise(0, 10, active=True, reject='slot not free')

    def test_uncertain_request_prevents_post(self):
        self.exercise(1, 10, uncertain=True, reject='Uncertain')


if __name__ == '__main__':
    unittest.main(verbosity=2)
