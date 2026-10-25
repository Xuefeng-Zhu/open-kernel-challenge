"""CPU orchestration/metadata tests only: no torch/CUDA/Triton imports required."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ownership_plan', HERE/'native_ownership_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeRunner:
    def __init__(self, fail=False):
        self.events, self.fail = [], fail
    def prepare(self, case, arm):
        self.events.append(('prepare', arm))
        return dict(arm=arm)
    def assert_isolation(self, states):
        assert states[0] is not states[1]
        self.events.append(('isolation',))
    def restore_and_launch(self, state, label):
        self.events.append(('launch', state['arm'], label))
    def wait_all(self, states):
        self.events.append(('wait',))
    def check(self, state, label, cycle):
        self.events.append(('check', state['arm'], label))
        return dict(arm=state['arm'], label=label, cycle=cycle, passed=not self.fail)
    def profile_pair(self, states, label):
        self.events.append(('profile_pair', label))
        for state in states:
            self.restore_and_launch(state, label)
        self.wait_all(states)
        return dict(cross_graph_overlap_observed=False, evidence='fake_unverified')


class OwnershipProbeOrchestration(unittest.TestCase):
    def test_two_launches_precede_every_wait_and_check(self):
        runner, emitted = FakeRunner(), []
        records = probe.run_case(probe.CASES[0], runner, 2, lambda: None, emitted.append)
        self.assertEqual(len(records), 12)
        self.assertEqual(records, emitted)
        self.assertEqual(runner.events[:3], [('prepare',0),('prepare',1),('isolation',)])
        at = 3
        for _ in range(2):
            for label in ['A1','B','A2']:
                self.assertEqual(runner.events[at:at+5],
                    [('launch',0,label),('launch',1,label),('wait',),('check',0,label),('check',1,label)])
                at += 5

    def test_failure_and_deadline_stop_without_more_launches(self):
        runner, emitted = FakeRunner(fail=True), []
        with self.assertRaises(AssertionError):
            probe.run_case(probe.CASES[0], runner, 2, lambda: None, emitted.append)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(sum(e[0]=='launch' for e in runner.events), 2)
        runner = FakeRunner()
        def expired():
            raise TimeoutError('test deadline')
        with self.assertRaises(TimeoutError):
            probe.run_case(probe.CASES[0], runner, 1, expired, lambda _: None)
        self.assertEqual(runner.events, [])

    def test_dry_run_needs_no_torch_cuda_or_triton_import(self):
        script = (
            'import builtins,runpy,sys\n'
            'original=builtins.__import__\n'
            'def guarded(name,*args,**kwargs):\n'
            '    if name.split(".")[0] in {"torch","triton"}:\n'
            '        raise AssertionError("CUDA dependency imported during dry run")\n'
            '    return original(name,*args,**kwargs)\n'
            'builtins.__import__=guarded\n'
            'sys.argv=[sys.argv[1],"--list-cases"]\n'
            'runpy.run_path(sys.argv[0],run_name="__main__")\n')
        result = subprocess.run([sys.executable, '-B', '-c', script,
                                 str(HERE/'native_ownership_probe.py')],
                                capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['checked_replays'], 30)
        self.assertFalse(plan['same_graph_overlap'])
        self.assertFalse(plan['physical_kernel_overlap_measured'])
        # The guarded subprocess proves import isolation. Other tests in a
        # shared discovery process may legitimately have imported torch already.
        self.assertEqual({c['name'] for c in plan['cases']},
                         {'sliding_decode1','sliding_decode16','sliding_decode32','full_decode32',
                          'nearby_full_page64_nonunit'})

    def test_native_arguments_required_before_any_import(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            probe.main([])
        self.assertEqual(caught.exception.code, 2)
        for filename in ['native_ownership_probe.py','test_native_ownership_probe.py']:
            compile((HERE/filename).read_bytes(), filename, 'exec')

    def test_existing_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'evidence.json'
            output.write_text('preserve this evidence')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                probe.main(['--candidate', str(HERE/'native_ownership_probe.py'),
                            '--expected-sha', '0'*64, '--output', str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.read_text(), 'preserve this evidence')

    def test_bounded_plan_and_profile_do_not_add_replays(self):
        plan = probe.plan(1, True)
        self.assertEqual(len(plan['cases']), 7)
        self.assertEqual(plan['checked_replays'], 42)
        self.assertEqual(plan['cases'][0]['q_len'], 8)
        self.assertEqual(plan['cases'][0]['batch'], 2)
        self.assertEqual(plan['cases'][2]['batch'], 1)
        self.assertEqual(plan['cases'][2]['q_len'], 8)
        self.assertEqual(next(c['ctx'] for c in plan['cases'] if c['name']=='full_decode32'), 8192)
        runner, profiles = FakeRunner(), []
        records = probe.run_case(plan['cases'][0], runner, 1, lambda: None,
                                 lambda _: None, profiles.append)
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(e[0]=='launch' for e in runner.events), 6)
        self.assertEqual(sum(e[0]=='profile_pair' for e in runner.events), 1)
        self.assertFalse(profiles[0]['cross_graph_overlap_observed'])

    def test_gpu_interval_summary_requires_distinct_graphs_same_device(self):
        def kernel(start, duration, stream, graph, device=0):
            return dict(ph='X', cat='kernel', name='test_kernel', ts=start, dur=duration,
                        args={'stream':stream, 'graph id':graph, 'device':device})
        events = [kernel(0,10,1,100), kernel(3,4,2,200), kernel(10,2,3,300),
                  kernel(0,10,4,400,device=1)]
        result = probe.summarize_trace({'traceEvents':events})
        self.assertTrue(result['cross_graph_overlap_observed'])
        self.assertEqual(result['graph_overlap']['overlap_us'], 4.)
        self.assertEqual(result['stream_overlap']['max_concurrent'], 2)
        # Internal streams of ONE graph are insufficient cross-graph evidence.
        result = probe.summarize_trace({'traceEvents':[kernel(0,10,1,100),kernel(3,4,2,100)]})
        self.assertTrue(result['physical_kernel_overlap_observed'])
        self.assertFalse(result['cross_graph_overlap_observed'])
        # Touching endpoints, copies, malformed data and duplicate records cannot
        # create positive overlap evidence.
        good = kernel(0,10,1,None)
        bad = kernel(float('nan'),10,3,300)
        copy = {**kernel(1,8,2,200), 'cat':'gpu_memcpy'}
        result = probe.summarize_trace({'traceEvents':[good,good,bad,copy,kernel(10,2,2,None)]})
        self.assertEqual(result['kernel_count'], 2)
        self.assertFalse(result['physical_kernel_overlap_observed'])
        self.assertFalse(result['cross_graph_overlap_observed'])

    def test_original_stream_is_preferred_to_display_lane(self):
        events = [dict(ph='X',cat='kernel',name='k',ts=0,dur=10,
                       args={'device':0,'stream':lane,'original_stream':7,'graph id':9})
                  for lane in [1,2]]
        result = probe.summarize_trace({'traceEvents':events})
        self.assertFalse(result['physical_kernel_overlap_observed'])
        self.assertFalse(result['cross_graph_overlap_observed'])

    def test_compact_records_preserve_exact_values_and_default_schema(self):
        record = dict(case='full_decode32', arm=0, label='B', actual_seq_lens=list(range(32)),
                      passed=True, oracle_message='exact oracle details')
        compact = dict(plan=probe.plan(1,True),records=[])
        probe.append_record(compact, record)
        self.assertEqual(dict(zip(compact['record_columns'],compact['records'][0])), record)
        original = dict(plan=probe.plan(1),records=[])
        probe.append_record(original, record)
        self.assertEqual(original['records'],[record])
        with self.assertRaises(ValueError):
            probe.append_record(compact, {'different':'schema'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
