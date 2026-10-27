"""CPU-only fixed-scope, timing-summary and native-attribution regressions."""
import ast
import copy
from contextlib import redirect_stderr
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


w = load(HERE/'worker.py', 'b32_locality_worker_test')
trace_helper = load(HERE/'trace_helpers.py', 'b32_locality_trace_test')


def samples(random=20., contiguous=10.):
    # Pure arithmetic fixtures, never represented as measured GPU results.
    return [dict(index=i, order=w.measured_order(i),
        raw_us={w.ARMS[0]:random, w.ARMS[1]:contiguous}) for i in range(12)]


def fixtures(first=None, second=None):
    return [dict(seed=seed, measured=copy.deepcopy(rows)) for seed, rows in
        zip(w.SEEDS, (samples() if first is None else first, samples() if second is None else second))]


def synthetic_trace():
    events = []
    labels = ['PROFILE_SELECTED_FULL_VERIFY_B32_'+arm.upper() for arm in w.ARMS]
    for index, label in enumerate(labels):
        external, correlation = 11+index, 101+index
        host_start, gpu_start = 100+index*100, 5+index*30
        events += [
            dict(name=label, cat='user_annotation', ph='X', ts=host_start, dur=50,
                pid=10, tid=20, args={'External id':external}),
            dict(name='cudaGraphLaunch', cat='cuda_runtime', ph='X', ts=host_start+10, dur=5,
                pid=10, tid=20, args={'correlation':correlation}),
            dict(name=label, cat='gpu_user_annotation', ph='X', ts=gpu_start, dur=20,
                pid=0, tid=7, args={'External id':external}),
            dict(name='void FlashAttnFwdSm90<64, 128, 64>(Params)', cat='kernel', ph='X',
                ts=gpu_start+2, dur=10, args={'correlation':correlation,
                    'grid':[132,1,1], 'block':[256,1,1], 'registers per thread':255,
                    'shared memory':56320, 'graph id':2, 'graph node id':8589934594, 'stream':7}),
            # Foreign GPU work inside the CPU annotation must not be attributed.
            dict(name='unrelated_restore', cat='kernel', ph='X', ts=host_start+20, dur=5,
                args={'correlation':999+index, 'graph id':0}),
        ]
    return dict(traceEvents=events), labels


class Protocol(unittest.TestCase):
    def test_fixed_single_b32_l8_full_contract_and_counts(self):
        self.assertEqual(w.SEEDS, (135831,135832))
        self.assertEqual(len(w.CASES), 1)
        case = w.CASES[0]
        self.assertEqual([case[k] for k in ('batch','q_len','layer','ctx','page','heads','spread','dist','kv_scale')],
            [32,8,'full',80000,1,'tp8','agentic','coherent',1.])
        self.assertEqual((w.WARMUP_ROUNDS,w.REVERSE_PAIRS), (4,6))
        counts = w.expected_counts(True)
        for key, value in dict(fixtures=2, original_oracle_calls=4, warmup_checked_replays=16,
                measured_checked_replays=48, activity_checked_replays=2,
                total_checked_graph_replays=70, graph_replays_attempted=70,
                graph_replays_completed=70, capture_warmup_function_calls_excluded=3,
                capture_function_calls_excluded=1).items():
            self.assertEqual(counts[key], value)

    def test_six_reverse_pairs_do_not_mix_order(self):
        for index in range(0,12,2):
            self.assertEqual(w.measured_order(index), list(reversed(w.measured_order(index+1))))
        self.assertNotEqual(w.measured_order(0), w.measured_order(2))
        for index in (-1,12,True,1.2):
            with self.assertRaises(ValueError):
                w.measured_order(index)

    def test_raw_summary_no_floor_or_replay_subtraction(self):
        result = w.summarize_samples(samples())
        self.assertEqual(result['reverse_order_pair_ratios'], [2.]*6)
        self.assertEqual(result['raw_median_us'], dict(zip(w.ARMS,[20.,10.])))
        self.assertEqual(result['contiguous_latency_reduction_percent'], 50.)
        self.assertEqual(result['favorable_contiguous_pairs'], 6)
        self.assertEqual(w.summarize_samples(samples(10,20))['contiguous_latency_reduction_percent'], -100.)

    def test_invalid_or_incomplete_timing_samples_rejected(self):
        cases = [samples()[:-1]]
        for value in (float('nan'),float('inf'),0,-1):
            rows = samples(); rows[0]['raw_us'][w.ARMS[0]] = value; cases.append(rows)
        rows = samples(); rows[0]['order'].reverse(); cases.append(rows)
        rows = samples(); del rows[0]['raw_us'][w.ARMS[1]]; cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):
                w.summarize_samples(rows)

    def test_keep_requires_fifteen_percent_on_each_seed(self):
        self.assertEqual(w.threshold_decision(fixtures(samples(100,85),samples(100,85)),True)['decision'], 'KEEP')
        self.assertEqual(w.threshold_decision(fixtures(samples(),samples(100,85.1)),True)['decision'], 'DROP')
        self.assertEqual(w.threshold_decision(fixtures(samples(100,85.1),samples()),True)['decision'], 'DROP')

    def test_all_six_pairs_must_win_even_with_large_median(self):
        for first_ratio in (1.,.9):
            rows = samples()
            for row in rows[:2]:
                row['raw_us'] = {w.ARMS[0]:10*first_ratio,w.ARMS[1]:10.}
            self.assertGreater(w.summarize_samples(rows)['contiguous_latency_reduction_percent'],15.)
            self.assertEqual(w.threshold_decision(fixtures(rows,samples()),True)['decision'],'DROP')

    def test_identity_and_two_ordered_complete_fixtures_are_hard_gates(self):
        for identity in (False,None,'observed_equal',1):
            with self.assertRaises(ValueError):
                w.threshold_decision(fixtures(),identity)
        for rows in (fixtures()[:1],list(reversed(fixtures())),fixtures([],samples())):
            with self.assertRaises(ValueError):
                w.threshold_decision(rows,True)

    def test_score_projection_remains_conditional_not_a_candidate_gain(self):
        result = w.threshold_decision(fixtures(),True)
        fraction = .8280845119160929
        self.assertEqual(result['conditional_score_full_fraction'],fraction)
        self.assertAlmostEqual(result['conditional_score_factor_at_15_percent'],(1/(1-fraction*.15))**(1/6))
        self.assertIn('hypothesis',result['qualification'])
        self.assertIn('not a deployable candidate',result['qualification'])

    def test_exact_source_bindings_flat_payload_mutation_and_symlink(self):
        self.assertEqual(w.source_gate(),w.SOURCE_HASHES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, source in w.source_paths().items():
                (root/name).write_bytes(source.read_bytes())
            self.assertEqual(w.source_gate(root),w.SOURCE_HASHES)
            target = root/'fixture.py'; target.write_bytes(target.read_bytes()+b'\n# altered\n')
            with self.assertRaises(ValueError):
                w.source_gate(root)
            target.unlink(); target.symlink_to(HERE/'fixture.py')
            with self.assertRaises(ValueError):
                w.source_gate(root)

    def test_default_has_no_vendor_import_or_dynamic_execution(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        names = []
        for node in tree.body:
            if isinstance(node,ast.Import): names += [n.name.split('.')[0] for n in node.names]
            elif isinstance(node,ast.ImportFrom): names.append((node.module or '').split('.')[0])
        self.assertFalse(set(names)&{'torch','triton','sglang','sgl_kernel','modal','subprocess'})
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id in ('exec','eval','__import__') for node in ast.walk(tree)))

    def test_activity_is_mandatory_before_execute(self):
        with patch.object(sys,'argv',['worker.py','--execute']), patch.object(w,'execute') as execute, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            w.main()
        execute.assert_not_called()

    def test_scalar_byte_view_flattens_first(self):
        class Scalar:
            def __init__(self,flat=False):
                self.flat = flat
                self.shape = () if not flat else (1,)
                self.dtype = 'float32'
            def contiguous(self): return self
            def reshape(self,*shape):
                if shape != (-1,): raise ValueError('flatten required')
                return Scalar(True)
            def view(self,dtype):
                if not self.flat: raise ValueError('scalar dtype reinterpretation')
                return b'bytes'
        api = type('API',(),dict(uint8='uint8',equal=staticmethod(lambda a,b:a==b)))
        self.assertTrue(w.byte_equal(api,Scalar(),Scalar()))

    def test_no_extra_bookkeeping_inside_cuda_event_interval(self):
        tree = ast.parse((HERE/'worker.py').read_text()); found = 0
        for node in ast.walk(tree):
            if not isinstance(node,ast.For): continue
            for index,statement in enumerate(node.body):
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) \
                        and ast.unparse(statement.value.func)=='begin.record':
                    self.assertEqual([ast.unparse(row) for row in node.body[index:index+4]],
                        ['begin.record()','graph.replay()','end.record()','end.synchronize()'])
                    found += 1
        self.assertEqual(found,1)


class Attribution(unittest.TestCase):
    def test_correlation_excludes_foreign_events_despite_clock_offset(self):
        trace, labels = synthetic_trace()
        result = w.parse_activity(trace,labels,trace_helper)
        for label in labels:
            rows = result['groups'][label]['kernels']
            self.assertEqual(len(rows),1)
            self.assertIn('FlashAttnFwdSm90',rows[0]['name'])
            self.assertEqual(rows[0]['launch']['grid'],[132,1,1])
            self.assertNotIn('achieved_occupancy',rows[0])

    def test_actual_saved_b32_trace_is_attributed_without_cpu_containment(self):
        path = HERE.parent/'fa3-next-evidence/baseline-runtime-identity/activity-trace.json.gz'
        trace = json.loads(gzip.decompress(path.read_bytes()))
        label = 'PROFILE_SELECTED_FULL_VERIFY_B32'
        result = w.parse_activity(trace,[label],trace_helper)['groups'][label]
        self.assertEqual(result['kernel_count'],5)
        self.assertEqual(result['attribution']['original_cpu_containment_mainloop_count'],0)
        self.assertEqual(len(result['attribution']['full_mainloop_symbols'][0]),1380)

    def test_duplicate_missing_wrong_correlation_or_external_id_fails(self):
        trace, labels = synthetic_trace()
        changes = []
        duplicate = copy.deepcopy(trace); duplicate['traceEvents'].append(copy.deepcopy(duplicate['traceEvents'][0])); changes.append(duplicate)
        missing = copy.deepcopy(trace); missing['traceEvents']=[e for e in missing['traceEvents'] if e.get('cat')!='kernel']; changes.append(missing)
        wrong = copy.deepcopy(trace); wrong['traceEvents'][3]['args']['correlation']=9000; changes.append(wrong)
        external = copy.deepcopy(trace); external['traceEvents'][2]['args']['External id']=9999; changes.append(external)
        for changed in changes:
            with self.assertRaises(ValueError):
                w.parse_activity(changed,labels,trace_helper)

    def test_native_missing_or_different_identity_cannot_pass(self):
        trace, labels = synthetic_trace(); groups = w.parse_activity(trace,labels,trace_helper)['groups']
        left,right = [groups[label]['kernels'] for label in labels]
        self.assertTrue(w.native_identity_comparison(left,right)['same_symbol_and_launch_fields'])
        for field,value in (('grid',None),('block','256,1,1'),('registers per thread',None),('graph id',0)):
            altered = copy.deepcopy(right); altered[0]['launch'][field]=value
            result = w.native_identity_comparison(left,altered)
            self.assertEqual(result['status'],'UNKNOWN')
            self.assertIsNone(result['same_symbol_and_launch_fields'])
        for altered in ([],[dict(right[0],name=' ')]):
            self.assertEqual(w.native_identity_comparison(left,altered)['status'],'UNKNOWN')
        altered = copy.deepcopy(right); altered[0]['name'] += 'Different'
        self.assertFalse(w.native_identity_comparison(left,altered)['same_symbol_and_launch_fields'])
        altered = copy.deepcopy(right); altered[0]['launch']['grid']=[131,1,1]
        self.assertFalse(w.native_identity_comparison(left,altered)['same_symbol_and_launch_fields'])

    def test_trace_metadata_exact_and_bounds_fail_without_truncation(self):
        raw = json.dumps(synthetic_trace()[0]).encode(); compressed = gzip.compress(raw,mtime=0)
        result = w.trace_artifact(raw)
        self.assertEqual(result['raw_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['gzip_sha256'],hashlib.sha256(compressed).hexdigest())
        self.assertEqual(result['raw_bytes'],len(raw))
        self.assertEqual(result['gzip_bytes'],len(compressed))
        self.assertNotIn('data_b64',result)
        with patch.object(w,'MAX_RAW_TRACE',len(raw)-1), self.assertRaises(ValueError):
            w.trace_artifact(raw)
        with patch.object(w,'MAX_GZIP_TRACE',1), self.assertRaises(ValueError):
            w.trace_artifact(raw)


if __name__=='__main__':
    unittest.main()
