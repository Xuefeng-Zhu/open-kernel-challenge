"""CPU-only protocol/parser/bounds tests; no fake GPU speed result is emitted."""
import ast
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('locality_worker_under_test',HERE/'worker.py')
w=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(w)


def synthetic_rounds(a=20.,b=10.):
    # Pure summary arithmetic input, not a GPU execution or exported result.
    return [dict(index=i,order=w.measured_order(i),raw_us={w.ARMS[0]:a,w.ARMS[1]:b})
            for i in range(12)]


class Protocol(unittest.TestCase):
    def test_declared_cases_seeds_and_counts_are_fixed(self):
        self.assertEqual(w.SEEDS,(135831,135832))
        self.assertEqual([(r['batch'],r['q_len'],r['layer'],r['ctx'],r['page']) for r in w.CASES],
            [(1,1,'sliding',80000,1),(16,8,'full',80000,1),(32,1,'full',80000,1)])
        self.assertEqual(w.expected_counts(False)['original_oracle_calls'],12)
        self.assertEqual(w.expected_counts(False)['warmup_checked_replays'],48)
        self.assertEqual(w.expected_counts(False)['measured_checked_replays'],144)
        self.assertEqual(w.expected_counts(False)['total_checked_graph_replays'],204)
        self.assertEqual(w.expected_counts(True)['total_checked_graph_replays'],210)
        self.assertEqual(w.expected_counts(True)['graph_replays_attempted'],210)
        self.assertEqual(w.expected_counts(True)['original_oracle_calls_attempted'],12)

    def test_six_pairs_reverse_order_without_layout_or_seed_mixing(self):
        for i in range(0,12,2):
            self.assertEqual(w.measured_order(i),list(reversed(w.measured_order(i+1))))
        self.assertNotEqual(w.measured_order(0),w.measured_order(2))
        for bad in (-1,12,True,1.2):
            with self.assertRaises(ValueError):w.measured_order(bad)

    def test_raw_pair_summary_has_no_floor_subtraction(self):
        out=w.summarize_samples(synthetic_rounds())
        self.assertEqual(out['reverse_order_pair_ratios'],[2.]*6)
        self.assertEqual(out['raw_median_us'],dict(zip(w.ARMS,[20.,10.])))
        self.assertEqual(out['contiguous_latency_reduction_percent'],50.)
        self.assertEqual(out['favorable_contiguous_pairs'],6)
        self.assertAlmostEqual(w.summarize_samples(synthetic_rounds(10,20))['contiguous_latency_reduction_percent'],-100.)

    def test_incomplete_wrong_order_nonfinite_or_missing_samples_fail(self):
        candidates=[synthetic_rounds()[:-1]]
        for value in (float('nan'),float('inf'),0,-1):
            rows=synthetic_rounds();rows[3]['raw_us'][w.ARMS[0]]=value;candidates.append(rows)
        rows=synthetic_rounds();rows[0]['order']=list(reversed(rows[0]['order']));candidates.append(rows)
        rows=synthetic_rounds();rows[0]['raw_us'].pop(w.ARMS[0]);candidates.append(rows)
        for rows in candidates:
            with self.assertRaises(ValueError):w.summarize_samples(rows)

    def test_actual_source_bindings_and_mutated_or_symlink_source_refused(self):
        self.assertEqual(w.source_gate(),w.SOURCE_HASHES)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in w.SOURCE_HASHES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((w.V2/name).read_bytes())
            self.assertEqual(w.source_gate(root),w.SOURCE_HASHES)
            p=root/'locality-probe/fixture.py';p.write_bytes(p.read_bytes()+b'\n# changed\n')
            with self.assertRaises(ValueError):w.source_gate(root)
            p.unlink();p.symlink_to(w.HERE/'fixture.py')
            with self.assertRaises(ValueError):w.source_gate(root)

    def test_default_has_no_vendor_import_or_dynamic_execution(self):
        tree=ast.parse((HERE/'worker.py').read_text())
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        names=[alias.name.split('.')[0] for n in imports if isinstance(n,ast.Import) for alias in n.names]
        names += [n.module.split('.')[0] for n in imports if isinstance(n,ast.ImportFrom)]
        self.assertFalse(set(names)&{'torch','triton','sglang','sgl_kernel','modal'})
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        vendor=[n for n in ast.walk(execute) if isinstance(n,ast.Import) and any(a.name in ('torch','triton') for a in n.names)]
        self.assertEqual(len(vendor),2)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec','__import__') for n in ast.walk(tree)))
        self.assertNotIn('subprocess',names)

    def test_scalar_byte_check_flattens_before_reinterpreting(self):
        # Enforces the Torch dtype-view contract without importing Torch.
        class Scalar:
            shape=();dtype='float32'
            def __init__(self,flat=False):self.flat=flat
            def contiguous(self):return self
            def reshape(self,*shape):
                self.asserted_shape=shape
                if shape!=(-1,):raise ValueError('flatten required')
                return Scalar(True)
            def view(self,dtype):
                if not self.flat:raise RuntimeError('zero-dimensional dtype view forbidden')
                return b'\0\0\x80?'
        class TensorAPI:
            uint8='uint8'
            @staticmethod
            def equal(a,b):return a==b
        self.assertTrue(w.byte_equal(TensorAPI,Scalar(),Scalar()))

    def test_locality_contrast_refuses_identity_or_only_unattended_changes(self):
        spec=importlib.util.spec_from_file_location('locality_fixture_test_worker',HERE/'fixture.py')
        helper=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
        identity=helper.build_plan([list(range(2050))],[2050],[2049],1,2050)
        with self.assertRaises(ValueError):w.locality_contrast(identity,'sliding')
        tail_only=list(range(2050));tail_only[0],tail_only[1]=tail_only[1],tail_only[0]
        plan=helper.build_plan([tail_only],[2050],[2049],1,2050)
        with self.assertRaises(ValueError):w.locality_contrast(plan,'sliding')
        shifted=helper.build_plan([[2,3],[0,1]],[2,2],[3,1],1,4)
        with self.assertRaises(ValueError):w.locality_contrast(shifted,'full')
        result=w.locality_contrast(plan,'full')
        self.assertEqual(result['moved_attended_history_pages'],2)
        self.assertEqual(result['canonical_layout_contiguous_adjacent_pairs'],2049)
        self.assertNotEqual(result['page_table_A_sha256'],result['page_table_B_sha256'])

    def test_no_extra_bookkeeping_inside_event_interval(self):
        tree=ast.parse((HERE/'worker.py').read_text());found=0
        for node in ast.walk(tree):
            if not isinstance(node,ast.For):continue
            for i,statement in enumerate(node.body):
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func)=='begin.record':
                    self.assertEqual([ast.unparse(s) for s in node.body[i:i+4]],
                        ['begin.record()','graph.replay()','end.record()','end.synchronize()'])
                    found+=1
        self.assertEqual(found,1)


class ActivityParser(unittest.TestCase):
    def trace(self):
        return {'traceEvents':[
            {'name':'LOCALITY_A','cat':'user_annotation','ph':'X','ts':100,'dur':100},
            {'name':'native-long-symbol','cat':'kernel','ph':'X','ts':125,'dur':10,
             'args':{'grid':[132,1,1],'block':[256,1,1],'registers per thread':255,
                     'shared memory':56320,'graph id':1,'graph node id':7,'stream':5}},
            {'name':'outside-eviction','cat':'kernel','ph':'X','ts':10,'dur':15,'args':{}},
        ]}

    def test_only_fenced_kernels_with_full_identity_are_recorded(self):
        result=w.parse_activity(self.trace(),['LOCALITY_A'])
        rows=result['groups']['LOCALITY_A']['kernels']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['name'],'native-long-symbol')
        self.assertEqual(rows[0]['symbol_sha256'],hashlib.sha256(b'native-long-symbol').hexdigest())
        self.assertEqual(rows[0]['launch']['grid'],[132,1,1])
        self.assertNotIn('achieved_occupancy',rows[0])

    def test_missing_duplicate_or_no_kernel_range_is_failure(self):
        trace=self.trace()
        with self.assertRaises(ValueError):w.parse_activity(trace,['missing'])
        trace['traceEvents'].append(dict(trace['traceEvents'][0]))
        with self.assertRaises(ValueError):w.parse_activity(trace,['LOCALITY_A'])
        trace=self.trace();trace['traceEvents']=[e for e in trace['traceEvents'] if e['cat']!='kernel']
        with self.assertRaises(ValueError):w.parse_activity(trace,['LOCALITY_A'])

    def test_trace_transport_is_exact_and_oversize_is_not_truncated(self):
        raw=json.dumps(self.trace()).encode();artifact=w.trace_artifact(raw)
        compressed=base64.b64decode(artifact['data_b64'])
        self.assertEqual(gzip.decompress(compressed),raw)
        self.assertEqual(artifact['raw_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(artifact['compressed_sha256'],hashlib.sha256(compressed).hexdigest())
        with self.assertRaises(ValueError):w.trace_artifact(b'x'*(w.MAX_RAW_TRACE+1))

    def test_native_identity_ignores_event_order_and_missing_is_unknown(self):
        row=w.parse_activity(self.trace(),['LOCALITY_A'])['groups']['LOCALITY_A']['kernels'][0]
        other=json.loads(json.dumps(row));other['launch']['graph node id']=8
        result=w.native_identity_comparison([row,other],[other,row])
        self.assertTrue(result['same_symbol_and_launch_fields'])
        missing=json.loads(json.dumps(row));missing['launch']['grid']=None
        result=w.native_identity_comparison([missing],[missing])
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertIsNone(result['same_symbol_and_launch_fields'])
        changed=json.loads(json.dumps(row));changed['launch']['grid']=[64,1,1]
        self.assertFalse(w.native_identity_comparison([row],[changed])['same_symbol_and_launch_fields'])
        changed=json.loads(json.dumps(row));changed['launch']['block']='256,1,1'
        self.assertEqual(w.native_identity_comparison([changed],[changed])['status'],'UNKNOWN')

    def test_empty_native_symbol_does_not_count_as_identity(self):
        trace=self.trace();trace['traceEvents'][1]['name']=' '
        with self.assertRaises(ValueError):w.parse_activity(trace,['LOCALITY_A'])


if __name__=='__main__':unittest.main()
