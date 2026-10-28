"""Offline protocol tests: no candidate imports, CUDA, services or processes."""
import ast
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import worker

HERE=Path(__file__).resolve().parent
ATTENTION=HERE.parent/'attention-fusion-lookahead/reviews/attention-v2'


class WorkerProtocolTests(unittest.TestCase):
    def test_prerequisites_match_all_sixteen_task_rows(self):
        rows=worker.TEST_CASES
        self.assertEqual(len(rows),16)
        self.assertEqual([r['batch'] for r in rows],[1,3,5,4,8,2,4,4,1,3,2,4,4,2,4,2])
        self.assertEqual([r['ctx'] for r in rows],[10,2047,5000,3000,8192,100000,6000,6000,1048576,4000,9000,80000,6000,20000,80000,20000])
        self.assertEqual(sum(r.get('heads')=='dp' for r in rows),3)
        self.assertEqual(sum(r.get('kv_scale')==1/64 for r in rows),2)
        self.assertEqual(sum(r.get('dist')=='edges' for r in rows),2)
        self.assertEqual(sum(r.get('dist')=='coherent' for r in rows),2)

    def test_scored_coverage_distinct(self):
        rows=worker.scored_cases()
        self.assertEqual(len(rows),12)
        self.assertEqual({(r['batch'],r['q_len'],r['layer']) for r in rows},
            {(b,l,k) for b in (1,16,32) for l in (1,8) for k in ('sliding','full')})
        self.assertTrue(all(r['ctx']==80000 and r['page']==1 for r in rows))
        self.assertTrue(all(r['dist']==('coherent' if r['layer']=='full' else 'soft') for r in rows))

    def test_fresh_predeclared_seeds(self):
        self.assertEqual(worker.SEEDS,(135901,135902))
        self.assertNotIn(135871,worker.SEEDS)

    def test_reversed_order_every_pair(self):
        for i in range(0,12,2):
            self.assertEqual(worker.order(i),worker.order(i+1)[::-1])
        self.assertEqual(set(worker.order(0)),{'champion','v298'})

    def test_pair_aggregation_cancels_first_arm_bias(self):
        samples=[dict(champion=12.,v298=5.) if i%2==0 else dict(champion=10.,v298=6.) for i in range(12)]
        summary=worker.paired_summary(samples)
        self.assertAlmostEqual(summary['median_pair_speedup'],2.)
        self.assertEqual(summary['favorable_pairs'],6)

    def test_pair_incomplete_nonfinite_zero_rejected(self):
        for samples in ([],[dict(champion=1.,v298=1.)]*11,
                        [dict(champion=1.,v298=0.)]*12,
                        [dict(champion=math.nan,v298=1.)]*12,
                        [dict(champion=1.,other=1.)]*12):
            with self.assertRaises(ValueError): worker.paired_summary(samples)

    def test_whole_suite_exact_weights(self):
        rows=[]
        for spec in worker.scored_cases():
            c=10. if spec['layer']=='sliding' else 100.
            s=2. if spec['layer']=='sliding' else 4.
            rows.append(dict(case=spec,seed=worker.SEEDS[0],summary=dict(champion_median_us=c,median_pair_speedup=s)))
        out=worker.whole_suite(rows,worker.SEEDS[0])
        self.assertEqual(len(out['loads']),6)
        for load in out['loads']:
            self.assertEqual(load['champion_weighted_us'],2060.)
            self.assertEqual(load['v298_reference_normalized_weighted_us'],630.)
        self.assertAlmostEqual(out['relative_champion_geomean'],2060/630)
        self.assertIn('not an official score',out['qualification'])

    def test_whole_suite_missing_duplicate_or_invalid_rejected(self):
        rows=[dict(case=spec,seed=1,summary=dict(champion_median_us=10.,median_pair_speedup=2.)) for spec in worker.scored_cases()]
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError): worker.whole_suite(bad,1)
        rows[0]['summary']['median_pair_speedup']=float('inf')
        with self.assertRaises(ValueError): worker.whole_suite(rows,1)

    def test_all_uploaded_sources_match_frozen_hashes(self):
        paths={'selected.py':ATTENTION/'fusion-lookahead-probe/candidate.py',
               'v298.py':HERE/'upstream/muse_attention_decode.py',
               'reference.py':ATTENTION/'official/reference.py','checks.py':ATTENTION/'official/checks.py'}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,path in paths.items():
                raw=path.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),worker.SOURCES[name])
                (root/name).write_bytes(raw)
            self.assertEqual(worker.bind_sources(root),worker.SOURCES)
            (root/'v298.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'v298.py'): worker.bind_sources(root)

    def test_source_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'selected.py').symlink_to(ATTENTION/'fusion-lookahead-probe/candidate.py')
            with self.assertRaises(ValueError): worker.bind_sources(root)

    def test_cpu_import_is_inert(self):
        tree=ast.parse((HERE/'worker.py').read_text())
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                self.assertNotIn('torch',ast.unparse(node))
                self.assertNotIn('triton',ast.unparse(node))
                self.assertNotIn('v298',ast.unparse(node))
        self.assertFalse(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) for n in tree.body))

    def test_no_official_process_network_or_profiler_api(self):
        tree=ast.parse((HERE/'worker.py').read_text())
        forbidden={'subprocess','requests','urllib','socket','modal','os','ncu','eval','exec','__import__'}
        calls={ast.unparse(n.func).split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls&forbidden)
        imports={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports|={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        self.assertFalse(imports&forbidden)

    def test_proven_scalar_safe_byte_comparison(self):
        src=ast.get_source_segment((HERE/'worker.py').read_text(),next(n for n in ast.parse((HERE/'worker.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='bytes_equal'))
        self.assertIn('reshape(-1).view(torch.uint8)',src)

    def test_atomic_output_nondeterminism_not_bitwise_gate(self):
        src=ast.get_source_segment((HERE/'worker.py').read_text(),next(n for n in ast.parse((HERE/'worker.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='assert_replay'))
        self.assertIn('torch.isfinite(out)',src)
        self.assertIn('cache_bytes(',src)
        self.assertNotIn('ref.collect(work, out), expected',src)

    def test_cli_540_second_cap_before_native_run(self):
        with mock.patch.object(worker.sys, 'argv', ['worker.py', '--output', '/tmp/unused-summary.json']), mock.patch.object(worker, 'run') as run:
            worker.main()
            self.assertEqual(run.call_args.args[-1], 540)
        for limit in ('0', '-1', '541'):
            with mock.patch.object(worker.sys, 'argv', ['worker.py', '--output', '/tmp/unused-summary.json', '--deadline', limit]), mock.patch.object(worker, 'run') as run:
                with self.assertRaises(ValueError): worker.main()
                run.assert_not_called()

    def test_build_outside_graph_before_prerequisite(self):
        src=(HERE/'worker.py').read_text()
        self.assertLess(src.index("built=modules['v298']._module()"),src.index('for index,spec in enumerate(TEST_CASES)'))
        self.assertLess(src.index('for index,spec in enumerate(TEST_CASES)'),src.index('for seed in SEEDS:'))
        self.assertNotIn('flags=',src)


if __name__=='__main__': unittest.main()
