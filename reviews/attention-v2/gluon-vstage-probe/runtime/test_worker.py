"""CPU-only V-stage input, three-arm order and prospective benefit gates.

No Torch/Gluon/driver imports or native correctness/performance claim.
"""
import argparse
import ast
import importlib.util
from pathlib import Path
import json
import math
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gluon_vstage_runtime_protocol', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def measured_rows(before=(96.,96.), after=(100.,100.), fa3=(100.,100.)):
    rows = []
    for index in range(worker.WARMUP_SAMPLES, worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES):
        order, version = worker.order_and_version(index)
        rows.append(dict(index=index,order=order,input_version=version,
            raw_us=dict(fa3=fa3[version],v_after_wait=after[version],v_before_wait=before[version])))
    return rows


class RuntimeContracts(unittest.TestCase):
    def test_frozen_kernel_config_working_baseline_and_original_support(self):
        self.assertEqual(worker.source_gate(), {**worker.SOURCE_HASHES,**worker.SUPPORT_HASHES,**worker.LICENSE_HASHES})
        self.assertEqual(worker.SOURCE_HASHES['kernel.py'], '6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e')
        self.assertEqual(worker.SOURCE_HASHES['fusion-lookahead-probe/candidate.py'], '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8')
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory)
            for name in ['kernel.py','config.py']:
                (probe/name).write_bytes((worker.PROBE/name).read_bytes())
            (probe/'kernel.py').write_bytes(b'changed kernel\n')
            with self.assertRaises(ValueError):
                worker.source_gate(probe,worker.V2,worker.REPO)
            (probe/'kernel.py').unlink()
            (probe/'kernel.py').symlink_to(worker.PROBE/'kernel.py')
            with self.assertRaises(ValueError):
                worker.source_gate(probe,worker.V2,worker.REPO)

    def test_exact_three_cases_fresh_seeds_and_declared_count_arithmetic(self):
        self.assertEqual(worker.NAMES,['fa3','v_after_wait','v_before_wait'])
        self.assertEqual(worker.SEEDS,[135751,135752])
        self.assertEqual(worker.PT_STRIDE,86008)
        self.assertEqual(len(worker.CASES),3)
        self.assertEqual(len({case['name'] for case in worker.CASES}),3)
        for case in worker.CASES:
            self.assertEqual((case['batch'],case['q_len'],case['heads'],case['layer'],case['page']), (32,8,'tp8','full',1))
        self.assertEqual((worker.BOUNDARY_UNIT['ctx'],worker.BOUNDARY_UNIT['spread'],worker.BOUNDARY_UNIT['dist'],worker.BOUNDARY_UNIT['kv_scale']), (129,'fixed','soft',1.))
        self.assertEqual(worker.BOUNDARY_SCALED['kv_scale'],1/64)
        self.assertEqual((worker.TARGET['ctx'],worker.TARGET['spread'],worker.TARGET['dist'],worker.TARGET['kv_scale']), (4096,'agentic','coherent',1.))
        self.assertEqual(len(worker.CASES)*len(worker.NAMES)*3+1+2,30)
        self.assertEqual(len(worker.CASES)*len(worker.NAMES),9)
        self.assertEqual(len(worker.CASES)*3*2*2,36)  # Two FA3/prototype comparisons, K and V.
        self.assertEqual((worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES)*3,48)

    def test_boundary_new_tokens_tile_edges_and_causal_prefix_are_legal(self):
        values=[0,1,55,56,63,64,65,129]
        self.assertEqual(worker.boundary_lengths(0),values*4)
        self.assertEqual(worker.boundary_lengths(1),list(reversed(values*4)))
        self.assertEqual([seq+8 for seq in values],[8,9,63,64,71,72,73,137])
        self.assertEqual([(seq+8+63)//64 for seq in values],[1,1,1,1,2,2,2,3])
        for seq in values:
            total=seq+8
            valid=range(total)
            for query in range(8):
                visible=[key for key in valid if key<=seq+query]
                self.assertEqual(len(visible),seq+query+1)
                self.assertIn(seq+query,visible)
                self.assertTrue(all(key<total for key in visible))
        for version in [-1,2]:
            with self.assertRaises(ValueError):
                worker.boundary_lengths(version)

    def test_page1_padding_preserves_prefix_and_masks_only_unused_columns(self):
        for seq in [0,1,55,56,63,64,65,129]:
            prefix=list(reversed(range(seq+8)))
            padded=worker.padded_row(prefix)
            self.assertEqual(len(padded),86008)
            self.assertEqual(padded[:seq+8],prefix)
            self.assertTrue(all(page==-1 for page in padded[seq+8:]))
            self.assertEqual(sorted(padded[:seq+8]),list(range(seq+8)))
        # An unused negative suffix already present in the source table is
        # legal padding. The helper alone does not certify a live-prefix range.
        row=[8,0,4,-1,-1]
        self.assertEqual(worker.padded_row(row,8),row+[-1]*3)
        for row,width in [([0,1],1),([-2],3),([1.],3),([True],3),([0],-1)]:
            with self.assertRaises(ValueError):
                worker.padded_row(row,width)

    def test_three_arm_reverse_orders_rotate_positions_and_share_fixture(self):
        first_positions={'0':set(),'1':set()}
        for index in range(0,worker.WARMUP_SAMPLES+worker.MEASURED_SAMPLES,2):
            left,right=worker.order_and_version(index),worker.order_and_version(index+1)
            self.assertEqual(set(left[0]),set(worker.NAMES))
            self.assertEqual(left[0],list(reversed(right[0])))
            self.assertEqual(left[1],right[1])
            if index>=worker.WARMUP_SAMPLES:
                first_positions[str(left[1])].add(tuple(left[0]))
        self.assertEqual({key:len(value) for key,value in first_positions.items()},{'0':3,'1':3})
        for orders in first_positions.values():
            for arm in worker.NAMES:
                self.assertEqual({order.index(arm) for order in orders},{0,1,2})
        for index in [-1,1.,True]:
            with self.assertRaises(ValueError):
                worker.order_and_version(index)

    def test_summary_reproduces_six_pairs_and_three_per_fixture_per_control(self):
        summary=worker.timing_summary(measured_rows())
        self.assertEqual(set(summary['comparisons']), {'v_before_wait_vs_v_after_wait','v_before_wait_vs_fa3'})
        for comparison in summary['comparisons'].values():
            self.assertEqual(comparison['positive_reverse_order_pairs'],6)
            self.assertEqual(len(comparison['reverse_order_pair_geomeans']),6)
            for fixture in comparison['per_fixture'].values():
                self.assertEqual(len(fixture['pair_ratios']),3)
                self.assertEqual(fixture['positive_pairs'],3)
                self.assertAlmostEqual(fixture['median_ratio'],100/96)
                self.assertAlmostEqual(fixture['latency_reduction_percent'],4.)
        separate=worker.timing_summary(measured_rows(fa3=(10000.,10000.)))
        self.assertEqual(summary['comparisons']['v_before_wait_vs_v_after_wait'], separate['comparisons']['v_before_wait_vs_v_after_wait'])

    def test_mechanism_and_actual_fa3_gates_are_separate_and_no_promotion_follows(self):
        both=worker.benefit_gate(measured_rows())
        self.assertTrue(both['mechanism_passed'] and both['actual_fa3_passed'])
        self.assertTrue(both['prospective_followup_supported'])
        mechanism_only=worker.benefit_gate(measured_rows(before=(110.,110.),after=(120.,120.),fa3=(100.,100.)))
        self.assertTrue(mechanism_only['mechanism_passed'])
        self.assertFalse(mechanism_only['actual_fa3_passed'] or mechanism_only['prospective_followup_supported'])
        actual_only=worker.benefit_gate(measured_rows(before=(95.,95.),after=(90.,90.),fa3=(100.,100.)))
        self.assertFalse(actual_only['mechanism_passed'] or actual_only['prospective_followup_supported'])
        self.assertTrue(actual_only['actual_fa3_passed'])
        for result in [both,mechanism_only,actual_only]:
            self.assertFalse(result['production_replacement_admitted'])
            self.assertFalse(result['official_check_admitted'])

    def test_each_fixture_must_meet_two_percent_even_with_a_large_pooled_gain(self):
        result=worker.benefit_gate(measured_rows(before=(80.,99.)))
        self.assertFalse(result['mechanism_passed'] or result['actual_fa3_passed'])
        self.assertFalse(result['prospective_followup_supported'])
        self.assertFalse(worker.benefit_gate(measured_rows(before=(99.,99.)))['mechanism_passed'])

    def test_all_six_pairs_required_for_each_comparison_independently(self):
        lose_after=measured_rows(after=(100.,100.),fa3=(120.,120.))
        for row in lose_after[:2]: row['raw_us']['v_before_wait']=101.
        result=worker.benefit_gate(lose_after)
        self.assertFalse(result['mechanism_passed'])
        self.assertTrue(result['actual_fa3_passed'])
        lose_fa3=measured_rows(after=(120.,120.),fa3=(100.,100.))
        for row in lose_fa3[:2]: row['raw_us']['v_before_wait']=101.
        result=worker.benefit_gate(lose_fa3)
        self.assertTrue(result['mechanism_passed'])
        self.assertFalse(result['actual_fa3_passed'])

    def test_timing_rejects_missing_nonfinite_invalid_order_or_fixture(self):
        for samples in [[],measured_rows()[:-1],measured_rows(before=(0.,96.)),measured_rows(after=(float('nan'),100.)),measured_rows(fa3=(float('inf'),100.))]:
            with self.assertRaises(ValueError): worker.timing_summary(samples)
        for mutate in ['order','version','missing','extra']:
            samples=measured_rows()
            if mutate=='order': samples[1]['order']=samples[0]['order']
            elif mutate=='version': samples[1]['input_version']=1-samples[0]['input_version']
            elif mutate=='missing': del samples[0]['raw_us']['fa3']
            else: samples[0]['raw_us']['unused']=100.
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError): worker.timing_summary(samples)

    def test_cpu_default_int32_metadata_and_unmodified_actual_fa3_source_contract(self):
        source=(HERE/'worker.py').read_text(); tree=ast.parse(source)
        top_imports=[alias.name for node in tree.body if isinstance(node,ast.Import) for alias in node.names]
        self.assertFalse(set(top_imports)&{'torch','triton','modal'})
        self.assertIn("parser.add_argument('--execute'",source)
        self.assertIn("status='offline-preflight'",source)
        self.assertIn("table.dtype != t.int32 or data['seq_lens'].dtype != t.int32",source)
        self.assertIn("t.full((32,PT_STRIDE),-1,dtype=t.int32",source)
        assertions={ast.dump(node.test,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Assert)}
        for expression in [
                'live.numel() == seq+8 and bool(((live>=0)&(live<capacity)).all())',
                'bool((padded[b,seq+8:] == -1).all())']:
            self.assertIn(ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False),assertions)
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    self.assertFalse(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
                        and target.value.id in {'working','kernel','ref','helpers'})
        functions=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='functions' for target in node.targets))
        actual=next(keyword.value for keyword in functions.keywords if keyword.arg=='fa3')
        self.assertEqual(ast.unparse(actual),'working.kernel_fn')
        for forbidden in ['CACHE_TOL =','REL_L2_TOL =','t.profiler','import modal']:
            self.assertNotIn(forbidden,source)
        compile(tree,str(HERE/'worker.py'),'exec')

    def test_source_failure_retains_bounded_partial_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'result.json'
            with patch.object(worker,'source_gate',side_effect=ValueError('test source mismatch')):
                code=worker.main(argparse.Namespace(output=output,deadline=160.))
            self.assertEqual(code,1)
            report=json.loads(output.read_text())
            self.assertFalse(report['passed'] or report['correctness_complete'] or report['timing_complete'])
            self.assertFalse(report['official_check_admitted'] or report['production_replacement_admitted'])
            self.assertEqual(report['timings'],[])
            self.assertLess(output.stat().st_size,worker.MAX_REPORT)
            self.assertFalse(output.with_suffix('.tmp').exists())


if __name__=='__main__': unittest.main(verbosity=2)
