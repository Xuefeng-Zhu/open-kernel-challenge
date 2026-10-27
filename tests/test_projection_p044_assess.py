"""Offline artifact fixtures; no CUDA, network, or live Modal evidence."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import projection_p044_assess as assessor


def dump(path, value):
    path.write_text(json.dumps(value))


def fixture(directory):
    reports = {}
    for mode in ("timing", "activity"):
        report = dict(mode=mode, passed=True, source_sha256=assessor.SOURCE_SHA256,
            source_hashes={assessor.SOURCE:assessor.SOURCE_SHA256},
            environment={"submission_sha256":assessor.SOURCE_SHA256},
            protocol=dict(warmup_replays=6, measured_rounds=24 if mode == 'timing' else 0,
                          eviction_bytes=256*1024**2),
            preparation=dict(runners={name:dict(synchronized_wall_seconds=1.25+i)
                                      for i, name in enumerate(assessor.RUNNERS)}), cases=[])
        for index, (layer, m) in enumerate(assessor.REPRESENTATIVES):
            case = dict(layer=layer, m=m, passed=True,
                aba=[dict(runner=name, state=label, passed=True, same_input_address=True,
                          immutable_x_weights=True) for name in assessor.RUNNERS
                     for label in ('A','B','A-return')])
            if mode == 'timing':
                case['warmup'] = [{} for _ in range(12)]
                case['conditions'] = {condition:dict(samples=[dict(round=i,
                    order=list(assessor.RUNNERS if i%2 == 0 else reversed(assessor.RUNNERS)),
                    us=dict(candidate=10*factor, baseline=20*factor)) for i in range(24)])
                    for condition, factor in [('cold',1),('warm',.5)]}
            else:
                case['activities'] = []
                for runner in assessor.RUNNERS:
                    path = directory/f'activity.case{index:02d}.{layer}.m{m}.{runner}.trace.json'
                    raw = dict(traceEvents=[
                        dict(ph='X',cat='kernel',name='nvjet_sm90_test',dur=4.0,
                             args={'grid':[16,1,1],'block':[128,1,1],'registers per thread':64,
                                   'shared memory':8192}),
                        dict(ph='X',cat='kernel',name='cublasLt::splitKreduce_kernel',dur=1.5,args={}),
                        dict(ph='X',cat='cpu_op',name='aten::mm',dur=200,args={})])
                    dump(path,raw)
                    case['activities'].append(dict(runner=runner,trace=dict(file=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)))
            report['cases'].append(case)
        reports[mode] = report
        dump(directory/(mode+'.json'),report)
    return reports


class P044AssessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reports = fixture(self.root)

    def save(self, mode):
        dump(self.root/(mode+'.json'),self.reports[mode])

    def test_complete_source_bound_diagnostic_preserves_samples_and_resources(self):
        result = assessor.assess(self.root)
        self.assertTrue(result['complete'])
        self.assertTrue(result['source_binding']['valid'])
        self.assertNotIn('score',result)
        self.assertEqual(result['modes']['timing']['preparation']['runners']['candidate']
                         ['synchronized_wall_seconds'],1.25)
        for case in result['cases']:
            cold = case['conditions']['cold']
            self.assertEqual(cold['raw_sample_count'],24)
            self.assertEqual(cold['pair_ratios'],[2.0]*12)
            self.assertEqual(cold['paired_ratio']['median'],2.0)
            self.assertEqual(cold['runners']['candidate']['raw_gross_us'],[10]*24)
            self.assertEqual(case['warm_over_cold_gross_ratios']['candidate']['per_round_ratios'],[.5]*24)
            self.assertEqual(case['modes']['timing']['aba']['candidate']['completed_passes'],3)
            trace = case['activities']['candidate']
            self.assertTrue(trace['available'])
            self.assertEqual(trace['duration_us_by_symbol_class'],dict(main=4.0,reduction=1.5,unknown=0.0))
            self.assertEqual(trace['kernels'][0]['resources_as_emitted']['registers per thread'],64)
            self.assertEqual(trace['kernels'][1]['resources_as_emitted'],{})

    def test_empirical_percentiles_and_unknown_symbol(self):
        self.assertEqual(assessor.distribution(list(range(11))),
            dict(count=11,minimum=0,p10=1.,median=5,p90=9.,maximum=10))
        self.assertIsNone(assessor.distribution([])['median'])
        self.assertEqual(assessor.kernel_class('opaque_kernel'), 'unknown')

    def test_wrong_or_malformed_source_refuses_attribution(self):
        original = copy.deepcopy(self.reports['activity'])
        for field, bad in [('source_sha256','wrong'), ('source_hashes','malformed'),
                           ('environment',{'submission_sha256':'different'})]:
            with self.subTest(field=field):
                self.reports['activity'] = {**original, field:bad}
                self.save('activity')
                result = assessor.assess(self.root)
                self.assertFalse(result['source_binding']['valid'])
                self.assertFalse(result['complete'])

    def test_missing_trace_is_unavailable_even_when_worker_passed(self):
        entry = self.reports['activity']['cases'][0]['activities'][0]
        (self.root/entry['trace']['file']).unlink()
        result = assessor.assess(self.root)
        self.assertTrue(result['modes']['activity']['worker_passed'])
        self.assertFalse(result['complete'])
        trace = result['cases'][0]['activities']['candidate']
        self.assertFalse(trace['available'])
        self.assertEqual(trace['errors'],['Raw trace file missing'])

    def test_empty_device_trace_cannot_claim_activity_success(self):
        entry = self.reports['activity']['cases'][0]['activities'][0]
        path = self.root/entry['trace']['file']
        dump(path,dict(traceEvents=[dict(ph='X',cat='cpu_op',name='aten::mm',dur=10)]))
        entry['trace'].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        self.save('activity')
        result = assessor.assess(self.root)
        trace = result['cases'][0]['activities']['candidate']
        self.assertFalse(trace['available'])
        self.assertFalse(trace['resource_metadata_available'])
        self.assertFalse(result['complete'])

    def test_raw_trace_hash_or_path_mismatch_rejected(self):
        entry = self.reports['activity']['cases'][0]['activities'][0]
        entry['trace']['sha256'] = 'wrong'
        self.save('activity')
        self.assertIn('hash/size mismatch',assessor.assess(self.root)['cases'][0]['activities']['candidate']['errors'][0])
        entry['trace']['file'] = '../outside.json'
        self.save('activity')
        self.assertIn('unsafe',assessor.assess(self.root)['cases'][0]['activities']['candidate']['errors'][0])

    def test_partial_failure_keeps_completed_aba_and_timings(self):
        report = self.reports['timing']
        report.update(passed=False,failure=dict(type='TimeoutError',message='deadline'))
        report['cases'] = report['cases'][:1]
        case = report['cases'][0]
        case['passed'] = False
        case['aba'] = case['aba'][:2]
        for condition in case['conditions'].values():
            condition['samples'] = condition['samples'][:3]
        self.save('timing')
        result = assessor.assess(self.root)
        self.assertFalse(result['complete'])
        self.assertEqual(result['modes']['timing']['failure']['type'],'TimeoutError')
        self.assertEqual(result['cases'][0]['modes']['timing']['aba']['candidate']['completed_passes'],2)
        self.assertEqual(result['cases'][0]['conditions']['cold']['pair_ratios'],[2.0])
        self.assertEqual(result['cases'][0]['conditions']['cold']['raw_sample_count'],3)
        self.assertFalse(result['cases'][1]['modes']['timing']['present'])

    def test_invalid_sample_is_not_silently_paired(self):
        samples = self.reports['timing']['cases'][0]['conditions']['cold']['samples']
        samples[0]['us']['candidate'] = 0
        samples[3]['order'] = ['candidate','baseline']
        result = assessor.summarize_condition(dict(samples=samples))
        self.assertEqual(result['invalid_sample_indices'],[0,3])
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['pair_ratios']),10)
        self.assertIsNone(result['runners']['candidate']['raw_gross_us'][0])

    def test_missing_report_preserves_other_mode(self):
        (self.root/'timing.json').unlink()
        result = assessor.assess(self.root)
        self.assertFalse(result['complete'])
        self.assertTrue(result['modes']['activity']['worker_passed'])
        self.assertEqual(result['report_files']['timing']['error'],'Report missing')

    def test_output_is_deterministic_and_cli_does_not_execute_worker(self):
        output = self.root/'assessment.json'
        args = ['--artifacts-dir',str(self.root),'--output',str(output)]
        with patch('builtins.print'):
            self.assertEqual(assessor.main(args),0)
            first = output.read_bytes()
            self.assertEqual(assessor.main(args),0)
        self.assertEqual(output.read_bytes(),first)


if __name__ == '__main__':
    unittest.main()
