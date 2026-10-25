"""CPU-only protocol tests. Never import candidates, torch, Triton or Modal."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired_worker', HERE/'paired_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Protocol(unittest.TestCase):
    def test_boundary_lengths_fit_full_b32_fixture(self):
        a, b = worker.boundary_lengths(0), worker.boundary_lengths(1)
        self.assertEqual(len(a), 32)
        self.assertNotEqual(a, b)
        self.assertEqual(set(a), {0,1,63,64,65,8191,8192,8193})
        self.assertEqual(set(a), set(b))
        self.assertLessEqual(max(a), worker.ADVERSARIAL['ctx'])
        self.assertEqual((worker.ADVERSARIAL['ctx']+1+63)//64, 129)
        self.assertEqual((129*64+8191)//8192, 2)
        self.assertTrue(any(x < 8192 for x in a))

    def test_case_targets_guard_and_precision(self):
        for case in (worker.TARGET, worker.ADVERSARIAL):
            self.assertEqual((case['batch'],case['q_len'],case['layer']), (32,1,'full'))
        self.assertEqual(worker.TARGET['dist'], 'coherent')
        self.assertEqual(worker.TARGET['ctx'], 80000)
        self.assertEqual(worker.ADVERSARIAL['page'], 64)
        self.assertEqual(worker.ADVERSARIAL['kv_scale'], 1/64)

    def test_order_pairs_hold_input_constant(self):
        for i in range(0, worker.WARMUP_PAIRS+worker.MEASURED_PAIRS, 2):
            a, av = worker.order_and_version(i)
            b, bv = worker.order_and_version(i+1)
            self.assertEqual(a, list(reversed(b)))
            self.assertEqual(av, bv)
        self.assertEqual(worker.WARMUP_PAIRS, 4)
        self.assertGreaterEqual(worker.MEASURED_PAIRS, 12)

    def test_aba_is_ordered_and_fails_closed(self):
        calls, emitted = [], []
        class Fake:
            def __init__(self, name, fail=False): self.name,self.fail=name,fail
            def replay_and_check(self,label):
                calls.append((self.name,label))
                return {'passed':not(self.fail and label=='B')}
        runners = {'a213':Fake('a213'), 'a215':Fake('a215')}
        worker.execute_aba(runners, lambda:None, emitted.append)
        self.assertEqual(calls, [(name,label) for label in ['A1','B','A2'] for name in ['a213','a215']])
        self.assertEqual(len(emitted), 6)
        calls.clear();emitted.clear();runners['a215']=Fake('a215',True)
        with self.assertRaises(AssertionError):
            worker.execute_aba(runners,lambda:None,emitted.append)
        self.assertFalse(any(label=='A2' for _,label in calls))
        self.assertFalse(emitted[-1]['passed'])

    def test_deadline_prevents_next_replay(self):
        class Never:
            def replay_and_check(self,label): raise AssertionError('Should not launch')
        def expired(): raise TimeoutError('Expired')
        with self.assertRaises(TimeoutError):
            worker.execute_aba({'a213':Never(),'a215':Never()},expired,lambda _:None)

    def test_summary_uses_paired_ratios_not_ratio_of_medians(self):
        samples=[{'raw_us':{'a213':float(i+1),'a215':float(i+1)/2}} for i in range(12)]
        got=worker.summarize_timing(samples)
        self.assertEqual(got['median_reverse_order_pair_ratio'],2)
        self.assertEqual(len(got['reverse_order_pair_geomeans']),6)
        with self.assertRaises(ValueError): worker.summarize_timing(samples[:-1])
        samples[0]['raw_us']['a215'] = float('nan')
        with self.assertRaises(ValueError): worker.summarize_timing(samples)


if __name__ == '__main__':
    unittest.main(verbosity=2)
