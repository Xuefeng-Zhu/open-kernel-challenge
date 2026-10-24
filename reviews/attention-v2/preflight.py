"""Local attention-v2 preflight. Executes host helpers/CPU equations, never Triton."""
import ast
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'official'))
spec = importlib.util.spec_from_file_location('v2_official_reference', ROOT / 'official/reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(4)
SOURCE = (ROOT / 'submission.py').read_text()
TREE = ast.parse(SOURCE)
helpers = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in {'_tickets', '_num_splits'}]
namespace = dict(torch=torch, NUM_SMS=132, _TICKETS={})
exec(compile(ast.Module(body=helpers, type_ignores=[]), '<actual-host-helpers>', 'exec'), namespace)
RESULTS = {'kind': 'CPU host-helper execution and analytical/model tests only',
           'triton_executed': False, 'gpu_executed': False,
           'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(),
           'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                           'cuda_available': torch.cuda.is_available()}, 'cases': []}


def rope_cpu(x, pos, theta):
    """Model source exp2/log2 frequency and interleaved rotation using CPU Torch math."""
    d = x.shape[-1]
    dimensions = torch.arange(d)
    frequency = torch.exp2(-(2.0 * (dimensions // 2).float() / d) * torch.log2(torch.tensor(theta)))
    angle = pos.float()[..., None, None] * frequency
    partner = x[..., dimensions ^ 1]
    sign = torch.where(dimensions % 2 == 0, -1.0, 1.0)
    return x * angle.cos() + sign * partner * angle.sin()


def masks(seq, length, window, splits, block_n=64):
    lo = max(seq - window + 1, 0) if window else 0
    lo = min(lo, seq)
    span = seq - lo
    per = ((span + splits - 1) // splits + block_n - 1) // block_n * block_n
    parts = [list(range(lo + s * per, min(lo + (s + 1) * per, seq))) for s in range(splits)]
    visible = []
    for j in range(length):
        qpos = seq + j
        old = [t for part in parts for t in part if not window or t > qpos - window]
        # The source's new-token visibility has causality but no lower window bound.
        new = [seq + t for t in range(min(length, 16)) if seq + t <= qpos]
        visible.append(old + new)
    return visible


def attention_cpu_model(data):
    """Same mathematical ordering/rounding, CPU operations; not native GPU semantics."""
    cfg = data['config']
    batch, length, hq, hkv, d = (cfg[k] for k in ['batch', 'q_len', 'num_q_heads', 'num_kv_heads', 'head_dim'])
    group = hq // hkv
    q, k, v, gate = reference.split_qkv(data)
    pos = data['seq_lens'].long()[:, None] + torch.arange(length)
    q, k = reference._rms(q), reference._rms(k)
    if cfg['rope_theta']:
        q = rope_cpu(q, pos, cfg['rope_theta'])
        k = rope_cpu(k, pos, cfg['rope_theta'])
    q = (q * cfg['qk_scale_factor']).bfloat16().float()
    kn = (k / data['k_scale']).to(torch.float8_e4m3fn)
    vn = (v.float() / data['v_scale']).to(torch.float8_e4m3fn)
    keys_flat = data['k_cache'].view(-1, hkv, d).float()
    vals_flat = data['v_cache'].view(-1, hkv, d).float()
    span_max = cfg['window'] or data['page_table'].shape[1] * cfg['page_size']
    splits = namespace['_num_splits'](batch, hkv, span_max, 64)
    out = torch.empty((batch, length, hq, d), dtype=torch.float32)
    log2e = 1.4426950408889634

    def update(acc, maximum, total, logits, values):
        new_max = torch.maximum(maximum, logits.max(-1).values)
        safe = torch.where(torch.isneginf(new_max), 0.0, new_max)
        alpha = torch.exp2(maximum - safe)
        probability = torch.exp2(logits - safe[:, None])
        total = total * alpha + probability.sum(-1)
        acc = acc * alpha[:, None] + probability.bfloat16().float() @ values
        return acc, new_max, total

    for b in range(batch):
        seq = int(data['seq_lens'][b])
        lo = min(max(seq - cfg['window'] + 1, 0), seq) if cfg['window'] else 0
        per = (((seq - lo + splits - 1) // splits + 63) // 64) * 64
        for h in range(hkv):
            query = q[b, :, h * group:(h + 1) * group].reshape(length * group, d)
            qpos = seq + torch.arange(length).repeat_interleave(group)
            partials = []
            for split in range(splits):
                acc = torch.zeros((length * group, d))
                maximum = torch.full((length * group,), -math.inf)
                total = torch.zeros((length * group,))
                stop = min(lo + (split + 1) * per, seq)
                for start in range(lo + split * per, stop, 64):
                    ts = torch.arange(start, min(start + 64, stop))
                    page = data['page_table'][b, ts // cfg['page_size']].long().clamp_min(0)
                    rows = page * cfg['page_size'] + ts % cfg['page_size']
                    key = keys_flat[rows, h].bfloat16().float()
                    values = vals_flat[rows, h].bfloat16().float()
                    logits = query @ key.T * (cfg['sm_scale'] * data['k_scale'] * log2e)
                    if cfg['window']:
                        logits = torch.where(ts[None] > qpos[:, None] - cfg['window'], logits, -math.inf)
                    acc, maximum, total = update(acc, maximum, total, logits, values)
                if split == splits - 1:
                    new_pos = seq + torch.arange(min(length, 16))
                    key = kn[b, :16, h].float().bfloat16().float()
                    values = vn[b, :16, h].float().bfloat16().float()
                    logits = query @ key.T * (cfg['sm_scale'] * data['k_scale'] * log2e)
                    logits = torch.where(new_pos[None] <= qpos[:, None], logits, -math.inf)
                    acc, maximum, total = update(acc, maximum, total, logits, values)
                partials.append((acc, maximum, total))
            acc = torch.zeros((length * group, d))
            maximum = torch.full((length * group,), -math.inf)
            total = torch.zeros((length * group,))
            for value, m_s, l_s in partials:
                m_n = torch.maximum(maximum, m_s)
                m_z = torch.where(torch.isneginf(m_n), 0.0, m_n)
                w_t, w_s = torch.exp2(maximum - m_z), torch.exp2(m_s - m_z)
                total = total * w_t + l_s * w_s
                acc = acc * w_t[:, None] + value * w_s[:, None]
                maximum = m_n
            a = acc * data['v_scale'] / total[:, None]
            a = reference._rms(a)
            heads_gate = gate[b].view(length, hq, d)[:, h * group:(h + 1) * group].reshape(length * group, d)
            out[b, :, h * group:(h + 1) * group] = (a * heads_gate.float().sigmoid()).view(length, group, d)
    return out.reshape(batch, length, hq * d).bfloat16(), kn, vn, splits


class Preflight(unittest.TestCase):
    def test_normalization_only_repairs_five_links(self):
        import re
        original = (ROOT / 'paste.original.txt').read_text()
        fixed, count = re.subn(r'\[([A-Za-z_]\w*(?:\.\w+)+)\]\(http://\1\)', r'\1', original)
        self.assertEqual(count, 5)
        self.assertEqual(fixed, SOURCE)
        compile(SOURCE, 'submission.py', 'exec')

    def test_actual_host_helper_shares_ticket_storage(self):
        tickets = namespace['_tickets']
        a, b = tickets(torch.device('cpu'), 2), tickets(torch.device('cpu'), 128)
        self.assertIs(a, b)
        self.assertEqual(a.numel(), 4096)
        RESULTS['host_ticket_alias'] = {'same_object': True, 'capacity': a.numel(), 'device': 'cpu', 'gpu_execution': False}

    def test_concurrent_ticket_counterexample(self):
        # Sequentially consistent atomics already fail: stronger memory ordering cannot fix isolation.
        traces = {}
        for splits in [2, 3, 5, 9, 16, 24, 33, 66, 132]:
            counter, scratch, outputs, trace = 0, {'A': set(), 'B': set()}, {}, []
            order = ([('A', i) for i in range(splits - 1)] + [('B', splits - 1)] +
                     [('B', i) for i in range(splits - 1)] + [('A', splits - 1)])
            for call, split in order:
                scratch[call].add(split)
                prev = counter
                counter += 1
                elected = prev == splits - 1
                missing = sorted(set(range(splits)) - scratch[call]) if elected else []
                trace.append(dict(call=call, split=split, previous_ticket=prev, elected=elected, missing_splits=missing))
                if elected:
                    outputs[call] = not missing
                    counter = 0
            self.assertTrue(outputs['A'])
            self.assertFalse(outputs['B'])
            self.assertEqual(counter, 0)
            if splits in [2, 16]:
                traces[str(splits)] = trace
        RESULTS['concurrent_model_counterexamples'] = traces
        RESULTS['concurrent_split_counts_tested'] = [2, 3, 5, 9, 16, 24, 33, 66, 132]

    def test_serial_ticket_reset_replays(self):
        for splits in [1, 2, 3, 16, 17, 33, 132]:
            counter = 0
            for replay in range(7):
                arrived = set()
                elections = 0
                for split in reversed(range(splits)):
                    arrived.add(split)
                    previous = counter
                    counter += 1
                    if previous == splits - 1:
                        self.assertEqual(len(arrived), splits)
                        elections += 1
                        counter = 0
                self.assertEqual(elections, 1)
                self.assertEqual(counter, 0)

    def test_exact_published_window_masks(self):
        cases = 0
        for seq, length, window, splits in itertools.product([0, 1, 10, 2047, 2048, 2049, 5000], [1, 8], [0, 2048], [1, 2, 3, 16, 132]):
            for j, got in enumerate(masks(seq, length, window, splits)):
                p = seq + j
                expected = list(range(max(0, p - window + 1) if window else 0, p + 1))
                self.assertEqual(got, expected)
            cases += 1
        RESULTS['mask_cases'] = cases
        RESULTS['mask_note'] = 'Analytical set coverage, not Triton address/mask execution; seq0 is extra analytical coverage'

    def test_noncontract_generalization_counterexamples(self):
        self.assertNotEqual(masks(10, 8, 2, 2)[7], list(range(16, 18)))
        covered_new = [i for i in range(16) if i < 17]
        self.assertNotIn(16, covered_new)
        RESULTS['outside_contract_limits'] = ['window2/L8 omits new-key lower bound', 'L17 exceeds BLOCK_L16; new token16 not written']

    def test_official_cpu_oracle_against_equation_model(self):
        cases = [
            dict(batch=1, ctx=10, heads='tp8', layer='sliding', q_len=1, page=64),
            dict(batch=1, ctx=10, heads='tp8', layer='sliding', q_len=8, page=64),
            dict(batch=2, ctx=2050, heads='tp8', layer='sliding', q_len=8, page=1, spread='mixed', dist='edges'),
            dict(batch=1, ctx=2048, heads='tp8', layer='sliding', q_len=8, page=1, dist='edges', kv_scale=1/64),
            dict(batch=1, ctx=4097, heads='tp8', layer='full', q_len=1, page=1, dist='coherent'),
            dict(batch=1, ctx=8193, heads='tp8', layer='full', q_len=8, page=1, dist='coherent'),
            dict(batch=2, ctx=129, heads='tp8', layer='full', q_len=1, page=64, spread='mixed', kv_scale=1/64),
            dict(batch=1, ctx=65, heads='dp', layer='full', q_len=8, page=64, dist='soft'),
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=case):
                data = reference.generate_input(**case, seed=7200+i, device='cpu')
                output, k, v, splits = attention_cpu_model(data)
                raw = torch.cat([p.contiguous().view(torch.uint8).flatten() for p in [output, k, v]])
                ok, message = reference.check_implementation(data, raw)
                RESULTS['cases'].append(dict(case=case, passed=ok, message=message, splits=splits))
                self.assertTrue(ok, message)
                self.assertTrue(torch.equal(reference.new_slots(data).flatten(), data['cache_loc']))
                expected_pos = (data['seq_lens'][:, None] + torch.arange(case['q_len'])).flatten()
                self.assertTrue(torch.equal(expected_pos, data['positions']))

    def test_long_position_rope_cpu_difference(self):
        gen = torch.Generator().manual_seed(17129)
        x = torch.randn((1, 2, 2, 64), generator=gen).bfloat16()
        measurements = []
        for positions, scope in [([80000, 86007], 'published scored sliding-position range'),
                                 ([1048576, 1048583], 'extra robustness; published million-context case is full/no-RoPE')]:
            pos = torch.tensor([positions])
            model = rope_cpu(reference._rms(x), pos, 500000.0)
            oracle = reference._rope(reference._rms(x), pos, 500000.0)
            error = float(reference.relative_l2(model, oracle))
            measurements.append(dict(positions=positions, scope=scope, relative_l2=error))
            self.assertTrue(torch.isfinite(model).all())
        RESULTS['rope_cpu_differences'] = measurements
        RESULTS['rope_note'] = 'CPU Torch exp2/log2/trig only; not Triton sin/cos or GPU error bound'



if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Preflight)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    RESULTS.update(tests_run=outcome.testsRun, failures=len(outcome.failures), errors=len(outcome.errors), passed=outcome.wasSuccessful())
    (ROOT/'cpu-results.json').write_text(json.dumps(RESULTS, indent=2)+'\n')
    raise SystemExit(not outcome.wasSuccessful())
