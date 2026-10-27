"""Local source invariants and CPU models, not CUDA/Triton execution."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def expr(source):
    return ast.parse(source, mode='eval').body


def is_flag(node):
    return isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'DEFER_DENOM'


def historical_partition(seq, split, chunk=8192):
    return split * chunk, min((split + 1) * chunk, seq)


def masses(scores, valid, n=64, dtype=torch.float32, deferred=False):
    """Independent CPU online-softmax recurrence; logits are already scaled.

    All tensor operations are explicitly CPU. No tensor-core operations, native
    exp2 approximation, compiler layout or GPU graph execution is modeled.
    """
    scores = scores.to(device='cpu', dtype=dtype)
    rows, length = scores.shape
    peak = torch.full((rows,), -float('inf'), dtype=dtype, device='cpu')
    scalar = torch.zeros(rows, dtype=dtype, device='cpu')
    columns = torch.zeros((rows, n), dtype=dtype, device='cpu')
    for start in range(0, length, n):
        tile = torch.full((rows, n), -float('inf'), dtype=dtype, device='cpu')
        width = min(n, length-start)
        tile[:, :width] = torch.where(valid[:, start:start+width], scores[:, start:start+width], -float('inf'))
        new_peak = torch.maximum(peak, tile.amax(1))
        safe = torch.where(torch.isneginf(new_peak), 0., new_peak)
        alpha = torch.exp2(peak-safe)
        probability = torch.exp2(tile-safe[:, None])
        if deferred:
            columns = columns * alpha[:, None] + probability
        else:
            scalar = scalar * alpha + probability.sum(1)
        peak = new_peak
    return peak, columns.sum(1) if deferred else scalar


class SourceInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (ROOT/'mixed_fa3_fp8_qk.py').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
        cls.old = ast.parse(raw)
        cls.raw = (ROOT/'mixed_fa3_deferred_denom.py').read_text()
        cls.new = ast.parse(cls.raw)

    def test_syntax_and_full_parent_reconstruction(self):
        compile(self.raw, 'syntax only', 'exec')
        tree = copy.deepcopy(self.new)
        tree.body[0] = copy.deepcopy(self.old.body[0])
        fused = fn(tree, '_fused')
        flag = fused.args.args.pop()
        self.assertEqual(flag.arg, 'DEFER_DENOM')
        self.assertEqual(dump(flag.annotation), dump(expr('tl.constexpr')))
        top_flags = [n for n in fused.body if is_flag(n)]
        self.assertEqual(len(top_flags), 2)
        self.assertEqual(dump(top_flags[0].body[0]), dump(ast.parse('l_columns = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)').body[0]))
        self.assertEqual(dump(top_flags[1].body[0]), dump(ast.parse('l_i = tl.sum(l_columns, 1)').body[0]))
        for n in top_flags:
            self.assertFalse(n.orelse)
            fused.body.remove(n)
        loop = next(n for n in fused.body if isinstance(n, ast.For))
        flag = next(n for n in loop.body if is_flag(n))
        self.assertEqual(dump(flag.body[0]), dump(ast.parse('l_columns = l_columns * alpha[:, None] + p').body[0]))
        index = loop.body.index(flag)
        loop.body[index:index+1] = flag.orelse
        host = fn(tree, '_e2_kernel_fn')
        predicate = next(n for n in host.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'defer_denom')
        self.assertEqual(dump(predicate.value), dump(expr('batch == 32 and length == 1 and window == 0 and not rope')))
        host.body.remove(predicate)
        launch = next(n for n in ast.walk(host) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
                      and n.func.value.id == '_fused')
        keyword = next(k for k in launch.keywords if k.arg == 'DEFER_DENOM')
        self.assertEqual(dump(keyword.value), dump(expr('defer_denom')))
        launch.keywords.remove(keyword)
        self.assertEqual(dump(tree), dump(self.old))

    def test_single_scored_path_and_config_only_guard(self):
        predicate = next(n.value for n in fn(self.new, '_e2_kernel_fn').body
                         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                         and n.targets[0].id == 'defer_denom')
        code = compile(ast.Expression(predicate), 'shape predicate', 'eval')
        active = []
        for b in (1,16,32):
            for length in (1,8):
                for window in (0,2048):
                    scope = dict(batch=b, length=length, window=window, rope=bool(window))
                    if eval(code, {'__builtins__': {}}, scope):
                        active.append((b,length,window))
        self.assertEqual(active, [(32,1,0)])
        self.assertEqual({n.id for n in ast.walk(predicate) if isinstance(n, ast.Name)},
                         {'batch','length','window','rope'})
        self.assertEqual(dump(fn(self.new,'kernel_fn')), dump(fn(self.old,'kernel_fn')))

    def test_reduction_precedes_new_token_update_and_tickets_unchanged(self):
        fused = fn(self.new, '_fused')
        reduction = next(i for i,n in enumerate(fused.body) if is_flag(n)
                         and isinstance(n.body[0],ast.Assign) and n.body[0].targets[0].id=='l_i')
        live = fused.body[reduction+1]
        self.assertEqual(dump(live.test), dump(expr('split == SPLITS - 1')))
        old_live = next(n for n in fn(self.old,'_fused').body if isinstance(n,ast.If)
                        and dump(n.test)==dump(expr('split == SPLITS - 1')))
        self.assertEqual(dump(live),dump(old_live))
        for name in ('_merge','_tickets','_norm_rope'):
            self.assertEqual(dump(fn(self.new,name)),dump(fn(self.old,name)))


class DenominatorModels(unittest.TestCase):
    def test_fp64_identity_varied_lengths_masks_and_maxima(self):
        generator = torch.Generator(device='cpu').manual_seed(14001)
        for length in (0,1,15,16,63,64,65,127,128,129,8191,8192,8193):
            scores = torch.randn((3,length),generator=generator,device='cpu',dtype=torch.float64)*20
            valid = torch.rand((3,length),generator=generator,device='cpu') > .2
            for deferred in (False,True):
                peak,l = masses(scores,valid,dtype=torch.float64,deferred=deferred)
                if length:
                    masked = torch.where(valid,scores,-float('inf'))
                    maximum = masked.amax(1)
                    safe = torch.where(torch.isneginf(maximum),0.,maximum)
                    reference = torch.exp2(masked-safe[:,None]).sum(1)
                    torch.testing.assert_close(peak,maximum,rtol=0,atol=0)
                    torch.testing.assert_close(l,reference,rtol=2e-14,atol=1e-12)
                else:
                    self.assertTrue(bool(torch.isneginf(peak).all()))
                    self.assertTrue(bool((l==0).all()))

    def test_fp32_error_and_rms_epsilon_dependency(self):
        generator=torch.Generator(device='cpu').manual_seed(14002)
        for span in (65,1023,8192):
            for spread in (.01,1.,50.,1000.):
                s=torch.randn((4,span),generator=generator,device='cpu')*spread
                valid=torch.ones_like(s,dtype=torch.bool)
                peak,a=masses(s,valid)
                peak2,b=masses(s,valid,deferred=True)
                torch.testing.assert_close(peak,peak2,rtol=0,atol=0)
                torch.testing.assert_close(a,b,rtol=4e-6,atol=2e-4)
                # Identical numerator to isolate the changed denominator's effect.
                acc=torch.randn((4,64),generator=generator,device='cpu')*1e-4
                for v_scale in (1.,1/64):
                    def finish(l):
                        x=acc*v_scale/l[:,None]
                        return x*torch.rsqrt(x.square().mean(1,keepdim=True)+1e-5)
                    torch.testing.assert_close(finish(a),finish(b),rtol=5e-6,atol=1e-7)
                # A deliberately tiny numerator makes dropping epsilon invalid.
                x=acc/a[:,None]
                wrong=x*torch.rsqrt(x.square().mean(1,keepdim=True))
                self.assertGreater(float((wrong-finish(a)).abs().max()),.1)

    def test_empty_split_partial_tiles_and_new_token_transition(self):
        for seq in (0,1,8191,8192,8193,80000,81920):
            intervals=[]
            for split in range(11):
                lo,hi=historical_partition(seq,split)
                actual=list(range(lo,max(lo,hi)))
                intervals.extend(actual)
                if hi<=lo:
                    peak,l=masses(torch.empty((2,0)),torch.empty((2,0),dtype=torch.bool),deferred=True)
                    self.assertTrue(bool(torch.isneginf(peak).all()))
                    self.assertTrue(bool((l==0).all()))
                    # The last split's live token makes this otherwise empty mass positive.
                    score=torch.tensor([1.,-2.])
                    safe=torch.maximum(peak,score)
                    next_l=l*torch.exp2(peak-safe)+torch.exp2(score-safe)
                    self.assertTrue(bool((next_l==1).all()))
            self.assertEqual(intervals,list(range(seq)))

    def test_fresh_length_and_content_changes_are_not_reused(self):
        generator=torch.Generator(device='cpu').manual_seed(14003)
        a=torch.randn((2,130),generator=generator,device='cpu')
        b=a.clone(); b[:,80:]+=3
        outputs=[]
        for value,length in ((a,130),(b,81),(a,130)):
            outputs.append(masses(value[:,:length],torch.ones((2,length),dtype=torch.bool),deferred=True))
        self.assertTrue(torch.equal(outputs[0][0],outputs[2][0]))
        self.assertTrue(torch.equal(outputs[0][1],outputs[2][1]))
        self.assertFalse(torch.equal(outputs[0][1],outputs[1][1]))
        # This is a CPU dependency test, not CUDA graph A/B/A evidence.


if __name__ == '__main__':
    unittest.main(verbosity=2)
