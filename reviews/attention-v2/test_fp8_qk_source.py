"""A2-13 source/contract checks. Does not import or execute candidate/Triton."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
PARENT_SHA = 'c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0'


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(x):
    return ast.dump(x, include_attributes=False)


def expression(s):
    return ast.parse(s, mode='eval').body


def route(tree, cfg):
    # Execute only inspected pure shape predicates, never any candidate code.
    for n in fn(tree, 'kernel_fn').body[1:]:
        if isinstance(n, ast.If):
            if eval(compile(ast.Expression(n.test), 'shape predicate', 'eval'),
                    {'__builtins__': {}}, {'cfg': cfg}):
                return n.body[0].value.func.id
        elif isinstance(n, ast.Return):
            return n.value.func.id
    raise AssertionError('no route')


class SourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_bytes = (HERE/'mixed_fa3_decode_parallel_merge.py').read_bytes()
        cls.old = ast.parse(cls.parent_bytes)
        cls.source = (HERE/'mixed_fa3_fp8_qk.py').read_text()
        cls.new = ast.parse(cls.source)

    def test_frozen_parent_and_python_syntax(self):
        self.assertEqual(hashlib.sha256(self.parent_bytes).hexdigest(), PARENT_SHA)
        compile(self.source, 'syntax only, not Triton compilation', 'exec')

    def test_only_declared_qk_edits_reconstruct_entire_parent(self):
        new = copy.deepcopy(self.new)
        # Deliberate documentation change plus one additional operand selector.
        self.assertIsInstance(new.body[0], ast.Expr)
        new.body[0] = copy.deepcopy(self.old.body[0])
        selector = next(n for n in new.body if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_QK_DOT')
        self.assertEqual(dump(selector.value), dump(expression('tl.float32 if _INTERP else tl.float8e4nv')))
        new.body.remove(selector)
        fused = fn(new, '_fused')
        argument = next(a for a in fused.args.args if a.arg == 'QK_DOT')
        self.assertEqual(dump(argument.annotation), dump(expression('tl.constexpr')))
        fused.args.args.remove(argument)
        replacements = {
            'q.to(tl.bfloat16).to(tl.float8e4nv).to(QK_DOT)': '(q * qk_factor).to(tl.bfloat16).to(DOT)',
            '(sm_scale * qk_factor) * ks * LOG2E': 'sm_scale * ks * LOG2E',
            'tl.load(kc_ptr + kv_off, mask=t_ok[:, None], other=0.0).to(QK_DOT)':
                'tl.load(kc_ptr + kv_off, mask=t_ok[:, None], other=0.0).to(DOT)',
            'tl.dot(q, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)':
                'tl.dot(q, tl.trans(k))',
            'tl.dot(q, tl.trans(kq.to(QK_DOT)), out_dtype=tl.float32, max_num_imprecise_acc=0)':
                'tl.dot(q, tl.trans(kq.to(tl.float32).to(DOT)))',
        }
        counts = dict.fromkeys(replacements, 0)

        class Restore(ast.NodeTransformer):
            def visit(self, node):
                for before, after in replacements.items():
                    if dump(node) == dump(expression(before)):
                        counts[before] += 1
                        return expression(after)
                return super().visit(node)
        Restore().visit(fused)
        self.assertEqual(list(counts.values()), [1]*5)
        host = fn(new, '_e2_kernel_fn')
        launch = next(n for n in ast.walk(host) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
                      and n.func.value.id == '_fused')
        keyword = next(k for k in launch.keywords if k.arg == 'QK_DOT')
        self.assertEqual(dump(keyword.value), dump(expression('_QK_DOT')))
        launch.keywords.remove(keyword)
        self.assertEqual(dump(new), dump(self.old))

    def test_pv_remains_bf16_and_fp32_softmax_partials(self):
        def pv(tree):
            return [dump(n) for n in ast.walk(fn(tree, '_fused')) if isinstance(n, ast.Call)
                    and dump(n.func) == dump(expression('tl.dot')) and not
                    (isinstance(n.args[0], ast.Name) and n.args[0].id == 'q')]
        self.assertEqual(len(pv(self.old)), 2)
        self.assertEqual(pv(self.old), pv(self.new))
        for name in ('_norm_rope', '_merge', '_tickets', 'kernel_fn'):
            self.assertEqual(dump(fn(self.old, name)), dump(fn(self.new, name)))

    def test_all_scored_routes_four_fallbacks_and_eight_unchanged_fa3(self):
        changed = []
        for b in (1, 16, 32):
            for length in (1, 8):
                for window, theta in ((0, 0), (2048, 500000)):
                    cfg = dict(batch=b, q_len=length, window=window, rope_theta=theta)
                    self.assertEqual(route(self.old, cfg), route(self.new, cfg))
                    if route(self.new, cfg) == '_e2_kernel_fn':
                        changed.append((b, length, window))
        self.assertEqual(set(changed), {(1,1,2048), (16,1,2048), (32,1,2048), (32,1,0)})

    def test_official_prerequisite_routes_and_nearby_dispatch(self):
        task = (HERE/'official/TASK.md').read_text()
        rows = task.split('Before timing,')[1].split('```')[1].strip().splitlines()
        self.assertEqual(len(rows), 16)
        affected = []
        for index, row in enumerate(rows, 1):
            fields = dict(part.strip().split(' ', 1) for part in row.split(','))
            sliding = fields['layer'] == 'sliding'
            cfg = dict(batch=int(fields['batch']), q_len=int(fields.get('q_len', 1)),
                       window=2048 if sliding else 0, rope_theta=500000 if sliding else 0)
            self.assertEqual(route(self.new, cfg), route(self.old, cfg))
            if route(self.new, cfg) == '_e2_kernel_fn':
                affected.append(index)
        self.assertEqual(affected, [1,2,4,5,6,7,8,11,12,14,15,16])
        for b in (2,3,4,8,15,17,31,33):
            for length in (1,8):
                for window,theta in ((0,0),(2048,500000)):
                    cfg = dict(batch=b,q_len=length,window=window,rope_theta=theta)
                    self.assertEqual(route(self.new,cfg),route(self.old,cfg))

    def test_no_precision_change_to_fa3_functions(self):
        names = [n.name for n in self.old.body if isinstance(n, ast.FunctionDef)
                 and n.name.startswith('_fa3')]
        self.assertGreater(len(names), 3)
        for name in names:
            self.assertEqual(dump(fn(self.old,name)),dump(fn(self.new,name)))

    def test_both_qk_calls_use_fp32_result_precise_accumulation_request(self):
        dots = [n for n in ast.walk(fn(self.new, '_fused')) if isinstance(n, ast.Call)
                and dump(n.func) == dump(expression('tl.dot'))
                and isinstance(n.args[0],ast.Name) and n.args[0].id=='q']
        self.assertEqual(len(dots),2)
        for dot in dots:
            self.assertEqual({k.arg:dump(k.value) for k in dot.keywords},
                             {'out_dtype':dump(expression('tl.float32')),
                              'max_num_imprecise_acc':dump(expression('0'))})
        # Both QK reductions are D64. BLOCK_L16 is N, not K; PV's K16 stays BF16.
        for m,n,k in ((16,64,64),(16,16,64),(64,64,64),(64,16,64)):
            self.assertEqual(k%32,0)
            self.assertGreaterEqual(m,1)
            self.assertGreaterEqual(n,1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
