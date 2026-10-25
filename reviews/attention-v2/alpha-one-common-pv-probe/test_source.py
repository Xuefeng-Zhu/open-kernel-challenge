"""Offline source/recurrence checks. They do not establish native equivalence."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import struct
import unittest

HERE = Path(__file__).resolve().parent
CONTROL_SHA = '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
FAILED_SHA = 'e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764'
OLD = '        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(DOT), v)\n'


def dump(node):
    return ast.dump(node, include_attributes=False)


def functions(tree):
    return {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}


def is_dot(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'dot'


class FoldPageLookaheadFalse(ast.NodeTransformer):
    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Name) and node.test.id == 'PAGE_LOOKAHEAD':
            return node.orelse
        return node


def predicate(alpha, row_ok):
    return sum(bool(ok and value != 1.0) for value, ok in zip(alpha, row_ok)) == 0


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


class CommonPVSourceChecks(unittest.TestCase):
    def setUp(self):
        self.before_text = (HERE/'baseline.py').read_text()
        self.after_text = (HERE/'candidate.py').read_text()
        self.before = ast.parse(self.before_text)
        self.after = ast.parse(self.after_text)
        self.patch = next(x for x in ast.walk(functions(self.after)['_fused'])
            if isinstance(x, ast.If) and isinstance(x.test, ast.Name)
            and x.test.id == 'PAGE_LOOKAHEAD'
            and any(isinstance(y, ast.Name) and y.id == 'all_alpha_one' for y in ast.walk(x)))

    def test_exact_control_and_minimal_reverse_patch(self):
        self.assertEqual(hashlib.sha256(self.before_text.encode()).hexdigest(), CONTROL_SHA)
        self.assertEqual((HERE.parent/'fusion-lookahead-probe/candidate.py').read_text(), self.before_text)
        lines = self.after_text.splitlines(keepends=True)
        patch = ''.join(lines[self.patch.lineno-1:self.patch.end_lineno])
        self.assertEqual(self.after_text.replace(patch, OLD, 1), self.before_text)
        for name, node in functions(self.before).items():
            if name != '_fused':
                self.assertEqual(dump(node), dump(functions(self.after)[name]), name)

    def test_constexpr_false_recovers_entire_control_ast(self):
        before = FoldPageLookaheadFalse().visit(copy.deepcopy(self.before))
        after = FoldPageLookaheadFalse().visit(copy.deepcopy(self.after))
        self.assertEqual(dump(before), dump(after))
        self.assertEqual(dump(self.patch.orelse[0]), dump(ast.parse(OLD.strip()).body[0]))

    def test_exactly_one_common_target_pv_and_multiply_only_inner_region(self):
        inner = next(x for x in self.patch.body if isinstance(x, ast.If))
        self.assertEqual(dump(inner.test), dump(ast.parse('not all_alpha_one', mode='eval').body))
        self.assertEqual(len(inner.body), 1)
        self.assertEqual(dump(inner.body[0]), dump(ast.parse('acc = acc * alpha[:, None]').body[0]))
        self.assertEqual(inner.orelse, [])
        self.assertFalse(any(is_dot(x) for x in ast.walk(inner)))
        self.assertEqual(sum(is_dot(x) for x in ast.walk(ast.Module(body=self.patch.body, type_ignores=[]))), 1)
        self.assertEqual(dump(self.patch.body[-1]), dump(ast.parse('acc = acc + tl.dot(p.to(tl.bfloat16).to(DOT), v)').body[0]))
        self.assertIsInstance(self.patch.body[-1].value.left, ast.Name)
        self.assertEqual(self.patch.body[-1].value.left.id, 'acc')
        # No separate PV-from-zero assignment or new dot-precision parameter.
        dot = next(x for x in ast.walk(self.patch.body[-1]) if is_dot(x))
        self.assertEqual(len(dot.args), 2)
        self.assertEqual(dot.keywords, [])

    def test_exact_predicate_uses_only_valid_query_rows(self):
        assignment = self.patch.body[0]
        expected = ast.parse('all_alpha_one = tl.sum((row_ok & (alpha != 1.0)).to(tl.int32), 0) == 0').body[0]
        self.assertEqual(dump(assignment), dump(expected))
        valid = [True]*8 + [False]*8
        self.assertTrue(predicate([1.0]*8+[math.nan, 0., .5, -1., math.inf, -math.inf, -0., 2.], valid))
        for value in [0., -0., .5, math.nan, math.inf, math.nextafter(1., 0.), math.nextafter(1., 2.)]:
            alpha = [1.0]*16
            alpha[5] = value
            self.assertFalse(predicate(alpha, valid))

    def test_denominator_tail_and_dispatch_are_unchanged(self):
        tail = '    if split == SPLITS - 1:\n'
        self.assertEqual(self.before_text.split(tail, 1)[1], self.after_text.split(tail, 1)[1])
        self.assertEqual(self.after_text.count('l_i = l_i * alpha + tl.sum(p, 1)'), 2)
        for name in ['_e2_kernel_fn', '_merge', 'kernel_fn']:
            self.assertEqual(dump(functions(self.before)[name]), dump(functions(self.after)[name]))
        self.assertEqual(self.before_text.count('max_num_imprecise_acc=0'), self.after_text.count('max_num_imprecise_acc=0'))

    def test_scalar_carry_recurrence_model_and_first_tile(self):
        valid = [True]*8+[False]*8
        values = [0., -0., 2.**-149, -2.**-149, 2.**-126, -2.**-126, 1., -1., 2.**100]
        for alpha in [[1.]*8+[0.]*8, [.5]+[1.]*15, [0.]*16]:
            skip = predicate(alpha, valid)
            for row in range(8):
                for old in values:
                    slow_c = f32(old*alpha[row])
                    common_c = old if skip else f32(old*alpha[row])
                    self.assertEqual(struct.pack('<f', slow_c), struct.pack('<f', common_c))
                    for update in [0., -0., 2.**-149, 1., -1.]:
                        self.assertEqual(struct.pack('<f', f32(slow_c+update)), struct.pack('<f', f32(common_c+update)))
        self.assertFalse(predicate([0.]*16, valid))
        # Host scalar RN model only: no MMA order, compiler or GPU FTZ proof.

    def test_failed_variant_stays_immutable_and_is_not_the_control(self):
        failed = HERE.parent/'alpha-one-rescale-probe/candidate.py'
        self.assertEqual(hashlib.sha256(failed.read_bytes()).hexdigest(), FAILED_SHA)
        self.assertNotEqual(hashlib.sha256(self.after_text.encode()).hexdigest(), FAILED_SHA)
        self.assertNotEqual(hashlib.sha256(self.after_text.encode()).hexdigest(), CONTROL_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
