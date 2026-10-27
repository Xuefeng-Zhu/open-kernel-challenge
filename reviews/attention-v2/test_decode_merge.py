"""A2-10 source and CPU merge algebra checks; no Triton/GPU execution."""
import ast
import copy
import math
from pathlib import Path
import random
import unittest

HERE = Path(__file__).resolve().parent
OLD = ast.parse((HERE/'mixed_fa3_local_tickets.py').read_text())
NEW = ast.parse((HERE/'mixed_fa3_decode_merge.py').read_text())


def fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]))


class DecodeMerge(unittest.TestCase):
    def test_partial_math_partition_and_neutral_publication_unchanged(self):
        old, new = fn(OLD, '_fused'), fn(NEW, '_fused')
        self.assertEqual(dump(old.body[:-1]), dump(new.body[:-1]))
        self.assertEqual(dump(old.body[-1].body), dump(new.body[-1].body))
        self.assertEqual(dump(old.body[-1].orelse[:4]), dump(new.body[-1].orelse[:4]))
        conditional = new.body[-1].orelse[4]
        self.assertEqual(ast.unparse(conditional.test), 'not SEPARATE_MERGE')
        self.assertEqual(dump(old.body[-1].orelse[4:]), dump(conditional.body))

    def test_merge_recurrence_rounding_and_output_exact_copy(self):
        election = fn(OLD, '_fused').body[-1].orelse[-1]
        math_nodes = election.body[:-1]  # exclude only ticket reset
        self.assertEqual(dump(math_nodes), dump(fn(NEW, '_merge').body[-len(math_nodes):]))
        attrs = {n.attr for n in ast.walk(fn(NEW, '_merge')) if isinstance(n, ast.Attribute)}
        self.assertNotIn('atomic_add', attrs)
        self.assertNotIn('atomic_xchg', attrs)

    def test_other_functions_and_complete_entrypoint_unchanged(self):
        for node in OLD.body:
            if isinstance(node, ast.FunctionDef) and node.name not in ['_fused', '_e2_kernel_fn']:
                self.assertEqual(ast.dump(node), ast.dump(fn(NEW, node.name)), node.name)
        compile((HERE/'mixed_fa3_decode_merge.py').read_bytes(), 'candidate', 'exec')

    def test_exact_host_changes_and_same_partition_capacity(self):
        old, new = copy.deepcopy(fn(OLD, '_e2_kernel_fn')), copy.deepcopy(fn(NEW, '_e2_kernel_fn'))
        # Undo only the declared new flag, local-counter bypass, launch constexpr,
        # and conditional same-stream merge. All allocation/partition logic remains.
        flag = next(i for i,n in enumerate(new.body) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'separate_merge')
        new.body.pop(flag)
        ticket = new.body[flag]
        self.assertEqual(ast.unparse(ticket.value.test), 'separate_merge')
        self.assertEqual(ast.unparse(ticket.value.body), 'out')
        ticket.value = ticket.value.orelse
        launch = new.body[flag+1].value
        self.assertEqual(launch.keywords[-3].arg, 'SEPARATE_MERGE')
        launch.keywords = [k for k in launch.keywords if k.arg != 'SEPARATE_MERGE']
        branch = new.body.pop(-2)
        self.assertEqual(ast.unparse(branch.test), 'separate_merge and splits > 1')
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_merge_with_empty_chunks_matches_direct_softmax_120_cases(self):
        rng = random.Random(20261025)
        for n in [1, 2, 7, 17, 65, 129]:
            for splits in [1, 2, 5, 11]:
                for _ in range(5):
                    logits = [rng.uniform(-15, 15) for _ in range(n)]
                    values = [[rng.uniform(-3, 3) for _ in range(4)] for _ in range(n)]
                    chunk = math.ceil(n/splits)
                    m, l, a = -math.inf, 0., [0.]*4
                    for s in range(splits):
                        lo, hi = s*chunk, min((s+1)*chunk, n)
                        local_m = max(logits[lo:hi], default=-math.inf)
                        weights = [math.exp2(x-local_m) for x in logits[lo:hi]]
                        local_l = sum(weights)
                        local_a = [sum(w*values[lo+i][d] for i,w in enumerate(weights)) for d in range(4)]
                        next_m = max(m, local_m)
                        safe = 0. if next_m == -math.inf else next_m
                        wt, ws = math.exp2(m-safe), math.exp2(local_m-safe)
                        a = [x*wt+y*ws for x,y in zip(a, local_a)]
                        l = l*wt+local_l*ws
                        m = next_m
                    mx = max(logits)
                    p = [math.exp2(x-mx) for x in logits]
                    expected = [sum(w*v[d] for w,v in zip(p, values))/sum(p) for d in range(4)]
                    for got, want in zip(a, expected):
                        self.assertAlmostEqual(got/l, want, places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
