"""CPU ownership/protocol models and source invariants; no GPU execution."""
import ast
import copy
from itertools import combinations
from pathlib import Path
import unittest

import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE/'mixed_fa3_fused_full_small.py'
CANDIDATE = HERE/'mixed_fa3_local_tickets.py'


def helper(path):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_tickets')
    ns = {'torch': torch, '_TICKETS': {}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['_tickets']


def simulate(order, splits, shared):
    # Each event publishes its own partial, then does atomic increment, merge and
    # reset. Instant merge/reset is a valid strong-ordering interleaving; finding
    # a failure here requires neither weak GPU ordering nor cache assumptions.
    counters = [0, 0]
    published = [0, 0]
    merges = [0, 0]
    premature = []
    for owner in order:
        published[owner] += 1
        slot = 0 if shared else owner
        previous = counters[slot]
        counters[slot] += 1
        if previous == splits - 1:
            merges[owner] += 1
            if published[owner] != splits:
                premature.append((owner, published[owner]))
            counters[slot] = 0
    return premature, merges, counters


class InvocationTickets(unittest.TestCase):
    def test_parent_reproduces_alias_candidate_has_disjoint_live_storage(self):
        for path, shared in [(PARENT, True), (CANDIDATE, False)]:
            make = helper(path)
            a, b = make('cpu', 64), make('cpu', 2)
            self.assertEqual(a.data_ptr() == b.data_ptr(), shared)
            a[0] = 7
            self.assertEqual(int(b[0]), 7 if shared else 0)
            self.assertEqual(b.dtype, torch.int32)
            self.assertGreaterEqual(b.numel(), 2)
        fresh = make('cpu', 8193)
        self.assertEqual(fresh.numel(), 8193)
        self.assertEqual(int(torch.count_nonzero(fresh)), 0)

    def test_concrete_shared_counter_failure_with_separate_partials(self):
        premature, merges, _ = simulate([0, 1, 0, 1], 2, True)
        self.assertEqual(premature, [(1, 1)])
        self.assertEqual(merges, [0, 2])
        self.assertEqual(simulate([0, 1, 0, 1], 2, False), ([], [1, 1], [0, 0]))

    def test_all_96_two_call_interleavings_are_safe_with_private_counters(self):
        count = 0
        bad_shared = 0
        for splits in [2, 3, 4]:
            for a_positions in combinations(range(2*splits), splits):
                chosen = set(a_positions)
                order = [0 if i in chosen else 1 for i in range(2*splits)]
                self.assertEqual(simulate(order, splits, False), ([], [1, 1], [0, 0]))
                bad_shared += bool(simulate(order, splits, True)[0])
                count += 1
        self.assertEqual(count, 96)
        self.assertGreater(bad_shared, 0)

    def test_sequential_shared_protocol_is_not_falsely_marked_broken(self):
        for splits in [1, 2, 16]:
            self.assertEqual(simulate([0]*splits+[1]*splits, splits, True), ([], [1, 1], [0, 0]))

    def test_gpu_functions_dispatch_and_all_other_code_unchanged(self):
        old, new = [ast.parse(path.read_text()) for path in [PARENT, CANDIDATE]]
        def strip(tree):
            tree = copy.deepcopy(tree)
            tree.body = [n for n in tree.body if not
                (isinstance(n, ast.FunctionDef) and n.name == '_tickets') and not
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_TICKETS' for t in n.targets))]
            return ast.dump(tree)
        self.assertEqual(strip(old), strip(new))
        self.assertNotIn('_TICKETS', CANDIDATE.read_text())
        compile(CANDIDATE.read_bytes(), str(CANDIDATE), 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
