"""AST invariants for a single-factor refactor; not native GPU execution."""
import ast
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
BASE=ast.parse((HERE/'submission.py').read_text())
NEW=ast.parse((HERE/'separate_merge.py').read_text())


def fn(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
def same(a,b):return ast.dump(ast.Module(body=a,type_ignores=[]))==ast.dump(ast.Module(body=b,type_ignores=[]))


class SingleFactor(unittest.TestCase):
    def test_norm_and_partition_unchanged(self):
        for name in ['_norm_rope','_num_splits']:
            self.assertEqual(ast.dump(fn(BASE,name)),ast.dump(fn(NEW,name)))

    def test_attention_and_new_token_math_unchanged(self):
        base,new=fn(BASE,'_fused'),fn(NEW,'_partial')
        self.assertTrue(same(base.body[:-1],new.body[:-1]))
        self.assertTrue(same(base.body[-1].body,new.body[-1].body))
        self.assertEqual(len(new.body[-1].orelse),4) # pbase plus three stores
        self.assertTrue(same(base.body[-1].orelse[:4],new.body[-1].orelse))

    def test_merge_recurrence_and_output_math_unchanged(self):
        ticket_election=fn(BASE,'_fused').body[-1].orelse[-1]
        old=ticket_election.body[:-1] # omit only ticket reset
        new=fn(NEW,'_merge').body[-len(old):]
        self.assertTrue(same(old,new))

    def test_no_counter_or_global_tensor_state(self):
        names={n.id for n in ast.walk(NEW) if isinstance(n,ast.Name)}
        self.assertNotIn('_TICKETS',names);self.assertNotIn('ticket_ptr',names)
        attrs={n.attr for n in ast.walk(NEW) if isinstance(n,ast.Attribute)}
        self.assertNotIn('atomic_add',attrs);self.assertNotIn('atomic_xchg',attrs)
        source=(HERE/'separate_merge.py').read_text()
        self.assertNotIn('torch.zeros',source)


if __name__=='__main__':unittest.main(verbosity=2)
