"""Verify E3 is a single loop attribute change with untouched alternate paths."""
import ast
from pathlib import Path
import unittest
from prepare_pipeline import transform

HERE=Path(__file__).resolve().parent


class Pipeline(unittest.TestCase):
    def check_parent(self,name):
        original=ast.parse((HERE/name).read_text());changed=ast.parse(transform((HERE/name).read_text()))
        fused=lambda t:next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_fused')
        a,b=fused(original),fused(changed)
        i=next(i for i,x in enumerate(a.body) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='start')
        branch=b.body[i];self.assertIsInstance(branch,ast.If)
        self.assertEqual(branch.test.id,'PIPELINED_HISTORY')
        self.assertEqual(ast.dump(branch.orelse[0]),ast.dump(a.body[i]))
        piped=branch.body[0];piped.iter=ast.parse('range(s_lo, s_hi, BLOCK_N)',mode='eval').body
        self.assertEqual(ast.dump(piped),ast.dump(a.body[i]))
        b.body[i]=branch.orelse[0]
        b.args.args=[x for x in b.args.args if x.arg!='PIPELINED_HISTORY']
        self.assertEqual(ast.dump(a),ast.dump(b))
        dispatch=[k.value for node in ast.walk(changed) if isinstance(node,ast.Call)
                  for k in node.keywords if k.arg=='PIPELINED_HISTORY']
        self.assertEqual(len(dispatch),1)
        self.assertEqual(ast.dump(dispatch[0]),ast.dump(ast.parse('window == 0 and batch in (16, 32)',mode='eval').body))
        for node in ast.walk(changed):
            if isinstance(node,ast.Call):node.keywords=[x for x in node.keywords if x.arg!='PIPELINED_HISTORY']
        self.assertEqual(ast.dump(original),ast.dump(changed))

    def test_original_parent(self):self.check_parent('submission.py')
    def test_fixed_chunks_parent(self):self.check_parent('fixed_chunks.py')


if __name__=='__main__':unittest.main(verbosity=2)
