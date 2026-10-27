"""CPU/AST coverage for fixed historical chunks, not GPU validation."""
import ast
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent


class FixedChunks(unittest.TestCase):
    def test_all_valid_keys_once_and_empty_chunks(self):
        size=8192;bound=86008;splits=(bound+size-1)//size
        for length in [0,1,8191,8192,8193,16000,32768,74000,80000,81920,86008]:
            intervals=[(i*size,min((i+1)*size,length)) for i in range(splits)]
            valid=[(lo,hi) for lo,hi in intervals if hi>lo]
            self.assertEqual(sum(hi-lo for lo,hi in valid),length)
            if valid:
                self.assertEqual(valid[0][0],0);self.assertEqual(valid[-1][1],length)
                for first,second in zip(valid,valid[1:]):self.assertEqual(first[1],second[0])
            if length<81920:self.assertTrue(any(hi<=lo for lo,hi in intervals))
        self.assertEqual(splits,11)

    def test_only_partition_and_host_capacity_change(self):
        base=ast.parse((HERE/'submission.py').read_text());new=ast.parse((HERE/'fixed_chunks.py').read_text())
        find=lambda t,n:next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==n)
        for name in ['_norm_rope','_num_splits','_tickets']:
            self.assertEqual(ast.dump(find(base,name)),ast.dump(find(new,name)))
        a,b=find(base,'_fused'),find(new,'_fused')
        # Replacing only per/s_lo/s_hi by the new constexpr conditional preserves
        # the complete attention loop, new-token owner and neutral publication.
        old=next(i for i,x in enumerate(a.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='per' for t in x.targets))
        new_at=next(i for i,x in enumerate(b.body) if isinstance(x,ast.If) and isinstance(x.test,ast.Name) and x.test.id=='CHUNK_SIZE')
        self.assertEqual(ast.dump(ast.Module(body=a.body[:old],type_ignores=[])),ast.dump(ast.Module(body=b.body[:new_at],type_ignores=[])))
        self.assertEqual(ast.dump(ast.Module(body=a.body[old+3:],type_ignores=[])),ast.dump(ast.Module(body=b.body[new_at+1:],type_ignores=[])))
        source=(HERE/'fixed_chunks.py').read_text()
        self.assertIn('window == 0 and batch in (16, 32)',source)
        self.assertIn('num_warps=4, num_stages=3',source)


if __name__=='__main__':unittest.main(verbosity=2)
