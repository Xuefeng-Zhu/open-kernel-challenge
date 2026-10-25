"""CPU address-trace and AST tests only. Never imports or executes a candidate."""
import ast
import copy
import hashlib
from pathlib import Path
import random
import unittest

ROOT=Path(__file__).resolve().parent
PARENT_HASH='5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0'
CANDIDATE_HASH='3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'


def dump(n): return ast.dump(n,include_attributes=False)
def statements(s): return ast.parse(s).body
def function(t,name): return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
def flagged(n): return isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='PAGE_LOOKAHEAD'

BOOT='''first_t = s_lo + offs_n
first_ok = first_t < s_hi
first_pg = tl.load(pt_ptr + b * stride_pt + first_t // page, mask=first_ok, other=0).to(tl.int64)
first_pg = tl.maximum(first_pg, 0)
row_current = first_pg * page + first_t % page
'''
NEXT='''next_t = t + BLOCK_N
next_ok = next_t < s_hi
next_pg = tl.load(pt_ptr + b * stride_pt + next_t // page, mask=next_ok, other=0)
'''


class SourceIsolation(unittest.TestCase):
    def test_exact_measured_donor_and_preserved_fusion_selected(self):
        v2 = ROOT.parent
        donor = (v2/'mixed_fa3_page_lookahead.py').read_bytes()
        self.assertEqual(hashlib.sha256(donor).hexdigest(),
            'fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c')
        d = ast.parse(donor)
        for name in ['_fused', '_e2_kernel_fn']:
            self.assertEqual(dump(function(self.candidate, name)), dump(function(d, name)))
            self.assertEqual(ast.get_source_segment((ROOT/'candidate.py').read_text(), function(self.candidate, name)),
                             ast.get_source_segment(donor.decode(), function(d, name)))
        self.assertEqual(hashlib.sha256((v2/'fused-preparation-probe/candidate.py').read_bytes()).hexdigest(), PARENT_HASH)
        self.assertEqual(hashlib.sha256((v2/'mixed_fa3_decode_parallel_merge.py').read_bytes()).hexdigest(),
            'c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0')

    @classmethod
    def setUpClass(cls):
        p=(ROOT/'baseline.py').read_bytes();c=(ROOT/'candidate.py').read_bytes()
        assert hashlib.sha256(p).hexdigest()==PARENT_HASH
        assert hashlib.sha256(c).hexdigest()==CANDIDATE_HASH
        cls.parent=ast.parse(p);cls.candidate=ast.parse(c)
        compile(c,'syntax only','exec')

    def test_complete_parent_reconstruction(self):
        t=copy.deepcopy(self.candidate);t.body[0]=copy.deepcopy(self.parent.body[0])
        f=function(t,'_fused');arg=f.args.args.pop()
        self.assertEqual(arg.arg,'PAGE_LOOKAHEAD');self.assertEqual(dump(arg.annotation),dump(ast.parse('tl.constexpr',mode='eval').body))
        bootstrap=[n for n in f.body if flagged(n)]
        self.assertEqual(len(bootstrap),1);self.assertFalse(bootstrap[0].orelse)
        self.assertEqual([dump(n) for n in bootstrap[0].body],[dump(n) for n in statements(BOOT)])
        f.body.remove(bootstrap[0])
        loop=next(n for n in f.body if isinstance(n,ast.For))
        flags=[n for n in loop.body if flagged(n)];self.assertEqual(len(flags),3)
        use,prefetch,carry=flags
        self.assertEqual([dump(n) for n in use.body],[dump(n) for n in statements('row = row_current')])
        self.assertEqual([dump(n) for n in prefetch.body],[dump(n) for n in statements(NEXT)])
        self.assertFalse(prefetch.orelse)
        self.assertEqual([dump(n) for n in carry.body],[dump(n) for n in statements('next_pg64 = tl.maximum(next_pg.to(tl.int64), 0)\nrow_next = next_pg64 * page + next_t % page\nrow_current = row_next')])
        self.assertFalse(carry.orelse)
        i=loop.body.index(use);loop.body[i:i+1]=use.orelse
        loop.body.remove(prefetch);loop.body.remove(carry)
        host=function(t,'_e2_kernel_fn')
        guard=next(n for n in host.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='page_lookahead')
        host.body.remove(guard)
        launch=next(n for n in ast.walk(host) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id=='_fused')
        kw=next(k for k in launch.keywords if k.arg=='PAGE_LOOKAHEAD');self.assertEqual(dump(kw.value),dump(ast.Name(id='page_lookahead',ctx=ast.Load())))
        launch.keywords.remove(kw)
        self.assertEqual(dump(t),dump(self.parent))

    def test_single_path_dispatch_and_unchanged_entrypoint(self):
        guard=next(n.value for n in function(self.candidate,'_e2_kernel_fn').body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='page_lookahead')
        self.assertEqual(dump(guard),dump(ast.parse('batch == 32 and length == 1 and window == 0 and not rope',mode='eval').body))
        code=compile(ast.Expression(guard),'config guard','eval');on=[]
        for b in (1,16,32):
            for l in (1,8):
                for w in (0,2048):
                    if eval(code,{'__builtins__':{}},dict(batch=b,length=l,window=w,rope=bool(w))):on.append((b,l,w))
        self.assertEqual(on,[(32,1,0)])
        self.assertEqual(dump(function(self.parent,'kernel_fn')),dump(function(self.candidate,'kernel_fn')))

    def test_prefetch_order_payload_and_scalar_denom(self):
        loop=next(n for n in function(self.candidate,'_fused').body if isinstance(n,ast.For))
        names=lambda n:n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
        k=next(i for i,n in enumerate(loop.body) if names(n)=='k')
        self.assertTrue(flagged(loop.body[k+1]));self.assertEqual(names(loop.body[k+2]),'s')
        carry=loop.body[-1];self.assertTrue(flagged(carry))
        self.assertEqual(names(loop.body[-2]),'m_i')
        self.assertEqual(names(loop.body[-3]),'acc')
        # No use of the prefetched value before the end-of-tile carry.
        self.assertFalse(any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='next_pg' for stmt in loop.body[k+2:-1] for n in ast.walk(stmt)))
        self.assertEqual([n.func.id for n in ast.walk(loop.body[k+1]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)],[])
        loads=[n for n in ast.walk(loop.body[k+1]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
        self.assertEqual(len(loads),1)
        pointer_names={n.id for n in ast.walk(loads[0].args[0]) if isinstance(n,ast.Name)}
        self.assertIn('pt_ptr',pointer_names);self.assertFalse(pointer_names & {'kc_ptr','vc_ptr','qkv_ptr','loc_ptr'})
        denom=next(n for n in loop.body if names(n)=='l_i')
        self.assertEqual(dump(denom),dump(statements('l_i = l_i * alpha + tl.sum(p, 1)')[0]))
        self.assertNotIn('l_columns',{n.id for n in ast.walk(self.candidate) if isinstance(n,ast.Name)})


def address_trace(table,b,stride,page,seq,split,hkv=2,head=0,d=64,lookahead=False):
    """Integer-only model. `table` contains valid entries only: masked read raises if attempted."""
    lo=split*8192;hi=min(lo+8192,seq);n=64;reads=[];keys=[];values=[]
    def rows(start):
        result=[]
        for t in range(start,start+n):
            valid=t<hi
            if valid:
                address=b*stride+t//page
                pg=table[address]  # No silent out-of-bounds fallback in the test model.
                reads.append((t,address))
            else:pg=0
            result.append((t,valid,max(pg,0)*page+t%page))
        return result
    def payload(items,output):
        for t,valid,row in items:
            if valid:output.append((t,row*hkv*d+head*d,row*hkv*d+head*d+d-1))
    if lookahead:current=rows(lo)
    for start in range(lo,hi,n):
        current=current if lookahead else rows(start)
        payload(current,keys)
        if lookahead:following=rows(start+n)
        payload(current,values)
        if lookahead:current=following
    return dict(page_reads=reads,k=keys,v=values)


def make_table(seq,page,b=2,stride=None,seed=1):
    pages=(seq+page-1)//page
    stride=stride or max(1,pages+7)
    ids=list(range(pages));random.Random(seed).shuffle(ids)
    return {b*stride+i:pg for i,pg in enumerate(ids)},stride


class AddressModel(unittest.TestCase):
    def test_pages_scatter_tails_empty_chunks_and_coverage(self):
        for page in (1,64):
            for seq in (0,1,63,64,65,8191,8192,8193,80000,81920):
                table,stride=make_table(seq,page)
                covered=[]
                for split in range((seq+8191)//8192+2):
                    a=address_trace(table,2,stride,page,seq,split)
                    b=address_trace(table,2,stride,page,seq,split,lookahead=True)
                    self.assertEqual(a,b)
                    self.assertEqual(a['k'],a['v'])
                    self.assertTrue(all(t<seq for t,_,_ in b['k']))
                    covered.extend(t for t,_,_ in b['k'])
                self.assertEqual(covered,list(range(seq)))

    def test_page_zero_masked_negative_padding_and_64bit_offsets(self):
        # Existing valid-page clamp is preserved, including zero. Negative valid
        # pages are a parent-parity robustness test, not an official valid input.
        table={17:0,18:-1,19:2**30}
        for look in (False,True):
            a=address_trace(table,1,17,64,129,0,hkv=16,head=15,lookahead=look)
            self.assertEqual(a['k'][0][1],15*64)
            self.assertEqual(a['k'][64][1],15*64)
            self.assertGreater(a['k'][-1][1],2**32)
            self.assertEqual(len(a['page_reads']),129)
        self.assertEqual(address_trace(table,1,17,64,129,0,hkv=16,head=15),address_trace(table,1,17,64,129,0,hkv=16,head=15,lookahead=True))

    def test_fresh_same_container_length_and_mapping_changes(self):
        # CPU object/address reuse only; this is not CUDA graph evidence.
        table={};stride=9000
        a,_=make_table(8193,1,stride=stride,seed=15)
        b,_=make_table(65,1,stride=stride,seed=29)
        outcomes=[]
        for contents,seq in ((a,8193),(b,65),(a,8193)):
            table.clear();table.update(contents)
            traces=[address_trace(table,2,stride,1,seq,s,lookahead=True) for s in (0,1,2)]
            expected=[address_trace(table,2,stride,1,seq,s) for s in (0,1,2)]
            self.assertEqual(traces,expected);outcomes.append(traces)
        self.assertEqual(outcomes[0],outcomes[2]);self.assertNotEqual(outcomes[0],outcomes[1])

    def test_random_mappings_strides_heads_and_lengths(self):
        rng=random.Random(215)
        for i in range(40):
            seq=rng.randrange(0,20000);page=rng.choice((1,64));batch=rng.randrange(1,5)
            stride=(seq+page-1)//page+rng.randrange(1,17)
            table,_=make_table(seq,page,batch,stride,i)
            hkv=rng.choice((2,16));head=rng.randrange(hkv)
            for split in range((seq+8191)//8192+1):
                self.assertEqual(address_trace(table,batch,stride,page,seq,split,hkv,head),address_trace(table,batch,stride,page,seq,split,hkv,head,lookahead=True))


if __name__=='__main__': unittest.main(verbosity=2)
