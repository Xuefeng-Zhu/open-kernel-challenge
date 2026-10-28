"""CPU metadata/scalar/lifecycle checks; never import candidate or call services."""
import ast
import hashlib
import inspect
from pathlib import Path
import unittest
import worker as w

class MetadataTests(unittest.TestCase):
    def test_exact_cpu_page1_mapping(self):
        out=w.metadata_cpu([[3,0,2,1],[6,7,5,4]],[0,2],1,1)
        self.assertEqual(out,dict(positions=[0,2],cache_loc=[3,5],cache_seqlens=[1,3],cu_seqlens_q=[0,1,2]))
    def test_page64_l8_cross_page_tail(self):
        out=w.metadata_cpu([[1,0],[2,3]],[61,120],8,64)
        self.assertEqual(out['cache_loc'][:3],[125,126,127]);self.assertEqual(out['cache_loc'][3:8],[0,1,2,3,4]);self.assertEqual(out['cache_loc'][-1],255)
    def test_bijection_and_bounds_reject(self):
        for args in [([[0,0]],[0],1,1),([[1,2]],[0],1,1),([[0,1]],[-1],1,1),([[0,1]],[2],1,1),([[0],[1,2]],[0,0],1,1)]:
            with self.assertRaises(ValueError):w.metadata_cpu(*args)
    def test_boundary_lengths_stay_in_capacity(self):
        for l in (1,8):
            for variant in range(5):
                a=w.length_pattern(16,86008,l,variant)
                self.assertEqual(len(a),16);self.assertTrue(all(0<=n<=86008-l for n in a));self.assertIn(0,a);self.assertIn(2048,a)
    def test_default_family_and_growth_scope(self):
        self.assertEqual(len(w.FAMILIES),6)
        self.assertEqual({(x['batch'],x['q_len'],x['layer']) for x in w.FAMILIES},{(1,1,'full'),(16,1,'full'),(1,8,'full'),(16,8,'full'),(1,1,'sliding'),(16,8,'sliding')})
        self.assertGreater(w.GROWTH['batch']*8*8*16,32768)
        self.assertEqual(w.GROWTH['batch']*8*8*16,33792)

class RealTorchTests(unittest.TestCase):
    def base(self,length=8,window=0,page=1):
        import torch
        b,width,hkv,d=2,80,2,64
        table=torch.arange(b*width,dtype=torch.int32).view(b,width)
        lens=torch.tensor([65,68],dtype=torch.int32);pos=lens.long()[:,None]+torch.arange(length)
        loc=(torch.gather(table.long(),1,pos//page)*page+pos%page).reshape(-1)
        cache=torch.linspace(-2,2,b*width*page*hkv*d).view(b*width,page,hkv,d).to(torch.float8_e4m3fn)
        return dict(qkv=torch.zeros(b,length,2304,dtype=torch.bfloat16),positions=pos.reshape(-1),cache_loc=loc,cache_seqlens=lens+length,cu_seqlens_q=torch.arange(0,(b+1)*length,length,dtype=torch.int32),scheduler_metadata=None,k_cache=cache.clone(),v_cache=cache.clone(),k_scale=torch.tensor(1.),v_scale=torch.tensor(1.),page_table=table,seq_lens=lens,config=dict(batch=b,q_len=length,num_q_heads=16,num_kv_heads=2,head_dim=64,page_size=page,window=window,rope_theta=500000. if window else 0.,kv_scale=1.))
    def test_real_torch_import_is_not_stub(self):
        import torch
        self.assertEqual(torch.tensor(2.).square().item(),4.)
        self.assertTrue(hasattr(torch,'__version__'))
    def test_variant_matches_cpu_metadata_and_scales(self):
        import torch
        base=self.base();fresh=w.variant(torch,base,101)
        modeled=w.metadata_cpu(fresh['page_table'].tolist(),fresh['seq_lens'].tolist(),8,1)
        for key,values in modeled.items():self.assertEqual(fresh[key].tolist(),values)
        self.assertEqual(fresh['k_scale'].item(),1/64);self.assertEqual(fresh['config']['kv_scale'],1/64)
        self.assertTrue(torch.isfinite(fresh['k_cache'].float()).all())
        self.assertTrue(w.h.bytes_equal(torch,base['k_scale'],torch.tensor(1.)))
        self.assertEqual(base['seq_lens'].tolist(),[65,68])
        self.assertFalse(torch.equal(fresh['qkv'],base['qkv']))
    def test_coherent_value_eight_stays_finite_at_nonunit_scale(self):
        import torch
        base=self.base();base['v_cache'].fill_(8.)
        fresh=w.variant(torch,base,101)
        self.assertEqual(w.V_HISTORY_AMPLITUDE,.25)
        self.assertTrue(torch.isfinite(fresh['v_cache'].float()).all())
        self.assertTrue(torch.equal(fresh['v_cache'].float(),torch.full_like(fresh['v_cache'].float(),128.)))
        self.assertTrue(torch.equal(fresh['k_cache'].float()*fresh['k_scale'],base['k_cache'].float()*base['k_scale']))
    def test_actual_torch_page64_builder_matches_cpu_model(self):
        import torch
        base=self.base(page=64);fresh=w.variant(torch,base,101)
        modeled=w.metadata_cpu(fresh['page_table'].tolist(),fresh['seq_lens'].tolist(),8,64)
        for key,values in modeled.items():self.assertEqual(fresh[key].tolist(),values)
    def test_other_retained_output_clobber_is_detected(self):
        import torch
        from types import SimpleNamespace
        ref=SimpleNamespace(collect=lambda work,out:out)
        slots={'A':dict(work=None,out=torch.tensor([1.]),last_raw=torch.tensor([1.])), 'B':dict(work=None,out=torch.tensor([2.]),last_raw=torch.tensor([2.]))}
        self.assertEqual(w.unchanged_other_outputs(torch,ref,slots,'A'),1)
        slots['B']['out'].fill_(3.)
        with self.assertRaises(ValueError):w.unchanged_other_outputs(torch,ref,slots,'A')
    def test_full_decode_live_metadata_kept_consistent(self):
        import torch
        base=self.base(length=1);fresh=w.variant(torch,base,101)
        for key in ['seq_lens','positions','cache_loc','page_table','cache_seqlens','cu_seqlens_q']:
            self.assertTrue(w.h.bytes_equal(torch,fresh[key],base[key]))
    def test_sliding_decode_can_change_lengths_and_page_addresses(self):
        import torch
        base=self.base(length=1,window=2048);fresh=w.variant(torch,base,101)
        self.assertFalse(w.h.bytes_equal(torch,fresh['seq_lens'],base['seq_lens']))
        modeled=w.metadata_cpu(fresh['page_table'].tolist(),fresh['seq_lens'].tolist(),1,1)
        self.assertEqual(fresh['cache_loc'].tolist(),modeled['cache_loc'])
    def test_inplace_restore_keeps_zero_dim_scale_addresses(self):
        import torch
        base=self.base();work=w.h.clone(torch,base);state=w.h.pointers(torch,work)
        fresh=w.variant(torch,base,101);w.restore(torch,work,fresh)
        self.assertEqual(state,w.h.pointers(torch,work))
        self.assertEqual(work['k_scale'].shape,torch.Size([]));self.assertEqual(work['k_scale'].item(),1/64)
        for key in w.USED:self.assertTrue(w.h.bytes_equal(torch,work[key],fresh[key]))
    def test_launch_configuration_change_rejected(self):
        import torch
        base=self.base();fresh=w.variant(torch,base,101);fresh['config']['window']=2048
        with self.assertRaises(ValueError):w.restore(torch,base,fresh)

class SourceTests(unittest.TestCase):
    def test_helper_hash_and_original_bindings(self):
        root=Path(__file__).resolve().parent
        self.assertEqual(w.h.sha(root/'helpers.py'),w.SOURCES['helpers.py'])
        self.assertEqual(w.SOURCES['v298.py'],'0cc4a0668acf7553c0d2c6d5c3a2e7d5c85374a70ad774463e9d2fd9e5d0f883')
    def test_native_imports_deferred_and_no_service_subprocess(self):
        tree=ast.parse(Path(w.__file__).read_text());top=[ast.unparse(x) for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in x or 'modal' in x or 'subprocess' in x for x in top))
        self.assertNotIn('ncu', '\n'.join(top))
    def test_growth_does_not_pin_or_clear_old_scratch(self):
        source=inspect.getsource(w.run)
        self.assertIn('weakref.ref(x)',source)
        self.assertNotIn('old_ws =',source);self.assertNotIn('._WS.clear',source)
        self.assertNotIn('._WS[device] =',source)
        self.assertLess(source.index("phase('main-serial-complete')"),source.index("phase('optional-growth-capture')"))
    def test_parity_counters_recorded_not_forced_zero(self):
        source=inspect.getsource(w.scratch_snapshot)
        self.assertIn('counter_epochs',source)
        self.assertNotIn('count_nonzero(ws[2]',source)

if __name__=='__main__':unittest.main()
