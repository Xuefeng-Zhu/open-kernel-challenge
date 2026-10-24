"""CPU-only restoration and validation guards; does not import the Triton candidate."""
import importlib.util
from pathlib import Path
import sys
import re
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'official'))
import modal_worker as worker
spec=importlib.util.spec_from_file_location('attention_fixture_reference',HERE/'official/reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


class InputIsolation(unittest.TestCase):
    def setUp(self):
        self.fresh=reference.generate_input(batch=1,ctx=10,heads='tp8',layer='full',
            q_len=8,page=64,kv_scale=1/64,seed=301,device='cpu')
        self.work=worker.clone_inputs(self.fresh)

    def test_clones_restore_scalars_fp8_and_metadata(self):
        self.work['qkv'].zero_();self.work['k_scale'].fill_(1)
        worker.restore(self.work,self.fresh)
        self.assertTrue(worker.unchanged(self.work,self.fresh)[0])
        self.assertNotEqual(self.work['k_cache'].data_ptr(),self.fresh['k_cache'].data_ptr())

    def test_only_minimal_compiler_repair_after_original_normalization(self):
        original=(HERE/'paste.original.txt').read_text()
        normalized,count=re.subn(r'\[([A-Za-z_]\w*(?:\.\w+)+)\]\(http://\1\)',r'\1',original)
        self.assertEqual(count,5)
        expected=normalized.replace('LOG2E = 1.4426950408889634','LOG2E = tl.constexpr(1.4426950408889634)')
        self.assertEqual((HERE/'submission.py').read_text(),expected)

    def test_only_new_cache_rows_may_change(self):
        slots=self.fresh['cache_loc'];cache=self.work['k_cache'].view(-1,128).view(torch.uint8)
        cache[slots]=0
        self.assertTrue(worker.unchanged(self.work,self.fresh)[0])
        original=self.fresh['k_cache'].view(-1,128).view(torch.uint8)
        forbidden=next(i for i in range(len(cache)) if i not in slots.tolist())
        cache[forbidden,0]=original[forbidden,0]^1
        self.assertFalse(worker.unchanged(self.work,self.fresh)[0])

    def test_immutable_tensor_and_config_are_checked(self):
        self.work['qkv'][0,0,0]+=1
        self.assertFalse(worker.unchanged(self.work,self.fresh)[0])
        worker.restore(self.work,self.fresh);self.work['config']['window']=2
        self.assertFalse(worker.unchanged(self.work,self.fresh)[0])

    def test_graph_buffers_cannot_silently_change_shape(self):
        changed=worker.clone_inputs(self.fresh);changed['page_table']=changed['page_table'][:,:0]
        with self.assertRaises(ValueError):worker.restore(self.work,changed)


if __name__=='__main__':unittest.main(verbosity=2)
