"""Standalone source integration and CPU oracle coverage, never GPU evidence."""
import ast
import hashlib
from unittest.mock import patch
import unittest
from tests import test_projection_pooler_codec as old
from harness import ROOT, load_module

SOURCE=ROOT/'candidates/projection_pooler_codec_explicit.py'
RAW=ROOT/'candidates/projection_pooler_codec_explicit_raw.py'

class Standalone(old.PoolerCodec):
    @classmethod
    def setUpClass(cls):
        with patch.object(old,'SOURCE',SOURCE),patch.object(old,'CONTROL',RAW):
            super().setUpClass()

    def test_parent_immutable_other_paths_exact_ast(self):
        def funcs(p):return {n.name:n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
        parent=funcs(old.SOURCE);new=funcs(SOURCE)
        self.assertEqual(parent.keys(),new.keys())
        for name in parent.keys()-{'_pooler_codec_partial'}:
            self.assertEqual(ast.dump(parent[name]),ast.dump(new[name]),name)
        frozen=funcs(ROOT/'projection_p065_explicit_operand.py')['_pooler_codec_partial']
        self.assertEqual(ast.dump(frozen),ast.dump(new['_pooler_codec_partial']))
        self.assertEqual(SOURCE.read_text().replace('POOLER_COMPRESSED = True','POOLER_COMPRESSED = False',1),RAW.read_text())
        self.assertEqual(hashlib.sha256(old.SOURCE.read_bytes()).hexdigest(),'9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837')
        self.assertEqual(hashlib.sha256((ROOT/'candidates/projection_pooler_column_batch2_hybrid.py').read_bytes()).hexdigest(),'001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e')

    def test_no_host_work_in_timed_gpu_branch_and_invocation_local_scratch(self):
        with patch.object(old,'SOURCE',SOURCE):super().test_no_host_work_in_timed_gpu_branch_and_invocation_local_scratch()

if __name__=='__main__':unittest.main()
