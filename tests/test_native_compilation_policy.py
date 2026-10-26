"""Regression tests for our narrow preflight, not official submission approval."""
import ast
import unittest

from static_check import local_policy_errors


class NativeCompilationPolicyTests(unittest.TestCase):
    def errors(self, source):
        return local_policy_errors(ast.parse(source))

    def test_official_documented_initialization_exception(self):
        for body in ("extension = load_inline(name='probe', cpp_sources='source')",
                     "def prepare(weights):\n    return load_inline(name='probe', cpp_sources='source')"):
            self.assertEqual(self.errors("from torch.utils.cpp_extension import load_inline\n" + body), [])

    def test_reviewed_gluon_api_only(self):
        allowed = """import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language import DotOperandLayout, NVMMADistributedLayout, BlockedLayout, SliceLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
"""
        self.assertEqual(self.errors(allowed), [])
        for source in ("import triton.runtime.driver", "from triton.experimental.gluon._runtime import GluonASTSource",
                       "from triton.experimental.gluon.language import *", "import subprocess"):
            self.assertTrue(self.errors(source), source)

    def test_timed_compile_stays_rejected(self):
        for body in ("def kernel_fn(data):\n    return load_inline()",
                     "def kernel_fn(data):\n    def prepare():\n        return load_inline()\n    return prepare()"):
            self.assertTrue(self.errors(body))

    def test_no_blanket_process_or_download_exception(self):
        for body in ("import subprocess", "import os", "import urllib.request",
                     "from torch.utils.cpp_extension import load", "from torch.utils.cpp_extension import *",
                     "from torch.utils.cpp_extension import load_inline as build", "exec('arbitrary')"):
            self.assertTrue(self.errors(body), body)

    def test_ordinary_diagnostics_separate_from_timed_sync(self):
        self.assertEqual(self.errors("print('compile diagnostics')"), [])
        self.assertTrue(self.errors("def kernel_fn(data):\n    print(data)"))
        self.assertTrue(self.errors("def kernel_fn(data):\n    return data.cpu()"))


if __name__ == "__main__":
    unittest.main()
