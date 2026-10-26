import ast
import tempfile
import unittest
from pathlib import Path
import projection_p063_offline_compile as p


class OfflineSourceContract(unittest.TestCase):
    def test_exact_bodies_match_and_bad_hash_rejected(self):
        bodies=[p.extract(p.ROOT/'candidates'/n,h) for n,h in p.SOURCES.items()]
        self.assertEqual(bodies[0],bodies[1])
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            p.extract(p.ROOT/'candidates'/next(iter(p.SOURCES)), 'bad')

    def test_payload_is_only_owned_kernel_and_triton_imports(self):
        with tempfile.TemporaryDirectory() as d:
            m=p.prepare(Path(d));tree=ast.parse((Path(d)/'kernel_source.py').read_text())
            self.assertEqual([type(x).__name__ for x in tree.body],['Import','Import','FunctionDef'])
            self.assertEqual(tree.body[-1].name,p.KERNEL)
            self.assertEqual(m['variants'],[False,True])
            self.assertEqual(m['module_sha256'],p.sha((Path(d)/'kernel_source.py').read_bytes()))

    def test_launch_constants_match_frozen_source_call(self):
        tree=ast.parse((p.ROOT/'candidates'/next(iter(p.SOURCES))).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)
               and isinstance(n.func.value,ast.Name) and n.func.value.id==p.KERNEL]
        self.assertEqual(len(calls),1)
        call=calls[0]
        self.assertEqual([ast.literal_eval(x) for x in call.args[13:17]],[8,16,64,64])
        self.assertEqual({x.arg:ast.literal_eval(x.value) for x in call.keywords},p.OPTIONS)
        self.assertEqual(list(p.POINTERS),[a.arg for a in ast.parse(p.extract(p.ROOT/'candidates'/next(iter(p.SOURCES)),next(iter(p.SOURCES.values())))[0]).body[0].args.args[:8]])

    def test_cpu_models_do_not_import_runtime_at_module_import(self):
        tree=ast.parse(Path(p.__file__).read_text())
        imports=[a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('triton',imports)
        self.assertNotIn('torch',imports)
        self.assertEqual(p.CONSTANTS['M'],16)
        self.assertEqual(p.CONSTANTS['K'],8192)


if __name__=='__main__':unittest.main()
