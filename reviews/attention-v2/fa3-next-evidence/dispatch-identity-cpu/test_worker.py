import ast
import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_identity_worker',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def elf(bits=64,endian='<'):
    note=struct.pack(endian+'III',4,4,3)+b'GNU\0'+b'\x01\x23\x45\x67'
    ident=b'\x7fELF'+bytes([2 if bits==64 else 1,1 if endian=='<' else 2,1])+b'\0'*9
    if bits==64:
        header=struct.pack(endian+'HHIQQQIHHHHHH',3,62,1,0,64,0,0,64,56,1,0,0,0)
        ph=struct.pack(endian+'IIQQQQQQ',4,0,120,0,0,len(note),len(note),4)
    else:
        header=struct.pack(endian+'HHIIIIIHHHHHH',3,3,1,0,52,0,0,52,32,1,0,0,0)
        ph=struct.pack(endian+'IIIIIIII',4,84,0,0,len(note),len(note),0,4)
    return ident+header+ph+note


def parse(raw):
    return w.elf_build_ids(lambda offset,count:raw[offset:offset+count],len(raw))


class WorkerTests(unittest.TestCase):
    def test_elf_all_classes_endianness(self):
        for bits in (32,64):
            for endian in ('<','>'):
                with self.subTest(bits=bits,endian=endian):
                    got=parse(elf(bits,endian))
                    self.assertEqual(got['build_ids'],['01234567'])
                    self.assertEqual(got['elf_class'],bits)

    def test_malformed_elf_is_not_source_identity(self):
        self.assertEqual(parse(b'not an ELF')['status'],'not_elf')
        self.assertEqual(parse(b'\x7fELF\x02\x01'+b'\0'*10)['status'],'truncated_elf_header')
        raw=bytearray(elf());struct.pack_into('<H',raw,56,65535)
        self.assertEqual(parse(raw)['status'],'program_header_bound')
        raw=bytearray(elf());struct.pack_into('<Q',raw,72,2**60)
        self.assertEqual(parse(raw)['status'],'note_segment_bound')

    def test_notes_do_not_overread(self):
        self.assertEqual(w.note_build_ids(struct.pack('<III',10000,4,3),'<'),[])
        raw=struct.pack('<III',4,10000,3)+b'GNU\0'
        self.assertEqual(w.note_build_ids(raw,'<'),[])

    def test_wrapper_is_syntax_only_return_flag_not_native_arg(self):
        raw=b'''def flash_attn_with_kvcache(q,num_splits=0,pack_gqa=None,sm_margin=0,ver=3,return_softmax_lse=False):
    out,lse,*rest=torch.ops.sgl_kernel.fwd.default(q,num_splits,pack_gqa)
    return (out,lse,*rest) if return_softmax_lse else out
'''
        x=w.wrapper_ast(raw)
        self.assertEqual(x['defaults']['pack_gqa'],'None')
        self.assertFalse(x['return_flag_in_native_arguments'])
        self.assertEqual(x['native_positions_1based']['num_splits'],2)
        self.assertEqual(x['native_positions_1based']['pack_gqa'],3)
        self.assertEqual(x['native_positional_count'],3)

    def test_wrapper_forwarded_flag_not_falsely_return_only(self):
        x=w.wrapper_ast(b'def flash_attn_with_kvcache(return_softmax_lse=False):\n return torch.ops.sgl_kernel.fwd.default(return_softmax_lse)\n')
        self.assertTrue(x['return_flag_in_native_arguments'])

    def test_vendor_path_no_private_origin(self):
        for path in ['/Users/user/site-packages/mod.py','/root/site-packages/mod.py','/opt/sglang/local-notes/auth.py','/etc/credentials']:
            self.assertFalse(w.vendor_path(path))
        self.assertTrue(w.vendor_path('/opt/sglang/lib/python3.12/site-packages/sgl_kernel/flash_attn.py'))

    def test_binary_hash_and_note_are_file_identity_only(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,'vendor_path',return_value=True):
            p=Path(tmp)/'flash_ops.so';p.write_bytes(elf())
            a=w.Audit();a.read_binary(p,Path(tmp))
            row=a.report['binary_files'][0]
            self.assertEqual(row['sha256'],hashlib.sha256(elf()).hexdigest())
            self.assertEqual(row['elf']['build_ids'],['01234567'])
            self.assertIn('not loaded or executed',row['qualification'])

    def test_source_and_binary_limits_stop_without_partial_hash_claim(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,'vendor_path',return_value=True):
            p=Path(tmp)/'flash_ops.so';p.write_bytes(elf())
            a=w.Audit()
            with patch.object(w,'MAX_BINARY',1):a.read_binary(p,Path(tmp))
            self.assertFalse(a.report['binary_files']);self.assertIn('binary_hash_bound',a.report['limits_reached'])
            q=Path(tmp)/'flash.h';q.write_text('PackGQA\n'*100)
            with patch.object(w,'MAX_SOURCE',1):a.read_source(q,Path(tmp))
            self.assertFalse(a.report['files'])

    def test_symlink_escape_cannot_read_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,'vendor_path',return_value=True):
            root=Path(tmp)/'vendor';root.mkdir();secret=Path(tmp)/'outside';secret.write_text('not exported')
            link=root/'flash.h';link.symlink_to(secret)
            a=w.Audit();a.read_source(link,root)
            self.assertFalse(a.report['files']);self.assertTrue(a.report['skips'])

    def test_directory_walk_is_bounded_before_materializing_entries(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,'vendor_path',return_value=True):
            root=Path(tmp)
            for i in range(20):(root/f'{i}.h').write_text('x')
            a=w.Audit()
            with patch.object(w,'MAX_ENTRIES',3):a.walk(root,root)
            self.assertLessEqual(a.report['walk_entries'],4)
            self.assertIn('walk_entries',a.report['limits_reached'])

    def test_deadline_errors_are_never_swallowed(self):
        for method in ('read_source','read_binary','walk'):
            a=w.Audit(seconds=-1)
            with self.subTest(method=method),self.assertRaises(TimeoutError):
                getattr(a,method)(Path('/opt/sglang/file'),Path('/opt/sglang'))
        with patch.object(w,'bind_sources',return_value={}),patch.object(w,'MODULES',()),patch.object(w,'DISTRIBUTIONS',()),patch.object(w,'FIXED_DIRS',()):
            with self.assertRaises(TimeoutError):w.run(None,None,seconds=-1)

    def test_distribution_version_uses_bounded_file_not_version_property(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,'vendor_path',return_value=True):
            root=Path(tmp);(root/'METADATA').write_text('Name: fixture\nVersion: 3.8.0\n')
            class Distribution:
                _path=root
                @property
                def version(self):raise AssertionError('unbounded metadata accessor forbidden')
            with patch.object(w,'bind_sources',return_value={}),patch.object(w,'MODULES',()),patch.object(w,'DISTRIBUTIONS',('fixture',)),patch.object(w,'FIXED_DIRS',()),patch.object(w.importlib.metadata,'distribution',return_value=Distribution()):
                report=w.run(None,None)
            self.assertEqual(report['distributions']['fixture']['version'],'3.8.0')
            self.assertEqual(report['distributions']['fixture']['status'],'bounded_metadata_file')
            self.assertEqual(report['source_bytes_read'],(root/'METADATA').stat().st_size)

    def test_no_vendor_imports_subprocess_or_dynamic_execution(self):
        tree=ast.parse((HERE/'worker.py').read_text())
        forbidden={'torch','sglang','sgl_kernel','triton','modal','subprocess','ctypes','socket','requests'}
        imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        names=[a.name.split('.')[0] for n in imports if isinstance(n,ast.Import) for a in n.names]
        names += [n.module.split('.')[0] for n in imports if isinstance(n,ast.ImportFrom)]
        self.assertFalse(forbidden.intersection(names))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','__import__'} for n in calls))


if __name__=='__main__':unittest.main()
