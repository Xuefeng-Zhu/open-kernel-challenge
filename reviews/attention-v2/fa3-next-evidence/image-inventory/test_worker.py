"""CPU-only local guard tests using synthetic vendor trees; no real inventory."""
import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('bounded_image_inventory', HERE/'worker.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.vendor = self.base/'site-packages'
        self.vendor.mkdir()

    def write(self, relative, content):
        path = self.vendor/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def inventory(self):
        report = worker.initial_report()
        return report, worker.Inventory(report, 30)

    def fake_collect(self, modules, versions=None):
        versions = versions or {}

        def spec(name):
            if name not in modules:
                return None
            root = modules[name]
            return SimpleNamespace(origin=str(root/'__init__.py'),
                                   submodule_search_locations=[str(root)])

        def version(name):
            if name not in versions:
                raise worker.importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        with patch.object(worker.importlib.util, 'find_spec', side_effect=spec) as discovery, \
             patch.object(worker.importlib.metadata, 'version', side_effect=version), \
             patch.object(worker, 'FIXED_VENDOR_DIRS', ()):
            result = worker.collect_inventory()
        return result, discovery

    def test_only_stdlib_imports_and_top_level_discovery(self):
        tree = ast.parse((HERE/'worker.py').read_text())
        roots = {node.names[0].name.split('.')[0] for node in ast.walk(tree)
                 if isinstance(node, ast.Import)}
        roots |= {node.module.split('.')[0] for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom)}
        self.assertEqual(roots, {'argparse', 'ast', 'datetime', 'hashlib', 'importlib',
                                'json', 'math', 'os', 'pathlib', 're', 'signal', 'stat', 'time'})
        result, discovery = self.fake_collect({})
        self.assertEqual([call.args[0] for call in discovery.call_args_list], list(worker.MODULES))
        self.assertTrue(all('.' not in call.args[0] for call in discovery.call_args_list))
        self.assertEqual(result['package_imports_executed'], [])
        self.assertFalse(result['gpu_requested'])
        self.assertFalse(result['cuda_context_requested'])
        self.assertEqual(result['candidate_kernel_launches'], 0)

    def test_module_presence_separate_from_distribution_metadata(self):
        init = self.write('sgl_kernel/__init__.py', 'raise RuntimeError("do not execute")\n')
        wrapper = self.write('sgl_kernel/flash_attn.py',
            'def flash_attn_with_kvcache(q, k_cache, v_cache, num_splits=0): pass\n')
        result, _ = self.fake_collect({'sgl_kernel': init.parent}, {'triton': '3.8.0'})
        self.assertTrue(result['modules']['sgl_kernel']['found'])
        self.assertIsNone(result['distributions']['sgl-kernel'])
        self.assertEqual(result['distributions']['triton'], '3.8.0')
        self.assertFalse(result['modules']['flashinfer']['found'])
        self.assertEqual(result['modules']['sgl_kernel']['origin'], str(init.resolve()))
        row = next(row for row in result['files'] if row['path'] == str(wrapper.resolve()))
        self.assertEqual(row['sha256'], hashlib.sha256(wrapper.read_bytes()).hexdigest())
        self.assertEqual(row['api_signatures'][0]['name'], 'flash_attn_with_kvcache')
        self.assertEqual(result['status'], 'complete')

    def test_source_ast_signatures_page_guards_and_never_executes(self):
        raw = b'''raise RuntimeError("must not execute")
def warpgroup_mma(a, b, acc, *, is_async=False): pass
def xqa(page_size):
    if page_size not in [16,32,64,128]: raise ValueError()
class BatchDecodeWithPagedKVCacheWrapper:
    def plan(self, indptr, page_size=16):
        """CUDA graph capture does not prove plan is capture safe."""
        pass
'''
        result = worker.source_summary(raw, '.py')
        self.assertEqual([row['name'] for row in result['api_signatures']],
                         ['warpgroup_mma', 'xqa', 'BatchDecodeWithPagedKVCacheWrapper.plan'])
        self.assertEqual(result['page_size_guards'][0]['values'], [16,32,64,128])
        self.assertEqual(result['page_size_guards'][0]['operator'], 'NotIn')
        self.assertIn('CUDA graph', result['source_excerpt'])
        self.assertLessEqual(len(result['source_excerpt'].encode()), worker.MAX_SNIPPET)

    def test_triton_fixed_sources_find_hopper_and_async_copy(self):
        init = self.write('triton/__init__.py', '__version__ = "3.8.0"\n')
        self.write('triton/experimental/gluon/__init__.py', 'def jit(fn): pass\n')
        self.write('triton/experimental/gluon/language/nvidia/hopper/__init__.py',
                   'def warpgroup_mma(a,b,acc,is_async=False): pass\n'
                   'def warpgroup_mma_wait(num_outstanding, deps): pass\n')
        self.write('triton/experimental/gluon/language/nvidia/ampere/async_copy.py',
                   'def async_load(dst, src, mask=None): pass\ndef wait_group(n): pass\n')
        result, _ = self.fake_collect({'triton': init.parent}, {'triton':'3.8.0'})
        names = {sig['name'] for row in result['files'] for sig in row.get('api_signatures', [])}
        self.assertTrue({'warpgroup_mma', 'warpgroup_mma_wait', 'async_load', 'wait_group'} <= names)

    def test_outside_symlink_not_read_and_skip_list_bounded(self):
        outside = self.base/'outside.py'
        outside.write_text('harmless fixture outside allowroot')
        link = self.vendor/'link.py'
        link.symlink_to(outside)
        report, inventory = self.inventory()
        for _ in range(100):
            inventory.inspect(link, self.vendor)
        self.assertEqual(report['files_examined'], 0)
        self.assertEqual(report['hashed_file_bytes'], 0)
        self.assertEqual(len(report['skips']), 64)
        self.assertEqual(report['skips'][0]['reason'], 'symlink_outside_allowroot')

    def test_large_binary_stat_only_without_open_or_hash(self):
        binary = self.write('sgl_kernel/flash_ops.abi3.so', b'')
        with binary.open('r+b') as stream:
            stream.truncate(1_122_619_328)
        report, inventory = self.inventory()
        with patch.object(worker.os, 'open', side_effect=AssertionError('no binary open')):
            inventory.inspect(binary, self.vendor, stat_only=True)
        self.assertEqual(report['files'][0]['size'], 1_122_619_328)
        self.assertEqual(report['files'][0]['skip_reason'], 'stat_only_binary_no_hash')
        self.assertNotIn('sha256', report['files'][0])
        self.assertEqual(report['hashed_file_bytes'], 0)

    def test_file_and_total_byte_limits(self):
        paths = [self.write(f'p{i}.py', b'#'+b'x'*179) for i in range(3)]
        report, inventory = self.inventory()
        with patch.object(worker, 'MAX_FILE', 200), patch.object(worker, 'MAX_BYTES', 300):
            for path in paths:
                inventory.inspect(path, self.vendor)
        self.assertEqual(report['hashed_file_bytes'], 180)
        self.assertEqual(report['files_read'], 1)
        self.assertIn('max_bytes', report['limits_reached'])
        report, inventory = self.inventory()
        with patch.object(worker, 'MAX_FILE', 100), \
             patch.object(worker.os, 'open', side_effect=AssertionError('oversized file read')):
            inventory.inspect(paths[0], self.vendor)
        self.assertEqual(report['files'][0]['skip_reason'], 'file_exceeds_2MiB')

    def test_max_files_dedup_and_directory_entry_limit(self):
        report, inventory = self.inventory()
        paths = [self.write(f'f{i}.py', 'pass\n') for i in range(4)]
        with patch.object(worker, 'MAX_FILES', 2):
            inventory.inspect(paths[0], self.vendor)
            inventory.inspect(paths[0], self.vendor)
            for path in paths[1:]:
                inventory.inspect(path, self.vendor)
        self.assertEqual(report['files_examined'], 2)
        self.assertIn('max_files', report['limits_reached'])
        report, inventory = self.inventory()
        with patch.object(worker, 'MAX_WALK_ENTRIES', 2):
            inventory.walk_headers(self.vendor, self.vendor)
        self.assertLessEqual(report['walk_entries'], 2)
        self.assertIn('walk_entries', report['limits_reached'])

    def test_walk_depth_names_and_no_symlink_directory(self):
        self.write('headers/page.cuh', 'page_size == 1;\n')
        self.write('headers/not_allowlisted.cuh', 'must not be read')
        self.write('headers/a/b/page.cuh', 'too deep')
        outside = self.base/'external'
        outside.mkdir()
        (outside/'flash.h').write_text('outside')
        (self.vendor/'headers/link').symlink_to(outside, target_is_directory=True)
        report, inventory = self.inventory()
        with patch.object(worker, 'MAX_WALK_DEPTH', 1):
            inventory.walk_headers(self.vendor/'headers', self.vendor)
        self.assertEqual([Path(row['path']).name for row in report['files']], ['page.cuh'])
        self.assertEqual(report['walk_directories'], 2)

    def test_excerpt_and_signature_aggregate_budgets(self):
        text = '\n'.join('# mbarrier ' + '字'*2000 for _ in range(10))
        report, inventory = self.inventory()
        with patch.object(worker, 'MAX_SNIPPETS_TOTAL', 500), \
             patch.object(worker, 'MAX_SIGNATURES_TOTAL', 1):
            for i in range(3):
                path = self.write(f'e{i}.py', 'def wait_group(n): pass\n#'+text)
                inventory.inspect(path, self.vendor)
        self.assertLessEqual(sum(len(row['source_excerpt'].encode()) for row in report['files']), 500)
        self.assertTrue(all(not row['api_signatures'] for row in report['files']))
        self.assertIn('signature_export_budget', report['limits_reached'])
        self.assertLessEqual(len(worker.encoded(report)), worker.MAX_JSON)

    def test_vendor_location_rejects_user_and_config_roots(self):
        for path in ('/Users/a/site-packages/triton', '/root/site-packages/triton',
                     '/opt/sglang/.config/triton', '/etc/site-packages/triton', '/tmp/unscoped/triton'):
            self.assertFalse(worker.safe_vendor_location(Path(path)))
        self.assertTrue(worker.safe_vendor_location(Path('/opt/sglang/lib/python3.12/site-packages/triton')))

    def test_fixed_allowroot_cannot_resolve_outside_opt_sglang(self):
        with patch.object(worker.Path, 'resolve', return_value=self.vendor.resolve()):
            with self.assertRaises(ValueError):
                worker.resolve_allowroot(Path('/opt/sglang'))
        outside = self.base/'outside'
        outside.mkdir()
        root = self.vendor/'root-link'
        root.symlink_to(outside, target_is_directory=True)
        report, inventory = self.inventory()
        inventory.walk_headers(root, root)
        inventory.inspect(outside/'page.cuh', root)
        self.assertEqual(report['files_read'], 0)
        self.assertEqual(report['skips'][0]['reason'], 'invalid_vendor_allowroot')

    def test_protected_subtree_inside_vendor_allowroot_is_never_read(self):
        path = self.write('headers/.config/page.cuh', 'harmless fixture never read')
        report, inventory = self.inventory()
        with patch.object(worker.os, 'open', side_effect=AssertionError('protected read')):
            inventory.inspect(path, self.vendor)
            inventory.walk_headers(self.vendor/'headers', self.vendor)
        self.assertEqual(report['files_read'], 0)
        self.assertEqual(report['walk_directories'], 1)
        self.assertEqual(report['skips'][0]['reason'], 'protected_or_nonvendor_source_path')

    def test_deadline_and_timeout_not_swallowed(self):
        for value in (0, -1, 31, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                worker.collect_inventory(value)
        report, inventory = self.inventory()
        with patch.object(worker.Path, 'resolve', side_effect=TimeoutError('watchdog')):
            with self.assertRaises(TimeoutError):
                inventory.walk_headers(self.vendor, self.vendor)
            with self.assertRaises(TimeoutError):
                inventory.inspect(self.vendor/'x', self.vendor)

    def test_atomic_json_bound_and_symlink_output_rejection(self):
        output = self.base/'result.json'
        report = worker.initial_report()
        worker.save(output, report)
        self.assertEqual(json.loads(output.read_text())['status'], 'starting')
        self.assertFalse(output.with_suffix('.tmp').exists())
        with self.assertRaises(ValueError):
            worker.save(output, {'value':'x'*worker.MAX_JSON})
        output.unlink()
        output.symlink_to(self.base/'outside.json')
        with self.assertRaises(ValueError):
            worker.save(output, report)

    def test_main_timeout_report_and_invalid_deadline(self):
        output = self.base/'result.json'
        with patch.object(worker, 'collect_inventory', side_effect=TimeoutError('test')), \
             patch.object(worker.signal, 'signal'), patch.object(worker.signal, 'alarm'):
            self.assertEqual(worker.main(['--output', str(output)]), 1)
        result = json.loads(output.read_text())
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertEqual(result['worker_sha256'], hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest())
        with redirect_stderr(io.StringIO()):
            for value in ('0', '-1', '31', 'nan', 'inf'):
                with self.assertRaises(SystemExit):
                    worker.main(['--output', str(output), '--deadline', value])


if __name__ == '__main__':
    unittest.main()
