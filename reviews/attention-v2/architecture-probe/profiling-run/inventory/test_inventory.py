"""Local safety/transport checks. No inventory executable or Modal operation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('inventory_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

worker=load('worker')
controller=load('controller')


class InventoryTests(unittest.TestCase):
    def test_no_cuda_or_candidate_imports_or_gpu_queries(self):
        tree=ast.parse((HERE/'worker.py').read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
        self.assertFalse(any(n and n.split('.')[0] in {'torch','triton','cupy','ctypes','modal'} for n in imports))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) for n in ast.walk(tree)))
        flattened=[arg for commands in worker.COMMANDS.values() for command in commands for arg in command]
        self.assertNotIn('--query-metrics',flattened)
        self.assertNotIn('--query-metrics-mode',flattened)
        self.assertNotIn('nvidia-smi',flattened)

    def test_only_metadata_commands_and_bounded_search(self):
        self.assertEqual(worker.COMMANDS,{'ncu':[('--version',),('--help',),('--list-sections',)],
            'nsys':[('--version',),('profile','--help')]})
        self.assertEqual(set(worker.TOOL_PATTERNS),{'ncu','nsys'})
        self.assertTrue(all(p.startswith(('/opt/nvidia/','/usr/local/cuda')) for rows in worker.TOOL_PATTERNS.values() for p in rows))
        self.assertLessEqual(worker.MAX_COMMAND_OUTPUT,20000)
        self.assertLessEqual(worker.MAX_RESULT,128000)
        self.assertEqual(worker.DEADLINE,20)

    def test_cpu_resource_and_payload_allowlist(self):
        tree=ast.parse((HERE/'controller.py').read_text())
        functions=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='function']
        self.assertEqual(len(functions),1)
        kw={k.arg:ast.literal_eval(k.value) for k in functions[0].keywords if k.arg not in {'image'}}
        for name,want in {'gpu':None,'cpu':(2,2),'memory':(8192,8192),'max_containers':1,
                'min_containers':0,'buffer_containers':0,'retries':0,'timeout':30,'startup_timeout':15,
                'include_source':False,'block_network':True,'restrict_modal_access':True}.items():
            self.assertEqual(kw[name],want)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(sum(n.func.attr=='add_local_file' for n in calls),1)
        self.assertFalse(any(n.func.attr in {'add_local_dir','pip_install','apt_install'} for n in calls))
        self.assertEqual(sum(n.func.attr=='remote' for n in calls),1)

    def test_transport_rejects_changed_or_non_cpu_report(self):
        report=dict(gpu_requested=False,cuda_context_requested=False,candidate_kernel_launches=0)
        raw=json.dumps(report)
        reply=dict(report_json=raw,report_sha256=hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(controller.verify_result(reply)[1],report)
        with self.assertRaises(AssertionError):controller.verify_result({**reply,'report_sha256':'0'*64})
        for field,bad in [('gpu_requested',True),('cuda_context_requested',True),('candidate_kernel_launches',1)]:
            badraw=json.dumps({**report,field:bad})
            with self.assertRaises(AssertionError):
                controller.verify_result(dict(report_json=badraw,report_sha256=hashlib.sha256(badraw.encode()).hexdigest()))

    def test_remote_entry_has_no_local_closure_and_hard_kill(self):
        self.assertIsNone(controller.make_remote_entry().__closure__)
        tree=ast.parse((HERE/'controller.py').read_text())
        entry=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='entry')
        text=ast.unparse(entry)
        self.assertIn('timeout=25',text)
        self.assertIn('signal.SIGKILL',text)
        self.assertNotIn('AUTH',text)
        self.assertNotIn('modal',text.lower())

    def test_reserve_and_repository_root(self):
        from decimal import Decimal
        rate=(2*Decimal('.0473')+8*Decimal('.008'))/3600
        self.assertLessEqual(70*rate*2+Decimal('.01'),controller.RESERVE)
        self.assertEqual(controller.RESERVE,Decimal('.02'))
        self.assertEqual(controller.REPO,HERE.parents[4])
        self.assertEqual(controller.MAX_RESULT,worker.MAX_RESULT)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InventoryTests))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        network_calls=0,gpu_launches=0,remote_jobs=0,worker_sha256=hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest(),
        controller_sha256=hashlib.sha256((HERE/'controller.py').read_bytes()).hexdigest())
    (HERE/'local-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
