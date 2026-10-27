"""Local source, budget, and interrupted-log preservation guards. No network/GPU."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load(name):
 s=importlib.util.spec_from_file_location('access_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
w=load('access_worker');c=load('controller')

class Guards(unittest.TestCase):
 def test_one_metric_and_exact_child(self):
  argv=w.command(Path('/source'),Path('/scratch'))
  self.assertEqual(argv[argv.index('--metrics')+1],'dram__bytes_read.sum')
  self.assertEqual(argv[argv.index('--arm')+1],'fa3')
  self.assertNotIn('--query-metrics',argv)
  self.assertIn('/source/counter_target.py',argv)
  self.assertEqual(w.FROZEN['worker.py'],'07c6c36b06995df46b49b90208b0454eba1aa3efb17787c7a2fbdf67b1efa326')
  self.assertEqual(w.FROZEN['counter_target.py'],'910e0b3613681edf318d43df25356d8c9ed286b416c2757063d4c0d431142c3f')
 def test_no_cuda_parent_and_one_process(self):
  tree=ast.parse((HERE/'access_worker.py').read_text())
  imports=[n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]
  self.assertFalse(any(n.split('.')[0] in {'torch','triton','ctypes','cupy','modal'} for n in imports))
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  self.assertEqual(sum(n.func.attr=='Popen' for n in calls),1)
  self.assertFalse(any(n.func.attr=='profile' for n in calls))
 def test_actual_and_recorded_resources(self):
  tree=ast.parse((HERE/'controller.py').read_text())
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='function')
  kw={v.arg:ast.literal_eval(v.value) for v in node.keywords if v.arg!='image'}
  self.assertEqual((kw['gpu'],kw['cpu'],kw['memory'],kw['timeout'],kw['startup_timeout']),('H200',(2,2),(8192,8192),70,10))
  text=(HERE/'controller.py').read_text()
  self.assertIn('worker_alarm=60, child_hard_timeout=65, function_timeout=70',text)
  self.assertIn('startup_timeout=10, app_watchdog=80',text)
  self.assertEqual(set(c.ALLOWED_ARTIFACTS),{'summary.json','fa3.ncu-rep','fa3-counter-check.json'})
 def test_budget_closed_actuals(self):
  from decimal import Decimal
  rates=dict(cpu_hour_cost='.0473',mem_gib_hour_cost='.008',gpu_hour_cost_h200='4.54')
  a,b=c.closed_allowances(dict(client_wall_seconds=12.749726542038843),dict(client_wall_seconds=131.11781441699713),rates)
  self.assertEqual(a,Decimal('.011586'))
  self.assertLessEqual(a+b+c.RESERVE,Decimal('.67'))
  self.assertLessEqual(100*c.resource_rate(rates)*2+Decimal('.01'),c.RESERVE)
  with self.assertRaises(AssertionError):c.closed_allowances(dict(client_wall_seconds=999),dict(client_wall_seconds=999),rates)
 def test_global_alarm_logs_survive_scratch_cleanup(self):
  class Child:
   pid=123456
   def __init__(self,argv,**kw):
    kw['stdout'].write(b'partial stdout before global alarm\n');kw['stdout'].flush()
    kw['stderr'].write(b'CUPTI_ERROR_INSUFFICIENT_PRIVILEGES\n');kw['stderr'].flush()
    folder=Path(argv[argv.index('--output')+1]).parent
    (folder/'fa3-counter-check.json').write_text('{"passed":false,"partial":true}')
    (folder/'fa3.ncu-rep').write_bytes(b'partial binary')
    self.waits=0
   def wait(self,timeout):
    self.waits+=1
    if self.waits==1:raise TimeoutError('mock global alarm')
    return -9
   def poll(self):return None
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'artifacts'
   def fake_sha(path):return w.NCU_SHA if str(path)==w.NCU else w.FROZEN.get(Path(path).name,'source')
   with patch.object(w,'sha',side_effect=fake_sha),patch.object(w.Path,'is_file',return_value=True),patch.object(w.subprocess,'Popen',Child),patch.object(w.os,'killpg'),patch.object(w.signal,'alarm'),patch.object(w.signal,'signal'):
    code=w.run(dest)
   report=json.loads((dest/'summary.json').read_text())
   self.assertEqual(code,1);self.assertEqual(report['hardware_counter_sessions'],1)
   self.assertTrue(report['permission_denied']);self.assertTrue(report['timed_out'])
   self.assertIn('partial stdout',report['stdout']['text'])
   self.assertEqual(report['validation']['partial'],True)
   self.assertEqual((dest/'fa3.ncu-rep').read_bytes(),b'partial binary')

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
 import hashlib
 receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,gpu_calls=0,network_calls=0,
 source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['access_worker.py','controller.py','test_access.py']})
 (HERE/'local-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful() else 1)
