"""Pure-local counter policy/parser guards; no tool, CUDA or network execution."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('counter_policy',HERE/'counters.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def result(out='',code=0):
    return dict(exit_code=code,stdout=out,stderr='',output_bytes=len(out),truncated=False)


class CounterGuards(unittest.TestCase):
    def test_query_selects_only_exact_available_max_eight(self):
        names=[v[0] for _,v in c.METRIC_GROUPS]
        self.assertEqual(len(names),8)
        self.assertEqual(set(c.select_metrics('\n'.join(names)).values()),set(names))
        self.assertEqual(c.select_metrics('NOTgpu__time_duration.sumEXTRA'),{})
        self.assertTrue(c.denied(result('CUPTI_ERROR_INSUFFICIENT_PRIVILEGES',1)))

    def test_command_is_fixed_replay_safe_no_privilege_changes(self):
        argv=c.command('/ncu','fa3',['gpu__time_duration.sum'],Path('/tmp/out'),Path('/target'),Path('/check'))
        text=' '.join(argv)
        for fragment in ['--replay-mode application','--app-replay-match grid','--app-replay-mode strict',
            '--nvtx-include PROFILE_ONE/','--launch-count 1','--cache-control none','--clock-control none']:
            self.assertIn(fragment,text)
        self.assertFalse(any(x in argv for x in ['sudo','--set','--kill']))
        self.assertIn('FlashAttnFwdSm90',text)
        self.assertIn('_fused',' '.join(c.command('/ncu','bf16_custom',['x'],Path('/o'),Path('/t'),Path('/c'))))

    def test_first_permission_denial_stops_second_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dest=root/'dest';dest.mkdir()
            (root/'inventory.json').write_text(json.dumps({'tools':{'ncu':{'status':'found'}}}))
            replies=[result(' '.join(c.REQUIRED_FLAGS)),result('gpu__time_duration.sum'),result('ERR_NVGPUCTRPERM: denied',1)]
            with patch.object(c,'NCU',sys.executable),patch.object(c,'execute',side_effect=replies) as invoke:
                report=c.run_counters(root,dest,time.monotonic()+100)
            self.assertEqual(report['status'],'counter_permission_denied')
            self.assertEqual(report['sessions_attempted'],1)
            self.assertEqual(len(report['records']),1)
            self.assertEqual(invoke.call_count,3)
            query_argv=invoke.call_args_list[1].args[0]
            self.assertIn('suffix',query_argv);self.assertIn('--metrics',query_argv)
            self.assertNotIn('all',query_argv)
            self.assertEqual(list(dest.iterdir()),[])

    def test_process_group_killed_on_timeout(self):
        child=MagicMock();child.pid=123;child.communicate.side_effect=[subprocess.TimeoutExpired('ncu',1),(b'',b'')]
        with patch.object(c.subprocess,'Popen',return_value=child),patch.object(c.os,'killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):c.execute(['ncu'],1,100)
            kill.assert_called_once_with(123,c.signal.SIGKILL)

    def test_csv_layouts_preserve_units_and_values(self):
        self.assertEqual(c.csv_metrics('"Metric Name","Metric Unit","Metric Value"\n"x","byte","42"\n',{'x'}),
                         [dict(name='x',value='42',unit='byte')])
        self.assertEqual(c.csv_metrics('"ID","x"\n"","byte"\n"0","42"\n',{'x'}),
                         [dict(name='x',value='42',unit='byte')])
        self.assertEqual(c.csv_metrics('unexpected',{'x'}),[])

    def test_target_has_only_fenced_single_profile_replay_and_oracles(self):
        text=(HERE/'counter_target.py').read_text();tree=ast.parse(text)
        self.assertIn("range_push('PROFILE_ONE')",text)
        self.assertIn("check('pre_profile')",text);self.assertIn("check('post_profile')",text)
        self.assertIn('ref.check_implementation(work, raw)',text)
        self.assertIn('h.unchanged(work, fresh)',text)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in {'remote','remote_gen','pip_install','apt_install'} for n in ast.walk(tree)))


if __name__=='__main__':unittest.main(verbosity=2)
