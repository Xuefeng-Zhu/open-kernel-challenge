"""P059 controller fixtures; no Modal SDK, compilation, CUDA or network calls."""
import ast
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib

import projection_p059_modal as p


def fixture_report():
    # Nonstandard fixture enum encodings ensure receipt validation uses the
    # returned compiled map rather than silently assuming CUDA enum integers.
    names = ['CUBLASLT_REDUCTION_SCHEME_' + n for n in ('NONE','COMPUTE_TYPE','INPLACE','OUTPUT_TYPE')]
    enums = dict(zip(names, (71,503,101,997)))
    def plan(algorithm,tile,split,scheme,position):
        return dict(m=128,n=2304,k=8192,algorithm_id=algorithm,tile_id=tile,stages_id=35,
            split_k=split,reduction_scheme=scheme,cta_swizzle=0,custom_option=0,heuristic_position=position,
            workspace_required_bytes=1024,compute='CUBLAS_COMPUTE_32F',scale='CUDA_R_32F',input_output='CUDA_R_16BF')
    plans=[plan(41,21,1,enums[names[0]],0),plan(66,20,1,enums[names[0]],1),
           plan(9,6,8,enums[names[1]],2)]
    plans[2]['reduction_scheme_name']=names[1]
    def inv(entries):return dict(requested=32,returned=len(entries),eligible=entries,rejected=[])
    report=dict(complete=True,passed=True,status='completed',phase='cleanup',native_build_complete=True,
        native_execution=True,compile=dict(complete=True,builders={n:dict(complete=True,wall_seconds=1.0) for n in ('no_split','fp32_split')}),
        source_hashes={n:p.FIXED.get(n,'1'*64) for n in p.SOURCE_NAMES},
        native_source_string_sha256=dict(no_split='0e427a632fe12c410b2f0389deadf9c3757b919999287fea5705a892aad69fc8',
            fp32_split='e3493f7e6d56c18bea112587b620a8bda0bda93cf20bb0d8ade3994ac2865c0c'),
        inventories=dict(no_split=inv(plans[:2]),fp32_split=dict(inv(plans[2:]),reduction_scheme_constants=enums)),
        selected_configurations=3,oracle_checks=298,
        nearby=[dict(layer='qkv_gate',m=127,executed=False,correctness=[])],
        assessment=dict(screen_passed=False,automatic_selection=False,official_score=False))
    runners=('p046','row41','row66','split')
    case=dict(layer='qkv_gate',m=128,n=2304,k=8192,selected=plans,runner_names=list(runners),passed=True,
        same_address_aba=True,native_private_workspaces=True,
        correctness=[dict(phase=ph,label=l,runner=r,passed=True,output_poison='NaN',same_input_address=True)
            for ph in ('before','after') for l in ('A','B','A-return','column_probes','zero','cancel') for r in runners],
        independent_graphs=dict(complete=True,separate_streams=True,distinct_mutable_buffers=True,kernel_overlap_verified=False,
            checks=[dict(repeat=i,runner=r,passed=True) for i in range(3) for r in ('original','independent')]),
        series={},activities={r:dict(available=True,correctness_passed=True,copy_events=[],
            cuda_events=[dict(symbol='fixture_kernel',duration_us=1.0,kind='kernel_or_other')]) for r in runners})
    mappings=dict(anchor_row41=('p046','row41'),anchor_row66=('p046','row66'),
                  anchor_split=('p046','split'),row66_vs_split=('row66','split'))
    for name,(parent,candidate) in mappings.items():
        series=dict(passed=True,runner_mapping=dict(parent=parent,candidate=candidate))
        for key,length,warm in (('warmup',6,True),('samples',24,False)):
            series[key]=[dict(round=i,order=['parent','candidate'] if i%2==0 else ['candidate','parent'],
                excluded_warmup=warm,correctness=dict(parent=True,candidate=True),us=dict(parent=20.0,candidate=21.0)) for i in range(length)]
        pairs=[math.sqrt((20/21)*(20/21))]*12
        series['summary']=dict(reverse_order_pair_ratios=pairs,median_paired_raw_ratio=pairs[0],winning_pairs=0)
        case['series'][name]=series
    report['cases']=[case]
    return report


def fixture_content():
    return {'report.json':json.dumps(fixture_report()).encode(), 'screen.log':b'fixture diagnostics',
            'summary.json':json.dumps(dict(complete=True,stages=[dict(mode='screen',exit_code=0)],worker_wall_seconds=1)).encode()}


def encode(content):
    raw=json.dumps({n:base64.b64encode(v).decode() for n,v in content.items()},sort_keys=True).encode()
    packed=zlib.compress(raw)
    return dict(chunks=(len(packed)+p.CHUNK-1)//p.CHUNK,packed_bytes=len(packed),encoded_bytes=len(raw),
                artifact_bytes=sum(map(len,content.values())),sha256=hashlib.sha256(packed).hexdigest(),
                artifact_sha256={n:hashlib.sha256(v).hexdigest() for n,v in content.items()}), [packed[i:i+p.CHUNK] for i in range(0,len(packed),p.CHUNK)]


class P059ControllerTests(unittest.TestCase):
    def test_exact_payload_and_artifact_scope(self):
        self.assertEqual(len(p.PAYLOAD),17);self.assertEqual(len(set(p.PAYLOAD)),17)
        self.assertEqual(len(p.SOURCE_NAMES),13)
        self.assertIn(p.SPLIT,p.PAYLOAD);self.assertNotIn('native_cublaslt_column_probe.py',p.PAYLOAD);self.assertIn(p.BRIDGE,p.PAYLOAD);self.assertIn(p.PROVENANCE_WORKER,p.PAYLOAD)
        self.assertEqual(p.ARTIFACTS,{'summary.json','screen.log','report.json'})
        self.assertIn(p.NATIVE,p.PAYLOAD);self.assertIn(p.PARENT,p.PAYLOAD)
        self.assertFalse(any('local-notes' in n or 'auth' in n or n.endswith('_modal.py') for n in p.PAYLOAD))
        self.assertEqual(p.FIXED[p.NATIVE],'a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc')

    def test_source_freeze_and_p056_preservation(self):
        for name,expected in p.FIXED.items():
            self.assertEqual(hashlib.sha256((p.ROOT/name).read_bytes()).hexdigest(),expected)
        historical={"projection_p056_modal.py":"ea8b99f4b6b7df4027e4d0d93498f67b358e6833403f3151530c7b5704f0ff22",
                    "tests/test_projection_p056_modal.py":"e78b4ba59afff4965980998a52af029ad3a27d2be813a7ca62f3c7e52d6c940d"}
        for name,digest in historical.items():
            self.assertEqual(hashlib.sha256((p.ROOT/name).read_bytes()).hexdigest(),digest)

    def test_lifecycle_validation_unchanged_beyond_explicit_identity(self):
        old=ast.parse((p.ROOT/'projection_p057_modal.py').read_text())
        new=ast.parse(Path(p.__file__).read_text().replace('P059','P057').replace('p059','p057'))
        old_functions={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
        new_functions={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
        for name in ('reserve','budget_guard','frozen_payload','make_remote_entry'):
            self.assertEqual(new_functions[name],old_functions[name],name)
        # Only receipt fields are new in launch; all resource/admission/cleanup
        # logic must normalize to the frozen P057 implementation exactly.
        class ReceiptFields(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                node.keywords=[k for k in node.keywords if k.arg not in
                    ('split_source_sha256','host_worker_sha256','capability_blocked')]
                return node
        new_launch=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='launch')
        self.assertEqual(ast.dump(ReceiptFields().visit(new_launch),include_attributes=False),old_functions['launch'])

    def test_one_invocation_bounded_resources_and_no_remote_closure(self):
        source=Path(p.__file__).read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual([n.func.attr for n in calls if n.func.attr in ('remote','remote_gen','spawn')],['remote_gen'])
        opts={k.arg:ast.literal_eval(k.value) for n in calls if n.func.attr=='function' for k in n.keywords if k.arg!='image'}
        self.assertEqual((opts['gpu'],opts['cpu'],opts['memory']),('H200',(2,2),(8192,8192)))
        self.assertEqual((opts['min_containers'],opts['max_containers'],opts['buffer_containers']),(0,1,0))
        self.assertEqual((opts['timeout'],opts['startup_timeout'],opts['scaledown_window']),(300,45,2))
        self.assertIsNone(opts['retries']);self.assertTrue(opts['block_network'] and opts['restrict_modal_access'])
        self.assertFalse(opts['include_source']);self.assertIsNone(p.make_remote_entry().__closure__)
        for token in ('child.communicate(timeout=270)','threading.Timer(345, watchdog)','start_new_session=True',
                      'os.killpg(child.pid, signal.SIGKILL)','app.run(detach=False','"--compile-native"'):
            self.assertIn(token,source)

    def test_reserve_covers_bounds_and_rate_floor(self):
        self.assertEqual(p.reserve(),Decimal(375)*p.RATE_FLOOR*Decimal('1.1')+Decimal('.02'))
        self.assertLess(p.reserve(),Decimal('.56'))
        self.assertEqual(p.reserve(Decimal('.0001')),p.reserve())
        self.assertGreater(p.reserve(Decimal('.002')),p.reserve())

    def test_budget_credits_lag_and_paid_zero(self):
        rates=dict(gpu_hour_cost_h200='4.5396',cpu_hour_cost='.04728',mem_gib_hour_cost='.00807')
        bill=dict(billed_cost='0',metered_cost='2',metered_cost_breakdown={'app':'2.1'})
        pre=dict(verified_credit_balance_usd='.65',total_usage_usd='2')
        with self.assertRaisesRegex(ValueError,'reserve'):p.budget_guard(pre,bill,rates)
        pre['verified_credit_balance_usd']='28'
        rate,reserve,floor=p.budget_guard(pre,bill,rates);self.assertEqual(floor,Decimal('27.9'))
        with self.assertRaisesRegex(ValueError,'zero'):p.budget_guard(pre,dict(bill,billed_cost='.01'),rates)

    def test_preflight_fresh_cap_workspace_and_slot(self):
        t=datetime.now(timezone.utc)
        good=dict(observed_utc=t.isoformat(),workspace='xuefeng-zhu',gpu_slot_granted=True,
                  verified_net_spend_cap_usd='0',total_spend_usd='0',verified_credit_balance_usd='28',total_usage_usd='2')
        p.validate_preflight(good,t)
        for bad in (dict(observed_utc=(t-timedelta(minutes=16)).isoformat()),dict(workspace='other'),
                    dict(gpu_slot_granted=False),dict(verified_net_spend_cap_usd='1')):
            with self.assertRaises(ValueError):p.validate_preflight(dict(good,**bad),t)

    def test_frozen_payload_exact_bytes_and_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);head='a'*40
            blobs={n:('fixture '+n).encode() for n in (*p.PAYLOAD,'projection_p059_modal.py','projection_p044_modal.py')}
            for n,v in blobs.items():path=root/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(v)
            def git(cmd,**kwargs):return head+'\n' if cmd[1]=='rev-parse' else blobs[cmd[2].split(':',1)[1]]
            fixed={n:hashlib.sha256(blobs[n]).hexdigest() for n in p.FIXED}
            with patch.object(p.subprocess,'check_output',side_effect=git):
                with self.assertRaisesRegex(ValueError,'Frozen'):p.frozen_payload(head,root)
                with patch.object(p,'FIXED',fixed):
                    files,hashes=p.frozen_payload(head,root);self.assertEqual(set(files),set(p.PAYLOAD))
                    (root/p.WORKER).write_text('changed')
                    with self.assertRaisesRegex(ValueError,'Uncommitted'):p.frozen_payload(head,root)

    def test_bad_head_and_symlink_refused(self):
        with patch.object(p.subprocess,'check_output') as run:
            with self.assertRaisesRegex(ValueError,'reviewed commit'):p.frozen_payload('HEAD')
            run.assert_not_called()
        with patch.object(p.subprocess,'check_output',return_value='b'*40+'\n'):
            with self.assertRaisesRegex(ValueError,'HEAD differs'):p.frozen_payload('a'*40)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'target').write_text('fixture');(root/p.WORKER).symlink_to(root/'target')
            with patch.object(p.subprocess,'check_output',return_value='a'*40+'\n'):
                with self.assertRaisesRegex(ValueError,'symlink'):p.frozen_payload('a'*40,root)

    def test_existing_ledger_blocks_before_auth_or_external_action(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'experiments').mkdir();(root/'experiments/projection-p059-native-modal.json').write_text('{}')
            with patch.object(p,'ROOT',root),patch.object(p,'configure_existing_auth') as auth,patch.object(p,'frozen_payload') as freeze:
                with self.assertRaisesRegex(ValueError,'reconcile'):p.launch('a'*40,root)
                auth.assert_not_called();freeze.assert_not_called()
        self.assertIn('with record_path.open("x")',Path(p.__file__).read_text())

    def test_report_source_binding(self):
        report=fixture_report();self.assertTrue(p.validate_report(report,report['source_hashes']))
        for change in (dict(complete=False),dict(failure={'type':'fixture'}),dict(source_hashes={})):
            with self.assertRaises(ValueError):p.validate_report(dict(report,**change))
        bad=json.loads(json.dumps(report));bad['source_hashes'][p.NATIVE]='0'*64
        with self.assertRaisesRegex(ValueError,'Frozen'):p.validate_report(bad)
        with self.assertRaisesRegex(ValueError,'admitted payload'):
            p.validate_report(report,dict(report['source_hashes'],**{p.WORKER:'0'*64}))

    def test_success_gate_checks_actual_native_coverage_and_counts(self):
        r=fixture_report();self.assertTrue(p.validate_report(r));self.assertEqual(r['oracle_checks'],298)
        mutations=(lambda r:r.update(native_build_complete=False),lambda r:r.update(native_execution=False),
            lambda r:r.update(passed=False),lambda r:r.update(oracle_checks=297),
            lambda r:r['cases'][0]['selected'][0].update(tile_id=22),
            lambda r:r['cases'][0]['selected'][2].update(split_k=1),
            lambda r:r['cases'][0]['selected'][2].update(reduction_scheme_name='CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE'),
            lambda r:r['cases'][0]['selected'][2].update(scale='CUDA_R_16BF'),
            lambda r:r['cases'][0]['selected'][2].update(workspace_required_bytes=64*1024*1024+1),
            lambda r:r['cases'][0]['correctness'].pop(),
            lambda r:r['cases'][0]['correctness'][0].update(passed=False),
            lambda r:r['cases'][0]['correctness'][0].update(output_poison='zero'),
            lambda r:r['cases'][0]['independent_graphs']['checks'].pop(),
            lambda r:r['cases'][0]['independent_graphs'].update(kernel_overlap_verified=True),
            lambda r:r['cases'][0]['series']['row66_vs_split']['samples'].pop(),
            lambda r:r['cases'][0]['series']['anchor_row41']['warmup'].pop(),
            lambda r:r['cases'][0]['series']['anchor_split']['samples'][0]['us'].update(parent=float('nan')),
            lambda r:r['cases'][0]['series']['anchor_split']['samples'][0]['us'].update(candidate=0),
            lambda r:r['cases'][0]['series']['anchor_split']['samples'][0].update(order=['candidate','parent']),
            lambda r:r['cases'][0]['series']['anchor_split']['summary'].update(winning_pairs=12),
            lambda r:r['cases'][0]['activities'].pop('split'),
            lambda r:r['cases'][0]['activities']['split'].update(correctness_passed=False),
            lambda r:r['cases'][0]['activities']['split']['cuda_events'][0].update(duration_us=float('inf')),
            lambda r:r['nearby'][0].update(executed=True),
            lambda r:r['assessment'].update(screen_passed=True))
        for mutate in mutations:
            r=fixture_report();mutate(r)
            with self.assertRaises(ValueError):p.validate_report(r)

    def test_compiled_enum_inventory_and_body_binding(self):
        for mutate in (
            lambda r:r['inventories']['fp32_split'].pop('reduction_scheme_constants'),
            lambda r:r['inventories']['fp32_split']['reduction_scheme_constants'].update(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE=71),
            lambda r:r['inventories']['fp32_split'].update(returned=2),
            lambda r:r['native_source_string_sha256'].update(fp32_split='0'*64),
            lambda r:r['compile']['builders']['fp32_split'].update(complete=False)):
            r=fixture_report();mutate(r)
            with self.assertRaises(ValueError):p.validate_report(r)
        # Empty CUPTI events are explicitly unavailable, not a fabricated failure.
        r=fixture_report();r['cases'][0]['activities']['split'].update(available=False,cuda_events=[])
        self.assertTrue(p.validate_report(r))

    def test_capability_absence_retained_never_execution_success(self):
        r=fixture_report();r.update(passed=False,status='capability_blocked',phase='interface_and_inventory',
            native_execution=False,oracle_checks=0,selected_configurations=0,cases=[],nearby=[],
            capability_blocked=dict(type='CapabilityBlocked',message='No actual split'))
        self.assertFalse(p.validate_report(r))
        content=fixture_content();content['report.json']=json.dumps(r).encode()
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'out';summary=p.unpack_artifacts(*encode(content),dest)
            self.assertFalse(summary['complete']);self.assertTrue(summary['capability_blocked'])
            self.assertNotIn('receipt_validation_error',summary)
            self.assertEqual((dest/'report.json').read_bytes(),content['report.json'])
        r['native_execution']=True
        with self.assertRaises(ValueError):p.validate_report(r)

    def test_complete_bytes_preserved_and_negative_speed_allowed(self):
        data=fixture_content()
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'out';summary=p.unpack_artifacts(*encode(data),dest)
            self.assertTrue(summary['complete'])
            for n,v in data.items():self.assertEqual((dest/n).read_bytes(),v)
            self.assertFalse(json.loads((dest/'report.json').read_text())['assessment']['screen_passed'])
            with self.assertRaisesRegex(ValueError,'destination exists'):p.unpack_artifacts(*encode(data),dest)

    def test_partial_failure_retained_without_success(self):
        data=fixture_content();data['report.json']=b'{"complete":false,"phase":"compile","failure":{"type":"fixture"}}'
        data['summary.json']=b'{"complete":false,"stages":[{"mode":"screen","exit_code":1}],"worker_wall_seconds":1}'
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'out';self.assertFalse(p.unpack_artifacts(*encode(data),dest)['complete'])
            self.assertEqual((dest/'report.json').read_bytes(),data['report.json'])

    def test_invalid_completed_receipt_retained_but_not_accepted(self):
        for name in ('report.json','summary.json'):
            data=fixture_content()
            data[name]=b'{"complete":true}'
            with tempfile.TemporaryDirectory() as d:
                dest=Path(d)/'out';summary=p.unpack_artifacts(*encode(data),dest)
                self.assertFalse(summary['complete']);self.assertIn('receipt_validation_error',summary)
                self.assertEqual((dest/name).read_bytes(),data[name])

    def test_unsafe_corrupt_truncated_transport_refused(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/'out';data=fixture_content();data['../bad']=b'bad'
            with self.assertRaisesRegex(ValueError,'Unsafe artifact path'):p.unpack_artifacts(*encode(data),dest)
            m,c=encode(fixture_content())
            with self.assertRaisesRegex(ValueError,'Packed hash'):p.unpack_artifacts(dict(m,sha256='0'*64),c,dest)
            with self.assertRaisesRegex(ValueError,'chunk count'):p.unpack_artifacts(m,[],dest)
            self.assertFalse(dest.exists())

    def remote_fixture(self,kind):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';dest=root/'remote';source.mkdir();path_type=type(root)
            maps={'/opt/projection-p059-native':str(source),'/tmp/projection-p059-native':str(dest)}
            commands=[]
            def popen(command,**kwargs):
                commands.append(command)
                self.assertEqual(command[2:], [str(source/p.WORKER),'--compile-native','--output',str(dest/'report.json'),
                    '--deadline-seconds','240','--outer-timeout-seconds','270'])
                self.assertTrue(kwargs['start_new_session']);kwargs['stdout'].write(b'bounded fixture log')
                (dest/'report.json').write_text(json.dumps(fixture_report()))
                if kind=='timeout':
                    (dest/'report.json.tmp').write_text('unfinished')
                    return Mock(pid=123,communicate=Mock(side_effect=[subprocess.TimeoutExpired('fixture',270),None]))
                if kind=='unknown':(dest/'secret.txt').write_text('do not transfer')
                if kind=='symlink_temp':(dest/'report.json.tmp').symlink_to(dest/'report.json')
                return Mock(returncode=1 if kind=='failure' else 0,communicate=Mock(return_value=None))
            with patch('pathlib.Path',side_effect=lambda x:path_type(maps.get(str(x),x))),patch('subprocess.Popen',side_effect=popen),patch('os.killpg') as kill:
                items=list(p.make_remote_entry()())
            self.assertEqual(len(commands),1)
            if kind in ('unknown','symlink_temp'):
                self.assertEqual(items,[dict(kind='failure',reason='artifact_path_guard')]);return
            manifest=items[0];chunks=[x['data'] for x in items if x['kind']=='chunk']
            self.assertEqual(items[-1]['kind'],'complete');self.assertNotIn('report.json.tmp',manifest['artifact_sha256'])
            summary=p.unpack_artifacts(manifest,chunks,root/'retrieved')
            self.assertEqual(summary['complete'],kind=='valid')
            if kind=='timeout':kill.assert_called_once_with(123,9);self.assertEqual(summary['stages'][0]['exit_code'],124)
            self.assertEqual((root/'retrieved/screen.log').read_bytes(),b'bounded fixture log')

    def test_remote_single_known_command_and_timeout_preservation(self):
        for kind in ('valid','failure','timeout'):self.remote_fixture(kind)

    def test_remote_unknown_files_and_symlink_temporary_refused(self):
        self.remote_fixture('unknown');self.remote_fixture('symlink_temp')

    def test_shutdown_exact_tasks_and_paid_zero_required(self):
        source=Path(p.__file__).read_text()
        for s in ('stop()\n        timer.cancel()','exact and all_stopped([exact])','record["all_apps_stopped_tasks_zero"] = all_stopped(apps)',
                  'record["paid_after_zero"] = Decimal(after["billed_cost"]) == 0','"needs-reconciliation"'):
            self.assertIn(s,source)
        self.assertTrue(p.all_stopped([dict(state='stopped',tasks='0')]))
        self.assertFalse(p.all_stopped([dict(state='stopped',tasks='1')]))

    def test_self_test_has_no_sdk_or_external_calls(self):
        with patch.object(p.subprocess,'run',side_effect=AssertionError('external')),patch.object(p.subprocess,'check_output',side_effect=AssertionError('external')), \
             patch.object(p.Path,'is_file',return_value=True),patch.object(p.Path,'is_symlink',return_value=False),patch('builtins.print'):
            self.assertEqual(p.main(['self-test','--control-root','/unused']),0)


if __name__=='__main__':unittest.main()
