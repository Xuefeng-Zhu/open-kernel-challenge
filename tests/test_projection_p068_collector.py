import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import projection_p068_collector as c
import projection_p046_confirmation as safe


def valid_receipt():
    record=dict(stage="timing",complete=True,oracle_checks=240,source_hashes={"fixture":"sha"},cases=[],
        environment=dict(python="3.12",packages={"torch":"fixture"},cuda_runtime="13.0",gpu="NVIDIA H200",
            capability=[9,0],sms=132,tf32_allowed=False,bf16_reduced_precision_reduction=True,python_executable_sha256="0"*64),
        library_identity=dict(status="unavailable",libraries=[]))
    for index,(layer,m) in enumerate(c.CASES):
        case=dict(layer=layer,m=m,passed=True,source_sha256=c.FROZEN[c.SOURCE],
            correctness=[dict(phase=p,label=l,runner=r,passed=True) for p in ("before","after")
                         for l in c.LABELS for r in ("parent","candidate")])
        for key,n,offset in (("warmup",6,100),("samples",24,200)):
            case[key]=[dict(round=i,seed=6800000+index*1000+offset+i,order=list(safe.runner_order(i)),
                excluded_warmup=key=="warmup",us={"parent":10.,"candidate":11.},
                correctness={"parent":True,"candidate":True}) for i in range(n)]
        case["graph_full_call"]=dict(median_paired_raw_ratio=10/11,aa_relative_movement=.1,aa_drift_exceeds_2pct=True)
        record["cases"].append(case)
    return record


class FakeTensor:
    def __init__(self,value,address=42):self.value,self.address=value,address
    def clone(self):return FakeTensor(self.value,self.address+1)
    def view(self,dtype):return self
    def copy_(self,other):self.value=other.value


class CollectorTests(unittest.TestCase):
    def test_dry_run_no_native_import_or_process(self):
        script="import sys, unittest.mock as m; import projection_p068_collector as c; " \
               "p=m.patch('subprocess.Popen', side_effect=AssertionError('spawn')); p.start(); " \
               "x=c.plan('/tmp/not-created-p068'); assert 'torch' not in sys.modules; " \
               "assert x['expected_checks']['timing_oracles']==240; assert not x['counter_execution_enabled']"
        subprocess.run([sys.executable,"-B","-c",script],cwd=c.ROOT,check=True,timeout=5)

    def test_frozen_payload_and_command_are_exact(self):
        p=c.plan("/tmp/path with spaces")
        self.assertEqual(set(p["payload_files"]),set(c.FROZEN)|{"projection_p068_collector.py"})
        self.assertEqual(p["source_hashes"][c.SOURCE],c.FROZEN[c.SOURCE])
        self.assertTrue(all(x["hard_seconds"]==120 for x in p["stages"]))
        self.assertIn(str(Path("/tmp/path with spaces").resolve()),p["stages"][0]["argv"])
        self.assertFalse(p["counter_execution_enabled"])
        with self.assertRaises(ValueError):c.plan("/tmp/x","not-an-image-digest")

    def test_receipt_mutations_block_activity(self):
        good=valid_receipt();c.validate_timing(good,{"fixture":"sha"})
        mutations=[lambda x:x.update(complete=False),lambda x:x.update(oracle_checks=239),
            lambda x:x["source_hashes"].update(fixture="changed"),lambda x:x["cases"].pop(),
            lambda x:x["cases"][0]["samples"].pop(),lambda x:x["cases"][0]["correctness"].pop(),
            lambda x:x["cases"][0]["samples"][0].update(seed=1),
            lambda x:x["cases"][0]["samples"][0].update(order=["candidate","parent"]),
            lambda x:x["cases"][0]["samples"][0]["us"].update(parent=float('nan')),
            lambda x:x["cases"][0]["samples"][0]["correctness"].update(parent=False),
            lambda x:x["cases"][0].update(passed=False),lambda x:x.pop("environment"),
            lambda x:x.pop("library_identity"),lambda x:x["cases"][0].pop("graph_full_call"),
            lambda x:x["cases"][0]["graph_full_call"].update(aa_relative_movement=0)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                r=copy.deepcopy(good);mutate(r)
                with self.assertRaises(ValueError):c.validate_timing(r,{"fixture":"sha"})

    def test_pair_order_keeps_oracles_after_both_members(self):
        events=[]
        class Timer:
            def measure(self,graph,*args):events.extend(["evict","start",graph,"end","sync"]);return 10.
        sample=dict(order=["parent","candidate"],us={},correctness={})
        c.measure_pair(Timer(),{"parent":"A","candidate":"B"},None,None,sample,
            lambda:events.append("input"),lambda role:events.append("oracle-"+role),lambda:None)
        self.assertEqual(events,["evict","start","A","end","sync","input",
            "evict","start","B","end","sync","input","oracle-parent","oracle-candidate"])

    def test_pair_mutation_stops_second_member(self):
        events=[]
        class Timer:
            def measure(self,graph,*args):events.append(graph);return 1.
        def changed():raise RuntimeError("mutated")
        with self.assertRaises(RuntimeError):c.measure_pair(Timer(),{"parent":"A","candidate":"B"},None,None,
            dict(order=["parent","candidate"],us={},correctness={}),changed,lambda r:None,lambda:None)
        self.assertEqual(events,["A"])

    def test_capture_mutation_is_rejected_before_fresh_input(self):
        class Torch:
            uint8="bytes"
            class cuda:
                @staticmethod
                def synchronize():pass
            @staticmethod
            def equal(a,b):return a.value==b.value
        data={"x":FakeTensor(7),"config":{}}
        def capture(kernel,data,state):
            data["x"].value=99
            return object(),FakeTensor(21)
        with patch.object(safe,"tensor_metadata",lambda x:(x.address,)):
            with self.assertRaisesRegex(RuntimeError,"[Ii]nput"):
                c.capture_checked(Torch,safe,capture,None,data,{}, {},{}, {},{},(42,))

    def test_real_replay_helper_rejects_stale_output(self):
        # The fake oracle depends on current X and fixed weights, not an always-true mock.
        class Torch:
            uint8="bytes"
            class cuda:
                @staticmethod
                def synchronize():pass
            @staticmethod
            def equal(a,b):return a.value==b.value
        class Reference:
            @staticmethod
            def check_implementation(data,out,weights):return out.value==data["x"].value*weights["w"],"linear oracle"
        data={"x":FakeTensor(0),"config":{}}
        outputs={r:FakeTensor(0) for r in ("parent","candidate")}
        class Graph:
            def __init__(self,role,stale):self.role,self.stale,self.first=role,stale,None
            def replay(self):
                if not self.stale or self.first is None:self.first=data["x"].value*3
                outputs[self.role].value=self.first
        def make(seed,dist):return {"x":FakeTensor(0 if dist=="zero" else seed)}
        metadata=lambda x:(x.address,)
        for stale in (False,True):
            graphs={r:Graph(r,stale) for r in outputs}
            with patch.object(safe,"tensor_metadata",metadata):
                def call():safe.correctness_phase(Torch,Reference,data,graphs,outputs,{"w":3},{},(42,),
                    make,100,"before",[],lambda:None,lambda:None)
                if stale:
                    with self.assertRaisesRegex(RuntimeError,"oracle failed"):call()
                else:call()

    def test_maps_filter_and_explicit_unknowns(self):
        text="1-2 r-xp 0 0 0 /usr/lib/libcublasLt.so.13\n1-2 r-xp 0 0 0 /secret/private.so\n" \
             "1-2 r-xp 0 0 0 /usr/lib/libcuda.so.1 (deleted)\n1-2 r-xp 0 0 0 /usr/lib/libcublasLt.so.13\n"
        self.assertEqual(c.mapped_library_paths(text),[Path('/usr/lib/libcublasLt.so.13')])
        t=dict(traceEvents=[dict(ph="X",cat="kernel",name="vendor",ts=1.,dur=2.,args={})])
        row=c.trace_summary(t)["events"][0]
        self.assertIsNone(row["exact_kernel_binary_sha256"])
        self.assertIsNone(row["metadata"]["grid"])
        with self.assertRaises(ValueError):c.trace_summary({})

    def test_timeout_reaps_worker_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/"worker.log"
            r=c.run_bounded([sys.executable,"-u","-c","import time;print('started');time.sleep(30)"],log,.3)
            self.assertTrue(r["timed_out"]);self.assertTrue(r["worker_reaped"])
            self.assertFalse(r["cloud_app_stop_verified"])
            self.assertIn("started",log.read_text())
            self.assertEqual(log.stat().st_mode & 0o777,0o600)

    def test_controller_does_not_launch_activity_after_timeout(self):
        with tempfile.TemporaryDirectory() as d,patch.object(c.platform,"system",return_value="Linux"), \
             patch.object(c,"run_bounded",return_value=dict(exit_code=-9,timed_out=True,worker_reaped=True)) as run:
            output=Path(d)/"new"
            self.assertEqual(c.run(output),1);self.assertEqual(run.call_count,1)
            self.assertFalse(json.loads((output/"controller.json").read_text())["complete"])
            with self.assertRaises(FileExistsError):c.run(output)

    def test_supervisor_terminates_descendant_group(self):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/"descendant-terminated"
            child="import signal,time,sys; from pathlib import Path; " \
                  "signal.signal(signal.SIGTERM,lambda *a:(Path(sys.argv[1]).write_text('term'),sys.exit(0))); time.sleep(30)"
            parent=f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r},{str(marker)!r}]); time.sleep(30)"
            r=c.run_bounded([sys.executable,"-c",parent],Path(d)/"group.log",.8)
            self.assertTrue(r["timed_out"]);self.assertTrue(r["worker_reaped"])
            self.assertEqual(marker.read_text(),"term")

    def test_unsupervised_worker_refuses_before_torch_import(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True),patch.object(c.traceback,"print_exc"):
            self.assertEqual(c.native_worker("timing",Path(d)),1)
            r=json.loads((Path(d)/"timing.json").read_text())
            self.assertEqual(r["failure"]["type"],"RuntimeError")
            self.assertEqual(r["oracle_checks"],0)


if __name__=="__main__":unittest.main()
