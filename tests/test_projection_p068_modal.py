"""P068 launch/transport fixtures only: no SDK, GPU, authentication or services."""
import ast
import base64
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import projection_p068_modal as p
from tests.test_projection_p068_collector import valid_receipt


def fixture():
    expected=p.collector.sources()
    timing=valid_receipt();timing["source_hashes"]=expected
    content={"timing.json":json.dumps(timing).encode()}
    activity=dict(complete=True,stage="activity",source_hashes=expected,oracle_checks=18,
        timing_receipt_sha256=hashlib.sha256(content["timing.json"]).hexdigest(),
        environment=timing["environment"],library_identity=timing["library_identity"],cases=[])
    for layer,m in p.collector.CASES:
        case=dict(layer=layer,m=m,passed=True,source_sha256=p.collector.FROZEN[p.collector.SOURCE],
            activity=[],eager_full_call=[dict(repeat=i,cuda_event_us=10,acceptance_eligible=False) for i in range(3)])
        for i in range(3):
            name=f"{layer}.m{m}.repeat{i}.trace.local.json"
            raw=json.dumps(dict(traceEvents=[dict(ph="X",cat="kernel",name="fixture",ts=1,dur=2,args={})])).encode()
            content[name]=raw
            case["activity"].append(dict(repeat=i,trace_file=name,sha256=hashlib.sha256(raw).hexdigest(),
                source_sha256=p.collector.FROZEN[p.collector.SOURCE]))
        activity["cases"].append(case)
    content["activity.json"]=json.dumps(activity).encode()
    content["controller.json"]=json.dumps(dict(complete=True,stages=[dict(stage=s,exit_code=0,
        timed_out=False,worker_reaped=True) for s in ("timing","activity")])).encode()
    content["remote.json"]=json.dumps(dict(complete=True,artifact_retrieval_complete=True,worker_wall_seconds=12)).encode()
    return content,expected


def encode(content):
    raw=json.dumps({n:base64.b64encode(v).decode() for n,v in content.items()},sort_keys=True).encode()
    packed=zlib.compress(raw)
    chunks=[packed[i:i+p.CHUNK] for i in range(0,len(packed),p.CHUNK)]
    return dict(chunks=len(chunks),packed_bytes=len(packed),encoded_bytes=len(raw),
        artifact_bytes=sum(map(len,content.values())),sha256=hashlib.sha256(packed).hexdigest(),
        artifact_sha256={n:hashlib.sha256(v).hexdigest() for n,v in content.items()}),chunks


def preflight(stamp):
    return dict(observed_utc=stamp.isoformat(),workspace="xuefeng-zhu",gpu_slot_granted=True,
        verified_net_spend_cap_usd="0",total_spend_usd="0",verified_credit_balance_usd="27.53",
        total_usage_usd="2.47",authentication_verified=True,
        authentication_expiry_utc=(stamp+timedelta(hours=12)).isoformat())


class P068LauncherTests(unittest.TestCase):
    def test_exact_payload_and_no_remote_helper_file(self):
        self.assertEqual(set(p.PAYLOAD),set(p.collector.FROZEN)|{p.WORKER})
        self.assertEqual(len(p.PAYLOAD),12)
        self.assertFalse(any("auth" in n or "local-notes" in n or n.endswith("_modal.py") for n in p.PAYLOAD))
        self.assertEqual(len(p.ARTIFACTS),17);self.assertEqual(len(p.TRACES),9)
        self.assertIsNone(p.make_remote_entry().__closure__)

    def test_resources_one_invocation_and_no_outer_subprocess(self):
        source=Path(p.__file__).read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual([n.func.attr for n in calls if n.func.attr in ("remote","remote_gen","spawn")],["remote_gen"])
        settings={k.arg:ast.literal_eval(k.value) for n in calls if n.func.attr=="function" for k in n.keywords if k.arg!="image"}
        self.assertEqual((settings["gpu"],settings["cpu"],settings["memory"]),("H200",(2,2),(8192,8192)))
        self.assertEqual((settings["timeout"],settings["startup_timeout"],settings["scaledown_window"]),(300,45,2))
        self.assertEqual((settings["min_containers"],settings["max_containers"],settings["buffer_containers"]),(0,1,0))
        self.assertIsNone(settings["retries"])
        self.assertTrue(settings["block_network"] and settings["restrict_modal_access"])
        self.assertFalse(settings["include_source"])
        entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="make_remote_entry")
        attrs=[n.func.attr for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn("Popen",attrs);self.assertNotIn("check_output",attrs)
        self.assertIn('c.run(dest)',source);self.assertIn('threading.Timer(330,watchdog)',source)
        self.assertIn('app.run(detach=False',source)
        self.assertNotIn("modal-login-20261025",source)

    def test_current_auth_metadata_only_and_expiry_margin(self):
        now=datetime.now(timezone.utc);good=preflight(now)
        with tempfile.TemporaryDirectory() as d,patch.object(p,"check_owned_file") as stat, \
                patch.object(p.subprocess,"run") as run,patch.object(p.subprocess,"check_output",return_value=b""), \
                patch.dict(os.environ,{},clear=True),patch.object(Path,"read_bytes",side_effect=AssertionError("credential bytes")):
            auth,expiry=p.configure_existing_auth(Path(d),good,now)
            self.assertEqual(auth,Path(d)/"local-notes/modal-auth.toml");stat.assert_called_once_with(auth)
            self.assertEqual(expiry,good["authentication_expiry_utc"])
            self.assertEqual(run.call_args.args[0][:3],["git","check-ignore","-q"])
            for change in (dict(authentication_verified=False),dict(authentication_expiry_utc=(now+timedelta(seconds=990)).isoformat()),
                           dict(verified_net_spend_cap_usd="1")):
                with self.assertRaises(ValueError):p.configure_existing_auth(Path(d),dict(good,**change),now)

    def test_budget_reserve_rate_floor_and_delayed_billing(self):
        self.assertEqual(p.reserve(),Decimal(390)*p.RATE_FLOOR*Decimal("1.1")+Decimal(".02"))
        rates=dict(gpu_hour_cost_h200="4.5396",cpu_hour_cost=".04728",mem_gib_hour_cost=".00807")
        billing=dict(billed_cost="0",metered_cost="2.47",metered_cost_breakdown={"gpu":"2.57"})
        pre=dict(verified_credit_balance_usd="27.53",total_usage_usd="2.47")
        rate,cost,floor=p.budget_guard(pre,billing,rates)
        self.assertEqual(floor,Decimal("27.43"));self.assertGreaterEqual(rate,p.RATE_FLOOR)
        with self.assertRaises(ValueError):p.budget_guard(dict(pre,verified_credit_balance_usd=".1"),billing,rates)
        with self.assertRaises(ValueError):p.budget_guard(pre,dict(billing,billed_cost=".01"),rates)

    def test_complete_artifacts_and_owner_only_extraction(self):
        content,expected=fixture();p.validate_complete(content,expected)
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"artifacts"
            result=p.unpack_artifacts(*encode(content),dest,expected)
            self.assertTrue(result["validated_complete"])
            self.assertEqual(dest.stat().st_mode & 0o777,0o700)
            for name,raw in content.items():
                self.assertEqual((dest/name).read_bytes(),raw)
                self.assertEqual((dest/name).stat().st_mode & 0o777,0o600)
            with self.assertRaises(ValueError):p.unpack_artifacts(*encode(content),dest,expected)

    def test_activity_counts_environment_sources_and_traces_fail_closed(self):
        for mutation in (lambda a:a.update(oracle_checks=17),lambda a:a["cases"].pop(),
                         lambda a:a["cases"][0]["activity"].pop(),
                         lambda a:a["cases"][0]["eager_full_call"].pop(),
                         lambda a:a["cases"][0].update(passed=False),
                         lambda a:a.update(timing_receipt_sha256="0"*64),
                         lambda a:a["environment"].update(gpu="wrong")):
            content,expected=fixture();a=json.loads(content["activity.json"]);mutation(a)
            content["activity.json"]=json.dumps(a).encode()
            with self.assertRaises(ValueError):p.validate_complete(content,expected)
        content,expected=fixture();name=next(iter(p.TRACES));content[name]+=b" "
        with self.assertRaises(ValueError):p.validate_complete(content,expected)

    def test_failure_and_invalid_complete_receipts_are_preserved(self):
        content,expected=fixture();content["activity.json"]=b'{"complete":false}'
        with tempfile.TemporaryDirectory() as d:
            result=p.unpack_artifacts(*encode(content),Path(d)/"out",expected)
            self.assertFalse(result["validated_complete"])
            self.assertEqual((Path(d)/"out/activity.json").read_bytes(),b'{"complete":false}')
        content={"remote.json":b'{"complete":false,"artifact_retrieval_complete":false,"failure_type":"TimeoutError"}'}
        with tempfile.TemporaryDirectory() as d:
            result=p.unpack_artifacts(*encode(content),Path(d)/"out",expected)
            self.assertFalse(result["validated_complete"]);self.assertEqual(result["failure_type"],"TimeoutError")

    def test_transport_corruption_and_paths_rejected(self):
        content,expected=fixture()
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"out"
            manifest,chunks=encode(content)
            with self.assertRaises(ValueError):p.unpack_artifacts(manifest,chunks[:-1],dest,expected)
            with self.assertRaises(ValueError):p.unpack_artifacts(dict(manifest,sha256="0"*64),chunks,dest,expected)
            content["../escape"]=b"no"
            with self.assertRaises(ValueError):p.unpack_artifacts(*encode(content),dest,expected)
            self.assertFalse(dest.exists())

    def test_stream_rejects_truncation_reordering_duplicates(self):
        content,_=fixture();manifest,chunks=encode(content)
        good=[dict(kind="manifest",**manifest)]+[dict(kind="chunk",index=i,data=c) for i,c in enumerate(chunks)]
        good.append(dict(kind="complete",chunks=len(chunks)))
        self.assertEqual(p.receive_stream(iter(good)),(good[0],chunks))
        for bad in (good[:-1],[good[0],good[0]],good+[good[-1]],
                    [good[0],dict(kind="chunk",index=2,data=b"x")],
                    [good[0],dict(kind="chunk",index=0,data=b"x"*(p.CHUNK+1))]):
            with self.assertRaises(ValueError):p.receive_stream(iter(bad))

    def test_remote_calls_collector_directly_and_retains_failure(self):
        self._remote(False)
        self._remote(True)

    def test_remote_size_guard_keeps_small_receipts(self):
        self._remote(False,large=True)

    def _remote(self,failure,large=False):
        content,expected=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=root/"result";realpath=type(root)
            def mapped(path):
                return realpath(str(dest) if str(path)=="/tmp/projection-p068" else str(root/"source")
                                if str(path)=="/opt/projection-p068" else path)
            def run(output):
                self.assertEqual(output,dest);output.mkdir(mode=0o700)
                for name,raw in content.items():
                    if name!="remote.json":(output/name).write_bytes(raw)
                if large:(output/next(iter(p.TRACES))).write_bytes(b"x"*(9*1024*1024))
                return 1 if failure else 0
            with patch("pathlib.Path",side_effect=mapped),patch.object(p.collector,"run",side_effect=run) as called, \
                    patch("shutil.which",return_value=None),patch.object(sys,"path",list(sys.path)):
                items=list(p.make_remote_entry()())
            called.assert_called_once()
            manifest,chunks=p.receive_stream(iter(items))
            result=p.unpack_artifacts(manifest,chunks,root/"retrieved",expected)
            self.assertEqual(result["complete"],not failure)
            self.assertEqual(result["artifact_retrieval_complete"],not large)
            self.assertEqual(result["validated_complete"],not failure and not large)
            self.assertFalse(json.loads((root/"retrieved/runtime-metadata.json").read_text())["profiler_executed"])
            if large:
                self.assertIn(next(iter(p.TRACES)),result["omitted_artifacts"])
                self.assertTrue((root/"retrieved/timing.json").exists())

    def test_existing_record_refuses_before_auth_or_service(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"experiments").mkdir();(root/"experiments/projection-p068-modal.json").write_text("{}")
            with patch.object(p,"ROOT",root),patch.object(p,"configure_existing_auth") as auth,patch.object(p,"frozen_payload") as freeze:
                with self.assertRaisesRegex(ValueError,"reconcile"):p.launch("a"*40,root)
                auth.assert_not_called();freeze.assert_not_called()

    def test_self_test_no_native_import_or_external(self):
        with patch.object(p.subprocess,"run",side_effect=AssertionError("external")), \
                patch.object(p.subprocess,"check_output",side_effect=AssertionError("external")),patch("builtins.print"):
            self.assertEqual(p.main(["self-test"]),0)


if __name__=="__main__":unittest.main()
