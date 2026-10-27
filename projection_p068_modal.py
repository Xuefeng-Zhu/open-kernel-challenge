"""One bounded P068 P046-only diagnostic; self-test performs no service calls.

The collector owns native child supervision. This local controller owns the
ephemeral app, result transport, budget reservation and verified app shutdown.
Only the collector's 12 frozen files are uploaded; this launcher stays local.
"""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import zlib

import projection_p068_collector as collector
from projection_p044_modal import (now, require, resource_rate, gross,
    validate_preflight, check_owned_file, cli, all_stopped, MAX_RAW, MAX_PACKED, CHUNK)

ROOT = Path(__file__).resolve().parent
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
PROFILE = "xuefeng-zhu"
RATE_FLOOR = Decimal("0.001305166667")
WORKER = "projection_p068_collector.py"
PAYLOAD = tuple(collector.FROZEN) + (WORKER,)
TRACES = frozenset(f"{layer}.m{m}.repeat{i}.trace.local.json"
    for layer, m in collector.CASES for i in range(3))
ARTIFACTS = TRACES | frozenset(("plan.json", "controller.json", "timing.json", "activity.json",
    "runtime-metadata.json", "remote.json", "timing.log.local.txt", "activity.log.local.txt"))
RESERVE_SECONDS = 390
save = collector.atomic_json


def reserve(rate=RATE_FLOOR):
    return Decimal(RESERVE_SECONDS) * max(rate, RATE_FLOOR) * Decimal("1.1") + Decimal(".02")


def budget_guard(preflight, billing, rates):
    require(Decimal(billing["billed_cost"]) == 0, "Actual paid usage must remain zero")
    rate = max(RATE_FLOOR, resource_rate(rates))
    amount = reserve(rate)
    free_floor = Decimal(preflight["verified_credit_balance_usd"]) - max(
        Decimal(0), gross(billing) - Decimal(preflight["total_usage_usd"]))
    require(free_floor > amount, "Free credits cannot cover conservative reserve")
    return rate, amount, free_floor


def configure_existing_auth(control_root, preflight, observed=None):
    """Use verified expiry metadata and stat only; never read credential bytes."""
    observed = observed or datetime.now(timezone.utc)
    validate_preflight(preflight, observed)
    require(preflight.get("authentication_verified") is True, "Existing authentication not verified")
    expiry = datetime.fromisoformat(preflight["authentication_expiry_utc"].replace("Z", "+00:00"))
    require(expiry.tzinfo is not None and (expiry-observed).total_seconds() > RESERVE_SECONDS + 600,
            "Existing authentication expiry cannot cover this diagnostic")
    auth = control_root / "local-notes/modal-auth.toml"
    check_owned_file(auth)
    blocked = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN",
               "MODAL_OAUTH_CLIENT_ID", "MODAL_OAUTH_CLIENT_SECRET")
    require(not any(os.environ.get(k) for k in blocked), "Unexpected credential environment override")
    subprocess.run(["git", "check-ignore", "-q", "local-notes/modal-auth.toml"],
                   cwd=control_root, capture_output=True, check=True)
    require(not subprocess.check_output(["git", "ls-files", "local-notes/modal-auth.toml"],
                                       cwd=control_root), "Credential file is tracked")
    os.environ["MODAL_CONFIG_PATH"] = str(auth)
    os.environ["MODAL_PROFILE"] = PROFILE
    return auth, expiry.isoformat()


def frozen_payload(head, root=ROOT):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", head)), "Exact reviewed commit required")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                    text=True).strip() == head, "HEAD differs from freeze")
    hashes = {}
    for name in (*PAYLOAD, "projection_p068_modal.py", "projection_p044_modal.py"):
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Payload missing or symlink: " + name)
        raw = path.read_bytes()
        require(raw == subprocess.check_output(["git", "show", head + ":" + name], cwd=root),
                "Uncommitted or changed payload: " + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    require(all(hashes[name] == value for name, value in collector.FROZEN.items()), "Frozen source mismatch")
    return {name: root / name for name in PAYLOAD}, hashes


def validate_complete(content, expected):
    """Validate the exact three-case receipt and nine trace bindings, not P053 counts."""
    controller = json.loads(content["controller.json"])
    require(controller.get("complete") is True and
        [(s.get("stage"), s.get("exit_code"), s.get("timed_out")) for s in controller.get("stages", [])]
        == [("timing", 0, False), ("activity", 0, False)], "Collector stages incomplete")
    require(all(s.get("worker_reaped") is True for s in controller["stages"]), "Native worker not reaped")
    timing = collector.validate_timing(json.loads(content["timing.json"]), expected)
    activity = json.loads(content["activity.json"])
    require(activity.get("complete") is True and activity.get("stage") == "activity"
            and not activity.get("failure") and activity.get("oracle_checks") == 18
            and activity.get("source_hashes") == expected, "Activity coverage/source mismatch")
    require(activity.get("timing_receipt_sha256") == hashlib.sha256(content["timing.json"]).hexdigest(),
            "Activity timing receipt hash mismatch")
    require(activity.get("environment") == timing["environment"], "Stage environment mismatch")
    libs = activity.get("library_identity", {})
    require(libs.get("status") in ("available", "unavailable") and isinstance(libs.get("libraries"), list),
            "Activity DSO identity/absence missing")
    old = {(x["basename"], x.get("sha256")) for x in timing["library_identity"]["libraries"]}
    new = {(x["basename"], x.get("sha256")) for x in libs["libraries"]}
    require(old.issubset(new), "Activity DSO identity changed")
    cases = activity.get("cases", [])
    require([(c.get("layer"), c.get("m")) for c in cases] == list(collector.CASES), "Activity case mismatch")
    names = set()
    for case in cases:
        require(case.get("passed") is True and case.get("source_sha256") == collector.FROZEN[collector.SOURCE],
                "Activity correctness/source mismatch")
        profiles = case.get("activity", [])
        require(len(profiles) == 3 and [p.get("repeat") for p in profiles] == [0, 1, 2], "Activity repetitions missing")
        eager = case.get("eager_full_call", [])
        require(len(eager) == 3 and [p.get("repeat") for p in eager] == [0, 1, 2], "Eager diagnostic repetitions missing")
        for profile in profiles:
            name = f"{case['layer']}.m{case['m']}.repeat{profile['repeat']}.trace.local.json"
            require(profile.get("trace_file") == name and name in content
                    and profile.get("sha256") == hashlib.sha256(content[name]).hexdigest()
                    and profile.get("source_sha256") == collector.FROZEN[collector.SOURCE], "Trace identity mismatch")
            collector.trace_summary(json.loads(content[name]))
            names.add(name)
    require(names == TRACES, "Nine traces required")
    return True


def unpack_artifacts(manifest, chunks, dest, expected):
    require(not dest.exists() and not dest.is_symlink(), "Artifact destination exists; reconcile instead of replacing")
    require(0 < manifest["chunks"] <= MAX_PACKED // CHUNK and len(chunks) == manifest["chunks"], "Chunk count mismatch")
    require(all(isinstance(c, bytes) and 0 < len(c) <= CHUNK for c in chunks), "Invalid chunk size")
    packed = b"".join(chunks)
    require(len(packed) == manifest["packed_bytes"] <= MAX_PACKED
            and hashlib.sha256(packed).hexdigest() == manifest["sha256"], "Packed size/hash mismatch")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, 12 * 1024 * 1024 + 1)
    require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
            and len(raw) == manifest["encoded_bytes"] <= 12 * 1024 * 1024, "Invalid/oversized compressed stream")
    encoded = json.loads(raw)
    require(isinstance(encoded, dict) and set(encoded) == set(manifest["artifact_sha256"])
            and "remote.json" in encoded and set(encoded) <= ARTIFACTS, "Unsafe artifact manifest/path")
    content = {n: base64.b64decode(v, validate=True) for n, v in encoded.items()}
    require(sum(map(len, content.values())) == manifest["artifact_bytes"] <= MAX_RAW
            and all(hashlib.sha256(v).hexdigest() == manifest["artifact_sha256"][n] for n, v in content.items()),
            "Artifact size/hash mismatch")
    # Preserve authentic failure receipts even if native validation is incomplete.
    dest.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, value in content.items():
        fd = os.open(dest / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
    result = json.loads(content["remote.json"])
    result["validated_complete"] = False
    if result.get("complete") and result.get("artifact_retrieval_complete"):
        try:
            validate_complete(content, expected)
            result["validated_complete"] = True
        except (ValueError, KeyError, TypeError) as error:
            result["validation_error_type"] = type(error).__name__
    return result


def make_remote_entry():
    # No local auth/config/client closure, extra remote helper file, or outer child.
    def entry():
        import base64, hashlib, json, os, shutil, sys, time, zlib
        from pathlib import Path
        root = Path("/opt/projection-p068")
        dest = Path("/tmp/projection-p068")
        sys.path.insert(0, str(root))
        import projection_p068_collector as c
        started = time.monotonic()
        result = dict(complete=False, artifact_retrieval_complete=True, cloud_app_stop_verified=False)
        # Presence/restriction metadata only. Never execute a profiler or query counters.
        metadata = dict(executables_present={name: shutil.which(name) is not None for name in ("ncu", "nsys", "nvcc")},
            profiler_executed=False, counter_query_executed=False, profiling_admin_restriction=None,
            restriction_scope="Driver parameter metadata; not a proof counters are accessible")
        try:
            params = Path("/proc/driver/nvidia/params")
            if params.is_file() and params.stat().st_size < 1024 * 1024:
                for line in params.read_text().splitlines():
                    key, sep, value = line.partition(":")
                    if sep and key.strip() == "RmProfilingAdminOnly" and value.strip() in ("0", "1"):
                        metadata["profiling_admin_restriction"] = int(value.strip())
        except OSError:
            metadata["restriction_status"] = "unreadable"
        # Inspect availability first, but let the collector create its exclusive
        # output directory. Metadata absence does not imply activity cannot run.
        try:
            result["collector_exit_code"] = c.run(dest)
            result["complete"] = result["collector_exit_code"] == 0
        except Exception as error:
            result["failure_type"] = type(error).__name__
        if not dest.exists():
            dest.mkdir(mode=0o700, exist_ok=False)
        c.atomic_json(dest / "runtime-metadata.json", metadata)
        result["worker_wall_seconds"] = time.monotonic() - started
        c.atomic_json(dest / "remote.json", result)
        allowed = {"plan.json", "controller.json", "timing.json", "activity.json", "runtime-metadata.json", "remote.json",
                   "timing.log.local.txt", "activity.log.local.txt"}
        allowed.update(f"{layer}.m{m}.repeat{i}.trace.local.json" for layer,m in c.CASES for i in range(3))
        files = []
        for path in dest.iterdir():
            if path.name.endswith(".json.tmp") and path.name[:-4] in allowed and path.is_file() and not path.is_symlink():
                continue
            if path.name not in allowed or not path.is_file() or path.is_symlink():
                yield dict(kind="failure", reason="artifact_path_guard")
                return
            files.append(path)
        inventory = {}
        for path in files:
            inventory[path.name] = dict(bytes=path.stat().st_size,
                sha256=c.sha(path) if path.stat().st_size <= 32*1024*1024 else None)
        def encode(content):
            raw = json.dumps({n:base64.b64encode(v).decode() for n,v in content.items()},sort_keys=True).encode()
            return raw, zlib.compress(raw, 9)
        content = {p.name:p.read_bytes() for p in files} if sum(x["bytes"] for x in inventory.values()) <= 8*1024*1024 else {}
        raw, packed = encode(content)
        if not content or len(packed) > 2*1024*1024:
            # Preserve small receipts and hashes rather than lose all evidence to a large trace.
            content = {p.name:p.read_bytes() for p in files if not p.name.endswith(".trace.local.json")
                       and p.stat().st_size <= 512*1024}
            result.update(artifact_retrieval_complete=False, omission_reason="transport_size_guard",
                          artifact_inventory=inventory, omitted_artifacts=sorted(set(inventory)-set(content)))
            content["remote.json"] = json.dumps(result).encode()
            raw, packed = encode(content)
        if sum(map(len,content.values())) > 8*1024*1024 or len(packed)>2*1024*1024:
            content = {"remote.json":json.dumps(dict(complete=False,artifact_retrieval_complete=False,
                omission_reason="summary_size_guard",artifact_inventory=inventory)).encode()}
            raw,packed=encode(content)
        count=(len(packed)+16383)//16384
        yield dict(kind="manifest",chunks=count,packed_bytes=len(packed),encoded_bytes=len(raw),
            artifact_bytes=sum(map(len,content.values())),sha256=hashlib.sha256(packed).hexdigest(),
            artifact_sha256={n:hashlib.sha256(v).hexdigest() for n,v in content.items()})
        for index in range(count):
            yield dict(kind="chunk",index=index,data=packed[index*16384:(index+1)*16384])
        yield dict(kind="complete",chunks=count)
    return entry


def receive_stream(items):
    manifest, chunks, complete = None, [], False
    for item in items:
        require(not complete, "Data after complete marker")
        if item.get("kind") == "manifest":
            require(manifest is None and 0 < item["packed_bytes"] <= MAX_PACKED, "Repeated/invalid manifest")
            manifest = item
        elif item.get("kind") == "chunk":
            require(manifest is not None and item["index"] == len(chunks), "Chunk ordering mismatch")
            require(isinstance(item["data"], bytes) and 0 < len(item["data"]) <= CHUNK, "Chunk size invalid")
            chunks.append(item["data"])
            require(sum(map(len,chunks)) <= MAX_PACKED, "Packed size guard")
        elif item.get("kind") == "complete":
            require(manifest is not None and item["chunks"] == len(chunks) == manifest["chunks"], "Incomplete stream")
            complete = True
        else:
            raise RuntimeError("Remote artifact guard failed; inspect existing app")
    require(complete and manifest is not None, "Incomplete result; reconcile same app, never relaunch")
    return manifest, chunks


def launch(head, control_root):
    control_root = control_root.resolve()
    record_path = ROOT / "experiments/projection-p068-modal.json"
    artifacts = ROOT / "local-notes/projection-p068-artifacts"
    require(not record_path.exists() and not record_path.is_symlink()
            and not artifacts.exists() and not artifacts.is_symlink(), "Existing diagnostic state; reconcile, never relaunch")
    files, hashes = frozen_payload(head)
    expected = {name:hashes[name] for name in PAYLOAD}
    pre_path = control_root / "local-notes/modal-p068-preflight.local.json"
    require(pre_path.is_file() and not pre_path.is_symlink(), "Missing current protected preflight")
    pre = json.loads(pre_path.read_text())
    auth, expiry = configure_existing_auth(control_root, pre)
    subprocess.run(["git","check-ignore","-q","local-notes/projection-p068-artifacts/remote.json"],
                   cwd=ROOT,capture_output=True,check=True)
    slot = control_root / "local-notes/gpu-slot.local.json"
    prior = json.loads(slot.read_text())
    require(prior.get("state") in ("idle","terminal") and not prior.get("active_reservation"), "GPU reservation needs reconciliation")
    billing = cli(control_root,"billing","summary","--json")
    require(all_stopped(cli(control_root,"app","list","--env","main","--json")), "An app is active")
    rate, cost, free = budget_guard(pre,billing,cli(control_root,"billing","rates","--json"))
    import warnings
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings("ignore",message=r"Your Modal token .* expires",category=ServerWarning)
    require(Path(modal.config.user_config_path).resolve()==auth.resolve(),"SDK uses unexpected config")
    image=modal.Image.from_id(IMAGE)
    for name,path in files.items():image=image.add_local_file(path,"/opt/projection-p068/"+name,copy=False)
    app=modal.App("okc-projection-p068-baseline",include_source=False)
    fn=app.function(image=image,gpu="H200",cpu=(2,2),memory=(8192,8192),
        min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=2,
        timeout=300,startup_timeout=45,retries=None,serialized=True,include_source=False,
        block_network=True,restrict_modal_access=True)(make_remote_entry())
    record=dict(experiment="P068 fixed P046 baseline diagnosis",state="reserved",app_id=None,
        created_utc=now(),source_commit=head,payload_sha256=expected,controller_sha256=hashes["projection_p068_modal.py"],
        shared_controller_sha256=hashes["projection_p044_modal.py"],base_image=IMAGE,
        cost_reserve_usd=str(cost),resource_rate_usd_per_second=str(rate),verified_free_credit_floor_usd=str(free),
        paid_limit_usd="0",billing_before=billing,authentication_expiry_utc=expiry,
        preflight_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        resources=dict(gpu="H200",cpu=[2,2],memory_mib=[8192,8192],max_containers=1,min_containers=0,
            buffer_containers=0,function_timeout=300,startup_timeout=45,app_watchdog=330,
            stage_internal=90,stage_hard=120,stages=2,reserve_seconds=390,retry_policy=None,explicit_invocations=1),
        no_hardware_counters=True,not_official_score=True)
    fd=os.open(record_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w") as stream:stream.write(json.dumps(record,indent=2)+"\n")
    save(slot,dict(state="reserved",provider="modal",experiment="P068",app_id=None))
    started=time.monotonic();lock=threading.Lock();stop_result={};expired=threading.Event()
    def stop():
        with lock:
            if not app.app_id or stop_result:return
            stop_result.update(app_id=app.app_id,requested_utc=now())
            try:stop_result.update(cli(control_root,"app","stop",app.app_id,"--yes","--env","main"))
            except Exception as error:stop_result["error_type"]=type(error).__name__
    def watchdog():expired.set();stop()
    timer=threading.Timer(330,watchdog);timer.daemon=True
    try:
        timer.start()
        with app.run(detach=False,environment_name="main"):
            record.update(state="accepted",app_id=app.app_id,accepted_utc=now());save(record_path,record)
            save(slot,dict(state="accepted",provider="modal",experiment="P068",app_id=app.app_id))
            print("Accepted P068 app "+app.app_id,flush=True)
            require(not expired.is_set(),"Watchdog elapsed before invocation")
            manifest,chunks=receive_stream(fn.remote_gen())
            record["transport_manifest"]=manifest;save(record_path,record)
            result=unpack_artifacts(manifest,chunks,artifacts,expected)
            record.update(state="function-terminal",artifact_retrieval_complete=result.get("artifact_retrieval_complete",False),
                diagnostic_complete=result.get("validated_complete",False),worker_wall_seconds=result.get("worker_wall_seconds"),
                native_complete=result.get("complete",False),validation_error_type=result.get("validation_error_type"))
            save(record_path,record);stop()
    except Exception as error:
        record.update(state="needs-reconciliation",app_id=app.app_id,error_type=type(error).__name__)
        print("P068 needs reconciliation of the existing request; no relaunch.",flush=True)
    finally:
        stop();timer.cancel()
        record.update(explicit_stop=stop_result,client_finished_utc=now(),client_wall_seconds=time.monotonic()-started)
        try:
            # Bounded read-only reconciliation; a stop acknowledgement is not stopped/tasks0.
            observations=[]
            for attempt in range(3):
                apps=cli(control_root,"app","list","--env","main","--json")
                exact=next((a for a in apps if a["app_id"]==app.app_id),None)
                observations.append({k:exact.get(k) for k in ("app_id","state","tasks","created_at","stopped_at")} if exact else None)
                if exact and all_stopped([exact]):break
                if attempt<2:time.sleep(2)
            record["shutdown_observations"]=observations
            record["shutdown_verified"]=bool(exact and all_stopped([exact]))
            record["all_apps_stopped_tasks_zero"]=all_stopped(apps)
            after=cli(control_root,"billing","summary","--json")
            record.update(billing_after=after,paid_after_zero=Decimal(after["billed_cost"])==0,
                observed_workspace_gross_delta_usd=str(max(Decimal(0),gross(after)-gross(billing))),
                billing_qualification="Workspace snapshots can lag; not a settled per-app invoice; reserve retained")
        except Exception as error:record["postcheck_error_type"]=type(error).__name__
        terminal=record.get("shutdown_verified") and record.get("all_apps_stopped_tasks_zero") and record.get("paid_after_zero")
        record["state"]="terminal" if terminal else "needs-reconciliation";save(record_path,record)
        save(slot,dict(state="terminal" if terminal else "uncertain",provider="modal",experiment="P068",app_id=app.app_id))
        print(json.dumps({k:record.get(k) for k in ("app_id","state","diagnostic_complete","shutdown_verified","paid_after_zero")}),flush=True)
    return 0 if terminal and record.get("diagnostic_complete") and record.get("artifact_retrieval_complete") else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("self-test","launch"))
    parser.add_argument("--frozen-head")
    parser.add_argument("--control-root",type=Path)
    args=parser.parse_args(argv)
    if args.action=="self-test":
        require(len(PAYLOAD)==12 and len(set(PAYLOAD))==12,"Payload allowlist changed")
        require(not any("local-notes" in n or "auth" in n or n.endswith("_modal.py") for n in PAYLOAD),"Unsafe payload")
        require(len(TRACES)==9 and len(ARTIFACTS)==17,"Artifact allowlist changed")
        collector.sources();print("P068 payload/transport guards passed; no SDK or external call.");return 0
    require(args.frozen_head is not None and args.control_root is not None,"Freeze and control root required")
    return launch(args.frozen_head,args.control_root)


if __name__=="__main__":raise SystemExit(main())
