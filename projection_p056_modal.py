"""One bounded P056 native cuBLASLt diagnostic on existing image/login.
Local imports/tests do not compile or run CUDA. Coordinator-only launch checks
credits, existing authentication, exact committed payload and an idle GPU slot.
No official check, credential creation, hardware counters or retry.
"""
import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import zlib
from projection_p044_modal import (now, require, save, resource_rate, gross,
    validate_preflight, configure_existing_auth, cli, all_stopped, MAX_RAW, MAX_PACKED, CHUNK)

ROOT = Path(__file__).resolve().parent
IMAGE = "im-cPTbGpNiX3RarKResPgX7h"
PROFILE = "xuefeng-zhu"
RATE_FLOOR = Decimal("0.001305166667")
WORKER = "projection_p056_native_screen.py"
NATIVE = "native_cublaslt_probe.py"
PARENT = "candidates/projection_pooler_column_batch2_hybrid.py"
FIXED = {
    NATIVE: "a40916423e9eb7da4b93258e7e986b55777a56ca93268e9c2e3105f437b409cc",
    PARENT: "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e",
    "projection_p046_confirmation.py": "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029",
    "official/reference.py": "e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d",
    "official/checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
}
PAYLOAD = (WORKER, NATIVE, PARENT, "projection_p046_confirmation.py", "benchmark.py",
    "harness.py", "official/reference.py", "official/checks.py", "official/muse_bf16_proj.json",
    "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Triton-MIT.txt",
    "licenses/June-9-Researcher-Reciprocity.txt")
SOURCE_NAMES = PAYLOAD[:9]
ARTIFACTS = frozenset(("summary.json", "report.json", "screen.log"))


def validate_report(report, expected_source_hashes=None):
    """Completed native evidence is separate from transport or a speedup verdict."""
    require(report.get("complete") is True and report.get("passed") is True and not report.get("failure")
            and report.get("native_build_complete") is True and report.get("native_execution") is True
            and report.get("compile", {}).get("complete") is True,
            "Worker did not complete its native diagnostic")
    reported = report.get("source_hashes", {})
    require(set(reported) == set(SOURCE_NAMES), "Source coverage mismatch")
    require(all(reported.get(name) == digest for name, digest in FIXED.items()),
            "Frozen native or parent source mismatch")
    if expected_source_hashes is not None:
        require(all(reported[name] == expected_source_hashes[name] for name in reported),
                "Sources differ from admitted payload")
    cases = report.get("cases", [])
    require([(c.get("layer"), c.get("m"), c.get("n"), c.get("k")) for c in cases]
            == [("qkv_gate", 128, 2304, 8192), ("unpooler", 16, 8192, 4096)],
            "Native case coverage mismatch")
    count = 0
    selected_count = 0
    labels = ("A", "B", "A-return", "column_probes", "zero", "cancel")
    for case in cases:
        plans = case.get("selected", [])
        require(case.get("passed") is True and case.get("same_address_aba") is True
                and case.get("native_private_workspaces") is True and 1 <= len(plans) <= 2,
                "Native case correctness or plan selection incomplete")
        selected_count += len(plans)
        for plan in plans:
            require((plan.get("m"), plan.get("n"), plan.get("k"))
                    == (case["m"], case["n"], case["k"])
                    and plan.get("split_k") in (0, 1) and plan.get("reduction_scheme") == 0
                    and plan.get("compute") == "CUBLAS_COMPUTE_32F"
                    and plan.get("input_output") == "CUDA_R_16BF"
                    and 0 <= plan.get("workspace_required_bytes", -1) <= 64 * 1024 * 1024,
                    "Native plan outside recorded shape/precision/workspace scope")
        runners = ("p046",) + tuple("native" + str(i) for i in range(len(plans)))
        checks = case.get("correctness", [])
        require([(c.get("phase"), c.get("label"), c.get("runner"), c.get("passed")) for c in checks]
                == [(phase, label, runner, True) for phase in ("before", "after")
                    for label in labels for runner in runners], "Adversarial/ABA checks incomplete")
        count += len(checks)
        independent = case.get("independent_graphs", {})
        require(independent.get("complete") is True and independent.get("separate_streams") is True
                and independent.get("distinct_mutable_buffers") is True,
                "Independent graph validation incomplete")
        require([(r.get("repeat"), r.get("runner"), r.get("passed")) for r in independent.get("checks", [])]
                == [(i, runner, True) for i in range(3) for runner in ("original", "independent")],
                "Independent graph checks incomplete")
        count += 6
        mappings = {"anchor_native0": {"parent": "p046", "candidate": "native0"}}
        if len(plans) == 2:
            mappings.update(anchor_native1={"parent": "p046", "candidate": "native1"},
                            native_control={"parent": "native0", "candidate": "native1"})
        series = case.get("series", {})
        require(set(series) == set(mappings), "Native timing comparison coverage mismatch")
        for name, mapping in mappings.items():
            value = series[name]
            require(value.get("passed") is True and value.get("runner_mapping") == mapping,
                    "Native timing mapping incomplete")
            for key, length, warmup in (("warmup", 6, True), ("samples", 24, False)):
                rows = value.get(key, [])
                require(len(rows) == length, "Native timing sample count mismatch")
                for i, row in enumerate(rows):
                    order = ["parent", "candidate"] if i % 2 == 0 else ["candidate", "parent"]
                    require(row.get("round") == i and row.get("order") == order
                            and row.get("excluded_warmup") is warmup
                            and row.get("correctness") == {"parent": True, "candidate": True},
                            "Native timing order or correctness incomplete")
                    times = row.get("us", {})
                    require(set(times) == {"parent", "candidate"} and all(
                        isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0
                        for t in times.values()), "Native raw timing must be finite and positive")
                    count += 2
    require(report.get("selected_configurations") == selected_count <= 4,
            "Native configuration count mismatch")
    require(report.get("oracle_checks") == count, "Native oracle count mismatch")
    return True


def reserve(rate=RATE_FLOOR):
    # 45s startup + 300s function hard bound + 30s teardown/resolution margin.
    return Decimal(375) * max(rate, RATE_FLOOR) * Decimal("1.1") + Decimal(".02")


def budget_guard(preflight, billing, rates):
    require(Decimal(billing["billed_cost"]) == 0, "Actual paid usage must remain zero")
    rate = max(RATE_FLOOR, resource_rate(rates))
    cost_reserve = reserve(rate)
    free_floor = Decimal(preflight["verified_credit_balance_usd"]) - max(
        Decimal(0), gross(billing) - Decimal(preflight["total_usage_usd"]))
    require(free_floor > cost_reserve, "Verified free credits cannot cover conservative reserve")
    return rate, cost_reserve, free_floor


def frozen_payload(head, root=ROOT):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", head)), "Exact reviewed commit required")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                    text=True).strip() == head, "HEAD differs from freeze")
    hashes = {}
    for name in (*PAYLOAD, "projection_p056_modal.py", "projection_p044_modal.py"):
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Payload missing or symlink: " + name)
        raw = path.read_bytes()
        require(raw == subprocess.check_output(["git", "show", head + ":" + name], cwd=root),
                "Uncommitted or changed payload: " + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    require(all(hashes[name] == digest for name, digest in FIXED.items()), "Frozen native or parent source mismatch")
    return {name: root / name for name in PAYLOAD}, hashes


def unpack_artifacts(manifest, chunks, dest, expected_source_hashes=None):
    require(not dest.exists() and not dest.is_symlink(), "Artifact destination exists; reconcile instead of replacing")
    require(0 < manifest["chunks"] <= MAX_PACKED // CHUNK and
            len(chunks) == manifest["chunks"], "Invalid chunk count")
    require(all(isinstance(c, bytes) and 0 < len(c) <= CHUNK for c in chunks), "Invalid chunk size")
    packed = b"".join(chunks)
    require(len(packed) == manifest["packed_bytes"] <= MAX_PACKED, "Packed size mismatch")
    require(hashlib.sha256(packed).hexdigest() == manifest["sha256"], "Packed hash mismatch")
    maximum_encoded = 12 * 1024 * 1024
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, maximum_encoded + 1)
    require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
            "Invalid or oversized compressed stream")
    require(len(raw) == manifest["encoded_bytes"] <= maximum_encoded, "Encoded size mismatch")
    content = json.loads(raw)
    require(isinstance(content, dict) and set(content) == set(manifest["artifact_sha256"]),
            "Artifact manifest mismatch")
    require("summary.json" in content and set(content) <= ARTIFACTS, "Unsafe artifact path")
    decoded = {n: base64.b64decode(v, validate=True) for n, v in content.items()}
    require(sum(map(len, decoded.values())) == manifest["artifact_bytes"] <= MAX_RAW, "Raw size mismatch")
    require(all(hashlib.sha256(v).hexdigest() == manifest["artifact_sha256"][n]
                for n, v in decoded.items()), "Artifact hash mismatch")
    summary = json.loads(decoded["summary.json"])
    # Preserve safe, hash-bound raw bytes even if compilation/correctness failed.
    # A semantic validation error never becomes a successful diagnostic.
    dest.mkdir(parents=True, exist_ok=False)
    for name, value in decoded.items():
        (dest / name).write_bytes(value)
    if summary.get("complete"):
        try:
            require([(s.get("mode"), s.get("exit_code")) for s in summary.get("stages", [])]
                    == [("screen", 0)], "Unexpected stage history")
            require("report.json" in decoded, "Completed native report missing")
            validate_report(json.loads(decoded["report.json"]), expected_source_hashes)
        except (ValueError, TypeError, KeyError) as error:
            summary = dict(summary, complete=False, receipt_validation_error=str(error))

    return summary


def make_remote_entry():
    # No control-root, credentials, SDK client, config, or local state closure.
    def entry():
        import base64, hashlib, json, os, signal, subprocess, sys, time, zlib
        from pathlib import Path
        root = Path("/opt/projection-p056-native")
        dest = Path("/tmp/projection-p056-native")
        dest.mkdir(exist_ok=False)
        started = time.monotonic()
        stage_start = time.monotonic()
        command = [sys.executable, "-B", str(root / "projection_p056_native_screen.py"),
                   "--compile-native", "--output", str(dest / "report.json"),
                   "--deadline-seconds", "240", "--outer-timeout-seconds", "270"]
        # This fixed child is the sole supervised command. Native compilation
        # happens inside its explicitly reviewed load_inline call.
        with (dest / "screen.log").open("wb") as log:
            child = subprocess.Popen(command, cwd=root, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                child.communicate(timeout=270)
                code = child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=2)
                code = 124
        log_path = dest / "screen.log"
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 16000))
            tail = log.read(16000)
        log_path.write_bytes(tail)
        stages = [dict(mode="screen", exit_code=code, wall_seconds=time.monotonic() - stage_start)]
        (dest / "summary.json").write_text(json.dumps(dict(stages=stages,
            complete=code == 0, worker_wall_seconds=time.monotonic() - started)))
        files = sorted(dest.glob("*"))
        # A killed atomic report save can leave this one incomplete temp file.
        # Preserve the last committed report/log, never transfer temp bytes.
        # Symlinks and every other unexpected entry still fail the guard below.
        files = [path for path in files if not (path.name == "report.json.tmp"
                 and path.is_file() and not path.is_symlink())]
        # Independent remote-side allowlist before any artifact is returned.
        def allowed(name):
            return name in {"summary.json", "report.json", "screen.log"}
        if any(not p.is_file() or p.is_symlink() or not allowed(p.name) for p in files):
            yield {"kind": "failure", "reason": "artifact_path_guard"}
            return
        raw_bytes = sum(p.stat().st_size for p in files)
        if raw_bytes > 8 * 1024 * 1024:
            yield {"kind": "failure", "reason": "artifact_raw_size_guard"}
            return
        encoded = json.dumps({p.name: base64.b64encode(p.read_bytes()).decode() for p in files},
                             sort_keys=True).encode()
        packed = zlib.compress(encoded, 9)
        if len(packed) > 2 * 1024 * 1024:
            yield {"kind": "failure", "reason": "artifact_compressed_size_guard"}
            return
        count = (len(packed) + 16383) // 16384
        yield {"kind": "manifest", "chunks": count, "packed_bytes": len(packed),
               "encoded_bytes": len(encoded), "artifact_bytes": raw_bytes,
               "sha256": hashlib.sha256(packed).hexdigest(),
               "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        for i in range(count):
            yield {"kind": "chunk", "index": i, "data": packed[i * 16384:(i + 1) * 16384]}
        yield {"kind": "complete", "chunks": count}
    return entry



def launch(head, control_root):
    control_root = control_root.resolve()
    record_path = ROOT / "experiments/projection-p056-native-modal.json"
    artifacts = ROOT / "local-notes/projection-p056-native-artifacts"
    require(not record_path.exists() and not record_path.is_symlink()
            and not artifacts.exists() and not artifacts.is_symlink(),
            "Existing diagnostic state; reconcile, never relaunch")
    subprocess.run(["git", "check-ignore", "-q", "local-notes/projection-p056-native-artifacts/summary.json"],
                   cwd=ROOT, capture_output=True, check=True)
    files, hashes = frozen_payload(head)
    auth, expiry = configure_existing_auth(control_root)
    pre_path = control_root / "local-notes/projection-p056-native-preflight.local.json"
    require(pre_path.is_file() and not pre_path.is_symlink(), "Missing local preflight")
    pre = json.loads(pre_path.read_text())
    validate_preflight(pre)
    slot = control_root / "local-notes/gpu-slot.local.json"
    prior_slot = json.loads(slot.read_text())
    require(prior_slot["state"] in ("idle", "terminal"), "Existing GPU reservation needs reconciliation")
    require(not prior_slot.get("active_reservation"), "Existing active reservation")
    billing = cli(control_root, "billing", "summary", "--json")
    require(Decimal(billing["billed_cost"]) == 0, "Actual paid usage must remain zero")
    require(all_stopped(cli(control_root, "app", "list", "--env", "main", "--json")), "An app is still active")
    rates = cli(control_root, "billing", "rates", "--json")
    rate, cost_reserve, free_floor = budget_guard(pre, billing, rates)
    import warnings
    import modal
    from modal.exception import ServerWarning
    warnings.filterwarnings("ignore", message=r"Your Modal token .* expires", category=ServerWarning)
    require(Path(modal.config.user_config_path).resolve() == auth.resolve(), "SDK uses unexpected config")
    image = modal.Image.from_id(IMAGE)
    for name, path in files.items():
        image = image.add_local_file(path, "/opt/projection-p056-native/" + name, copy=False)
    app = modal.App("okc-projection-p056-native-diagnostic", include_source=False)
    fn = app.function(image=image, gpu="H200", cpu=(2, 2), memory=(8192, 8192),
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        timeout=300, startup_timeout=45, retries=None, serialized=True,
        include_source=False, block_network=True, restrict_modal_access=True)(make_remote_entry())
    record = dict(experiment="P056 native cuBLASLt diagnostic", state="reserved", app_id=None,
        created_utc=now(), source_commit=head, worker_sha256=hashes[WORKER], native_source_sha256=hashes[NATIVE], parent_source_sha256=hashes[PARENT], base_image=IMAGE,
        payload_sha256={n: hashes[n] for n in PAYLOAD}, controller_sha256=hashes["projection_p056_modal.py"],
        cost_reserve_usd=str(cost_reserve), verified_free_credit_floor_usd=str(free_floor),
        resource_rate_usd_per_second=str(rate), paid_limit_usd="0", billing_before=billing,
        authentication_expiry_utc=expiry, preflight_sha256=hashlib.sha256(pre_path.read_bytes()).hexdigest(),
        shared_controller_sha256=hashes["projection_p044_modal.py"],
        resources=dict(gpu="H200", cpu=[2, 2], memory_mib=[8192, 8192], max_containers=1,
            min_containers=0, buffer_containers=0, scaledown_window=2, function_timeout=300,
            startup_timeout=45, child_internal_timeout=240, child_hard_timeout=270,
            app_watchdog=345, stop_allowance=30, retry_policy=None, explicit_invocations=1),
        compile_only_requested=False, performance_measurement_requested=True,
        no_hardware_counters=True, not_official_score=True)
    # The exclusive record is the durable no-duplicate reservation. If interrupted
    # between its write and the slot update, it still prevents another launch.
    with record_path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    save(slot, dict(state="reserved", provider="modal", experiment="P056 native", app_id=None))
    started = time.monotonic()
    stop_lock = threading.Lock()
    stop_result = {}
    expired = threading.Event()

    def stop():
        with stop_lock:
            if not app.app_id or stop_result:
                return
            stop_result.update(app_id=app.app_id, requested_utc=now())
            try:
                stop_result.update(cli(control_root, "app", "stop", app.app_id, "--yes", "--env", "main"))
            except Exception as error:
                stop_result["error_type"] = type(error).__name__

    def watchdog():
        expired.set()
        stop()

    timer = threading.Timer(345, watchdog)
    timer.daemon = True
    try:
        timer.start()
        with app.run(detach=False, environment_name="main"):
            record.update(state="accepted", app_id=app.app_id, accepted_utc=now())
            save(record_path, record)
            save(slot, dict(state="accepted", provider="modal", experiment="P056 native", app_id=app.app_id))
            print("Accepted P056 native app " + app.app_id, flush=True)
            require(not expired.is_set(), "App watchdog elapsed before invocation")
            manifest, chunks, complete = None, [], False
            for item in fn.remote_gen():
                require(not complete, "Unexpected data after complete marker")
                if item["kind"] == "manifest":
                    require(manifest is None and 0 < item["packed_bytes"] <= MAX_PACKED,
                            "Invalid or repeated manifest")
                    manifest = item
                    record["transport_manifest"] = manifest
                    save(record_path, record)
                elif item["kind"] == "chunk":
                    require(manifest is not None and item["index"] == len(chunks), "Chunk ordering mismatch")
                    require(isinstance(item["data"], bytes) and 0 < len(item["data"]) <= CHUNK, "Chunk size invalid")
                    chunks.append(item["data"])
                    require(sum(map(len, chunks)) <= MAX_PACKED, "Packed-size guard")
                elif item["kind"] == "complete":
                    require(manifest is not None and item["chunks"] == len(chunks) == manifest["chunks"],
                            "Incomplete stream")
                    complete = True
                else:
                    raise RuntimeError("Remote worker/artifact guard failed")
            require(complete and manifest is not None, "Incomplete result; reconcile same app")
            summary = unpack_artifacts(manifest, chunks, artifacts, hashes)
            record.update(state="function-terminal", artifact_retrieval_complete=True,
                          diagnostic_complete=summary["complete"], stages=summary["stages"],
                          worker_wall_seconds=summary["worker_wall_seconds"],
                          receipt_validation_error=summary.get("receipt_validation_error"))
            save(record_path, record)
            stop()
    except Exception as error:
        record.update(state="needs-reconciliation", app_id=app.app_id, error_type=type(error).__name__)
        print("P056 native needs reconciliation of the existing request; no relaunch.", flush=True)
    finally:
        stop()
        timer.cancel()
        record.update(explicit_stop=stop_result, client_finished_utc=now(),
                      client_wall_seconds=time.monotonic() - started)
        try:
            apps = cli(control_root, "app", "list", "--env", "main", "--json")
            exact = next((a for a in apps if a["app_id"] == app.app_id), None)
            record["shutdown_observation"] = ({k: exact.get(k) for k in
                ("app_id", "state", "tasks", "created_at", "stopped_at")} if exact else None)
            record["shutdown_verified"] = bool(exact and all_stopped([exact]))
            record["all_apps_stopped_tasks_zero"] = all_stopped(apps)
            after = cli(control_root, "billing", "summary", "--json")
            record["billing_after"] = after
            record["paid_after_zero"] = Decimal(after["billed_cost"]) == 0
            record["observed_workspace_gross_delta_usd"] = str(max(Decimal(0), gross(after) - gross(billing)))
            record["billing_qualification"] = "Workspace snapshot may lag; not a settled per-app invoice; reserve retained"
        except Exception as error:
            record["postcheck_error_type"] = type(error).__name__
        terminal = record.get("shutdown_verified") and record.get("all_apps_stopped_tasks_zero") and record.get("paid_after_zero")
        record["state"] = "terminal" if terminal else "needs-reconciliation"
        save(record_path, record)
        save(slot, dict(state="terminal" if terminal else "uncertain", provider="modal", experiment="P056 native", app_id=app.app_id))
        print(json.dumps({k: record.get(k) for k in ("app_id", "state", "diagnostic_complete", "shutdown_verified", "paid_after_zero")}), flush=True)
    return 0 if terminal and record.get("diagnostic_complete") and record.get("artifact_retrieval_complete") else 1



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "launch"))
    parser.add_argument("--frozen-head")
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "self-test":
        require(len(PAYLOAD) == 13 and len(set(PAYLOAD)) == 13, "Payload allowlist changed")
        require(all("local-notes" not in p and "auth" not in p for p in PAYLOAD), "Unsafe payload")
        require(all((ROOT / name).is_file() and not (ROOT / name).is_symlink()
                    for name in PAYLOAD), "Missing or symlink payload")
        require(len(ARTIFACTS) == 3, "Artifact allowlist changed")
        print("4 local guards passed; no Modal SDK import or external call.")
        return 0
    require(args.frozen_head is not None, "Launch requires reviewed committed freeze")
    return launch(args.frozen_head, args.control_root)


if __name__ == "__main__":
    raise SystemExit(main())
