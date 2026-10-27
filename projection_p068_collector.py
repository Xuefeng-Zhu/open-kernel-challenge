"""P068: fixed P046 diagnostic collector. Default is an offline dry-run plan.

No Modal/OKC client, credentials, package install, candidate edits or NCU launch.
The run command is for an already authorized ephemeral H200 container. Its owner
must separately enforce app lifetime/cost and verify stopped/tasks0 afterwards.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SOURCE = "candidates/projection_pooler_column_batch2_hybrid.py"
FROZEN = {
    SOURCE: "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e",
    "projection_p046_confirmation.py": "6c57daf6ab5aca4a73aa1a608b6ae978af284a02d41c489b4e7b62b307505029",
    "benchmark.py": "5d7f2db870a1e82142be4b5f91e319c944673b31bdb66bacf02159d2032dde78",
    "harness.py": "d46ea816a0091cea8b6e429a6e8facfbe0a82c20488a8c9b89023c1561777fd8",
    "official/reference.py": "e62491abc9ddcad33f5faba8b0ee74cf6370b7be5dd2dc05031742f3ebf7147d",
    "official/checks.py": "9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055",
    "official/muse_bf16_proj.json": "ee02afa1e848c15b16d677fc1e405617a51182fbda7ed6b21c9e5f6a7c5ef0e9",
    "LICENSE": "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747",
    # Integration refresh: combined notices only; all computational inputs remain frozen.
    "THIRD_PARTY_NOTICES.md": "77dcaee42d3b5c94a0700174ba0f6d807646f939558f7d790562209fdd189d3b",
    "licenses/June-9-Researcher-Reciprocity.txt": "9eb39e0fa3b25c043d6165a8ce5aca29501c057e6aae9e54df63b8e92c0efda5",
    "licenses/Triton-MIT.txt": "80af2216e0ceefa0c103fecad3a188a0d68c0165000bd21b3cfb17ba7e7ed517",
}
CASES = (("qkv_gate", 128), ("pooler", 128), ("pooler", 8))
WARMUP, ROUNDS, REPEATS = 6, 24, 3
INTERNAL_SECONDS, HARD_SECONDS = 90, 120
LABELS = ("A", "B", "A-return", "sparse", "zero")
PACKAGES = ("torch", "triton", "nvidia-cutlass-dsl", "apache-tvm-ffi", "quack-kernels", "einops")
LIBRARY = re.compile(r"^(libcublas(?:Lt)?|libcuda|libcudart|libnvJitLink|libtorch_cuda|libcupti)\.so(?:\.[0-9]+)*$")
ABSENT = {
    "exact_vendor_kernel_binary": "Not exposed by this collector; DSO identity is not a kernel/cubin binding",
    "vendor_tactic_and_tile_assignment": "Not exposed by Torch/CUPTI activity",
    "hardware_counters": "Not collected; previous NCU timeouts require a materially changed capability before retry",
    "achieved_occupancy_or_stalls": "Not measured; trace estimates are excluded",
    "concurrency": "Sequential replay only; independent hidden vendor workspace is not established",
    "official_score": "Diagnostic only; no calibration or official score",
}


def sha(path, gate=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            gate()
            h.update(chunk)
    return h.hexdigest()


def sources(root=ROOT):
    result = {}
    for name, expected in FROZEN.items():
        result[name] = sha(root / name)
        if result[name] != expected:
            raise ValueError("Frozen source mismatch: " + name)
    result[Path(__file__).name] = sha(root / Path(__file__).name)
    return result


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def worker_command(stage, output, image_digest=None):
    if stage not in ("timing", "activity"):
        raise ValueError("Unknown stage")
    cmd = [sys.executable, "-B", str(ROOT / Path(__file__).name), "_worker",
           "--stage", stage, "--output-dir", str(Path(output).resolve())]
    if image_digest:
        cmd += ["--image-digest", image_digest]
    return cmd


def plan(output, image_digest=None):
    if image_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("Image digest must be sha256:<64 lowercase hex digits>")
    return dict(kind="P068 dry-run, no native execution", source_hashes=sources(),
        payload_files=list(FROZEN) + [Path(__file__).name], cases=[list(c) for c in CASES],
        stages=[dict(stage=s, argv=worker_command(s, output, image_digest),
                     shell_display=shlex.join(worker_command(s, output, image_digest)),
                     internal_seconds=INTERNAL_SECONDS, hard_seconds=HARD_SECONDS) for s in ("timing", "activity")],
        image_digest=dict(value=image_digest, provenance="caller supplied, not independently verified" if image_digest else "unavailable"),
        expected_checks=dict(timing_oracles=240, activity_oracles=18,
            timing_capture_warmups=18, timing_capture_executions=6,
            activity_capture_warmups=9, activity_capture_executions=3),
        counter_requirements=["per-SM active cycles/CTA instances or supported per-CTA timeline",
            "PC-correlated barrier/membar waits", "DRAM/L2 bytes and transactions",
            "eligible/active warps and issue activity"],
        counter_execution_enabled=False, absence=ABSENT,
        cleanup="Controller kills and reaps its worker process group; external container owner must stop app and verify tasks0",
        prerequisites=["existing valid authorized H200 access", "verified free credits and net paid cap $0",
                       "one serialized ephemeral app with an external lifetime/cost watchdog"],
        no_cloud_or_credential_actions=True)


def run_bounded(cmd, log_path, timeout=HARD_SECONDS):
    """Linux/POSIX child supervision; never a shell, retry or detached child."""
    started = time.monotonic()
    env = {k: v for k, v in os.environ.items() if k in {
        "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_HOME",
        "PYTHONPATH", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_VISIBLE_DEVICES"}}
    env["P068_SUPERVISOR_PID"] = str(os.getpid())
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    timed_out, terminated = False, False
    with os.fdopen(fd, "wb") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also remove any descendant left behind after a worker exits.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                terminated = True
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=3)
    return dict(exit_code=proc.returncode, timed_out=timed_out, worker_reaped=True,
                group_termination_sent=terminated, wall_seconds=time.monotonic()-started,
                cloud_app_stop_verified=False)


def mapped_library_paths(text):
    """Allowlisted DSOs only; never export maps or arbitrary mapped file paths."""
    result = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if LIBRARY.fullmatch(path.name):
                result.add(path)
    return sorted(result)


def library_identity(gate=lambda: None):
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return dict(status="unavailable", reason="/proc/self/maps absent", libraries=[])
    libraries = []
    for path in mapped_library_paths(maps.read_text()):
        gate()
        try:
            if not path.is_file():raise OSError("Not a regular library file")
            before=path.stat();digest=sha(path,gate);after=path.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                raise OSError("Library changed during hashing")
            libraries.append(dict(basename=path.name, sha256=digest, bytes=after.st_size))
        except OSError:
            libraries.append(dict(basename=path.name, sha256=None, status="unreadable"))
    return dict(status="available" if libraries else "unavailable", libraries=libraries,
                scope="Loaded process DSOs, not a vendor kernel/cubin or tactic binding")


def environment(torch, image_digest):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    prop = torch.cuda.get_device_properties(0)
    return dict(python=platform.python_version(), platform=platform.system(), packages=packages,
        cuda_runtime=torch.version.cuda, gpu=prop.name, capability=list(torch.cuda.get_device_capability()),
        sms=prop.multi_processor_count, tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        image_digest=image_digest, image_digest_provenance="caller supplied" if image_digest else "unavailable",
        driver_version=None, driver_version_reason="May be reported by activity trace; no assumed version",
        python_executable_sha256=sha(Path(sys.executable)))


def validate_timing(record, expected_sources):
    if not (record.get("complete") and record.get("stage")=="timing" and record.get("source_hashes")==expected_sources
            and record.get("oracle_checks")==240 and not record.get("failure")):
        raise ValueError("Incomplete or foreign timing receipt")
    env=record.get("environment",{})
    required={"python","packages","cuda_runtime","gpu","capability","sms","tf32_allowed",
              "bf16_reduced_precision_reduction","python_executable_sha256"}
    if (not required.issubset(env) or env.get("capability")!=[9,0] or "H200" not in env.get("gpu","")
            or not isinstance(env.get("packages"),dict) or not env["packages"].get("torch")
            or not isinstance(env.get("sms"),int) or env["sms"]<=0
            or not re.fullmatch(r"[0-9a-f]{64}",env.get("python_executable_sha256",""))):
        raise ValueError("Missing timing environment identity")
    libs=record.get("library_identity",{})
    if libs.get("status") not in ("available","unavailable") or not isinstance(libs.get("libraries"),list):
        raise ValueError("Missing timing DSO identity/absence marker")
    for lib in libs["libraries"]:
        if not LIBRARY.fullmatch(lib.get("basename","")) or not (
                re.fullmatch(r"[0-9a-f]{64}",lib.get("sha256") or "") or lib.get("status")=="unreadable"):
            raise ValueError("Invalid DSO identity")
    if [(c.get("layer"), c.get("m")) for c in record.get("cases", [])] != list(CASES):
        raise ValueError("Timing case coverage mismatch")
    count = 0
    for case_index, case in enumerate(record["cases"]):
        if case.get("passed") is not True or case.get("source_sha256") != FROZEN[SOURCE]:
            raise ValueError("Case correctness/source binding missing")
        checks = case.get("correctness", [])
        expected = {(phase, label, role) for phase in ("before", "after") for label in LABELS for role in ("parent", "candidate")}
        if len(checks)!=20 or {(c["phase"],c["label"],c["runner"]) for c in checks}!=expected or not all(c.get("passed") for c in checks):
            raise ValueError("Missing ABA/boundary checks")
        count += len(checks)
        for key, n in (("warmup", WARMUP), ("samples", ROUNDS)):
            samples = case.get(key, [])
            if len(samples)!=n:
                raise ValueError("Missing predefined samples")
            for i, sample in enumerate(samples):
                order = ["parent", "candidate"] if i%2==0 else ["candidate", "parent"]
                expected_seed=6800000+case_index*1000+(100 if key=="warmup" else 200)+i
                if (sample["round"]!=i or sample.get("seed")!=expected_seed or sample["order"]!=order or sample["excluded_warmup"]!=(key=="warmup")
                    or sample.get("correctness")!={"parent":True,"candidate":True}
                    or set(sample["us"])!={"parent","candidate"}
                    or not all(math.isfinite(v) and v>0 for v in sample["us"].values())):
                    raise ValueError("Invalid order, timing or correctness sample")
            count += 2*n
        samples=case["samples"]
        ratios=[s["us"]["parent"]/s["us"]["candidate"] for s in samples]
        paired=statistics.median(math.sqrt(ratios[i]*ratios[i+1]) for i in range(0,ROUNDS,2))
        movement=1/paired-1
        summary=case.get("graph_full_call",{})
        if (not math.isclose(summary.get("median_paired_raw_ratio",-1),paired,rel_tol=1e-10)
                or not math.isclose(summary.get("aa_relative_movement",999),movement,abs_tol=1e-10)
                or summary.get("aa_drift_exceeds_2pct")!=(abs(movement)>.02)):
            raise ValueError("Missing or inconsistent A/A timing summary")
    if count!=240:
        raise ValueError("Oracle count mismatch")
    return record


def measure_pair(timer, graphs, eviction, eviction_out, sample, input_check, output_check, gate):
    """Shared input; both timed members finish before either FP64 oracle."""
    for role in sample["order"]:
        gate()
        sample["us"][role]=timer.measure(graphs[role],eviction,eviction_out)
        input_check()
    for role in sample["order"]:
        output_check(role)
        sample["correctness"][role]=True


def capture_checked(torch, safe, capture, kernel, data, state, weights, original, frozen, config, pointer):
    """Reject capture-time mutation before a fresh probe can overwrite it."""
    initial = {"x": data["x"].clone()}
    graph, output = capture(kernel, data, state)
    torch.cuda.synchronize()
    safe.assert_input(torch, data, initial, config, pointer)
    safe.assert_frozen(torch, weights, original)
    safe.assert_frozen(torch, state, frozen)
    return graph, output


def trace_summary(trace):
    events = []
    for event in trace.get("traceEvents", []):
        if event.get("ph")!="X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        start, duration = event.get("ts"), event.get("dur")
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start,duration)) or duration<0:
            raise ValueError("Invalid trace interval")
        args = event.get("args", {})
        events.append(dict(symbol=event.get("name"), kind=event["cat"], start_us=start, duration_us=duration,
            metadata={k:args.get(k) for k in ("grid","block","stream","registers per thread","shared memory","cluster")},
            exact_kernel_binary_sha256=None, tactic_id=None))
    events.sort(key=lambda e:e["start_us"])
    kernels=[e for e in events if e["kind"]=="kernel"]
    if not kernels:
        raise ValueError("No CUDA kernel events; activity unavailable")
    return dict(events=events, gaps_us=[b["start_us"]-a["start_us"]-a["duration_us"] for a,b in zip(kernels,kernels[1:])],
        trace_versions={k:trace.get(k) for k in ("cuda_driver_version","cuda_runtime_version","cupti_version")},
        qualification="Instrumented intervals/static launch fields only; no counter/occupancy or exact executable binding")


def native_worker(stage, output, image_digest=None):
    started = time.monotonic()
    def gate():
        if time.monotonic()-started>INTERNAL_SECONDS:
            raise TimeoutError("Internal diagnostic deadline")
    record=dict(stage=stage, complete=False, source_hashes=sources(), cases=[], oracle_checks=0,
                absence=ABSENT, diagnostic_only=True)
    def save():
        atomic_json(output/(stage+".json"),record)
    save()
    try:
        if os.environ.get("P068_SUPERVISOR_PID") != str(os.getppid()):
            raise RuntimeError("Use the bounded run controller; unsupervised worker refused")
        timing=None
        if stage=="activity":
            timing=validate_timing(json.loads((output/"timing.json").read_text()),record["source_hashes"])
            record["timing_receipt_sha256"]=sha(output/"timing.json")
        import torch
        import projection_p046_confirmation as safe
        from benchmark import capture
        from harness import load_module
        safe.require_h200(torch)
        record["environment"]=environment(torch,image_digest)
        if timing and timing["environment"]!=record["environment"]:
            raise ValueError("Timing/activity environment changed")
        module=load_module(ROOT/SOURCE,"p068_p046")
        reference=load_module(ROOT/"official/reference.py","p068_oracle")
        with torch.inference_mode():
            gate()
            weights=reference.generate_weights(reference.WEIGHTS_SEED,device="cuda")
            original=safe.freeze_tensors(torch,weights)
            oracle_weights={k:v[1] for k,v in original.items()}
            state=module.prepare(weights)
            frozen=safe.freeze_tensors(torch,state)
            safe.assert_frozen(torch,weights,original)
            eviction=torch.ones(safe.EVICTION_BYTES//4,dtype=torch.float32,device="cuda")
            eviction_out=torch.empty((),dtype=torch.float32,device="cuda")
            timer=safe.ColdGraphTimer(torch)
            for index,(layer,m) in enumerate(CASES):
                gate()
                seed=6800000+index*1000
                make=lambda s,dist="normal":safe.fresh_input(torch,reference,layer,m,s,dist)
                data=make(seed)
                config,pointer=dict(data["config"]),safe.tensor_metadata(data["x"])
                roles=("parent","candidate") if stage=="timing" else ("parent",)
                graphs,outputs={},{}
                for role in roles:
                    gate()
                    graphs[role],outputs[role]=capture_checked(torch,safe,capture,module.kernel_fn,data,state,
                        weights,original,frozen,config,pointer)
                if len({o.data_ptr() for o in outputs.values()})!=len(roles):
                    raise ValueError("Outputs alias; sequential graph isolation failed")
                item=dict(layer=layer,m=m,source_sha256=FROZEN[SOURCE],config=config,
                    x_shape=list(data["x"].shape),x_stride=list(data["x"].stride()),
                    capture_warmups_per_graph=3,capture_executions_per_graph=1,
                    capture_calls_checked_individually=False,correctness=[],warmup=[],samples=[])
                record["cases"].append(item);save()
                if stage=="timing":
                    safe.correctness_phase(torch,reference,data,graphs,outputs,oracle_weights,config,pointer,
                        make,seed+10,"before",item["correctness"],gate,save)
                    record["oracle_checks"]+=10
                    for warm,n,offset in ((True,WARMUP,100),(False,ROUNDS,200)):
                        for i in range(n):
                            gate();fresh=make(seed+offset+i);data["x"].copy_(fresh["x"])
                            order=list(safe.runner_order(i))
                            sample=dict(round=i,seed=seed+offset+i,order=order,us={},correctness={},excluded_warmup=warm)
                            item["warmup" if warm else "samples"].append(sample)
                            measure_pair(timer,graphs,eviction,eviction_out,sample,
                                lambda:safe.assert_input(torch,data,fresh,config,pointer),
                                lambda role:safe.check_output(torch,reference,data,outputs[role],fresh,config,pointer,oracle_weights),gate)
                            record["oracle_checks"]+=2
                            save()
                    safe.correctness_phase(torch,reference,data,graphs,outputs,oracle_weights,config,pointer,
                        make,seed+10,"after",item["correctness"],gate,save)
                    record["oracle_checks"]+=10
                    summary=safe.summarize_samples(item["samples"])
                    summary.pop("candidate_raw_latency_cut")
                    summary["aa_relative_movement"]=1/summary["median_paired_raw_ratio"]-1
                    summary["aa_drift_exceeds_2pct"]=abs(summary["aa_relative_movement"])>.02
                    summary["qualification"]="Two identical P046 graphs; gross whole-kernel_fn replay time, not gain or score. All tails retained; unstable tails need review."
                    item["graph_full_call"]=summary
                else:
                    item["activity"],item["eager_full_call"]=[],[]
                    for i in range(REPEATS):
                        gate();fresh=make(seed+600+i);data["x"].copy_(fresh["x"])
                        torch.sum(eviction,dim=0,out=eviction_out);torch.cuda.synchronize()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,with_stack=False,profile_memory=False) as prof:
                            graphs["parent"].replay();torch.cuda.synchronize()
                        path=output/f"{layer}.m{m}.repeat{i}.trace.local.json"
                        prof.export_chrome_trace(str(path));os.chmod(path,0o600)
                        trace=trace_summary(json.loads(path.read_text()))
                        trace.update(trace_file=path.name,sha256=sha(path),source_sha256=FROZEN[SOURCE],repeat=i)
                        item["activity"].append(trace);save()
                        safe.check_output(torch,reference,data,outputs["parent"],fresh,config,pointer,oracle_weights)
                        record["oracle_checks"]+=1
                        # Eager execution is a separate descriptive control, never judge-equivalent timing.
                        torch.sum(eviction,dim=0,out=eviction_out);torch.cuda.synchronize();tick=time.monotonic()
                        timer.start.record();out=module.kernel_fn(data,state);timer.end.record();timer.end.synchronize()
                        item["eager_full_call"].append(dict(repeat=i,cuda_event_us=timer.start.elapsed_time(timer.end)*1000,
                            synchronized_host_wall_us=(time.monotonic()-tick)*1e6,acceptance_eligible=False))
                        safe.check_output(torch,reference,data,out,fresh,config,pointer,oracle_weights)
                        record["oracle_checks"]+=1;save()
                safe.assert_frozen(torch,weights,original);safe.assert_frozen(torch,state,frozen)
                item["passed"]=True;save();del graphs,outputs
            gate();record["library_identity"]=library_identity(gate)
            if timing:
                before=timing["library_identity"]
                old={(x["basename"],x.get("sha256")) for x in before["libraries"]}
                now={(x["basename"],x.get("sha256")) for x in record["library_identity"]["libraries"]}
                if not old.issubset(now):raise ValueError("Loaded DSO identity differs between stages")
                record["library_identity_comparison"]="matched timing DSOs" if old and all(v for _,v in old) else "incomplete; absence is not equality"
            torch.cuda.synchronize()
        record["complete"]=True
        if stage=="timing":validate_timing(record,record["source_hashes"])
        elif record["oracle_checks"]!=18:raise ValueError("Incomplete activity correctness")
        record["wall_seconds"]=time.monotonic()-started;save();return 0
    except Exception as error:
        traceback.print_exc()  # Owner-only local log; never auto-published.
        record["complete"]=False
        record["failure"]={"type":type(error).__name__,"message":"Native stage failed; review owner-only worker log locally"}
        save();return 1


def run(output,image_digest=None):
    if platform.system()!="Linux":raise RuntimeError("Run requires an authorized Linux H200 container; dry-run is available locally")
    spec=plan(output,image_digest)
    output.mkdir(mode=0o700,parents=False,exist_ok=False)  # Never overwrite/duplicate a prior run.
    atomic_json(output/"plan.json",spec)
    receipt=dict(stages=[],complete=False,cloud_app_stop_verified=False,
        external_cleanup_required="Stop ephemeral app and verify stopped/tasks0 even after a controller failure")
    try:
        for stage in ("timing","activity"):
            result=run_bounded(worker_command(stage,output,image_digest),output/(stage+".log.local.txt"))
            receipt["stages"].append(dict(stage=stage,**result));atomic_json(output/"controller.json",receipt)
            if result["exit_code"]!=0 or result["timed_out"]:return 1
            actual=json.loads((output/(stage+".json")).read_text())
            if not actual.get("complete"):return 1
            if stage=="timing":validate_timing(actual,spec["source_hashes"])
        receipt["complete"]=True;return 0
    finally:
        atomic_json(output/"controller.json",receipt)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","run","_worker"),nargs="?",default="plan")
    parser.add_argument("--output-dir",type=Path,default=Path("/tmp/p068-baseline"))
    parser.add_argument("--image-digest")
    parser.add_argument("--stage",choices=("timing","activity"))
    args=parser.parse_args(argv)
    if args.action=="plan":print(json.dumps(plan(args.output_dir,args.image_digest),indent=2));return 0
    if args.action=="run":return run(args.output_dir,args.image_digest)
    if args.stage is None:parser.error("Worker stage required")
    return native_worker(args.stage,args.output_dir,args.image_digest)


if __name__=="__main__":
    raise SystemExit(main())
