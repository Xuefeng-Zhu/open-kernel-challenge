"""Exact-source P043/P062 SM90 compile comparison, with no device or benchmark.

prepare() uses only the Python standard library. compile_payload() requires
official Triton 3.8 on Linux and its bundled NVIDIA tools; it never launches or
loads a GPU kernel. This program is not a contest submission.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "projection_pooler_lossless_codec.py": "9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837",
    "projection_pooler_lossless_raw_control.py": "8ee5189831d7dcd209284b14a1ed907140d45ad9beba8c266122dc07e93a43de",
}
KERNEL = "_pooler_codec_partial"
POINTERS = dict(X="*bf16", W="*bf16", SM="*u8", DELTA="*u8", BASE="*u8",
                ESC_IDX="*i32", ESC_EXP="*u8", P="*fp32")
CONSTANTS = dict(M=16, N=4096, K=8192, WN=4096, KP=8192,
                 S=8, BM=16, BN=64, BK=64)
OPTIONS = dict(num_warps=4, num_stages=2)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError("Frozen source hash mismatch: " + path.name)
    text = data.decode()
    nodes = [n for n in ast.walk(ast.parse(text))
             if isinstance(n, ast.FunctionDef) and n.name == KERNEL]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one owned kernel")
    n = nodes[0]
    first = min([n.lineno] + [d.lineno for d in n.decorator_list])
    body = textwrap.dedent("\n".join(text.splitlines()[first-1:n.end_lineno])) + "\n"
    return body, ast.dump(n, include_attributes=False)


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=True)
    sources = [extract(ROOT / "candidates" / name, digest) for name,digest in SOURCES.items()]
    if sources[0] != sources[1]:
        raise ValueError("Raw and compressed kernel bodies differ")
    module = "# Derived verbatim from the frozen project sources; see manifest.json.\nimport triton\nimport triton.language as tl\n\n" + sources[0][0]
    parsed = ast.parse(module)
    fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef))
    expected = list(POINTERS) + list(CONSTANTS) + ["COMPRESSED"]
    if [a.arg for a in fn.args.args] != expected:
        raise ValueError("Kernel ABI changed")
    manifest = dict(sources=SOURCES, kernel=KERNEL, body_ast_sha256=sha(sources[0][1].encode()),
                    module_sha256=sha(module.encode()), target=["cuda",90,32],
                    pointers=POINTERS, constants=CONSTANTS, options=OPTIONS,
                    alignment_bytes=16, variants=[False,True],
                    scope="Offline code generation only; no GPU execution, correctness, timing or judge-artifact identity claim.")
    (destination / "kernel_source.py").write_text(module)
    (destination / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    shutil.copyfile(__file__, destination / Path(__file__).name)
    return manifest


def compile_payload(directory):
    # No CUDA availability monkeypatch, torch import, active driver or launch.
    if platform.system() != "Linux":
        raise RuntimeError("Use the pinned Linux CPU-only container")
    if list(Path('/dev').glob('nvidia*')):
        raise RuntimeError("Diagnostic container must have no NVIDIA device nodes")
    manifest = json.loads((directory / "manifest.json").read_text())
    module_path = directory / "kernel_source.py"
    if sha(module_path.read_bytes()) != manifest["module_sha256"]:
        raise ValueError("Extracted module changed after preparation")
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    if triton.__version__ != "3.8.0":
        raise RuntimeError("Pinned Triton 3.8.0 required")
    spec = importlib.util.spec_from_file_location("p063_owned_kernel", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, KERNEL)
    # JITFunction.__init__ already defines params/arg_names. create_binder()
    # discovers the active device and must not be called in this offline path.
    tools = {k: Path(getattr(triton.knobs.nvidia,k).path) for k in ("ptxas","cuobjdump")}
    versions = {k: subprocess.run([str(p),"--version"],capture_output=True,text=True,
                                  check=True,timeout=10).stdout for k,p in tools.items()}
    tool_hashes = {k:sha(p.read_bytes()) for k,p in tools.items()}
    out = directory / "artifacts"
    out.mkdir(exist_ok=True)
    result = dict(manifest=manifest, python=platform.python_version(), host_machine=platform.machine(),
                  triton=triton.__version__, versions=versions, tool_sha256=tool_hashes,
                  gpu_execution=False, benchmark=False, variants=[])
    triton.knobs.nvidia.dump_ptxas_log = True
    for compressed in (False,True):
        label = "encoded" if compressed else "raw"
        constants = dict(CONSTANTS, COMPRESSED=compressed)
        signature = dict(POINTERS, **{k:"constexpr" for k in constants})
        attrs = {(i,): [["tt.divisibility",16]] for i in range(len(POINTERS))}
        src = ASTSource(fn=kernel, signature=signature, constexprs=constants, attrs=attrs)
        capture = io.StringIO()
        started = time.monotonic()
        with contextlib.redirect_stdout(capture):
            compiled = triton.compile(src,target=GPUTarget("cuda",90,32),options=OPTIONS)
        hashes = {}
        for ext in ("ttir","ttgir","llir","ptx","cubin"):
            value = compiled.asm[ext]
            data = value if isinstance(value,bytes) else value.encode()
            (out/f"{label}.{ext}").write_bytes(data)
            hashes[ext] = sha(data)
        (out/f"{label}.ptxas.txt").write_text(capture.getvalue())
        for flag,ext in (("--dump-resource-usage","resources.txt"),("--dump-sass","sass")):
            r = subprocess.run([str(tools["cuobjdump"]),flag,str(out/f"{label}.cubin")],
                               capture_output=True,text=True,check=True,timeout=30)
            (out/f"{label}.{ext}").write_text(r.stdout)
            hashes[ext] = sha(r.stdout.encode())
        metadata = compiled.metadata._asdict()
        kept = {k:metadata[k] for k in ("name","num_warps","num_ctas","shared","global_scratch_size",
                "global_scratch_align","tmem_size","arch","enable_fp_fusion") if k in metadata}
        result["variants"].append(dict(label=label,constants=constants,options=OPTIONS,
             metadata=kept,hashes=hashes,compile_and_disasm_wall_seconds=time.monotonic()-started,
             resource_source="Bundled ptxas log and cuobjdump static cubin metadata, not a device-loaded kernel."))
        print(json.dumps(dict(completed=label,metadata=kept)),flush=True)
    (out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",type=Path)
    parser.add_argument("--compile",type=Path)
    args=parser.parse_args()
    if bool(args.prepare)==bool(args.compile):parser.error("Choose exactly one operation")
    if args.prepare:print(json.dumps(prepare(args.prepare)))
    else:compile_payload(args.compile)
