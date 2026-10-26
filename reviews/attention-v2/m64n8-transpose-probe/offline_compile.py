"""One owned SM90 Gluon tile: preparation or GPU-free offline compilation."""
from __future__ import annotations
import argparse, ast, contextlib, hashlib, importlib.util, io, json, platform
from pathlib import Path
import re, shutil, signal, subprocess, sys, time

HERE=Path(__file__).resolve().parent
KERNEL_SHA='118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033'
SELECTED_SHA='3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
POINTERS=dict(Q='*bf16',K='*fp8e4nv',V='*fp8e4nv',KScale='*fp32',VScale='*fp32',Out='*fp32',RowMax='*fp32',RowSum='*fp32')
CONSTANTS=dict(QK_FACTOR=5.4730064863875,SM_SCALE=0.125)
OPTIONS=dict(num_warps=4,num_stages=3)
API_SHA={
 'triton/experimental/gluon/_runtime.py':'739a9c208c3c73ff5d4ec2d38a4051c20140e4f149c442a05cc37010443d5684',
 'triton/experimental/gluon/language/_layouts.py':'817880af6cd4814150a7312b5fc5484da75c1e876013a95afdc2e045a3d225bb',
 'triton/experimental/gluon/language/nvidia/hopper/__init__.py':'c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc'}
DEPENDENCY=dict(filename='triton-3.8.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl',sha256='a9c404c69ed4a39e8ec632eaf6b9fe058a060bf98979c177f6ef666f06bb8d50',size=226474486)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def bounded_write(path,value):
 raw=value if isinstance(value,bytes) else value.encode()
 if len(raw)>8*1024*1024:raise ValueError('Artifact exceeds8MiB; no truncation')
 path.write_bytes(raw);return sha(raw)

def source_contract(raw):
 if sha(raw)!=KERNEL_SHA:raise ValueError('Owned kernel hash mismatch')
 tree=ast.parse(raw);funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
 if len(funcs)!=1 or funcs[0].name!='m64n8_history_tile':raise ValueError('One owned kernel required')
 if [a.arg for a in funcs[0].args.args]!=list(POINTERS)+list(CONSTANTS):raise ValueError('Kernel ABI changed')
 for n in tree.body:
  if not isinstance(n,(ast.ImportFrom,ast.FunctionDef,ast.Expr)):raise ValueError('Unexpected module content')
  if isinstance(n,ast.Expr) and not(isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):raise ValueError('No host call in kernel module')
 return sha(ast.dump(funcs[0],include_attributes=False).encode())

def prepare(dest):
 raw=(HERE/'kernel.py').read_bytes();body=source_contract(raw)
 if sha((HERE.parent/'fusion-lookahead-probe/candidate.py').read_bytes())!=SELECTED_SHA:raise ValueError('Selected source binding mismatch')
 dest.mkdir(parents=True,exist_ok=True);(dest/'kernel.py').write_bytes(raw)
 shutil.copyfile(__file__,dest/Path(__file__).name)
 m=dict(kernel_sha256=KERNEL_SHA,helper_sha256=sha(Path(__file__).read_bytes()),body_ast_sha256=body,selected_sha256=SELECTED_SHA,kernel='m64n8_history_tile',pointers=POINTERS,constants=CONSTANTS,options=OPTIONS,target=['cuda',90,32],alignment_bytes=16,dependency=DEPENDENCY,API_source_sha256=API_SHA,scope='One full historical tile only; no producer/tail/online carry/merge/GPU loading/execution/numerical/performance or judge-binary claim')
 save(dest/'manifest.json',m);return m

def instruction_summary(ptx,sass):
 return dict(wgmma_m64n8k16_sites=len(re.findall(r'wgmma\.mma_async[^;\n]*\.m64n8k16\.',ptx)),mma_sync_sites=len(re.findall(r'\bmma\.sync\.',ptx)),local_load_sites=len(re.findall(r'\bld\.local\.',ptx)),local_store_sites=len(re.findall(r'\bst\.local\.',ptx)),sass_hgmma_sites=len(re.findall(r'\bHGMMA\.',sass)),qualification='Static sites only; verify both QK/PV dataflow separately, not executed counts or performance')

def compile_payload(directory):
 if platform.system()!='Linux':raise RuntimeError('Pinned Linux CPU-only container required')
 if list(Path('/dev').glob('nvidia*')):raise RuntimeError('NVIDIA devices must be absent')
 m=json.loads((directory/'manifest.json').read_text())
 if m['kernel_sha256']!=KERNEL_SHA or m['pointers']!=POINTERS or m['constants']!=CONSTANTS or m['options']!=OPTIONS or m['target']!=['cuda',90,32]:raise ValueError('Frozen specialization changed')
 if sha(Path(__file__).read_bytes())!=m['helper_sha256']:raise ValueError('Offline helper changed')
 source_contract((directory/'kernel.py').read_bytes())
 import triton
 from triton.backends.compiler import GPUTarget
 from triton.experimental.gluon._runtime import GluonASTSource
 if triton.__version__!='3.8.0':raise ValueError('Pinned Triton3.8.0 required')
 package=Path(triton.__file__).resolve().parent
 for relative,expected in API_SHA.items():
  if sha((package.parent/relative).read_bytes())!=expected:raise ValueError('Compiler API source differs:'+relative)
 spec=importlib.util.spec_from_file_location('attention_m64n8_owned_tile',directory/'kernel.py')
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 # Construct directly. A runtime binder discovers a device and is forbidden.
 signature=dict(POINTERS,**{k:'constexpr' for k in CONSTANTS})
 attrs={(i,):[['tt.divisibility',16]] for i in range(len(POINTERS))}
 src=GluonASTSource(mod.m64n8_history_tile,signature,CONSTANTS,attrs)
 tools={k:Path(getattr(triton.knobs.nvidia,k).path) for k in ('ptxas','cuobjdump')}
 versions={k:subprocess.run([str(p),'--version'],capture_output=True,text=True,check=True,timeout=10).stdout for k,p in tools.items()}
 tool_hashes={k:sha(p.read_bytes()) for k,p in tools.items()}
 out=directory/'artifacts';out.mkdir(exist_ok=True);capture=io.StringIO();started=time.monotonic()
 triton.knobs.nvidia.dump_ptxas_log=True
 try:
  with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):compiled=triton.compile(src,target=GPUTarget('cuda',90,32),options=OPTIONS)
 except Exception as exc:
  bounded_write(out/'compile.log',capture.getvalue())
  save(out/'failure.json',dict(status='compile_failed',reason_type=type(exc).__name__,reason=str(exc)[:12000],kernel_sha256=KERNEL_SHA,gpu_execution=False,variant_retries=0));raise
 hashes={}
 for ext in ('ttgir','llir','ptx','cubin'):
  if ext not in compiled.asm:raise ValueError('Required compiler artifact absent:'+ext)
  hashes[ext]=bounded_write(out/f'tile.{ext}',compiled.asm[ext])
 if 'ttir' in compiled.asm:hashes['ttir']=bounded_write(out/'tile.ttir',compiled.asm['ttir'])
 hashes['ptxas.txt']=bounded_write(out/'tile.ptxas.txt',capture.getvalue())
 for flag,ext in [('--dump-resource-usage','resources.txt'),('--dump-sass','sass')]:
  r=subprocess.run([str(tools['cuobjdump']),flag,str(out/'tile.cubin')],capture_output=True,text=True,check=True,timeout=30)
  hashes[ext]=bounded_write(out/f'tile.{ext}',r.stdout)
 meta=compiled.metadata._asdict();kept={k:meta[k] for k in ('name','num_warps','num_ctas','shared','global_scratch_size','global_scratch_align','tmem_size','arch','enable_fp_fusion') if k in meta}
 result=dict(status='offline_compile_complete_pending_dataflow_review',manifest=m,python=platform.python_version(),host_machine=platform.machine(),triton=triton.__version__,compiler_API_source_sha256=API_SHA,tool_sha256=tool_hashes,tool_versions=versions,metadata=kept,artifact_sha256=hashes,instruction_inventory=instruction_summary((out/'tile.ptx').read_text(),(out/'tile.sass').read_text()),compile_and_disasm_wall_seconds=time.monotonic()-started,gpu_execution=False,numerical_correctness=False,performance=False,resource_gate='Both native64x8+strict precision+0spill/local; registers/shared recorded, not a whole-producer admission')
 save(out/'manifest.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',type=Path);g.add_argument('--compile',type=Path);a=p.parse_args()
 def alarm(signum,frame):raise TimeoutError('240second offline compile deadline')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(240)
 try:print(json.dumps(prepare(a.prepare) if a.prepare else compile_payload(a.compile)))
 finally:signal.alarm(0)

if __name__=='__main__':main()
