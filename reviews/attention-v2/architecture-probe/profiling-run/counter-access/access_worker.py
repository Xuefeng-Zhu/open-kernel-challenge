"""Final single-metric NCU access triage. Parent stays stdlib-only/no CUDA.

Exactly one NCU subprocess, one FA3 target, no activity session or query. Child
uses the already-reviewed immutable target; timeout logs are retained on disk.
"""
import argparse
import base64
import csv
import io
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

NCU='/usr/local/cuda-13.0/bin/ncu'
NCU_SHA='18d8a8aa13c156cd43ebdf24ac950b5386cb229f2a4c931db47ef91cfe785920'
METRIC='dram__bytes_read.sum'
MAX_LOG=64*1024
MAX_REPORT=2*1024*1024
MAX_SUMMARY=512*1024
FROZEN={'worker.py':'07c6c36b06995df46b49b90208b0454eba1aa3efb17787c7a2fbdf67b1efa326',
        'counter_target.py':'910e0b3613681edf318d43df25356d8c9ed286b416c2757063d4c0d431142c3f'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(root,folder):
    return [NCU,'--config-file','0','--replay-mode','application','--app-replay-match','grid',
        '--app-replay-mode','strict','--nvtx','--nvtx-include','PROFILE_ONE/',
        '--graph-profiling','node','--kernel-name-base','demangled','--kernel-name',
        'regex:.*FlashAttnFwdSm90.*','--launch-count','1','--cache-control','none',
        '--clock-control','none','--metrics',METRIC,'--csv','--page','raw','-o',str(folder/'fa3'),
        sys.executable,str(root/'counter_target.py'),'--arm','fa3','--output',str(folder/'fa3-counter-check.json')]


def bounded_log(path):
    size=path.stat().st_size
    with path.open('rb') as f:
        if size<=MAX_LOG:raw=f.read()
        else:
            first=f.read(MAX_LOG//2);f.seek(-MAX_LOG//2,2);raw=first+f.read()
    text=raw.decode(errors='replace')
    return dict(bytes=size,preserved_bytes=len(raw),truncated=size>MAX_LOG,
        preserved_sha256=hashlib.sha256(raw).hexdigest(),preserved_base64=base64.b64encode(raw).decode(),text=text)


def run(dest):
    root=Path(__file__).resolve().parent;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();report=dict(kind='One remaining FA3 hardware metric access triage',
        status='starting',passed=False,started_utc=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(__file__),source_sha256=FROZEN,metric=METRIC,arm='fa3',
        activity_sessions=0,hardware_counter_sessions=0,prior_counter_sessions=1,
        overall_counter_sessions_upper_bound=2,benchmark_claim=False,exact_cta_trace=False,
        parent_cuda_context=False,permission_denied=False)
    def save():
        report['worker_wall_seconds']=time.monotonic()-started
        raw=json.dumps(report,separators=(',',':'))+'\n'
        assert len(raw.encode())<=MAX_SUMMARY
        p=dest/'summary.tmp';p.write_text(raw);p.replace(dest/'summary.json')
    def alarm(*_):raise TimeoutError('60s access triage alarm')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(60);save()
    try:
        for name,expected in FROZEN.items():assert sha(root/name)==expected,name
        assert Path(NCU).is_file() and sha(NCU)==NCU_SHA,'Existing NCU launcher mismatch'
        with tempfile.TemporaryDirectory(prefix='ncu-access-') as scratch:
            folder=Path(scratch);argv=command(root,folder)
            report.update(command=argv,hardware_counter_sessions=1,timeout_seconds=55)
            save()
            code=125
            try:
                with (folder/'stdout.log').open('wb') as out,(folder/'stderr.log').open('wb') as err:
                    child=subprocess.Popen(argv,cwd=root,stdout=out,stderr=err,start_new_session=True)
                    try:
                        code=child.wait(timeout=55)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2);code=124
                        report['timed_out']=True
                    except BaseException as exc:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
                        report['wait_error_type']=type(exc).__name__
                        report['timed_out']=isinstance(exc,TimeoutError)
            finally:
                # Handles are flushed/closed before reading, including global
                # alarm/exception paths. Logs survive TemporaryDirectory cleanup.
                signal.alarm(0)
                for stream in ('stdout','stderr'):
                    path=folder/(stream+'.log')
                    if path.is_file():report[stream]=bounded_log(path)
                report['exit_code']=code
                save()
                logs=(report['stdout']['text']+'\n'+report['stderr']['text']).lower()
                report['permission_denied']=any(s in logs for s in ['err_nvgpuctrperm','permission to access nvidia gpu performance counters','insufficient privileges','cupti_error_insufficient_privileges','access to the gpu performance counters is not permitted'])
                check=folder/'fa3-counter-check.json'
                if check.is_file() and not check.is_symlink() and check.stat().st_size<=64*1024:
                    raw=check.read_bytes();(dest/check.name).write_bytes(raw);report['validation']=json.loads(raw)
                else:report['validation']=None
                binary=folder/'fa3.ncu-rep'
                if binary.is_file() and not binary.is_symlink():
                    report['raw_report_bytes']=binary.stat().st_size
                    if binary.stat().st_size<=MAX_REPORT:
                        raw=binary.read_bytes();(dest/binary.name).write_bytes(raw);report.update(raw_report_preserved=True,raw_report_sha256=hashlib.sha256(raw).hexdigest())
                    else:report.update(raw_report_preserved=False,report_size_guard=True)
                report['metric_rows']=[]
                rows=list(csv.reader(io.StringIO(report['stdout']['text'])))
                for index,row in enumerate(rows):
                    if 'Metric Name' in row and 'Metric Value' in row:
                        ni,vi=row.index('Metric Name'),row.index('Metric Value')
                        ui=row.index('Metric Unit') if 'Metric Unit' in row else None
                        for values in rows[index+1:]:
                            if len(values)>max(ni,vi) and values[ni]==METRIC:
                                report['metric_rows'].append(dict(name=METRIC,value=values[vi],unit=values[ui] if ui is not None and len(values)>ui else None))
                        break
                if report['permission_denied']:report['status']='counter_permission_denied_stop'
                elif report.get('timed_out'):report['status']='timeout_partial_logs_preserved'
                elif code or not report.get('validation',{} ) or not report['validation'].get('passed'):
                    report['status']='failed_or_incomplete_logs_preserved'
                elif not report.get('raw_report_preserved'):report['status']='no_bounded_raw_report'
                elif not report['metric_rows']:report['status']='raw_report_collected_metric_unparsed'
                else:report.update(status='counter_access_collected',passed=True)
                report['causal_confidence']='Access diagnostic only; no two-arm comparison or bottleneck inference.'
    except Exception as exc:report.update(status='failed_before_or_during_session',error_type=type(exc).__name__,error=str(exc)[:500])
    finally:
        signal.alarm(0);save()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    raise SystemExit(run(p.parse_args().output_dir))
