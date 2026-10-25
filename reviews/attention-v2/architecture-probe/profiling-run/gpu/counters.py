"""Bounded installed-NCU collection; never install or expand permission.

Activity tracing must already have passed. Counter failures remain observation
limits, not workload bottleneck findings. No exact CTA tracing is claimed.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

NCU='/usr/local/cuda-13.0/bin/ncu'
MAX_TOTAL=8*1024*1024
SUMMARY_RESERVE=256*1024
MAX_LOG=16*1024
METRIC_GROUPS=[
 ('duration',['gpu__time_duration.sum']),
 ('dram_read_bytes',['dram__bytes_read.sum']),
 ('dram_write_bytes',['dram__bytes_write.sum']),
 ('tensor_active',['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']),
 ('warps_active',['sm__warps_active.avg.pct_of_peak_sustained_active']),
 ('ctas_active',['sm__ctas_active.avg.per_cycle_active','sm__ctas_active.avg']),
 ('long_scoreboard',['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']),
 ('barrier',['smsp__warp_issue_stalled_barrier_per_warp_active.pct'])]
REQUIRED_FLAGS=['--replay-mode','--app-replay-match','--app-replay-mode','--nvtx',
 '--nvtx-include','--graph-profiling','--kernel-name-base','--kernel-name',
 '--launch-count','--cache-control','--clock-control','--metrics','--config-file']


def execute(argv, timeout, limit):
    child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,start_new_session=True)
    try:
        out,err=child.communicate(timeout=timeout)
    except BaseException:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.communicate(timeout=2)
        raise
    raw=out+b'\n'+err
    return dict(exit_code=child.returncode,stdout=out[:limit].decode(errors='replace'),
        stderr=err[:max(0,limit-min(len(out),limit))].decode(errors='replace'),
        output_bytes=len(raw),truncated=len(raw)>limit)


def denied(result):
    return bool(re.search(r'ERR_NVGPUCTRPERM|CUPTI_ERROR_INSUFFICIENT_PRIVILEGES|permission.*(?:counter|profil)|(?:counter|profil).*permission',
        result.get('stdout','')+'\n'+result.get('stderr',''),re.I))


def select_metrics(text):
    selected={}
    for semantic,names in METRIC_GROUPS:
        for name in names:
            if re.search(r'(?<![A-Za-z0-9_.])'+re.escape(name)+r'(?![A-Za-z0-9_.])',text):
                selected[semantic]=name;break
    return selected


def command(ncu,arm,metrics,output,target,check):
    kernel='regex:.*FlashAttnFwdSm90.*' if arm=='fa3' else 'regex:.*_fused.*'
    return [ncu,'--config-file','0','--replay-mode','application','--app-replay-match','grid',
      '--app-replay-mode','strict','--nvtx','--nvtx-include','PROFILE_ONE/',
      '--graph-profiling','node','--kernel-name-base','demangled','--kernel-name',kernel,
      '--launch-count','1','--cache-control','none','--clock-control','none',
      '--metrics',','.join(metrics),'--csv','--page','raw','-o',str(output),
      sys.executable,str(target),'--arm',arm,'--output',str(check)]


def csv_metrics(text, requested):
    """Accept only explicit named metric/value or requested-column CSV layouts."""
    rows=list(csv.reader(io.StringIO(text)))
    for i,row in enumerate(rows):
        if 'Metric Name' in row and 'Metric Value' in row:
            names={name:j for j,name in enumerate(row)};out=[]
            for values in rows[i+1:]:
                if len(values)<=max(names.values()):continue
                name=values[names['Metric Name']]
                if name in requested:out.append(dict(name=name,value=values[names['Metric Value']],
                    unit=values[names['Metric Unit']] if 'Metric Unit' in names else None))
            return out
        found=[(j,name) for j,name in enumerate(row) if name in requested]
        if found:
            # Raw wide-format reports have units in the next row, then values.
            if i+2>=len(rows):return []
            units,values=rows[i+1],rows[i+2]
            if len(values)!=len(row):return []
            return [dict(name=name,value=values[j],unit=units[j] if j<len(units) else None) for j,name in found]
    return []


def run_counters(root,dest,deadline_monotonic):
    root,dest=Path(root),Path(dest)
    report=dict(status='starting',sessions_attempted=0,records=[],
        exact_cta_trace=False,sampled_tail_measured=False,
        qualification='Instrumented counters only. No benchmark latency or causal bottleneck claim. Aggregate residency is not exact CTA trace.')
    def remaining():return deadline_monotonic-time.monotonic()
    try:
        inv=json.loads((root/'inventory.json').read_text())
        ncu_info=inv['tools']['ncu']
        if ncu_info['status']!='found' or not Path(NCU).is_file():
            report['status']='unavailable_installed_tool_not_found';return report
        help_result=execute([NCU,'--help'],min(3,max(.1,remaining()-5)),128*1024)
        if help_result['exit_code'] or any(flag not in help_result['stdout'] for flag in REQUIRED_FLAGS):
            report.update(status='unsupported_installed_cli_flags',help_exit_code=help_result['exit_code']);return report
        report['help_sha256']=hashlib.sha256(help_result['stdout'].encode()).hexdigest()
        bases=','.join(dict.fromkeys(names[0].split('.')[0] for _,names in METRIC_GROUPS))
        query=execute([NCU,'--query-metrics','--query-metrics-mode','suffix','--metrics',bases,'--devices','0'],min(5,max(.1,remaining()-5)),1024*1024)
        report['query']=dict(exit_code=query['exit_code'],output_bytes=query['output_bytes'],truncated=query['truncated'])
        if denied(query):
            report.update(status='counter_access_denied_during_query',query_error=(query['stdout']+'\n'+query['stderr'])[:MAX_LOG]);return report
        if query['exit_code'] or query['truncated']:
            report.update(status='metric_query_failed_or_truncated',query_error=(query['stdout']+'\n'+query['stderr'])[:MAX_LOG]);return report
        selected=select_metrics(query['stdout'])
        report['selected_metrics']=selected
        report['unavailable_semantics']=[k for k,_ in METRIC_GROUPS if k not in selected]
        if not selected:
            report['status']='no_requested_metric_supported';return report
        for arm in ['fa3','bf16_custom']:
            if remaining()<25:
                report['status']='deadline_before_next_session';break
            current=sum(p.stat().st_size for p in dest.iterdir() if p.is_file())
            if current+SUMMARY_RESERVE+64*1024>=MAX_TOTAL:
                report['status']='artifact_budget_before_next_session';break
            with tempfile.TemporaryDirectory(prefix='ncu-bounded-') as scratch:
                folder=Path(scratch);check=folder/(arm+'-counter-check.json');output=folder/arm
                argv=command(NCU,arm,list(selected.values()),output,root/'counter_target.py',check)
                row=dict(arm=arm,command=argv,status='starting');report['records'].append(row)
                report['sessions_attempted']+=1
                session_timeout=min(60,remaining()-5)
                row['timeout_seconds']=session_timeout
                try:
                    result=execute(argv,session_timeout,MAX_LOG)
                except subprocess.TimeoutExpired:
                    row.update(status='timeout')
                    report['status']='counter_session_timeout';break
                row.update(result)
                if denied(result):
                    row['status']='permission_denied';report['status']='counter_permission_denied';break
                if check.is_file() and check.stat().st_size<=32*1024:
                    validation=json.loads(check.read_text());row['validation']=validation
                    name=arm+'-counter-check.json';shutil.copyfile(check,dest/name)
                else:
                    row['validation']=None
                binary=output.with_suffix('.ncu-rep')
                if binary.is_file():
                    size=binary.stat().st_size;row['report_bytes']=size
                    available=MAX_TOTAL-SUMMARY_RESERVE-sum(p.stat().st_size for p in dest.iterdir() if p.is_file())
                    if size<=available:
                        raw=binary.read_bytes();row['report_sha256']=hashlib.sha256(raw).hexdigest()
                        shutil.copyfile(binary,dest/(arm+'.ncu-rep'));row['report_preserved']=True
                    else:
                        row['report_preserved']=False;row['status']='counter_report_size_guard';report['status']='counter_report_size_guard';break
                if result['exit_code'] or not row['validation'] or not row['validation'].get('passed') or not row.get('report_preserved'):
                    row['status']='failed_or_incomplete';report['status']='counter_session_failed';break
                row['metrics']=csv_metrics(result['stdout'],set(selected.values()))
                row['missing_requested_metrics']=sorted(set(selected.values())-{m['name'] for m in row['metrics']})
                row['status']='collected' if not row['missing_requested_metrics'] else 'raw_report_collected_metrics_partial_or_unparsed'
        else:
            report['metrics_complete']=all(row['status']=='collected' for row in report['records'])
            report['status']='two_sessions_complete' if report['metrics_complete'] else 'two_sessions_complete_metrics_partial'
    except subprocess.TimeoutExpired:
        report['status']='inventory_query_timeout'
    except TimeoutError:
        report['status']='global_deadline_counter_branch_stopped'
    except Exception as exc:
        report.update(status='counter_branch_failed',error_type=type(exc).__name__,error=str(exc)[:1000])
    assert report['sessions_attempted']<=2
    if len(json.dumps(report).encode())>64*1024:
        for row in report['records']:
            row['stdout']=row.get('stdout','')[:4096];row['stderr']=row.get('stderr','')[:4096]
        report['logs_reduced_for_summary_bound']=True
    return report
