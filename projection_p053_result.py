"""Offline P053 receipt/trace audit and sanitized publication; never executes GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import projection_p053_phase_screen as worker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(raw, dest, ledger, output):
    if dest.exists():
        raise ValueError('Publication exists; inspect instead of replacing')
    timing_path, activity_path = raw/'timing.json', raw/'activity.json'
    timing, activity = json.loads(timing_path.read_text()), json.loads(activity_path.read_text())
    worker.validate_timing_receipt(timing_path, worker.verify_sources())
    if not (activity['complete'] and activity['oracle_checks'] == 18 and len(activity['profiles']) == 18
            and activity['sources_sha256'] == timing['sources_sha256']
            and activity['timing_receipt']['sha256'] == digest(timing_path)):
        raise ValueError('Activity receipt mismatch')
    rows=[]
    for case in timing['cases']:
        for name, series in case['series'].items():
            summary=worker.summarize(series['samples'],name)
            if summary != series['summary']:
                raise ValueError('Saved paired summary mismatch')
            rows.append(dict(m=case['m'],comparison=name,**summary))
    for profile in activity['profiles']:
        path=raw/profile['trace_file']
        if digest(path)!=profile['trace_sha256'] or profile['passed'] is not True:
            raise ValueError('Raw trace binding mismatch')
        actual=worker.trace_intervals(path,profile['runner'])
        if any(actual[key]!=profile[key] for key in actual):
            raise ValueError('Trace interpretation mismatch')
    dest.mkdir(parents=True)
    manifest={}
    for path in sorted(raw.iterdir()):
        target=dest/path.name
        removed=[]
        if path.name.endswith('.trace.json'):
            data=json.loads(path.read_text())
            for key in ('host_name','trace_id'):
                if key in data:
                    del data[key]
                    removed.append(key)
            target.write_text(json.dumps(data,indent=2)+'\n')
        else:
            shutil.copyfile(path,target)
        manifest[path.name]=dict(raw_sha256=digest(path),published_sha256=digest(target),removed_top_level_fields=removed)
    phases=[]
    for m in worker.CASES:
        for runner in worker.PROFILE_RUNNERS:
            profiles=[p for p in activity['profiles'] if p['m']==m and p['runner']==runner]
            intervals=[p['kernel_intervals'] for p in profiles]
            resources=[]
            for profile in profiles:
                trace=json.loads((raw/profile['trace_file']).read_text())
                resources.append([dict(name=e['name'],**{k:e.get('args',{}).get(k) for k in
                    ('registers per thread','shared memory','grid','block')})
                    for e in trace['traceEvents'] if e.get('cat')=='kernel'])
            phases.append(dict(m=m,runner=runner,raw_intervals=intervals,resource_metadata=resources,
                median_duration_us_by_sequence_position=[statistics.median(seq[i]['duration_us'] for seq in intervals)
                    for i in range(len(intervals[0]))],gaps_us=[p['inter_kernel_gaps_us'] for p in profiles]))
    run=json.loads(ledger.read_text())
    result=dict(experiment='P053',source_commit=run['source_commit'],app_id=run['app_id'],
        decision='Reject reducer-only rescue of the fixed P051/P052 route; preserve P046',
        original_oracle_checks=598,timing_oracle_checks=580,activity_oracle_checks=18,
        timing_receipt_sha256=digest(timing_path),activity_receipt_sha256=digest(activity_path),
        timing=rows,assessment=worker.assess(timing['cases']),separate_activity=phases,
        artifact_manifest=manifest,worker_wall_seconds=run['worker_wall_seconds'],
        client_wall_seconds=run['client_wall_seconds'],shutdown=run['independent_shutdown_observation'],
        cost_reserve_usd=run['cost_reserve_usd'],paid_after_zero=run['paid_after_zero'],
        observed_workspace_gross_delta_usd=run['observed_workspace_gross_delta_usd'],
        delayed_workspace_gross_delta_usd=run.get('delayed_workspace_gross_delta_usd'),
        delayed_billing_observed_utc=run.get('delayed_billing_observed_utc'),
        limitations=['Raw CUDA graph event times; no calibration, official score or formal interval',
            'Partial-only includes layout conversion and FP32 scratch stores; incomplete output until untimed reducer',
            'Three separately instrumented traces per whole path; no median subtraction or measured overlap claim',
            'No hardware counter measurements; profiler occupancy estimates deliberately not interpreted',
            'Raw trace host_name and trace_id removed only from published copies; all events and resource data preserved',
            'Current M8/M7 only; earlier 390-check P052 evidence inherited, not rerun; no concurrent calls',
            'Delayed billing is not a settled per-app invoice; reserve retained'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checked=598,files=len(manifest),decision=result['decision'],gpu_executions_added=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    publish(args.raw,args.dest,args.ledger,args.output)
