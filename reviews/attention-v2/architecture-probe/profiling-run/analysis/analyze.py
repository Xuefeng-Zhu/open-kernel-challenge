"""Pure-local audit of one fenced activity session and optional NCU evidence.

Intervals describe instrumented activity layout. They are never substituted for
unprofiled paired latency, and missing counters never establish a bottleneck.
No CUDA, Torch, Modal, browser, network or candidate import occurs here.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
PROFILING=HERE.parent
ARCH=PROFILING.parent
ATTENTION=ARCH.parent
EXECUTION_COMMIT='e26bb0cf291dfdbfc6e70065b6d121bf315423ca'
PROFILER_PAYLOAD={
    'worker.py':'07c6c36b06995df46b49b90208b0454eba1aa3efb17787c7a2fbdf67b1efa326',
    'counters.py':'cdf06463a840d75ce8cd3e7ef497fa60c0481b63ff7de6300d4061b23ea12408',
    'counter_target.py':'910e0b3613681edf318d43df25356d8c9ed286b416c2757063d4c0d431142c3f',
    'inventory.json':'f7f114b91ef6fded836f4502e15dbd4e0972a2872ac3e1142c20ad8ad847c382',
}
ARMS=('fa3','bf16_custom')
RANGES={'fa3':'PROFILE_ONE_FA3','bf16_custom':'PROFILE_ONE_BF16_CUSTOM'}
GPU_CATEGORIES={'kernel','gpu_memcpy','gpu_memset'}
STAGES={'qk_norm','fp8_conversion','kv_cache_store','length_schedule_prepare',
        'fa3_mainloop','fa3_combine','output_norm_gate'}
FROZEN={
    'candidate.py':('e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e',ARCH/'probe/candidate.py'),
    'candidatecustomhelper.py':('59fa2bf6aa1e97e55a23b821b56bdd5573b3fdf2edb0613913c2e93dc7acde82',ARCH/'paired/candidatecustomhelper.py'),
    'modal_worker.py':('9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e',ATTENTION/'modal_worker.py'),
    'official/reference.py':('386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b',ATTENTION/'official/reference.py'),
    'official/checks.py':('9a046c3c38db28b325dcd1557baecdf6bc9ef941d2175f9a900ba4c230bb8055',ATTENTION/'official/checks.py'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union(intervals):
    values=sorted((float(a),float(b)) for a,b in intervals if math.isfinite(a) and math.isfinite(b) and b>=a)
    if not values:return 0.
    total=0.;low,high=values[0]
    for a,b in values[1:]:
        if a>high:total+=high-low;low,high=a,b
        else:high=max(high,b)
    return total+high-low


def stage(arm,name,category):
    if arm=='bf16_custom':
        if '_fused' in name:return 'integrated_producer_and_merge'
        if category=='gpu_memset' or 'memset' in name.lower():return 'zero_or_memset_not_independently_attributed'
    else:
        for token,label in [('_qk_rms_norm','qk_norm'),('ConvertToFloat8E4M3','fp8_conversion'),
            ('store_kvcache','kv_cache_store'),('prepare_varlen_num_blocks','length_schedule_prepare'),
            ('FlashAttnFwdSm90','fa3_mainloop'),('FlashAttnFwdCombine','fa3_combine'),('norm_gate','output_norm_gate')]:
            if token in name:return label
    return 'other'


def metadata_ids(args):
    return {k:v for k,v in args.items() if any(token in str(k).lower()
        for token in ('correlation','external id','graph','node','stream'))}


def activity(trace):
    raw=trace.get('traceEvents',[])
    complete=[e for e in raw if e.get('ph')=='X' and isinstance(e.get('ts'),(int,float))
        and isinstance(e.get('dur'),(int,float)) and math.isfinite(e['ts'])
        and math.isfinite(e['dur']) and e['dur']>=0]
    result=dict(trace_events=len(raw),display_time_unit=trace.get('displayTimeUnit'),
        timestamp_duration_unit='microseconds per Chrome trace event format',arms={},issues=[])
    for arm,label in RANGES.items():
        markers=[e for e in complete if e.get('name')==label and e.get('cat') in ('user_annotation','cpu_op')]
        if len(markers)!=1:
            result['issues'].append(f'{arm}: expected one fenced CPU annotation, got {len(markers)}')
            continue
        marker=markers[0];low=marker['ts'];high=low+marker['dur']
        within=[e for e in complete if e is not marker and low<=e['ts'] and e['ts']+e['dur']<=high
            and e.get('cat') in GPU_CATEGORIES|{'cuda_runtime','cuda_driver'}]
        gpu=sorted([e for e in within if e.get('cat') in GPU_CATEGORIES],key=lambda e:(e['ts'],e['dur']))
        kernels=[e for e in gpu if e.get('cat')=='kernel']
        events=[];by_stage={}
        for e in gpu:
            name=str(e.get('name',''));label_stage=stage(arm,name,e.get('cat'))
            row=dict(name=name,name_sha256=hashlib.sha256(name.encode()).hexdigest(),category=e.get('cat'),
                stage=label_stage,start_us=e['ts'],end_us=e['ts']+e['dur'],duration_us=e['dur'],
                relative_to_cpu_marker_us=e['ts']-low,pid=e.get('pid'),tid=e.get('tid'),
                available_ids=metadata_ids(e.get('args',{})))
            events.append(row)
            by_stage.setdefault(label_stage,[]).append(row)
        required=STAGES if arm=='fa3' else {'integrated_producer_and_merge'}
        missing=sorted(required-set(by_stage))
        if missing:result['issues'].append(arm+': missing expected stages '+','.join(missing))
        mainloop=[e for e in kernels if ('FlashAttnFwdSm90' if arm=='fa3' else '_fused') in str(e.get('name',''))]
        if len(mainloop)!=1:result['issues'].append(arm+': mainloop/integrated kernel count is '+str(len(mainloop)))
        api=[]
        for e in sorted([e for e in within if e.get('cat') in ('cuda_runtime','cuda_driver')],key=lambda e:e['ts']):
            api.append(dict(name=e['name'],category=e.get('cat'),start_us=e['ts'],end_us=e['ts']+e['dur'],
                duration_us=e['dur'],available_ids=metadata_ids(e.get('args',{}))))
        launches=[e for e in api if 'graphlaunch' in e['name'].lower()]
        handoffs=[]
        for previous,current in zip(events,events[1:]):
            handoffs.append(dict(from_stage=previous['stage'],to_stage=current['stage'],
                previous_end_us=previous['end_us'],next_start_us=current['start_us'],
                pairwise_separation_us=current['start_us']-previous['end_us'],
                from_ids=previous['available_ids'],to_ids=current['available_ids'],
                interpretation='Pairwise separation only: an earlier event may cover this interval. This is not a global idle gap or a causal handoff. IDs are preserved for explicit stream/graph matching.'))
        intervals=[(e['start_us'],e['end_us']) for e in events]
        kintervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
        span=(max(b for a,b in intervals)-min(a for a,b in intervals)) if intervals else None
        busy=union(intervals)
        uncovered=[]
        if intervals:
            frontier=intervals[0][1]
            for start,end in intervals[1:]:
                if start>frontier:uncovered.append(dict(start_us=frontier,end_us=start,duration_us=start-frontier))
                frontier=max(frontier,end)
        result['arms'][arm]=dict(cpu_annotation_span_us=high-low,
            cpu_span_qualification='Includes launch, completion synchronization and profiler overhead; not unprofiled host launch latency.',
            gpu_activity_span_us=span,gpu_activity_union_us=busy,
            gpu_kernel_union_us=union(kintervals),gpu_activity_duration_sum_us=sum(e['duration_us'] for e in events),
            uncovered_activity_span_us=(span-busy if span is not None else None),
            missing_required_stages=missing,kernel_count=len(kernels),gpu_event_count=len(events),
            stage_union_us={s:union((e['start_us'],e['end_us']) for e in rows) for s,rows in by_stage.items()},
            stage_event_counts={s:len(rows) for s,rows in by_stage.items()},
            gpu_events=events,adjacent_recorded_event_separations=handoffs,
            uncovered_recorded_gpu_activity_intervals=uncovered,cpu_api_events=api,
            cpu_graph_launch_api_events=launches,
            cpu_graph_launch_api_union_us=union((e['start_us'],e['end_us']) for e in launches),
            interpretation='Instrumented activity layout only. Do not subtract these numbers from prior benchmark latency or equate uncovered recorded span with device stall counters.')
    return result


def oracle_audit(summary):
    checks=summary.get('checks',[])
    expected={(arm,label) for arm in ARMS for label in ('pre_profile','post_profile')}
    observed=[(x.get('arm'),x.get('label')) for x in checks]
    problems=[]
    if len(checks)!=4 or set(observed)!=expected:problems.append('Expected exactly four pre/post oracle checks')
    for row in checks:
        valid=all(row.get(k) is True for k in ('passed','oracle_called','stable_addresses','immutable_inputs_and_untouched_rows'))
        if row.get('label')=='post_profile':valid=valid and row.get('exact_output_newkv_equal_pre_profile') is True
        if not valid:problems.append(row.get('arm','?')+'/'+row.get('label','?')+': '+str(row.get('oracle_message')))
    return dict(checks=checks,passed=not problems,issues=problems)


def number(value):
    if not isinstance(value,str):return None
    cleaned=value.strip().replace(',','')
    try:
        parsed=float(cleaned)
        return parsed if math.isfinite(parsed) else None
    except ValueError:return None


def parse_csv_text(text,requested=()):
    """Retain literal NCU units, raw values, IDs and pass fields; no rescaling guesses."""
    lines=text.splitlines()
    headers=[]
    for i,line in enumerate(lines):
        try:cells=next(csv.reader([line]))
        except csv.Error:continue
        if 'Metric Name' in cells and 'Metric Value' in cells:headers.append((i,cells))
    rows=[];errors=[]
    for pos,(offset,columns) in enumerate(headers):
        end=headers[pos+1][0] if pos+1<len(headers) else len(lines)
        reader=csv.reader(io.StringIO('\n'.join(lines[offset+1:end])))
        for cells in reader:
            if not cells or len(cells)!=len(columns):continue
            row=dict(zip(columns,cells))
            if not row.get('Metric Name') or row['Metric Name']=='Metric Name':continue
            row['parsed_numeric_value']=number(row.get('Metric Value'))
            rows.append(row)
    wide_header=None
    if not headers:
        all_rows=list(csv.reader(io.StringIO(text)))
        for i,columns in enumerate(all_rows):
            found=[(j,name) for j,name in enumerate(columns) if name in set(requested)]
            if not found or i+2>=len(all_rows):continue
            wide_header=columns
            units=all_rows[i+1]
            for values in all_rows[i+2:]:
                if len(values)!=len(columns):continue
                for j,name in found:
                    rows.append({'Metric Name':name,'Metric Value':values[j],
                        'Metric Unit':units[j] if j<len(units) else '',
                        'parsed_numeric_value':number(values[j]),
                        'raw_wide_identity_fields':{key:value for key,value in zip(columns,values) if key not in set(requested)}})
            break
        if wide_header is None:errors.append('No supported long/wide NCU metric CSV header found')
    metrics=sorted({r.get('Metric Name') for r in rows})
    units={m:sorted({r.get('Metric Unit','') for r in rows if r.get('Metric Name')==m}) for m in metrics}
    pass_fields={k:sorted({r[k] for r in rows if k in r}) for k in sorted({k for r in rows for k in r if 'pass' in k.lower() or 'replay' in k.lower()})}
    return dict(raw_text_sha256=hashlib.sha256(text.encode()).hexdigest(),rows=rows,row_count=len(rows),metric_names=metrics,
        literal_units=units,literal_pass_replay_fields=pass_fields,parse_issues=errors,
        qualification='Literal tool values/units retained. Kernel replay and metric collection are instrumented; do not compare their duration directly with unprofiled latency.')


def counter_audit(folder,summary):
    files=sorted(p for p in folder.iterdir() if p.is_file() and 'ncu' in p.name.lower())
    denials=[];tool_messages=[];sessions=[];issues=[]
    branch=summary.get('counter_branch') or {}
    selected=branch.get('selected_metrics') or {}
    records=branch.get('records',[])
    attempted=branch.get('sessions_attempted')
    if (isinstance(attempted,(int,float)) and attempted>2) or len(records)>2:
        issues.append('Counter-session bound exceeded')
    if len({r.get('arm') for r in records})!=len(records):issues.append('Duplicate counter arm')
    if attempted is not None and attempted!=len(records):issues.append('Recorded counter session count disagrees')
    denial_re=r'ERR_NVGPUCTRPERM|does not have permission|permission denied|insufficient permission|permission.*(?:counter|profil)|(?:counter|profil).*permission'
    if re.search(denial_re,str(branch.get('query_error','')),re.I):denials.append('metric_query')
    for index,row in enumerate(records):
        arm=row.get('arm');text=row.get('stdout','')+'\n'+row.get('stderr','')
        denied=bool(re.search(denial_re,text,re.I)) or row.get('status')=='permission_denied'
        if denied:denials.append(str(arm))
        parsed=parse_csv_text(row.get('stdout',''),selected.values())
        session_issues=[]
        validation=row.get('validation') or {}
        check_path=folder/(str(arm)+'-counter-check.json')
        if check_path.is_file():
            saved_check=json.loads(check_path.read_text())
            if saved_check!=validation:session_issues.append('Counter validation file differs from summary')
        elif row.get('status')=='collected':session_issues.append('Counter validation artifact missing')
        checks=validation.get('checks',[])
        validated=(validation.get('passed') is True and validation.get('algorithm_changed') is False
            and len(checks)==2 and [c.get('label') for c in checks]==['pre_profile','post_profile'])
        for check in checks:
            validated=validated and all(check.get(k) is True for k in ('passed','oracle_called','stable_addresses','immutable_inputs_and_untouched_rows'))
            if check.get('label')=='post_profile':validated=validated and check.get('exact_output_newkv_return') is True
        for name,(expected,_) in FROZEN.items():
            validated=validated and validation.get('source_sha256',{}).get(name)==expected
        binary=folder/(str(arm)+'.ncu-rep')
        binary_valid=(binary.is_file() and row.get('report_preserved') is True and
            sha(binary)==row.get('report_sha256') and binary.stat().st_size==row.get('report_bytes'))
        if row.get('status')=='collected' and not binary_valid:session_issues.append('Collected NCU report missing or hash/size mismatch')
        if row.get('status')=='collected' and not validated:session_issues.append('Collected counter target pre/post oracle gate failed or unavailable')
        returned=row.get('metrics',[])
        for metric in returned:
            matching=[r for r in parsed['rows'] if r.get('Metric Name')==metric.get('name') and
                r.get('Metric Value')==metric.get('value') and r.get('Metric Unit')==(metric.get('unit') or '')]
            if not matching:session_issues.append('Metric does not match raw stdout CSV: '+str(metric.get('name')))
        for line in text.splitlines():
            if re.search(r'pass(?:es)?|replay|error|warning|profiling|permission',line,re.I):
                tool_messages.append(dict(arm=arm,message=line[:3000]))
        pass_counts=[int(n) for n in re.findall(r'\b(\d+)\s+pass(?:es)?\b',text,re.I)]
        if denied and index!=len(records)-1:session_issues.append('A further counter session was recorded after permission denial')
        admitted=(not session_issues and not denied and row.get('status')=='collected' and
            row.get('exit_code')==0 and not row.get('truncated') and not branch.get('logs_reduced_for_summary_bound')
            and validated and binary_valid and bool(returned) and bool(parsed['rows']))
        identities=[]
        for metric_row in parsed['rows']:
            fields=metric_row.get('raw_wide_identity_fields',metric_row)
            identities.append({k:v for k,v in fields.items() if k in ('ID','Kernel Name','Context','Stream','Grid Size','Block Size','Device')})
        semantic_values={semantic:[dict(value=r.get('Metric Value'),unit=r.get('Metric Unit'),
            numeric_value=r.get('parsed_numeric_value')) for r in parsed['rows'] if r.get('Metric Name')==metric]
            for semantic,metric in selected.items()}
        returned_names={m.get('name') for m in returned}
        if returned and returned_names!=set(selected.values()):
            session_issues.append('Requested/returned metric set differs')
        unavailable_numeric=[semantic for semantic,values in semantic_values.items()
            if not values or not all(v['numeric_value'] is not None and str(v['unit'] or '').strip().lower() not in ('','n/a','na') for v in values)]
        kernel_names=sorted({str(identity.get('Kernel Name')) for identity in identities if identity.get('Kernel Name')})
        expected_token='FlashAttnFwdSm90' if arm=='fa3' else '_fused'
        kernel_identity_verified=len(kernel_names)==1 and expected_token in kernel_names[0]
        if returned and unavailable_numeric:session_issues.append('Nonfinite/missing value or literal unit for '+','.join(unavailable_numeric))
        if returned and not kernel_identity_verified:session_issues.append('Exact selected main-kernel identity unavailable or ambiguous in CSV')
        admitted=admitted and not session_issues and not unavailable_numeric and kernel_identity_verified
        sessions.append(dict(arm=arm,status=row.get('status'),command=row.get('command'),exit_code=row.get('exit_code'),
            output_bytes=row.get('output_bytes'),output_truncated=row.get('truncated'),
            metrics_admitted=admitted,session_issues=session_issues,validation_passed=validated,
            binary_report_verified=binary_valid,parsed_csv=parsed,returned_metrics=returned,
            selected_metric_names=selected,literal_semantic_values=semantic_values,
            numeric_metric_semantics_unavailable=unavailable_numeric,
            selected_kernel_identity_verified=kernel_identity_verified,
            observed_kernel_names=kernel_names,observed_kernel_identity_fields=identities,
            observed_pass_counts=pass_counts,pass_count_qualification='Only explicit tool log counts; command/session count is not hardware replay/pass count.',
            stdout_sha256=hashlib.sha256(row.get('stdout','').encode()).hexdigest(),
            stderr_sha256=hashlib.sha256(row.get('stderr','').encode()).hexdigest()))
    has_metrics=any(s['metrics_admitted'] for s in sessions)
    availability=('denied' if denials else 'validated_instrumented_metrics' if has_metrics else
        'unknown_or_not_collected:'+str(branch.get('status','unavailable')))
    return dict(recorded_branch=branch,recorded_hardware_counter_sessions=summary.get('hardware_counter_sessions'),
        ncu_files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        sessions=sessions,audit_issues=issues,permission_denial_sources=denials,tool_messages=tool_messages,
        metrics_observed=has_metrics,availability=availability,
        unknown_without_validated_counters=['achieved DRAM throughput','tensor-pipe utilization','warp stall reasons','achieved residency/occupancy','per-CTA service-time distribution'],
        qualification='Denial ends counter branch. Absence of data is not evidence of low throughput, low utilization or a bottleneck.')


def analyze(folder,receipt_path=None):
    summary_path=folder/'summary.json';summary=json.loads(summary_path.read_text())
    problems=[];sources={}
    for name,(expected,path) in FROZEN.items():
        actual=sha(path)
        reported=summary.get('source_sha256',{}).get(name)
        sources[name]=dict(local_sha256=actual,reported_sha256=reported,expected_sha256=expected)
        if actual!=expected or reported!=expected:problems.append('Frozen source hash mismatch/missing: '+name)
    worker=PROFILING/'gpu/worker.py'
    worker_hash=sha(worker)
    if summary.get('worker_sha256')!=worker_hash:problems.append('Worker source hash mismatch')
    profiler_sources={}
    for name,expected in PROFILER_PAYLOAD.items():
        actual=sha(PROFILING/'gpu'/name)
        profiler_sources[name]=dict(expected_sha256=expected,local_sha256=actual)
        if actual!=expected:problems.append('Frozen profiling payload mismatch: '+name)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()}
    transport=dict(published_receipt_present=False,artifact_hashes_reverified=False,
        limitation='Summary/raw trace hashes verified locally; transport receipt not re-audited unless published receipt supplied.')
    if receipt_path is not None and receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text())
        transport=dict(published_receipt_present=True,receipt_sha256=sha(receipt_path),
            source_commit=receipt.get('source_commit'),app_id=receipt.get('app_id'),artifact_hashes_reverified=True)
        if receipt.get('source_commit')!=EXECUTION_COMMIT:problems.append('Published receipt execution commit differs')
        for name,expected in {**PROFILER_PAYLOAD,**{k:v[0] for k,v in FROZEN.items()}}.items():
            if receipt.get('payload_sha256',{}).get(name)!=expected:problems.append('Published receipt payload differs: '+name)
        artifact_map=receipt.get('transport_manifest',{}).get('artifact_sha256',{})
        if set(artifact_map)!=set(files):problems.append('Published transport artifact set differs')
        for name,meta in files.items():
            if artifact_map.get(name)!=meta['sha256']:
                transport['artifact_hashes_reverified']=False
                problems.append('Published transport artifact hash differs: '+name)
    oracle=oracle_audit(summary)
    if not oracle['passed']:problems.extend(oracle['issues'])
    trace_path=folder/'activity-trace.json';raw_analysis=None
    if trace_path.is_file():
        saved=summary.get('artifacts',{}).get('activity-trace.json',{})
        if saved.get('sha256')!=sha(trace_path) or saved.get('bytes')!=trace_path.stat().st_size:
            problems.append('Raw activity trace hash/size mismatch or unavailable')
        raw_analysis=activity(json.loads(trace_path.read_text()))
        problems.extend(raw_analysis['issues'])
        compact=summary.get('activity',{}).get('arms',{})
        for arm,row in raw_analysis['arms'].items():
            saved=compact.get(arm,{})
            for key in ('gpu_activity_span_us','gpu_activity_union_us','gpu_kernel_union_us'):
                value=saved.get(key)
                if not isinstance(value,(int,float)) or not isinstance(row[key],(int,float)) or not math.isclose(value,row[key],rel_tol=1e-9,abs_tol=1e-6):
                    problems.append(arm+': saved compact activity differs for '+key)
    else:problems.append('Raw activity trace missing')
    if summary.get('activity_sessions')!=1 or summary.get('profiled_graph_replays')!={a:1 for a in ARMS}:
        problems.append('Expected exactly one activity session and one replay per arm')
    admitted=(not problems and summary.get('status')=='complete' and summary.get('passed') is True)
    return dict(kind='Offline activity/counter evidence audit, not benchmark latency or automatic bottleneck diagnosis',
        analyzer_sha256=sha(Path(__file__)),artifact_files=files,source_provenance=sources,
        execution_commit=EXECUTION_COMMIT,profiler_payload_provenance=profiler_sources,
        transport_audit=transport,
        worker_sha256=worker_hash,summary_status=summary.get('status'),summary_passed=summary.get('passed'),
        environment=summary.get('environment'),input_metadata=summary.get('input_metadata'),
        oracle_audit=oracle,activity_evidence_admitted=admitted,audit_issues=problems,
        activity=raw_analysis,counters=counter_audit(folder,summary),
        benchmark_claim=False,exact_cta_trace=False,
        limitations=['One fixed input and one instrumented replay per unchanged arm; not an independent performance sample.',
          'Stage unions/spans/handoffs are instrumented timestamps, not unprofiled wall latency or causal hardware-stall evidence.',
          'Custom producer/softmax/merge remain integrated and cannot be timed internally from one kernel event.',
          'No subtraction from previous paired benchmark and no automatic cause assignment from instruction or duration totals.'])


def render(report):
    lines=['# Fixed B32 activity and counter evidence','',
        'Frozen execution source commit `'+report['execution_commit']+'`.',
        f"Activity evidence admitted: `{report['activity_evidence_admitted']}`. Four pre/post oracle checks passed: `{report['oracle_audit']['passed']}`.",
        '','This is one instrumented CPU/CUDA activity session with one unchanged FA3 and one unchanged BF16-custom graph replay. Durations below describe activity layout; they are not unprofiled latency and must not be subtracted from earlier paired benchmark results.']
    if report['audit_issues']:
        lines+=['','Audit limitations:','']+['- '+x for x in report['audit_issues']]
    if report['activity_evidence_admitted']:
        lines+=['','| Arm | GPU activity union us | GPU activity span us | CPU annotation span us | Instrumented graph-launch API union us |',
            '|---|---:|---:|---:|---:|']
        for arm,row in report['activity']['arms'].items():
            lines.append(f"| {arm} | {row['gpu_activity_union_us']:.3f} | {row['gpu_activity_span_us']:.3f} | {row['cpu_annotation_span_us']:.3f} | {row['cpu_graph_launch_api_union_us']:.3f} |")
        lines+=['','CPU annotations include completion synchronization and profiler overhead. API event durations are themselves instrumented; they do not establish ordinary launch overhead. GPU span can include gaps, while union avoids double-counting overlaps.','',
            '| Arm | Observed stage | Events | Interval union us |','|---|---|---:|---:|']
        for arm,row in report['activity']['arms'].items():
            for name,value in row['stage_union_us'].items():
                lines.append(f"| {arm} | {name} | {row['stage_event_counts'][name]} | {value:.3f} |")
        lines+=['','FA3 stage attribution uses complete observed kernel names. FP8 conversion names alone do not identify query versus cache conversion. The custom kernel includes producer and merge; their internal costs are unresolved. Memset/fill destination attribution to ticket storage remains unproven. Exact event IDs, streams, pairwise stage separations and full names are retained in `analysis.json` and the raw trace.','',
            '| Arm | Adjacent recorded stages | Pairwise separation us |','|---|---|---:|']
        for arm,row in report['activity']['arms'].items():
            for pair in row['adjacent_recorded_event_separations']:
                if pair['from_stage']!=pair['to_stage']:
                    lines.append(f"| {arm} | {pair['from_stage']} → {pair['to_stage']} | {pair['pairwise_separation_us']:+.3f} |")
        lines+=['','Negative separation means overlapping recorded intervals; positive separation is only pairwise and an earlier event may cover it. These are not inferred SM idle times, causal handoff delays or warp stalls. The complete union-frontier uncovered intervals are retained separately. In particular, overlapping schedule/mainloop/combine intervals prevent summing their durations into wall latency.']
    counters=report['counters']
    lines+=['','Counter branch: `'+counters['availability']+'`.']
    for session in counters['sessions']:
        if session['status']=='timeout':
            original=next((r for r in counters['recorded_branch'].get('records',[]) if r.get('arm')==session['arm']),{})
            lines+=['',str(session['arm'])+' was the only attempted main-kernel NCU session and timed out at '+str(original.get('timeout_seconds','unknown'))+' seconds. No second arm was attempted. The timeout path returned no stdout/stderr, target validation or raw NCU report; pass/replay progress and permission status cannot be determined. This is a timeout, not a recorded permission denial. No retry or extra experiment was performed.']
    if counters['permission_denial_sources']:
        lines+=['','The counter tool explicitly denied permission. No counter retry, permission bypass or inferred metric is supported by this evidence.']
    if not counters['metrics_observed']:
        lines+=['','Achieved DRAM throughput, tensor utilization, warp stalls, achieved residency and per-CTA service times remain unknown. Activity stage duration is insufficient to call the kernel bandwidth-, tensor- or occupancy-limited.']
    else:
        lines+=['','| Arm | Metric semantic | Exact metric | Literal value | Literal unit |',
            '|---|---|---|---|---|']
        for session in counters['sessions']:
            if not session['metrics_admitted']:continue
            for semantic,values in session['literal_semantic_values'].items():
                for value in values:
                    lines.append('| '+str(session['arm'])+' | '+semantic+' | `'+session['selected_metric_names'][semantic]+'` | '+str(value['value'])+' | '+str(value['unit'])+' |')
        for session in counters['sessions']:
            lines+=['',str(session['arm'])+' explicit pass counts in tool logs: '+str(session['observed_pass_counts'])+'. Session count is not replay/pass count.']
        lines+=['','NCU kernel IDs, raw units/values and available replay/pass fields are preserved in `analysis.json`. Collection/replay duration is instrumented, and counters target one main kernel per arm rather than complete-call work. Aggregate active warps/CTAs do not supply exact CTA service times or tail imbalance. A causal conclusion requires convergent validated metrics; no single utilization metric proves a bottleneck and no metric is substituted for whole-call latency.']
    lines+=['','Exact source and raw artifact hashes are in `analysis.json`; deliverable hashes are in `manifest.json`. No model, candidate route, benchmark or official result was changed by this local postprocessor.']
    return '\n'.join(lines)+'\n'


def self_test():
    assert union([(0,4),(2,6),(10,12)])==8
    assert stage('bf16_custom','_fused','kernel')=='integrated_producer_and_merge'
    assert stage('fa3','void FlashAttnFwdSm90<...>','kernel')=='fa3_mainloop'
    checks=[dict(arm=a,label=l,passed=True,oracle_called=True,stable_addresses=True,
        immutable_inputs_and_untouched_rows=True,exact_output_newkv_equal_pre_profile=True if l=='post_profile' else None)
        for a in ARMS for l in ('pre_profile','post_profile')]
    assert oracle_audit(dict(checks=checks))['passed']
    checks[-1]['exact_output_newkv_equal_pre_profile']=False
    assert not oracle_audit(dict(checks=checks))['passed']
    assert number('1,234.5')==1234.5 and number('N/A') is None
    long='"ID","Metric Name","Metric Unit","Metric Value"\n"0","gpu__time_duration.sum","nsecond","1234"\n'
    parsed=parse_csv_text(long)
    assert parsed['rows'][0]['Metric Unit']=='nsecond' and parsed['rows'][0]['parsed_numeric_value']==1234
    wide='"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum"\n"","","nsecond","byte"\n"0","_fused","1234","2,048"\n'
    parsed=parse_csv_text(wide,['gpu__time_duration.sum','dram__bytes_read.sum'])
    assert len(parsed['rows'])==2 and parsed['rows'][1]['parsed_numeric_value']==2048
    labels=['_qk_rms_norm','ConvertToFloat8E4M3','store_kvcache','prepare_varlen_num_blocks','FlashAttnFwdSm90','FlashAttnFwdCombine','norm_gate']
    events=[dict(ph='X',cat='user_annotation',name=RANGES['fa3'],ts=100,dur=1000),
        dict(ph='X',cat='user_annotation',name=RANGES['bf16_custom'],ts=2000,dur=1000),
        dict(ph='X',cat='cuda_runtime',name='cudaGraphLaunch',ts=110,dur=5),
        dict(ph='X',cat='kernel',name='_fused',ts=2100,dur=400)]
    events += [dict(ph='X',cat='kernel',name=name,ts=200+i*100,dur=80) for i,name in enumerate(labels)]
    profiled=activity(dict(traceEvents=events))
    assert not profiled['issues'] and profiled['arms']['fa3']['gpu_activity_union_us']==560
    assert profiled['arms']['fa3']['gpu_activity_span_us']==680 and profiled['arms']['fa3']['cpu_graph_launch_api_union_us']==5
    nested=[dict(ph='X',cat='user_annotation',name=RANGES['bf16_custom'],ts=0,dur=200),
        dict(ph='X',cat='kernel',name='_fused',ts=10,dur=100),
        dict(ph='X',cat='gpu_memset',name='memsetA',ts=20,dur=10),
        dict(ph='X',cat='gpu_memset',name='memsetB',ts=40,dur=10)]
    nested_result=activity(dict(traceEvents=nested))['arms']['bf16_custom']
    assert nested_result['uncovered_recorded_gpu_activity_intervals']==[] and nested_result['uncovered_activity_span_us']==0
    assert nested_result['adjacent_recorded_event_separations'][-1]['pairwise_separation_us']==10
    print('9 local interval/stage/oracle/CSV/fenced-trace guards passed; no GPU/network/browser activity')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=PROFILING/'gpu/artifacts')
    parser.add_argument('--receipt',type=Path,default=PROFILING/'gpu/launch-receipt.json',
        help='Optional published transport receipt. Never reads an ignored local ledger implicitly.')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return 0
    report=analyze(args.artifacts,args.receipt)
    (HERE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'RESULT.md').write_text(render(report))
    names=['analyze.py','analysis.json','RESULT.md']
    names += [n for n in ('local-verification.json','README.md') if (HERE/n).is_file()]
    (HERE/'manifest.json').write_text(json.dumps({n:sha(HERE/n) for n in names},indent=2)+'\n')
    print(json.dumps(dict(activity_evidence_admitted=report['activity_evidence_admitted'],
        issues=report['audit_issues'],counter_status=report['counters']['availability'],analysis_sha256=sha(HERE/'analysis.json')),indent=2))
    return 0 if report['activity_evidence_admitted'] else 2


if __name__=='__main__':raise SystemExit(main())
