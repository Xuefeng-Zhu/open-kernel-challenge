"""Offline supplement for the separate final single-metric access session.

Never launches NCU or imports CUDA/Torch. Original activity artifacts and result
are left unchanged. A byte counter is not a bandwidth or performance result.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile

HERE=Path(__file__).resolve().parent
PROFILING=HERE.parent
ACCESS=PROFILING/'counter-access'
METRIC='dram__bytes_read.sum'
EXECUTION_COMMIT='052665dd2496f12952d684ee6fd5a4b0e1075228'
ACCESS_WORKER_SHA256='7c2b3cfd6b53151cc69d0d17b7673d6f481be5ecbe891d3c851440d3df88f9f7'
ORIGINAL_SUMMARY_SHA256='a48846ac5a4a79de771822130eee0d46705fa5350c49fafff9508125d977ba15'
ORIGINAL_TRACE_SHA256='e49b1677de2dfbc9fcca43e63924a026cd7c101321e2bee1131f87ee41355ffc'
EXPECTED_SUPPORT={
    'worker.py':'07c6c36b06995df46b49b90208b0454eba1aa3efb17787c7a2fbdf67b1efa326',
    'counter_target.py':'910e0b3613681edf318d43df25356d8c9ed286b416c2757063d4c0d431142c3f',
}
spec=importlib.util.spec_from_file_location('offline_activity_audit',HERE/'analyze.py')
BASE=importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def log_audit(value):
    if not isinstance(value,dict):return dict(available=False,verified=False,text='',issue='No retained log record')
    try:raw=base64.b64decode(value.get('preserved_base64',''),validate=True)
    except (ValueError,TypeError):return dict(available=True,verified=False,text=value.get('text',''),issue='Invalid preserved base64')
    original_size=value.get('bytes');preserved_size=value.get('preserved_bytes');truncated=value.get('truncated')
    sizes_valid=(type(original_size) is int and type(preserved_size) is int and
        0<=preserved_size<=original_size and type(truncated) is bool and truncated==(preserved_size<original_size))
    valid=(sizes_valid and len(raw)==preserved_size and hashlib.sha256(raw).hexdigest()==value.get('preserved_sha256')
        and raw.decode(errors='replace')==value.get('text'))
    return dict(available=True,verified=valid,text=value.get('text',''),original_bytes=value.get('bytes'),
        preserved_bytes=value.get('preserved_bytes'),truncated=value.get('truncated'),
        preserved_sha256=value.get('preserved_sha256'),
        qualification='Truncated logs retain head and tail; absence of a marker in such logs does not prove it never appeared.')


def analyze(folder,receipt_path):
    summary_path=folder/'summary.json';summary=json.loads(summary_path.read_text())
    problems=[];provenance={}
    worker=ACCESS/'access_worker.py'
    provenance['access_worker.py']=dict(local_sha256=sha(worker),reported_sha256=summary.get('worker_sha256'),expected_sha256=ACCESS_WORKER_SHA256)
    if sha(worker)!=ACCESS_WORKER_SHA256 or summary.get('worker_sha256')!=ACCESS_WORKER_SHA256:problems.append('Frozen access worker hash differs')
    for name,expected in EXPECTED_SUPPORT.items():
        actual=sha(PROFILING/'gpu'/name)
        provenance[name]=dict(local_sha256=actual,reported_sha256=summary.get('source_sha256',{}).get(name),expected_sha256=expected)
        if actual!=expected or summary.get('source_sha256',{}).get(name)!=expected:problems.append('Frozen support mismatch: '+name)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()}
    transport=dict(published_receipt_present=False,artifact_hashes_reverified=False)
    if receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text())
        transport=dict(published_receipt_present=True,receipt_sha256=sha(receipt_path),
            source_commit=receipt.get('source_commit'),app_id=receipt.get('app_id'),artifact_hashes_reverified=True)
        if receipt.get('source_commit')!=EXECUTION_COMMIT:problems.append('Published access execution commit differs')
        artifact_map=receipt.get('transport_manifest',{}).get('artifact_sha256',{})
        if set(artifact_map)!=set(files):problems.append('Published transport artifact set differs')
        for name,meta in files.items():
            if artifact_map.get(name)!=meta['sha256']:
                transport['artifact_hashes_reverified']=False;problems.append('Transport hash mismatch: '+name)
        if receipt.get('payload_sha256',{}).get('access_worker.py')!=sha(worker):problems.append('Published worker payload hash differs')
        for name,expected in EXPECTED_SUPPORT.items():
            if receipt.get('payload_sha256',{}).get(name)!=expected:problems.append('Published support payload hash differs: '+name)
    protocol=(summary.get('activity_sessions')==0 and summary.get('prior_counter_sessions')==1
        and summary.get('hardware_counter_sessions') in (0,1) and summary.get('overall_counter_sessions_upper_bound')==2
        and summary.get('arm')=='fa3' and summary.get('metric')==METRIC and summary.get('parent_cuda_context') is False)
    if not protocol:problems.append('Fixed single-arm/session/metric protocol not established')
    stdout=log_audit(summary.get('stdout'));stderr=log_audit(summary.get('stderr'))
    if summary.get('hardware_counter_sessions')==1:
        for label,log in [('stdout',stdout),('stderr',stderr)]:
            if not log['verified']:problems.append('Retained '+label+' bytes not verified')
    combined=stdout['text']+'\n'+stderr['text']
    denial_lines=[line for line in combined.splitlines() if re.search(r'ERR_NVGPUCTRPERM|permission.*(?:counter|profil)|insufficient privileges|access.*performance counters.*not permitted',line,re.I)]
    parsed=BASE.parse_csv_text(stdout['text'],[METRIC])
    validation=summary.get('validation') or {}
    checks=validation.get('checks',[])
    oracle=(validation.get('passed') is True and validation.get('arm')=='fa3' and validation.get('algorithm_changed') is False
        and [c.get('label') for c in checks]==['pre_profile','post_profile'])
    for check in checks:
        oracle=oracle and all(check.get(k) is True for k in ('passed','oracle_called','stable_addresses','immutable_inputs_and_untouched_rows'))
        if check.get('label')=='post_profile':oracle=oracle and check.get('exact_output_newkv_return') is True
    for name,(expected,_) in BASE.FROZEN.items():oracle=oracle and validation.get('source_sha256',{}).get(name)==expected
    case=validation.get('case',{})
    expected_case=dict(batch=32,q_len=8,ctx=80000,heads='tp8',layer='full',page=1,
        kv_scale=1.,spread='agentic',dist='coherent',seed=135701)
    oracle=oracle and case==expected_case
    original=json.loads((PROFILING/'gpu/artifacts/summary.json').read_text())
    if sha(PROFILING/'gpu/artifacts/summary.json')!=ORIGINAL_SUMMARY_SHA256 or sha(PROFILING/'gpu/artifacts/activity-trace.json')!=ORIGINAL_TRACE_SHA256:
        problems.append('Original activity raw artifacts changed')
    oracle=oracle and validation.get('actual_lengths')==original.get('input_metadata',{}).get('actual_lengths')
    check_path=folder/'fa3-counter-check.json'
    check_artifact_valid=check_path.is_file() and json.loads(check_path.read_text())==validation
    if check_path.is_file() and not check_artifact_valid:problems.append('Counter check artifact differs from summary')
    oracle=oracle and check_artifact_valid
    binary=folder/'fa3.ncu-rep'
    binary_valid=(binary.is_file() and binary.stat().st_size>0 and summary.get('raw_report_preserved') is True
        and sha(binary)==summary.get('raw_report_sha256') and binary.stat().st_size==summary.get('raw_report_bytes'))
    metrics=[r for r in parsed['rows'] if r.get('Metric Name')==METRIC]
    kernel_names=set()
    for row in metrics:
        identity=row.get('raw_wide_identity_fields',row)
        if identity.get('Kernel Name'):kernel_names.add(identity['Kernel Name'])
    kernel_valid=len(kernel_names)==1 and 'FlashAttnFwdSm90' in next(iter(kernel_names))
    value_valid=bool(metrics) and all(r.get('parsed_numeric_value') is not None and
        r['parsed_numeric_value']>=0 and str(r.get('Metric Unit') or '').strip().lower() not in ('','n/a','na') for r in metrics)
    logs_complete=stdout['verified'] and stderr['verified'] and stdout.get('truncated') is False and stderr.get('truncated') is False
    denied=summary.get('permission_denied') is True or bool(denial_lines)
    raw_collected=(not problems and protocol and summary.get('hardware_counter_sessions')==1 and not denied and summary.get('status')=='counter_access_collected'
        and summary.get('passed') is True and summary.get('exit_code')==0 and oracle and binary_valid)
    metric_admitted=raw_collected and value_valid and kernel_valid and logs_complete
    if denied:status='permission_denied'
    elif summary.get('timed_out'):status='timeout_partial_logs_preserved'
    elif metric_admitted:status='validated_single_metric_access'
    elif raw_collected:status='raw_counter_report_collected_numeric_value_not_admitted'
    else:status='failed_or_incomplete_access_evidence'
    messages=[line for line in combined.splitlines() if re.search(r'pass(?:es)?|replay|error|warning|profiling|permission|connected',line,re.I)]
    return dict(kind='Separate final single-metric FA3 access supplement; no performance or BF16 comparison',
        status=status,analyzer_sha256=sha(Path(__file__)),base_parser_sha256=sha(HERE/'analyze.py'),
        artifact_files=files,source_provenance=provenance,transport_audit=transport,
        execution_commit=EXECUTION_COMMIT,
        original_activity_summary_sha256=sha(PROFILING/'gpu/artifacts/summary.json'),
        original_activity_trace_sha256=sha(PROFILING/'gpu/artifacts/activity-trace.json'),
        recorded_status=summary.get('status'),recorded_passed=summary.get('passed'),exit_code=summary.get('exit_code'),
        timeout_seconds=summary.get('timeout_seconds'),timed_out=summary.get('timed_out',False),
        hardware_counter_sessions=summary.get('hardware_counter_sessions'),prior_counter_sessions=summary.get('prior_counter_sessions'),
        protocol_verified=protocol,audit_issues=problems,logs=dict(stdout=stdout,stderr=stderr),
        permission_denied=denied,denial_evidence=denial_lines,tool_messages=messages,
        observed_pass_counts=[int(x) for x in re.findall(r'\b(\d+)\s+pass(?:es)?\b',combined,re.I)],
        target_validation=validation,target_oracles_passed=oracle,raw_report_verified=binary_valid,
        raw_counter_access_completed=raw_collected,metric_value_admitted=metric_admitted,
        parsed_csv=parsed,exact_metric_rows=metrics,selected_kernel_names=sorted(kernel_names),
        benchmark_claim=False,bf16_comparison=False,bandwidth_claim=False,
        limitations=['One FA3 DRAM-byte counter only, not a complete-call measurement or a two-arm comparison.',
            'Bytes alone do not establish bytes/second without a matched elapsed-time measurement.',
            'No tensor utilization, stall, achieved residency, exact CTA tail or causal bottleneck inference.',
            'The original activity session and its first 60-second counter timeout remain unchanged.'])


def render(report):
    lines=['# Separate final counter-access supplement','',
        'Status: `'+report['status']+'`. This uses the remaining counter authorization separately from the original activity run and first 60-second NCU timeout. Recorded counter sessions here: '+str(report['hardware_counter_sessions'])+'. No new activity trace or algorithm change was made.','',
        'Frozen supplemental execution commit `'+report['execution_commit']+'`.',
        f"Raw supplement summary SHA256 `{report['artifact_files']['summary.json']['sha256']}`.",
        f"Original activity summary remains `{report['original_activity_summary_sha256']}` and trace `{report['original_activity_trace_sha256']}`."]
    if report['permission_denied']:
        lines+=['','NCU explicitly denied hardware-counter access for the fixed FA3 B32/L8/80K/page1 kernel. The action is denied, and no counter value is admitted.','']+['> '+line for line in report['denial_evidence']]
    elif report['timed_out']:
        verified=all(log.get('verified') is True for log in report['logs'].values())
        qualifier='retained stdout/stderr bytes are hash-verified' if verified else 'retained stdout/stderr integrity is incomplete; see audit issues'
        lines+=['',f"The one FA3 `dram__bytes_read.sum` session timed out after {report['timeout_seconds']} seconds; {qualifier}. This does not establish permission denial or a performance result."]
    elif report['metric_value_admitted']:
        lines+=['','The fixed FA3 target passed both pre/post official oracles and exact output/new-KV return; raw NCU report and log bytes verified. The following literal hardware metric value is admitted:','',
            '| Metric | Literal value | Literal unit |','|---|---:|---|']
        for metric in report['exact_metric_rows']:
            lines.append('| `'+METRIC+'` | '+str(metric['Metric Value'])+' | '+str(metric['Metric Unit'])+' |')
        lines+=['','This establishes access to this counter for the named FA3 main kernel. It does not establish a speedup or a BF16 comparison.']
    else:lines+=['','Collection or numeric validation did not complete. No usable hardware metric or performance claim is admitted.']
    lines+=['','FA3 target pre/post oracle and exact post-profile output/new-KV validation passed: `'+str(report['target_oracles_passed'])+'`. Workload validation does not establish successful hardware-counter collection.']
    for name,log in report['logs'].items():
        lines+=['',name+' retained bytes: '+str(log.get('preserved_bytes'))+' of '+str(log.get('original_bytes'))+'; truncated: `'+str(log.get('truncated'))+'`; byte/hash/text verification: `'+str(log.get('verified'))+'`.']
    if report['tool_messages']:
        lines+=['','Preserved diagnostic lines:','']+['> '+line for line in report['tool_messages']]
        lines+=['','A profiler connection message establishes connection to the child process; it does not establish permission to read hardware counters or a collected metric.']
    if report['audit_issues']:lines+=['','Audit issues:','']+['- '+x for x in report['audit_issues']]
    lines+=['','Explicit pass counts observed in logs: '+str(report['observed_pass_counts'])+'. Requested session count is not replay/pass count.','',
        'This supplemental session changed both the requested metric count and the parent process/CUPTI context arrangement. Even successful access cannot isolate the cause of the first session\'s timeout. The authorized totals are now one activity session and at most two counter sessions; no further collection follows from this analyzer.',
        '',
        'DRAM bytes alone do not establish bandwidth. Tensor utilization, stall causes, residency and exact CTA tail behavior remain unknown. No permission bypass, further collection or candidate/model change was performed by this analyzer.','',
        'Complete retained log text/base64 hashes, source hashes and raw report checks are in `counter-access-analysis.json`; deliverable hashes are in `counter-access-manifest.json`.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=ACCESS/'artifacts')
    parser.add_argument('--receipt',type=Path,default=ACCESS/'launch-receipt.json')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        raw=b'==ERROR== ERR_NVGPUCTRPERM: permission to access NVIDIA GPU performance counters\n'
        log=dict(bytes=len(raw),preserved_bytes=len(raw),truncated=False,
            preserved_sha256=hashlib.sha256(raw).hexdigest(),preserved_base64=base64.b64encode(raw).decode(),text=raw.decode())
        assert log_audit(log)['verified']
        invalid=dict(log,preserved_sha256='0'*64)
        assert not log_audit(invalid)['verified']
        assert not log_audit(dict(log,bytes=len(raw)+1,truncated=False))['verified']
        assert not log_audit({k:v for k,v in log.items() if k!='truncated'})['verified']
        empty=dict(bytes=0,preserved_bytes=0,truncated=False,preserved_sha256=hashlib.sha256(b'').hexdigest(),preserved_base64='',text='')
        synthetic=dict(worker_sha256=sha(ACCESS/'access_worker.py'),source_sha256=EXPECTED_SUPPORT,
            activity_sessions=0,prior_counter_sessions=1,hardware_counter_sessions=1,overall_counter_sessions_upper_bound=2,
            parent_cuda_context=False,arm='fa3',metric=METRIC,stdout=log,stderr=empty,
            permission_denied=True,status='counter_permission_denied_stop',passed=False)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'summary.json').write_text(json.dumps(synthetic))
            audit=analyze(folder,folder/'absent-receipt.json')
            assert audit['status']=='permission_denied' and not audit['metric_value_admitted']
            synthetic.update(permission_denied=False,timed_out=True,status='timeout_partial_logs_preserved',stdout=empty)
            (folder/'summary.json').write_text(json.dumps(synthetic))
            audit=analyze(folder,folder/'absent-receipt.json')
            assert audit['status']=='timeout_partial_logs_preserved' and not audit['permission_denied'] and not audit['metric_value_admitted']
        print('6 retained-log integrity/metadata/denial/timeout guards passed; no GPU/network/browser calls')
        return 0
    report=analyze(args.artifacts,args.receipt)
    (HERE/'counter-access-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'COUNTER_ACCESS_SUPPLEMENT.md').write_text(render(report))
    names=['analyze_access.py','counter-access-analysis.json','COUNTER_ACCESS_SUPPLEMENT.md']
    names += [name for name in ('counter-access-local-verification.json',) if (HERE/name).is_file()]
    (HERE/'counter-access-manifest.json').write_text(json.dumps({name:sha(HERE/name) for name in names},indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],metric_admitted=report['metric_value_admitted'],
        audit_issues=report['audit_issues'],analysis_sha256=sha(HERE/'counter-access-analysis.json')),indent=2))
    return 0 if not report['audit_issues'] else 2


if __name__=='__main__':raise SystemExit(main())
