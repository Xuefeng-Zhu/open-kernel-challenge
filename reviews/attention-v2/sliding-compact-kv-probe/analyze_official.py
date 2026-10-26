"""Offline analysis of one compact-window official check; never runs a service."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
SELECTED = V2 / 'fusion-lookahead-probe/official-check'
SELECTED_ID = '5090b83f1d63407b'
SELECTED_SHA = '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
SELECTED_JOB_SHA = 'a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb'
SELECTED_PROFILE_SHA = '3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300'
CANDIDATE_SHA = 'c4307029dac71deb6ee67754e09be9aceedd0d7fcfa3c6421837ef5eaab71594'
LOADS = tuple(f'{kind} {batch}' for kind in ('decode', 'verify') for batch in (1, 16, 32))
KEYS = {(load, layer) for load in LOADS for layer in ('sliding', 'full')}
TARGETS = {(f'verify {batch}', 'sliding') for batch in (1, 16, 32)}
MARKER = 'OKC_COMPACT_FA3_METADATA'
MAX_JSON = 16 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise ValueError('Input symlink/size guard: ' + path.name)
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def key(row):
    case = row['case']
    return case['load'], case['layer']


def rows_from_job(job):
    rows = [r for r in (job.get('result') or {}).get('records', [])
            if r.get('event') == 'benchmark']
    result = {key(r): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate benchmark shape')
    return result


def finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def complete_rows(rows):
    return set(rows) == KEYS and all(
        r.get('status') == 'pass' and not r.get('flags', [])
        and finite_positive(r.get('median_ns')) and finite_positive(r.get('baseline_ns'))
        for r in rows.values())


def score_from_rows(rows, times=None):
    """Exact fixed-reference six-load weighted score, with optional candidate times."""
    if not complete_rows(rows):
        raise ValueError('Expected twelve passing positive benchmark rows')
    loads = []
    for load in LOADS:
        reference = sum(rows[(load, layer)]['baseline_ns'] * weight
                        for layer, weight in (('sliding', 46), ('full', 16)))
        candidate = sum((times[(load, layer)] if times is not None
                         else rows[(load, layer)]['median_ns']) * weight
                        for layer, weight in (('sliding', 46), ('full', 16)))
        if not finite_positive(candidate):
            raise ValueError('Nonpositive modeled load')
        loads.append({'load': load, 'reference_weighted_us': reference / 1000,
                      'candidate_weighted_us': candidate / 1000,
                      'speedup': reference / candidate})
    return math.exp(sum(math.log(r['speedup']) for r in loads) / len(LOADS)), loads


def differences(a, b, prefix=''):
    """All metadata differences, including contextual node/temperature/power values."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out.extend(differences(a.get(k), b.get(k), prefix + ('.' if prefix else '') + k))
        return out
    return [] if a == b else [{'field': prefix, 'selected': a, 'new': b}]


def gpu_identity(result):
    """Stable reported GPU/driver/clock/power-limit fields; not a utilization claim."""
    machine = result.get('machine') or {}
    fields = ('name', 'driver_version', 'clocks.max.sm', 'clocks.sm', 'clocks.mem', 'power.limit')
    return {'gpu': result.get('gpu'), **{
        phase: {name: (machine.get(phase) or {}).get(name) for name in fields}
        for phase in ('gpu_before', 'gpu_after')}}


def identity_known(identity):
    return identity.get('gpu') is not None and all(
        value is not None for phase in ('gpu_before', 'gpu_after')
        for value in identity[phase].values())


def metadata_from_log(text):
    """Only the bounded import-time marker; no guessed runtime output fields."""
    found = [line.strip() for line in str(text or '').splitlines() if line.strip().startswith(MARKER)]
    if not found:
        return {'status': 'UNKNOWN', 'reason': 'metadata marker absent from returned log',
                'resolved_pack_gqa': 'UNKNOWN', 'resolved_splits': 'UNKNOWN',
                'internal_QK_PV_dtype': 'UNKNOWN', 'binary_identity': 'UNKNOWN'}
    if len(found) > 64 or len(set(found)) > 1:
        return {'status': 'UNKNOWN', 'reason': 'inconsistent/too many markers; not selecting one posthoc',
                'marker_count': len(found)}
    line = found[0]
    if len(line.encode('utf-8')) > 4608:
        return {'status': 'UNKNOWN', 'reason': 'marker exceeds bounded metadata envelope'}
    if line == MARKER + ' unavailable':
        return {'status': 'UNKNOWN', 'reason': 'candidate reported metadata unavailable'}
    prefix = MARKER + ' torch='
    if not line.startswith(prefix) or ' cuda=' not in line or ' fwd_schema=' not in line:
        return {'status': 'UNKNOWN', 'reason': 'unrecognized marker format'}
    torch_version, tail = line[len(prefix):].split(' cuda=', 1)
    cuda_version, schema = tail.split(' fwd_schema=', 1)
    if len(torch_version.encode()) > 128 or len(cuda_version.encode()) > 128 or len(schema.encode()) > 4096:
        return {'status': 'UNKNOWN', 'reason': 'metadata field size guard'}
    return {'status': 'observed_API_metadata', 'torch_version': torch_version,
            'cuda_version': cuda_version,
            'identical_marker_observations': len(found),
            'operator_schema': None if schema == 'oversize_not_retained' else schema,
            'operator_schema_status': 'oversize_not_retained' if schema == 'oversize_not_retained' else 'observed',
            'resolved_pack_gqa': 'UNKNOWN', 'resolved_splits': 'UNKNOWN',
            'executed_tile': 'UNKNOWN', 'internal_QK_PV_dtype': 'UNKNOWN',
            'binary_identity': 'UNKNOWN',
            'qualification': 'Registration schema is not resolved packing/split output, an instruction dtype or binary hash.'}


def profile_agrees(job, profile, rows):
    if not isinstance(profile, dict):
        return False, 'profile JSON unavailable'
    if profile.get('job_id') != job.get('id') or profile.get('problem') != 'muse_attention_decode':
        return False, 'profile identity mismatch'
    summary = profile.get('summary') or {}
    speedup = summary.get('speedup')
    if not (summary.get('passed') is True and summary.get('tests') == 16
            and summary.get('tests_passed') == 16 and summary.get('error') is None
            and finite_positive(speedup)
            and math.isclose(speedup, job['result'].get('speedup', float('nan')), rel_tol=1e-12)):
        return False, 'profile summary contradicts raw passing result'
    benchmark = profile.get('benchmarks') or []
    other = {key(r): r for r in benchmark}
    if len(other) != len(benchmark) or set(other) != set(rows):
        return False, 'profile shape set mismatch'
    for k, row in rows.items():
        for name in ('median_ns', 'baseline_ns', 'paired_ratio'):
            a, b = row.get(name), other[k].get(name)
            if not finite_positive(a) or not finite_positive(b) or not math.isclose(a, b, rel_tol=1e-12):
                return False, 'profile/record timing disagreement'
    return True, 'complete profile and raw records agree'


def analyze(job, profile, selected, selected_profile, request=None):
    """Pure object analysis for tests and CLI. No imported candidate/vendor code."""
    result, old_result = job.get('result') or {}, selected.get('result') or {}
    source_bound = (request.get('job_id') == job.get('id') and request.get('source_sha256') == CANDIDATE_SHA
                    if isinstance(request, dict) else None)
    out = {'job_id': job.get('id'), 'mode': job.get('mode'), 'terminal_status': job.get('status'),
           'selected_job_id': selected.get('id'), 'selected_source_sha256': SELECTED_SHA,
           'candidate_source_sha256': CANDIDATE_SHA, 'automatic_promotion': False,
           'source_request_receipt_verified': source_bound,
           'observed_result_passed': result.get('passed'), 'observed_result_error': result.get('error'),
           'new_metadata': metadata_from_log(result.get('log_tail')),
           'qualification': 'Official pairs candidate with its own SGLang reference. Historical normalized selected comparison is descriptive, not a paired candidate-versus-selected experiment.'}
    if job.get('board') != 'muse_attention_decode' or job.get('mode') != 'check':
        return dict(out, classification='invalid_receipt', reason='Expected Attention nonranking check')
    if source_bound is False:
        return dict(out, classification='invalid_receipt', reason='Request does not bind this job to the frozen candidate source')
    if selected.get('id') != SELECTED_ID:
        raise ValueError('Unexpected selected control job')
    if job.get('duplicate_of') is not None:
        return dict(out, classification='inconclusive', reason='Service returned a reused result; no fresh measurement')
    if job.get('status') not in ('done', 'failed', 'timeout', 'cancelled', 'canceled', 'error'):
        return dict(out, classification='not_terminal', reason='Wait for terminal official evidence')
    tests = [r for r in result.get('records', []) if r.get('event') == 'test']
    failed = [r for r in tests if r.get('status') != 'pass']
    checks = result.get('checks') or {}
    out['correctness'] = {'prerequisites_recorded': len(tests),
        'prerequisites_passed': sum(r.get('status') == 'pass' for r in tests),
        'first_failed_case': failed[0].get('case') if failed else None,
        'first_failed_message': failed[0].get('message') if failed else None,
        'result_passed': result.get('passed'), 'returncode': result.get('returncode'),
        'checks_ok': checks.get('ok'), 'error': result.get('error')}
    if failed:
        return dict(out, classification='original_prerequisite_failure',
                    reason='Original prerequisite did not pass; native eligibility versus numerical cause must be read from failure evidence, timings not required')
    if not result.get('passed') or result.get('returncode') != 0 or result.get('error') is not None:
        return dict(out, classification='native_or_execution_failure', reason='Official execution did not pass; timings not required')
    indexes = [r.get('index') for r in tests]
    if (len(tests) != 16 or any(type(i) is not int for i in indexes)
            or set(indexes) != set(range(16)) or not checks.get('ok')
            or checks.get('flagged') or checks.get('rejected') or result.get('flags')):
        return dict(out, classification='incomplete_or_flagged_evidence', reason='Expected all16 original prerequisites and clean official checks')
    rows, old_rows = rows_from_job(job), rows_from_job(selected)
    if not complete_rows(rows) or not complete_rows(old_rows):
        return dict(out, classification='incomplete_or_flagged_evidence', reason='Expected twelve passing unflagged timing rows')
    profile_ok, profile_reason = profile_agrees(job, profile, rows)
    old_profile_ok, old_profile_reason = profile_agrees(selected, selected_profile, old_rows)
    out['profile_agreement'] = {'new': profile_ok, 'new_reason': profile_reason,
        'selected': old_profile_ok, 'selected_reason': old_profile_reason}
    if not profile_ok or not old_profile_ok:
        return dict(out, classification='inconclusive', reason='Profile and raw official timing evidence not verified')
    score, loads = score_from_rows(rows)
    old_score, _ = score_from_rows(old_rows)
    if not math.isclose(score, result.get('speedup', float('nan')), rel_tol=1e-12):
        raise ValueError('New six-load score reconstruction mismatch')
    if not math.isclose(old_score, old_result.get('speedup', float('nan')), rel_tol=1e-12):
        raise ValueError('Selected six-load score reconstruction mismatch')
    comparisons, modeled_times = [], {k: r['median_ns'] for k, r in old_rows.items()}
    for k in sorted(KEYS):
        row, old = rows[k], old_rows[k]
        ratio = (row['median_ns'] / row['baseline_ns']) / (old['median_ns'] / old['baseline_ns'])
        changed = k in TARGETS
        if changed:
            modeled_times[k] *= ratio
        comparisons.append({'load': k[0], 'layer': k[1], 'target': changed,
            'new_candidate_us': row['median_ns'] / 1000, 'new_reference_us': row['baseline_ns'] / 1000,
            'new_candidate_vs_own_reference_paired_ratio': row['paired_ratio'],
            'selected_candidate_us': old['median_ns'] / 1000, 'selected_reference_us': old['baseline_ns'] / 1000,
            'historical_reference_normalized_latency_ratio': ratio,
            'historical_reference_normalized_latency_change_percent': 100 * (ratio - 1)})
    modeled_score, modeled_loads = score_from_rows(old_rows, modeled_times)
    gain = 100 * (modeled_score / old_score - 1)
    target = [r for r in comparisons if r['target']]
    controls = [r for r in comparisons if not r['target']]
    target_worst = max(r['historical_reference_normalized_latency_change_percent'] for r in target)
    control_flags = [r for r in controls if abs(r['historical_reference_normalized_latency_change_percent']) > 3 + 1e-10]
    evaluator_same = (result.get('evaluator') == old_result.get('evaluator')
                      and isinstance(result.get('evaluator'), dict))
    gpu_new, gpu_old = gpu_identity(result), gpu_identity(old_result)
    gpu_known = identity_known(gpu_new) and identity_known(gpu_old)
    gpu_same = gpu_known and gpu_new == gpu_old
    out.update(official_speedup=score, official_speedup_ci=result.get('speedup_ci'),
        selected_speedup=old_score, official_point_score_delta_percent=100 * (score / old_score - 1),
        loads=loads, shapes=comparisons, target_only_fixed_reference_modeled_score=modeled_score,
        target_only_fixed_reference_modeled_gain_percent=gain, target_only_modeled_loads=modeled_loads,
        evaluator={'identical_code_image_files_dictionary': evaluator_same,
                   'differences': differences(old_result.get('evaluator'), result.get('evaluator')),
                   'selected': old_result.get('evaluator'), 'new': result.get('evaluator')},
        gpu_environment={'stable_fields_known': gpu_known, 'stable_identity_clock_limit_fields_equal': gpu_same,
                         'stable_field_differences': differences(gpu_old, gpu_new),
                         'all_reported_machine_differences': differences(old_result.get('machine'), result.get('machine')),
                         'selected_machine': old_result.get('machine'), 'new_machine': result.get('machine'),
                         'qualification': 'Node, temperature and instantaneous draw are retained as context; static identity is not measured utilization.'},
        gates={'all16_prerequisites_and12_shapes_pass': True, 'source_request_receipt_verified': source_bound,
               'same_evaluator': evaluator_same, 'compatible_reported_GPU': gpu_same,
               'target_only_modeled_gain_at_least3_percent': gain >= 3 - 1e-10,
               'no_target_normalized_regression_above3_percent': target_worst <= 3 + 1e-10,
               'no_unchanged_absolute_normalized_drift_above3_percent': not control_flags},
        worst_target_normalized_latency_change_percent=target_worst,
        unchanged_control_drift_flags=control_flags)
    if not evaluator_same or not gpu_same:
        return dict(out, classification='inconclusive_comparison', reason='Evaluator or reported stable GPU environment differs/is unknown')
    if control_flags:
        return dict(out, classification='inconclusive_comparison', reason='Unchanged controls drifted beyond the declared3percent gate')
    if target_worst > 3 + 1e-10 or gain < 3 - 1e-10:
        return dict(out, classification='performance_rejected', reason='Declared target regression/model-score gate failed')
    if source_bound is not True:
        return dict(out, classification='inconclusive_comparison', reason='Fresh job/source request binding not verified')
    return dict(out, classification='descriptive_gain_requires_confirmation',
                reason='Prospective screen passed; actual-best paired/fresh-input confirmation still required, no automatic promotion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--request', type=Path)
    p.add_argument('--selected-job', type=Path, default=SELECTED/'official-result.json')
    p.add_argument('--selected-profile', type=Path, default=SELECTED/'artifacts/candidate.5090b83f1d63407b.json')
    p.add_argument('--candidate-source', type=Path, default=HERE/'candidate.py')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    job, job_hash = read_json(a.job)
    selected, selected_hash = read_json(a.selected_job)
    selected_profile, selected_profile_hash = read_json(a.selected_profile)
    if selected_hash != SELECTED_JOB_SHA or selected_profile_hash != SELECTED_PROFILE_SHA:
        raise ValueError('Selected reference artifact hash differs from the saved reviewed evidence')
    profile, profile_hash = read_json(a.profile) if a.profile and a.profile.exists() else (None, None)
    request, request_hash = read_json(a.request) if a.request else (None, None)
    candidate_hash = digest(a.candidate_source.read_bytes())
    if candidate_hash != CANDIDATE_SHA:
        raise ValueError('Candidate bytes differ from the reviewed frozen source')
    output = analyze(job, profile, selected, selected_profile, request)
    output['artifact_sha256'] = {'job': job_hash, 'profile': profile_hash, 'request': request_hash,
        'selected_job': selected_hash, 'selected_profile': selected_profile_hash,
        'candidate_source': candidate_hash, 'analyzer': digest(Path(__file__).read_bytes())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output.get(k) for k in ['job_id', 'classification', 'reason',
        'official_speedup', 'target_only_fixed_reference_modeled_gain_percent', 'gates']}, indent=2))


if __name__ == '__main__':
    main()
