"""Verify complete official evidence, with validated fusion as primary control."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
TARGET = ('decode 32', 'full')


def benchmark_rows(job):
    return {(x['case']['load'], x['case']['layer']): x for x in job['result']['records']
            if x['event'] == 'benchmark'}


def reconstruct(job):
    rows = benchmark_rows(job)
    loads = [f'{kind} {batch}' for kind in ['decode', 'verify'] for batch in [1, 16, 32]]
    assert len(rows) == 12 and set(rows) == {(load, layer) for load in loads for layer in ['sliding', 'full']}
    result = []
    for load in loads:
        control = sum(rows[(load, layer)]['baseline_ns']*weight for layer, weight in [('sliding', 46), ('full', 16)])
        candidate = sum(rows[(load, layer)]['median_ns']*weight for layer, weight in [('sliding', 46), ('full', 16)])
        assert control > 0 and candidate > 0
        result.append(dict(load=load, reference_weighted_us=control/1000,
                           candidate_weighted_us=candidate/1000, speedup=control/candidate))
    score = math.exp(sum(math.log(x['speedup']) for x in result)/6)
    assert math.isclose(score, job['result']['speedup'], rel_tol=1e-12)
    return score, result


def main():
    raw = (HERE/'official-result.json').read_bytes()
    job = json.loads(raw)
    terminal = json.loads((HERE/'terminal-receipt.json').read_bytes())
    request = json.loads((HERE/'request-receipt.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == terminal['published_response_sha256']
    assert job['id'] == request['job_id'] == terminal['job_id'] == '5090b83f1d63407b'
    assert job['mode'] == 'check' and job['status'] == 'done' and job['duplicate_of'] is None
    assert request['source_commit'] == 'c9062363c665e37535d3370c1b6bfb54e04fc7c8'
    assert request['source_sha256'] == hashlib.sha256((HERE.parent/'candidate.py').read_bytes()).hexdigest()
    assert request['source_sha256'] == '3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8'
    assert terminal['passed'] and terminal['terminal_status'] == 'done'
    result = job['result']
    assert result['passed'] and result['returncode'] == 0 and result['error'] is None
    assert result['checks']['ok'] and not result['checks']['flagged'] and not result['checks']['rejected']
    tests = [x for x in result['records'] if x['event'] == 'test']
    assert len(tests) == 16 and all(x['status'] == 'pass' for x in tests)
    assert len([x for x in result['records'] if x['event'] == 'benchmark']) == 12
    assert len([x for x in result['records'] if x['event'] == 'profile']) == 12
    current = benchmark_rows(job)
    assert all(not x['flags'] and x['median_ns'] > 0 and x['baseline_ns'] > 0 for x in current.values())
    score, loads = reconstruct(job)
    fusion_path = V2/'fused-preparation-probe/official-check/official-result.json'
    fusion = json.loads(fusion_path.read_bytes())
    reconstruct(fusion)
    old = benchmark_rows(fusion)
    comparisons = []
    for key, row in current.items():
        before = old[key]
        normalized = (row['median_ns']/row['baseline_ns'])/(before['median_ns']/before['baseline_ns'])
        comparisons.append(dict(load=key[0], layer=key[1], changed=key == TARGET,
            official_candidate_us=row['median_ns']/1000, same_run_reference_us=row['baseline_ns']/1000,
            same_run_paired_speedup=row['paired_ratio'], fusion_candidate_us=before['median_ns']/1000,
            fusion_reference_us=before['baseline_ns']/1000,
            reference_normalized_cross_run_latency_reduction_percent=100*(1-normalized)))
    history = {'fusion': fusion}
    for name in ['A2-13', 'A2-11']:
        history[name] = json.loads((V2/(name+'-official-result.json')).read_bytes())
        reconstruct(history[name])
    selected_rows = benchmark_rows(history['A2-11'])
    selected_comparisons = []
    for key, row in current.items():
        before = selected_rows[key]
        normalized = (row['median_ns']/row['baseline_ns'])/(before['median_ns']/before['baseline_ns'])
        selected_comparisons.append(dict(load=key[0], layer=key[1],
            A2_11_candidate_us=before['median_ns']/1000,
            composition_candidate_us=row['median_ns']/1000,
            reference_normalized_cross_run_latency_reduction_percent=100*(1-normalized)))
    target = next(x for x in comparisons if x['changed'])
    unchanged = [x['reference_normalized_cross_run_latency_reduction_percent'] for x in comparisons if not x['changed']]
    evaluation_match = result['evaluator'] == fusion['result']['evaluator']
    output = dict(job_id=job['id'], passed=True, mode='check', source_sha256=request['source_sha256'],
        published_result_sha256=hashlib.sha256(raw).hexdigest(),
        official_speedup=result['speedup'], official_speedup_ci=result['speedup_ci'],
        official_score_us=result['score_us'], reconstructed_speedup=score, loads=loads, shapes=comparisons,
        primary_control_job_id=fusion['id'], primary_control_result_sha256=hashlib.sha256(fusion_path.read_bytes()).hexdigest(),
        primary_control_speedup=fusion['result']['speedup'],
        historical_point_score_deltas_percent={name: 100*(score/value['result']['speedup']-1)
                                              for name, value in history.items()},
        historical_score_context={name: dict(job_id=value['id'], speedup=value['result']['speedup'],
                                            speedup_ci=value['result']['speedup_ci'])
                                  for name, value in history.items()},
        comparison_with_selected_A2_11=selected_comparisons,
        same_evaluator_code_files_image=evaluation_match,
        unchanged_path_normalized_reduction_range_percent=[min(unchanged), max(unchanged)],
        changed_target=target,
        qualifications=['Cross-run reference normalization is descriptive, not a paired composition-versus-fusion score experiment.',
                        'Native 4.8226% passed the new 4% replication gate; the official normalized target cut is a separate value and must not be labeled >=4%.',
                        'Marginal score confidence intervals do not provide a significance test of the cross-run difference.',
                        'A2-13 and A2-11 point comparisons include additional pre-existing source changes; fusion is the primary control.',
                        'A2-11 source remains unchanged as rollback; working selection is recorded separately, with no ranked entry.'])
    (HERE/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ['official_speedup', 'official_speedup_ci',
        'historical_point_score_deltas_percent', 'same_evaluator_code_files_image',
        'unchanged_path_normalized_reduction_range_percent', 'changed_target']}, indent=2))


if __name__ == '__main__':
    main()
