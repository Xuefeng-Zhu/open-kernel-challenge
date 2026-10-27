"""Reconstruct the official six-load score and qualified cross-run comparisons."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
TARGETS = {('decode 16', 'full'), ('verify 16', 'full'), ('verify 32', 'full')}


def rows(job):
    return {(row['case']['load'], row['case']['layer']): row
            for row in job['result']['records'] if row['event'] == 'benchmark'}


def main():
    raw = (HERE/'official-result.json').read_bytes()
    job = json.loads(raw)
    terminal = json.loads((HERE/'terminal-receipt.json').read_bytes())
    request = json.loads((HERE/'request-receipt.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == terminal['published_response_sha256']
    assert job['id'] == request['job_id'] == terminal['job_id'] == '2f5f1fb54b48486d'
    assert job['mode'] == 'check' and job['status'] == 'done' and job['duplicate_of'] is None
    assert request['source_commit'] == '0280972637e600e5f82a45f35778aed3f72886b1'
    assert request['source_sha256'] == hashlib.sha256((HERE.parent/'candidate.py').read_bytes()).hexdigest()
    result = job['result']
    assert result['passed'] and result['returncode'] == 0 and result['error'] is None
    assert result['checks']['ok'] and not result['checks']['flagged'] and not result['checks']['rejected']
    tests = [x for x in result['records'] if x['event'] == 'test']
    assert len(tests) == 16 and all(x['status'] == 'pass' for x in tests)
    current = rows(job)
    loads = [f'{kind} {batch}' for kind in ['decode', 'verify'] for batch in [1, 16, 32]]
    assert set(current) == {(load, layer) for load in loads for layer in ['sliding', 'full']}
    assert all(not row['flags'] and row['median_ns'] > 0 and row['baseline_ns'] > 0
               for row in current.values())
    load_rows = []
    for load in loads:
        ref = sum(current[(load, layer)]['baseline_ns']*weight for layer, weight in [('sliding', 46), ('full', 16)])
        candidate = sum(current[(load, layer)]['median_ns']*weight for layer, weight in [('sliding', 46), ('full', 16)])
        load_rows.append(dict(load=load, reference_weighted_us=ref/1000,
                              candidate_weighted_us=candidate/1000, speedup=ref/candidate))
    reconstructed = math.exp(sum(math.log(x['speedup']) for x in load_rows)/6)
    assert math.isclose(reconstructed, result['speedup'], rel_tol=1e-12)
    history, hashes = {}, {}
    for name in ['A2-11', 'A2-13']:
        path = V2/(name+'-official-result.json')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        history[name] = json.loads(path.read_bytes())
    comparisons = []
    old = rows(history['A2-13'])
    for key, row in current.items():
        previous = old[key]
        normalized_ratio = (row['median_ns']/row['baseline_ns'])/(previous['median_ns']/previous['baseline_ns'])
        comparisons.append(dict(load=key[0], layer=key[1], changed=key in TARGETS,
            official_candidate_us=row['median_ns']/1000, same_run_reference_us=row['baseline_ns']/1000,
            same_run_paired_speedup=row['paired_ratio'], A2_13_candidate_us=previous['median_ns']/1000,
            A2_13_reference_us=previous['baseline_ns']/1000,
            reference_normalized_cross_run_latency_reduction_percent=100*(1-normalized_ratio)))
    output = dict(job_id=job['id'], mode=job['mode'], passed=True,
        source_sha256=request['source_sha256'], published_result_sha256=hashlib.sha256(raw).hexdigest(),
        official_speedup=result['speedup'], official_speedup_ci=result['speedup_ci'],
        reconstructed_speedup=reconstructed, official_score_us=result['score_us'],
        official_reference_us=result['baseline_us'], loads=load_rows, shapes=comparisons,
        historical_receipt_hashes=hashes,
        historical_score_deltas_percent={name: 100*(result['speedup']/value['result']['speedup']-1)
                                         for name, value in history.items()},
        qualifications=['Cross-run reference normalization reduces machine drift but is not a paired new-versus-parent experiment.',
                        'Historical A2-11 comparison also includes the pre-existing A2-13 B32 decode QK change.',
                        'Overlapping marginal score intervals are not a significance test of the difference.',
                        'A2-11 remains selected; no ranked entry, adoption, additional Modal job or quota-filling duplicate.'])
    (HERE/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ['job_id', 'official_speedup', 'official_speedup_ci',
          'reconstructed_speedup', 'historical_score_deltas_percent']}, indent=2))
    print(json.dumps([x for x in comparisons if x['changed']], indent=2))


if __name__ == '__main__':
    main()
