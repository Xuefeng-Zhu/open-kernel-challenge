"""Reproduce the bounded fusion-series report from saved terminal receipts only."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ledger = json.loads((HERE/'attention-cycle2-budget.json').read_text())
    rows = []
    for name in ['A2-05', 'A2-06', 'A2-07', 'A2-08']:
        entry = next(r for r in ledger['requests'] if r['experiment'] == name)
        assert entry['state'] == 'terminal', 'Never summarize a pending job as completed'
        a = json.loads((HERE/f'{name}-assessment.json').read_text())
        receipt = json.loads((HERE/f'{name}-official-result.json').read_text())
        assert receipt['id'] == entry['job_id'] == a['job_id']
        source = HERE/Path(entry['candidate']).name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['source_sha256']
        result = receipt.get('result') or {}
        benches = [r for r in result.get('records', []) if r['event'] == 'benchmark']
        rows.append(dict(experiment=name, job_id=entry['job_id'], source=source.name,
            source_commit=entry['source_commit'], source_sha256=entry['source_sha256'],
            passed=entry.get('passed'), score=entry.get('speedup'), ci=entry.get('speedup_ci'),
            prerequisite_count=len(a['test_records']), timed_shapes=len(benches),
            timing_sample_counts=[dict(shape=[r['case']['load'], r['case']['layer']],
                candidate=len(r['runs_ns']), baseline=len(r['baseline_runs_ns']),
                ratios=len(r['ratios'])) for r in benches],
            targets=[r for r in a.get('shape_rows', []) if r['target']],
            controls_over5pct=a.get('unchanged_controls_over5pct_latency_regression'),
            aggregate_ci_overlap=a.get('aggregate_ci_overlap'),
            target_goal_met=a.get('target_goal_met'), evaluator=result.get('evaluator'),
            gpu=result.get('gpu'), machine=result.get('machine')))
    initial = json.loads((HERE/'A2-04-assessment.json').read_text())
    selected = ledger['selection']
    report = dict(experiments=rows, selected=selected,
        starting_score=initial['score'], starting_job=initial['job_id'],
        selected_relative_to_start=selected['score']/initial['score']-1,
        user_attention_checks_used=len(ledger['requests']),
        user_attention_checks_left=ledger['remaining'],
        actual_platform_quota=ledger['last_platform_quota'],
        quota_observed_utc=ledger['quota_observed_utc'],
        all_admitted_requests_terminal=all(r['state']=='terminal' for r in ledger['requests']),
        uncertain_requests=sum(r['state'] in ('sending','uncertain') for r in ledger['requests']),
        no_modal_launch_in_series=True, no_paid_or_scored_operation=True,
        model_configuration='Not exposed to the execution tools; no claim that a model switch caused gains')
    lines = ['# Bounded attention prologue-fusion results', '',
        f"Selected `{selected['source']}`, source `{selected['source_commit']}`.",
        f"Official off-board job `{selected['official_job']}`: **{selected['score']:.8f}x**, CI {selected['ci']}.",
        f"Starting A2-04: {initial['score']:.8f}x; selected point change {100*report['selected_relative_to_start']:+.3f}%.",
        'Separate jobs are not directly paired trials of our candidates. Each is paired against SGLang.', '',
        '| Experiment | Job | Official score | Target normalized latency cut |',
        '|---|---|---:|---|']
    for row in rows:
        cuts=', '.join(f"{r['load']}/{r['layer']}: {100*r['normalized_latency_cut']:.2f}%" for r in row['targets'])
        score=f"{row['score']:.8f}x" if row['passed'] else 'failed'
        lines.append(f"| [{row['experiment']}]({row['experiment']}-RESULT.md) | {row['job_id']} | {score} | {cuts} |")
    lines += ['', '## Method and exact limits', '',
        'Each passing run contains16 prerequisite cases and12 timed shapes. Native sample counts',
        'per shape are recorded in the accompanying JSON; they count timing sample entries,',
        'not a proven total number of graph replays. Each run has11shapes with60samples',
        'and fullverify32 with56samples:716entries per side. All service-returned entries',
        'are retained without filtering by us. Raw/calibrated times and ratios stay separate.',
        'Score uses six loads, each46*sliding+16*full, then geometric mean(SGLang/ours).',
        'Profiles are separate instrumented runs with PDL overlap, not additive walltime.',
        'No register/spill/shared-memory/counter measurements are exposed for these kernels.', '',
        'A2-05 new path: B1/L8/sliding/page1/scale1. A2-06 adds B16/B32 sliding',
        'plus three native prerequisites (mixed lengths, scale1/64, soft inputs), all page1.',
        'A2-07 adds fullB1/L8, including the1,048,576-context/page64 prerequisite.',
        'A2-08 targets fullB1/L1 with the supplied actual FA3 scheduler, page1/scale1.',
        'For a failed experiment, its full intended coverage must not be inferred as passing.',
        'Explicit same-address A-B-A CUDA graphs, untouched-old-cache stress and concurrent',
        'invocations were not separately tested for these new fused sources. CPU models do',
        'check these invariants but do not execute Triton/FA3. E2 fallback shared-ticket',
        'concurrency limitations remain. Historical Modal replay tests use different sources.', '',
        'Initial local A2-05 preparation incorrectly attributed explicit saturation to the',
        'reference. The extra clamp was removed before upload; official direct-cast semantics',
        'and finite CPU oracle checks were restored. Original preparation/history is retained.', '',
        '## Budget and reproduction', '',
        f"Attention cycle: {len(ledger['requests'])}/10 accepted, {ledger['remaining']} remain; old six-cycle is separate.",
        f"Platform quota at {ledger['quota_observed_utc']}: {ledger['last_platform_quota']}.",
        'Projection remains at five authorized checks; no projection request in this series.',
        'No paid compute, ranked entry, credentials, or Modal launch. Conservative historical',
        'Modal accounting remains$1.708917914/$2; previous shutdown evidence is retained.', '',
        '```sh', '.venv/bin/python reviews/attention-v2/summarize_fusion_series.py', '```', '',
        'Per-experiment cards list source hashes, exact commands and CPU/native evidence.',
        'Original source commit IDs map to reviewed private-export commits in',
        '`analysis/PUSH_EXPORT_MAP.json`. Model configuration is not exposed to tools.']
    (HERE/'FUSION_SERIES_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'FUSION_SERIES_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['selected_relative_to_start','user_attention_checks_used',
        'user_attention_checks_left','all_admitted_requests_terminal','uncertain_requests']}))


if __name__ == '__main__':
    main()
