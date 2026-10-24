"""Render current budget/results from preserved evidence; performs no network calls."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def commit_for(path):
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", path], cwd=ROOT).decode().strip() or None


def main():
    ledger = json.loads((ROOT / "experiments/remote-budget.json").read_text())
    admitted = [x for x in ledger["requests"] if x["state"] != "not_admitted"]
    terminal = [x for x in admitted if x["state"] == "terminal"]
    uncertain = [x for x in admitted if x["state"] in {"sending", "uncertain"}]
    remaining = ledger["maximum_new_experiments"] - len(admitted)
    lines = ["# Remote experiment results and budget", "",
             f"**{len(admitted)}/10 new requests admitted/reserved; {remaining} remain.** "
             f"{len(terminal)} terminal, {len(admitted)-len(terminal)} pending, {len(uncertain)} uncertain.",
             "Prior baseline job `e5a5b2d20da54187` is excluded. Only off-board checks and $0 spend are authorized; "
             "no scored/final submission, new credentials or material agreement.", "",
             "| Experiment | Job | State | Correctness | Speedup vs SGLang (95% interval) | vs original, normalized |",
             "|---|---|---|---|---|---|"]
    if ledger.get("admission_paused"):
        lines.insert(4, "**New uploads are paused by automatic approval review.** Planned R06 was denied before process creation; no POST or budget reservation occurred. See `DENIED_R06.md`. Existing-job monitoring remained allowed.")
        lines.insert(5, "")
    best = None
    remotes = []
    for entry in admitted:
        path = ROOT / "results" / ("remote-" + entry["experiment"] + ".json")
        summary = json.loads(path.read_text()) if path.is_file() and entry["state"] == "terminal" else None
        result = summary["result"] if summary else {}
        ok = result.get("passed")
        tests = "9/9 + 30/30" if ok else ("failed" if summary else "pending")
        speed = "—"
        relative = "—"
        if result.get("speedup"):
            ci = result.get("speedup_ci")
            speed = f"{result['speedup']:.4f}× [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else str(result["speedup"])
            relative = f"{summary['speedup_over_original_baseline']:.4f}×"
        lines.append(f"| {entry['experiment']} | `{entry.get('job_id')}` | {entry['status']} | {tests} | {speed} | {relative} |")
        record = "experiments/remote-" + entry["experiment"] + ".md"
        remotes.append(dict(id=entry["experiment"], source_commit=entry["source_commit"],
                            submission_sha256=entry["submission_sha256"], candidate=entry["candidate"],
                            job_id=entry.get("job_id"), mode="check", status=entry["status"],
                            result_commit=commit_for(record), record=record,
                            results=path.relative_to(ROOT).as_posix() if summary else None,
                            timings=path.with_suffix(".csv").relative_to(ROOT).as_posix() if summary and result.get("passed") else None))
        if ok and result.get("checks", {}).get("ok") and not result.get("flags"):
            if best is None or result["speedup"] > best["result"]["speedup"]:
                best = summary
    lines += ["", "The original starter scored 0.9076063758×. Comparisons to it are reference-normalized "
              "across separate runs, not paired same-input measurements. Official within-run SGLang pairings "
              "and their intervals are authoritative. Missing results/timings are not inferred from CPU models.", "",
              "Ledger: `experiments/remote-budget.json`. Exact source/result commits and per-shape evidence: "
              "`experiments/index.json`. Uncertain requests stay reserved until reconciled; never blindly retry a POST."]
    if ledger.get("last_account_quota"):
        lines += ["", "Latest account quota (distinct from this authorization): `" + json.dumps(ledger["last_account_quota"]) + "`."]
    (ROOT / "REMOTE_RESULTS.md").write_text("\n".join(lines) + "\n")
    active = [x for x in admitted if x['state'] != 'terminal']
    status = ["# Current experiment checkpoint", "",
              f"{len(admitted)}/10 new checks admitted/reserved; {len(terminal)} terminal, "
              f"{len(active)} pending, {len(uncertain)} uncertain, {remaining} allowances remain.",
              "$0; no scored/public/final entry. Original baseline is separately preserved.", "",
              "| Experiment | Job | Status | Official speedup vs SGLang |",
              "|---|---|---|---|"]
    for e in admitted:
        value = f"{e['speedup']:.7f}x" if e.get('speedup') else 'unmeasured'
        status.append(f"| {e['experiment']} | `{e.get('job_id')}` | {e['status']} | {value} |")
    status += ["", "Exact source/result hashes, confidence intervals, correctness, per-shape times and "
               "experiment qualifications: `experiments/index.json`, `REMOTE_RESULTS.md`, `BEST.md`.",
               "The ledger records each read timestamp and quota; queued jobs are not completed results."]
    if ledger.get('admission_paused'):
        status += ["", "New uploads paused by automatic approval review; see `DENIED_R06.md` and ledger."]
    elif ledger.get('denied_operation_resolutions'):
        status += ["", "The historical R06 denial was resolved by specific new human approval; its single "
                   "retry was accepted. Do not repeat it; preserve `DENIED_R06.md` history/resolution."]
    (ROOT / 'STATUS.md').write_text('\n'.join(status) + '\n')
    resume = ["# Resume this same project and exact accepted jobs", "",
              f"Budget {len(admitted)}/10, {remaining} remain, {len(uncertain)} uncertain; $0, check mode only.",
              "Read `STATUS.md`, `BEST.md`, `REMOTE_RESULTS.md` and `experiments/remote-budget.json`.",
              "Never resend admitted experiments. At most two active jobs; inspect terminal results before "
              "choosing a meaningful replacement. No paid compute, scored/public/final entry, new credentials "
              "or material agreement. Uncertain requests stay reserved until reconciled read-only.", ""]
    if active:
        resume += ["## Already accepted — monitor to terminal", "", "```sh"]
        resume += [f"python3 remote_check.py poll {e['experiment']}  # {e.get('job_id')}" for e in active]
        resume += ["```", "", "Official CLI `status <exact-ID> --wait --json` monitors may also be running; "
                   "these are read-only, not replacement jobs. Do not end iteration merely because jobs queue."]
    else:
        resume += ["All admitted jobs are terminal. Choose further work only from actual evidence and the "
                   "remaining authorization; stop early if no meaningful hypothesis or a real blocker exists."]
    resume += ["", "For terminal results: `python3 summarize_check.py <experiment>`, inspect all correctness "
               "and timing/profile records, record failures too, then independently commit sanitized evidence. "
               "`python3 update_progress.py` refreshes derived status offline. Push only sanitized files to "
               "the existing private repository; preserve original baseline and immutable experiment sources.", "",
               "Historical R06 denial and its sole approved retry: `DENIED_R06.md`. Never retry that upload again. "
               "Any new denial must be honored without alternate transfer routes.", "",
               "Candidate-specific hypotheses and CPU/GPU limitations are in `experiments/`. R06 pooler split8 "
               "regressed and must not be composed into the best. QKV split8 stays local-only after that evidence. "
               "Current remaining hypotheses: router split-count isolation; single-row SIMT pooler; combine only "
               "measured winning regions in a separately committed/checked final candidate if justified.", "",
               "The unrelated attention attachment review remains ignored/local-only under "
               "`local-notes/attention-review/REVIEW.md`; never upload/commit it as part of this project."]
    (ROOT / 'CONTINUE.md').write_text('\n'.join(resume) + '\n')
    index_path = ROOT / "experiments/index.json"
    index = json.loads(index_path.read_text())
    index["remote_experiments"] = remotes
    index["remote_budget"] = dict(maximum=10, admitted_or_reserved=len(admitted), remaining=remaining,
                                 prior_baseline_excluded=ledger["prior_baseline_excluded"])
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    if best:
        entry, result = best["admission"], best["result"]
        note = "The interval is above 1; this off-board trial measured an improvement over SGLang." if result["speedup_ci"][0] > 1 else "The interval includes 1; this is parity, not a demonstrated win over SGLang."
        (ROOT / "BEST.md").write_text(f"""# Best completed, correctness-passing observed candidate

Candidate: `{entry['candidate']}`; source commit `{entry['source_commit']}`.
SHA256: `{entry['submission_sha256']}`. Job `{entry['job_id']}`, mode check.

Official speedup **{result['speedup']:.6f}× SGLang**, 95% interval
**[{result['speedup_ci'][0]:.6f}, {result['speedup_ci'][1]:.6f}]**.
{note}
9/9 prerequisites and 30/30 scored-shape correctness checks pass. This is the highest
observed passing score so far, not a claim that every difference between candidates
is statistically resolved or that it is a final leaderboard placement.

Reference-normalized ratio over our original starter: **{best['speedup_over_original_baseline']:.6f}×**;
that is a cross-run comparison. Exact timings/environment/risks are in
`results/remote-{entry['experiment']}.json` and its CSV/experiment record.

The original `submission.py` remains unchanged and separately preserved. Job states
and the authorized remaining budget are in `REMOTE_RESULTS.md`; no scored entry or
paid resource is authorized.
""")
        if entry['experiment'] == 'R05-wide-layout':
            with (ROOT / "BEST.md").open("a") as handle:
                handle.write("""
## Candidate comparison and remaining risk

This is the highest observed aggregate point score, not a statistically established
winner over R02 or R04: their intervals overlap. R05 router m=128/256 is 11.66/12.63 µs
versus R02's 10.06/11.30 µs, so the new layout regresses those targeted cases despite
its slightly higher whole-workload score. Untouched projection ratios vary across runs.
R02's contiguous larger-M path and R04's low-M native path are the better-supported
components for a future controlled hybrid; that native/wide combination has not been
GPU-tested. The prepared library-only hybrid, pooler and split32 variants also lack
their own GPU results.
""")
        if ledger.get("admission_paused"):
            with (ROOT / "BEST.md").open("a") as handle:
                handle.write(f"\n{len(admitted)} new checks were admitted; {len(terminal)} are terminal. "
                             f"{remaining} allowances remain, but further uploads are paused because automatic "
                             "approval review denied R06 before its command ran. No uncertain request "
                             "or budget use resulted; see `DENIED_R06.md`.\n")
    print(f"Updated progress: {len(admitted)}/10 admitted, {len(terminal)} terminal, {remaining} remaining")


if __name__ == "__main__":
    main()
