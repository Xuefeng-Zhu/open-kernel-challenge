"""Auditable approved-budget wrapper. Only off-board checks; never automatic POST retries."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "experiments/remote-budget.json"
CLI = ROOT / "bin/okc"
TERMINAL = {"done", "failed", "rejected", "duplicate"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def cli(*args):
    p = subprocess.run([str(CLI), *args], cwd=ROOT, capture_output=True, text=True)
    try:
        data = json.loads(p.stdout)
    except ValueError:
        data = None
    return p, data


def quota():
    p, data = cli("problems", "--json")
    if p.returncode or data is None:
        raise RuntimeError("Read-only quota request failed; no POST will be sent")
    return next(x["limits"] for x in data["problems"] if x["name"] == "muse_bf16_proj")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    send = sub.add_parser("send")
    send.add_argument("experiment")
    send.add_argument("candidate", type=Path)
    poll = sub.add_parser("poll")
    poll.add_argument("experiment")
    sub.add_parser("quota")
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text())
    entries = ledger["requests"]
    if args.action == "quota":
        ledger["last_account_quota"] = quota()
        ledger["last_account_quota_utc"] = now()
        save(LEDGER, ledger)
        print(json.dumps(ledger["last_account_quota"]))
        return 0
    if args.action == "send":
        if ledger.get("admission_paused"):
            raise RuntimeError("New uploads paused after automatic approval rejection; resolve with materially new trusted authorization")
        if any(x["experiment"] == args.experiment for x in entries):
            raise RuntimeError("Experiment already journaled; reconcile it, never blindly resend")
        if any(x["state"] in {"sending", "uncertain"} for x in entries):
            raise RuntimeError("Uncertain request reserved; inspect account jobs before any new POST")
        counted = [x for x in entries if x["state"] != "not_admitted"]
        if len(counted) >= ledger["maximum_new_experiments"]:
            raise RuntimeError("New-experiment authorization exhausted")
        active = [x for x in entries if x.get("job_id") and x.get("status") not in TERMINAL]
        if len(active) >= ledger["max_active_jobs"]:
            raise RuntimeError("Bounded active-job limit reached")
        path = args.candidate.resolve()
        relative = path.relative_to(ROOT).as_posix()
        if not relative.startswith("candidates/") or not path.read_text().startswith("# okc: problem=muse_bf16_proj\n"):
            raise RuntimeError("Only a committed dense-projection candidate is eligible")
        source_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT).decode().strip()
        committed = subprocess.check_output(["git", "show", source_commit + ":" + relative], cwd=ROOT)
        source = path.read_bytes()
        if source != committed:
            raise RuntimeError("Candidate differs from its source commit; commit before admitting it")
        rules_reply = subprocess.run(
            ["curl", "-fsSL", "https://openkernelchallenge.dev/api/v1/rules"],
            capture_output=True, text=True)
        if rules_reply.returncode or json.loads(rules_reply.stdout).get("version") != ledger["rules_version"]:
            raise RuntimeError("Rules version changed/unavailable; no POST or new agreement")
        before = quota()
        if before["checks"]["left"] < 1:
            raise RuntimeError("Official check quota exhausted")
        entry = dict(experiment=args.experiment, candidate=relative, source_commit=source_commit,
                     submission_sha256=hashlib.sha256(source).hexdigest(), mode="check",
                     created_utc=now(), quota_before=before, state="sending", status="unknown", job_id=None)
        entries.append(entry)
        save(LEDGER, ledger)  # Reserve BEFORE the sole POST; interruption is an uncertain outcome.
        p, data = cli("check", relative, "--no-wait", "--json")
        (ROOT / "results" / (args.experiment + "-accept.local.json")).write_text(p.stdout)
        (ROOT / "results" / (args.experiment + "-accept.local.txt")).write_text(p.stderr)
        entry["cli_exit_code"] = p.returncode
        job_id = data.get("job_id") if isinstance(data, dict) else None
        if job_id:
            entry.update(state="accepted", status="accepted", job_id=job_id, accepted_utc=now())
        else:
            entry.update(state="uncertain", reconciliation="Inspect own jobs read-only before any retry; reservation remains charged")
        save(LEDGER, ledger)
        print(json.dumps(entry))
        return 0 if job_id else 2
    entry = next(x for x in entries if x["experiment"] == args.experiment)
    if not entry.get("job_id"):
        raise RuntimeError("No accepted job ID; reconcile jobs first")
    p, data = cli("status", entry["job_id"], "--json")
    (ROOT / "results" / (args.experiment + "-status-reply.local.json")).write_text(p.stdout)
    if not isinstance(data, dict):
        raise RuntimeError("Read-only status failed; existing job reservation unchanged")
    # CLI may return a minimal acknowledgement while the accepted job is dispatching.
    # Its identity must still match; never treat this as a result or a new admission.
    if (data.get("job_id") == entry["job_id"]
            and data.get("status") in {"received", "queued", "running"}
            and "result" not in data and data.get("mode", "check") == "check"):
        entry.update(status=data["status"], status_cli_exit_code=p.returncode,
                     last_checked_utc=now())
        save(LEDGER, ledger)
        print(json.dumps(entry))
        return 0
    if data.get("mode") != "check" or data.get("id") != entry["job_id"]:
        (ROOT / "results" / (args.experiment + "-status-error.local.json")).write_text(p.stdout)
        raise RuntimeError("Job identity/mode mismatch")
    save(ROOT / "results" / (args.experiment + "-job.local.json"), data)
    entry.update(status=data["status"], status_cli_exit_code=p.returncode, last_checked_utc=now())
    if data["status"] in TERMINAL:
        result = data.get("result") or {}
        entry.update(state="terminal", quota_after=quota(), passed=result.get("passed"),
                     speedup=result.get("speedup"), speedup_ci=result.get("speedup_ci"))
    save(LEDGER, ledger)
    print(json.dumps(entry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
