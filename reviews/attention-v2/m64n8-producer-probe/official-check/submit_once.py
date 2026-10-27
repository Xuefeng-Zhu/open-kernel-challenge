"""One frozen official mode=check request; never retry an uncertain POST."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "candidate.py"
EXPECTED_SHA = "96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970"
SOURCE_COMMIT = "04447c1cc8f64fdb489b74e1e942a64e0fae3cbb"
TERMINAL = {"done", "failed", "rejected", "duplicate", "cancelled", "canceled", "expired"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    cli = Path(sys.argv[1]).resolve(strict=True)
    if (HERE / "request-receipt.json").exists():
        raise RuntimeError("Request already reserved; inspect outcome instead of repeating")
    raw = CANDIDATE.read_bytes()
    if CANDIDATE.is_symlink() or hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
        raise ValueError("Frozen source mismatch")
    compile(raw, str(CANDIDATE), "exec")
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip())
    relative = CANDIDATE.relative_to(repo).as_posix()
    if raw != subprocess.check_output(["git", "show", SOURCE_COMMIT + ":" + relative], cwd=repo):
        raise ValueError("Committed source mismatch")

    def read(*args):
        result = subprocess.run([str(cli), *args], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=60, check=True, cwd=repo)
        return json.loads(result.stdout)

    jobs = read("jobs", "--json")
    active = [j for j in jobs if j["status"] not in TERMINAL]
    save("active-jobs-admission.json", {"observed_utc": now(), "rows_seen": len(jobs), "active_jobs": active})
    if active:
        raise RuntimeError("Another official measurement is active; no POST")
    problems = read("problems", "--json")
    quota = next(p["limits"] for p in problems["problems"] if p["name"] == "muse_attention_decode")
    save("quota-admission.json", {"observed_utc": now(), "problem": "muse_attention_decode", "limits": quota})
    if quota["checks"]["left"] < 1:
        raise RuntimeError("No free check quota; no POST")

    receipt = {"state": "sending", "mode": "check", "problem": "muse_attention_decode",
               "candidate": relative, "source_sha256": EXPECTED_SHA, "source_commit": SOURCE_COMMIT,
               "created_utc": now(), "quota_before": quota, "active_jobs_before": 0,
               "expected_monetary_cost_usd": 0, "formal_entry": False, "Modal_apps_created": 0,
               "POST_attempts": 1, "job_id": None}
    save("request-receipt.json", receipt)
    try:
        result = subprocess.run([str(cli), "check", relative, "--no-wait", "--json"],
                                cwd=repo, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=120)
    except subprocess.TimeoutExpired:
        receipt.update(state="uncertain", reason="CLI timeout; reconcile account jobs before any repeat")
        save("request-receipt.json", receipt)
        raise
    (HERE / "accept.stdout.txt").write_text(result.stdout)
    (HERE / "accept.stderr.txt").write_text(result.stderr)
    try:
        data = json.loads(result.stdout)
    except ValueError:
        data = None
    job = data.get("job_id") if isinstance(data, dict) else None
    receipt.update(cli_exit_code=result.returncode, observed_utc=now())
    if job:
        save("accept.json", data)
        receipt.update(state="accepted", job_id=job)
    else:
        receipt.update(state="uncertain", reason="No admitted ID; reconcile account jobs, do not repeat")
    save("request-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
