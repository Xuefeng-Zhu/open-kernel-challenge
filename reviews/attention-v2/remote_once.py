"""One approved attention check; no automatic POST retries, no publishing."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CLI = PROJECT/'bin/okc'
POLICY = ROOT/'remote-policy.json'
TERMINAL = {'done', 'failed', 'rejected', 'duplicate'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(obj):
    temp = POLICY.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2)+'\n')
    temp.replace(POLICY)


def cli(*args):
    p = subprocess.run([str(CLI), *args], cwd=PROJECT, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    try:
        value = json.loads(p.stdout)
    except ValueError:
        value = None
    return p, value


def quota():
    p, data = cli('problems', '--json')
    if p.returncode or not isinstance(data, dict):
        raise RuntimeError('Read-only quota request failed; no admission')
    return next(x['limits'] for x in data['problems'] if x['name']=='muse_attention_decode')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['send', 'poll'])
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text())
    if args.action == 'send':
        if policy.get('remote_check_approval') != 'approved_one_normalized_only':
            raise RuntimeError('No exact source-transfer approval')
        if policy['maximum_authorized_remote_checks'] != 1 or policy['requests']:
            raise RuntimeError('Single allowance already reserved/used; never retry blindly')
        source = ROOT/'submission.py'
        data = source.read_bytes()
        relative = source.relative_to(PROJECT).as_posix()
        commit = subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=PROJECT).decode().strip()
        if data != subprocess.check_output(['git','show',commit+':'+relative],cwd=PROJECT):
            raise RuntimeError('Candidate differs from committed source')
        sha = hashlib.sha256(data).hexdigest()
        if sha != policy['candidate_sha256'] or not data.startswith(b'# okc: problem=muse_attention_decode\n'):
            raise RuntimeError('Only the approved attention source is eligible')
        import re
        original = (ROOT/'paste.original.txt').read_text()
        fixed, count = re.subn(r'\[([A-Za-z_]\w*(?:\.\w+)+)\]\(http://\1\)',r'\1',original)
        if count != 5 or fixed.encode() != data:
            raise RuntimeError('Only the five approved link repairs are permitted')
        rules = subprocess.run(['curl','-fsSL','https://openkernelchallenge.dev/api/v1/rules'],capture_output=True,text=True)
        if rules.returncode or json.loads(rules.stdout)['version'] != policy['rules_version']:
            raise RuntimeError('Rules changed/unavailable; do not accept new terms')
        before = quota()
        if before['checks']['left'] < 1:
            raise RuntimeError('Attention quota exhausted')
        entry = dict(created_utc=now(),state='sending',status='unknown',job_id=None,
                     mode='check',problem='muse_attention_decode',source_commit=commit,
                     source_sha256=sha,quota_before=before)
        policy['requests'].append(entry)
        save(policy)  # Reserve BEFORE the sole admission command.
        p, reply = cli('check',relative,'--no-wait','--json')
        (ROOT/'accept.local.json').write_text(p.stdout)
        (ROOT/'accept.local.txt').write_text(p.stderr)
        job = reply.get('job_id') if isinstance(reply,dict) else None
        entry.update(cli_exit_code=p.returncode)
        if job:
            entry.update(state='accepted',status='accepted',job_id=job,accepted_utc=now())
        else:
            entry.update(state='uncertain',reconciliation='Inspect own jobs and accepted source before any retry; allowance remains reserved')
        save(policy)
        print(json.dumps(entry))
        return 0 if job else 2
    if len(policy['requests']) != 1 or not policy['requests'][0].get('job_id'):
        raise RuntimeError('No accepted ID; reconcile own jobs read-only')
    entry=policy['requests'][0]
    p, reply=cli('status',entry['job_id'],'--json')
    (ROOT/'status-reply.local.json').write_text(p.stdout)
    if not isinstance(reply,dict):
        raise RuntimeError('Status read failed; existing reservation unchanged')
    if reply.get('job_id') == entry['job_id'] and reply.get('status') in {'received','queued','running'} and 'result' not in reply:
        entry.update(status=reply['status'],last_checked_utc=now(),status_cli_exit_code=p.returncode)
    else:
        if reply.get('id') != entry['job_id'] or reply.get('mode') != 'check':
            raise RuntimeError('Status identity/mode mismatch; do not resend')
        (ROOT/'job.local.json').write_text(json.dumps(reply,indent=2)+'\n')
        entry.update(status=reply['status'],last_checked_utc=now(),status_cli_exit_code=p.returncode)
        if reply['status'] in TERMINAL:
            result=reply.get('result') or {}
            entry.update(state='terminal',passed=result.get('passed'),speedup=result.get('speedup'),
                         speedup_ci=result.get('speedup_ci'),quota_after=quota())
    save(policy)
    print(json.dumps(entry))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
