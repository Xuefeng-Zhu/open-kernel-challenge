"""Append related research commits to the sanitized private-export branch locally.

No network/push, worktree mutation, source-ref rewrite, force or candidate execution.
The original research branch remains authoritative for receipt source commit IDs;
the export manifest maps those IDs to content-identical exported experiments.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'attention-fa3-local'
EXPORT = 'attention-fa3-sanitized'
MANIFEST = 'analysis/PUSH_EXPORT_MAP.json'
SENSITIVE = [rb'\b(?:ak|as)-[A-Za-z0-9]{20,}', rb'gh[pousr]_[A-Za-z0-9]{20,}',
             rb'github_pat_[A-Za-z0-9_]{30,}',
             rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']


def git(*args, data=None, env=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, input=data, env=env)


def metadata(headers):
    env = dict(os.environ)
    for label, prefix in [('author', 'AUTHOR'), ('committer', 'COMMITTER')]:
        line = next(v for v in headers.decode().splitlines() if v.startswith(label+' '))
        found = re.fullmatch(label+r' (.*) <([^>]+)> (\d+ [+-]\d{4})', line)
        if not found:
            raise ValueError('Unrecognized commit metadata; do not guess')
        env['GIT_'+prefix+'_NAME'], env['GIT_'+prefix+'_EMAIL'], env['GIT_'+prefix+'_DATE'] = found.groups()
    return env


def main():
    source = git('rev-parse', SOURCE).decode().strip()
    previous = git('rev-parse', EXPORT).decode().strip()
    report = json.loads(git('show', previous+':'+MANIFEST))
    anchor = report['source_snapshot']
    if source == anchor:
        print(json.dumps(dict(export_commit=previous, source_snapshot=source, new_commits=0)))
        return
    subprocess.run(['git', 'merge-base', '--is-ancestor', anchor, source], cwd=ROOT, check=True)
    commits = git('rev-list', '--reverse', '--topo-order', source, '^'+anchor).decode().splitlines()
    mapping = report['commit_mapping']
    expected_parent = anchor
    exported_parent = previous
    with tempfile.TemporaryDirectory(prefix='okc-export-index-') as folder:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder)/'index'))
        git('read-tree', previous, env=env)
        for old in commits:
            raw = git('cat-file', 'commit', old)
            headers, message = raw.split(b'\n\n', 1)
            parents = [v[7:] for v in headers.decode().splitlines() if v.startswith('parent ')]
            if parents != [expected_parent]:
                raise ValueError('Only the reviewed linear related branch is exported; inspect merge/divergence')
            patch = git('diff-tree', '--binary', '--no-commit-id', '-p', '-r', expected_parent, old)
            names = git('diff-tree', '--no-commit-id', '--name-only', '-r', expected_parent, old).decode().splitlines()
            if MANIFEST in names or any(re.search(pattern, patch) for pattern in SENSITIVE):
                raise ValueError('Source patch changes export manifest or contains sensitive material; inspect without printing it')
            if patch:
                git('apply', '--cached', '--whitespace=nowarn', data=patch, env=env)
            tree = git('write-tree', env=env).decode().strip()
            exported_parent = git('commit-tree', tree, '-p', exported_parent,
                                  data=message, env=metadata(headers)).decode().strip()
            mapping[old] = exported_parent
            expected_parent = old
        differences = git('diff', '--name-only', source, exported_parent).decode().splitlines()
        if differences != [MANIFEST]:
            raise ValueError('Export tree differs beyond provenance manifest')
        report.setdefault('initial_source_snapshot', anchor)
        if 'running_job_at_snapshot' in report:
            report['initial_export_job_note'] = report.pop('running_job_at_snapshot')
        candidate_paths = [p for p in git('ls-tree', '-r', '--name-only', source,
                                         '--', 'reviews/attention-v2').decode().splitlines()
                           if Path(p).name.startswith('mixed_fa3') and p.endswith('.py')]
        report.update(source_snapshot=source, sanitized_snapshot=exported_parent,
                      previous_export_tip=previous, incremental_related_replay=True,
                      commit_mapping=mapping,
                      current_export_note='Runtime state is in the exported ledgers/reports, not inferred by this local exporter',
                      candidate_sha256={p: hashlib.sha256(git('show', source+':'+p)).hexdigest()
                                        for p in candidate_paths})
        blob = git('hash-object', '-w', '--stdin', data=(json.dumps(report, indent=2)+'\n').encode()).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644', blob, MANIFEST, env=env)
        tree = git('write-tree', env=env).decode().strip()
        tip = git('commit-tree', tree, '-p', exported_parent,
                  data=f'Document sanitized export provenance through {source}\n'.encode()).decode().strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', previous, tip], cwd=ROOT, check=True)
    git('update-ref', 'refs/heads/'+EXPORT, tip, previous)
    local = dict(report, export_commit=tip)
    (ROOT/'local-notes/attention-safe-export-map.local.json').write_text(json.dumps(local, indent=2)+'\n')
    print(json.dumps(dict(source_snapshot=source, export_commit=tip, related_commits_appended=len(commits),
                          original_source_ref_unchanged=True, worktree_unchanged=True)))


if __name__ == '__main__':
    main()
