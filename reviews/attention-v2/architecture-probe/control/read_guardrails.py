"""Read-only preflight. Uses existing supported Modal config, never reads secrets.

This tool cannot authorize a launch: Modal 1.6.2 billing summary does not expose
remaining free credits or the net spend limit. Live dashboard evidence is still
required from the human while browser control is prohibited.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = args.original_workspace.resolve()
    auth = original / 'local-notes/modal-auth.toml'
    assert auth.is_file() and not auth.is_symlink()
    assert auth.stat().st_uid == os.getuid() and auth.stat().st_mode & 0o777 == 0o600
    blocked = ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'MODAL_OAUTH_REFRESH_TOKEN',
               'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET']
    assert not any(os.environ.get(k) for k in blocked)
    subprocess.run(['git', 'check-ignore', '-q', 'local-notes/modal-auth.toml'],
                   cwd=original, check=True)
    assert not subprocess.check_output(['git', 'ls-files', 'local-notes/modal-auth.toml'], cwd=original)
    env = os.environ.copy()
    env.update(MODAL_CONFIG_PATH=str(auth), MODAL_PROFILE='xuefeng-zhu')
    result = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'workspace': 'xuefeng-zhu', 'credential_copied': False,
              'config_regular_owned_mode600_ignored_untracked': True,
              'gpu_launch_attempted': False, 'browser_used': False,
              'launch_admitted': False,
              'missing_live_guardrails': ['remaining_free_credit_balance', 'net_spend_limit_usd'],
              'blocker_kind': 'supported_read_capability_unavailable_pending_human_evidence'}
    for name, cmd in [('billing', ['billing', 'summary', '--json']),
                      ('apps', ['app', 'list', '--env', 'main', '--json'])]:
        child = subprocess.run([str(original/'.venv/bin/python'), '-m', 'modal', *cmd],
                               capture_output=True, text=True, timeout=20, env=env)
        # CLI stderr can contain a credential identifier in expiry warnings;
        # never print, persist or forward it, including on failure.
        if child.returncode:
            result.update(state='read_failed_no_retry', failed_action='modal '+' '.join(cmd),
                          failed_exit_code=child.returncode)
            break
        result[name] = json.loads(child.stdout)
    else:
        from decimal import Decimal
        result['paid_cost_zero'] = Decimal(result['billing']['billed_cost']) == 0
        result['all_apps_stopped_tasks_zero'] = all(
            x['state'] == 'stopped' and int(x['tasks']) == 0 for x in result['apps'])
        result['state'] = 'blocked_pending_fresh_credit_cap_evidence'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result.get(k) for k in ['observed_utc', 'state', 'paid_cost_zero',
          'all_apps_stopped_tasks_zero', 'launch_admitted']}, indent=2))


if __name__ == '__main__':
    main()
