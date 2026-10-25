"""Push only pre-reviewed existing private experiment branches; no auth setup/retry."""
import datetime,json,subprocess,sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    branch=sys.argv[1]
    assert branch in {'attention-fa3-sanitized','projection-pipeline-preflight'}
    remote=subprocess.check_output(['git','remote','get-url','origin'],text=True).strip()
    assert remote=='https://github.com/Xuefeng-Zhu/open-kernel-challenge.git','Unexpected remote; no push'
    repo=json.loads(subprocess.check_output(['gh','repo','view','Xuefeng-Zhu/open-kernel-challenge','--json','visibility,nameWithOwner'],text=True,timeout=20))
    assert repo=={'nameWithOwner':'Xuefeng-Zhu/open-kernel-challenge','visibility':'PRIVATE'},'Expected private repository; no push'
    tip=subprocess.check_output(['git','rev-parse',branch],text=True).strip()
    old=subprocess.check_output(['git','ls-remote','--heads','origin',branch],text=True,timeout=20).strip().split()
    assert len(old)==2 and old[1]=='refs/heads/'+branch,'Expected existing remote branch; inspect before push'
    subprocess.run(['git','merge-base','--is-ancestor',old[0],tip],check=True)
    subprocess.run(['git','push','origin',branch+':refs/heads/'+branch],check=True,timeout=55)
    actual=subprocess.check_output(['git','ls-remote','--heads','origin',branch],text=True,timeout=20).strip().split()[0]
    assert actual==tip,'Push outcome requires investigation; never blindly retry'
    r=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branch=branch,old_remote_sha=old[0],remote_sha=actual,visibility='PRIVATE',force=False)
    p=Path('local-notes/auto-push-receipts.local.json');items=json.loads(p.read_text()) if p.exists() else [];items.append(r);p.write_text(json.dumps(items,indent=2)+'\n')
    print(json.dumps(r))


if __name__ == "__main__":
    main()
