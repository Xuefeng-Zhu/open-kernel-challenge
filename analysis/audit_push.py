"""Read-only outgoing Git history audit; never prints matched material."""
import json,re,subprocess,sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    ref,base=sys.argv[1:]
    patterns=[rb'\b(?:ak|as)-[A-Za-z0-9]{20,}',rb'gh[pousr]_[A-Za-z0-9]{20,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'(?i)authorization\s*[:=]\s*["\x27]?Bearer\s+[A-Za-z0-9._~+-]{20,}']
    def safe_path(name):
     assert not any(x in name.split('/') for x in ['local-notes','.venv','.okc','.aws','.config']) and not name.endswith(('.local.json','.local.txt','.pem','.key')) and 'modal-auth' not in name,'Private path: '+name
    objects={}
    for line in subprocess.check_output(['git','rev-list','--objects',ref,'^'+base],text=True).splitlines():
     parts=line.split(' ',1)
     if len(parts)==2:objects[parts[0]]=parts[1];safe_path(parts[1])
    for path in subprocess.check_output(['git','diff','--name-only',base,ref],text=True).splitlines():safe_path(path)
    if not objects:
        print(json.dumps(dict(branch=ref,tip=subprocess.check_output(['git','rev-parse',ref],text=True).strip(),historical_blobs_reviewed=0,bytes=0,private_paths_absent=True,sensitive_scan_passed=True)))
        return
    info=subprocess.check_output(['git','cat-file','--batch-check'],input=('\n'.join(objects)+'\n').encode()).decode().splitlines()
    ids=[s.split()[0] for s in info if s.split()[1]=='blob']
    raw=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join(ids)+'\n').encode());at=total=0
    for oid in ids:
     end=raw.index(b'\n',at);header=raw[at:end].split();n=int(header[2]);body=raw[end+1:end+1+n];at=end+n+2;name=objects[oid]
     assert header[0].decode()==oid and len(body)==n
     assert n<25*1024*1024,'Oversized: '+name
     assert not any(re.search(p,body) for p in patterns),'Sensitive pattern; inspect without printing: '+name
     total+=n
    print(json.dumps(dict(branch=ref,tip=subprocess.check_output(['git','rev-parse',ref],text=True).strip(),historical_blobs_reviewed=len(ids),bytes=total,private_paths_absent=True,sensitive_scan_passed=True)))


if __name__ == "__main__":
    main()
