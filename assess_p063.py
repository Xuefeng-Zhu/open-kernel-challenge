"""Read owned offline compiler artifacts; no GPU or performance inference."""
import hashlib
import json
import math
from pathlib import Path
import re
from collections import Counter

ROOT=Path(__file__).resolve().parent


def parse_ptxas(text):
    registers=re.search(r'Used (\d+) registers',text)
    spills=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',text)
    if not registers or not spills:
        raise ValueError('Missing actual assembler register/spill report')
    return dict(registers=int(registers[1]),stack_bytes=int(spills[1]),
                spill_store_bytes=int(spills[2]),spill_load_bytes=int(spills[3]))


def sass_ops(text):
    # Static textual instruction sites, not dynamic executions or measured work.
    out=Counter()
    for line in text.splitlines():
        match=re.match(r'^\s*/\*[0-9a-f]+\*/\s+(.+?);',line)
        if not match:continue
        words=match[1].split()
        if words[0].startswith('@'):words=words[1:]
        if not words or not re.fullmatch(r'[A-Z][A-Z0-9_.]*',words[0]):
            raise ValueError('Unrecognized disassembly instruction')
        out[words[0]]+=1
    return out


def score_scenario(assessment,selected,retained_fraction):
    if not 0 < retained_fraction <= 1:raise ValueError('Invalid retained fraction')
    targets={r['m']:r['raw_over_selected']*retained_fraction for r in assessment['targets']}
    loads={}
    for r in selected['result']['records']:
        if r.get('event')!='benchmark':continue
        c=r['case'];base=r['median_ns']*c['weight']
        old,new=loads.get(c['load'],(0,0))
        multiplier=targets[c['m']] if c['layer']=='pooler' and c['m'] in targets else 1
        loads[c['load']]=(old+base,new+base*multiplier)
    if len(loads)!=6:raise ValueError('Expected six loads')
    factor=math.exp(sum(math.log(a/b) for a,b in loads.values())/6)
    return dict(target_relative_latency=targets,score_factor=factor,
                modeled_score=selected['result']['speedup']*factor,
                assumption='Stored-byte ratio scales ALL raw latency with zero decoding cost; not a physical bound or performance forecast.')


def assess(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['gpu_execution'] or manifest['benchmark']:raise ValueError('Expected offline artifacts')
    rows=[]
    for v in manifest['variants']:
        name=v['label']
        if 'ptxas.txt' not in v['hashes']:
            raise ValueError('Assembler resource log is not hash-bound')
        for ext,h in v['hashes'].items():
            f=directory/f'{name}.{ext}'
            # Cubin is retained locally; its hash is preserved in published metadata.
            if ext=='cubin' and not f.exists():continue
            if hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError('Artifact hash mismatch: '+str(f))
        resources=parse_ptxas((directory/f'{name}.ptxas.txt').read_text())
        ptx=(directory/f'{name}.ptx').read_text()
        gir=(directory/f'{name}.ttgir').read_text()
        sass=sass_ops((directory/f'{name}.sass').read_text())
        rows.append(dict(label=name,**resources,compiler_shared_request_bytes=v['metadata']['shared'],
            ptx_matrix_instruction_families=sorted(set(re.findall(r'(?:wgmma|mma)\.[\w.]+',ptx))),
            static_sass_instruction_counts=dict(sorted(sass.items())),
            static_sass_instruction_sites=sum(sass.values()),
            shared_staging_ops={op:len(re.findall(r'\b'+re.escape(op)+r'\b',gir))
              for op in ('ttg.local_alloc','ttg.local_load','ttg.async_copy_global_to_local','ttg.async_wait','ttg.convert_layout')},
            scope='Static generated-code evidence only; not executed counts, achieved occupancy, memory traffic, correctness or timing.'))
    return dict(variants=rows,judge_codegen_identity_verified=False,
                sources=manifest['manifest']['sources'],compiler=manifest['triton'],tool_hashes=manifest['tool_sha256'])


if __name__=='__main__':
    d=ROOT/'evidence/p063'
    result=assess(d)
    a=json.loads((ROOT/'results/projection-p062-assessment.json').read_text())
    s=json.loads((ROOT/'results/remote-P2-09-pooler-column-batch2-hybrid.json').read_text())
    # P061 measured stored-byte scenario; explicit constant, not an HBM measurement.
    result['storage_only_scenario']=score_scenario(a,s,1-.23274564743041992)
    result['selected_preserved']=True
    (ROOT/'results/router-p063-offline-assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
