"""Reproduce P064 static gate; never infer native correctness or latency."""
import hashlib
import json
from pathlib import Path
import re
from assess_p063 import assess

ROOT=Path(__file__).resolve().parent


def sass_instructions(text):
    return '\n'.join(match[1] for line in text.splitlines()
                     if (match:=re.match(r'^\s*/\*[0-9a-f]+\*/\s+(.+?);',line)))


def reconstructed_bf16_allocations(text):
    # Source-valued BF16 tile allocs, not uninitialized pipeline buffers.
    return re.findall(r'^\s*(%[\w]+) = ttg.local_alloc (%[\w]+) : \(tensor<(\d+x\d+)xbf16,',text,re.M)


def evaluate(parent,candidate):
    old=assess(parent);new=assess(candidate)
    by_old={v['label']:v for v in old['variants']}
    by_new={v['label']:v for v in new['variants']}
    a=sass_instructions((parent/'raw.sass').read_text())
    b=sass_instructions((candidate/'raw.sass').read_text())
    decoded_allocs=reconstructed_bf16_allocations((candidate/'encoded.ttgir').read_text())
    encoded=by_new['encoded'];baseline=by_old['encoded']
    same_raw=a==b
    gates=dict(raw_instruction_text_identical=same_raw,
        decoded_bf16_shared_staging_removed=not decoded_allocs,
        same_bf16_matrix_family=encoded['ptx_matrix_instruction_families']==baseline['ptx_matrix_instruction_families'],
        no_spills=encoded['spill_store_bytes']==encoded['spill_load_bytes']==0,
        register_reduction=encoded['registers']<baseline['registers'],
        shared_request_reduction=encoded['compiler_shared_request_bytes']<baseline['compiler_shared_request_bytes'],
        instruction_site_reduction=encoded['static_sass_instruction_sites']<baseline['static_sass_instruction_sites'])
    passed=all(gates.values())
    return dict(parent=old,candidate=new,gates=gates,compiler_gate_pass=passed,
        raw_instruction_text_sha256=hashlib.sha256(b.encode()).hexdigest(),
        decoded_bf16_source_allocations=decoded_allocs,
        decision='Native review required before check' if passed else 'Reject P064 for official admission: decoded shared staging remains.',
        native_correctness_verified=False,gpu_performance_measured=False,selected='P046',
        caveat='Counts are static instruction sites, not executed counts, occupancy, traffic or latency. No automatic GPU admission.')


if __name__=='__main__':
    result=evaluate(ROOT/'evidence/p063',ROOT/'evidence/p064')
    (ROOT/'results/projection-p064-assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('gates','compiler_gate_pass','decision')},indent=2))
