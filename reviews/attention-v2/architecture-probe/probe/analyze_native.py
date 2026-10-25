"""Offline native-code audit; never imports CUDA, Triton, Modal, or candidate.

Usage: python analyze_native.py [../native_artifacts] [--output FILE]
No WGMMA is attributed to QK merely because WGMMA exists elsewhere. Dot roles
are mapped from frozen source assignments through IR location aliases and PTX
source locations. Missing/ambiguous mappings remain explicit review blockers.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCE_HASH = 'e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'
LABELS = [f'{kind}.b{batch}.fused' for kind in ('fp8','bf16') for batch in (16,32)]
REQUIRED_KINDS = ('ttir','ttgir','ptx','cubin','sass')
DOT_RE = re.compile(r'\b(tt\.dot|ttng\.warp_group_dot|nvgpu\.wgmma|ttng\.tc_gen5_mma)\b')
LOC_REF = re.compile(r'#loc\d*\b')
SOURCE_LOC = re.compile(r'"([^"\n]+)":(\d+):(\d+)')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_dots(text):
    tree = ast.parse(text)
    fused = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fused')
    result = {}
    for node in ast.walk(fused):
        if not isinstance(node,ast.Assign):
            continue
        targets = [n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)]
        for call in ast.walk(node.value):
            if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                    and isinstance(call.func.value,ast.Name) and call.func.value.id=='tl'
                    and call.func.attr=='dot'):
                continue
            role = 'QK' if targets==['s'] else 'PV' if targets==['acc'] else 'ambiguous'
            precision = next((ast.literal_eval(k.value) for k in call.keywords
                              if k.arg=='max_num_imprecise_acc' and isinstance(k.value,ast.Constant)),None)
            result[call.lineno] = dict(role=role,assignment_targets=targets,
                max_num_imprecise_acc_source_literal=precision,
                source=ast.get_source_segment(text,call))
    return result


def classify(locations, source_map):
    relevant = sorted({int(l['source_line']) for l in locations
        if Path(l['source_file']).name in ('candidate.py','mixed_fa3_fp8_qk.py')
        and int(l['source_line']) in source_map})
    roles = {source_map[n]['role'] for n in relevant}
    return (next(iter(roles)) if len(roles)==1 and 'ambiguous' not in roles else 'ambiguous'), relevant


def ir_dots(text, source_map):
    definitions = {}
    for line in text.splitlines():
        match = re.match(r'\s*(#loc\d*)\s*=\s*(.*)',line)
        if match:
            definitions[match[1]]=match[2]

    def resolve(ref, seen=None):
        seen = set() if seen is None else seen
        if ref in seen:
            return []
        seen = seen | {ref}
        expr = definitions.get(ref,'')
        result = [dict(source_file=f,source_line=int(n),source_column=int(c))
                  for f,n,c in SOURCE_LOC.findall(expr)]
        for parent in LOC_REF.findall(expr):
            result.extend(resolve(parent,seen))
        # Stable de-duplication retains the source/callsite evidence.
        return list({(x['source_file'],x['source_line'],x['source_column']):x for x in result}.values())

    result=[]
    for number,line in enumerate(text.splitlines(),1):
        match=DOT_RE.search(line)
        if not match:
            continue
        refs = LOC_REF.findall(line.rsplit('loc(',1)[-1]) if 'loc(' in line else []
        locations=[]
        for ref in refs:
            locations.extend(resolve(ref))
        locations.extend(dict(source_file=f,source_line=int(n),source_column=int(c))
                         for f,n,c in SOURCE_LOC.findall(line))
        role,mapped = classify(locations,source_map)
        precision = re.search(r'(?:maxNumImpreciseAcc|max_num_imprecise_acc)\s*=\s*(\d+)',line)
        result.append(dict(artifact_line=number,op=match[1],role=role,
            mapped_source_dot_lines=mapped,locations=locations,
            tensor_element_types=re.findall(r'tensor<[^>]*?(f8E4M3FN|f8E5M2|bf16|f16|f32)(?=[,>])',line),
            max_num_imprecise_acc_ir=(int(precision[1]) if precision else None),
            precision_attribute_state=('explicit' if precision else 'omitted_or_unavailable'),
            evidence=line.strip()))
    return result


def ptx_ops(text,source_map):
    files={int(i):f for i,f in re.findall(r'(?m)^\s*\.file\s+(\d+)\s+"([^"]+)"',text)}
    active=[]
    result=[]
    for number,line in enumerate(text.splitlines(),1):
        loc = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)',line)
        if loc:
            indices=[(loc[1],loc[2],loc[3])]+re.findall(r'inlined_at\s+(\d+)\s+(\d+)\s+(\d+)',loc[4])
            active=[dict(source_file=files.get(int(i),f'<file-index:{i}>'),source_line=int(n),source_column=int(c))
                    for i,n,c in indices]
        ins = re.match(r'\s*(?:@!?%\w+\s+)?(wgmma\.[\w.]+|mma\.sync[\w.]*|cp\.async[\w.]*)(?=\s|;)',line)
        if not ins:
            continue
        role,mapped=classify(active,source_map)
        result.append(dict(artifact_line=number,opcode=ins[1],role=role,
            mapped_source_dot_lines=mapped,locations=active[:],evidence=line.strip()))
    return dict(source_files=files,instructions=result,
        counts=dict(Counter(x['opcode'] for x in result)),
        matrix_counts_by_role={role:dict(Counter(x['opcode'] for x in result
            if x['role']==role and (x['opcode'].startswith('mma.sync') or x['opcode'].startswith('wgmma.mma'))))
            for role in ('QK','PV','ambiguous')},
        wgmma_commit_count=sum(x['opcode'].startswith('wgmma.commit_group') for x in result),
        wgmma_wait_count=sum(x['opcode'].startswith('wgmma.wait_group') for x in result),
        wgmma_fence_count=sum(x['opcode'].startswith('wgmma.fence') for x in result),
        cp_async_commit_count=sum(x['opcode'].startswith('cp.async.commit_group') for x in result),
        cp_async_wait_count=sum(x['opcode'].startswith('cp.async.wait_group') for x in result),
        local_memory_load_count=len(re.findall(r'\bld\.local[.\w]*\b',text)),
        local_memory_store_count=len(re.findall(r'\bst\.local[.\w]*\b',text)))


def sass_ops(text):
    rows=[]
    for number,line in enumerate(text.splitlines(),1):
        match=re.search(r'/\*([0-9a-fA-F]{4,})\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Za-z0-9_.]*)\s',line)
        if match:
            rows.append(dict(artifact_line=number,address=match[1],opcode=match[2],evidence=line.strip()))
    return dict(opcode_counts=dict(Counter(x['opcode'] for x in rows)),
        hgmma_count=sum(x['opcode'].startswith('HGMMA') for x in rows),
        hmma_count=sum(x['opcode'].startswith('HMMA') for x in rows),
        warpgroup_control=[x for x in rows if x['opcode'].startswith('WARPGROUP')],
        local_load_store=[x for x in rows if x['opcode'].split('.')[0] in ('LDL','STL')],
        qualification='Static counts only. HGMMA presence alone cannot distinguish QK from PV. Local-memory operations are not automatically spill bytes.')


def analyze_record(row,folder,source_map):
    label=row['label']
    result=dict(label=label,resources={k:row.get(k) for k in
        ('registers','driver_spills_raw','shared_memory_bytes','kernel_hash','sass_tool_exit_code')},
        files={},missing=[],hash_errors=[],options=row.get('options'),grid=row.get('grid'))
    texts={}
    artifacts=row.get('artifacts',{})
    for kind in (*REQUIRED_KINDS,'llir'):
        item=artifacts.get(kind)
        name=item.get('file') if isinstance(item,dict) else f'{label}.{kind}'
        path=folder/name
        if not path.is_file():
            if kind in REQUIRED_KINDS:
                result['missing'].append(kind)
            continue
        actual=sha(path)
        result['files'][kind]=dict(file=name,bytes=path.stat().st_size,sha256=actual,
            expected_sha256=item.get('sha256') if isinstance(item,dict) else None)
        if not isinstance(item,dict) or item.get('sha256')!=actual:
            result['hash_errors'].append(kind)
        if kind!='cubin':
            texts[kind]=path.read_text()
    for kind in ('ttir','ttgir'):
        if kind in texts:
            result[kind]=dict(dots=ir_dots(texts[kind],source_map),
                mma_layout_versions=sorted({int(v) for v in re.findall(r'versionMajor\s*=\s*(\d+)',texts[kind])}))
    if 'ptx' in texts:
        result['ptx']=ptx_ops(texts['ptx'],source_map)
    if 'sass' in texts:
        result['sass']=sass_ops(texts['sass'])
    qk_ir=[d for d in result.get('ttgir',{}).get('dots',[]) if d['role']=='QK']
    qk_ptx=[d for d in result.get('ptx',{}).get('instructions',[]) if d['role']=='QK'
            and (d['opcode'].startswith('mma.sync') or d['opcode'].startswith('wgmma.mma'))]
    ambiguous_ir=[d for d in result.get('ttgir',{}).get('dots',[]) if d['role']=='ambiguous']
    expected_qk={line for line,info in source_map.items() if info['role']=='QK'}
    ir_mapped={line for d in qk_ir for line in d['mapped_source_dot_lines']}
    ptx_mapped={line for d in qk_ptx for line in d['mapped_source_dot_lines']}
    mapping_incomplete=bool(ambiguous_ir or ir_mapped!=expected_qk or ptx_mapped!=expected_qk)
    qk_wgmma=bool(qk_ir and qk_ptx) and all('warp_group_dot' in d['op'] for d in qk_ir) and all(d['opcode'].startswith('wgmma.mma') for d in qk_ptx)
    qk_wgmma_any=any('warp_group_dot' in d['op'] for d in qk_ir) and any(d['opcode'].startswith('wgmma.mma') for d in qk_ptx)
    qk_ttir=[d for d in result.get('ttir',{}).get('dots',[]) if d['role']=='QK']
    explicit=[d['max_num_imprecise_acc_ir'] for d in qk_ttir]
    source_zero=all(x['max_num_imprecise_acc_source_literal']==0 for x in source_map.values() if x['role']=='QK')
    contradiction=any(n is not None and n!=0 for n in explicit)
    blockers=[]
    if result['missing']: blockers.append('Missing required artifacts')
    if result['hash_errors']: blockers.append('Artifact receipt hash mismatch/missing expected hash')
    if mapping_incomplete: blockers.append('QK/PV source mapping requires manual review')
    if not source_zero or contradiction: blockers.append('Precision contract not established or explicit IR contradiction')
    result['qk_wgmma_gate']=dict(qk_wgmma_mapped_in_ttgir_and_ptx=qk_wgmma,
        any_qk_wgmma_mapped_in_ttgir_and_ptx=qk_wgmma_any,
        expected_qk_source_lines=sorted(expected_qk),
        per_source_dot_lowering={str(line):dict(source_role=info['role'],
            ttgir_ops=sorted({d['op'] for d in result.get('ttgir',{}).get('dots',[]) if line in d['mapped_source_dot_lines']}),
            ptx_matrix_ops=sorted({d['opcode'] for d in result.get('ptx',{}).get('instructions',[])
                if line in d['mapped_source_dot_lines'] and (d['opcode'].startswith('mma.sync') or d['opcode'].startswith('wgmma.mma'))}))
            for line,info in sorted(source_map.items())},
        source_qk_precision_literal_zero=source_zero,
        ttir_qk_precision_values=explicit,
        ttir_explicit_zero_for_all_qk=bool(explicit) and all(n==0 for n in explicit),
        precision_qualification=('Omitted TTIR attribute is not by itself zero proof; frozen source literals establish requested policy, and explicit nonzero IR would contradict it.'),
        needs_manual_mapping=mapping_incomplete,blockers=blockers,
        disposition=('incomplete_or_requires_manual_review' if blockers else
            'QK_WGMMA_mechanism_observed_runtime_unqualified' if qk_wgmma else
            'mixed_QK_lowering_requires_manual_mechanism_review' if qk_wgmma_any else
            'QK_WGMMA_not_observed_reject_this_operand_widening_hypothesis'))
    return result


def self_test():
    source={141:dict(role='QK'),154:dict(role='PV')}
    ir='''%s = ttng.warp_group_dot %a, %b, %c {maxNumImpreciseAcc = 0 : i32} : tensor<64x64xbf16> loc(#loc9)
%acc = ttng.warp_group_dot %a, %v, %c : tensor<64x64xbf16> loc(#loc10)
#loc9 = loc("s"(#loc1))
#loc10 = loc(callsite(#loc2 at #loc3))
#loc1 = loc("/opt/probe/candidate.py":141:13)
#loc2 = loc("/opt/probe/candidate.py":154:29)
#loc3 = loc(unknown)'''
    parsed=ir_dots(ir,source)
    assert [d['role'] for d in parsed]==['QK','PV']
    assert parsed[0]['max_num_imprecise_acc_ir']==0 and parsed[1]['max_num_imprecise_acc_ir'] is None
    ptx='''.file 1 "/opt/probe/candidate.py"
.loc 1 154 29
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%r0}, 0, 0, 1;
wgmma.commit_group.sync.aligned;
.loc 1 141 13
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r0}, {%r1}, {%r2}, {%r0};'''
    p=ptx_ops(ptx,source)
    assert p['matrix_counts_by_role']['QK'] and not any(k.startswith('wgmma') for k in p['matrix_counts_by_role']['QK'])
    assert any(k.startswith('wgmma') for k in p['matrix_counts_by_role']['PV'])
    assert p['wgmma_commit_count']==1
    assert len(sass_ops('/*0120*/ HGMMA.64x64x16.F32 R1, gdesc[UR2], R4; /* 0x00 */')['opcode_counts'])==1
    print('Semantic parser checks passed: recursive role mapping, precision omission, PV-only WGMMA does not establish QK, SASS counting.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',nargs='?',type=Path,default=HERE.parent/'native_artifacts')
    parser.add_argument('--source',type=Path,default=HERE/'candidate.py')
    parser.add_argument('--output',type=Path,default=HERE/'native-analysis.json')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(); return 0
    assert sha(args.source)==SOURCE_HASH, 'Frozen A2-13 source mismatch'
    source_map=source_dots(args.source.read_text())
    assert len(source_map)==4 and sum(x['role']=='QK' for x in source_map.values())==2
    summary_path=args.folder/'summary.json'
    summary=json.loads(summary_path.read_text())
    assert summary.get('source_sha256')==SOURCE_HASH, 'Compiled source hash mismatch'
    records=summary.get('records',[])
    assert len({r['label'] for r in records})==len(records),'Duplicate specialization labels'
    report=dict(kind='Offline compile/disassembly evidence only; no runtime qualification',
        analyzer_sha256=sha(Path(__file__)),source_sha256=sha(args.source),
        summary_sha256=sha(summary_path),summary_status=summary.get('status'),
        candidate_kernel_launches=summary.get('candidate_kernel_launches'),
        correctness_tested=summary.get('correctness_tested'),performance_tested=summary.get('performance_tested'),
        environment=summary.get('environment'),source_dot_contract=source_map,
        installed_fa3=summary.get('installed_fa3'),
        missing_labels=sorted(set(LABELS)-{r['label'] for r in records}),
        records=[analyze_record(r,args.folder,source_map) for r in records],comparisons=[],
        limitations=['Static instruction counts and lexical commit/wait placement do not establish overlap, stalls, throughput or latency.',
          'SASS HGMMA alone does not attribute QK vs PV; role evidence comes from TTGIR and PTX source mapping.',
          'Driver spill field retains reported raw units; local loads/stores are not automatically spill bytes.',
          'No correctness, cache ownership, graph replay, paired end-to-end timing, or official score is established.'])
    assert report['candidate_kernel_launches']==0 and report['correctness_tested'] is False and report['performance_tested'] is False
    by_label={r['label']:r for r in report['records']}
    for batch in (16,32):
        a,b=by_label.get(f'fp8.b{batch}.fused'),by_label.get(f'bf16.b{batch}.fused')
        if a and b:
            deltas={}
            for key in ('registers','driver_spills_raw','shared_memory_bytes'):
                x,y=a['resources'].get(key),b['resources'].get(key)
                deltas[key]=(y-x if isinstance(x,(int,float)) and isinstance(y,(int,float)) else None)
            report['comparisons'].append(dict(batch=batch,bf16_minus_fp8=deltas))
    complete=not report['missing_labels'] and summary.get('status')=='compiled' and all(
        not r['missing'] and not r['hash_errors'] for r in report['records'])
    report['artifact_audit_complete']=complete
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),artifact_audit_complete=complete,
        gates={r['label']:r['qk_wgmma_gate']['disposition'] for r in report['records']}),indent=2))
    return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
