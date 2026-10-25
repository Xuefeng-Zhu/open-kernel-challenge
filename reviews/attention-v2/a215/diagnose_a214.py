"""Reproduce a source-level cost audit from local sources and official receipts.

No candidate imports, network, GPU, profiling counters or native-code assertions.
"""
import hashlib
import json
from pathlib import Path
import re
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXPECTED={'A2-13':('mixed_fa3_fp8_qk.py','e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e'),'A2-14':('mixed_fa3_deferred_denom.py','bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d')}


def source_counts(rows,cols,tiles):
    assert rows>0 and cols>0 and tiles>=0
    before=dict(persistent_fp32_elements=rows,multiplies=rows*tiles,additions=rows*cols*tiles,horizontal_sum_calls=tiles)
    after=dict(persistent_fp32_elements=rows*cols,multiplies=rows*cols*tiles,additions=rows*cols*tiles+rows*(cols-1),horizontal_sum_calls=1)
    return dict(rows=rows,cols=cols,tiles=tiles,before=before,after=after,added_multiplies=after['multiplies']-before['multiplies'],added_additions=after['additions']-before['additions'],added_persistent_bytes=4*(after['persistent_fp32_elements']-before['persistent_fp32_elements']),unchanged_max_reductions=tiles)


def data():
    result={}
    for name,(file,sha) in EXPECTED.items():
        source=ROOT/file;assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
        receipt_path=ROOT/(name+'-official-result.json');receipt=json.loads(receipt_path.read_text())
        r=receipt['result'];assert receipt['status']=='done' and r['passed']
        target=next(x for x in r['records'] if x['event']=='benchmark' and x['case']['load']=='decode 32' and x['case']['layer']=='full')
        profile=next(x['report'] for x in r['records'] if x['event']=='profile' and x['case']['load']=='decode 32' and x['case']['layer']=='full')
        activities={n:float(t) for n,t in re.findall(r'^(_fused|_merge)\s+([0-9.]+) µs',profile,re.M)}
        resources=[]
        def walk(value,path='result'):
            if isinstance(value,dict):
                for k,v in value.items():
                    if k.lower() in ('n_regs','num_regs','registers','n_spills','spills','shared','shared_memory','ptx','sass','ttir','ttgir','cubin'):resources.append(path+'.'+k)
                    walk(v,path+'.'+k)
            elif isinstance(value,list):
                for i,v in enumerate(value):walk(v,path+'['+str(i)+']')
        walk(r)
        result[name]=dict(source=file,source_sha256=sha,job=receipt['id'],receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),whole_call_adjusted_us=target['median_ns']/1000,paired_ratio=target['paired_ratio'],profile_activity_us=activities,resource_fields_found=resources,log_tail_characters=len(r.get('log_tail') or ''),evaluator=r['evaluator'])
    extensions={'.ptx','.cubin','.sass','.ttir','.ttgir','.ll'}
    # Filename inventory only; never reads credential files or unrelated JSON.
    roots=[ROOT,ROOT.parents[1]/'local-notes']
    files=[]
    for folder in roots:
        files.extend(str(p.relative_to(ROOT.parents[1])) for p in folder.rglob('*') if p.is_file() and p.suffix in extensions)
    ratio=result['A2-13']['paired_ratio']/result['A2-14']['paired_ratio']
    return dict(evidence=result,same_evaluator=result['A2-13']['evaluator']==result['A2-14']['evaluator'],normalized_target_latency_increase=ratio-1,full_chunk_source_model=source_counts(16,64,128),artifact_search=dict(roots=['reviews/attention-v2','local-notes'],filename_extensions=sorted(extensions),matching_files=sorted(files),scope='Only saved project attention and local-note artifact paths; no claim about inaccessible worker cache.'),next_prototype=dict(source='../mixed_fa3_page_lookahead.py',sha256=hashlib.sha256((ROOT/'mixed_fa3_page_lookahead.py').read_bytes()).hexdigest(),parent='A2-13',carried_row_vector_logical_bytes=64*8,prefetched_raw_page_ids_logical_bytes=64*4,native_allocation_delta_unknown=True,page1_logical_page_id_bytes_per_tile=64*4,logical_fp8_KV_bytes_per_head_tile=2*64*64,valid_read_traffic_removed=0),limitations=['Source scalar arithmetic counts are pre-optimization and not native instructions or runtime cost.','FMA contraction, first-iteration simplification, inactive rows and compiler layout may change emitted work.','Logical state bytes are not register allocation, spill traffic or occupancy.','Historical projection or failed A09 metadata cannot diagnose these candidate bytes.'])


class Counts(unittest.TestCase):
    def test_full_chunk_expansion(self):
        x=source_counts(16,64,128)
        self.assertEqual((x['before']['multiplies'],x['before']['additions']),(2048,131072))
        self.assertEqual((x['after']['multiplies'],x['after']['additions']),(131072,132080))
        self.assertEqual((x['added_multiplies'],x['added_additions'],x['added_persistent_bytes']),(129024,1008,4032))
    def test_one_column_identity(self):
        x=source_counts(16,1,128)
        self.assertEqual(x['added_multiplies'],0);self.assertEqual(x['added_additions'],0)
        self.assertEqual(x['added_persistent_bytes'],0)
    def test_actual_receipt_identity_and_profile(self):
        x=data();self.assertTrue(x['same_evaluator'])
        self.assertEqual(x['evidence']['A2-13']['profile_activity_us'],{'_fused':311.1,'_merge':2.6})
        self.assertEqual(x['evidence']['A2-14']['profile_activity_us'],{'_fused':422.0,'_merge':2.7})
        self.assertAlmostEqual(x['normalized_target_latency_increase'],.339263163,places=8)


if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv:
        unittest.main(argv=[sys.argv[0]],verbosity=2)
    else:
        x=data();(HERE/'diagnosis.json').write_text(json.dumps(x,indent=2)+'\n')
        print(json.dumps({'target_normalized_latency_increase':x['normalized_target_latency_increase'],'source_count_model':x['full_chunk_source_model'],'compiler_artifacts_found':len(x['artifact_search']['matching_files'])}))
