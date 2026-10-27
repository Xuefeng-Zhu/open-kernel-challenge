"""Deterministic NCU application-replay target; unchanged algorithms/tolerances."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time


def main(arm, output):
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from worker import CASE, SOURCES, load
    report = dict(arm=arm, passed=False, checks=[], source_sha256=SOURCES,
                  algorithm_changed=False, case=CASE)
    def save():
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, separators=(',', ':'))+'\n')
        tmp.replace(output)
    def alarm(*_):
        raise TimeoutError('Counter target55s deadline')
    signal.signal(signal.SIGALRM, alarm); signal.alarm(55)
    save()
    try:
        for name, expected in SOURCES.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
        import torch as t
        assert t.cuda.is_available() and 'H200' in t.cuda.get_device_name()
        sys.path.insert(0, str(root/'official'))
        h = load(root/'modal_worker.py', 'counter_helpers')
        ref = load(root/'official/reference.py', 'counter_reference')
        candidate = load(root/'candidate.py', 'candidate')
        custom = load(root/'candidatecustomhelper.py', 'candidatecustomhelper')
        fn = candidate.kernel_fn if arm == 'fa3' else custom.bf16_custom_fn
        fresh = ref.generate_input(**CASE, device='cuda')
        report['actual_lengths'] = fresh['seq_lens'].cpu().tolist()
        work = h.clone_inputs(fresh)
        graph, out = h.capture(fn, work, fresh)
        addresses = {k:v.data_ptr() for k,v in work.items() if isinstance(v,t.Tensor)}
        out_address = out.data_ptr()
        expected = None
        def check(label):
            nonlocal expected
            raw = ref.collect(work, out)
            intact, why = h.unchanged(work, fresh)
            okay, message = ref.check_implementation(work, raw)
            exact = bool(t.equal(raw,expected)) if expected is not None else None
            stable = addresses == {k:v.data_ptr() for k,v in work.items() if isinstance(v,t.Tensor)} and out.data_ptr()==out_address
            row = dict(label=label,passed=bool(okay and intact and stable and exact is not False),
                oracle_called=True,oracle_message=message,immutable_inputs_and_untouched_rows=intact,
                immutable_message=why,stable_addresses=stable,exact_output_newkv_return=exact)
            report['checks'].append(row);save()
            assert row['passed'], 'Counter target correctness gate failed'
            if expected is None: expected=raw.clone()
        h.restore(work,fresh);graph.replay();t.cuda.synchronize();check('pre_profile')
        eviction=t.ones(256*1024*1024//4,device='cuda')
        h.restore(work,fresh);eviction.sum();t.cuda.synchronize()
        t.cuda.nvtx.range_push('PROFILE_ONE')
        try:
            graph.replay();t.cuda.synchronize()
        finally:
            t.cuda.nvtx.range_pop()
        check('post_profile')
        report['passed']=True;save();return 0
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:1000]);save();return 1
    finally:
        signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fa3','bf16_custom'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit(main(a.arm,a.output))
