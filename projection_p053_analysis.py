"""Offline P052 phase-cost evidence, not a hardware-counter or speed model."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RECEIPT = "results/remote-P052-screen/timing.json"
RECEIPT_SHA = "2ae054ec40d6f39a119dfe0c97fd330dfde09e315ba51d45b0cffb6cc71cd30f"


def logical_work(m, n=4096, k=8192):
    if type(m) is not int or not 1 <= m <= 16 or n != 4096 or k != 8192:
        raise ValueError("This source model covers the fixed canonical P051 tile only")
    tile_m, tile_n, tile_k, splits, threads = 16, 64, 64, 8, 128
    column_tiles = n // tile_n
    partial_ctas = column_tiles * splits
    return dict(m=m, n=n, k=k, partial_ctas=partial_ctas,
        reducer_ctas=(m*n+127)//128, iterations_per_partial_cta=k//splits//tile_k,
        useful_flops=2*m*n*k, padded_tensor_flops=2*tile_m*n*k,
        padded_over_useful_flops=tile_m/m,
        logical_weight_read_bytes=2*n*k,
        logical_activation_read_bytes=2*m*k*column_tiles,
        unique_activation_bytes=2*m*k,
        partial_write_bytes=4*splits*m*n, reducer_read_bytes=4*splits*m*n,
        output_write_bytes=2*m*n,
        partial_epilogue_shared_read_plus_write_bytes=64*threads*partial_ctas,
        qualification="Source-address valid byte counts and padded tensor arithmetic; not HBM transactions, cache misses, achieved FLOPS or measured shared traffic")


def ptx_body(text):
    # Exact executable text comparison omits source locations/debug sections.
    text = text.split("\n\t.file", 1)[0]
    return "\n".join(line.split("//", 1)[0].rstrip() for line in text.splitlines()
        if not line.lstrip().startswith(".loc") and line.split("//", 1)[0].strip())


def instruction_sites(text):
    body = ptx_body(text)
    lines = []
    for raw in body.splitlines():
        match = re.match(r"^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.:]*)\b", raw)
        if match:
            lines.append(match.group(1))
    def count(prefix):
        return sum(x.startswith(prefix) for x in lines)
    return dict(static_instruction_lines=len(lines),
        async_data_copy_sites=count("cp.async.cg.shared.global")+count("cp.async.ca.shared.global"),
        async_commit_sites=count("cp.async.commit_group"), async_wait_sites=count("cp.async.wait_group"),
        mma_sync_sites=count("mma.sync"), wgmma_sites=count("wgmma.mma_async"),
        shared_vector2_stores=count("st.shared::cta.v2.b32"),
        shared_vector4_stores=count("st.shared::cta.v4.b32"),
        shared_vector2_loads=count("ld.shared.v2.b32"),
        shared_vector4_loads=count("ld.shared.v4.b32"),
        global_vector4_stores=count("st.global.v4.b32"),
        qualification="Static PTX sites, not dynamic counts, transaction efficiency or per-instruction latency")


def analyze(root=ROOT):
    raw=(root/RECEIPT).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=RECEIPT_SHA:
        raise ValueError("Frozen P052 timing receipt changed")
    receipt=json.loads(raw)
    records=[]
    for case in receipt["cases"]:
        m=case["m"]
        item=dict(work=logical_work(m), timing={name:s["summary"] for name,s in case["series"].items()}, ptx={})
        ratio=case["series"]["selected"]["summary"]["median_paired_raw_ratio"]
        item["candidate_cut_to_match_p046_under_same_paired_statistic"]=1-ratio
        for arm in ("control","weight_first"):
            path=root/f"results/remote-P052-screen/correctness.pooler_m{m}.{arm}.ptx"
            text=path.read_text()
            facts=instruction_sites(text)
            facts["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            if m==8:
                previous=root/f"results/remote-P051-compile/timing.pooler_m8.{arm}.ptx"
                facts["same_executable_text_as_p051"]=ptx_body(previous.read_text())==ptx_body(text)
            item["ptx"][arm]=facts
        records.append(item)
    return dict(kind="P053 offline discriminating-test rationale", source_receipt_sha256=RECEIPT_SHA,
        gpu_runs_added=0, official_checks_added=0, cases=records,
        missing=["P052 partial and reducer activity durations", "Measured layout-conversion cost",
                 "Measured HBM/cache transactions", "Measured matrix/copy overlap or stalls"],
        decision="Prepare same-body whole/partial comparisons and separate activity; no new candidate or promotion")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(analyze(),indent=2)+"\n")


if __name__=="__main__":
    main()
