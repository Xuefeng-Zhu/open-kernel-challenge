"""P061: immutable source/address/score audit. No candidate imports or GPU work."""
import hashlib
import json
from pathlib import Path

import projection_gap_model as model

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "candidates/projection_pooler_column_batch2_hybrid.py": "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e",
    "candidates/projection_pooler_lossless_codec.py": "9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837",
    "candidates/projection_pooler_lossless_raw_control.py": "8ee5189831d7dcd209284b14a1ed907140d45ad9beba8c266122dc07e93a43de",
}


def address_model(n, k, escapes):
    """Canonical full K256 blocks only; not issued transactions or HBM bytes."""
    assert k % 256 == 0 and 0 <= escapes <= n * (k // 256)
    elements, blocks = n * k, n * (k // 256)
    # Deduplicate the two k lanes addressing each packed delta byte, but only
    # within each BK64 iteration. Base/index are requested four times per block.
    unique_iteration_bytes = elements + (blocks - escapes) * 128 + escapes * 256 + blocks * 5 * 4
    # The literal tensor load operands name one exponent byte per weight lane:
    # duplicate nibble addresses count twice here, unlike the line above.
    scalar_operand_bytes = elements * 2 + blocks * 5 * 4
    return dict(raw_weight_bytes=elements * 2,
                distinct_addresses_summed_per_iteration_bytes=unique_iteration_bytes,
                distinct_address_ratio=unique_iteration_bytes / (elements * 2),
                scalar_operand_bytes=scalar_operand_bytes,
                scalar_operand_ratio=scalar_operand_bytes / (elements * 2),
                qualification="Logical source address accounting only; compiler reuse/vectorization, transactions, caches, spills and hardware traffic are unknown.")


def build():
    # Independent small-block enumeration checks both normal and escape formulas.
    for escaped in (False, True):
        count = 0
        for start in range(0, 256, 64):
            addresses = {("base", 0)} | {("index", byte) for byte in range(4)}
            addresses |= {("sign_fraction", k) for k in range(start, start+64)}
            addresses |= {("escape" if escaped else "delta", k if escaped else k//2)
                          for k in range(start, start+64)}
            count += len(addresses)
        assert count == address_model(1, 256, int(escaped))["distinct_addresses_summed_per_iteration_bytes"]
    for path, expected in SOURCES.items():
        assert model.sha(ROOT / path) == expected, path
    compressed = (ROOT / "candidates/projection_pooler_lossless_codec.py").read_text()
    raw = (ROOT / "candidates/projection_pooler_lossless_raw_control.py").read_text()
    assert compressed.replace("POOLER_COMPRESSED = True", "POOLER_COMPRESSED = False", 1) == raw
    rows = model.read_rows(ROOT)
    current, loads = model.score(rows)
    p046 = json.loads((ROOT / model.RECEIPT).read_text())
    assert abs(current - p046["result"]["speedup"]) < 1e-12
    storage_path = "results/router-p043-codec-storage-cpu.json"
    p043_path = "results/remote-P2-07-pooler-lossless-codec.json"
    storage = json.loads((ROOT / storage_path).read_text())
    p043 = json.loads((ROOT / p043_path).read_text())
    reduction = min(row["byte_reduction"] for row in storage["results"])
    scopes = {
        "existing_three_pooler_targets": lambda r: r["layer"] == "pooler" and r["m"] in (8, 16, 32),
        "all_pooler_hypothetical_extension": lambda r: r["layer"] == "pooler",
        "three_families_hypothetical_extension": lambda r: r["layer"] in model.MAIN,
    }
    coverage = {}
    for name, selected in scopes.items():
        scenarios = []
        for cut in (.1, .2, reduction, 1.):
            changed = [{**r, "candidate_us": r["candidate_us"] * (1-cut)} if selected(r) else r for r in rows]
            score = model.score(changed)[0]
            scenarios.append(dict(uniform_target_cut=cut, modeled_score=score, modeled_gain=score/current-1))
        coverage[name] = dict(shapes=sum(selected(r) for r in rows),
            mean_load_share=sum(r["weight"]*r["candidate_us"]/loads[r["load"]]["candidate_us"] for r in rows if selected(r))/6,
            scenarios=scenarios)
    comparisons = []
    for record in p043["result"]["records"]:
        if record.get("event") != "benchmark" or not scopes["existing_three_pooler_targets"](record["case"]):
            continue
        parent = next(r for r in rows if (r["load"], r["layer"]) == (record["case"]["load"], "pooler"))
        ratio = (parent["reference_us"]/parent["candidate_us"])/record["paired_ratio"]
        comparisons.append(dict(m=parent["m"], p046_calibrated_us=parent["candidate_us"],
            p043_calibrated_us=record["median_ns"]/1000,
            p043_over_p046_reference_normalized_latency=ratio,
            p043_own_cut_needed_for_p046_parity=1-1/ratio))
    return dict(kind="Local source, address and conditional score audit; no new measurement",
        selected="P046", selected_job="41ba778763a5433b", selected_score=current,
        source_hashes=SOURCES, matched_raw_diff="Only POOLER_COMPRESSED True versus False",
        matched_raw_gpu_evidence=False, p043_codegen_available=False,
        old_candidate_comparison=comparisons, coverage=coverage,
        per_seed_address_models=[dict(seed=r["seed"], stored_byte_reduction=r["byte_reduction"],
            **address_model(*r["shape"], r["escape_blocks"])) for r in storage["results"]],
        geometry={str(m): dict(ctas=64*8*((m+15)//16), iterations_per_cta=16,
            weight_decode_passes=(m+15)//16, partial_scratch_bytes=8*m*4096*4,
            partial_write_plus_read_bytes=2*8*m*4096*4) for m in (8,16,32)},
        optimistic_storage_only_screen=dict(ratio=1-reduction,
            max_raw_over_p046_for_parity=1/(1-reduction),
            max_raw_over_p046_for_ten_percent_target_cut=.9/(1-reduction),
            qualification="Assume ALL raw time scales with stored weight bytes and zero incremental decode cost. Not a hardware bound or prediction; a reason to deprioritize slow consumers, not proof of impossibility."),
        evidence_hashes={p:model.sha(ROOT/p) for p in (model.CSV, model.RECEIPT, storage_path, p043_path, "TASK.md")},
        decision="Do not relaunch unchanged P043 as an optimization. Preserve optional three-arm attribution protocol; no distinct competitive consumer-aligned mechanism has been demonstrated.",
        no_gpu_launch=True, no_official_check=True)


if __name__ == "__main__":
    report = build()
    (ROOT / "results/projection-compression-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"selected_score":report["selected_score"], "coverage":report["coverage"],
                      "source_hashes_verified":True, "gpu_launched":False}))
