"""Frozen P046 family coverage and conditional score leverage; local analysis only."""
import hashlib
import json
from pathlib import Path

import projection_gap_model as model

ROOT = Path(__file__).resolve().parent
SOURCE = "candidates/projection_pooler_column_batch2_hybrid.py"
SOURCE_SHA = "001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e"


def build():
    assert hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA
    rows = model.read_rows(ROOT)
    current, loads = model.score(rows)
    receipt = json.loads((ROOT / model.RECEIPT).read_text())
    assert abs(current - receipt["result"]["speedup"]) < 1e-12
    families = {}
    for family in model.FAMILIES:
        shares = {load: sum(r["weight"] * r["candidate_us"] for r in rows
                           if r["load"] == load and r["layer"] == family) / totals["candidate_us"]
                  for load, totals in loads.items()}
        families[family] = dict(
            per_load_share=shares, mean_load_share=sum(shares.values()) / 6,
            official_calibrated_us={str(r["m"]): r["candidate_us"] for r in rows if r["layer"] == family},
            conditional_uniform_cuts=[dict(latency_cut=cut,
                score_factor=model.score(model.change(rows, (family,), cut))[0] / current)
                for cut in (.10, .20)],
        )
    return dict(selected="P046", source=SOURCE, source_sha256=SOURCE_SHA,
        official_job="41ba778763a5433b", official_score=current,
        csv_sha256=model.CSV_SHA256, families=families,
        scoring="Geometric mean of six load speedups; all five family times within each load have weight62",
        qualification="Shares and hypothetical cuts are arithmetic on one official receipt. They do not identify a hardware bottleneck or predict attainable gains.",
        gpu_executed=False, new_official_check=False)


if __name__ == "__main__":
    result = build()
    (ROOT / "results/projection-selected-baseline-matrix.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"official_score": result["official_score"], "families": len(result["families"]),
                      "no_external_action": True}))
