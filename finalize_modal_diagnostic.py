"""Reconcile only an existing completed app via supported read-only CLI calls."""
import argparse
import csv
from datetime import datetime
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import re

from modal_analysis import summarize_case,aggregate_graph
from modal_diagnostics import ROOT,LEDGER,now,read_cli,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",choices=["M19-unpooler","M20-qkv-repeat","M23-composition","M26-qkv-layout","M28-unpooler-layout", "M30-pooler-grouped", "M32-pooler-narrow", "M33-resource-probe", "M34-resource-probe-fixed", "M36-qkv-m128-singlepass"])
    args=parser.parse_args()
    os.environ["MODAL_CONFIG_PATH"]=str(ROOT/"local-notes/modal-auth.toml")
    ledger=json.loads(LEDGER.read_text())
    entry=next(e for e in ledger["attempts"] if e["experiment"]==args.experiment)
    apps=read_cli("app","list","--json")
    app=next(a for a in apps if a["app_id"]==entry["app_id"])
    if app["state"]!="stopped" or int(app["tasks"])!=0:
        raise RuntimeError("Exact app not yet stopped; no subsequent launch allowed")
    billing=read_cli("billing","summary","--json")
    raw=json.loads((ROOT/f"local-notes/modal-{args.experiment}-result.local.json").read_text())
    graph=raw["records"].get("graph-diagnostic.json",{})
    summaries=[summarize_case(c,graph.get("execution_orders")) for c in graph.get("cases",[])]
    runner_names=tuple(graph.get("source_sha256",dict(candidate=None,r10=None,sglang=None)))
    passed=(raw["exit_code"]==0 and raw["records"].get("stages.json",{}).get("passed",False)
            and len(summaries)==30 and all(raw["records"][n+"-correctness.json"]["passed"]
                                           for n in runner_names))
    if entry.get("protocol")=="resource-v1":
        passed=raw["exit_code"]==0 and raw["records"].get("resource-probe.json",{}).get("passed",False)
    elapsed=(datetime.fromisoformat(app["stopped_at"])-datetime.fromisoformat(app["created_at"])).total_seconds()
    # Conservative bound covers the entire app lifetime (including non-GPU setup),
    # plus5s and10%margin, not just worker time. No extra resource is provisioned.
    upper=(Decimal(math.ceil(elapsed+5))*(Decimal('.001261')+2*Decimal('.0000131')+8*Decimal('.00000222'))
           *Decimal('1.1')+Decimal('.01'))
    entry.update(state="terminal",platform_state=app["state"],tasks=0,termination_verified_utc=now(),
                 correctness_and_graph_passed=passed,billing_after=billing,app_lifetime_seconds=elapsed,
                 conservative_cost_upper_usd=str(upper))
    initial=Decimal('0.12054281')
    bound=initial+sum((Decimal(a.get("conservative_cost_upper_usd","0")) for a in ledger["attempts"]),Decimal(0))
    observed=max(Decimal(billing["metered_cost"]),sum((Decimal(v) for v in billing["metered_cost_breakdown"].values()),Decimal(0)))
    ledger["conservative_stage_spent_usd"]=str(bound)
    ledger["observed_gross_high_water_usd"]=str(max(Decimal(ledger["observed_gross_high_water_usd"]),observed))
    ledger["latest_billing_observed_utc"]=now()
    write(ledger)
    evidence=dict(kind="Modal H200 diagnostic, no official score/CI",experiment=args.experiment,
                  attempt=entry,app=app,result={k:v for k,v in raw.items() if k!="logs"},
                  failure_logs=raw.get("logs") if raw["exit_code"] else None,
                  summaries=summaries,aggregate=aggregate_graph(graph) if graph.get("protocol") in {"composition-v1","screen-v1"} else None,
                  billing_snapshot=billing,billing_observed_utc=now(),
                  qualifications=["All samples retained; no outlier exclusion or overhead subtraction",
                                  "Stage upper bound is a conservative budget guard, not actual billed cost",
                                  "Billing API counters may settle asynchronously",
                                  "App lifetime/worker wall time are not GPU busy time"])
    encoded=json.dumps(evidence,indent=2)+"\n"
    if re.search(r'\b(?:ak|as)-[A-Za-z0-9]{8,}',encoded) or '/Users/' in encoded:
        raise ValueError("Unexpected credential identifier or personal path in evidence")
    (ROOT/f"results/remote-modal-{args.experiment}.json").write_text(encoded)
    with (ROOT/f"results/remote-modal-{args.experiment}.csv").open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['load','layer','m',*[name+'_'+metric for name in runner_names for metric in ['median_us','p90_us','max_us']],
                         *['paired_'+name+'_over_candidate' for name in runner_names[1:]]])
        for item in summaries:
            d=item['gross_us_distribution'];c=item['case'];r=item['median_paired_gross_ratio']
            writer.writerow([c['load'],c['layer'],c['m'],*[d[name][metric] for name in runner_names for metric in ['median','p90','maximum']],
                             *[r[name] for name in runner_names[1:]]])
    print(args.experiment,'pass',passed,'stopped app',app['app_id'],'lifetime',elapsed,
          'actual cumulative metered',billing['metered_cost'],'net',billing['billed_cost'],
          'conservative stage bound',bound)
    if Decimal(billing['billed_cost'])!=0 or bound>2 or observed>2:
        raise RuntimeError("Cost gate: stop all further work")


if __name__=="__main__":
    main()
