"""Offline diagnostic summaries. Retain every sample; no judge score or CI."""
import itertools
import math
import statistics


def quantile(values, fraction):
    ordered=sorted(values)
    index=(len(ordered)-1)*fraction
    low=math.floor(index);high=math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def summarize_case(case,execution_orders=None):
    samples=case["samples"]
    names=tuple(execution_orders[0]) if execution_orders else ("candidate","r10","sglang")
    if names[0]!="candidate":
        raise ValueError("Candidate must anchor the declared order")
    orders=[tuple(x) for x in execution_orders] if execution_orders else list(itertools.permutations(names))
    expected=set(orders)
    if len(expected)!=len(orders) or any(set(order)!=set(names) or len(order)!=len(names) for order in orders):
        raise ValueError("Declared orders must be unique permutations")
    if any(tuple(reversed(order)) not in expected for order in orders):
        raise ValueError("Every declared order must have its reverse")
    for name in names:
        counts=[sum(order[position]==name for order in orders) for position in range(len(names))]
        if len(set(counts))!=1:
            raise ValueError("Declared orders must balance execution positions")
    period=len(orders)
    if not samples or len(samples)%period:
        raise ValueError("Complete predefined order blocks required")
    pairs=[]
    for base in range(0,len(samples),period):
        block=samples[base:base+period]
        lookup={tuple(sample["order"]):i for i,sample in enumerate(block)}
        if set(lookup)!=expected:
            raise ValueError("Each block must contain every declared execution order")
        for i,sample in enumerate(block):
            reverse=lookup[tuple(reversed(sample["order"]))]
            if i<reverse:
                other=block[reverse]
                ratios={name:math.sqrt(sample["us"][name]/sample["us"]["candidate"]*
                                       other["us"][name]/other["us"]["candidate"])
                        for name in names[1:]}
                pairs.append(dict(indices=[base+i,base+reverse],seeds=[sample["seed"],other["seed"]],
                                  geometric_control_over_candidate=ratios))
    all_indices=[index for pair in pairs for index in pair["indices"]]
    if sorted(all_indices)!=list(range(len(samples))):
        raise ValueError("Every raw sample must participate once; no outlier removal")
    if len({s["seed"] for s in samples})!=len(samples):
        raise ValueError("Fresh-input seeds must be distinct")
    distributions={}
    for name in names:
        values=[sample["us"][name] for sample in samples]
        if not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError("Nonpositive or nonfinite timing")
        median=statistics.median(values)
        distributions[name]=dict(n=len(values),minimum=min(values),p10=quantile(values,.1),
                                  median=median,p90=quantile(values,.9),maximum=max(values),
                                  median_absolute_deviation=statistics.median(abs(v-median) for v in values))
    return dict(case=case["case"],gross_us_distribution=distributions,opposite_order_pairs=pairs,
                median_paired_gross_ratio={name:statistics.median(p["geometric_control_over_candidate"][name]
                                                                 for p in pairs) for name in names[1:]},
                limitations="Gross diagnostic, all samples retained, no official score or confidence interval")


def aggregate_graph(graph):
    """Contract-weighted gross load diagnostic; synthetic ids are NOT input seeds."""
    cases=graph["cases"]
    orders=graph["execution_orders"]
    names=tuple(orders[0])
    loads={}
    for row in cases:loads.setdefault(row["case"]["load"],[]).append(row)
    summaries=[]
    for load,rows in loads.items():
        samples=[]
        for i in range(graph["repeats"]):
            order=rows[0]["samples"][i]["order"]
            if any(row["samples"][i]["order"]!=order for row in rows):
                raise ValueError("Per-load order mismatch")
            times={name:sum(row["case"]["weight"]*row["samples"][i]["us"][name] for row in rows) for name in names}
            samples.append(dict(seed=i,repetition_index=i,seed_note="Synthetic pairing id; actual component seeds below",
                                component_seeds=[row["samples"][i]["seed"] for row in rows],order=order,us=times))
        reduced=summarize_case(dict(case=dict(load=load,kind="contract-weighted gross load sum"),samples=samples),orders)
        reduced["weighted_samples"]=samples
        summaries.append(reduced)
    ratios={name:math.exp(statistics.mean(math.log(row["median_paired_gross_ratio"][name]) for row in summaries)) for name in names[1:]}
    return dict(loads=summaries,geometric_mean_over_loads_control_over_candidate=ratios,
                qualification="Weighted gross diagnostic across equally weighted loads; no official score or confidence interval")
