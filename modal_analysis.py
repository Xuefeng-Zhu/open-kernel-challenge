"""Offline diagnostic summaries. Retain every sample; no judge score or CI."""
import itertools
import math
import statistics


def quantile(values, fraction):
    ordered=sorted(values)
    index=(len(ordered)-1)*fraction
    low=math.floor(index);high=math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def summarize_case(case):
    samples=case["samples"]
    names=("candidate","r10","sglang")
    if not samples or len(samples)%6:
        raise ValueError("Complete six-order blocks required")
    pairs=[]
    expected=set(itertools.permutations(names))
    for base in range(0,len(samples),6):
        block=samples[base:base+6]
        lookup={tuple(sample["order"]):i for i,sample in enumerate(block)}
        if set(lookup)!=expected:
            raise ValueError("Each block must contain all six execution orders")
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
