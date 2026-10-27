# Interpretation references

Runtime support and the actual values come from the preserved installed NCU
help, metric queries and raw CSV. Documentation supplies terminology, not an
alternative measurement or proof of a bottleneck.

- [NVIDIA CUPTI 13.0 metric mappings](https://docs.nvidia.com/cupti/13.0.0/main/main.html)
  maps long-scoreboard to memory-dependency stalls and DRAM utilization to
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`. A stall percentage does
  not locate the responsible instruction or memory level.
- [NVIDIA Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)
  discusses bandwidth, available requests, active warps and instruction-level
  parallelism. Bytes in flight cannot be established from a single counter.
  These relationships motivate a conditional hypothesis; they do not establish
  causal transfer from the eager profile to the timed graph.
