# NIR synthetic netlist performance baseline

This run captures the initial performance envelope for building module graphs and normalizing
large synthetic netlists so we can detect regressions quickly. Measurements were taken on the
CI container with `cargo bench --bench nir_perf` after enabling the Criterion-based benchmark
suite.

## Target metrics

- **Allocation share:** keep the allocator at or below **4 allocations/node** and
  **256 B/node** while building module graphs, and approximately **51 allocations/node**
  and **4 kB/node** for normalization. These values reflect the current steady-state behavior
  of the synthetic workload and act as upper bounds for future changes.
- **Linear scaling:** maintain per-node runtime within ±15% as the netlist grows from 10k to
  100k nodes. The baseline stays within 7.4% for module graph construction and 8.5% for
  normalization.

## Baseline results

| Stage                        | Nodes  | Total time | ns/node | alloc/node | bytes/node |
|------------------------------|-------:|-----------:|--------:|-----------:|-----------:|
| ModuleGraph::from_module     | 10,000 |    92.54ms |  9,254.5|      28.00 |     1,822.4|
| normalize_module             | 10,000 |   105.17ms | 10,517.0|      51.01 |     3,966.5|
| ModuleGraph::from_module     | 50,000 |   478.08ms |  9,561.7|      28.00 |     1,807.0|
| normalize_module             | 50,000 |   534.70ms | 10,694.1|      51.00 |     3,837.3|
| ModuleGraph::from_module     |100,000 |   885.73ms |  8,857.3|      28.00 |     1,807.7|
| normalize_module             |100,000 |  1.15s     | 11,499.4|      51.00 |     3,839.0|

Per-node runtime spread: **7.37%** (ModuleGraph::from_module) and **8.54%** (normalize_module).

## CI hook

A dedicated `nir-perf` job is available via `workflow_dispatch` to rerun these measurements
on demand. Triggering the workflow publishes the full Criterion report as a build artifact,
allowing the team to compare subsequent runs against this baseline.
