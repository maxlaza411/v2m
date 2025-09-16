# Evaluator v2 design

## Purpose and scope

Evaluator v2 is the golden software model for Minecraft circuit designs in NIR. It provides deterministic, bit-parallel simulation that is fast enough for continuous integration and strong enough to serve as the oracle for other back ends. This document locks the public API, key data structures, and execution model that the `v2m-evaluator` crate implements.

The evaluator sits beside existing NIR passes and executes the top-level module with realistic tick semantics:

```
RTL ─▶ NIR Lint ─▶ NIR Opt ─▶ Retiming
           │            │          │
           └────────────┴──────┬───┘
                               ▼
                        Evaluator v2
                               ▲
             Tests / property checks / equivalence
```

Evaluator v2 must:

* execute many stimulus vectors in parallel without sacrificing determinism,
* support arbitrarily wide buses and wide DFFs/latches without copies,
* model synchronous and asynchronous reset precisely at tick boundaries, and
* produce bit-identical outputs so other passes can trust it as a reference.

The initial implementation focuses on sequential logic driven by a single clock. Event-driven optimisations, X-propagation, and multi-phase clocks remain extensible but are not required for v2.

## Batch model

Evaluator v2 operates on *batches* of stimulus vectors. A batch size is the number of independent simulations evaluated together. Each vector corresponds to one row of input/output samples over time. We process vectors in units of 64 bits to leverage native word operations.

Two axes control the packed representation of a signal:

* **Bit lanes** — a `width`-bit signal requires `L = ceil(width / 64)` lanes. A lane stores up to 64 adjacent bits from the signal.
* **Vector words** — `N` vectors are processed in groups of 64, yielding `W = ceil(N / 64)` words per lane. Each bit inside a word is one vector.

The storage for a signal therefore forms an `L × W` matrix of `u64`. All logic kernels operate on this matrix layout.

### Packed storage

`Packed` is the reusable storage arena for these matrices. A `Packed` value owns an SoA buffer sized for a specific batch. Slices inside the arena are described by `PackedIndex`, which stores the offset and lane count for a signal. The same index can be reused across multiple `Packed` values that share a layout (for example the current and next register images).

```rust
pub struct Packed {
    num_vectors: usize,
    words_per_lane: usize,
    storage: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedIndex {
    pub(crate) offset: usize,
    pub(crate) lanes: usize,
}
```

Key helpers:

* `Packed::new(num_vectors)` creates an empty arena for the batch.
* `Packed::allocate(width_bits)` reserves space for a signal and returns a `PackedIndex`.
* `Packed::duplicate_layout()` clones the arena shape while zeroing the data.
* `Packed::slice(index)` / `slice_mut(index)` provide read/write access to a signal.
* `Packed::copy_from(other)` copies the entire arena after verifying that the layouts match.

`PackedBitMask` reuses the vector-word layout to represent per-vector reset states. A word count of `ceil(num_vectors / 64)` keeps masks compatible with the rest of the pipeline.

## Public API

The `v2m-evaluator` crate exposes a minimal surface today while leaving room to add specialised helpers (packed iterators, operator kernels, etc.) as the implementation grows.

```rust
use v2m_evaluator::{Evaluator, Packed, PackedBitMask, SimOptions};

let nir = v2m_formats::load_nir("design.json")?;
let mut eval = Evaluator::new(&nir, 512, SimOptions::default())?;

let mut inputs = Packed::new(512);
// ... pack port vectors into `inputs`
let reset = PackedBitMask::new(512);

let outputs = eval.tick(&inputs, &reset)?;
```

### Configuration

`SimOptions` collects behavioural switches that do not affect the arena layout. For v2 we fix two flags:

* `allow_x` — allow dual-rail X/Z tracking (off by default, planned for later).
* `async_reset_is_high` — whether asynchronous reset asserts on a high level.

More options (event-driven threshold, tracing callbacks, power-aware kernels) can be added without breaking the ABI.

### Building an evaluator

`Evaluator::new(nir, num_vectors, opts)` performs the one-time compilation step:

1. Find the top module in the NIR design and construct a `ModuleGraph`.
2. Build a combinational topological order for the graph.
3. Allocate `Packed` arenas for nets, temporary values, current/next registers, and port staging areas. Allocation uses the `Packed` API above.
4. Prebind each node input/output pin to a `PackedIndex` so combinational kernels can index directly into the arenas without resolving text names.

Failure to build the graph or detect a combinational loop surfaces as an `Error` from `Evaluator::new`.

### Tick execution

`Evaluator` follows strict tick semantics so all clients observe the same behaviour:

1. `set_inputs` copies a batch of packed port values into the input staging arena.
2. `comb_eval` visits nodes in topological order, reading from net/register slices and writing the results into net slices. Sequential elements contribute their `Q` slices as stable inputs during this phase.
3. `step_clock` updates each register slice based on the per-vector reset mask and the pre-bound `D` slices, swapping the current/next register arenas afterwards.
4. `get_outputs` exposes the packed output staging arena; `tick` wraps the above steps into a single convenience call.

Reset masks share the same vector packing so per-vector reset sequences can be expressed efficiently. Asynchronous resets update `regs_cur` immediately in `step_clock`; synchronous resets select the reset value instead of `D`.

### Error handling

`Evaluator::new` and `Evaluator::tick` return `Result` with a domain-specific `Error` enum. The error variants forward issues from `v2m-nir` (graph construction, combinational loops) and report shape mismatches when packed buffers do not align.

Future extensions (event-driven short-circuiting, trace recording, activity stats) can reuse the same error channel without modifying the core API.

## Integration points

* **Pass validation** — each NIR optimisation pass can re-run the evaluator on cached stimuli and compare packed outputs to ensure no functional drift.
* **Retiming checks** — sequential equivalence becomes a bounded model check: run pre- and post-retime designs for `K` cycles with identical inputs and compare outputs per tick.
* **Backend comparison** — datapack interpreters and the redstone emulator can consume the same packed vectors and assert cycle-by-cycle equality against the evaluator’s outputs.

## Testing strategy

* Functional tests instantiate the evaluator on a library of NIR examples (full adder, ALU, FIFOs) and compare against Yosys or previously captured golden results.
* Reset edge cases: verify synchronous vs. asynchronous reset behaviour, mixed polarities, and partial-vector reset masks.
* Packing edge cases: slices spanning lane boundaries, concatenation across nets, carries at 64-bit word edges.
* Performance target: evaluate a 10k-node, 32-bit ALU over 4096 vectors and achieve at least a 10× speedup versus scalar execution.

This design document is the contract for the forthcoming implementation. The stub crate added alongside this document ensures the API is stable enough for other crates to start integrating with Evaluator v2.
