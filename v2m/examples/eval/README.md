# Evaluator CLI quick start

This directory contains a minimal smoke test for the `v2m nir eval` command. The example drives the bundled `full_adder` design with four stimulus vectors and demonstrates how to capture profiling metrics.

## Files

* `full_adder.inputs.json` — one cycle of input vectors for the full adder (`a`, `b`, `cin`).
* `full_adder.expected.json` — golden outputs (`sum`, `cout`) for the same vectors.

## Run the evaluation

From the repository root:

```bash
cargo run --package v2m-cli --bin v2m -- \
  nir eval --nir examples/nir/full_adder.json \
  --inputs examples/eval/full_adder.inputs.json \
  --out examples/eval/full_adder.expected.json \
  --profile --profile-json examples/eval/profile.json
```

The command prints a JSON blob of the simulated outputs, verifies them against the golden file, and finally shows a kernel-by-kernel profile. The `--profile-json` flag writes the same metrics to `examples/eval/profile.json` for inspection or automation.

Use this example as a template when bringing up new designs: copy the input skeleton, update the port names, and tweak the vector count with `--vectors` as needed.
