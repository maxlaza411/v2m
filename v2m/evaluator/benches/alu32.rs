use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use v2m_evaluator::{hash_packed_outputs, Evaluator, PackedBitMask, SimOptions};
use v2m_formats::nir::{
    BitRef, BitRefConcat, BitRefNet, Module, Net, Nir, Node, NodeOp, Port, PortDirection,
};

const NUM_VECTORS: usize = 4096;
const STAGES: usize = 1500;
const DEFAULT_ITERATIONS: usize = 20;
const INPUT_SEED: u64 = 0xB3F3_CAFE;
const MIN_THROUGHPUT_TARGET: f64 = 100_000.0; // vectors per second

fn main() {
    let config = BenchConfig::parse();
    if cfg!(debug_assertions) {
        println!("Skipping benchmark in debug mode; run with --release to measure performance.");
        return;
    }
    let nir = build_benchmark_alu32(STAGES);
    let module = nir
        .modules
        .get(nir.top.as_str())
        .expect("top module must exist");
    let node_count = module.nodes.len();
    assert!(
        node_count >= 10_000,
        "expected at least 10k nodes, found {node_count}"
    );

    let report = run_benchmark(&nir, module, config.iterations);

    println!(
        "Evaluated {} nodes for {} iterations over {} vectors in {:.3}s (throughput {:.2} Mvec/s)",
        node_count,
        config.iterations,
        NUM_VECTORS,
        report.total_time.as_secs_f64(),
        report.throughput / 1_000_000.0
    );

    if let Err(error) = write_report(&config.output_path, node_count, &report) {
        eprintln!(
            "failed to write benchmark report `{}`: {error}",
            config.output_path.display()
        );
    } else {
        println!("Report written to {}", config.output_path.display());
    }

    if report.throughput < config.min_throughput {
        eprintln!(
            "throughput {:.2} vectors/s below target {:.2} vectors/s",
            report.throughput, config.min_throughput
        );
        std::process::exit(1);
    }
}

struct BenchConfig {
    output_path: PathBuf,
    min_throughput: f64,
    iterations: usize,
}

impl BenchConfig {
    fn parse() -> Self {
        let mut output_path = default_output_path();
        let mut min_throughput = MIN_THROUGHPUT_TARGET;
        let mut iterations = DEFAULT_ITERATIONS;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--bench" | "--test" | "--nocapture" | "--quiet" => {}
                value if value.starts_with("--color") => {
                    if !value.contains('=') {
                        let _ = args.next();
                    }
                }
                "--output" => {
                    let path = args.next().unwrap_or_else(|| {
                        eprintln!("--output requires a path argument");
                        std::process::exit(2);
                    });
                    output_path = PathBuf::from(path);
                }
                "--min-throughput" => {
                    let value = args.next().unwrap_or_else(|| {
                        eprintln!("--min-throughput requires a numeric argument");
                        std::process::exit(2);
                    });
                    min_throughput = value.parse().unwrap_or_else(|_| {
                        eprintln!("invalid value for --min-throughput: {value}");
                        std::process::exit(2);
                    });
                }
                "--iterations" => {
                    let value = args.next().unwrap_or_else(|| {
                        eprintln!("--iterations requires a numeric argument");
                        std::process::exit(2);
                    });
                    iterations = value.parse().unwrap_or_else(|_| {
                        eprintln!("invalid value for --iterations: {value}");
                        std::process::exit(2);
                    });
                }
                other if other.starts_with("--") => {
                    eprintln!("unrecognised option `{other}`");
                    std::process::exit(2);
                }
                _ => {
                    // Positional arguments are not expected.
                    eprintln!("unexpected positional argument `{arg}`");
                    std::process::exit(2);
                }
            }
        }

        Self {
            output_path,
            min_throughput,
            iterations,
        }
    }
}

struct BenchReport {
    iterations: usize,
    total_time: Duration,
    throughput: f64,
    output_hash: [u8; 32],
}

fn run_benchmark(nir: &Nir, module: &Module, iterations: usize) -> BenchReport {
    let mut evaluator = Evaluator::new(nir, NUM_VECTORS, SimOptions::default())
        .expect("failed to construct evaluator");
    let mut rng = StdRng::seed_from_u64(INPUT_SEED);
    let stimulus = generate_random_inputs(module, NUM_VECTORS, &mut rng);
    let packed_inputs = evaluator
        .pack_inputs_from_biguints(&stimulus)
        .expect("failed to pack stimulus");
    let reset_mask = PackedBitMask::new(NUM_VECTORS);

    // Warm-up run to populate caches.
    let warm_outputs = evaluator
        .tick(&packed_inputs, &reset_mask)
        .expect("warm-up tick");
    black_box(&warm_outputs);

    let start = Instant::now();
    let mut last_outputs = warm_outputs;
    for _ in 0..iterations {
        let outputs = evaluator
            .tick(&packed_inputs, &reset_mask)
            .expect("evaluation tick");
        black_box(&outputs);
        last_outputs = outputs;
    }
    let elapsed = start.elapsed();

    let vectors_processed = iterations as f64 * NUM_VECTORS as f64;
    let throughput = vectors_processed / elapsed.as_secs_f64();
    let hash = hash_packed_outputs(&last_outputs);

    BenchReport {
        iterations,
        total_time: elapsed,
        throughput,
        output_hash: hash,
    }
}

fn write_report(path: &PathBuf, node_count: usize, report: &BenchReport) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;
    writeln!(file, "# Evaluator ALU32 Benchmark")?;
    writeln!(file)?;
    writeln!(file, "| Metric | Value |")?;
    writeln!(file, "| --- | --- |")?;
    writeln!(file, "| Nodes | {node_count} |")?;
    writeln!(file, "| Stages | {STAGES} |")?;
    writeln!(file, "| Vectors | {NUM_VECTORS} |")?;
    writeln!(file, "| Iterations | {} |", report.iterations)?;
    writeln!(
        file,
        "| Total time | {:.6} s |",
        report.total_time.as_secs_f64()
    )?;
    writeln!(
        file,
        "| Time per tick | {:.6} s |",
        report.total_time.as_secs_f64() / report.iterations as f64
    )?;
    writeln!(
        file,
        "| Throughput | {:.3} Mvec/s |",
        report.throughput / 1_000_000.0
    )?;
    writeln!(
        file,
        "| Output hash | {} |",
        hex_digest(&report.output_hash)
    )?;
    Ok(())
}

fn default_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/benchmarks/alu32.md")
}

fn build_benchmark_alu32(stages: usize) -> Nir {
    assert!(stages > 0, "benchmark requires at least one stage");

    let mut ports = BTreeMap::new();
    ports.insert(
        "a".to_string(),
        Port {
            dir: PortDirection::Input,
            bits: 32,
            attrs: None,
        },
    );
    ports.insert(
        "b".to_string(),
        Port {
            dir: PortDirection::Input,
            bits: 32,
            attrs: None,
        },
    );
    ports.insert(
        "op".to_string(),
        Port {
            dir: PortDirection::Input,
            bits: 2,
            attrs: None,
        },
    );
    ports.insert(
        "y".to_string(),
        Port {
            dir: PortDirection::Output,
            bits: 32,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    nets.insert(
        "a".to_string(),
        Net {
            bits: 32,
            attrs: None,
        },
    );
    nets.insert(
        "b".to_string(),
        Net {
            bits: 32,
            attrs: None,
        },
    );
    nets.insert(
        "op".to_string(),
        Net {
            bits: 2,
            attrs: None,
        },
    );
    nets.insert(
        "y".to_string(),
        Net {
            bits: 32,
            attrs: None,
        },
    );

    for stage in 0..stages {
        let prefix = format!("stage_{stage:04}");
        for suffix in ["sum", "and", "or", "xor", "lo", "hi", "out"] {
            nets.insert(
                format!("{prefix}_{suffix}"),
                Net {
                    bits: 32,
                    attrs: None,
                },
            );
        }
    }

    let mut nodes = BTreeMap::new();
    let sel_low = BitRef::Concat(BitRefConcat {
        concat: vec![net_range("op", 0, 0)],
    });
    let sel_high = BitRef::Concat(BitRefConcat {
        concat: vec![net_range("op", 1, 1)],
    });

    for stage in 0..stages {
        let prefix = format!("stage_{stage:04}");
        let prev_out = if stage == 0 {
            net_full("a")
        } else {
            net_full(&format!("stage_{:04}_out", stage - 1))
        };
        let alt_input = if stage % 2 == 0 { "b" } else { "a" };
        let input_b = net_full(alt_input);

        let sum_net = net_full(&format!("{prefix}_sum"));
        let and_net = net_full(&format!("{prefix}_and"));
        let or_net = net_full(&format!("{prefix}_or"));
        let xor_net = net_full(&format!("{prefix}_xor"));
        let lo_net = net_full(&format!("{prefix}_lo"));
        let hi_net = net_full(&format!("{prefix}_hi"));
        let out_net = net_full(&format!("{prefix}_out"));

        nodes.insert(
            format!("{prefix}_add"),
            Node {
                uid: format!("{prefix}_add"),
                op: NodeOp::Add,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), prev_out.clone()),
                    ("B".to_string(), input_b.clone()),
                    ("Y".to_string(), sum_net.clone()),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            format!("{prefix}_and"),
            Node {
                uid: format!("{prefix}_and"),
                op: NodeOp::And,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), prev_out.clone()),
                    ("B".to_string(), input_b.clone()),
                    ("Y".to_string(), and_net.clone()),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            format!("{prefix}_or"),
            Node {
                uid: format!("{prefix}_or"),
                op: NodeOp::Or,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), prev_out.clone()),
                    ("B".to_string(), input_b.clone()),
                    ("Y".to_string(), or_net.clone()),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            format!("{prefix}_xor"),
            Node {
                uid: format!("{prefix}_xor"),
                op: NodeOp::Xor,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), prev_out.clone()),
                    ("B".to_string(), input_b.clone()),
                    ("Y".to_string(), xor_net.clone()),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            format!("{prefix}_mux_lo"),
            Node {
                uid: format!("{prefix}_mux_lo"),
                op: NodeOp::Mux,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), sum_net.clone()),
                    ("B".to_string(), and_net.clone()),
                    ("S".to_string(), sel_low.clone()),
                    ("Y".to_string(), lo_net.clone()),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            format!("{prefix}_mux_hi"),
            Node {
                uid: format!("{prefix}_mux_hi"),
                op: NodeOp::Mux,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), or_net.clone()),
                    ("B".to_string(), xor_net.clone()),
                    ("S".to_string(), sel_low.clone()),
                    ("Y".to_string(), hi_net.clone()),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            format!("{prefix}_mux_out"),
            Node {
                uid: format!("{prefix}_mux_out"),
                op: NodeOp::Mux,
                width: 32,
                pin_map: BTreeMap::from([
                    ("A".to_string(), lo_net),
                    ("B".to_string(), hi_net),
                    ("S".to_string(), sel_high.clone()),
                    ("Y".to_string(), out_net),
                ]),
                params: None,
                attrs: None,
            },
        );
    }

    let final_source = format!("stage_{:04}_out", stages - 1);
    nodes.insert(
        "export_y".to_string(),
        Node {
            uid: "export_y".to_string(),
            op: NodeOp::Slice,
            width: 32,
            pin_map: BTreeMap::from([
                ("A".to_string(), net_full(&final_source)),
                ("Y".to_string(), net_full("y")),
            ]),
            params: None,
            attrs: None,
        },
    );

    let mut modules = BTreeMap::new();
    modules.insert("alu32_bench".to_string(), Module { ports, nets, nodes });

    Nir {
        v: "nir-1.1".to_string(),
        design: "alu32_bench".to_string(),
        top: "alu32_bench".to_string(),
        attrs: None,
        modules,
        generator: None,
        cmdline: None,
        source_digest_sha256: None,
    }
}

fn generate_random_inputs(
    module: &Module,
    num_vectors: usize,
    rng: &mut StdRng,
) -> HashMap<String, Vec<BigUint>> {
    let mut map = HashMap::new();
    for (name, port) in &module.ports {
        if matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
            let width = port.bits as usize;
            let mut values = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                values.push(random_biguint(width, rng));
            }
            map.insert(name.clone(), values);
        }
    }
    map
}

fn random_biguint(width_bits: usize, rng: &mut StdRng) -> BigUint {
    if width_bits == 0 {
        return BigUint::default();
    }

    let byte_len = (width_bits + 7) / 8;
    let mut bytes = vec![0u8; byte_len];
    rng.fill_bytes(&mut bytes);

    let excess_bits = byte_len * 8 - width_bits;
    if excess_bits > 0 {
        let keep = 8 - excess_bits;
        let mask = if keep == 0 {
            0
        } else {
            (1u16 << keep) as u8 - 1
        };
        if let Some(last) = bytes.last_mut() {
            *last &= mask;
        }
    }

    BigUint::from_bytes_le(&bytes)
}

fn net_full(name: &str) -> BitRef {
    net_range(name, 0, 31)
}

fn net_range(name: &str, lsb: u32, msb: u32) -> BitRef {
    BitRef::Net(BitRefNet {
        net: name.to_string(),
        lsb,
        msb,
    })
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut text, "{:02x}", byte);
    }
    text
}
