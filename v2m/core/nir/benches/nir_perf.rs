use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use v2m_formats::nir::{BitRef, BitRefNet, Module, Net, Node, NodeOp, Port, PortDirection};
use v2m_nir::{normalize_module, ModuleGraph};

const SYNTHETIC_SIZES: &[usize] = &[10_000, 50_000, 100_000];
const TARGET_ALLOC_PER_NODE: f64 = 4.0;
const TARGET_BYTES_PER_NODE: f64 = 256.0;
const TARGET_NS_SPREAD_PERCENT: f64 = 15.0;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

struct TrackingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = System.realloc(ptr, old_layout, new_size);
        if !ptr.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        ptr
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AllocStats {
    allocations: u64,
    allocated_bytes: u64,
}

fn reset_alloc_stats() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn snapshot_alloc_stats() -> AllocStats {
    AllocStats {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

fn synthetic_modules() -> &'static Vec<(usize, Module)> {
    static MODULES: OnceLock<Vec<(usize, Module)>> = OnceLock::new();
    MODULES.get_or_init(|| {
        SYNTHETIC_SIZES
            .iter()
            .copied()
            .map(|size| (size, build_synthetic_module(size)))
            .collect()
    })
}

fn build_synthetic_module(node_count: usize) -> Module {
    assert!(
        node_count > 0,
        "synthetic modules require at least one node"
    );

    let mut ports = BTreeMap::new();
    ports.insert(
        "a".to_string(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "b".to_string(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "out".to_string(),
        Port {
            dir: PortDirection::Output,
            bits: 1,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    nets.insert(
        "a".to_string(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "b".to_string(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "out".to_string(),
        Net {
            bits: 1,
            attrs: None,
        },
    );

    let mut nodes = BTreeMap::new();
    let mut previous_a = "a".to_string();
    let mut previous_b = "b".to_string();

    for index in 0..node_count {
        let node_name = format!("node_{index}");
        let output_net = if index + 1 == node_count {
            "out".to_string()
        } else {
            let net_name = format!("n{index}");
            nets.insert(
                net_name.clone(),
                Net {
                    bits: 1,
                    attrs: None,
                },
            );
            net_name
        };

        let mut pin_map = BTreeMap::new();
        pin_map.insert(
            "A".to_string(),
            BitRef::Net(BitRefNet {
                net: previous_a.clone(),
                lsb: 0,
                msb: 0,
            }),
        );
        pin_map.insert(
            "B".to_string(),
            BitRef::Net(BitRefNet {
                net: previous_b.clone(),
                lsb: 0,
                msb: 0,
            }),
        );
        pin_map.insert(
            "Y".to_string(),
            BitRef::Net(BitRefNet {
                net: output_net.clone(),
                lsb: 0,
                msb: 0,
            }),
        );

        nodes.insert(
            node_name.clone(),
            Node {
                uid: node_name,
                op: NodeOp::Xor,
                width: 1,
                pin_map,
                params: None,
                attrs: None,
            },
        );

        previous_b = previous_a;
        previous_a = output_net;
    }

    Module { ports, nets, nodes }
}

fn ensure_baseline_logged() {
    static BASELINE: Once = Once::new();
    BASELINE.call_once(|| {
        log_target_metrics(synthetic_modules());
    });
}

fn log_target_metrics(modules: &[(usize, Module)]) {
    println!("=== NIR performance baseline (synthetic netlists) ===");
    println!(
        "Target: ≤ {TARGET_ALLOC_PER_NODE:.1} allocations/node, ≤ {TARGET_BYTES_PER_NODE:.1} B/node, and linear scaling within ±{TARGET_NS_SPREAD_PERCENT:.1}%"
    );

    let mut graph_ns_per_node = Vec::new();
    let mut norm_ns_per_node = Vec::new();

    for (nodes, module) in modules {
        let (graph_duration, graph_allocs) = measure_module_graph(module);
        let (norm_duration, norm_allocs) = measure_normalization(module);

        let graph_ns = graph_duration.as_secs_f64() * 1e9 / *nodes as f64;
        let graph_alloc_per_node = graph_allocs.allocations as f64 / *nodes as f64;
        let graph_bytes_per_node = graph_allocs.allocated_bytes as f64 / *nodes as f64;
        println!(
            "ModuleGraph::from_module: {nodes} nodes -> {graph_duration:.2?} total ({graph_ns:.1} ns/node, {graph_alloc_per_node:.2} alloc/node, {graph_bytes_per_node:.1} B/node)"
        );
        graph_ns_per_node.push(graph_ns);

        let norm_ns = norm_duration.as_secs_f64() * 1e9 / *nodes as f64;
        let norm_alloc_per_node = norm_allocs.allocations as f64 / *nodes as f64;
        let norm_bytes_per_node = norm_allocs.allocated_bytes as f64 / *nodes as f64;
        println!(
            "normalize_module:          {nodes} nodes -> {norm_duration:.2?} total ({norm_ns:.1} ns/node, {norm_alloc_per_node:.2} alloc/node, {norm_bytes_per_node:.1} B/node)"
        );
        norm_ns_per_node.push(norm_ns);
    }

    let graph_spread = percent_spread(&graph_ns_per_node);
    let norm_spread = percent_spread(&norm_ns_per_node);
    println!(
        "Observed per-node time spread: ModuleGraph {graph_spread:.2}% | normalize {norm_spread:.2}%"
    );
    println!("========================================================\n");
}

fn percent_spread(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let min = values
        .iter()
        .fold(f64::INFINITY, |acc, &value| acc.min(value));
    let max = values
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
    if max == 0.0 {
        0.0
    } else {
        ((max - min) / max) * 100.0
    }
}

fn measure_module_graph(module: &Module) -> (Duration, AllocStats) {
    reset_alloc_stats();
    let start = Instant::now();
    let graph = ModuleGraph::from_module(module).expect("build module graph");
    let duration = start.elapsed();
    black_box(&graph);
    drop(graph);
    (duration, snapshot_alloc_stats())
}

fn measure_normalization(module: &Module) -> (Duration, AllocStats) {
    reset_alloc_stats();
    let start = Instant::now();
    let normalized = normalize_module(module).expect("normalize module");
    let duration = start.elapsed();
    black_box(&normalized);
    drop(normalized);
    (duration, snapshot_alloc_stats())
}

fn module_graph_bench(c: &mut Criterion) {
    ensure_baseline_logged();
    let modules = synthetic_modules();

    let mut group = c.benchmark_group("module_graph_from_module");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for (nodes, module) in modules.iter() {
        group.throughput(Throughput::Elements(*nodes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nodes), module, |b, module| {
            b.iter(|| {
                let graph = ModuleGraph::from_module(module).expect("build module graph");
                black_box(graph);
            });
        });
    }

    group.finish();
}

fn normalization_bench(c: &mut Criterion) {
    ensure_baseline_logged();
    let modules = synthetic_modules();

    let mut group = c.benchmark_group("normalize_module");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for (nodes, module) in modules.iter() {
        group.throughput(Throughput::Elements(*nodes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nodes), module, |b, module| {
            b.iter(|| {
                let normalized = normalize_module(module).expect("normalize module");
                black_box(normalized);
            });
        });
    }

    group.finish();
}

fn benchmark_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
}

criterion_group! {
    name = nir_perf;
    config = benchmark_config();
    targets = module_graph_bench, normalization_bench
}
criterion_main!(nir_perf);
