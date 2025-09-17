use std::collections::{BTreeMap, HashMap};

use num_bigint::BigUint;
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
use v2m_evaluator::{Evaluator, PackedBitMask, SimOptions};
use v2m_formats::nir::{BitRef, BitRefNet, Module, Net, Nir, Node, NodeOp, Port, PortDirection};
use v2m_nir::{normalize_module, NormalizedModule, NormalizedNodeKind};

const NUM_VECTORS: usize = 256;

#[derive(Clone)]
struct Signal {
    name: String,
    width: usize,
}

#[derive(Clone, Copy)]
enum OpKind {
    Not,
    And,
    Or,
    Xor,
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]
    #[test]
    fn strash_preserves_behavior(seed in any::<u64>()) {
        let nir = build_random_nir(seed);
        let module = nir
            .modules
            .get(nir.top.as_str())
            .expect("top module must exist");

        let mut evaluator = Evaluator::new(&nir, NUM_VECTORS, SimOptions::default())
            .expect("construct evaluator");

        let inputs = generate_random_inputs(module, NUM_VECTORS, seed);
        let packed_inputs = evaluator
            .pack_inputs_from_biguints(&inputs)
            .expect("pack random inputs");
        let reset_mask = PackedBitMask::new(NUM_VECTORS);
        let outputs = evaluator
            .tick(&packed_inputs, &reset_mask)
            .expect("evaluate original design");
        let baseline = evaluator
            .unpack_outputs_to_biguints(&outputs)
            .expect("unpack baseline outputs");

        let normalized = normalize_module(module).expect("normalize module");
        let normalized_outputs = evaluate_normalized(&normalized, &inputs)
            .expect("evaluate normalized module");

        prop_assert_eq!(baseline, normalized_outputs);
    }
}

#[test]
fn random_nir_seed_is_stable() {
    let seed = 0xDEAD_BEEF_u64;
    let first = build_random_nir(seed);
    let second = build_random_nir(seed);
    assert_eq!(first, second);
}

fn build_random_nir(seed: u64) -> Nir {
    let mut rng = StdRng::seed_from_u64(seed);

    let input_ports = rng.gen_range(1..=4);
    let mut ports = BTreeMap::new();
    let mut nets = BTreeMap::new();
    let mut nodes = BTreeMap::new();

    let mut signals = Vec::new();
    for index in 0..input_ports {
        let width = rng.gen_range(1..=8);
        let name = format!("in{index}");
        ports.insert(
            name.clone(),
            Port {
                dir: PortDirection::Input,
                bits: width as u32,
                attrs: None,
            },
        );
        nets.insert(
            name.clone(),
            Net {
                bits: width as u32,
                attrs: None,
            },
        );
        signals.push(Signal { name, width });
    }

    let target_nodes = rng.gen_range(0..=8);
    let mut node_counter = 0usize;

    for _ in 0..target_nodes {
        if signals.is_empty() {
            break;
        }

        let mut width_counts = HashMap::new();
        for signal in &signals {
            *width_counts.entry(signal.width).or_insert(0usize) += 1;
        }
        let mut binary_widths: Vec<usize> = width_counts
            .into_iter()
            .filter_map(|(width, count)| if count >= 2 { Some(width) } else { None })
            .collect();

        let mut possible_ops = Vec::new();
        if !signals.is_empty() {
            possible_ops.push(OpKind::Not);
        }
        if !binary_widths.is_empty() {
            possible_ops.extend_from_slice(&[OpKind::And, OpKind::Or, OpKind::Xor]);
        }
        if possible_ops.is_empty() {
            break;
        }

        let op = possible_ops[rng.gen_range(0..possible_ops.len())];
        let node_name = format!("n{node_counter}");
        node_counter += 1;

        match op {
            OpKind::Not => {
                let signal_index = rng.gen_range(0..signals.len());
                let input = signals[signal_index].clone();
                let width = input.width;
                nets.insert(
                    node_name.clone(),
                    Net {
                        bits: width as u32,
                        attrs: None,
                    },
                );
                let mut pin_map = BTreeMap::new();
                pin_map.insert(
                    "A".to_string(),
                    BitRef::Net(BitRefNet {
                        net: input.name,
                        lsb: 0,
                        msb: width as u32 - 1,
                    }),
                );
                pin_map.insert(
                    "Y".to_string(),
                    BitRef::Net(BitRefNet {
                        net: node_name.clone(),
                        lsb: 0,
                        msb: width as u32 - 1,
                    }),
                );
                nodes.insert(
                    node_name.clone(),
                    Node {
                        uid: node_name.clone(),
                        op: NodeOp::Not,
                        width: width as u32,
                        pin_map,
                        params: None,
                        attrs: None,
                    },
                );
                signals.push(Signal {
                    name: node_name,
                    width,
                });
            }
            OpKind::And | OpKind::Or | OpKind::Xor => {
                binary_widths.sort_unstable();
                let width = binary_widths[rng.gen_range(0..binary_widths.len())];
                let candidates: Vec<_> = signals
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, sig)| {
                        if sig.width == width {
                            Some((idx, sig.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                if candidates.len() < 2 {
                    continue;
                }
                let (_, a_signal) = candidates[rng.gen_range(0..candidates.len())].clone();
                let (_, b_signal) = if candidates.len() == 2 {
                    if rng.gen_bool(0.5) {
                        candidates[0].clone()
                    } else {
                        candidates[1].clone()
                    }
                } else {
                    candidates[rng.gen_range(0..candidates.len())].clone()
                };
                let mut pin_map = BTreeMap::new();
                pin_map.insert(
                    "A".to_string(),
                    BitRef::Net(BitRefNet {
                        net: a_signal.name,
                        lsb: 0,
                        msb: width as u32 - 1,
                    }),
                );
                pin_map.insert(
                    "B".to_string(),
                    BitRef::Net(BitRefNet {
                        net: b_signal.name,
                        lsb: 0,
                        msb: width as u32 - 1,
                    }),
                );
                pin_map.insert(
                    "Y".to_string(),
                    BitRef::Net(BitRefNet {
                        net: node_name.clone(),
                        lsb: 0,
                        msb: width as u32 - 1,
                    }),
                );
                nets.insert(
                    node_name.clone(),
                    Net {
                        bits: width as u32,
                        attrs: None,
                    },
                );
                let op_kind = match op {
                    OpKind::And => NodeOp::And,
                    OpKind::Or => NodeOp::Or,
                    OpKind::Xor => NodeOp::Xor,
                    OpKind::Not => unreachable!(),
                };
                nodes.insert(
                    node_name.clone(),
                    Node {
                        uid: node_name.clone(),
                        op: op_kind,
                        width: width as u32,
                        pin_map,
                        params: None,
                        attrs: None,
                    },
                );
                signals.push(Signal {
                    name: node_name,
                    width,
                });
            }
        }
    }

    if signals.is_empty() {
        let name = "in0".to_string();
        signals.push(Signal { name, width: 1 });
    }

    let max_outputs = signals.len().min(4).max(1);
    let output_ports = rng.gen_range(1..=max_outputs);
    for index in 0..output_ports {
        let signal_index = rng.gen_range(0..signals.len());
        let signal = signals[signal_index].clone();
        let port_name = format!("out{index}");
        let node_name = format!("out_node{index}");
        ports.insert(
            port_name.clone(),
            Port {
                dir: PortDirection::Output,
                bits: signal.width as u32,
                attrs: None,
            },
        );
        nets.insert(
            port_name.clone(),
            Net {
                bits: signal.width as u32,
                attrs: None,
            },
        );
        let mut pin_map = BTreeMap::new();
        pin_map.insert(
            "A".to_string(),
            BitRef::Net(BitRefNet {
                net: signal.name,
                lsb: 0,
                msb: signal.width as u32 - 1,
            }),
        );
        pin_map.insert(
            "Y".to_string(),
            BitRef::Net(BitRefNet {
                net: port_name.clone(),
                lsb: 0,
                msb: signal.width as u32 - 1,
            }),
        );
        nodes.insert(
            node_name.clone(),
            Node {
                uid: node_name,
                op: NodeOp::Slice,
                width: signal.width as u32,
                pin_map,
                params: None,
                attrs: None,
            },
        );
    }

    let module = Module { ports, nets, nodes };
    let mut modules = BTreeMap::new();
    modules.insert("top".to_string(), module);

    Nir {
        v: "nir-1.1".to_string(),
        design: "fuzz".to_string(),
        top: "top".to_string(),
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
    seed: u64,
) -> HashMap<String, Vec<BigUint>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut map = HashMap::new();
    for (name, port) in &module.ports {
        if matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
            let mut values = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                values.push(random_biguint(port.bits as usize, &mut rng));
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

#[derive(Clone)]
struct NodeValue {
    bits: Vec<Vec<bool>>,
}

impl NodeValue {
    fn width(&self) -> usize {
        self.bits.len()
    }

    fn single(bit_vec: Vec<bool>) -> Self {
        Self {
            bits: vec![bit_vec],
        }
    }

    fn constant(pattern: Vec<bool>, num_vectors: usize) -> Self {
        let bits = pattern
            .into_iter()
            .map(|bit| vec![bit; num_vectors])
            .collect();
        Self { bits }
    }
}

fn evaluate_normalized(
    module: &NormalizedModule,
    inputs: &HashMap<String, Vec<BigUint>>,
) -> Result<HashMap<String, Vec<BigUint>>, String> {
    let mut node_values: Vec<Option<NodeValue>> = vec![None; module.nodes.len()];

    for (port, literals) in &module.inputs {
        let values = inputs
            .get(port)
            .ok_or_else(|| format!("missing input values for port `{port}`"))?;
        let bit_vectors = biguints_to_bits(values, literals.len());
        for (bit_index, literal) in literals.iter().enumerate() {
            let mut bits = bit_vectors[bit_index].clone();
            if literal.inverted {
                for bit in &mut bits {
                    *bit = !*bit;
                }
            }
            node_values[literal.node] = Some(NodeValue::single(bits));
        }
    }

    for node in &module.nodes {
        if node_values[node.id].is_some() {
            continue;
        }
        match &node.kind {
            NormalizedNodeKind::Input => {
                return Err(format!("input node {} was not assigned", node.id));
            }
            NormalizedNodeKind::Const { bits } => {
                let pattern = parse_const_bits(bits, node.width as usize)?;
                node_values[node.id] = Some(NodeValue::constant(pattern, NUM_VECTORS));
            }
            NormalizedNodeKind::Op { op, inputs, .. } => {
                if node.width != 1 {
                    return Err(format!(
                        "unsupported node width {} for op {:?}",
                        node.width, op
                    ));
                }
                let mut result = match op {
                    NodeOp::And => vec![true; NUM_VECTORS],
                    NodeOp::Or => vec![false; NUM_VECTORS],
                    NodeOp::Xor => vec![false; NUM_VECTORS],
                    other => {
                        return Err(format!("unsupported op in normalized graph: {other:?}"));
                    }
                };
                for literal in inputs {
                    let source = node_values
                        .get(literal.node)
                        .and_then(|value| value.as_ref())
                        .ok_or_else(|| format!("missing value for node {}", literal.node))?;
                    if source.width() != 1 {
                        return Err(format!(
                            "literal width mismatch: node {} has width {}",
                            literal.node,
                            source.width()
                        ));
                    }
                    let bits = &source.bits[0];
                    for (index, slot) in result.iter_mut().enumerate() {
                        let mut bit = bits[index];
                        if literal.inverted {
                            bit = !bit;
                        }
                        match op {
                            NodeOp::And => *slot &= bit,
                            NodeOp::Or => *slot |= bit,
                            NodeOp::Xor => *slot ^= bit,
                            _ => unreachable!(),
                        }
                    }
                }
                node_values[node.id] = Some(NodeValue::single(result));
            }
        }
    }

    let mut outputs = HashMap::new();
    for (name, bits) in &module.outputs {
        let mut values = vec![BigUint::default(); NUM_VECTORS];
        for (bit_index, literal) in bits.iter().enumerate() {
            let source = node_values
                .get(literal.node)
                .and_then(|value| value.as_ref())
                .ok_or_else(|| format!("missing value for node {}", literal.node))?;
            if source.width() != 1 {
                return Err(format!(
                    "output literal width mismatch: node {} has width {}",
                    literal.node,
                    source.width()
                ));
            }
            let bits = &source.bits[0];
            for (vec_index, value) in values.iter_mut().enumerate() {
                let mut bit = bits[vec_index];
                if literal.inverted {
                    bit = !bit;
                }
                if bit {
                    value.set_bit(bit_index as u64, true);
                }
            }
        }
        outputs.insert(name.clone(), values);
    }

    Ok(outputs)
}

fn biguints_to_bits(values: &[BigUint], width: usize) -> Vec<Vec<bool>> {
    let mut result = vec![vec![false; values.len()]; width];
    for (vec_idx, value) in values.iter().enumerate() {
        for bit_idx in 0..width {
            result[bit_idx][vec_idx] = value.bit(bit_idx as u64);
        }
    }
    result
}

fn parse_const_bits(bits: &str, width: usize) -> Result<Vec<bool>, String> {
    if bits.len() != width && !(bits.is_empty() && width == 0) {
        return Err(format!(
            "constant bit string length mismatch (expected {width}, got {})",
            bits.len()
        ));
    }
    Ok(bits
        .chars()
        .rev()
        .map(|ch| match ch {
            '0' => Ok(false),
            '1' => Ok(true),
            other => Err(format!("invalid bit character `{other}`")),
        })
        .collect::<Result<Vec<_>, _>>()?)
}
