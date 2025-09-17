use std::collections::{BTreeMap, HashMap};

use num_bigint::BigUint;
use num_traits::Num;
use v2m_formats::nir::{BitRef, Module, NodeOp};
use v2m_nir::{ModuleGraph, NetId, NodeId};

use crate::packed::{mask_for_word, PackedIndex};
use crate::pin_binding::{BitBinding, ConstId, ConstPool, SignalId, LANE_BITS};

#[derive(Default)]
pub(crate) struct NodePinBindings {
    pub(crate) pins: BTreeMap<String, BitBinding>,
}

#[derive(Clone, Debug)]
pub(crate) struct UnaryKernel {
    pub(crate) input: PackedIndex,
    pub(crate) output: PackedIndex,
}

#[derive(Clone, Debug)]
pub(crate) struct BinaryKernel {
    pub(crate) input_a: PackedIndex,
    pub(crate) input_b: PackedIndex,
    pub(crate) output: PackedIndex,
}

#[derive(Clone, Debug)]
pub(crate) struct MuxKernel {
    pub(crate) input_a: PackedIndex,
    pub(crate) input_b: PackedIndex,
    pub(crate) select: PackedIndex,
    pub(crate) output: PackedIndex,
    pub(crate) select_masks: Vec<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct TransferKernel {
    pub(crate) plan: Vec<TransferSegment>,
}

#[derive(Clone, Debug)]
pub(crate) struct TransferSegment {
    pub(crate) dest_offset: usize,
    pub(crate) lanes: usize,
    pub(crate) source: SegmentSource,
}

#[derive(Clone, Debug)]
pub(crate) enum SegmentSource {
    Net { offset: usize },
    Const { words: Vec<u64> },
}

#[derive(Clone, Debug)]
pub(crate) struct ArithKernel {
    pub(crate) bits: Vec<ArithBitPlan>,
    pub(crate) invert_b: bool,
    pub(crate) initial_carry: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ArithBitPlan {
    pub(crate) dest_offset: usize,
    pub(crate) a: BitSource,
    pub(crate) b: BitSource,
}

#[derive(Clone, Debug)]
pub(crate) enum NodeKernel {
    Const(TransferKernel),
    Slice(TransferKernel),
    Cat(TransferKernel),
    Not(UnaryKernel),
    And(BinaryKernel),
    Or(BinaryKernel),
    Xor(BinaryKernel),
    Xnor(BinaryKernel),
    Mux(MuxKernel),
    Add(ArithKernel),
    Sub(ArithKernel),
}

#[derive(Clone, Copy, Debug)]
enum PackedSourceKind {
    Net(NetId, usize),
    Const(ConstId, usize),
    Literal(bool),
}

#[derive(Clone, Copy, Debug)]
struct PackedDestination {
    net: NetId,
    lane: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BitSource {
    Net { offset: usize },
    Const { value: bool },
}

pub(crate) fn build_comb_kernels(
    module: &Module,
    graph: &ModuleGraph,
    net_indices: &HashMap<String, PackedIndex>,
    net_indices_by_id: &HashMap<NetId, PackedIndex>,
    node_bindings: &HashMap<NodeId, NodePinBindings>,
    const_pool: &ConstPool,
    words_per_lane: usize,
    num_vectors: usize,
) -> HashMap<NodeId, NodeKernel> {
    let mut kernels = HashMap::new();

    for (name, node) in &module.nodes {
        if matches!(node.op, NodeOp::Dff | NodeOp::Latch) {
            continue;
        }

        let node_id = graph
            .node_id(name.as_str())
            .expect("module node must exist in graph");

        let kernel = match node.op {
            NodeOp::Const => node_bindings
                .get(&node_id)
                .and_then(|bindings| {
                    build_const_kernel(
                        node,
                        bindings,
                        net_indices_by_id,
                        const_pool,
                        words_per_lane,
                        num_vectors,
                    )
                })
                .map(NodeKernel::Const),
            NodeOp::Slice => node_bindings
                .get(&node_id)
                .and_then(|bindings| {
                    build_copy_kernel(
                        bindings,
                        net_indices_by_id,
                        const_pool,
                        words_per_lane,
                        num_vectors,
                    )
                })
                .map(NodeKernel::Slice),
            NodeOp::Cat => node_bindings
                .get(&node_id)
                .and_then(|bindings| {
                    build_copy_kernel(
                        bindings,
                        net_indices_by_id,
                        const_pool,
                        words_per_lane,
                        num_vectors,
                    )
                })
                .map(NodeKernel::Cat),
            NodeOp::Add => node_bindings
                .get(&node_id)
                .and_then(|bindings| {
                    build_arith_kernel(
                        bindings,
                        net_indices_by_id,
                        const_pool,
                        words_per_lane,
                        false,
                        false,
                    )
                })
                .map(NodeKernel::Add),
            NodeOp::Sub => node_bindings
                .get(&node_id)
                .and_then(|bindings| {
                    build_arith_kernel(
                        bindings,
                        net_indices_by_id,
                        const_pool,
                        words_per_lane,
                        true,
                        true,
                    )
                })
                .map(NodeKernel::Sub),
            NodeOp::Not => build_not_kernel(node, net_indices).map(NodeKernel::Not),
            NodeOp::And => build_and_kernel(node, net_indices).map(NodeKernel::And),
            NodeOp::Or => build_or_kernel(node, net_indices).map(NodeKernel::Or),
            NodeOp::Xor => build_xor_kernel(node, net_indices).map(NodeKernel::Xor),
            NodeOp::Xnor => build_xnor_kernel(node, net_indices).map(NodeKernel::Xnor),
            NodeOp::Mux => build_mux_kernel(node, net_indices, words_per_lane, num_vectors)
                .map(NodeKernel::Mux),
            _ => None,
        };

        if let Some(kernel) = kernel {
            kernels.insert(node_id, kernel);
        }
    }

    kernels
}

fn build_const_kernel(
    node: &v2m_formats::nir::Node,
    bindings: &NodePinBindings,
    net_indices_by_id: &HashMap<NetId, PackedIndex>,
    const_pool: &ConstPool,
    words_per_lane: usize,
    num_vectors: usize,
) -> Option<TransferKernel> {
    let literal = node
        .params
        .as_ref()
        .and_then(|params| params.get("value"))
        .and_then(|value| value.as_str())?;
    let bits = parse_const_bits(literal, node.width)?;

    let output_binding = bindings.pins.get("Y")?;
    let destinations = expand_binding_destinations(output_binding)?;
    if destinations.len() != bits.len() {
        return None;
    }

    let sources: Vec<PackedSourceKind> = bits.into_iter().map(PackedSourceKind::Literal).collect();

    let plan = build_transfer_plan(
        &sources,
        &destinations,
        net_indices_by_id,
        const_pool,
        words_per_lane,
        num_vectors,
    )?;

    Some(TransferKernel { plan })
}

fn build_copy_kernel(
    bindings: &NodePinBindings,
    net_indices_by_id: &HashMap<NetId, PackedIndex>,
    const_pool: &ConstPool,
    words_per_lane: usize,
    num_vectors: usize,
) -> Option<TransferKernel> {
    let input_binding = bindings.pins.get("A")?;
    let output_binding = bindings.pins.get("Y")?;

    let sources = expand_binding_sources(input_binding);
    let destinations = expand_binding_destinations(output_binding)?;

    if sources.len() != destinations.len() {
        return None;
    }

    let plan = build_transfer_plan(
        &sources,
        &destinations,
        net_indices_by_id,
        const_pool,
        words_per_lane,
        num_vectors,
    )?;

    Some(TransferKernel { plan })
}

fn build_arith_kernel(
    bindings: &NodePinBindings,
    net_indices_by_id: &HashMap<NetId, PackedIndex>,
    const_pool: &ConstPool,
    words_per_lane: usize,
    invert_b: bool,
    initial_carry: bool,
) -> Option<ArithKernel> {
    let input_a = bindings.pins.get("A")?;
    let input_b = bindings.pins.get("B")?;
    let output = bindings.pins.get("Y")?;

    let sources_a = expand_binding_sources(input_a);
    let sources_b = expand_binding_sources(input_b);
    let destinations = expand_binding_destinations(output)?;

    if destinations.len() != sources_a.len() || destinations.len() != sources_b.len() {
        return None;
    }

    if words_per_lane == 0 {
        return Some(ArithKernel {
            bits: Vec::new(),
            invert_b,
            initial_carry,
        });
    }

    let mut bits = Vec::with_capacity(destinations.len());
    for idx in 0..destinations.len() {
        let dest = &destinations[idx];
        let dest_index = *net_indices_by_id.get(&dest.net)?;
        let dest_offset = dest_index.offset() + dest.lane * words_per_lane;
        let a_source = convert_bit_source(
            &sources_a[idx],
            net_indices_by_id,
            const_pool,
            words_per_lane,
        )?;
        let b_source = convert_bit_source(
            &sources_b[idx],
            net_indices_by_id,
            const_pool,
            words_per_lane,
        )?;
        bits.push(ArithBitPlan {
            dest_offset,
            a: a_source,
            b: b_source,
        });
    }

    Some(ArithKernel {
        bits,
        invert_b,
        initial_carry,
    })
}

fn expand_binding_sources(binding: &BitBinding) -> Vec<PackedSourceKind> {
    let mut sources = Vec::new();
    let bits_per_lane = LANE_BITS as usize;
    for descriptor in binding.descriptors() {
        let base_lane =
            descriptor.lane_offset as usize * bits_per_lane + descriptor.bit_offset as usize;
        for offset in 0..descriptor.width as usize {
            let lane = base_lane + offset;
            match descriptor.source {
                SignalId::Net(net_id) => {
                    sources.push(PackedSourceKind::Net(net_id, lane));
                }
                SignalId::Const(const_id) => {
                    sources.push(PackedSourceKind::Const(const_id, lane));
                }
            }
        }
    }
    sources
}

fn expand_binding_destinations(binding: &BitBinding) -> Option<Vec<PackedDestination>> {
    let mut destinations = Vec::new();
    let bits_per_lane = LANE_BITS as usize;
    for descriptor in binding.descriptors() {
        let base_lane =
            descriptor.lane_offset as usize * bits_per_lane + descriptor.bit_offset as usize;
        match descriptor.source {
            SignalId::Net(net_id) => {
                for offset in 0..descriptor.width as usize {
                    destinations.push(PackedDestination {
                        net: net_id,
                        lane: base_lane + offset,
                    });
                }
            }
            SignalId::Const(_) => return None,
        }
    }
    Some(destinations)
}

fn const_pool_bit(const_pool: &ConstPool, const_id: ConstId, lane: usize) -> bool {
    let value = const_pool.get(const_id);
    if lane as u32 >= value.width {
        return false;
    }
    let word_index = lane / LANE_BITS as usize;
    let bit_index = lane % LANE_BITS as usize;
    let chunk = value.words.get(word_index).copied().unwrap_or(0);
    ((chunk >> bit_index) & 1) != 0
}

fn convert_bit_source(
    source: &PackedSourceKind,
    net_indices_by_id: &HashMap<NetId, PackedIndex>,
    const_pool: &ConstPool,
    words_per_lane: usize,
) -> Option<BitSource> {
    match source {
        PackedSourceKind::Net(net_id, lane) => {
            let index = *net_indices_by_id.get(net_id)?;
            Some(BitSource::Net {
                offset: index.offset() + lane * words_per_lane,
            })
        }
        PackedSourceKind::Const(const_id, lane) => Some(BitSource::Const {
            value: const_pool_bit(const_pool, *const_id, *lane),
        }),
        PackedSourceKind::Literal(bit) => Some(BitSource::Const { value: *bit }),
    }
}

fn build_transfer_plan(
    sources: &[PackedSourceKind],
    destinations: &[PackedDestination],
    net_indices_by_id: &HashMap<NetId, PackedIndex>,
    const_pool: &ConstPool,
    words_per_lane: usize,
    num_vectors: usize,
) -> Option<Vec<TransferSegment>> {
    if sources.len() != destinations.len() {
        return None;
    }

    if words_per_lane == 0 {
        return Some(Vec::new());
    }

    #[derive(Debug)]
    enum PendingSource {
        Net { offset: usize },
        Const { bits: Vec<bool> },
    }

    struct PendingSegment {
        dest_offset: usize,
        source: PendingSource,
        lanes: usize,
    }

    fn finalize_pending(
        pending: Option<PendingSegment>,
        plan: &mut Vec<TransferSegment>,
        words_per_lane: usize,
        num_vectors: usize,
    ) {
        if let Some(segment) = pending {
            match segment.source {
                PendingSource::Net { offset } => {
                    plan.push(TransferSegment {
                        dest_offset: segment.dest_offset,
                        lanes: segment.lanes,
                        source: SegmentSource::Net { offset },
                    });
                }
                PendingSource::Const { bits } => {
                    let mut words = Vec::with_capacity(bits.len() * words_per_lane);
                    for bit in bits {
                        for word_idx in 0..words_per_lane {
                            let mask = mask_for_word(word_idx, words_per_lane, num_vectors);
                            words.push(if bit { mask } else { 0 });
                        }
                    }
                    plan.push(TransferSegment {
                        dest_offset: segment.dest_offset,
                        lanes: segment.lanes,
                        source: SegmentSource::Const { words },
                    });
                }
            }
        }
    }

    let mut plan = Vec::new();
    let mut pending: Option<PendingSegment> = None;

    for (source_kind, dest) in sources.iter().zip(destinations.iter()) {
        let dest_index = *net_indices_by_id.get(&dest.net)?;
        let dest_offset = dest_index.offset() + dest.lane * words_per_lane;
        let source =
            convert_bit_source(source_kind, net_indices_by_id, const_pool, words_per_lane)?;

        match source {
            BitSource::Net { offset } => {
                if let Some(existing) = pending.as_mut() {
                    if let PendingSource::Net {
                        offset: current_offset,
                    } = &mut existing.source
                    {
                        if existing.dest_offset + existing.lanes * words_per_lane == dest_offset
                            && *current_offset + existing.lanes * words_per_lane == offset
                        {
                            existing.lanes += 1;
                            continue;
                        }
                    }
                }

                let to_flush = pending.take();
                finalize_pending(to_flush, &mut plan, words_per_lane, num_vectors);
                pending = Some(PendingSegment {
                    dest_offset,
                    source: PendingSource::Net { offset },
                    lanes: 1,
                });
            }
            BitSource::Const { value } => {
                if let Some(existing) = pending.as_mut() {
                    if let PendingSource::Const { bits } = &mut existing.source {
                        if existing.dest_offset + existing.lanes * words_per_lane == dest_offset {
                            bits.push(value);
                            existing.lanes += 1;
                            continue;
                        }
                    }
                }

                let to_flush = pending.take();
                finalize_pending(to_flush, &mut plan, words_per_lane, num_vectors);
                pending = Some(PendingSegment {
                    dest_offset,
                    source: PendingSource::Const { bits: vec![value] },
                    lanes: 1,
                });
            }
        }
    }

    finalize_pending(pending, &mut plan, words_per_lane, num_vectors);
    Some(plan)
}

fn parse_const_bits(literal: &str, width: u32) -> Option<Vec<bool>> {
    let (base, digits) = if let Some(rest) = literal.strip_prefix("0b") {
        (2u32, rest)
    } else if let Some(rest) = literal.strip_prefix("0B") {
        (2u32, rest)
    } else if let Some(rest) = literal.strip_prefix("0x") {
        (16u32, rest)
    } else if let Some(rest) = literal.strip_prefix("0X") {
        (16u32, rest)
    } else {
        (10u32, literal)
    };

    let digits: String = digits.chars().filter(|c| *c != '_').collect();
    if digits.is_empty() {
        return None;
    }

    let value = match base {
        2 => BigUint::from_str_radix(&digits, 2).ok()?,
        16 => BigUint::from_str_radix(&digits, 16).ok()?,
        10 => BigUint::parse_bytes(digits.as_bytes(), 10)?,
        _ => unreachable!(),
    };

    let width = width as usize;
    if width == 0 {
        return Some(Vec::new());
    }

    if value.bits() > width as u64 {
        return None;
    }

    let mut bits = Vec::with_capacity(width);
    for index in 0..width {
        bits.push(value.bit(index as u64));
    }

    Some(bits)
}

fn build_not_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
) -> Option<UnaryKernel> {
    let input_ref = node.pin_map.get("A")?;
    let output_ref = node.pin_map.get("Y")?;
    let input_net = bitref_full_net(input_ref, node.width)?;
    let output_net = bitref_full_net(output_ref, node.width)?;
    let input_index = *net_indices
        .get(input_net)
        .expect("NOT input net must be allocated");
    let output_index = *net_indices
        .get(output_net)
        .expect("NOT output net must be allocated");

    Some(UnaryKernel {
        input: input_index,
        output: output_index,
    })
}

fn build_and_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
) -> Option<BinaryKernel> {
    build_bitwise_kernel(node, net_indices, "AND")
}

fn build_or_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
) -> Option<BinaryKernel> {
    build_bitwise_kernel(node, net_indices, "OR")
}

fn build_xor_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
) -> Option<BinaryKernel> {
    build_bitwise_kernel(node, net_indices, "XOR")
}

fn build_xnor_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
) -> Option<BinaryKernel> {
    build_bitwise_kernel(node, net_indices, "XNOR")
}

fn build_bitwise_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
    op_name: &'static str,
) -> Option<BinaryKernel> {
    let input_a_ref = node.pin_map.get("A")?;
    let input_b_ref = node.pin_map.get("B")?;
    let output_ref = node.pin_map.get("Y")?;
    let input_a_net = bitref_full_net(input_a_ref, node.width)?;
    let input_b_net = bitref_full_net(input_b_ref, node.width)?;
    let output_net = bitref_full_net(output_ref, node.width)?;
    let input_a = *net_indices
        .get(input_a_net)
        .unwrap_or_else(|| panic!("{} input A net must be allocated", op_name));
    let input_b = *net_indices
        .get(input_b_net)
        .unwrap_or_else(|| panic!("{} input B net must be allocated", op_name));
    let output = *net_indices
        .get(output_net)
        .unwrap_or_else(|| panic!("{} output net must be allocated", op_name));

    Some(BinaryKernel {
        input_a,
        input_b,
        output,
    })
}

fn build_mux_kernel(
    node: &v2m_formats::nir::Node,
    net_indices: &HashMap<String, PackedIndex>,
    words_per_lane: usize,
    num_vectors: usize,
) -> Option<MuxKernel> {
    let input_a_ref = node.pin_map.get("A")?;
    let input_b_ref = node.pin_map.get("B")?;
    let select_ref = node.pin_map.get("S")?;
    let output_ref = node.pin_map.get("Y")?;

    let input_a_net = bitref_full_net(input_a_ref, node.width)?;
    let input_b_net = bitref_full_net(input_b_ref, node.width)?;
    let select_net = bitref_full_net(select_ref, 1)?;
    let output_net = bitref_full_net(output_ref, node.width)?;

    let input_a = *net_indices
        .get(input_a_net)
        .expect("MUX A input net must be allocated");
    let input_b = *net_indices
        .get(input_b_net)
        .expect("MUX B input net must be allocated");
    let select = *net_indices
        .get(select_net)
        .expect("MUX select net must be allocated");
    let output = *net_indices
        .get(output_net)
        .expect("MUX output net must be allocated");

    if select.lanes() != 1 {
        return None;
    }

    let select_masks = if words_per_lane == 0 {
        Vec::new()
    } else {
        (0..words_per_lane)
            .map(|word| mask_for_word(word, words_per_lane, num_vectors))
            .collect()
    };

    Some(MuxKernel {
        input_a,
        input_b,
        select,
        output,
        select_masks,
    })
}

fn bitref_full_net<'a>(bitref: &'a BitRef, expected_width: u32) -> Option<&'a str> {
    match bitref {
        BitRef::Net(net) if net.lsb == 0 && net.msb >= net.lsb => {
            let width = net.msb - net.lsb + 1;
            if width == expected_width {
                Some(net.net.as_str())
            } else {
                None
            }
        }
        _ => None,
    }
}
