use std::collections::{BTreeMap, HashMap};
use std::ptr;

use num_bigint::BigUint;
use num_traits::{Num, Zero};

use thiserror::Error;
use v2m_formats::nir::{BitRef, Module, Nir, NodeOp, PortDirection};
use v2m_nir::{BuildError as GraphBuildError, EvalOrderError, ModuleGraph, NetId, NodeId};

mod pin_binding;

use pin_binding::{
    bind_bitref, BitBinding, ConstId, ConstPool, PinBindingError, SignalId, LANE_BITS,
};

const WORD_BITS: usize = 64;

#[inline]
fn div_ceil(value: usize, divisor: usize) -> usize {
    debug_assert!(divisor > 0);
    if value == 0 {
        0
    } else {
        1 + (value - 1) / divisor
    }
}

#[inline]
fn lanes_for_width(width_bits: usize) -> usize {
    width_bits
}

#[inline]
fn words_for_vectors(num_vectors: usize) -> usize {
    div_ceil(num_vectors, WORD_BITS)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimOptions {
    pub allow_x: bool,
    pub async_reset_is_high: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    num_vectors: usize,
    words_per_lane: usize,
    storage: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedIndex {
    offset: usize,
    lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBitMask {
    words: Vec<u64>,
}

#[derive(Debug, Error)]
pub enum PackedError {
    #[error(
        "packed buffers must have the same shape (lanes {expected_lanes}x{expected_words}, got {actual_lanes}x{actual_words})"
    )]
    ShapeMismatch {
        expected_lanes: usize,
        expected_words: usize,
        actual_lanes: usize,
        actual_words: usize,
    },
}

#[derive(Debug, Error)]
pub enum PortValueError {
    #[error("missing data for port `{name}`")]
    MissingPort { name: String },
    #[error("unexpected port `{name}`")]
    UnexpectedPort { name: String },
    #[error("port `{name}` expects {expected} vectors, got {actual}")]
    VectorCountMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("value for port `{name}` vector {vector} exceeds width {width_bits} bits")]
    ValueTooWide {
        name: String,
        width_bits: usize,
        vector: usize,
    },
    #[error("packed buffer expects {expected} vectors, got {actual}")]
    PackedVectorMismatch { expected: usize, actual: usize },
    #[error("packed buffer words-per-lane mismatch (expected {expected}, got {actual})")]
    WordsPerLaneMismatch { expected: usize, actual: usize },
    #[error("packed buffer layout mismatch for port `{name}`")]
    PackedLayoutMismatch { name: String },
    #[error("port `{name}` has unsupported direction `{actual:?}` for this operation")]
    DirectionMismatch { name: String, actual: PortDirection },
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("top module `{top}` not found in design `{design}`")]
    MissingTop { design: String, top: String },
    #[error(transparent)]
    ModuleGraph(#[from] GraphBuildError),
    #[error(transparent)]
    EvalOrder(#[from] EvalOrderError),
    #[error(transparent)]
    Packed(#[from] PackedError),
    #[error(transparent)]
    PortValues(#[from] PortValueError),
    #[error("failed to bind pin `{pin}` on node `{node}`: {source}")]
    PinBinding {
        node: String,
        pin: String,
        #[source]
        source: PinBindingError,
    },
}

#[allow(dead_code)]
pub struct Evaluator<'nir> {
    nir: &'nir Nir,
    module: &'nir Module,
    graph: ModuleGraph,
    topo: Vec<NodeId>,
    topo_level_offsets: Vec<usize>,
    topo_level_map: Vec<Option<usize>>,
    options: SimOptions,
    num_vectors: usize,
    nets: Packed,
    net_indices: HashMap<String, PackedIndex>,
    net_indices_by_id: HashMap<NetId, PackedIndex>,
    regs_cur: Packed,
    regs_next: Packed,
    reg_indices: HashMap<String, PackedIndex>,
    inputs: Packed,
    input_ports: HashMap<String, PackedIndex>,
    outputs: Packed,
    output_ports: HashMap<String, PackedIndex>,
    node_bindings: HashMap<NodeId, NodePinBindings>,
    dff_nodes: Vec<DffInfo>,
    const_pool: ConstPool,
    comb_kernels: HashMap<NodeId, NodeKernel>,
    carry_scratch: Vec<u64>,
}

impl Packed {
    pub fn new(num_vectors: usize) -> Self {
        Self {
            num_vectors,
            words_per_lane: words_for_vectors(num_vectors),
            storage: Vec::new(),
        }
    }

    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    #[inline]
    pub fn words_per_lane(&self) -> usize {
        self.words_per_lane
    }

    #[inline]
    pub fn total_lanes(&self) -> usize {
        if self.words_per_lane == 0 {
            0
        } else {
            self.storage.len() / self.words_per_lane
        }
    }

    pub fn allocate(&mut self, width_bits: usize) -> PackedIndex {
        let lanes = lanes_for_width(width_bits);
        let offset = self.storage.len();
        self.storage.resize(offset + lanes * self.words_per_lane, 0);
        PackedIndex { offset, lanes }
    }

    pub fn duplicate_layout(&self) -> Self {
        Self {
            num_vectors: self.num_vectors,
            words_per_lane: self.words_per_lane,
            storage: vec![0; self.storage.len()],
        }
    }

    #[inline]
    fn lane_offset(&self, index: PackedIndex, lane: usize) -> usize {
        debug_assert!(lane < index.lanes);
        index.offset + lane * self.words_per_lane
    }

    pub fn lane(&self, index: PackedIndex, lane: usize) -> &[u64] {
        let start = self.lane_offset(index, lane);
        let end = start + self.words_per_lane;
        &self.storage[start..end]
    }

    pub fn lane_mut(&mut self, index: PackedIndex, lane: usize) -> &mut [u64] {
        let start = self.lane_offset(index, lane);
        let end = start + self.words_per_lane;
        &mut self.storage[start..end]
    }

    #[inline]
    pub fn slice(&self, index: PackedIndex) -> &[u64] {
        let end = index.offset + index.lanes * self.words_per_lane;
        &self.storage[index.offset..end]
    }

    #[inline]
    pub fn slice_mut(&mut self, index: PackedIndex) -> &mut [u64] {
        let end = index.offset + index.lanes * self.words_per_lane;
        &mut self.storage[index.offset..end]
    }

    pub fn copy_from(&mut self, other: &Packed) -> Result<(), PackedError> {
        if self.words_per_lane != other.words_per_lane || self.storage.len() != other.storage.len()
        {
            return Err(PackedError::ShapeMismatch {
                expected_lanes: self.total_lanes(),
                expected_words: self.words_per_lane,
                actual_lanes: other.total_lanes(),
                actual_words: other.words_per_lane,
            });
        }

        self.storage.copy_from_slice(&other.storage);
        Ok(())
    }
}

impl PackedIndex {
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    #[inline]
    pub fn lane_offset(&self, lane: usize, words_per_lane: usize) -> usize {
        debug_assert!(lane < self.lanes);
        self.offset + lane * words_per_lane
    }

    #[inline]
    pub fn word_offset(&self, lane: usize, word: usize, words_per_lane: usize) -> usize {
        debug_assert!(words_per_lane > 0);
        debug_assert!(word < words_per_lane);
        self.lane_offset(lane, words_per_lane) + word
    }
}

impl PackedBitMask {
    pub fn new(num_vectors: usize) -> Self {
        Self {
            words: vec![0; words_for_vectors(num_vectors)],
        }
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

fn pack_port_biguints(
    target: &mut Packed,
    index: PackedIndex,
    width_bits: usize,
    values: &[BigUint],
    name: &str,
) -> Result<(), PortValueError> {
    if width_bits > index.lanes() {
        return Err(PortValueError::PackedLayoutMismatch {
            name: name.to_string(),
        });
    }

    if values.len() != target.num_vectors() {
        return Err(PortValueError::VectorCountMismatch {
            name: name.to_string(),
            expected: target.num_vectors(),
            actual: values.len(),
        });
    }

    let words_per_lane = target.words_per_lane();
    let slice = target.slice_mut(index);
    slice.fill(0);

    if width_bits == 0 {
        for (vec_idx, value) in values.iter().enumerate() {
            if !value.is_zero() {
                return Err(PortValueError::ValueTooWide {
                    name: name.to_string(),
                    width_bits,
                    vector: vec_idx,
                });
            }
        }
        return Ok(());
    }

    for (vec_idx, value) in values.iter().enumerate() {
        if value.bits() > width_bits as u64 {
            return Err(PortValueError::ValueTooWide {
                name: name.to_string(),
                width_bits,
                vector: vec_idx,
            });
        }

        if value.is_zero() {
            continue;
        }

        let word_idx = vec_idx / WORD_BITS;
        let bit_in_word = vec_idx % WORD_BITS;
        let bit_mask = 1u64 << bit_in_word;

        for (chunk_idx, mut chunk) in value.to_u64_digits().into_iter().enumerate() {
            if chunk == 0 {
                continue;
            }

            let base_lane = chunk_idx * WORD_BITS;
            while chunk != 0 {
                let bit = chunk.trailing_zeros() as usize;
                let lane = base_lane + bit;
                if lane >= width_bits {
                    break;
                }

                let offset = index.offset + lane * words_per_lane + word_idx;
                target.storage[offset] |= bit_mask;
                chunk &= chunk - 1;
            }
        }
    }

    Ok(())
}

fn unpack_port_biguints(
    source: &Packed,
    index: PackedIndex,
    width_bits: usize,
    name: &str,
) -> Result<Vec<BigUint>, PortValueError> {
    if width_bits > index.lanes() {
        return Err(PortValueError::PackedLayoutMismatch {
            name: name.to_string(),
        });
    }

    let words_per_lane = source.words_per_lane();
    let end = index.offset + index.lanes() * words_per_lane;
    if end > source.storage.len() {
        return Err(PortValueError::PackedLayoutMismatch {
            name: name.to_string(),
        });
    }

    let num_vectors = source.num_vectors();
    let mut result = vec![BigUint::default(); num_vectors];

    if width_bits == 0 {
        return Ok(result);
    }

    let slice = source.slice(index);
    for lane in 0..width_bits {
        let lane_offset = lane * words_per_lane;
        for word_idx in 0..words_per_lane {
            let word = slice[lane_offset + word_idx];
            if word == 0 {
                continue;
            }

            let base_vector = word_idx * WORD_BITS;
            let mut mask = word;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let vector_idx = base_vector + bit;
                if vector_idx >= num_vectors {
                    break;
                }

                result[vector_idx].set_bit(lane as u64, true);
                mask &= mask - 1;
            }
        }
    }

    Ok(result)
}

#[derive(Default)]
struct NodePinBindings {
    pins: BTreeMap<String, BitBinding>,
}

#[derive(Clone, Debug)]
struct UnaryKernel {
    input: PackedIndex,
    output: PackedIndex,
}

#[derive(Clone, Debug)]
struct BinaryKernel {
    input_a: PackedIndex,
    input_b: PackedIndex,
    output: PackedIndex,
}

#[derive(Clone, Debug)]
struct MuxKernel {
    input_a: PackedIndex,
    input_b: PackedIndex,
    select: PackedIndex,
    output: PackedIndex,
    select_masks: Vec<u64>,
}

#[derive(Clone, Debug)]
struct TransferSegment {
    dest_offset: usize,
    lanes: usize,
    source: SegmentSource,
}

#[derive(Clone, Debug)]
enum SegmentSource {
    Net { offset: usize },
    Const { words: Vec<u64> },
}

#[derive(Clone, Debug)]
struct TransferKernel {
    plan: Vec<TransferSegment>,
}

#[derive(Clone, Copy, Debug)]
enum BitSource {
    Net { offset: usize },
    Const { value: bool },
}

#[derive(Clone, Debug)]
struct ArithBit {
    a: BitSource,
    b: BitSource,
    dest_offset: usize,
}

#[derive(Clone, Debug)]
struct ArithKernel {
    bits: Vec<ArithBit>,
    invert_b: bool,
    initial_carry: bool,
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

#[derive(Clone, Debug)]
enum NodeKernel {
    Const(TransferKernel),
    Not(UnaryKernel),
    And(BinaryKernel),
    Mux(MuxKernel),
    Slice(TransferKernel),
    Cat(TransferKernel),
    Add(ArithKernel),
    Sub(ArithKernel),
}

#[derive(Clone, Debug)]
struct DffInfo {
    reg_index: PackedIndex,
    d_binding: BitBinding,
    q_binding: BitBinding,
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

fn mask_for_word(word_index: usize, words_per_lane: usize, num_vectors: usize) -> u64 {
    if words_per_lane == 0 {
        return 0;
    }

    if word_index + 1 < words_per_lane {
        return u64::MAX;
    }

    let remainder = num_vectors % WORD_BITS;
    if remainder == 0 {
        if num_vectors == 0 {
            0
        } else {
            u64::MAX
        }
    } else {
        (1u64 << remainder) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use std::collections::{BTreeMap, HashMap};
    use v2m_formats::nir::Port;

    fn build_test_nir() -> Nir {
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
                bits: 17,
                attrs: None,
            },
        );
        ports.insert(
            "c".to_string(),
            Port {
                dir: PortDirection::Inout,
                bits: 65,
                attrs: None,
            },
        );
        ports.insert(
            "zero_in".to_string(),
            Port {
                dir: PortDirection::Input,
                bits: 0,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            Port {
                dir: PortDirection::Output,
                bits: 7,
                attrs: None,
            },
        );
        ports.insert(
            "z".to_string(),
            Port {
                dir: PortDirection::Output,
                bits: 128,
                attrs: None,
            },
        );
        ports.insert(
            "zero_out".to_string(),
            Port {
                dir: PortDirection::Output,
                bits: 0,
                attrs: None,
            },
        );

        let module = Module {
            ports,
            nets: BTreeMap::new(),
            nodes: BTreeMap::new(),
        };

        let mut modules = BTreeMap::new();
        modules.insert("top".to_string(), module);

        Nir {
            v: "0.1".to_string(),
            design: "test".to_string(),
            top: "top".to_string(),
            attrs: None,
            modules,
            generator: None,
            cmdline: None,
            source_digest_sha256: None,
        }
    }

    fn random_biguint(width_bits: usize, rng: &mut StdRng) -> BigUint {
        if width_bits == 0 {
            return BigUint::default();
        }

        let byte_len = div_ceil(width_bits, 8);
        let mut bytes = vec![0u8; byte_len];
        rng.fill_bytes(&mut bytes);

        let excess_bits = byte_len * 8 - width_bits;
        if excess_bits > 0 {
            let keep = 8 - excess_bits;
            let mask = if keep == 0 {
                0
            } else {
                ((1u16 << keep) - 1) as u8
            };
            if let Some(last) = bytes.last_mut() {
                *last &= mask;
            }
        }

        BigUint::from_bytes_le(&bytes)
    }

    fn random_biguints(width_bits: usize, count: usize, rng: &mut StdRng) -> Vec<BigUint> {
        (0..count)
            .map(|_| random_biguint(width_bits, rng))
            .collect()
    }

    fn biguints_to_fixed_bytes(values: &[BigUint], width_bits: usize) -> Vec<Vec<u8>> {
        let byte_len = div_ceil(width_bits, 8);
        values
            .iter()
            .map(|value| {
                let mut bytes = value.to_bytes_le();
                if bytes.len() < byte_len {
                    bytes.resize(byte_len, 0);
                } else if bytes.len() > byte_len {
                    bytes.truncate(byte_len);
                }
                bytes
            })
            .collect()
    }
    use serde_json::Value;
    use v2m_formats::nir::{
        BitRef, BitRefConcat, BitRefNet, Module as NirModule, Net as NirNet, Node as NirNode,
        Port as NirPort,
    };

    fn build_and_module(width: u32) -> Nir {
        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: width,
                attrs: None,
            },
        );
        ports.insert(
            "b".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: width,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: width,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: width,
                attrs: None,
            },
        );
        nets.insert(
            "b".to_string(),
            NirNet {
                bits: width,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: width,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "and0".to_string(),
            NirNode {
                uid: "and0".to_string(),
                op: NodeOp::And,
                width,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("a")),
                    ("B".to_string(), net_bit("b")),
                    ("Y".to_string(), net_bit("y")),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        build_nir(module)
    }

    #[test]
    fn packed_lane_and_word_indexing() {
        let mut packed = Packed::new(128);
        assert_eq!(packed.words_per_lane(), 2);

        let first = packed.allocate(96);
        assert_eq!(first.offset(), 0);
        assert_eq!(first.lanes(), 96);

        let second = packed.allocate(32);
        assert_eq!(second.lanes(), 32);
        assert_eq!(
            second.offset(),
            first.offset() + first.lanes() * packed.words_per_lane()
        );

        for lane in 0..first.lanes() {
            let lane_slice = packed.lane_mut(first, lane);
            for word in 0..lane_slice.len() {
                lane_slice[word] = ((lane as u64) << 16) | word as u64;
            }
        }

        {
            let lane_slice = packed.lane_mut(second, 0);
            for word in 0..lane_slice.len() {
                lane_slice[word] = 0xABCD_0000 | word as u64;
            }
        }

        assert_eq!(packed.total_lanes(), first.lanes() + second.lanes());
        assert_eq!(packed.lane(first, 0), &[0, 1]);
        assert_eq!(packed.lane(first, 1), &[0x0001_0000, 0x0001_0001]);
        assert_eq!(packed.lane(first, 95), &[0x005F_0000, 0x005F_0001]);

        let slice = packed.slice(first);
        assert_eq!(&slice[0..4], &[0, 1, 0x0001_0000, 0x0001_0001]);
        let tail_start = (first.lanes() - 1) * packed.words_per_lane();
        assert_eq!(
            &slice[tail_start..tail_start + packed.words_per_lane()],
            &[0x005F_0000, 0x005F_0001]
        );
        assert_eq!(packed.lane(second, 0), &[0xABCD_0000, 0xABCD_0001]);

        let words_per_lane = packed.words_per_lane();
        assert_eq!(first.lane_offset(1, words_per_lane), words_per_lane);
        assert_eq!(
            first.word_offset(first.lanes() - 1, 1, words_per_lane),
            (first.lanes() - 1) * words_per_lane + 1
        );
    }

    #[test]
    fn packed_handles_width_not_multiple_of_word() {
        let mut packed = Packed::new(70);
        assert_eq!(packed.words_per_lane(), 2);

        let index = packed.allocate(65);
        assert_eq!(index.lanes(), 65);
        assert_eq!(packed.slice(index).len(), 130);

        packed.lane_mut(index, 64)[1] = 0xFFFF_FFFF_FFFFu64;

        let lane = packed.lane(index, 64);
        assert_eq!(lane[1], 0xFFFF_FFFF_FFFFu64);

        let slice = packed.slice(index);
        let start = 64 * packed.words_per_lane();
        assert_eq!(slice[start + 1], 0xFFFF_FFFF_FFFFu64);
        assert!(slice[..start + 1].iter().all(|&word| word == 0));
    }

    #[test]
    fn bitmask_matches_vector_layout() {
        let num_vectors = 130;
        let mut mask = PackedBitMask::new(num_vectors);
        assert_eq!(
            mask.words().len(),
            Packed::new(num_vectors).words_per_lane()
        );

        mask.words_mut()[0] = u64::MAX;
        mask.words_mut()[2] = 0x55AAu64;

        assert_eq!(mask.words()[0], u64::MAX);
        assert_eq!(mask.words()[2], 0x55AAu64);

        let empty_mask = PackedBitMask::new(0);
        assert!(empty_mask.words().is_empty());
    }

    #[test]
    fn inputs_packing_round_trip() {
        let nir = build_test_nir();
        let num_vectors = 130;
        let evaluator = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        let mut biguint_inputs: HashMap<String, Vec<BigUint>> = HashMap::new();
        for (name, port) in &evaluator.module.ports {
            if matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
                let values = random_biguints(port.bits as usize, num_vectors, &mut rng);
                biguint_inputs.insert(name.clone(), values);
            }
        }

        let packed_biguint = evaluator
            .pack_inputs_from_biguints(&biguint_inputs)
            .expect("packing inputs from bigints should succeed");

        for (name, index) in &evaluator.input_ports {
            let width = evaluator.module.ports[name].bits as usize;
            let expected = &biguint_inputs[name];
            let actual = unpack_port_biguints(&packed_biguint, *index, width, name)
                .expect("unpacking should succeed");
            assert_eq!(&actual, expected);
        }

        let mut byte_inputs: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for (name, values) in &biguint_inputs {
            let width = evaluator.module.ports[name].bits as usize;
            byte_inputs.insert(name.clone(), biguints_to_fixed_bytes(values, width));
        }

        let packed_bytes = evaluator
            .pack_inputs_from_bytes(&byte_inputs)
            .expect("packing inputs from bytes should succeed");

        for index in evaluator.input_ports.values() {
            assert_eq!(packed_bytes.slice(*index), packed_biguint.slice(*index));
        }
    }

    #[test]
    fn outputs_unpack_round_trip() {
        let nir = build_test_nir();
        let num_vectors = 96;
        let evaluator = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();
        let mut rng = StdRng::seed_from_u64(0x12345678);

        let mut packed_outputs = evaluator.outputs.duplicate_layout();
        let mut expected = HashMap::new();

        for (name, index) in &evaluator.output_ports {
            let width = evaluator.module.ports[name].bits as usize;
            let values = random_biguints(width, num_vectors, &mut rng);
            pack_port_biguints(&mut packed_outputs, *index, width, &values, name)
                .expect("packing test outputs should succeed");
            expected.insert(name.clone(), values);
        }

        let bigints = evaluator
            .unpack_outputs_to_biguints(&packed_outputs)
            .expect("unpacking bigints should succeed");
        assert_eq!(bigints, expected);

        let bytes = evaluator
            .unpack_outputs_to_bytes(&packed_outputs)
            .expect("unpacking bytes should succeed");

        for (name, values) in &expected {
            let width = evaluator.module.ports[name].bits as usize;
            let expected_bytes = biguints_to_fixed_bytes(values, width);
            assert_eq!(bytes[name], expected_bytes);
        }
    }

    #[test]
    fn packing_reports_errors() {
        let nir = build_test_nir();
        let num_vectors = 64;
        let evaluator = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();
        let mut rng = StdRng::seed_from_u64(0xCAFEBABE);

        let mut partial_inputs = HashMap::new();
        partial_inputs.insert("a".to_string(), random_biguints(1, num_vectors, &mut rng));
        partial_inputs.insert("b".to_string(), random_biguints(17, num_vectors, &mut rng));
        partial_inputs.insert("zero_in".to_string(), vec![BigUint::default(); num_vectors]);

        let err = evaluator
            .pack_inputs_from_biguints(&partial_inputs)
            .unwrap_err();
        assert!(matches!(err, PortValueError::MissingPort { .. }));

        let mut full_inputs = partial_inputs.clone();
        full_inputs.insert("c".to_string(), random_biguints(65, num_vectors, &mut rng));

        let mut inputs_with_extra = full_inputs.clone();
        inputs_with_extra.insert("extra".to_string(), vec![BigUint::default(); num_vectors]);
        let err = evaluator
            .pack_inputs_from_biguints(&inputs_with_extra)
            .unwrap_err();
        assert!(matches!(err, PortValueError::UnexpectedPort { .. }));

        let mut wrong_len = full_inputs.clone();
        wrong_len.insert(
            "c".to_string(),
            random_biguints(65, num_vectors - 1, &mut rng),
        );
        let err = evaluator.pack_inputs_from_biguints(&wrong_len).unwrap_err();
        assert!(matches!(err, PortValueError::VectorCountMismatch { .. }));

        let mut too_wide = full_inputs;
        let mut values = random_biguints(65, num_vectors, &mut rng);
        values[0].set_bit(65, true);
        too_wide.insert("c".to_string(), values);
        let err = evaluator.pack_inputs_from_biguints(&too_wide).unwrap_err();
        assert!(matches!(err, PortValueError::ValueTooWide { .. }));
    }

    #[test]
    fn unpacking_detects_vector_mismatch() {
        let nir = build_test_nir();
        let evaluator = Evaluator::new(&nir, 32, SimOptions::default()).unwrap();
        let wrong_vectors = Packed::new(16);
        let err = evaluator
            .unpack_outputs_to_biguints(&wrong_vectors)
            .unwrap_err();
        assert!(matches!(err, PortValueError::PackedVectorMismatch { .. }));
    }

    fn net_bit(name: &str) -> BitRef {
        BitRef::Net(BitRefNet {
            net: name.to_string(),
            lsb: 0,
            msb: 0,
        })
    }

    fn net_bus(name: &str, width: u32) -> BitRef {
        assert!(width > 0);
        BitRef::Net(BitRefNet {
            net: name.to_string(),
            lsb: 0,
            msb: width - 1,
        })
    }

    fn build_nir(module: NirModule) -> Nir {
        Nir {
            v: "nir-1.1".to_string(),
            design: "test".to_string(),
            top: "Top".to_string(),
            attrs: None,
            modules: BTreeMap::from([("Top".to_string(), module)]),
            generator: None,
            cmdline: None,
            source_digest_sha256: None,
        }
    }

    #[test]
    fn comb_eval_const_node_drives_output() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 1,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );

        let mut params = BTreeMap::new();
        params.insert("value".to_string(), Value::String("1".to_string()));

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "const0".to_string(),
            NirNode {
                uid: "const0".to_string(),
                op: NodeOp::Const,
                width: 1,
                pin_map: BTreeMap::from([("Y".to_string(), net_bit("y"))]),
                params: Some(params),
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);

        let mut eval = Evaluator::new(&nir, 70, SimOptions::default()).expect("create evaluator");
        eval.comb_eval().expect("comb eval");

        let output_index = *eval.output_ports.get("y").expect("output port");
        let outputs = eval.get_outputs();
        let words = outputs.slice(output_index);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], u64::MAX);
        let expected_mask = mask_for_word(1, outputs.words_per_lane(), 70);
        assert_eq!(words[1], expected_mask);
    }

    #[test]
    fn comb_eval_not_gate_inverts_input() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 1,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 1,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "not0".to_string(),
            NirNode {
                uid: "not0".to_string(),
                op: NodeOp::Not,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("a")),
                    ("Y".to_string(), net_bit("y")),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);

        let mut eval = Evaluator::new(&nir, 64, SimOptions::default()).expect("create evaluator");

        let mut inputs = Packed::new(64);
        let input_index = inputs.allocate(1);
        inputs.lane_mut(input_index, 0)[0] = 0xAA55_AA55_AA55_AA55u64;

        eval.set_inputs(&inputs).expect("set inputs");
        eval.comb_eval().expect("comb eval");

        let output_index = *eval.output_ports.get("y").expect("output port");
        let outputs = eval.get_outputs();
        let value = outputs.lane(output_index, 0)[0];
        assert_eq!(value, !0xAA55_AA55_AA55_AA55u64);
    }

    #[test]
    fn comb_eval_and_gate_masks_inputs() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 1,
                attrs: None,
            },
        );
        ports.insert(
            "b".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 1,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 1,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );
        nets.insert(
            "b".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "and0".to_string(),
            NirNode {
                uid: "and0".to_string(),
                op: NodeOp::And,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("a")),
                    ("B".to_string(), net_bit("b")),
                    ("Y".to_string(), net_bit("y")),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);

        let mut eval = Evaluator::new(&nir, 64, SimOptions::default()).expect("create evaluator");

        let mut inputs = Packed::new(64);
        let input_a = inputs.allocate(1);
        let input_b = inputs.allocate(1);
        inputs.lane_mut(input_a, 0)[0] = 0xFFFF_0000_FFFF_0000u64;
        inputs.lane_mut(input_b, 0)[0] = 0x0F0F_0F0F_0F0F_0F0Fu64;

        eval.set_inputs(&inputs).expect("set inputs");
        eval.comb_eval().expect("comb eval");

        let output_index = *eval.output_ports.get("y").expect("output port");
        let outputs = eval.get_outputs();
        let value = outputs.lane(output_index, 0)[0];
        assert_eq!(value, 0xFFFF_0000_FFFF_0000u64 & 0x0F0F_0F0F_0F0F_0F0Fu64);
    }

    #[test]
    fn batch_execution_matches_single_vector() {
        let nir = build_and_module(8);
        let mut rng = StdRng::seed_from_u64(0xFACE_CAFE);
        let batch_sizes = [1usize, 63, 64, 65, 4096];

        for &num_vectors in &batch_sizes {
            let mut batch_eval =
                Evaluator::new(&nir, num_vectors, SimOptions::default()).expect("batch eval");
            let output_names: Vec<String> = batch_eval.output_ports.keys().cloned().collect();

            let mut input_vectors: HashMap<String, Vec<BigUint>> = HashMap::new();
            input_vectors.insert("a".to_string(), random_biguints(8, num_vectors, &mut rng));
            input_vectors.insert("b".to_string(), random_biguints(8, num_vectors, &mut rng));

            let packed_inputs = batch_eval
                .pack_inputs_from_biguints(&input_vectors)
                .expect("pack batch inputs");
            let reset = PackedBitMask::new(num_vectors);
            let packed_outputs = batch_eval.tick(&packed_inputs, &reset).expect("batch tick");
            let batch_outputs = batch_eval
                .unpack_outputs_to_biguints(&packed_outputs)
                .expect("unpack batch outputs");

            let mut expected: HashMap<String, Vec<BigUint>> = output_names
                .iter()
                .map(|name| (name.clone(), Vec::with_capacity(num_vectors)))
                .collect();

            let mut scalar_eval =
                Evaluator::new(&nir, 1, SimOptions::default()).expect("scalar eval");
            let scalar_reset = PackedBitMask::new(1);

            for vector_idx in 0..num_vectors {
                let mut single_inputs = HashMap::new();
                for (name, values) in &input_vectors {
                    single_inputs.insert(name.clone(), vec![values[vector_idx].clone()]);
                }

                let packed_single = scalar_eval
                    .pack_inputs_from_biguints(&single_inputs)
                    .expect("pack scalar inputs");
                let single_outputs_packed = scalar_eval
                    .tick(&packed_single, &scalar_reset)
                    .expect("scalar tick");
                let single_outputs = scalar_eval
                    .unpack_outputs_to_biguints(&single_outputs_packed)
                    .expect("unpack scalar outputs");

                for (name, values) in single_outputs {
                    let value = values.into_iter().next().expect("single vector");
                    expected
                        .get_mut(&name)
                        .expect("expected output")
                        .push(value);
                }
            }

            assert_eq!(batch_outputs, expected, "num_vectors = {}", num_vectors);
        }
    }

    #[test]
    fn comb_eval_mux_selects_inputs() {
        let data_width = 5u32;

        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: data_width,
                attrs: None,
            },
        );
        ports.insert(
            "b".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: data_width,
                attrs: None,
            },
        );
        ports.insert(
            "sel".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 1,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: data_width,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: data_width,
                attrs: None,
            },
        );
        nets.insert(
            "b".to_string(),
            NirNet {
                bits: data_width,
                attrs: None,
            },
        );
        nets.insert(
            "sel".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: data_width,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "mux0".to_string(),
            NirNode {
                uid: "mux0".to_string(),
                op: NodeOp::Mux,
                width: data_width,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bus("a", data_width)),
                    ("B".to_string(), net_bus("b", data_width)),
                    ("S".to_string(), net_bit("sel")),
                    ("Y".to_string(), net_bus("y", data_width)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);

        let num_vectors = 80;
        let mut eval =
            Evaluator::new(&nir, num_vectors, SimOptions::default()).expect("create evaluator");

        let mut rng = StdRng::seed_from_u64(0xDEADBEEFu64);

        let mut inputs = Packed::new(num_vectors);
        let width = data_width as usize;
        let input_a = inputs.allocate(width);
        let input_b = inputs.allocate(width);
        let select_index = inputs.allocate(1);

        let words_per_lane = inputs.words_per_lane();

        for lane in 0..width {
            let lane_words = inputs.lane_mut(input_a, lane);
            for word_idx in 0..words_per_lane {
                lane_words[word_idx] = rng.next_u64();
            }
        }

        for lane in 0..width {
            let lane_words = inputs.lane_mut(input_b, lane);
            for word_idx in 0..words_per_lane {
                lane_words[word_idx] = rng.next_u64();
            }
        }

        {
            let select_words = inputs.lane_mut(select_index, 0);
            for word_idx in 0..words_per_lane {
                let mask = mask_for_word(word_idx, words_per_lane, num_vectors);
                select_words[word_idx] = rng.next_u64() & mask;
            }
        }

        eval.set_inputs(&inputs).expect("set inputs");
        eval.comb_eval().expect("comb eval");

        let outputs = eval.get_outputs();
        let output_index = *eval.output_ports.get("y").expect("output port");

        let select_words = inputs.lane(select_index, 0);
        for lane in 0..width {
            let a_lane = inputs.lane(input_a, lane);
            let b_lane = inputs.lane(input_b, lane);
            let expected: Vec<u64> = (0..words_per_lane)
                .map(|word_idx| {
                    let mask = mask_for_word(word_idx, words_per_lane, num_vectors);
                    let sel = select_words[word_idx] & mask;
                    let not_sel = (!sel) & mask;
                    (sel & b_lane[word_idx]) | (not_sel & a_lane[word_idx])
                })
                .collect();
            assert_eq!(outputs.lane(output_index, lane), expected.as_slice());
        }
    }

    #[test]
    fn comb_eval_mux_lane_boundaries() {
        let data_width = 2u32;

        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: data_width,
                attrs: None,
            },
        );
        ports.insert(
            "b".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: data_width,
                attrs: None,
            },
        );
        ports.insert(
            "sel".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 1,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: data_width,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: data_width,
                attrs: None,
            },
        );
        nets.insert(
            "b".to_string(),
            NirNet {
                bits: data_width,
                attrs: None,
            },
        );
        nets.insert(
            "sel".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: data_width,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "mux0".to_string(),
            NirNode {
                uid: "mux0".to_string(),
                op: NodeOp::Mux,
                width: data_width,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bus("a", data_width)),
                    ("B".to_string(), net_bus("b", data_width)),
                    ("S".to_string(), net_bit("sel")),
                    ("Y".to_string(), net_bus("y", data_width)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);

        let num_vectors = 96;
        let mut eval =
            Evaluator::new(&nir, num_vectors, SimOptions::default()).expect("create evaluator");

        let mut inputs = Packed::new(num_vectors);
        let width = data_width as usize;
        let input_a = inputs.allocate(width);
        let input_b = inputs.allocate(width);
        let select_index = inputs.allocate(1);

        let words_per_lane = inputs.words_per_lane();
        assert_eq!(words_per_lane, 2);

        let mask0 = mask_for_word(0, words_per_lane, num_vectors);
        let mask1 = mask_for_word(1, words_per_lane, num_vectors);

        {
            let select_words = inputs.lane_mut(select_index, 0);
            select_words[0] = 0;
            select_words[1] = mask1;
        }

        inputs
            .lane_mut(input_a, 0)
            .copy_from_slice(&[0x1234_5678_9ABC_DEF0u64, 0x0F0F_0F0F_0F0F_0F0Fu64]);
        inputs
            .lane_mut(input_a, 1)
            .copy_from_slice(&[0xAAAA_AAAA_AAAA_AAAAu64, 0x5555_5555_5555_5555u64]);

        inputs
            .lane_mut(input_b, 0)
            .copy_from_slice(&[0x1111_1111_1111_1111u64, 0xFFFF_FFFF_FFFF_FFFFu64]);
        inputs
            .lane_mut(input_b, 1)
            .copy_from_slice(&[0x2222_2222_2222_2222u64, 0x3333_3333_3333_3333u64]);

        eval.set_inputs(&inputs).expect("set inputs");
        eval.comb_eval().expect("comb eval");

        let outputs = eval.get_outputs();
        let output_index = *eval.output_ports.get("y").expect("output port");

        let lane0 = outputs.lane(output_index, 0);
        assert_eq!(lane0[0], inputs.lane(input_a, 0)[0] & mask0);
        assert_eq!(lane0[1], inputs.lane(input_b, 0)[1] & mask1);

        let lane1 = outputs.lane(output_index, 1);
        assert_eq!(lane1[0], inputs.lane(input_a, 1)[0] & mask0);
        assert_eq!(lane1[1], inputs.lane(input_b, 1)[1] & mask1);
    }

    #[test]
    fn slice_kernel_transfers_bits_across_lanes() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "wide".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 96,
                attrs: None,
            },
        );
        ports.insert(
            "slice".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 70,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "wide".to_string(),
            NirNet {
                bits: 96,
                attrs: None,
            },
        );
        nets.insert(
            "slice".to_string(),
            NirNet {
                bits: 70,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "slice0".to_string(),
            NirNode {
                uid: "slice0".to_string(),
                op: NodeOp::Slice,
                width: 70,
                pin_map: BTreeMap::from([
                    (
                        "A".to_string(),
                        BitRef::Net(BitRefNet {
                            net: "wide".to_string(),
                            lsb: 13,
                            msb: 82,
                        }),
                    ),
                    (
                        "Y".to_string(),
                        BitRef::Net(BitRefNet {
                            net: "slice".to_string(),
                            lsb: 0,
                            msb: 69,
                        }),
                    ),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);
        let num_vectors = 256;
        let mut eval = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();

        let mut rng = StdRng::seed_from_u64(0x5EED_5EED);
        let wide_values = random_biguints(96, num_vectors, &mut rng);
        let mut inputs = HashMap::new();
        inputs.insert("wide".to_string(), wide_values.clone());
        let packed_inputs = eval.pack_inputs_from_biguints(&inputs).unwrap();
        eval.set_inputs(&packed_inputs).unwrap();
        eval.comb_eval().unwrap();

        let outputs = eval.get_outputs();
        let unpacked = eval.unpack_outputs_to_biguints(&outputs).unwrap();
        let slice_values = unpacked.get("slice").expect("slice output");
        let mask: BigUint = (BigUint::from(1u8) << 70) - BigUint::from(1u8);

        for (input, output) in wide_values.iter().zip(slice_values.iter()) {
            let mut expected = input.clone();
            expected >>= 13u32;
            expected &= mask.clone();
            assert_eq!(output, &expected);
        }
    }

    #[test]
    fn cat_kernel_concatenates_segments() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "left".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 60,
                attrs: None,
            },
        );
        ports.insert(
            "right".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 45,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 105,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "left".to_string(),
            NirNet {
                bits: 60,
                attrs: None,
            },
        );
        nets.insert(
            "right".to_string(),
            NirNet {
                bits: 45,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 105,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "cat0".to_string(),
            NirNode {
                uid: "cat0".to_string(),
                op: NodeOp::Cat,
                width: 105,
                pin_map: BTreeMap::from([
                    (
                        "A".to_string(),
                        BitRef::Concat(BitRefConcat {
                            concat: vec![
                                BitRef::Net(BitRefNet {
                                    net: "left".to_string(),
                                    lsb: 0,
                                    msb: 59,
                                }),
                                BitRef::Net(BitRefNet {
                                    net: "right".to_string(),
                                    lsb: 0,
                                    msb: 44,
                                }),
                            ],
                        }),
                    ),
                    (
                        "Y".to_string(),
                        BitRef::Net(BitRefNet {
                            net: "y".to_string(),
                            lsb: 0,
                            msb: 104,
                        }),
                    ),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);
        let num_vectors = 192;
        let mut eval = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();

        let mut rng = StdRng::seed_from_u64(0xFACE_FEED);
        let left_values = random_biguints(60, num_vectors, &mut rng);
        let right_values = random_biguints(45, num_vectors, &mut rng);
        let mut inputs = HashMap::new();
        inputs.insert("left".to_string(), left_values.clone());
        inputs.insert("right".to_string(), right_values.clone());
        let packed = eval.pack_inputs_from_biguints(&inputs).unwrap();
        eval.set_inputs(&packed).unwrap();
        eval.comb_eval().unwrap();

        let outputs = eval.get_outputs();
        let unpacked = eval.unpack_outputs_to_biguints(&outputs).unwrap();
        let y_values = unpacked.get("y").expect("cat output");

        for ((left, right), result) in left_values
            .iter()
            .zip(right_values.iter())
            .zip(y_values.iter())
        {
            let expected = left.clone() + (right.clone() << 60);
            assert_eq!(result, &expected);
        }
    }

    #[test]
    fn const_kernel_prefills_multiple_bits() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 76,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 76,
                attrs: None,
            },
        );

        let mut params = BTreeMap::new();
        params.insert(
            "value".to_string(),
            Value::String("0x1234_5678_9ABC_DEF0_123".to_string()),
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "const0".to_string(),
            NirNode {
                uid: "const0".to_string(),
                op: NodeOp::Const,
                width: 76,
                pin_map: BTreeMap::from([("Y".to_string(), net_bus("y", 76))]),
                params: Some(params),
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);
        let num_vectors = 130;
        let mut eval = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();
        eval.comb_eval().unwrap();
        let outputs = eval.get_outputs();
        let unpacked = eval.unpack_outputs_to_biguints(&outputs).unwrap();
        let values = unpacked.get("y").expect("const output");

        let expected = BigUint::from_str_radix("123456789ABCDEF0123", 16).unwrap();
        for value in values {
            assert_eq!(value, &expected);
        }
    }

    #[test]
    fn add_kernel_matches_expected_results() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 70,
                attrs: None,
            },
        );
        ports.insert(
            "b".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 70,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 70,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: 70,
                attrs: None,
            },
        );
        nets.insert(
            "b".to_string(),
            NirNet {
                bits: 70,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 70,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "add0".to_string(),
            NirNode {
                uid: "add0".to_string(),
                op: NodeOp::Add,
                width: 70,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bus("a", 70)),
                    ("B".to_string(), net_bus("b", 70)),
                    ("Y".to_string(), net_bus("y", 70)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);
        let num_vectors = 10_000;
        let mut eval = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();

        let mut rng = StdRng::seed_from_u64(0xDEC0_ADD0);
        let mut a_values = random_biguints(70, num_vectors, &mut rng);
        let mut b_values = random_biguints(70, num_vectors, &mut rng);

        let max_value: BigUint = (BigUint::from(1u8) << 70) - BigUint::from(1u8);
        if !a_values.is_empty() {
            a_values[0] = max_value.clone();
            b_values[0] = BigUint::from(1u8);
        }
        if num_vectors > 1 {
            a_values[1] = BigUint::default();
            b_values[1] = max_value.clone();
        }
        if num_vectors > 2 {
            a_values[2] = (BigUint::from(1u8) << 64) - BigUint::from(1u8);
            b_values[2] = BigUint::from(1u8);
        }

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), a_values.clone());
        inputs.insert("b".to_string(), b_values.clone());
        let packed = eval.pack_inputs_from_biguints(&inputs).unwrap();
        eval.set_inputs(&packed).unwrap();
        eval.comb_eval().unwrap();

        let outputs = eval.get_outputs();
        let unpacked = eval.unpack_outputs_to_biguints(&outputs).unwrap();
        let sums = unpacked.get("y").expect("add output");
        let mask = &max_value;

        for ((a, b), result) in a_values.iter().zip(b_values.iter()).zip(sums.iter()) {
            let expected = (a + b) & mask;
            assert_eq!(result, &expected);
        }
    }

    #[test]
    fn sub_kernel_matches_expected_results() {
        let mut ports = BTreeMap::new();
        ports.insert(
            "a".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 68,
                attrs: None,
            },
        );
        ports.insert(
            "b".to_string(),
            NirPort {
                dir: PortDirection::Input,
                bits: 68,
                attrs: None,
            },
        );
        ports.insert(
            "y".to_string(),
            NirPort {
                dir: PortDirection::Output,
                bits: 68,
                attrs: None,
            },
        );

        let mut nets = BTreeMap::new();
        nets.insert(
            "a".to_string(),
            NirNet {
                bits: 68,
                attrs: None,
            },
        );
        nets.insert(
            "b".to_string(),
            NirNet {
                bits: 68,
                attrs: None,
            },
        );
        nets.insert(
            "y".to_string(),
            NirNet {
                bits: 68,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "sub0".to_string(),
            NirNode {
                uid: "sub0".to_string(),
                op: NodeOp::Sub,
                width: 68,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bus("a", 68)),
                    ("B".to_string(), net_bus("b", 68)),
                    ("Y".to_string(), net_bus("y", 68)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule { ports, nets, nodes };
        let nir = build_nir(module);
        let num_vectors = 10_000;
        let mut eval = Evaluator::new(&nir, num_vectors, SimOptions::default()).unwrap();

        let mut rng = StdRng::seed_from_u64(0x5EED_BAAD);
        let mut a_values = random_biguints(68, num_vectors, &mut rng);
        let mut b_values = random_biguints(68, num_vectors, &mut rng);

        let modulus = BigUint::from(1u8) << 68;
        let mask: BigUint = (&modulus) - BigUint::from(1u8);

        if !a_values.is_empty() {
            a_values[0] = BigUint::default();
            b_values[0] = BigUint::from(1u8);
        }
        if num_vectors > 1 {
            a_values[1] = mask.clone();
            b_values[1] = mask.clone();
        }
        if num_vectors > 2 {
            a_values[2] = BigUint::from(1u8);
            b_values[2] = mask.clone();
        }

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), a_values.clone());
        inputs.insert("b".to_string(), b_values.clone());
        let packed = eval.pack_inputs_from_biguints(&inputs).unwrap();
        eval.set_inputs(&packed).unwrap();
        eval.comb_eval().unwrap();

        let outputs = eval.get_outputs();
        let unpacked = eval.unpack_outputs_to_biguints(&outputs).unwrap();
        let diffs = unpacked.get("y").expect("sub output");

        for ((a, b), result) in a_values.iter().zip(b_values.iter()).zip(diffs.iter()) {
            let expected = (a + &modulus - b) & &mask;
            assert_eq!(result, &expected);
        }
    }

    fn net_bit_index(name: &str, index: u32) -> BitRef {
        BitRef::Net(BitRefNet {
            net: name.to_string(),
            lsb: index,
            msb: index,
        })
    }

    fn build_counter_nir(width: u32) -> Nir {
        let ports = BTreeMap::new();

        let mut nets = BTreeMap::new();
        nets.insert(
            "clk".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );

        for bit in 0..=width {
            nets.insert(
                format!("carry_{bit}"),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
        }

        for bit in 0..width {
            nets.insert(
                format!("value_{bit}"),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
            nets.insert(
                format!("next_{bit}"),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
            nets.insert(
                format!("not_{bit}"),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
        }

        let mut nodes = BTreeMap::new();

        let mut const_params = BTreeMap::new();
        const_params.insert("value".to_string(), Value::String("1".to_string()));
        nodes.insert(
            "carry_init".to_string(),
            NirNode {
                uid: "carry_init".to_string(),
                op: NodeOp::Const,
                width: 1,
                pin_map: BTreeMap::from([("Y".to_string(), net_bit("carry_0"))]),
                params: Some(const_params),
                attrs: None,
            },
        );

        for bit in 0..width {
            let value_net = format!("value_{bit}");
            let next_net = format!("next_{bit}");
            let carry_net = format!("carry_{bit}");
            let carry_next_net = format!("carry_{}", bit + 1);
            let not_net = format!("not_{bit}");
            let state_name = format!("state_{bit}");

            nodes.insert(
                format!("inv_{bit}"),
                NirNode {
                    uid: format!("inv_{bit}"),
                    op: NodeOp::Not,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit_index(&value_net, 0)),
                        ("Y".to_string(), net_bit_index(&not_net, 0)),
                    ]),
                    params: None,
                    attrs: None,
                },
            );

            nodes.insert(
                format!("sum_{bit}"),
                NirNode {
                    uid: format!("sum_{bit}"),
                    op: NodeOp::Mux,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit_index(&value_net, 0)),
                        ("B".to_string(), net_bit_index(&not_net, 0)),
                        ("S".to_string(), net_bit_index(&carry_net, 0)),
                        ("Y".to_string(), net_bit_index(&next_net, 0)),
                    ]),
                    params: None,
                    attrs: None,
                },
            );

            nodes.insert(
                format!("carry_and_{bit}"),
                NirNode {
                    uid: format!("carry_and_{bit}"),
                    op: NodeOp::And,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit_index(&value_net, 0)),
                        ("B".to_string(), net_bit_index(&carry_net, 0)),
                        ("Y".to_string(), net_bit_index(&carry_next_net, 0)),
                    ]),
                    params: None,
                    attrs: None,
                },
            );

            nodes.insert(
                state_name.clone(),
                NirNode {
                    uid: state_name,
                    op: NodeOp::Dff,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("D".to_string(), net_bit_index(&next_net, 0)),
                        ("Q".to_string(), net_bit_index(&value_net, 0)),
                        ("CLK".to_string(), net_bit("clk")),
                    ]),
                    params: None,
                    attrs: None,
                },
            );
        }

        let module = NirModule { ports, nets, nodes };
        build_nir(module)
    }

    fn packed_scalar_value(packed: &Packed, index: PackedIndex, width: usize) -> u64 {
        let words_per_lane = packed.words_per_lane();
        if words_per_lane == 0 {
            return 0;
        }

        let mut value = 0u64;
        for bit in 0..width {
            let lane_words = packed.lane(index, bit);
            if !lane_words.is_empty() && (lane_words[0] & 1) != 0 {
                value |= 1u64 << bit;
            }
        }
        value
    }

    fn packed_registers_value(packed: &Packed, indices: &[PackedIndex]) -> u64 {
        if packed.words_per_lane() == 0 {
            return 0;
        }

        indices.iter().enumerate().fold(0u64, |acc, (bit, index)| {
            let lane = packed.lane(*index, 0);
            if !lane.is_empty() && (lane[0] & 1) != 0 {
                acc | (1u64 << bit)
            } else {
                acc
            }
        })
    }

    #[test]
    fn dff_register_state_persists() {
        let nir = build_counter_nir(1);
        let mut eval = Evaluator::new(&nir, 1, SimOptions::default()).expect("create evaluator");
        let reset_mask = PackedBitMask::new(1);
        let state_index = eval.register_index("state_0").expect("register index");

        let mut expected_state = packed_scalar_value(&eval.get_registers_q(), state_index, 1);
        assert_eq!(expected_state, 0);

        for _ in 0..8 {
            eval.comb_eval().expect("comb eval");

            let current_state = packed_scalar_value(&eval.get_registers_q(), state_index, 1);
            assert_eq!(current_state, expected_state);

            let next_state = packed_scalar_value(&eval.get_registers_d(), state_index, 1);
            let expected_next = expected_state ^ 1;
            assert_eq!(next_state, expected_next);

            eval.step_clock(&reset_mask).expect("step clock");
            expected_state = packed_scalar_value(&eval.get_registers_q(), state_index, 1);
            assert_eq!(expected_state, expected_next);
        }
    }

    #[test]
    fn dff_32bit_counter_increments() {
        let width = 32u32;
        let nir = build_counter_nir(width);
        let mut eval = Evaluator::new(&nir, 1, SimOptions::default()).expect("create evaluator");
        let reset_mask = PackedBitMask::new(1);
        let register_indices: Vec<PackedIndex> = (0..width as usize)
            .map(|bit| {
                eval.register_index(&format!("state_{bit}"))
                    .expect("register index")
            })
            .collect();

        let mut current = packed_registers_value(&eval.get_registers_q(), &register_indices);
        assert_eq!(current, 0);

        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };

        for _ in 0..70 {
            eval.comb_eval().expect("comb eval");
            let next = packed_registers_value(&eval.get_registers_d(), &register_indices);
            assert_eq!(next, (current + 1) & mask);

            eval.step_clock(&reset_mask).expect("step clock");
            current = packed_registers_value(&eval.get_registers_q(), &register_indices);
            assert_eq!(current, next);
        }
    }
}

impl<'nir> Evaluator<'nir> {
    pub fn new(nir: &'nir Nir, num_vectors: usize, options: SimOptions) -> Result<Self, Error> {
        let module = nir
            .modules
            .get(nir.top.as_str())
            .ok_or_else(|| Error::MissingTop {
                design: nir.design.clone(),
                top: nir.top.clone(),
            })?;

        let graph = ModuleGraph::from_module(module)?;
        let (topo, topo_level_offsets, topo_level_map) =
            graph.combinational_topological_levels()?.into_parts();

        let mut nets = Packed::new(num_vectors);
        let mut net_indices = HashMap::new();
        let mut net_indices_by_id = HashMap::new();
        for (name, net) in &module.nets {
            let index = nets.allocate(net.bits as usize);
            net_indices.insert(name.clone(), index);
            let net_id = graph
                .net_id(name.as_str())
                .expect("net must exist in module graph");
            net_indices_by_id.insert(net_id, index);
        }

        let mut regs_cur = Packed::new(num_vectors);
        let mut reg_indices = HashMap::new();
        for (name, node) in &module.nodes {
            if matches!(node.op, NodeOp::Dff | NodeOp::Latch) {
                let index = regs_cur.allocate(node.width as usize);
                reg_indices.insert(name.clone(), index);
            }
        }
        let regs_next = regs_cur.duplicate_layout();

        let mut node_bindings: HashMap<NodeId, NodePinBindings> = HashMap::new();
        let mut const_pool = ConstPool::default();
        for (node_name, node) in &module.nodes {
            let node_id = graph
                .node_id(node_name.as_str())
                .expect("node must exist in graph");
            let entry = node_bindings
                .entry(node_id)
                .or_insert_with(NodePinBindings::default);
            for (pin_name, bitref) in &node.pin_map {
                let binding =
                    bind_bitref(module, &graph, bitref, &mut const_pool).map_err(|source| {
                        Error::PinBinding {
                            node: node_name.clone(),
                            pin: pin_name.clone(),
                            source,
                        }
                    })?;
                entry.pins.insert(pin_name.clone(), binding);
            }
        }

        let mut dff_nodes = Vec::new();
        for (node_name, node) in &module.nodes {
            if matches!(node.op, NodeOp::Dff) {
                let node_id = graph
                    .node_id(node_name.as_str())
                    .expect("node must exist in graph");
                let bindings = node_bindings
                    .get(&node_id)
                    .expect("DFF node must have pin bindings");
                let d_binding = bindings
                    .pins
                    .get("D")
                    .expect("DFF node must bind D pin")
                    .clone();
                let q_binding = bindings
                    .pins
                    .get("Q")
                    .expect("DFF node must bind Q pin")
                    .clone();
                let reg_index = *reg_indices
                    .get(node_name)
                    .expect("DFF node must have allocated register");
                dff_nodes.push(DffInfo {
                    reg_index,
                    d_binding,
                    q_binding,
                });
            }
        }

        let mut inputs = Packed::new(num_vectors);
        let mut outputs = Packed::new(num_vectors);
        let mut input_ports = HashMap::new();
        let mut output_ports = HashMap::new();

        for (name, port) in &module.ports {
            let width = port.bits as usize;
            match port.dir {
                PortDirection::Input => {
                    let index = inputs.allocate(width);
                    input_ports.insert(name.clone(), index);
                }
                PortDirection::Output => {
                    let index = outputs.allocate(width);
                    output_ports.insert(name.clone(), index);
                }
                PortDirection::Inout => {
                    let input_index = inputs.allocate(width);
                    let output_index = outputs.allocate(width);
                    input_ports.insert(name.clone(), input_index);
                    output_ports.insert(name.clone(), output_index);
                }
            }
        }

        let words_per_lane = nets.words_per_lane();
        let comb_kernels = Self::build_comb_kernels(
            module,
            &graph,
            &net_indices,
            &net_indices_by_id,
            &node_bindings,
            &const_pool,
            words_per_lane,
            num_vectors,
        );
        let carry_scratch = vec![0; words_per_lane];

        Ok(Self {
            nir,
            module,
            graph,
            topo,
            topo_level_offsets,
            topo_level_map,
            options,
            num_vectors,
            nets,
            net_indices,
            net_indices_by_id,
            regs_cur,
            regs_next,
            reg_indices,
            inputs,
            input_ports,
            outputs,
            output_ports,
            node_bindings,
            dff_nodes,
            const_pool,
            comb_kernels,
            carry_scratch,
        })
    }

    pub fn pack_inputs_from_biguints(
        &self,
        port_vectors: &HashMap<String, Vec<BigUint>>,
    ) -> Result<Packed, PortValueError> {
        let mut packed = self.inputs.duplicate_layout();

        for (name, index) in &self.input_ports {
            let port = self
                .module
                .ports
                .get(name.as_str())
                .expect("input port must exist");

            if !matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
                return Err(PortValueError::DirectionMismatch {
                    name: name.clone(),
                    actual: port.dir.clone(),
                });
            }

            let values = port_vectors
                .get(name)
                .ok_or_else(|| PortValueError::MissingPort { name: name.clone() })?;

            pack_port_biguints(&mut packed, *index, port.bits as usize, values, name)?;
        }

        for name in port_vectors.keys() {
            if !self.input_ports.contains_key(name) {
                return Err(PortValueError::UnexpectedPort { name: name.clone() });
            }
        }

        Ok(packed)
    }

    pub fn pack_inputs_from_bytes(
        &self,
        port_vectors: &HashMap<String, Vec<Vec<u8>>>,
    ) -> Result<Packed, PortValueError> {
        let mut packed = self.inputs.duplicate_layout();

        for (name, index) in &self.input_ports {
            let port = self
                .module
                .ports
                .get(name.as_str())
                .expect("input port must exist");

            if !matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
                return Err(PortValueError::DirectionMismatch {
                    name: name.clone(),
                    actual: port.dir.clone(),
                });
            }

            let values = port_vectors
                .get(name)
                .ok_or_else(|| PortValueError::MissingPort { name: name.clone() })?;

            if values.len() != self.num_vectors {
                return Err(PortValueError::VectorCountMismatch {
                    name: name.clone(),
                    expected: self.num_vectors,
                    actual: values.len(),
                });
            }

            let mut converted = Vec::with_capacity(values.len());
            for bytes in values {
                converted.push(BigUint::from_bytes_le(bytes));
            }

            pack_port_biguints(&mut packed, *index, port.bits as usize, &converted, name)?;
        }

        for name in port_vectors.keys() {
            if !self.input_ports.contains_key(name) {
                return Err(PortValueError::UnexpectedPort { name: name.clone() });
            }
        }

        Ok(packed)
    }

    pub fn unpack_outputs_to_biguints(
        &self,
        outputs: &Packed,
    ) -> Result<HashMap<String, Vec<BigUint>>, PortValueError> {
        if outputs.num_vectors() != self.num_vectors {
            return Err(PortValueError::PackedVectorMismatch {
                expected: self.num_vectors,
                actual: outputs.num_vectors(),
            });
        }

        let expected_words = self.outputs.words_per_lane();
        if outputs.words_per_lane() != expected_words {
            return Err(PortValueError::WordsPerLaneMismatch {
                expected: expected_words,
                actual: outputs.words_per_lane(),
            });
        }

        let mut result = HashMap::with_capacity(self.output_ports.len());

        for (name, index) in &self.output_ports {
            let port = self
                .module
                .ports
                .get(name.as_str())
                .expect("output port must exist");

            if !matches!(port.dir, PortDirection::Output | PortDirection::Inout) {
                return Err(PortValueError::DirectionMismatch {
                    name: name.clone(),
                    actual: port.dir.clone(),
                });
            }

            let values = unpack_port_biguints(outputs, *index, port.bits as usize, name)?;
            result.insert(name.clone(), values);
        }

        Ok(result)
    }

    pub fn unpack_outputs_to_bytes(
        &self,
        outputs: &Packed,
    ) -> Result<HashMap<String, Vec<Vec<u8>>>, PortValueError> {
        let bigints = self.unpack_outputs_to_biguints(outputs)?;
        let mut result = HashMap::with_capacity(bigints.len());

        for (name, values) in bigints {
            let width_bits = self
                .module
                .ports
                .get(name.as_str())
                .expect("output port must exist")
                .bits as usize;
            let byte_len = div_ceil(width_bits, 8);

            let mut port_bytes = Vec::with_capacity(values.len());
            for value in values {
                let mut bytes = value.to_bytes_le();
                if bytes.len() < byte_len {
                    bytes.resize(byte_len, 0);
                } else if bytes.len() > byte_len {
                    bytes.truncate(byte_len);
                }
                port_bytes.push(bytes);
            }

            result.insert(name, port_bytes);
        }

        Ok(result)
    }

    pub fn set_inputs(&mut self, port_values: &Packed) -> Result<(), Error> {
        self.inputs.copy_from(port_values)?;
        Ok(())
    }

    pub fn comb_eval(&mut self) -> Result<(), Error> {
        self.stage_inputs();
        self.stage_registers();

        let levels = self.topo_level_offsets.len().saturating_sub(1);
        for level in 0..levels {
            let start = self.topo_level_offsets[level];
            let end = self.topo_level_offsets[level + 1];
            let level_nodes: Vec<NodeId> = self.topo[start..end].to_vec();
            for node_id in level_nodes {
                let kernel = self.comb_kernels.get(&node_id).cloned();
                if let Some(kernel) = kernel {
                    self.execute_kernel(&kernel);
                }
            }
        }

        self.capture_registers();
        self.stage_outputs();
        Ok(())
    }

    fn stage_inputs(&mut self) {
        for (port_name, &port_index) in &self.input_ports {
            if let Some(&net_index) = self.net_indices.get(port_name.as_str()) {
                let values = self.inputs.slice(port_index);
                self.nets.slice_mut(net_index).copy_from_slice(values);
            }
        }
    }

    fn stage_registers(&mut self) {
        let bits_per_lane = LANE_BITS as usize;

        for dff in &self.dff_nodes {
            let mut reg_lane = 0usize;
            for descriptor in dff.q_binding.descriptors() {
                let width = descriptor.width as usize;
                let base_bit = descriptor.lane_offset as usize * bits_per_lane
                    + descriptor.bit_offset as usize;
                match descriptor.source {
                    SignalId::Net(net_id) => {
                        let net_index = *self
                            .net_indices_by_id
                            .get(&net_id)
                            .expect("net index for descriptor");
                        for bit in 0..width {
                            let reg_lane_idx = reg_lane + bit;
                            debug_assert!(reg_lane_idx < dff.reg_index.lanes());
                            let net_lane = base_bit + bit;
                            debug_assert!(net_lane < net_index.lanes());
                            let source = self.regs_cur.lane(dff.reg_index, reg_lane_idx);
                            let dest = self.nets.lane_mut(net_index, net_lane);
                            dest.copy_from_slice(source);
                        }
                    }
                    SignalId::Const(_) => {
                        debug_assert!(false, "DFF Q pins should not bind constants");
                    }
                }
                reg_lane += width;
            }
            debug_assert_eq!(reg_lane, dff.reg_index.lanes());
        }
    }

    fn stage_outputs(&mut self) {
        for (port_name, &port_index) in &self.output_ports {
            if let Some(&net_index) = self.net_indices.get(port_name.as_str()) {
                let values = self.nets.slice(net_index);
                self.outputs.slice_mut(port_index).copy_from_slice(values);
            }
        }
    }

    fn capture_registers(&mut self) {
        let bits_per_lane = LANE_BITS as usize;
        let words_per_lane = self.regs_next.words_per_lane();

        for dff in &self.dff_nodes {
            {
                let slice = self.regs_next.slice_mut(dff.reg_index);
                slice.fill(0);
            }

            let mut reg_lane = 0usize;
            for descriptor in dff.d_binding.descriptors() {
                let width = descriptor.width as usize;
                let base_bit = descriptor.lane_offset as usize * bits_per_lane
                    + descriptor.bit_offset as usize;
                match descriptor.source {
                    SignalId::Net(net_id) => {
                        let net_index = *self
                            .net_indices_by_id
                            .get(&net_id)
                            .expect("net index for descriptor");
                        for bit in 0..width {
                            let reg_lane_idx = reg_lane + bit;
                            debug_assert!(reg_lane_idx < dff.reg_index.lanes());
                            let net_lane = base_bit + bit;
                            debug_assert!(net_lane < net_index.lanes());
                            let source = self.nets.lane(net_index, net_lane);
                            let dest = self.regs_next.lane_mut(dff.reg_index, reg_lane_idx);
                            dest.copy_from_slice(source);
                        }
                    }
                    SignalId::Const(const_id) => {
                        let const_value = self.const_pool.get(const_id);
                        let lane_index = descriptor.lane_offset as usize;
                        let chunk = const_value.words.get(lane_index).copied().unwrap_or(0);
                        for bit in 0..width {
                            let bit_index = descriptor.bit_offset as usize + bit;
                            let reg_lane_idx = reg_lane + bit;
                            debug_assert!(reg_lane_idx < dff.reg_index.lanes());
                            let dest = self.regs_next.lane_mut(dff.reg_index, reg_lane_idx);
                            if ((chunk >> bit_index) & 1) != 0 {
                                for word_idx in 0..words_per_lane {
                                    dest[word_idx] =
                                        mask_for_word(word_idx, words_per_lane, self.num_vectors);
                                }
                            } else {
                                dest.fill(0);
                            }
                        }
                    }
                }
                reg_lane += width;
            }
            debug_assert_eq!(reg_lane, dff.reg_index.lanes());
        }
    }

    fn execute_kernel(&mut self, kernel: &NodeKernel) {
        match kernel {
            NodeKernel::Const(kernel) | NodeKernel::Slice(kernel) | NodeKernel::Cat(kernel) => {
                self.run_transfer(kernel)
            }
            NodeKernel::Not(kernel) => self.run_not(kernel),
            NodeKernel::And(kernel) => self.run_and(kernel),
            NodeKernel::Mux(kernel) => self.run_mux(kernel),
            NodeKernel::Add(kernel) => self.run_arith(kernel),
            NodeKernel::Sub(kernel) => self.run_arith(kernel),
        }
    }

    fn run_transfer(&mut self, kernel: &TransferKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        for segment in &kernel.plan {
            let len = segment.lanes * words_per_lane;
            if len == 0 {
                continue;
            }
            let dest_start = segment.dest_offset;
            match &segment.source {
                SegmentSource::Net { offset } => {
                    let src_start = *offset;
                    unsafe {
                        let src_ptr = self.nets.storage.as_ptr().add(src_start);
                        let dest_ptr = self.nets.storage.as_mut_ptr().add(dest_start);
                        ptr::copy(src_ptr, dest_ptr, len);
                    }
                }
                SegmentSource::Const { words } => {
                    let dest_end = dest_start + len;
                    let dest_slice = &mut self.nets.storage[dest_start..dest_end];
                    dest_slice.copy_from_slice(words);
                }
            }
        }
    }

    fn run_not(&mut self, kernel: &UnaryKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        debug_assert_eq!(kernel.input.lanes, kernel.output.lanes);

        for lane in 0..kernel.output.lanes {
            let input_base = kernel.input.lane_offset(lane, words_per_lane);
            let output_base = kernel.output.lane_offset(lane, words_per_lane);

            for word_idx in 0..words_per_lane {
                let mask = mask_for_word(word_idx, words_per_lane, self.num_vectors);
                let input_word = self.nets.storage[input_base + word_idx];
                self.nets.storage[output_base + word_idx] = (!input_word) & mask;
            }
        }
    }

    fn run_and(&mut self, kernel: &BinaryKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        debug_assert_eq!(kernel.input_a.lanes, kernel.output.lanes);
        debug_assert_eq!(kernel.input_b.lanes, kernel.output.lanes);

        for lane in 0..kernel.output.lanes {
            let input_a_base = kernel.input_a.lane_offset(lane, words_per_lane);
            let input_b_base = kernel.input_b.lane_offset(lane, words_per_lane);
            let output_base = kernel.output.lane_offset(lane, words_per_lane);

            for word_idx in 0..words_per_lane {
                let mask = mask_for_word(word_idx, words_per_lane, self.num_vectors);
                let value = (self.nets.storage[input_a_base + word_idx]
                    & self.nets.storage[input_b_base + word_idx])
                    & mask;
                self.nets.storage[output_base + word_idx] = value;
            }
        }
    }

    fn run_mux(&mut self, kernel: &MuxKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        debug_assert_eq!(kernel.select.lanes(), 1);
        debug_assert_eq!(kernel.select_masks.len(), words_per_lane);

        let input_a = self.nets.slice(kernel.input_a).to_vec();
        let input_b = self.nets.slice(kernel.input_b).to_vec();
        let select_words = self.nets.slice(kernel.select).to_vec();
        debug_assert_eq!(select_words.len(), words_per_lane * kernel.select.lanes());
        let output_slice = self.nets.slice_mut(kernel.output);
        for (index, word) in output_slice.iter_mut().enumerate() {
            let lane_word = index % words_per_lane;
            let mask = kernel.select_masks[lane_word];
            let select = select_words[lane_word] & mask;
            let not_select = (!select) & mask;
            *word = (select & input_b[index]) | (not_select & input_a[index]);
        }
    }

    fn run_arith(&mut self, kernel: &ArithKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        if self.carry_scratch.len() < words_per_lane {
            self.carry_scratch.resize(words_per_lane, 0);
        }
        self.carry_scratch.fill(0);

        if kernel.initial_carry {
            for word_idx in 0..words_per_lane {
                self.carry_scratch[word_idx] =
                    mask_for_word(word_idx, words_per_lane, self.num_vectors);
            }
        }

        for bit in &kernel.bits {
            for word_idx in 0..words_per_lane {
                let mask = mask_for_word(word_idx, words_per_lane, self.num_vectors);
                let a_word = match bit.a {
                    BitSource::Net { offset } => self.nets.storage[offset + word_idx],
                    BitSource::Const { value } => {
                        if value {
                            mask
                        } else {
                            0
                        }
                    }
                };
                let mut b_word = match bit.b {
                    BitSource::Net { offset } => self.nets.storage[offset + word_idx],
                    BitSource::Const { value } => {
                        if value {
                            mask
                        } else {
                            0
                        }
                    }
                };

                if kernel.invert_b {
                    b_word = (!b_word) & mask;
                }

                let carry_in = self.carry_scratch[word_idx];
                let sum = (a_word ^ b_word) ^ carry_in;
                let carry_out = (a_word & b_word) | (a_word & carry_in) | (b_word & carry_in);

                self.nets.storage[bit.dest_offset + word_idx] = sum & mask;
                self.carry_scratch[word_idx] = carry_out & mask;
            }
        }
    }

    fn build_comb_kernels(
        module: &'nir Module,
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
                        Self::build_const_kernel(
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
                        Self::build_copy_kernel(
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
                        Self::build_copy_kernel(
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
                        Self::build_arith_kernel(
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
                        Self::build_arith_kernel(
                            bindings,
                            net_indices_by_id,
                            const_pool,
                            words_per_lane,
                            true,
                            true,
                        )
                    })
                    .map(NodeKernel::Sub),
                NodeOp::Not => Self::build_not_kernel(node, net_indices).map(NodeKernel::Not),
                NodeOp::And => Self::build_and_kernel(node, net_indices).map(NodeKernel::And),
                NodeOp::Mux => {
                    Self::build_mux_kernel(node, net_indices, words_per_lane, num_vectors)
                        .map(NodeKernel::Mux)
                }
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
        let bits = Self::parse_const_bits(literal, node.width)?;

        let output_binding = bindings.pins.get("Y")?;
        let destinations = Self::expand_binding_destinations(output_binding)?;
        if destinations.len() != bits.len() {
            return None;
        }

        let sources: Vec<PackedSourceKind> =
            bits.into_iter().map(PackedSourceKind::Literal).collect();

        let plan = Self::build_transfer_plan(
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

        let sources = Self::expand_binding_sources(input_binding);
        let destinations = Self::expand_binding_destinations(output_binding)?;

        if sources.len() != destinations.len() {
            return None;
        }

        let plan = Self::build_transfer_plan(
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

        let sources_a = Self::expand_binding_sources(input_a);
        let sources_b = Self::expand_binding_sources(input_b);
        let destinations = Self::expand_binding_destinations(output)?;

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
            let dest_offset = dest_index.offset + dest.lane * words_per_lane;
            let a_source = Self::convert_bit_source(
                &sources_a[idx],
                net_indices_by_id,
                const_pool,
                words_per_lane,
            )?;
            let b_source = Self::convert_bit_source(
                &sources_b[idx],
                net_indices_by_id,
                const_pool,
                words_per_lane,
            )?;
            bits.push(ArithBit {
                a: a_source,
                b: b_source,
                dest_offset,
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
                    offset: index.offset + lane * words_per_lane,
                })
            }
            PackedSourceKind::Const(const_id, lane) => Some(BitSource::Const {
                value: Self::const_pool_bit(const_pool, *const_id, *lane),
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
                        if words_per_lane != 0 {
                            for bit in bits {
                                for word_idx in 0..words_per_lane {
                                    let mask = mask_for_word(word_idx, words_per_lane, num_vectors);
                                    words.push(if bit { mask } else { 0 });
                                }
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
            let dest_offset = dest_index.offset + dest.lane * words_per_lane;
            let source = Self::convert_bit_source(
                source_kind,
                net_indices_by_id,
                const_pool,
                words_per_lane,
            )?;

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
                            if existing.dest_offset + existing.lanes * words_per_lane == dest_offset
                            {
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
        let input_a_ref = node.pin_map.get("A")?;
        let input_b_ref = node.pin_map.get("B")?;
        let output_ref = node.pin_map.get("Y")?;
        let input_a_net = bitref_full_net(input_a_ref, node.width)?;
        let input_b_net = bitref_full_net(input_b_ref, node.width)?;
        let output_net = bitref_full_net(output_ref, node.width)?;

        let input_a = *net_indices
            .get(input_a_net)
            .expect("AND A input net must be allocated");
        let input_b = *net_indices
            .get(input_b_net)
            .expect("AND B input net must be allocated");
        let output = *net_indices
            .get(output_net)
            .expect("AND output net must be allocated");

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

    pub fn step_clock(&mut self, _reset_mask: &PackedBitMask) -> Result<(), Error> {
        self.regs_cur.copy_from(&self.regs_next)?;
        Ok(())
    }

    pub fn get_outputs(&self) -> Packed {
        self.outputs.clone()
    }

    pub fn get_registers_q(&self) -> Packed {
        self.regs_cur.clone()
    }

    pub fn get_registers_d(&self) -> Packed {
        self.regs_next.clone()
    }

    pub fn register_index(&self, name: &str) -> Option<PackedIndex> {
        self.reg_indices.get(name).copied()
    }

    pub fn tick(&mut self, inputs: &Packed, reset_mask: &PackedBitMask) -> Result<Packed, Error> {
        self.set_inputs(inputs)?;
        self.comb_eval()?;
        self.step_clock(reset_mask)?;
        Ok(self.get_outputs())
    }

    pub fn options(&self) -> SimOptions {
        self.options
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
}
