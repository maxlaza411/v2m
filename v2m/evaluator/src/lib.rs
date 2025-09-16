use std::collections::{BTreeMap, HashMap};

use num_bigint::BigUint;
use num_traits::Zero;

use thiserror::Error;
use v2m_formats::nir::{BitRef, Module, Nir, NodeOp, PortDirection};
use v2m_nir::{BuildError as GraphBuildError, EvalOrderError, ModuleGraph, NodeId};

mod pin_binding;

use pin_binding::{bind_bitref, BitBinding, ConstPool, PinBindingError};

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
    net_indices_by_id: Vec<PackedIndex>,
    regs_cur: Packed,
    regs_next: Packed,
    reg_indices: HashMap<String, PackedIndex>,
    inputs: Packed,
    input_ports: HashMap<String, PackedIndex>,
    outputs: Packed,
    output_ports: HashMap<String, PackedIndex>,
    node_bindings: HashMap<NodeId, NodePinBindings>,
    const_pool: ConstPool,
    comb_kernels: HashMap<NodeId, NodeKernel>,
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
struct ConstKernel {
    output: PackedIndex,
    words: Vec<u64>,
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
enum NodeKernel {
    Const(ConstKernel),
    Not(UnaryKernel),
    And(BinaryKernel),
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

fn parse_const_bool(literal: &str) -> Option<bool> {
    let cleaned: String = literal.chars().filter(|c| *c != '_').collect();
    if cleaned.is_empty() {
        return None;
    }

    let (base, digits) = if let Some(rest) = cleaned.strip_prefix("0b") {
        (2, rest)
    } else if let Some(rest) = cleaned.strip_prefix("0B") {
        (2, rest)
    } else if let Some(rest) = cleaned.strip_prefix("0x") {
        (16, rest)
    } else if let Some(rest) = cleaned.strip_prefix("0X") {
        (16, rest)
    } else {
        (10, cleaned.as_str())
    };

    if digits.is_empty() {
        return None;
    }

    let value = u128::from_str_radix(digits, base).ok()?;
    if value > 1 {
        return None;
    }

    Some(value == 1)
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
        BitRef, BitRefNet, Module as NirModule, Net as NirNet, Node as NirNode, Port as NirPort,
    };

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
        let mut net_indices_by_id = Vec::with_capacity(module.nets.len());
        for (name, net) in &module.nets {
            let index = nets.allocate(net.bits as usize);
            net_indices.insert(name.clone(), index);
            net_indices_by_id.push(index);
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

        let comb_kernels = Self::build_comb_kernels(
            module,
            &graph,
            &net_indices,
            nets.words_per_lane(),
            num_vectors,
        );

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
            const_pool,
            comb_kernels,
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

        self.stage_outputs();
        Ok(())
    }

    fn stage_inputs(&mut self) {
        for (port_name, &port_index) in &self.input_ports {
            if let Some(&net_index) = self.net_indices.get(port_name.as_str()) {
                let values = self.inputs.slice(port_index).to_vec();
                self.nets.slice_mut(net_index).copy_from_slice(&values);
            }
        }
    }

    fn stage_outputs(&mut self) {
        for (port_name, &port_index) in &self.output_ports {
            if let Some(&net_index) = self.net_indices.get(port_name.as_str()) {
                let values = self.nets.slice(net_index).to_vec();
                self.outputs.slice_mut(port_index).copy_from_slice(&values);
            }
        }
    }

    fn execute_kernel(&mut self, kernel: &NodeKernel) {
        match kernel {
            NodeKernel::Const(kernel) => self.run_const(kernel),
            NodeKernel::Not(kernel) => self.run_not(kernel),
            NodeKernel::And(kernel) => self.run_and(kernel),
        }
    }

    fn run_const(&mut self, kernel: &ConstKernel) {
        let slice = self.nets.slice_mut(kernel.output);
        slice.copy_from_slice(&kernel.words);
    }

    fn run_not(&mut self, kernel: &UnaryKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        let input_words = self.nets.slice(kernel.input).to_vec();
        let output_slice = self.nets.slice_mut(kernel.output);
        for (index, word) in output_slice.iter_mut().enumerate() {
            let lane_word = index % words_per_lane;
            let mask = mask_for_word(lane_word, words_per_lane, self.num_vectors);
            *word = (!input_words[index]) & mask;
        }
    }

    fn run_and(&mut self, kernel: &BinaryKernel) {
        let words_per_lane = self.nets.words_per_lane();
        if words_per_lane == 0 {
            return;
        }

        let input_a = self.nets.slice(kernel.input_a).to_vec();
        let input_b = self.nets.slice(kernel.input_b).to_vec();
        let output_slice = self.nets.slice_mut(kernel.output);
        for (index, word) in output_slice.iter_mut().enumerate() {
            let lane_word = index % words_per_lane;
            let mask = mask_for_word(lane_word, words_per_lane, self.num_vectors);
            *word = (input_a[index] & input_b[index]) & mask;
        }
    }

    fn build_comb_kernels(
        module: &'nir Module,
        graph: &ModuleGraph,
        net_indices: &HashMap<String, PackedIndex>,
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
                NodeOp::Const => {
                    Self::build_const_kernel(node, net_indices, words_per_lane, num_vectors)
                        .map(NodeKernel::Const)
                }
                NodeOp::Not => Self::build_not_kernel(node, net_indices).map(NodeKernel::Not),
                NodeOp::And => Self::build_and_kernel(node, net_indices).map(NodeKernel::And),
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
        net_indices: &HashMap<String, PackedIndex>,
        words_per_lane: usize,
        num_vectors: usize,
    ) -> Option<ConstKernel> {
        if node.width != 1 {
            return None;
        }

        let output_ref = node.pin_map.get("Y").expect("CONST node must bind Y pin");
        let output_net = bitref_full_net(output_ref, node.width)?;
        let output_index = *net_indices
            .get(output_net)
            .expect("CONST node output net must be allocated");

        let literal = node
            .params
            .as_ref()
            .and_then(|params| params.get("value"))
            .and_then(|value| value.as_str())
            .expect("CONST node requires string `value` parameter");
        let bit_value =
            parse_const_bool(literal).expect("CONST node value must fit within one bit");

        let mut words = Vec::with_capacity(output_index.lanes() * words_per_lane);
        if words_per_lane == 0 {
            return Some(ConstKernel {
                output: output_index,
                words,
            });
        }

        for _ in 0..output_index.lanes() {
            for word_index in 0..words_per_lane {
                let mask = mask_for_word(word_index, words_per_lane, num_vectors);
                let value = if bit_value { mask } else { 0 };
                words.push(value);
            }
        }

        Some(ConstKernel {
            output: output_index,
            words,
        })
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

    pub fn step_clock(&mut self, _reset_mask: &PackedBitMask) -> Result<(), Error> {
        // Implementation to be added in a future revision.
        Ok(())
    }

    pub fn get_outputs(&self) -> Packed {
        self.outputs.clone()
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
