use std::collections::HashMap;

use thiserror::Error;
use v2m_formats::nir::{Module, Nir, NodeOp, PortDirection};
use v2m_nir::{BuildError as GraphBuildError, EvalOrderError, ModuleGraph, NodeId};

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
    div_ceil(width_bits, WORD_BITS)
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
pub enum Error {
    #[error("top module `{top}` not found in design `{design}`")]
    MissingTop { design: String, top: String },
    #[error(transparent)]
    ModuleGraph(#[from] GraphBuildError),
    #[error(transparent)]
    EvalOrder(#[from] EvalOrderError),
    #[error(transparent)]
    Packed(#[from] PackedError),
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
    regs_cur: Packed,
    regs_next: Packed,
    reg_indices: HashMap<String, PackedIndex>,
    inputs: Packed,
    input_ports: HashMap<String, PackedIndex>,
    outputs: Packed,
    output_ports: HashMap<String, PackedIndex>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_lane_and_word_indexing() {
        let mut packed = Packed::new(128);
        assert_eq!(packed.words_per_lane(), 2);

        let first = packed.allocate(96);
        assert_eq!(first.offset(), 0);
        assert_eq!(first.lanes(), 2);

        let second = packed.allocate(32);
        assert_eq!(second.lanes(), 1);
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
        assert_eq!(packed.slice(first), &[0, 1, 0x0001_0000, 0x0001_0001]);
        assert_eq!(packed.slice(second), &[0xABCD_0000, 0xABCD_0001]);

        let words_per_lane = packed.words_per_lane();
        assert_eq!(first.lane_offset(1, words_per_lane), words_per_lane);
        assert_eq!(first.word_offset(1, 1, words_per_lane), words_per_lane + 1);
    }

    #[test]
    fn packed_handles_width_not_multiple_of_word() {
        let mut packed = Packed::new(70);
        assert_eq!(packed.words_per_lane(), 2);

        let index = packed.allocate(65);
        assert_eq!(index.lanes(), 2);
        assert_eq!(packed.slice(index).len(), 4);

        packed.lane_mut(index, 1)[1] = 0xFFFF_FFFF_FFFFu64;

        assert_eq!(packed.lane(index, 1)[1], 0xFFFF_FFFF_FFFFu64);
        assert_eq!(packed.slice(index)[3], 0xFFFF_FFFF_FFFFu64);
        assert!(packed.slice(index)[0..3].iter().all(|&word| word == 0));
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
        for (name, net) in &module.nets {
            let index = nets.allocate(net.bits as usize);
            net_indices.insert(name.clone(), index);
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
            regs_cur,
            regs_next,
            reg_indices,
            inputs,
            input_ports,
            outputs,
            output_ports,
        })
    }

    pub fn set_inputs(&mut self, port_values: &Packed) -> Result<(), Error> {
        self.inputs.copy_from(port_values)?;
        Ok(())
    }

    pub fn comb_eval(&mut self) -> Result<(), Error> {
        // Implementation to be added in a future revision.
        Ok(())
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
