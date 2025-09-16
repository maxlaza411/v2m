use std::collections::{BTreeMap, HashMap};

use thiserror::Error;
use v2m_formats::nir::{Module, Nir, NodeOp, PortDirection};
use v2m_nir::{BuildError as GraphBuildError, EvalOrderError, ModuleGraph, NodeId};

mod pin_binding;

use pin_binding::{bind_bitref, BitBinding, ConstPool, PinBindingError};

const WORD_BITS: usize = 64;

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
}

impl Packed {
    pub fn new(num_vectors: usize) -> Self {
        let words_per_lane = (num_vectors + (WORD_BITS - 1)) / WORD_BITS;
        Self {
            num_vectors,
            words_per_lane,
            storage: Vec::new(),
        }
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn words_per_lane(&self) -> usize {
        self.words_per_lane
    }

    pub fn total_lanes(&self) -> usize {
        if self.words_per_lane == 0 {
            0
        } else {
            self.storage.len() / self.words_per_lane
        }
    }

    pub fn allocate(&mut self, width_bits: usize) -> PackedIndex {
        let lanes = (width_bits + (WORD_BITS - 1)) / WORD_BITS;
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

    pub fn slice(&self, index: PackedIndex) -> &[u64] {
        let end = index.offset + index.lanes * self.words_per_lane;
        &self.storage[index.offset..end]
    }

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
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }
}

impl PackedBitMask {
    pub fn new(num_vectors: usize) -> Self {
        let word_count = (num_vectors + (WORD_BITS - 1)) / WORD_BITS;
        Self {
            words: vec![0; word_count],
        }
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

#[derive(Default)]
struct NodePinBindings {
    pins: BTreeMap<String, BitBinding>,
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
        let topo = graph.combinational_topological_order()?;

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

        Ok(Self {
            nir,
            module,
            graph,
            topo,
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
