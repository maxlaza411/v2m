use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::{
    BuildError, EvalOrderError, Literal, ModuleGraph, NetId, NodeId, ParamMap, StrashKind,
    StructuralHasher,
};
use v2m_formats::nir::{Module, Nir, Node, NodeOp, PortDirection};
use v2m_formats::{resolve_bitref, BitRef, ResolvedBit, ResolvedNetBit};

#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    #[error("failed to build module graph: {0}")]
    BuildGraph(#[from] BuildError),
    #[error("failed to compute evaluation order: {0}")]
    EvalOrder(#[from] EvalOrderError),
    #[error("node `{node}` pin `{pin}`: {source}")]
    PinResolve {
        node: String,
        pin: String,
        #[source]
        source: v2m_formats::Error,
    },
    #[error("net `{net}` is not defined in module")]
    UnknownNet { net: String },
    #[error("net `{net}` bit {bit} has not been assigned a driver")]
    UnresolvedNetBit { net: String, bit: u32 },
    #[error("net `{net}` bit {bit} already has a driver")]
    DuplicateNetBit { net: String, bit: u32 },
    #[error("node `{node}` is missing pin `{pin}`")]
    MissingPin { node: String, pin: String },
    #[error("node `{node}` output is not connected to any net")]
    OutputWithoutNet { node: String },
    #[error("node `{node}` output width mismatch: expected {expected} bits, got {actual}")]
    OutputWidthMismatch {
        node: String,
        expected: usize,
        actual: usize,
    },
    #[error("node `{node}` output cannot drive a constant value")]
    OutputToConstant { node: String },
    #[error("node `{node}` has unsupported operation `{op:?}`")]
    UnsupportedOp { node: String, op: NodeOp },
    #[error("node `{node}` mux select width {width} is not supported")]
    UnsupportedMuxWidth { node: String, width: usize },
    #[error(
        "node `{node}` operand width mismatch for `{op:?}`: left is {lhs} bits, right is {rhs} bits"
    )]
    WidthMismatch {
        node: String,
        op: NodeOp,
        lhs: usize,
        rhs: usize,
    },
    #[error(
        "sequential element `{node}` width mismatch between Q ({expected} bits) and D ({actual} bits)"
    )]
    StateWidthMismatch {
        node: String,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Serialize)]
pub struct NormalizedNir {
    pub design: String,
    pub top: String,
    pub modules: BTreeMap<String, NormalizedModule>,
}

#[derive(Debug, Serialize)]
pub struct NormalizedModule {
    pub inputs: BTreeMap<String, Vec<NormalizedLiteral>>,
    pub outputs: BTreeMap<String, Vec<NormalizedLiteral>>,
    pub states: Vec<StateSnapshot>,
    pub nodes: Vec<NormalizedNode>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct NormalizedLiteral {
    pub node: usize,
    #[serde(default, skip_serializing_if = "skip_false")]
    pub inverted: bool,
}

#[derive(Debug, Serialize)]
pub struct NormalizedNode {
    pub width: u32,
    pub id: usize,
    pub kind: NormalizedNodeKind,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalizedNodeKind {
    Input,
    Const {
        bits: String,
    },
    Op {
        op: NodeOp,
        inputs: Vec<NormalizedLiteral>,
        #[serde(skip_serializing_if = "Option::is_none")]
        params: Option<ParamMap>,
    },
}

#[derive(Debug, Serialize)]
pub struct StateSnapshot {
    pub name: String,
    pub bits: Vec<StateBitSnapshot>,
}

#[derive(Debug, Serialize)]
pub struct StateBitSnapshot {
    pub net: String,
    pub bit: u32,
    pub q: NormalizedLiteral,
    pub d: NormalizedLiteral,
}

pub fn normalize_nir(nir: &Nir) -> Result<NormalizedNir, NormalizeError> {
    let mut modules = BTreeMap::new();
    for (name, module) in &nir.modules {
        let normalized = normalize_module(module)?;
        modules.insert(name.clone(), normalized);
    }

    Ok(NormalizedNir {
        design: nir.design.clone(),
        top: nir.top.clone(),
        modules,
    })
}

pub fn normalize_module(module: &Module) -> Result<NormalizedModule, NormalizeError> {
    Normalizer::new(module)?.run()
}

struct NetState {
    bits: Vec<Option<Literal>>,
}

impl NetState {
    fn new(width: u32) -> Self {
        Self {
            bits: vec![None; width as usize],
        }
    }
}

struct RegisterInfo {
    node_id: NodeId,
    q_refs: Vec<(NetId, u32)>,
    q_literals: Vec<Literal>,
    d_pin: BitRef,
    d_literals: Vec<Literal>,
}

struct NormalizerNode<'a> {
    name: &'a str,
    node: &'a Node,
}

struct Normalizer<'a> {
    module: &'a Module,
    graph: ModuleGraph,
    nodes: Vec<NormalizerNode<'a>>,
    order: Vec<NodeId>,
    net_indices: HashMap<&'a str, NetId>,
    hasher: StructuralHasher,
    nets: Vec<NetState>,
    inputs: BTreeMap<String, Vec<Literal>>,
    registers: Vec<RegisterInfo>,
}

impl<'a> Normalizer<'a> {
    fn new(module: &'a Module) -> Result<Self, NormalizeError> {
        let graph = ModuleGraph::from_module(module)?;
        let order = graph.combinational_topological_order()?;

        let mut net_indices = HashMap::with_capacity(module.nets.len());
        let nets: Vec<NetState> = module
            .nets
            .iter()
            .enumerate()
            .map(|(index, (name, net))| {
                let id = NetId(index);
                net_indices.insert(name.as_str(), id);
                NetState::new(net.bits)
            })
            .collect();

        let nodes = module
            .nodes
            .iter()
            .map(|(name, node)| NormalizerNode {
                name: name.as_str(),
                node,
            })
            .collect();

        Ok(Self {
            module,
            graph,
            nodes,
            order,
            net_indices,
            hasher: StructuralHasher::new(),
            nets,
            inputs: BTreeMap::new(),
            registers: Vec::new(),
        })
    }

    fn run(mut self) -> Result<NormalizedModule, NormalizeError> {
        self.initialize_primary_inputs()?;
        self.initialize_registers()?;
        self.evaluate_combinational_nodes()?;
        self.resolve_register_inputs()?;
        let outputs = self.collect_outputs()?;
        self.ensure_all_bits_resolved()?;

        let inputs = self
            .inputs
            .iter()
            .map(|(name, bits)| {
                (
                    name.clone(),
                    bits.iter().copied().map(NormalizedLiteral::from).collect(),
                )
            })
            .collect();

        let mut states = Vec::with_capacity(self.registers.len());
        for register in &self.registers {
            if register.q_literals.len() != register.d_literals.len() {
                return Err(NormalizeError::StateWidthMismatch {
                    node: self.node_name(register.node_id).to_string(),
                    expected: register.q_literals.len(),
                    actual: register.d_literals.len(),
                });
            }

            let mut bits = Vec::with_capacity(register.q_literals.len());
            for (index, &(net_id, bit)) in register.q_refs.iter().enumerate() {
                bits.push(StateBitSnapshot {
                    net: self.graph.net(net_id).name().to_string(),
                    bit,
                    q: register.q_literals[index].into(),
                    d: register.d_literals[index].into(),
                });
            }

            states.push(StateSnapshot {
                name: self.node_name(register.node_id).to_string(),
                bits,
            });
        }
        states.sort_by(|a, b| a.name.cmp(&b.name));

        let nodes = self
            .hasher
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| NormalizedNode {
                width: node.width(),
                id: index,
                kind: match node.kind() {
                    StrashKind::Input => NormalizedNodeKind::Input,
                    StrashKind::Const { bits } => NormalizedNodeKind::Const {
                        bits: bits_to_string(bits),
                    },
                    StrashKind::Op { op, inputs, params } => NormalizedNodeKind::Op {
                        op: op.clone(),
                        inputs: inputs
                            .iter()
                            .copied()
                            .map(NormalizedLiteral::from)
                            .collect(),
                        params: params.clone(),
                    },
                },
            })
            .collect();

        Ok(NormalizedModule {
            inputs,
            outputs: outputs
                .into_iter()
                .map(|(name, bits)| {
                    (
                        name,
                        bits.into_iter().map(NormalizedLiteral::from).collect(),
                    )
                })
                .collect(),
            states,
            nodes,
        })
    }

    fn initialize_primary_inputs(&mut self) -> Result<(), NormalizeError> {
        for (name, port) in &self.module.ports {
            if matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
                let net_id = self.lookup_net_id(name.as_str())?;
                let mut literals = Vec::with_capacity(port.bits as usize);
                for bit in 0..port.bits {
                    let literal = self.hasher.input(1);
                    self.set_net_bit(net_id, bit, literal)?;
                    literals.push(literal);
                }
                self.inputs.insert(name.clone(), literals);
            }
        }
        Ok(())
    }

    fn initialize_registers(&mut self) -> Result<(), NormalizeError> {
        for index in 0..self.nodes.len() {
            if !matches!(self.nodes[index].node.op, NodeOp::Dff | NodeOp::Latch) {
                continue;
            }

            let node_id = NodeId(index);
            let (node_name, q_bitref, d_pin) = {
                let entry = &self.nodes[index];
                let q = entry
                    .node
                    .pin_map
                    .get("Q")
                    .ok_or_else(|| NormalizeError::MissingPin {
                        node: entry.name.to_string(),
                        pin: "Q".to_string(),
                    })?
                    .clone();
                let d = entry
                    .node
                    .pin_map
                    .get("D")
                    .ok_or_else(|| NormalizeError::MissingPin {
                        node: entry.name.to_string(),
                        pin: "D".to_string(),
                    })?
                    .clone();
                (entry.name.to_string(), q, d)
            };

            let resolved = resolve_bitref(self.module, &q_bitref).map_err(|source| {
                NormalizeError::PinResolve {
                    node: node_name.clone(),
                    pin: "Q".to_string(),
                    source,
                }
            })?;

            let mut q_refs = Vec::with_capacity(resolved.len());
            let mut q_literals = Vec::with_capacity(resolved.len());
            for bit in resolved {
                match bit {
                    ResolvedBit::Net(ResolvedNetBit { net, bit }) => {
                        let net_id = self.lookup_net_id(net.as_str())?;
                        let literal = self.hasher.input(1);
                        self.set_net_bit(net_id, bit, literal)?;
                        q_refs.push((net_id, bit));
                        q_literals.push(literal);
                    }
                    ResolvedBit::Const(_) => {
                        return Err(NormalizeError::OutputToConstant {
                            node: node_name.clone(),
                        });
                    }
                }
            }

            self.registers.push(RegisterInfo {
                node_id,
                q_refs,
                q_literals,
                d_pin,
                d_literals: Vec::new(),
            });
        }
        Ok(())
    }

    fn evaluate_combinational_nodes(&mut self) -> Result<(), NormalizeError> {
        let order = self.order.clone();
        for node_id in order {
            let op = self.nodes[node_id.index()].node.op.clone();
            if matches!(op, NodeOp::Dff | NodeOp::Latch) {
                continue;
            }

            let outputs = match op {
                NodeOp::And => self.compute_bitwise_op(node_id, NodeOp::And)?,
                NodeOp::Or => self.compute_bitwise_op(node_id, NodeOp::Or)?,
                NodeOp::Xor => self.compute_bitwise_op(node_id, NodeOp::Xor)?,
                NodeOp::Xnor => self.compute_xnor(node_id)?,
                NodeOp::Not => self.compute_not(node_id)?,
                NodeOp::Mux => self.compute_mux(node_id)?,
                NodeOp::Add => self.compute_add(node_id)?,
                NodeOp::Sub => self.compute_sub(node_id)?,
                NodeOp::Slice => self.compute_slice(node_id)?,
                NodeOp::Cat => {
                    return Err(NormalizeError::UnsupportedOp {
                        node: self.node_name(node_id).to_string(),
                        op: NodeOp::Cat,
                    });
                }
                NodeOp::Const => {
                    return Err(NormalizeError::UnsupportedOp {
                        node: self.node_name(node_id).to_string(),
                        op: NodeOp::Const,
                    });
                }
                NodeOp::Dff | NodeOp::Latch => unreachable!(),
            };

            self.assign_node_outputs(node_id, outputs)?;
        }
        Ok(())
    }

    fn resolve_register_inputs(&mut self) -> Result<(), NormalizeError> {
        for idx in 0..self.registers.len() {
            let (node_id, d_pin, q_len) = {
                let register = &self.registers[idx];
                (
                    register.node_id,
                    register.d_pin.clone(),
                    register.q_literals.len(),
                )
            };

            let bits = self.resolve_bitref_literals(node_id, "D", &d_pin)?;

            if bits.len() != q_len {
                return Err(NormalizeError::StateWidthMismatch {
                    node: self.node_name(node_id).to_string(),
                    expected: q_len,
                    actual: bits.len(),
                });
            }

            self.registers[idx].d_literals = bits.into_iter().map(|(_, literal)| literal).collect();
        }
        Ok(())
    }

    fn collect_outputs(&self) -> Result<BTreeMap<String, Vec<Literal>>, NormalizeError> {
        let mut outputs = BTreeMap::new();
        for (name, port) in &self.module.ports {
            if matches!(port.dir, PortDirection::Output | PortDirection::Inout) {
                let net_id = self.lookup_net_id(name.as_str())?;
                let mut bits = Vec::with_capacity(port.bits as usize);
                for bit in 0..port.bits {
                    bits.push(self.get_net_bit(net_id, bit)?);
                }
                outputs.insert(name.clone(), bits);
            }
        }
        Ok(outputs)
    }

    fn ensure_all_bits_resolved(&self) -> Result<(), NormalizeError> {
        for (index, state) in self.nets.iter().enumerate() {
            let net_id = NetId(index);
            for (bit_index, literal) in state.bits.iter().enumerate() {
                if literal.is_none() {
                    return Err(NormalizeError::UnresolvedNetBit {
                        net: self.net_name(net_id).to_string(),
                        bit: bit_index as u32,
                    });
                }
            }
        }
        Ok(())
    }

    fn compute_bitwise_op(
        &mut self,
        node_id: NodeId,
        op: NodeOp,
    ) -> Result<Vec<Literal>, NormalizeError> {
        let a = self.pin_literals(node_id, "A")?;
        let b = self.pin_literals(node_id, "B")?;
        if a.len() != b.len() {
            return Err(NormalizeError::WidthMismatch {
                node: self.node_name(node_id).to_string(),
                op,
                lhs: a.len(),
                rhs: b.len(),
            });
        }

        Ok(a.into_iter()
            .zip(b)
            .map(|(lhs, rhs)| self.hasher.intern_node(op.clone(), [lhs, rhs], 1, None))
            .collect())
    }

    fn compute_xnor(&mut self, node_id: NodeId) -> Result<Vec<Literal>, NormalizeError> {
        let mut xor = self.compute_bitwise_op(node_id, NodeOp::Xor)?;
        for literal in &mut xor {
            *literal = literal.invert();
        }
        Ok(xor)
    }

    fn compute_not(&mut self, node_id: NodeId) -> Result<Vec<Literal>, NormalizeError> {
        let a = self.pin_literals(node_id, "A")?;
        Ok(a.into_iter()
            .map(|literal| self.hasher.intern_node(NodeOp::Not, [literal], 1, None))
            .collect())
    }

    fn compute_mux(&mut self, node_id: NodeId) -> Result<Vec<Literal>, NormalizeError> {
        let a = self.pin_literals(node_id, "A")?;
        let b = self.pin_literals(node_id, "B")?;
        let sel = self.pin_literals(node_id, "S")?;

        if sel.len() != 1 {
            return Err(NormalizeError::UnsupportedMuxWidth {
                node: self.node_name(node_id).to_string(),
                width: sel.len(),
            });
        }

        if a.len() != b.len() {
            return Err(NormalizeError::WidthMismatch {
                node: self.node_name(node_id).to_string(),
                op: NodeOp::Mux,
                lhs: a.len(),
                rhs: b.len(),
            });
        }

        let select = sel[0];
        let select_inv = select.invert();

        Ok(a.into_iter()
            .zip(b.into_iter())
            .map(|(a_bit, b_bit)| {
                let a_term = self
                    .hasher
                    .intern_node(NodeOp::And, [select_inv, a_bit], 1, None);
                let b_term = self
                    .hasher
                    .intern_node(NodeOp::And, [select, b_bit], 1, None);
                self.hasher
                    .intern_node(NodeOp::Or, [a_term, b_term], 1, None)
            })
            .collect())
    }

    fn compute_add(&mut self, node_id: NodeId) -> Result<Vec<Literal>, NormalizeError> {
        let a = self.pin_literals(node_id, "A")?;
        let b = self.pin_literals(node_id, "B")?;
        if a.len() != b.len() {
            return Err(NormalizeError::WidthMismatch {
                node: self.node_name(node_id).to_string(),
                op: NodeOp::Add,
                lhs: a.len(),
                rhs: b.len(),
            });
        }

        let mut result = Vec::with_capacity(a.len());
        let mut carry = self.hasher.constant_zero(1);
        for (a_bit, b_bit) in a.into_iter().zip(b.into_iter()) {
            let sum_ab = self
                .hasher
                .intern_node(NodeOp::Xor, [a_bit, b_bit], 1, None);
            let sum = self
                .hasher
                .intern_node(NodeOp::Xor, [sum_ab, carry], 1, None);
            let carry_ab = self
                .hasher
                .intern_node(NodeOp::And, [a_bit, b_bit], 1, None);
            let carry_sum = self
                .hasher
                .intern_node(NodeOp::And, [sum_ab, carry], 1, None);
            carry = self
                .hasher
                .intern_node(NodeOp::Or, [carry_ab, carry_sum], 1, None);
            result.push(sum);
        }

        Ok(result)
    }

    fn compute_sub(&mut self, node_id: NodeId) -> Result<Vec<Literal>, NormalizeError> {
        let a = self.pin_literals(node_id, "A")?;
        let b = self.pin_literals(node_id, "B")?;
        if a.len() != b.len() {
            return Err(NormalizeError::WidthMismatch {
                node: self.node_name(node_id).to_string(),
                op: NodeOp::Sub,
                lhs: a.len(),
                rhs: b.len(),
            });
        }

        let mut result = Vec::with_capacity(a.len());
        let mut carry = self.hasher.constant_one(1);
        for (a_bit, b_bit) in a.into_iter().zip(b.into_iter()) {
            let b_inv = b_bit.invert();
            let sum_ab = self
                .hasher
                .intern_node(NodeOp::Xor, [a_bit, b_inv], 1, None);
            let sum = self
                .hasher
                .intern_node(NodeOp::Xor, [sum_ab, carry], 1, None);
            let carry_ab = self
                .hasher
                .intern_node(NodeOp::And, [a_bit, b_inv], 1, None);
            let carry_sum = self
                .hasher
                .intern_node(NodeOp::And, [sum_ab, carry], 1, None);
            carry = self
                .hasher
                .intern_node(NodeOp::Or, [carry_ab, carry_sum], 1, None);
            result.push(sum);
        }

        Ok(result)
    }

    fn compute_slice(&mut self, node_id: NodeId) -> Result<Vec<Literal>, NormalizeError> {
        Ok(self.pin_literals(node_id, "A")?)
    }

    fn assign_node_outputs(
        &mut self,
        node_id: NodeId,
        outputs: Vec<Literal>,
    ) -> Result<(), NormalizeError> {
        let node_name = self.node_name(node_id).to_string();
        let bitref = {
            let entry = &self.nodes[node_id.index()];
            entry
                .node
                .pin_map
                .get("Y")
                .ok_or_else(|| NormalizeError::MissingPin {
                    node: node_name.clone(),
                    pin: "Y".to_string(),
                })?
                .clone()
        };
        let resolved =
            resolve_bitref(self.module, &bitref).map_err(|source| NormalizeError::PinResolve {
                node: node_name.clone(),
                pin: "Y".to_string(),
                source,
            })?;

        if resolved.is_empty() {
            return Err(NormalizeError::OutputWithoutNet {
                node: node_name.clone(),
            });
        }

        if resolved.len() != outputs.len() {
            return Err(NormalizeError::OutputWidthMismatch {
                node: node_name.clone(),
                expected: resolved.len(),
                actual: outputs.len(),
            });
        }

        for (bit, literal) in resolved.into_iter().zip(outputs.into_iter()) {
            match bit {
                ResolvedBit::Net(ResolvedNetBit { net, bit }) => {
                    let net_id = self.lookup_net_id(net.as_str())?;
                    self.set_net_bit(net_id, bit, literal)?;
                }
                ResolvedBit::Const(_) => {
                    return Err(NormalizeError::OutputToConstant {
                        node: node_name.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    fn pin_literals(&mut self, node_id: NodeId, pin: &str) -> Result<Vec<Literal>, NormalizeError> {
        let bitref = {
            let entry = &self.nodes[node_id.index()];
            entry
                .node
                .pin_map
                .get(pin)
                .ok_or_else(|| NormalizeError::MissingPin {
                    node: entry.name.to_string(),
                    pin: pin.to_string(),
                })?
                .clone()
        };
        Ok(self
            .resolve_bitref_literals(node_id, pin, &bitref)?
            .into_iter()
            .map(|(_, literal)| literal)
            .collect())
    }

    fn resolve_bitref_literals(
        &mut self,
        node_id: NodeId,
        pin: &str,
        bitref: &BitRef,
    ) -> Result<Vec<(Option<(NetId, u32)>, Literal)>, NormalizeError> {
        let node_name = self.node_name(node_id).to_string();
        let resolved =
            resolve_bitref(self.module, bitref).map_err(|source| NormalizeError::PinResolve {
                node: node_name.clone(),
                pin: pin.to_string(),
                source,
            })?;

        resolved
            .into_iter()
            .map(|bit| match bit {
                ResolvedBit::Const(value) => Ok((None, self.constant(value))),
                ResolvedBit::Net(ResolvedNetBit { net, bit }) => {
                    let net_id = self.lookup_net_id(net.as_str())?;
                    let literal = self.get_net_bit(net_id, bit)?;
                    Ok((Some((net_id, bit)), literal))
                }
            })
            .collect()
    }

    fn constant(&mut self, value: bool) -> Literal {
        if value {
            self.hasher.constant_one(1)
        } else {
            self.hasher.constant_zero(1)
        }
    }

    fn get_net_bit(&self, net: NetId, bit: u32) -> Result<Literal, NormalizeError> {
        self.nets[net.index()]
            .bits
            .get(bit as usize)
            .and_then(|entry| *entry)
            .ok_or_else(|| self.unresolved_net_bit_error(net, bit))
    }

    fn set_net_bit(
        &mut self,
        net: NetId,
        bit: u32,
        literal: Literal,
    ) -> Result<(), NormalizeError> {
        let net_name = self.net_name(net).to_string();
        let state = &mut self.nets[net.index()];
        let slot = match state.bits.get_mut(bit as usize) {
            Some(slot) => slot,
            None => {
                return Err(NormalizeError::UnresolvedNetBit {
                    net: net_name.clone(),
                    bit,
                })
            }
        };
        if slot.is_some() {
            return Err(NormalizeError::DuplicateNetBit { net: net_name, bit });
        }
        *slot = Some(literal);
        Ok(())
    }

    fn lookup_net_id(&self, name: &str) -> Result<NetId, NormalizeError> {
        self.net_indices
            .get(name)
            .copied()
            .ok_or_else(|| NormalizeError::UnknownNet {
                net: name.to_string(),
            })
    }

    fn node_name(&self, node_id: NodeId) -> &str {
        self.nodes[node_id.index()].name
    }

    fn net_name(&self, net: NetId) -> &str {
        self.graph.net(net).name()
    }

    fn unresolved_net_bit_error(&self, net: NetId, bit: u32) -> NormalizeError {
        NormalizeError::UnresolvedNetBit {
            net: self.net_name(net).to_string(),
            bit,
        }
    }
}

impl From<Literal> for NormalizedLiteral {
    fn from(literal: Literal) -> Self {
        Self {
            node: literal.node().index(),
            inverted: literal.is_inverted(),
        }
    }
}

fn bits_to_string(bits: &[bool]) -> String {
    bits.iter()
        .rev()
        .map(|bit| if *bit { '1' } else { '0' })
        .collect()
}

fn skip_false(value: &bool) -> bool {
    !*value
}
