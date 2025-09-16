use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::fmt;

use v2m_formats::nir::{Module, NodeOp};
use v2m_formats::{resolve_bitref, BitRef, ResolvedBit};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NetId(usize);

impl NetId {
    #[inline]
    fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    #[inline]
    fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("failed to resolve pin `{pin}` on node `{node}`: {source}")]
    PinResolve {
        node: String,
        pin: String,
        #[source]
        source: v2m_formats::Error,
    },
    #[error("net `{net}` referenced by pin `{pin}` on node `{node}` not found")]
    UnknownNet {
        node: String,
        pin: String,
        net: String,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvalOrderError {
    CombinationalLoop { cycle: Vec<String> },
}

impl fmt::Display for EvalOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CombinationalLoop { cycle } => {
                write!(f, "combinational loop detected: {}", cycle.join(" -> "))
            }
        }
    }
}

impl std::error::Error for EvalOrderError {}

#[derive(Debug, Clone)]
pub struct Net {
    name: String,
    width: u32,
    drivers: Vec<NodeId>,
    loads: Vec<NodeId>,
}

impl Net {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn drivers(&self) -> &[NodeId] {
        &self.drivers
    }

    pub fn loads(&self) -> &[NodeId] {
        &self.loads
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    name: String,
    op: NodeOp,
    width: u32,
    inputs: Vec<NetId>,
    outputs: Vec<NetId>,
    fanins: Vec<NodeId>,
    fanouts: Vec<NodeId>,
}

impl Node {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn op(&self) -> &NodeOp {
        &self.op
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn inputs(&self) -> &[NetId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[NetId] {
        &self.outputs
    }

    pub fn fanins(&self) -> &[NodeId] {
        &self.fanins
    }

    pub fn fanouts(&self) -> &[NodeId] {
        &self.fanouts
    }

    fn is_combinational(&self) -> bool {
        !matches!(self.op, NodeOp::Dff | NodeOp::Latch)
    }
}

pub struct ModuleGraph {
    nets: Vec<Net>,
    nodes: Vec<Node>,
    net_lookup: HashMap<String, NetId>,
    node_lookup: HashMap<String, NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleMetrics {
    pub node_count: usize,
    pub net_count: usize,
    pub max_depth: usize,
    pub depth_histogram: Vec<usize>,
}

impl ModuleMetrics {
    pub fn combinational_node_count(&self) -> usize {
        self.depth_histogram.iter().sum()
    }
}

impl ModuleGraph {
    pub fn from_module(module: &Module) -> Result<Self, BuildError> {
        let mut nets = Vec::with_capacity(module.nets.len());
        let mut net_lookup = HashMap::with_capacity(module.nets.len());
        for (index, (name, net)) in module.nets.iter().enumerate() {
            let id = NetId(index);
            net_lookup.insert(name.clone(), id);
            nets.push(Net {
                name: name.clone(),
                width: net.bits,
                drivers: Vec::new(),
                loads: Vec::new(),
            });
        }

        let mut nodes = Vec::with_capacity(module.nodes.len());
        let mut node_lookup = HashMap::with_capacity(module.nodes.len());
        for (index, (name, node)) in module.nodes.iter().enumerate() {
            let id = NodeId(index);
            node_lookup.insert(name.clone(), id);
            nodes.push(Node {
                name: name.clone(),
                op: node.op.clone(),
                width: node.width,
                inputs: Vec::new(),
                outputs: Vec::new(),
                fanins: Vec::new(),
                fanouts: Vec::new(),
            });
        }

        let mut net_driver_sets = vec![BTreeSet::new(); nets.len()];
        let mut net_load_sets = vec![BTreeSet::new(); nets.len()];
        let mut node_input_sets = vec![BTreeSet::new(); nodes.len()];
        let mut node_output_sets = vec![BTreeSet::new(); nodes.len()];

        for (node_name, node) in module.nodes.iter() {
            let &node_id = node_lookup
                .get(node_name.as_str())
                .expect("node must exist");
            for (pin_name, bitref) in node.pin_map.iter() {
                let nets_for_pin =
                    collect_pin_nets(module, bitref, &net_lookup, node_name, pin_name)?;

                if is_output_pin(&node.op, pin_name.as_str()) {
                    for net_id in nets_for_pin {
                        net_driver_sets[net_id.index()].insert(node_id);
                        node_output_sets[node_id.index()].insert(net_id);
                    }
                } else {
                    for net_id in nets_for_pin {
                        net_load_sets[net_id.index()].insert(node_id);
                        node_input_sets[node_id.index()].insert(net_id);
                    }
                }
            }
        }

        let mut fanin_sets = vec![BTreeSet::new(); nodes.len()];
        let mut fanout_sets = vec![BTreeSet::new(); nodes.len()];

        for net_index in 0..nets.len() {
            let drivers = &net_driver_sets[net_index];
            let loads = &net_load_sets[net_index];
            for &driver in drivers {
                if !nodes[driver.index()].is_combinational() {
                    continue;
                }
                for &load in loads {
                    if !nodes[load.index()].is_combinational() || driver == load {
                        continue;
                    }
                    fanout_sets[driver.index()].insert(load);
                    fanin_sets[load.index()].insert(driver);
                }
            }
        }

        for (idx, net) in nets.iter_mut().enumerate() {
            net.drivers = net_driver_sets[idx].iter().copied().collect();
            net.loads = net_load_sets[idx].iter().copied().collect();
        }

        for (idx, node) in nodes.iter_mut().enumerate() {
            node.inputs = node_input_sets[idx].iter().copied().collect();
            node.outputs = node_output_sets[idx].iter().copied().collect();
            node.fanins = fanin_sets[idx].iter().copied().collect();
            node.fanouts = fanout_sets[idx].iter().copied().collect();
        }

        Ok(Self {
            nets,
            nodes,
            net_lookup,
            node_lookup,
        })
    }

    pub fn nets(&self) -> &[Net] {
        &self.nets
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn net(&self, id: NetId) -> &Net {
        &self.nets[id.index()]
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    pub fn net_id(&self, name: &str) -> Option<NetId> {
        self.net_lookup.get(name).copied()
    }

    pub fn node_id(&self, name: &str) -> Option<NodeId> {
        self.node_lookup.get(name).copied()
    }

    pub fn combinational_topological_order(&self) -> Result<Vec<NodeId>, EvalOrderError> {
        let mut indegree = vec![0usize; self.nodes.len()];
        let mut ready = BinaryHeap::new();
        let mut order = Vec::new();
        let mut processed = vec![false; self.nodes.len()];
        let mut combinational_count = 0usize;

        for (index, node) in self.nodes.iter().enumerate() {
            if !node.is_combinational() {
                continue;
            }

            combinational_count += 1;
            indegree[index] = node.fanins.len();
            if indegree[index] == 0 {
                ready.push(Reverse(NodeId(index)));
            }
        }

        while let Some(Reverse(node_id)) = ready.pop() {
            order.push(node_id);
            processed[node_id.index()] = true;

            for &succ in self.nodes[node_id.index()].fanouts() {
                indegree[succ.index()] -= 1;
                if indegree[succ.index()] == 0 {
                    ready.push(Reverse(succ));
                }
            }
        }

        if order.len() != combinational_count {
            let mut candidates = vec![false; self.nodes.len()];
            for (index, node) in self.nodes.iter().enumerate() {
                if node.is_combinational() && !processed[index] {
                    candidates[index] = true;
                }
            }

            let cycle_nodes = self
                .find_cycle_in_candidates(&candidates)
                .unwrap_or_else(|| {
                    self.nodes
                        .iter()
                        .enumerate()
                        .filter_map(|(index, _)| {
                            if candidates[index] {
                                Some(NodeId(index))
                            } else {
                                None
                            }
                        })
                        .collect()
                });

            let cycle = cycle_nodes
                .into_iter()
                .map(|id| self.node(id).name().to_string())
                .collect();

            return Err(EvalOrderError::CombinationalLoop { cycle });
        }

        Ok(order)
    }

    pub fn compute_depth(&self, order: &[NodeId]) -> Vec<(NodeId, usize)> {
        let mut depth_by_node = vec![0usize; self.nodes.len()];
        let mut result = Vec::with_capacity(order.len());

        for &node_id in order {
            debug_assert!(self.nodes[node_id.index()].is_combinational());
            let level = self.nodes[node_id.index()]
                .fanins()
                .iter()
                .map(|fanin| depth_by_node[fanin.index()] + 1)
                .max()
                .unwrap_or(0);
            depth_by_node[node_id.index()] = level;
            result.push((node_id, level));
        }

        result
    }

    pub fn metrics(&self) -> Result<ModuleMetrics, EvalOrderError> {
        let order = self.combinational_topological_order()?;
        let depths = self.compute_depth(&order);

        let mut histogram = Vec::new();
        for &(_, level) in &depths {
            if histogram.len() <= level {
                histogram.resize(level + 1, 0);
            }
            histogram[level] += 1;
        }

        let max_depth = depths.iter().map(|&(_, level)| level).max().unwrap_or(0);

        Ok(ModuleMetrics {
            node_count: self.nodes.len(),
            net_count: self.nets.len(),
            max_depth,
            depth_histogram: histogram,
        })
    }

    fn find_cycle_in_candidates(&self, candidates: &[bool]) -> Option<Vec<NodeId>> {
        let mut visited = vec![false; self.nodes.len()];
        let mut on_stack = vec![false; self.nodes.len()];
        let mut stack = Vec::new();

        for (index, &candidate) in candidates.iter().enumerate() {
            if !candidate || visited[index] {
                continue;
            }

            if let Some(cycle) = self.find_cycle_dfs(
                NodeId(index),
                candidates,
                &mut visited,
                &mut on_stack,
                &mut stack,
            ) {
                return Some(cycle);
            }
        }

        None
    }

    fn find_cycle_dfs(
        &self,
        node_id: NodeId,
        candidates: &[bool],
        visited: &mut [bool],
        on_stack: &mut [bool],
        stack: &mut Vec<NodeId>,
    ) -> Option<Vec<NodeId>> {
        visited[node_id.index()] = true;
        on_stack[node_id.index()] = true;
        stack.push(node_id);

        for &succ in self.nodes[node_id.index()].fanouts() {
            if !candidates[succ.index()] {
                continue;
            }

            if !visited[succ.index()] {
                if let Some(cycle) = self.find_cycle_dfs(succ, candidates, visited, on_stack, stack)
                {
                    return Some(cycle);
                }
            } else if on_stack[succ.index()] {
                let start = stack
                    .iter()
                    .rposition(|&id| id == succ)
                    .expect("cycle node must be in stack");
                let mut cycle = stack[start..].to_vec();
                cycle.push(succ);
                return Some(cycle);
            }
        }

        stack.pop();
        on_stack[node_id.index()] = false;
        None
    }
}

fn collect_pin_nets(
    module: &Module,
    bitref: &BitRef,
    net_lookup: &HashMap<String, NetId>,
    node_name: &str,
    pin_name: &str,
) -> Result<Vec<NetId>, BuildError> {
    let resolved = resolve_bitref(module, bitref).map_err(|source| BuildError::PinResolve {
        node: node_name.to_string(),
        pin: pin_name.to_string(),
        source,
    })?;

    let mut nets = BTreeSet::new();
    for resolved_bit in resolved {
        if let ResolvedBit::Net(bit) = resolved_bit {
            let net_name = bit.net;
            let net_id = match net_lookup.get(net_name.as_str()) {
                Some(id) => *id,
                None => {
                    return Err(BuildError::UnknownNet {
                        node: node_name.to_string(),
                        pin: pin_name.to_string(),
                        net: net_name,
                    })
                }
            };
            nets.insert(net_id);
        }
    }

    Ok(nets.into_iter().collect())
}

fn is_output_pin(op: &NodeOp, pin_name: &str) -> bool {
    match op {
        NodeOp::Dff | NodeOp::Latch => pin_name == "Q",
        _ => pin_name == "Y",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use std::path::PathBuf;
    use v2m_formats::nir::{
        BitRefConcat, BitRefNet, Module as NirModule, Net as NirNet, Node as NirNode,
    };

    fn module_graph_from_example(name: &str) -> ModuleGraph {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/nir")
            .join(format!("{name}.json"));
        let nir = v2m_formats::load_nir(&path).expect("load nir example");
        let module = nir.modules.get(nir.top.as_str()).expect("top module");
        ModuleGraph::from_module(module).expect("build graph")
    }

    fn net_bit(net: &str, lsb: u32, msb: u32) -> BitRef {
        BitRef::Net(BitRefNet {
            net: net.to_string(),
            lsb,
            msb,
        })
    }

    fn ripple_xor_graph(stages: usize) -> ModuleGraph {
        assert!(stages > 0);

        let mut nets = BTreeMap::new();
        for index in 0..=stages {
            nets.insert(
                format!("in_{index}"),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
        }

        if stages > 1 {
            for index in 0..(stages - 1) {
                nets.insert(
                    format!("stage_{index}"),
                    NirNet {
                        bits: 1,
                        attrs: None,
                    },
                );
            }
        }

        nets.insert(
            "out".to_string(),
            NirNet {
                bits: 1,
                attrs: None,
            },
        );

        let mut nodes = BTreeMap::new();
        let mut prev_output: Option<String> = None;

        for stage in 0..stages {
            let node_name = format!("xor_{stage}");
            let input_a = if stage == 0 {
                format!("in_{stage}")
            } else {
                prev_output.as_ref().expect("previous stage output").clone()
            };
            let input_b = format!("in_{}", stage + 1);
            let output = if stage + 1 == stages {
                "out".to_string()
            } else {
                format!("stage_{stage}")
            };

            let node = NirNode {
                uid: node_name.clone(),
                op: NodeOp::Xor,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit(&input_a, 0, 0)),
                    ("B".to_string(), net_bit(&input_b, 0, 0)),
                    ("Y".to_string(), net_bit(&output, 0, 0)),
                ]),
                params: None,
                attrs: None,
            };
            nodes.insert(node_name, node);
            prev_output = Some(output);
        }

        let module = NirModule {
            ports: BTreeMap::new(),
            nets,
            nodes,
        };

        ModuleGraph::from_module(&module).expect("build graph")
    }

    fn tree_xor_graph(leaves: usize) -> ModuleGraph {
        assert!(leaves.is_power_of_two());
        assert!(leaves >= 2);

        let mut nets = BTreeMap::new();
        for index in 0..leaves {
            nets.insert(
                format!("in_{index}"),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
        }

        let mut nodes = BTreeMap::new();
        let mut level_inputs: Vec<String> =
            (0..leaves).map(|index| format!("in_{index}")).collect();
        let mut level = 0usize;

        while level_inputs.len() > 1 {
            let mut next_level = Vec::with_capacity(level_inputs.len() / 2);
            for pair in 0..(level_inputs.len() / 2) {
                let a_name = level_inputs[2 * pair].clone();
                let b_name = level_inputs[2 * pair + 1].clone();
                let output_name = if level_inputs.len() == 2 {
                    "out".to_string()
                } else {
                    format!("level{level}_{pair}")
                };

                nets.insert(
                    output_name.clone(),
                    NirNet {
                        bits: 1,
                        attrs: None,
                    },
                );

                let node_name = format!("xor_l{level}_{pair}");
                let node = NirNode {
                    uid: node_name.clone(),
                    op: NodeOp::Xor,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit(&a_name, 0, 0)),
                        ("B".to_string(), net_bit(&b_name, 0, 0)),
                        ("Y".to_string(), net_bit(&output_name, 0, 0)),
                    ]),
                    params: None,
                    attrs: None,
                };
                nodes.insert(node_name, node);
                next_level.push(output_name);
            }

            level_inputs = next_level;
            level += 1;
        }

        let module = NirModule {
            ports: BTreeMap::new(),
            nets,
            nodes,
        };

        ModuleGraph::from_module(&module).expect("build graph")
    }

    #[test]
    fn full_adder_adjacency() {
        let graph = module_graph_from_example("full_adder");

        let xor_ab = graph.node(graph.node_id("xor_ab").unwrap());
        let and_cin = graph.node(graph.node_id("and_cin").unwrap());
        let or_carry = graph.node(graph.node_id("or_carry").unwrap());

        assert_eq!(
            xor_ab.fanouts(),
            &[
                graph.node_id("and_cin").unwrap(),
                graph.node_id("xor_sum").unwrap()
            ]
        );
        assert!(xor_ab.fanins().is_empty());

        assert_eq!(and_cin.fanins(), &[graph.node_id("xor_ab").unwrap()]);
        assert_eq!(and_cin.fanouts(), &[graph.node_id("or_carry").unwrap()]);

        let and_ab = graph.node(graph.node_id("and_ab").unwrap());
        assert!(and_ab.fanins().is_empty());
        assert_eq!(and_ab.fanouts(), &[graph.node_id("or_carry").unwrap()]);

        assert_eq!(
            or_carry.fanins(),
            &[
                graph.node_id("and_ab").unwrap(),
                graph.node_id("and_cin").unwrap()
            ]
        );
        assert!(or_carry.fanouts().is_empty());

        let sum_ab = graph.net(graph.net_id("sum_ab").unwrap());
        assert_eq!(sum_ab.drivers(), &[graph.node_id("xor_ab").unwrap()]);
        assert_eq!(
            sum_ab.loads(),
            &[
                graph.node_id("and_cin").unwrap(),
                graph.node_id("xor_sum").unwrap()
            ]
        );
    }

    #[test]
    fn mux_graph_connectivity() {
        let mut nets = BTreeMap::new();
        for name in ["a", "b", "sel", "sel_n", "y"] {
            nets.insert(
                name.to_string(),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
        }

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "inv_sel".to_string(),
            NirNode {
                uid: "inv_sel".to_string(),
                op: NodeOp::Not,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("sel", 0, 0)),
                    ("Y".to_string(), net_bit("sel_n", 0, 0)),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            "mux".to_string(),
            NirNode {
                uid: "mux".to_string(),
                op: NodeOp::Mux,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("a", 0, 0)),
                    ("B".to_string(), net_bit("b", 0, 0)),
                    (
                        "S".to_string(),
                        BitRef::Concat(BitRefConcat {
                            concat: vec![net_bit("sel_n", 0, 0)],
                        }),
                    ),
                    ("Y".to_string(), net_bit("y", 0, 0)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule {
            ports: BTreeMap::new(),
            nets,
            nodes,
        };

        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let inv_sel_id = graph.node_id("inv_sel").unwrap();
        let mux_id = graph.node_id("mux").unwrap();

        assert_eq!(graph.node(inv_sel_id).fanouts(), &[mux_id]);
        assert!(graph.node(inv_sel_id).fanins().is_empty());
        assert_eq!(graph.node(mux_id).fanins(), &[inv_sel_id]);

        let sel_n = graph.net(graph.net_id("sel_n").unwrap());
        assert_eq!(sel_n.drivers(), &[inv_sel_id]);
        assert_eq!(sel_n.loads(), &[mux_id]);
    }

    #[test]
    fn combinational_topological_order_full_adder() {
        let graph = module_graph_from_example("full_adder");

        let order = graph
            .combinational_topological_order()
            .expect("topological order");

        let mut positions = HashMap::new();
        for (index, node_id) in order.iter().enumerate() {
            positions.insert(*node_id, index);
        }

        for (index, node_id) in order.iter().enumerate() {
            for &fanin in graph.node(*node_id).fanins() {
                let fanin_pos = *positions.get(&fanin).expect("fanin in order");
                assert!(
                    fanin_pos < index,
                    "node {} must come after fanin {}",
                    graph.node(*node_id).name(),
                    graph.node(fanin).name(),
                );
            }
        }

        let combinational_nodes = graph
            .nodes()
            .iter()
            .filter(|node| !matches!(node.op(), NodeOp::Dff | NodeOp::Latch))
            .count();
        assert_eq!(order.len(), combinational_nodes);
    }

    #[test]
    fn combinational_loop_detection() {
        let mut nets = BTreeMap::new();
        for name in ["a", "b"] {
            nets.insert(
                name.to_string(),
                NirNet {
                    bits: 1,
                    attrs: None,
                },
            );
        }

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "not_a".to_string(),
            NirNode {
                uid: "not_a".to_string(),
                op: NodeOp::Not,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("b", 0, 0)),
                    ("Y".to_string(), net_bit("a", 0, 0)),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            "not_b".to_string(),
            NirNode {
                uid: "not_b".to_string(),
                op: NodeOp::Not,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("a", 0, 0)),
                    ("Y".to_string(), net_bit("b", 0, 0)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule {
            ports: BTreeMap::new(),
            nets,
            nodes,
        };

        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let err = graph
            .combinational_topological_order()
            .expect_err("must detect loop");

        let cycle = match &err {
            EvalOrderError::CombinationalLoop { cycle } => cycle,
        };
        assert_eq!(
            cycle,
            &vec![
                "not_a".to_string(),
                "not_b".to_string(),
                "not_a".to_string()
            ]
        );

        assert_eq!(
            err.to_string(),
            "combinational loop detected: not_a -> not_b -> not_a"
        );
    }

    #[test]
    fn register_cuts_combinational_paths() {
        let mut nets = BTreeMap::new();
        for (name, bits) in [("clk", 1), ("rst", 1), ("notq", 1), ("q", 1)] {
            nets.insert(name.to_string(), NirNet { bits, attrs: None });
        }

        let mut nodes = BTreeMap::new();
        nodes.insert(
            "inv".to_string(),
            NirNode {
                uid: "inv".to_string(),
                op: NodeOp::Not,
                width: 1,
                pin_map: BTreeMap::from([
                    ("A".to_string(), net_bit("q", 0, 0)),
                    ("Y".to_string(), net_bit("notq", 0, 0)),
                ]),
                params: None,
                attrs: None,
            },
        );
        nodes.insert(
            "reg".to_string(),
            NirNode {
                uid: "reg".to_string(),
                op: NodeOp::Dff,
                width: 1,
                pin_map: BTreeMap::from([
                    ("D".to_string(), net_bit("notq", 0, 0)),
                    ("Q".to_string(), net_bit("q", 0, 0)),
                    ("CLK".to_string(), net_bit("clk", 0, 0)),
                    ("RST".to_string(), net_bit("rst", 0, 0)),
                ]),
                params: None,
                attrs: None,
            },
        );

        let module = NirModule {
            ports: BTreeMap::new(),
            nets,
            nodes,
        };

        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let inv_id = graph.node_id("inv").unwrap();
        let reg_id = graph.node_id("reg").unwrap();

        assert!(graph.node(inv_id).fanins().is_empty());
        assert!(graph.node(inv_id).fanouts().is_empty());
        assert!(graph.node(reg_id).fanins().is_empty());
        assert!(graph.node(reg_id).fanouts().is_empty());

        let notq = graph.net(graph.net_id("notq").unwrap());
        assert_eq!(notq.drivers(), &[inv_id]);
        assert_eq!(notq.loads(), &[reg_id]);

        let q = graph.net(graph.net_id("q").unwrap());
        assert_eq!(q.drivers(), &[reg_id]);
        assert_eq!(q.loads(), &[inv_id]);
    }

    #[test]
    fn ripple_xor_depth_increases_linearly() {
        let stages = 7;
        let graph = ripple_xor_graph(stages);
        let order = graph
            .combinational_topological_order()
            .expect("topological order");
        let depths = graph.compute_depth(&order);

        let mut depth_by_name = HashMap::new();
        for (node_id, level) in depths {
            depth_by_name.insert(graph.node(node_id).name().to_string(), level);
        }

        for stage in 0..stages {
            let name = format!("xor_{stage}");
            assert_eq!(
                depth_by_name.get(name.as_str()),
                Some(&stage),
                "expected {name} to have depth {stage}"
            );
        }

        let metrics = graph.metrics().expect("metrics");
        assert_eq!(metrics.node_count, graph.nodes().len());
        assert_eq!(metrics.net_count, graph.nets().len());
        assert_eq!(metrics.max_depth, stages - 1);
        assert_eq!(metrics.combinational_node_count(), stages);
        assert_eq!(metrics.depth_histogram.len(), stages);
        assert!(metrics.depth_histogram.iter().all(|&count| count == 1));
    }

    #[test]
    fn tree_xor_has_log_depth() {
        let leaves = 8;
        let graph = tree_xor_graph(leaves);
        let order = graph
            .combinational_topological_order()
            .expect("topological order");
        let depths = graph.compute_depth(&order);

        let mut depth_by_name = HashMap::new();
        for (node_id, level) in depths {
            depth_by_name.insert(graph.node(node_id).name().to_string(), level);
        }

        assert_eq!(depth_by_name.get("xor_l0_0"), Some(&0));
        assert_eq!(depth_by_name.get("xor_l1_0"), Some(&1));
        assert_eq!(depth_by_name.get("xor_l2_0"), Some(&2));

        let metrics = graph.metrics().expect("metrics");
        assert_eq!(metrics.node_count, graph.nodes().len());
        assert_eq!(metrics.net_count, graph.nets().len());
        assert_eq!(metrics.max_depth, 2);
        assert_eq!(metrics.combinational_node_count(), 7);
        assert_eq!(metrics.depth_histogram, vec![4, 2, 1]);
    }
}
