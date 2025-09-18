use std::collections::{BTreeMap, HashMap};
use std::mem::Discriminant;

use serde_json::Value;
use v2m_formats::nir::NodeOp;

pub type ParamMap = BTreeMap<String, Value>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StrashNodeId(usize);

impl StrashNodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Literal {
    node: StrashNodeId,
    inverted: bool,
}

impl Literal {
    #[inline]
    pub fn node(self) -> StrashNodeId {
        self.node
    }

    #[inline]
    pub fn is_inverted(self) -> bool {
        self.inverted
    }

    #[inline]
    pub fn invert(self) -> Self {
        Self {
            node: self.node,
            inverted: !self.inverted,
        }
    }
}

impl From<StrashNodeId> for Literal {
    fn from(node: StrashNodeId) -> Self {
        Self {
            node,
            inverted: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StrashNode {
    width: u32,
    kind: StrashKind,
}

impl StrashNode {
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn kind(&self) -> &StrashKind {
        &self.kind
    }
}

#[derive(Clone, Debug)]
pub enum StrashKind {
    Input,
    Const {
        bits: Vec<bool>,
    },
    Op {
        op: NodeOp,
        inputs: Vec<Literal>,
        params: Option<ParamMap>,
    },
}

impl StrashKind {
    pub fn inputs(&self) -> &[Literal] {
        match self {
            StrashKind::Op { inputs, .. } => inputs,
            _ => &[],
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructuralHasher {
    nodes: Vec<StrashNode>,
    node_hash: HashMap<NodeKey, StrashNodeId>,
    const_hash: HashMap<ConstKey, StrashNodeId>,
}

impl StructuralHasher {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_hash: HashMap::new(),
            const_hash: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[StrashNode] {
        &self.nodes
    }

    pub fn node(&self, id: StrashNodeId) -> &StrashNode {
        &self.nodes[id.index()]
    }

    pub fn literal_width(&self, literal: Literal) -> u32 {
        self.node(literal.node()).width()
    }

    pub fn input(&mut self, width: u32) -> Literal {
        let id = self.push_node(StrashNode {
            width,
            kind: StrashKind::Input,
        });
        id.into()
    }

    pub fn constant_bits(&mut self, bits: Vec<bool>) -> Literal {
        let key = ConstKey::from_bits(&bits);
        if let Some(&existing) = self.const_hash.get(&key) {
            return existing.into();
        }

        let width = bits.len() as u32;
        let id = self.push_node(StrashNode {
            width,
            kind: StrashKind::Const { bits: bits.clone() },
        });
        self.const_hash.insert(key, id);
        id.into()
    }

    pub fn constant_zero(&mut self, width: u32) -> Literal {
        self.constant_bits(vec![false; width as usize])
    }

    pub fn constant_one(&mut self, width: u32) -> Literal {
        self.constant_bits(vec![true; width as usize])
    }

    pub fn intern_node<I>(
        &mut self,
        op: NodeOp,
        inputs: I,
        width: u32,
        params: Option<ParamMap>,
    ) -> Literal
    where
        I: IntoIterator<Item = Literal>,
    {
        let mut op = op;
        let mut inputs: Vec<Literal> = inputs.into_iter().collect();

        if matches!(op, NodeOp::Not) {
            debug_assert_eq!(inputs.len(), 1, "NOT expects exactly one input");
            let literal = inputs.pop().expect("NOT inputs cannot be empty").invert();
            return literal;
        }

        let mut invert_output = false;
        if matches!(op, NodeOp::Xnor) {
            op = NodeOp::Xor;
            invert_output = true;
        }

        if matches!(op, NodeOp::And | NodeOp::Or | NodeOp::Xor) {
            inputs.sort();
        }

        match op {
            NodeOp::And => {
                let mut deduped = Vec::with_capacity(inputs.len());
                let mut previous: Option<Literal> = None;

                for &literal in &inputs {
                    if self.literal_is_zero(literal, width) {
                        let mut zero = self.constant_zero(width);
                        if invert_output {
                            zero = zero.invert();
                        }
                        return zero;
                    }

                    if self.literal_is_one(literal, width) {
                        continue;
                    }

                    if previous == Some(literal) {
                        continue;
                    }

                    previous = Some(literal);
                    deduped.push(literal);
                }

                inputs = deduped;

                if inputs.is_empty() {
                    let mut one = self.constant_one(width);
                    if invert_output {
                        one = one.invert();
                    }
                    return one;
                }

                if inputs.len() == 1 {
                    let mut literal = inputs[0];
                    if invert_output {
                        literal = literal.invert();
                    }
                    return literal;
                }
            }
            NodeOp::Xor => {
                let mut parity = invert_output;
                inputs.retain(|&literal| {
                    if self.literal_is_zero(literal, width) {
                        false
                    } else if self.literal_is_one(literal, width) {
                        parity = !parity;
                        false
                    } else {
                        true
                    }
                });

                if inputs.is_empty() {
                    let mut zero = self.constant_zero(width);
                    if parity {
                        zero = zero.invert();
                    }
                    return zero;
                }

                if inputs.len() == 1 {
                    let mut literal = inputs[0];
                    if parity {
                        literal = literal.invert();
                    }
                    return literal;
                }

                invert_output = parity;
            }
            NodeOp::Or => {
                let mut deduped = Vec::with_capacity(inputs.len());
                let mut previous: Option<Literal> = None;

                for &literal in &inputs {
                    if self.literal_is_one(literal, width) {
                        let mut one = self.constant_one(width);
                        if invert_output {
                            one = one.invert();
                        }
                        return one;
                    }

                    if self.literal_is_zero(literal, width) {
                        continue;
                    }

                    if previous == Some(literal) {
                        continue;
                    }

                    previous = Some(literal);
                    deduped.push(literal);
                }

                inputs = deduped;

                if inputs.is_empty() {
                    let mut zero = self.constant_zero(width);
                    if invert_output {
                        zero = zero.invert();
                    }
                    return zero;
                }

                if inputs.len() == 1 {
                    let mut literal = inputs[0];
                    if invert_output {
                        literal = literal.invert();
                    }
                    return literal;
                }
            }
            _ => {}
        }

        let params_key = ParamsKey::from_option(params.as_ref());
        let key = NodeKey {
            op: std::mem::discriminant(&op),
            width,
            params: params_key,
            inputs: inputs.clone(),
        };

        if let Some(&existing) = self.node_hash.get(&key) {
            let mut literal = Literal::from(existing);
            if invert_output {
                literal = literal.invert();
            }
            return literal;
        }

        let node = StrashNode {
            width,
            kind: StrashKind::Op {
                op,
                inputs: inputs.clone(),
                params,
            },
        };
        let id = self.push_node(node);
        self.node_hash.insert(key, id);

        let mut literal = Literal::from(id);
        if invert_output {
            literal = literal.invert();
        }
        literal
    }

    fn push_node(&mut self, node: StrashNode) -> StrashNodeId {
        let id = StrashNodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn literal_is_zero(&self, literal: Literal, expected_width: u32) -> bool {
        self.literal_matches_constant(literal, expected_width, false)
    }

    fn literal_is_one(&self, literal: Literal, expected_width: u32) -> bool {
        self.literal_matches_constant(literal, expected_width, true)
    }

    fn literal_matches_constant(
        &self,
        literal: Literal,
        expected_width: u32,
        expected_bit: bool,
    ) -> bool {
        if self.literal_width(literal) != expected_width {
            return false;
        }

        match &self.nodes[literal.node().index()].kind {
            StrashKind::Const { bits } => {
                if bits.len() as u32 != expected_width {
                    return false;
                }

                bits.iter().all(|&bit| {
                    let value = if literal.is_inverted() { !bit } else { bit };
                    value == expected_bit
                })
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    op: Discriminant<NodeOp>,
    width: u32,
    params: Option<ParamsKey>,
    inputs: Vec<Literal>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ParamsKey(Vec<(String, ParamHashValue)>);

impl ParamsKey {
    fn from_option(params: Option<&ParamMap>) -> Option<Self> {
        params.map(|map| {
            let entries = map
                .iter()
                .map(|(key, value)| (key.clone(), ParamHashValue::from(value)))
                .collect();
            Self(entries)
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ParamHashValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<ParamHashValue>),
    Object(Vec<(String, ParamHashValue)>),
}

impl From<&Value> for ParamHashValue {
    fn from(value: &Value) -> Self {
        match value {
            Value::Null => ParamHashValue::Null,
            Value::Bool(bit) => ParamHashValue::Bool(*bit),
            Value::Number(number) => ParamHashValue::Number(number.to_string()),
            Value::String(text) => ParamHashValue::String(text.clone()),
            Value::Array(entries) => {
                let values = entries.iter().map(ParamHashValue::from).collect();
                ParamHashValue::Array(values)
            }
            Value::Object(object) => {
                let mut pairs: Vec<_> = object
                    .iter()
                    .map(|(key, value)| (key.clone(), ParamHashValue::from(value)))
                    .collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                ParamHashValue::Object(pairs)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConstKey {
    bits: Vec<bool>,
}

impl ConstKey {
    fn from_bits(bits: &[bool]) -> Self {
        Self {
            bits: bits.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commutative_nodes_share_identity() {
        let mut hasher = StructuralHasher::new();
        let a = hasher.input(1);
        let b = hasher.input(1);

        let ab = hasher.intern_node(NodeOp::And, [a, b], 1, None);
        let ba = hasher.intern_node(NodeOp::And, [b, a], 1, None);

        assert_eq!(ab, ba);
        assert_eq!(hasher.len(), 3);

        match hasher.node(ab.node()).kind() {
            StrashKind::Op { op, inputs, .. } => {
                assert!(matches!(op, NodeOp::And));
                assert_eq!(inputs, &[a, b]);
            }
            kind => panic!("expected op node, got {kind:?}"),
        }
    }

    #[test]
    fn double_not_collapses() {
        let mut hasher = StructuralHasher::new();
        let signal = hasher.input(1);

        let inverted = hasher.intern_node(NodeOp::Not, [signal], 1, None);
        assert_ne!(signal, inverted);
        assert_eq!(hasher.len(), 1);

        let restored = hasher.intern_node(NodeOp::Not, [inverted], 1, None);
        assert_eq!(signal, restored);
        assert_eq!(hasher.len(), 1);
    }

    #[test]
    fn xor_with_zero_is_identity() {
        let mut hasher = StructuralHasher::new();
        let signal = hasher.input(1);
        let zero = hasher.constant_zero(1);

        let result = hasher.intern_node(NodeOp::Xor, [signal, zero], 1, None);
        assert_eq!(result, signal);
        assert_eq!(hasher.len(), 2);
    }

    #[test]
    fn and_with_zero_becomes_zero() {
        let mut hasher = StructuralHasher::new();
        let signal = hasher.input(1);
        let zero = hasher.constant_zero(1);

        let result = hasher.intern_node(NodeOp::And, [signal, zero], 1, None);
        assert_eq!(result, zero);
        assert_eq!(hasher.len(), 2);
    }

    #[test]
    fn and_with_one_is_identity() {
        let mut hasher = StructuralHasher::new();
        let signal = hasher.input(1);
        let one = hasher.constant_one(1);

        let result = hasher.intern_node(NodeOp::And, [signal, one], 1, None);
        assert_eq!(result, signal);
        assert_eq!(hasher.len(), 2);
    }

    #[test]
    fn and_all_ones_collapses_to_one() {
        let mut hasher = StructuralHasher::new();
        let one = hasher.constant_one(1);

        let result = hasher.intern_node(NodeOp::And, [one, one], 1, None);
        assert_eq!(result, one);
        assert_eq!(hasher.len(), 1);
    }

    #[test]
    fn constant_zero_deduplicates() {
        let mut hasher = StructuralHasher::new();
        let zero_a = hasher.constant_zero(1);
        let zero_b = hasher.constant_zero(1);

        assert_eq!(zero_a, zero_b);
        assert_eq!(hasher.len(), 1);
    }

    #[test]
    fn or_with_zero_is_identity() {
        let mut hasher = StructuralHasher::new();
        let signal = hasher.input(1);
        let zero = hasher.constant_zero(1);

        let result = hasher.intern_node(NodeOp::Or, [signal, zero], 1, None);
        assert_eq!(result, signal);
        assert_eq!(hasher.len(), 2);
    }

    #[test]
    fn or_with_one_becomes_one() {
        let mut hasher = StructuralHasher::new();
        let signal = hasher.input(1);
        let one = hasher.constant_one(1);

        let result = hasher.intern_node(NodeOp::Or, [signal, one], 1, None);
        assert_eq!(result, one);
        assert_eq!(hasher.len(), 2);
    }

    #[test]
    fn xnor_inverts_xor() {
        let mut hasher = StructuralHasher::new();
        let a = hasher.input(1);
        let b = hasher.input(1);

        let xor = hasher.intern_node(NodeOp::Xor, [a, b], 1, None);
        let xnor = hasher.intern_node(NodeOp::Xnor, [a, b], 1, None);

        assert_eq!(xnor, xor.invert());
        assert_eq!(hasher.len(), 3);
    }
}
