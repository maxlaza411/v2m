use std::collections::BTreeMap;

use serde_json::Value;
use v2m_formats::nir::{BitRef, Node, NodeOp};

#[derive(Debug, Clone)]
pub struct NirNode {
    pub op: NodeOp,
    pub width: u32,
    pub pin_map: BTreeMap<String, BitRef>,
    pub attrs: Option<BTreeMap<String, Value>>,
}

impl NirNode {
    pub fn new(op: NodeOp, width: u32, pin_map: BTreeMap<String, BitRef>) -> Self {
        Self {
            op,
            width,
            pin_map,
            attrs: None,
        }
    }

    pub fn with_attrs(mut self, attrs: BTreeMap<String, Value>) -> Self {
        self.attrs = if attrs.is_empty() { None } else { Some(attrs) };
        self
    }

    pub fn into_node(self, uid: String) -> Node {
        Node {
            uid,
            op: self.op,
            width: self.width,
            pin_map: self.pin_map,
            params: None,
            attrs: self.attrs,
        }
    }
}
