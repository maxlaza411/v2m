use std::collections::BTreeMap;

use serde_json::json;
use v2m_formats::nir::{
    BitRef, BitRefConcat, BitRefConst, BitRefNet, Module, Net, Node, NodeOp, Port, PortDirection,
};
use v2m_nir::{normalize_module, NormalizedNodeKind};

fn net(_name: &str, bits: u32) -> Net {
    Net { bits, attrs: None }
}

fn port(dir: PortDirection, bits: u32) -> Port {
    Port {
        dir,
        bits,
        attrs: None,
    }
}

#[test]
fn normalize_const_node_produces_expected_bits() {
    let mut ports = BTreeMap::new();
    ports.insert("y".to_string(), port(PortDirection::Output, 2));

    let mut nets = BTreeMap::new();
    nets.insert("y".to_string(), net("y", 2));

    let mut pin_map = BTreeMap::new();
    pin_map.insert(
        "Y".to_string(),
        BitRef::Net(BitRefNet {
            net: "y".to_string(),
            lsb: 0,
            msb: 1,
        }),
    );

    let mut params = BTreeMap::new();
    params.insert("value".to_string(), json!("0b10"));

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "const0".to_string(),
        Node {
            uid: "const0".to_string(),
            op: NodeOp::Const,
            width: 2,
            pin_map,
            params: Some(params),
            attrs: None,
        },
    );

    let module = Module { ports, nets, nodes };

    let normalized = normalize_module(&module).expect("normalize const module");

    let y_bits = normalized.outputs.get("y").expect("output y present");
    assert_eq!(y_bits.len(), 2);

    let const_node = normalized
        .nodes
        .iter()
        .find(|node| matches!(&node.kind, NormalizedNodeKind::Const { bits } if bits == "10"));
    assert!(const_node.is_some(), "multi-bit const node recorded");

    let bit0 = &normalized.nodes[y_bits[0].node];
    match &bit0.kind {
        NormalizedNodeKind::Const { bits } => assert_eq!(bits, "0"),
        other => panic!("expected constant node for bit 0, got {other:?}"),
    }

    let bit1 = &normalized.nodes[y_bits[1].node];
    match &bit1.kind {
        NormalizedNodeKind::Const { bits } => assert_eq!(bits, "1"),
        other => panic!("expected constant node for bit 1, got {other:?}"),
    }
}

#[test]
fn normalize_cat_node_concatenates_inputs_and_constants() {
    let mut ports = BTreeMap::new();
    ports.insert("a".to_string(), port(PortDirection::Input, 2));
    ports.insert("b".to_string(), port(PortDirection::Input, 1));
    ports.insert("y".to_string(), port(PortDirection::Output, 4));

    let mut nets = BTreeMap::new();
    nets.insert("a".to_string(), net("a", 2));
    nets.insert("b".to_string(), net("b", 1));
    nets.insert("y".to_string(), net("y", 4));

    let mut pin_map = BTreeMap::new();
    pin_map.insert(
        "A".to_string(),
        BitRef::Concat(BitRefConcat {
            concat: vec![
                BitRef::Net(BitRefNet {
                    net: "a".to_string(),
                    lsb: 0,
                    msb: 1,
                }),
                BitRef::Const(BitRefConst {
                    value: "0b1".to_string(),
                    width: 1,
                }),
                BitRef::Net(BitRefNet {
                    net: "b".to_string(),
                    lsb: 0,
                    msb: 0,
                }),
            ],
        }),
    );
    pin_map.insert(
        "Y".to_string(),
        BitRef::Net(BitRefNet {
            net: "y".to_string(),
            lsb: 0,
            msb: 3,
        }),
    );

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "cat0".to_string(),
        Node {
            uid: "cat0".to_string(),
            op: NodeOp::Cat,
            width: 4,
            pin_map,
            params: None,
            attrs: None,
        },
    );

    let module = Module { ports, nets, nodes };

    let normalized = normalize_module(&module).expect("normalize cat module");

    let a_bits = normalized.inputs.get("a").expect("input a");
    let b_bits = normalized.inputs.get("b").expect("input b");
    let y_bits = normalized.outputs.get("y").expect("output y");

    assert_eq!(y_bits.len(), 4);
    assert_eq!(a_bits.len(), 2);
    assert_eq!(b_bits.len(), 1);

    assert_eq!(y_bits[0].node, a_bits[0].node);
    assert_eq!(y_bits[0].inverted, a_bits[0].inverted);
    assert_eq!(y_bits[1].node, a_bits[1].node);
    assert_eq!(y_bits[1].inverted, a_bits[1].inverted);

    let const_bit = &normalized.nodes[y_bits[2].node];
    match &const_bit.kind {
        NormalizedNodeKind::Const { bits } => assert_eq!(bits, "1"),
        other => panic!("expected constant node for concatenated constant bit, got {other:?}"),
    }

    assert_eq!(y_bits[3].node, b_bits[0].node);
    assert_eq!(y_bits[3].inverted, b_bits[0].inverted);
}
