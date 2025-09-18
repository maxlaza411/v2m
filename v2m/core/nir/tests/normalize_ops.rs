use std::collections::BTreeMap;

use v2m_formats::nir::{
    BitRef, BitRefConst, BitRefNet, Module, Net, Node, NodeOp, Port, PortDirection,
};
use v2m_nir::{normalize_module, NormalizedNodeKind};

#[test]
fn normalize_cat_and_const_nodes() {
    let mut ports = BTreeMap::new();
    ports.insert(
        "a".into(),
        Port {
            dir: PortDirection::Input,
            bits: 2,
            attrs: None,
        },
    );
    ports.insert(
        "b".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "y".into(),
        Port {
            dir: PortDirection::Output,
            bits: 3,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    nets.insert(
        "a".into(),
        Net {
            bits: 2,
            attrs: None,
        },
    );
    nets.insert(
        "b".into(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "y".into(),
        Net {
            bits: 3,
            attrs: None,
        },
    );

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "const_bus".into(),
        Node {
            uid: "const_bus".into(),
            op: NodeOp::Const,
            width: 4,
            pin_map: BTreeMap::from([(
                "Y".into(),
                BitRef::Const(BitRefConst {
                    value: "0b1010".into(),
                    width: 4,
                }),
            )]),
            params: None,
            attrs: None,
        },
    );
    nodes.insert(
        "cat_out".into(),
        Node {
            uid: "cat_out".into(),
            op: NodeOp::Cat,
            width: 3,
            pin_map: BTreeMap::from([
                (
                    "A".into(),
                    BitRef::Net(BitRefNet {
                        net: "a".into(),
                        lsb: 0,
                        msb: 1,
                    }),
                ),
                (
                    "B".into(),
                    BitRef::Net(BitRefNet {
                        net: "b".into(),
                        lsb: 0,
                        msb: 0,
                    }),
                ),
                (
                    "Y".into(),
                    BitRef::Net(BitRefNet {
                        net: "y".into(),
                        lsb: 0,
                        msb: 2,
                    }),
                ),
            ]),
            params: None,
            attrs: None,
        },
    );

    let module = Module { ports, nets, nodes };

    let normalized = normalize_module(&module).expect("normalize module");

    let outputs = normalized.outputs.get("y").expect("cat output");
    let output_nodes: Vec<_> = outputs
        .iter()
        .map(|literal| (literal.node, literal.inverted))
        .collect();
    assert_eq!(output_nodes, vec![(0, false), (1, false), (2, false)]);

    assert!(matches!(
        normalized.nodes[0].kind,
        NormalizedNodeKind::Input
    ));
    assert!(matches!(
        normalized.nodes[1].kind,
        NormalizedNodeKind::Input
    ));
    assert!(matches!(
        normalized.nodes[2].kind,
        NormalizedNodeKind::Input
    ));

    let const_bits: Vec<_> = normalized
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            NormalizedNodeKind::Const { bits } => Some(bits.as_str()),
            _ => None,
        })
        .collect();
    assert!(const_bits.contains(&"0"));
    assert!(const_bits.contains(&"1"));
}
