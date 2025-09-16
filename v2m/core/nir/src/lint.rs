use std::collections::BTreeMap;
use std::fmt;

use v2m_formats::nir::{Module, Nir, NodeOp, PortDirection};
use v2m_formats::{resolve_bitref, ResolvedBit};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Driver {
    Port { port: String, dir: PortDirection },
    NodePin { node: String, pin: String },
}

impl fmt::Display for Driver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Driver::Port { port, dir } => write!(f, "port `{port}` ({dir:?})"),
            Driver::NodePin { node, pin } => write!(f, "node `{node}` pin `{pin}`"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LintError {
    PinWidthMismatch {
        module: String,
        node: String,
        pin: String,
        expected: u32,
        actual: u32,
    },
    CatInputWidthMismatch {
        module: String,
        node: String,
        expected: u32,
        actual: u32,
    },
    MultipleDrivers {
        module: String,
        net: String,
        bit: u32,
        drivers: Vec<Driver>,
    },
    PortDirectionViolation {
        module: String,
        port: String,
        dir: PortDirection,
        bit: u32,
        driver: Driver,
    },
    UndrivenNet {
        module: String,
        net: String,
        bit: u32,
    },
    UnusedNet {
        module: String,
        net: String,
        bit: u32,
    },
    PortWidthMismatch {
        module: String,
        port: String,
        expected: u32,
        actual: u32,
    },
    PortNetMissing {
        module: String,
        port: String,
    },
    PinResolutionFailure {
        module: String,
        node: String,
        pin: String,
        error: String,
    },
}

impl fmt::Display for LintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LintError::PinWidthMismatch {
                module,
                node,
                pin,
                expected,
                actual,
            } => write!(
                f,
                "module `{module}` node `{node}` pin `{pin}` expects width {expected} but got {actual}",
            ),
            LintError::CatInputWidthMismatch {
                module,
                node,
                expected,
                actual,
            } => write!(
                f,
                "module `{module}` node `{node}` concatenation inputs total {actual} bits but output width is {expected}",
            ),
            LintError::MultipleDrivers {
                module,
                net,
                bit,
                drivers,
            } => {
                let joined = drivers
                    .iter()
                    .map(|driver| driver.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "module `{module}` net `{net}[{bit}]` has multiple drivers: {joined}",
                )
            }
            LintError::PortDirectionViolation {
                module,
                port,
                dir,
                bit,
                driver,
            } => write!(
                f,
                "module `{module}` {dir:?} port `{port}[{bit}]` is also driven by {driver}",
            ),
            LintError::UndrivenNet { module, net, bit } => write!(
                f,
                "module `{module}` net `{net}[{bit}]` has no drivers",
            ),
            LintError::UnusedNet { module, net, bit } => write!(
                f,
                "module `{module}` net `{net}[{bit}]` has no loads",
            ),
            LintError::PortWidthMismatch {
                module,
                port,
                expected,
                actual,
            } => write!(
                f,
                "module `{module}` port `{port}` declares width {expected} but net has width {actual}",
            ),
            LintError::PortNetMissing { module, port } => write!(
                f,
                "module `{module}` port `{port}` has no matching net",
            ),
            LintError::PinResolutionFailure {
                module,
                node,
                pin,
                error,
            } => write!(
                f,
                "module `{module}` node `{node}` pin `{pin}` failed to resolve: {error}",
            ),
        }
    }
}

pub fn lint_nir(nir: &Nir) -> Vec<LintError> {
    let mut diagnostics = Vec::new();
    for (module_name, module) in &nir.modules {
        diagnostics.extend(lint_module(module_name, module));
    }
    diagnostics
}

pub fn lint_module(module_name: &str, module: &Module) -> Vec<LintError> {
    let mut diagnostics = Vec::new();

    let mut net_usage: BTreeMap<String, NetUsage> = module
        .nets
        .iter()
        .map(|(name, net)| (name.clone(), NetUsage::new(net.bits)))
        .collect();

    for (port_name, port) in &module.ports {
        match module.nets.get(port_name) {
            Some(net) => {
                if net.bits != port.bits {
                    diagnostics.push(LintError::PortWidthMismatch {
                        module: module_name.to_string(),
                        port: port_name.clone(),
                        expected: port.bits,
                        actual: net.bits,
                    });
                }

                if let Some(usage) = net_usage.get_mut(port_name) {
                    let driver = Driver::Port {
                        port: port_name.clone(),
                        dir: port.dir.clone(),
                    };
                    let load = Load::Port {
                        port: port_name.clone(),
                    };
                    for bit in &mut usage.bits {
                        match port.dir {
                            PortDirection::Input => bit.add_driver(driver.clone()),
                            PortDirection::Output => bit.add_load(load.clone()),
                            PortDirection::Inout => {
                                bit.add_driver(driver.clone());
                                bit.add_load(load.clone());
                            }
                        }
                    }
                }
            }
            None => diagnostics.push(LintError::PortNetMissing {
                module: module_name.to_string(),
                port: port_name.clone(),
            }),
        }
    }

    for (node_name, node) in &module.nodes {
        let mut cat_input_sum = 0u32;
        for (pin_name, bitref) in &node.pin_map {
            let resolved = match resolve_bitref(module, bitref) {
                Ok(bits) => bits,
                Err(err) => {
                    diagnostics.push(LintError::PinResolutionFailure {
                        module: module_name.to_string(),
                        node: node_name.clone(),
                        pin: pin_name.clone(),
                        error: err.to_string(),
                    });
                    continue;
                }
            };

            let actual_width = resolved.len() as u32;
            let output_pin = is_output_pin(&node.op, pin_name);
            if let Some(expected) = expected_pin_width(&node.op, node.width, pin_name, output_pin) {
                if actual_width != expected {
                    diagnostics.push(LintError::PinWidthMismatch {
                        module: module_name.to_string(),
                        node: node_name.clone(),
                        pin: pin_name.clone(),
                        expected,
                        actual: actual_width,
                    });
                }
            }

            if matches!(node.op, NodeOp::Cat) && !output_pin {
                cat_input_sum = cat_input_sum.saturating_add(actual_width);
            }

            for resolved_bit in resolved {
                if let ResolvedBit::Net(bit) = resolved_bit {
                    if let Some(usage) = net_usage.get_mut(&bit.net) {
                        let index = bit.bit as usize;
                        if let Some(bit_usage) = usage.bits.get_mut(index) {
                            if output_pin {
                                bit_usage.add_driver(Driver::NodePin {
                                    node: node_name.clone(),
                                    pin: pin_name.clone(),
                                });
                            } else {
                                bit_usage.add_load(Load::NodePin {
                                    node: node_name.clone(),
                                    pin: pin_name.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        if matches!(node.op, NodeOp::Cat) && cat_input_sum != node.width {
            diagnostics.push(LintError::CatInputWidthMismatch {
                module: module_name.to_string(),
                node: node_name.clone(),
                expected: node.width,
                actual: cat_input_sum,
            });
        }
    }

    for (net_name, usage) in &net_usage {
        for (bit_index, bit_usage) in usage.bits.iter().enumerate() {
            let bit = bit_index as u32;

            if bit_usage.drivers.len() > 1 {
                if let Some((port_dir, port_name)) =
                    bit_usage.drivers.iter().find_map(|driver| match driver {
                        Driver::Port {
                            port,
                            dir: PortDirection::Input,
                        } => Some((PortDirection::Input, port.clone())),
                        _ => None,
                    })
                {
                    if let Some(offending) = bit_usage
                        .drivers
                        .iter()
                        .find(|driver| !matches!(driver, Driver::Port { .. }))
                        .cloned()
                    {
                        diagnostics.push(LintError::PortDirectionViolation {
                            module: module_name.to_string(),
                            port: port_name,
                            dir: port_dir,
                            bit,
                            driver: offending,
                        });
                        continue;
                    }
                }

                diagnostics.push(LintError::MultipleDrivers {
                    module: module_name.to_string(),
                    net: net_name.clone(),
                    bit,
                    drivers: bit_usage.drivers.clone(),
                });
            } else if bit_usage.drivers.is_empty() {
                diagnostics.push(LintError::UndrivenNet {
                    module: module_name.to_string(),
                    net: net_name.clone(),
                    bit,
                });
            }

            if bit_usage.loads.is_empty() {
                diagnostics.push(LintError::UnusedNet {
                    module: module_name.to_string(),
                    net: net_name.clone(),
                    bit,
                });
            }
        }
    }

    diagnostics
}

#[derive(Clone, Debug)]
struct NetUsage {
    bits: Vec<BitUsage>,
}

impl NetUsage {
    fn new(width: u32) -> Self {
        Self {
            bits: vec![BitUsage::default(); width as usize],
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BitUsage {
    drivers: Vec<Driver>,
    loads: Vec<Load>,
}

impl BitUsage {
    fn add_driver(&mut self, driver: Driver) {
        if !self.drivers.contains(&driver) {
            self.drivers.push(driver);
        }
    }

    fn add_load(&mut self, load: Load) {
        if !self.loads.contains(&load) {
            self.loads.push(load);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Load {
    Port { port: String },
    NodePin { node: String, pin: String },
}

fn expected_pin_width(
    op: &NodeOp,
    node_width: u32,
    pin_name: &str,
    is_output: bool,
) -> Option<u32> {
    match op {
        NodeOp::Const => is_output.then_some(node_width),
        NodeOp::Not
        | NodeOp::And
        | NodeOp::Or
        | NodeOp::Xor
        | NodeOp::Xnor
        | NodeOp::Add
        | NodeOp::Sub => Some(node_width),
        NodeOp::Mux => {
            if !is_output && pin_name == "S" {
                Some(1)
            } else {
                Some(node_width)
            }
        }
        NodeOp::Slice => is_output.then_some(node_width),
        NodeOp::Cat => is_output.then_some(node_width),
        NodeOp::Dff => match pin_name {
            "Q" | "D" => Some(node_width),
            "CLK" | "RST" | "EN" => Some(1),
            _ => Some(node_width),
        },
        NodeOp::Latch => match pin_name {
            "Q" | "D" => Some(node_width),
            "EN" | "GATE" | "CLK" | "RST" => Some(1),
            _ => Some(node_width),
        },
    }
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
    use std::collections::BTreeMap;
    use v2m_formats::nir::{BitRef, BitRefConst, BitRefNet, Module, Net, Node, Port};

    fn net_bit(net: &str, lsb: u32, msb: u32) -> BitRef {
        BitRef::Net(BitRefNet {
            net: net.to_string(),
            lsb,
            msb,
        })
    }

    fn const_bits(value: &str, width: u32) -> BitRef {
        BitRef::Const(BitRefConst {
            value: value.to_string(),
            width,
        })
    }

    fn module_with_maps(
        ports: Vec<(&str, Port)>,
        nets: Vec<(&str, Net)>,
        nodes: Vec<(&str, Node)>,
    ) -> Module {
        Module {
            ports: ports
                .into_iter()
                .map(|(name, port)| (name.to_string(), port))
                .collect(),
            nets: nets
                .into_iter()
                .map(|(name, net)| (name.to_string(), net))
                .collect(),
            nodes: nodes
                .into_iter()
                .map(|(name, node)| (name.to_string(), node))
                .collect(),
        }
    }

    fn port(dir: PortDirection, bits: u32) -> Port {
        Port {
            dir,
            bits,
            attrs: None,
        }
    }

    fn net(bits: u32) -> Net {
        Net { bits, attrs: None }
    }

    #[test]
    fn reports_pin_width_mismatch() {
        let module = module_with_maps(
            vec![
                ("a", port(PortDirection::Input, 1)),
                ("b", port(PortDirection::Input, 2)),
                ("y", port(PortDirection::Output, 2)),
            ],
            vec![("a", net(1)), ("b", net(2)), ("y", net(2))],
            vec![(
                "and0",
                Node {
                    uid: "and0".to_string(),
                    op: NodeOp::And,
                    width: 2,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit("a", 0, 0)),
                        ("B".to_string(), net_bit("b", 0, 1)),
                        ("Y".to_string(), net_bit("y", 0, 1)),
                    ]),
                    params: None,
                    attrs: None,
                },
            )],
        );

        let diagnostics = lint_module("Example", &module);

        assert_eq!(
            diagnostics,
            vec![LintError::PinWidthMismatch {
                module: "Example".to_string(),
                node: "and0".to_string(),
                pin: "A".to_string(),
                expected: 2,
                actual: 1,
            }]
        );
    }

    #[test]
    fn detects_multiple_drivers() {
        let module = module_with_maps(
            vec![
                ("a", port(PortDirection::Input, 1)),
                ("b", port(PortDirection::Input, 1)),
                ("x", port(PortDirection::Output, 1)),
            ],
            vec![("a", net(1)), ("b", net(1)), ("x", net(1))],
            vec![
                (
                    "and0",
                    Node {
                        uid: "and0".to_string(),
                        op: NodeOp::And,
                        width: 1,
                        pin_map: BTreeMap::from([
                            ("A".to_string(), net_bit("a", 0, 0)),
                            ("B".to_string(), net_bit("b", 0, 0)),
                            ("Y".to_string(), net_bit("x", 0, 0)),
                        ]),
                        params: None,
                        attrs: None,
                    },
                ),
                (
                    "or0",
                    Node {
                        uid: "or0".to_string(),
                        op: NodeOp::Or,
                        width: 1,
                        pin_map: BTreeMap::from([
                            ("A".to_string(), net_bit("a", 0, 0)),
                            ("B".to_string(), net_bit("b", 0, 0)),
                            ("Y".to_string(), net_bit("x", 0, 0)),
                        ]),
                        params: None,
                        attrs: None,
                    },
                ),
            ],
        );

        let diagnostics = lint_module("Example", &module);

        assert_eq!(
            diagnostics,
            vec![LintError::MultipleDrivers {
                module: "Example".to_string(),
                net: "x".to_string(),
                bit: 0,
                drivers: vec![
                    Driver::NodePin {
                        node: "and0".to_string(),
                        pin: "Y".to_string(),
                    },
                    Driver::NodePin {
                        node: "or0".to_string(),
                        pin: "Y".to_string(),
                    },
                ],
            }]
        );
    }

    #[test]
    fn detects_undriven_net() {
        let module = module_with_maps(
            vec![
                ("a", port(PortDirection::Input, 1)),
                ("y", port(PortDirection::Output, 1)),
            ],
            vec![("a", net(1)), ("y", net(1)), ("floating", net(1))],
            vec![(
                "and0",
                Node {
                    uid: "and0".to_string(),
                    op: NodeOp::And,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit("a", 0, 0)),
                        ("B".to_string(), net_bit("floating", 0, 0)),
                        ("Y".to_string(), net_bit("y", 0, 0)),
                    ]),
                    params: None,
                    attrs: None,
                },
            )],
        );

        let diagnostics = lint_module("Example", &module);

        assert!(diagnostics.contains(&LintError::UndrivenNet {
            module: "Example".to_string(),
            net: "floating".to_string(),
            bit: 0,
        }));
    }

    #[test]
    fn detects_unused_net() {
        let module = module_with_maps(
            vec![("y", port(PortDirection::Output, 1))],
            vec![("dead", net(1)), ("y", net(1))],
            vec![(
                "const0",
                Node {
                    uid: "const0".to_string(),
                    op: NodeOp::Not,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), const_bits("0", 1)),
                        ("Y".to_string(), net_bit("dead", 0, 0)),
                    ]),
                    params: None,
                    attrs: None,
                },
            )],
        );

        let diagnostics = lint_module("Example", &module);

        assert!(diagnostics.contains(&LintError::UnusedNet {
            module: "Example".to_string(),
            net: "dead".to_string(),
            bit: 0,
        }));
    }

    #[test]
    fn detects_port_width_mismatch() {
        let module = module_with_maps(
            vec![
                ("a", port(PortDirection::Input, 2)),
                ("y", port(PortDirection::Output, 1)),
            ],
            vec![("a", net(1)), ("y", net(1))],
            vec![(
                "buf0",
                Node {
                    uid: "buf0".to_string(),
                    op: NodeOp::Not,
                    width: 1,
                    pin_map: BTreeMap::from([
                        ("A".to_string(), net_bit("a", 0, 0)),
                        ("Y".to_string(), net_bit("y", 0, 0)),
                    ]),
                    params: None,
                    attrs: None,
                },
            )],
        );

        let diagnostics = lint_module("Example", &module);

        assert!(diagnostics.contains(&LintError::PortWidthMismatch {
            module: "Example".to_string(),
            port: "a".to_string(),
            expected: 2,
            actual: 1,
        }));
    }

    #[test]
    fn detects_port_direction_violation() {
        let module = module_with_maps(
            vec![
                ("a", port(PortDirection::Input, 1)),
                ("y", port(PortDirection::Output, 1)),
            ],
            vec![("a", net(1)), ("y", net(1))],
            vec![
                (
                    "not0",
                    Node {
                        uid: "not0".to_string(),
                        op: NodeOp::Not,
                        width: 1,
                        pin_map: BTreeMap::from([
                            ("A".to_string(), const_bits("0", 1)),
                            ("Y".to_string(), net_bit("a", 0, 0)),
                        ]),
                        params: None,
                        attrs: None,
                    },
                ),
                (
                    "buf0",
                    Node {
                        uid: "buf0".to_string(),
                        op: NodeOp::Not,
                        width: 1,
                        pin_map: BTreeMap::from([
                            ("A".to_string(), net_bit("a", 0, 0)),
                            ("Y".to_string(), net_bit("y", 0, 0)),
                        ]),
                        params: None,
                        attrs: None,
                    },
                ),
            ],
        );

        let diagnostics = lint_module("Example", &module);

        assert!(diagnostics.contains(&LintError::PortDirectionViolation {
            module: "Example".to_string(),
            port: "a".to_string(),
            dir: PortDirection::Input,
            bit: 0,
            driver: Driver::NodePin {
                node: "not0".to_string(),
                pin: "Y".to_string(),
            },
        }));
    }
}
