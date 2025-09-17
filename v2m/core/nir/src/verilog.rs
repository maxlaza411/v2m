use std::collections::BTreeSet;
use std::fmt::Write;

use thiserror::Error;

use v2m_formats::nir::{BitRef, BitRefConcat, BitRefNet, Module, Nir, Node, NodeOp, PortDirection};

#[derive(Debug, Error)]
pub enum VerilogExportError {
    #[error("node `{node}` uses unsupported operation `{op:?}`")]
    UnsupportedOperation { node: String, op: NodeOp },
    #[error("node `{node}` is missing required pin `{pin}`")]
    MissingPin { node: String, pin: String },
    #[error("node `{node}` pin `{pin}` references unsupported constant destination")]
    ConstOnOutput { node: String, pin: String },
    #[error("net `{net}` referenced by node `{node}` is undefined in module")]
    UnknownNet { node: String, net: String },
    #[error("failed to format Verilog output: {0}")]
    FmtError(#[from] std::fmt::Error),
}

pub fn nir_to_verilog(nir: &Nir) -> Result<String, VerilogExportError> {
    let mut output = String::new();
    let mut first = true;
    for (name, module) in &nir.modules {
        if !first {
            output.push_str("\n");
        }
        first = false;
        render_module(&mut output, name, module)?;
        output.push_str("\n");
    }
    Ok(output)
}

fn render_module(
    output: &mut String,
    name: &str,
    module: &Module,
) -> Result<(), VerilogExportError> {
    writeln!(output, "module {}(", name)?;

    let mut port_names: Vec<_> = module.ports.keys().collect();
    port_names.sort();
    for (index, port_name) in port_names.iter().enumerate() {
        let port = module
            .ports
            .get(*port_name)
            .expect("port must exist while iterating names");
        let dir = match port.dir {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
            PortDirection::Inout => "inout",
        };
        let width = format_width(port.bits);
        let comma = if index + 1 == port_names.len() {
            ""
        } else {
            ","
        };
        writeln!(output, "    {} wire {}{}{}", dir, width, port_name, comma)?;
    }
    writeln!(output, ");")?;
    output.push_str("\n");

    let mut wire_names: Vec<_> = module.nets.keys().collect();
    wire_names.sort();

    let port_set: BTreeSet<_> = module.ports.keys().collect();
    for net_name in &wire_names {
        if port_set.contains(net_name) {
            continue;
        }
        let net = module
            .nets
            .get(*net_name)
            .expect("net must exist while iterating names");
        writeln!(output, "    wire {}{};", format_width(net.bits), net_name)?;
    }

    let mut sequential = Vec::new();
    let mut assignments = Vec::new();

    for (node_name, node) in &module.nodes {
        match node.op {
            NodeOp::Dff => {
                sequential.push(collect_dff(node_name, node, module)?);
            }
            NodeOp::Latch => {
                return Err(VerilogExportError::UnsupportedOperation {
                    node: node_name.clone(),
                    op: node.op.clone(),
                });
            }
            _ => {
                assignments.push(collect_assignment(node_name, node, module)?);
            }
        }
    }

    for reg in &sequential {
        let decl_width = format_width(reg.width);
        writeln!(output, "    reg {}{};", decl_width, reg.reg_name)?;
    }

    for assignment in &assignments {
        writeln!(
            output,
            "    assign {} = {};",
            assignment.lhs, assignment.rhs
        )?;
    }

    for reg in &sequential {
        writeln!(
            output,
            "    assign {} = {};",
            format_lhs(&reg.node_name, module, &reg.q_binding)?,
            reg.reg_name
        )?;
    }

    for reg in &sequential {
        writeln!(output, "    always @(posedge {}) begin", reg.clk_expr)?;
        writeln!(output, "        {} <= {};", reg.reg_name, reg.d_expr)?;
        writeln!(output, "    end")?;
    }

    writeln!(output, "endmodule")?;
    Ok(())
}

struct Assignment {
    lhs: String,
    rhs: String,
}

struct DffExport {
    reg_name: String,
    width: u32,
    node_name: String,
    q_binding: BitBinding,
    d_expr: String,
    clk_expr: String,
}

type BitBinding = BitRef;

fn collect_assignment(
    node_name: &str,
    node: &Node,
    module: &Module,
) -> Result<Assignment, VerilogExportError> {
    let y_pin = node
        .pin_map
        .get("Y")
        .ok_or_else(|| VerilogExportError::MissingPin {
            node: node_name.to_string(),
            pin: "Y".to_string(),
        })?;
    let lhs = format_lhs(node_name, module, y_pin)?;

    let rhs = match node.op {
        NodeOp::Const => {
            let params = node.params.as_ref().and_then(|map| map.get("value"));
            let value = params.and_then(|value| value.as_str()).ok_or_else(|| {
                VerilogExportError::MissingPin {
                    node: node_name.to_string(),
                    pin: "value".to_string(),
                }
            })?;
            format_const_expr(value, node.width)
        }
        NodeOp::Slice | NodeOp::Cat => format_expr(
            node_name,
            module,
            node.pin_map
                .get("A")
                .ok_or_else(|| VerilogExportError::MissingPin {
                    node: node_name.to_string(),
                    pin: "A".to_string(),
                })?,
        )?,
        NodeOp::Not => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            format!("~({})", a)
        }
        NodeOp::And => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            format!("({}) & ({})", a, b)
        }
        NodeOp::Or => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            format!("({}) | ({})", a, b)
        }
        NodeOp::Xor => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            format!("({}) ^ ({})", a, b)
        }
        NodeOp::Xnor => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            format!("({}) ~^ ({})", a, b)
        }
        NodeOp::Mux => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            let s = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("S")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "S".to_string(),
                    })?,
            )?;
            format!("({}) ? ({}) : ({})", s, b, a)
        }
        NodeOp::Add => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            format!("({}) + ({})", a, b)
        }
        NodeOp::Sub => {
            let a = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("A")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "A".to_string(),
                    })?,
            )?;
            let b = format_expr(
                node_name,
                module,
                node.pin_map
                    .get("B")
                    .ok_or_else(|| VerilogExportError::MissingPin {
                        node: node_name.to_string(),
                        pin: "B".to_string(),
                    })?,
            )?;
            format!("({}) - ({})", a, b)
        }
        _ => {
            return Err(VerilogExportError::UnsupportedOperation {
                node: node_name.to_string(),
                op: node.op.clone(),
            });
        }
    };

    Ok(Assignment { lhs, rhs })
}

fn collect_dff(
    node_name: &str,
    node: &Node,
    module: &Module,
) -> Result<DffExport, VerilogExportError> {
    let q_ref = node
        .pin_map
        .get("Q")
        .ok_or_else(|| VerilogExportError::MissingPin {
            node: node_name.to_string(),
            pin: "Q".to_string(),
        })?
        .clone();
    let d_ref = node
        .pin_map
        .get("D")
        .ok_or_else(|| VerilogExportError::MissingPin {
            node: node_name.to_string(),
            pin: "D".to_string(),
        })?
        .clone();
    let clk_ref = node
        .pin_map
        .get("CLK")
        .ok_or_else(|| VerilogExportError::MissingPin {
            node: node_name.to_string(),
            pin: "CLK".to_string(),
        })?;

    let reg_name = sanitize_identifier(&format!("{}_reg", node.uid));
    let clk_expr = format_expr(node_name, module, clk_ref)?;
    let d_expr = format_expr(node_name, module, &d_ref)?;

    Ok(DffExport {
        reg_name,
        width: node.width,
        node_name: node_name.to_string(),
        q_binding: q_ref,
        d_expr,
        clk_expr,
    })
}

fn format_expr(node: &str, module: &Module, bitref: &BitRef) -> Result<String, VerilogExportError> {
    match bitref {
        BitRef::Net(net) => format_net_ref(node, module, net),
        BitRef::Const(constant) => Ok(format_const_expr(&constant.value, constant.width)),
        BitRef::Concat(BitRefConcat { concat }) => {
            let mut parts = Vec::with_capacity(concat.len());
            for part in concat {
                parts.push(format_expr(node, module, part)?);
            }
            Ok(format!("{{{}}}", parts.join(", ")))
        }
    }
}

fn format_lhs(node: &str, module: &Module, bitref: &BitRef) -> Result<String, VerilogExportError> {
    match bitref {
        BitRef::Const(_) => Err(VerilogExportError::ConstOnOutput {
            node: node.to_string(),
            pin: "Y".to_string(),
        }),
        BitRef::Net(_) | BitRef::Concat(_) => format_expr(node, module, bitref),
    }
}

fn format_net_ref(
    node: &str,
    module: &Module,
    net: &BitRefNet,
) -> Result<String, VerilogExportError> {
    let definition = module
        .nets
        .get(&net.net)
        .ok_or_else(|| VerilogExportError::UnknownNet {
            node: node.to_string(),
            net: net.net.clone(),
        })?;
    if net.lsb == 0 && net.msb + 1 == definition.bits {
        Ok(net.net.clone())
    } else if net.lsb == net.msb {
        Ok(format!("{}[{}]", net.net, net.lsb))
    } else {
        Ok(format!("{}[{}:{}]", net.net, net.msb, net.lsb))
    }
}

fn format_const_expr(value: &str, width: u32) -> String {
    let mut cleaned: String = value.chars().filter(|ch| *ch != '_').collect();
    if cleaned.starts_with("0b") || cleaned.starts_with("0B") {
        let digits = cleaned.split_off(2);
        format!("{}'b{}", width, digits)
    } else if cleaned.starts_with("0x") || cleaned.starts_with("0X") {
        let digits = cleaned.split_off(2);
        format!("{}'h{}", width, digits)
    } else if cleaned.starts_with("0o") || cleaned.starts_with("0O") {
        let digits = cleaned.split_off(2);
        format!("{}'o{}", width, digits)
    } else {
        if cleaned.starts_with('+') {
            cleaned.remove(0);
        }
        format!("{}'d{}", width, cleaned)
    }
}

fn format_width(width: u32) -> String {
    if width <= 1 {
        String::new()
    } else {
        format!("[{}:0] ", width - 1)
    }
}

fn sanitize_identifier(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            result.push(ch);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() || result.chars().next().unwrap().is_ascii_digit() {
        result.insert(0, '_');
    }
    result
}
