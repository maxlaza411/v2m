use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use v2m_formats::nir::Net as NirNet;

use crate::context::Ctx;
use crate::nir_node::NirNode;
use yosys_bridge::loader::Cell;

pub mod arith;
pub mod compare;
pub mod logic;
pub mod seq;

#[derive(Debug, Default)]
pub struct LoweredCell {
    pub nodes: Vec<(String, NirNode)>,
    pub extra_nets: Vec<(String, NirNet)>,
}

impl LoweredCell {
    pub fn single(name: &str, node: NirNode) -> Self {
        Self {
            nodes: vec![(name.to_string(), node)],
            extra_nets: Vec::new(),
        }
    }

    pub fn push_node(&mut self, name: String, node: NirNode) {
        self.nodes.push((name, node));
    }

    pub fn push_net(&mut self, name: String, net: NirNet) {
        self.extra_nets.push((name, net));
    }
}

fn expect_param_value<'a>(cell: &'a Cell, ctx: &Ctx, name: &str) -> Result<&'a Value> {
    cell.parameters.get(name).ok_or_else(|| {
        anyhow!(
            "cell `{}` in module `{}` is missing parameter `{}`",
            ctx.cell_name(),
            ctx.module_name(),
            name
        )
    })
}

pub fn expect_param_u32(cell: &Cell, ctx: &Ctx, name: &str) -> Result<u32> {
    let value = expect_param_value(cell, ctx, name)?;
    parse_u32(value).with_context(|| parameter_error(ctx, name, "an unsigned integer"))
}

pub fn expect_param_bool(cell: &Cell, ctx: &Ctx, name: &str) -> Result<bool> {
    let value = expect_param_value(cell, ctx, name)?;
    parse_bool(value).with_context(|| parameter_error(ctx, name, "a boolean"))
}

fn parameter_error(ctx: &Ctx, name: &str, description: &str) -> String {
    format!(
        "parameter `{}` on cell `{}` in module `{}` must be {}",
        name,
        ctx.cell_name(),
        ctx.module_name(),
        description
    )
}

fn parse_u32(value: &Value) -> Result<u32> {
    match value {
        Value::Number(num) => {
            if let Some(u) = num.as_u64() {
                u32::try_from(u).context("numeric parameter value exceeds u32 range")
            } else if let Some(i) = num.as_i64() {
                bail!("numeric parameter value `{}` must not be negative", i)
            } else {
                bail!("numeric parameter value `{num}` is not an integer")
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                bail!("string parameter value must not be empty");
            }
            let parsed: u64 = trimmed
                .parse()
                .context("failed to parse numeric string parameter value")?;
            u32::try_from(parsed).context("string parameter value exceeds u32 range")
        }
        other => bail!("unsupported parameter value `{other}`; expected number or string"),
    }
}

fn parse_bool(value: &Value) -> Result<bool> {
    match value {
        Value::Bool(v) => Ok(*v),
        Value::Number(num) => {
            if let Some(u) = num.as_u64() {
                match u {
                    0 => Ok(false),
                    1 => Ok(true),
                    other => bail!("boolean parameter expected 0 or 1 but found {other}"),
                }
            } else if let Some(i) = num.as_i64() {
                match i {
                    0 => Ok(false),
                    1 => Ok(true),
                    other => bail!("boolean parameter expected 0 or 1 but found {other}"),
                }
            } else {
                bail!("boolean parameter `{num}` is not an integer");
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                bail!("string boolean parameter must not be empty");
            }
            match trimmed.to_ascii_lowercase().as_str() {
                "0" | "false" => Ok(false),
                "1" | "true" => Ok(true),
                other => bail!("boolean parameter expected 0/1/true/false but found `{other}`"),
            }
        }
        other => bail!("unsupported parameter value `{other}`; expected boolean"),
    }
}
