use std::collections::BTreeMap;

use anyhow::{bail, Result};
use v2m_formats::nir::NodeOp;

use crate::context::Ctx;
use crate::nir_node::NirNode;
use yosys_bridge::loader::Cell;

use super::expect_param_u32;

pub fn map_and(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    map_bitwise_binary(NodeOp::And, cell, ctx)
}

pub fn map_or(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    map_bitwise_binary(NodeOp::Or, cell, ctx)
}

pub fn map_xor(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    map_bitwise_binary(NodeOp::Xor, cell, ctx)
}

pub fn map_xnor(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    map_bitwise_binary(NodeOp::Xnor, cell, ctx)
}

pub fn map_not(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    let a = ctx.bitref(cell, "A")?;
    let y = ctx.bitref(cell, "Y")?;

    let a_width = expect_param_u32(cell, ctx, "A_WIDTH")?;
    let y_width = expect_param_u32(cell, ctx, "Y_WIDTH")?;

    if a_width != y_width {
        bail!(
            "cell `{}` in module `{}` has mismatched NOT widths: A_WIDTH={} and Y_WIDTH={}",
            ctx.cell_name(),
            ctx.module_name(),
            a_width,
            y_width
        );
    }

    ctx.expect_width("A", &a, a_width)?;
    ctx.expect_width("Y", &y, y_width)?;

    let mut pin_map = BTreeMap::new();
    pin_map.insert("A".to_string(), a);
    pin_map.insert("Y".to_string(), y);

    Ok(NirNode::new(NodeOp::Not, y_width, pin_map))
}

pub fn map_mux(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    let a = ctx.bitref(cell, "A")?;
    let b = ctx.bitref(cell, "B")?;
    let s = ctx.bitref(cell, "S")?;
    let y = ctx.bitref(cell, "Y")?;

    let width = expect_param_u32(cell, ctx, "WIDTH")?;
    let select_width = expect_param_u32(cell, ctx, "S_WIDTH")?;

    ctx.expect_width("A", &a, width)?;
    ctx.expect_width("B", &b, width)?;
    ctx.expect_width("Y", &y, width)?;
    ctx.expect_width("S", &s, select_width)?;

    let mut pin_map = BTreeMap::new();
    pin_map.insert("A".to_string(), a);
    pin_map.insert("B".to_string(), b);
    pin_map.insert("S".to_string(), s);
    pin_map.insert("Y".to_string(), y);

    Ok(NirNode::new(NodeOp::Mux, width, pin_map))
}

fn map_bitwise_binary(op: NodeOp, cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    let a = ctx.bitref(cell, "A")?;
    let b = ctx.bitref(cell, "B")?;
    let y = ctx.bitref(cell, "Y")?;

    let a_width = expect_param_u32(cell, ctx, "A_WIDTH")?;
    let b_width = expect_param_u32(cell, ctx, "B_WIDTH")?;
    let y_width = expect_param_u32(cell, ctx, "Y_WIDTH")?;

    if a_width != b_width || a_width != y_width {
        bail!(
            "cell `{}` in module `{}` has mismatched widths: A_WIDTH={}, B_WIDTH={}, Y_WIDTH={}",
            ctx.cell_name(),
            ctx.module_name(),
            a_width,
            b_width,
            y_width
        );
    }

    ctx.expect_width("A", &a, a_width)?;
    ctx.expect_width("B", &b, b_width)?;
    ctx.expect_width("Y", &y, y_width)?;

    let mut pin_map = BTreeMap::new();
    pin_map.insert("A".to_string(), a);
    pin_map.insert("B".to_string(), b);
    pin_map.insert("Y".to_string(), y);

    Ok(NirNode::new(op, y_width, pin_map))
}
