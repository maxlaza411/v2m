use std::collections::BTreeMap;

use anyhow::Result;
use serde_json::json;
use v2m_formats::nir::NodeOp;

use crate::context::Ctx;
use crate::nir_node::NirNode;
use yosys_bridge::loader::Cell;

use super::{expect_param_bool, expect_param_u32};

pub fn map_add(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    map_arithmetic(NodeOp::Add, cell, ctx)
}

pub fn map_sub(cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    map_arithmetic(NodeOp::Sub, cell, ctx)
}

fn map_arithmetic(op: NodeOp, cell: &Cell, ctx: &mut Ctx) -> Result<NirNode> {
    let a = ctx.bitref(cell, "A")?;
    let b = ctx.bitref(cell, "B")?;
    let y = ctx.bitref(cell, "Y")?;

    let a_width = expect_param_u32(cell, ctx, "A_WIDTH")?;
    let b_width = expect_param_u32(cell, ctx, "B_WIDTH")?;
    let y_width = expect_param_u32(cell, ctx, "Y_WIDTH")?;

    ctx.expect_width("A", &a, a_width)?;
    ctx.expect_width("B", &b, b_width)?;
    ctx.expect_width("Y", &y, y_width)?;

    let a_signed = expect_param_bool(cell, ctx, "A_SIGNED")?;
    let b_signed = expect_param_bool(cell, ctx, "B_SIGNED")?;
    let y_signed = expect_param_bool(cell, ctx, "Y_SIGNED")?;

    let mut pin_map = BTreeMap::new();
    pin_map.insert("A".to_string(), a);
    pin_map.insert("B".to_string(), b);
    pin_map.insert("Y".to_string(), y);

    let mut attrs = BTreeMap::new();
    attrs.insert("A_WIDTH".to_string(), json!(a_width));
    attrs.insert("B_WIDTH".to_string(), json!(b_width));
    attrs.insert("Y_WIDTH".to_string(), json!(y_width));
    attrs.insert("A_SIGNED".to_string(), json!(a_signed));
    attrs.insert("B_SIGNED".to_string(), json!(b_signed));
    attrs.insert("Y_SIGNED".to_string(), json!(y_signed));

    Ok(NirNode::new(op, y_width, pin_map).with_attrs(attrs))
}
