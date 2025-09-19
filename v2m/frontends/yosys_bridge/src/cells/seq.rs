use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use v2m_formats::nir::{BitRef, BitRefNet, Net as NirNet, NodeOp};

use crate::context::Ctx;
use crate::nir_node::NirNode;
use yosys_bridge::loader::Cell;

use super::{expect_param_bool, expect_param_u32, expect_param_value, LoweredCell};

pub fn map_dff(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    lower_dff(cell, ctx, None, None)
}

pub fn map_adff(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let arst = ctx.bitref(cell, "ARST")?;
    let arst_pol = expect_param_bool(cell, ctx, "ARST_POLARITY")?;
    let arst_value_param = expect_param_value(cell, ctx, "ARST_VALUE")?;
    let arst_value = normalize_reset_value(ctx, "ARST_VALUE", arst_value_param)?;

    let reset = AsyncReset {
        pin: arst,
        polarity_active_high: arst_pol,
        value: arst_value,
    };

    lower_dff(cell, ctx, Some(reset), None)
}

pub fn map_dffe(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let en = ctx.bitref(cell, "EN")?;
    let en_pol = expect_param_bool(cell, ctx, "EN_POLARITY")?;

    let enable = EnableInfo {
        pin: en,
        polarity_active_high: en_pol,
    };

    lower_dff(cell, ctx, None, Some(enable))
}

pub fn map_adffe(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let arst = ctx.bitref(cell, "ARST")?;
    let arst_pol = expect_param_bool(cell, ctx, "ARST_POLARITY")?;
    let arst_value_param = expect_param_value(cell, ctx, "ARST_VALUE")?;
    let arst_value = normalize_reset_value(ctx, "ARST_VALUE", arst_value_param)?;

    let reset = AsyncReset {
        pin: arst,
        polarity_active_high: arst_pol,
        value: arst_value,
    };

    let en = ctx.bitref(cell, "EN")?;
    let en_pol = expect_param_bool(cell, ctx, "EN_POLARITY")?;

    let enable = EnableInfo {
        pin: en,
        polarity_active_high: en_pol,
    };

    lower_dff(cell, ctx, Some(reset), Some(enable))
}

struct AsyncReset {
    pin: BitRef,
    polarity_active_high: bool,
    value: Value,
}

struct EnableInfo {
    pin: BitRef,
    polarity_active_high: bool,
}

fn lower_dff(
    cell: &Cell,
    ctx: &mut Ctx,
    reset: Option<AsyncReset>,
    enable: Option<EnableInfo>,
) -> Result<LoweredCell> {
    let width = expect_param_u32(cell, ctx, "WIDTH")?;
    if width == 0 {
        bail!(
            "cell `{}` in module `{}` has zero-width register",
            ctx.cell_name(),
            ctx.module_name()
        );
    }

    let d = ctx.bitref(cell, "D")?;
    let q = ctx.bitref(cell, "Q")?;
    let clk = ctx.bitref(cell, "CLK")?;

    ctx.expect_width("D", &d, width)?;
    ctx.expect_width("Q", &q, width)?;
    ctx.expect_width("CLK", &clk, 1)?;

    if let Some(info) = &reset {
        ctx.expect_width("ARST", &info.pin, 1)?;
    }
    if let Some(info) = &enable {
        ctx.expect_width("EN", &info.pin, 1)?;
    }

    let clk_pol = expect_param_bool(cell, ctx, "CLK_POLARITY")?;

    let mut lowered = LoweredCell::default();
    let mut d_source = d.clone();

    if let Some(info) = enable {
        let net_name = format!("{}__v2m_seq_d", ctx.cell_name());
        lowered.push_net(
            net_name.clone(),
            NirNet {
                bits: width,
                attrs: None,
            },
        );
        let mux_output = BitRef::Net(BitRefNet {
            net: net_name,
            lsb: 0,
            msb: width - 1,
        });

        let (a_pin, b_pin) = if info.polarity_active_high {
            (q.clone(), d.clone())
        } else {
            (d.clone(), q.clone())
        };

        let mut mux_pins = BTreeMap::new();
        mux_pins.insert("A".to_string(), a_pin);
        mux_pins.insert("B".to_string(), b_pin);
        mux_pins.insert("S".to_string(), info.pin);
        mux_pins.insert("Y".to_string(), mux_output.clone());

        let mux_node = NirNode::new(NodeOp::Mux, width, mux_pins);
        let mux_name = format!("{}__v2m_seq_en_mux", ctx.cell_name());
        lowered.push_node(mux_name, mux_node);

        d_source = mux_output;
    }

    let mut pin_map = BTreeMap::new();
    pin_map.insert("D".to_string(), d_source);
    pin_map.insert("Q".to_string(), q);
    pin_map.insert("CLK".to_string(), clk);

    let mut attrs = BTreeMap::new();
    attrs.insert("clk_pol".to_string(), json!(clk_pol));

    if let Some(info) = reset {
        pin_map.insert("RST".to_string(), info.pin);

        let mut reset_map = Map::new();
        reset_map.insert("kind".to_string(), Value::String("async".to_string()));
        let polarity = if info.polarity_active_high {
            "active_high"
        } else {
            "active_low"
        };
        reset_map.insert("polarity".to_string(), Value::String(polarity.to_string()));
        let reset_value = info.value;
        reset_map.insert("value".to_string(), reset_value.clone());
        reset_map.insert("init".to_string(), reset_value);

        attrs.insert("reset".to_string(), Value::Object(reset_map));
    }

    let node = NirNode::new(NodeOp::Dff, width, pin_map).with_attrs(attrs);
    lowered.push_node(ctx.cell_name().to_string(), node);

    Ok(lowered)
}

fn normalize_reset_value(ctx: &Ctx, name: &str, value: &Value) -> Result<Value> {
    match value {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Number(num) => Ok(Value::String(num.to_string())),
        Value::Bool(flag) => Ok(Value::String(if *flag { "1" } else { "0" }.to_string())),
        other => bail!(
            "parameter `{}` on cell `{}` in module `{}` must be string, number, or boolean, got `{}`",
            name,
            ctx.cell_name(),
            ctx.module_name(),
            other
        ),
    }
}
