use std::collections::BTreeMap;

use anyhow::Result;
use frontends_yosys_bridge::rtlil_to_nir;
use serde_json::{json, Value};
use v2m_formats::nir::{BitRef, BitRefNet, NodeOp};
use yosys_bridge::loader::{Cell, Module, RtlilJson};

#[test]
fn dff_maps_to_single_register_with_clk_attr() -> Result<()> {
    let cell = dff_cell(true, &[("CLK", &[1]), ("D", &[2, 3]), ("Q", &[4, 5])]);
    let design = design_with_cell(
        "reg",
        cell,
        &[("clk", &[1]), ("d", &[2, 3]), ("q", &[4, 5])],
    );

    let nir = rtlil_to_nir(&design)?;
    let module = nir.modules.get("top").expect("module");
    assert_eq!(module.nodes.len(), 1);
    assert!(!module.nets.contains_key("reg__v2m_seq_d"));

    let node = module.nodes.get("reg").expect("register node");
    assert_eq!(node.op, NodeOp::Dff);
    assert_eq!(node.width, 2);
    assert_eq!(node.pin_map["D"], net_bit("d", 0, 1));
    assert_eq!(node.pin_map["Q"], net_bit("q", 0, 1));
    assert_eq!(node.pin_map["CLK"], net_bit("clk", 0, 0));

    let attrs = node.attrs.as_ref().expect("clk polarity attr");
    assert_eq!(attrs.get("clk_pol"), Some(&Value::Bool(true)));
    assert!(!node.pin_map.contains_key("RST"));

    Ok(())
}

#[test]
fn adff_includes_reset_pin_and_attrs() -> Result<()> {
    let cell = adff_cell(
        true,
        false,
        json!("3"),
        &[
            ("CLK", &[10]),
            ("D", &[11, 12]),
            ("Q", &[13, 14]),
            ("ARST", &[15]),
        ],
    );
    let design = design_with_cell(
        "reg",
        cell,
        &[
            ("clk", &[10]),
            ("d", &[11, 12]),
            ("q", &[13, 14]),
            ("rst", &[15]),
        ],
    );

    let nir = rtlil_to_nir(&design)?;
    let module = nir.modules.get("top").expect("module");
    let node = module.nodes.get("reg").expect("register node");

    assert_eq!(node.op, NodeOp::Dff);
    assert_eq!(node.pin_map["RST"], net_bit("rst", 0, 0));

    let attrs = node.attrs.as_ref().expect("reset attrs");
    let reset = attrs.get("reset").expect("reset attr");
    let reset_obj = reset.as_object().expect("reset object");
    assert_eq!(
        reset_obj.get("kind"),
        Some(&Value::String("async".to_string()))
    );
    assert_eq!(
        reset_obj.get("polarity"),
        Some(&Value::String("active_low".to_string()))
    );
    assert_eq!(
        reset_obj.get("value"),
        Some(&Value::String("3".to_string()))
    );
    assert_eq!(reset_obj.get("init"), Some(&Value::String("3".to_string())));

    Ok(())
}

#[test]
fn dffe_rewrites_enable_with_mux() -> Result<()> {
    let cell = dffe_cell(
        true,
        true,
        &[
            ("CLK", &[20]),
            ("D", &[21, 22]),
            ("Q", &[23, 24]),
            ("EN", &[25]),
        ],
    );
    let design = design_with_cell(
        "reg",
        cell,
        &[
            ("clk", &[20]),
            ("d", &[21, 22]),
            ("q", &[23, 24]),
            ("en", &[25]),
        ],
    );

    let nir = rtlil_to_nir(&design)?;
    let module = nir.modules.get("top").expect("module");

    assert!(module.nets.contains_key("reg__v2m_seq_d"));
    let mux = module.nodes.get("reg__v2m_seq_en_mux").expect("enable mux");
    assert_eq!(mux.op, NodeOp::Mux);
    assert_eq!(mux.pin_map["A"], net_bit("q", 0, 1));
    assert_eq!(mux.pin_map["B"], net_bit("d", 0, 1));
    assert_eq!(mux.pin_map["S"], net_bit("en", 0, 0));
    assert_eq!(mux.pin_map["Y"], net_bit("reg__v2m_seq_d", 0, 1));

    let reg = module.nodes.get("reg").expect("register node");
    assert_eq!(reg.pin_map["D"], net_bit("reg__v2m_seq_d", 0, 1));

    Ok(())
}

#[test]
fn adffe_combines_enable_and_reset() -> Result<()> {
    let cell = adffe_cell(
        true,
        true,
        false,
        json!("0"),
        &[
            ("CLK", &[30]),
            ("D", &[31, 32]),
            ("Q", &[33, 34]),
            ("ARST", &[35]),
            ("EN", &[36]),
        ],
    );
    let design = design_with_cell(
        "reg",
        cell,
        &[
            ("clk", &[30]),
            ("d", &[31, 32]),
            ("q", &[33, 34]),
            ("rst", &[35]),
            ("en", &[36]),
        ],
    );

    let nir = rtlil_to_nir(&design)?;
    let module = nir.modules.get("top").expect("module");

    let reg = module.nodes.get("reg").expect("register node");
    assert_eq!(reg.pin_map["RST"], net_bit("rst", 0, 0));
    assert_eq!(reg.pin_map["D"], net_bit("reg__v2m_seq_d", 0, 1));

    Ok(())
}

fn dff_cell(clk_pol: bool, connections: &[(&str, &[i64])]) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("WIDTH".to_string(), json!(2));
    parameters.insert("CLK_POLARITY".to_string(), json!(clk_pol));
    make_cell("$dff", parameters, connections)
}

fn adff_cell(
    clk_pol: bool,
    arst_pol: bool,
    arst_value: Value,
    connections: &[(&str, &[i64])],
) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("WIDTH".to_string(), json!(2));
    parameters.insert("CLK_POLARITY".to_string(), json!(clk_pol));
    parameters.insert("ARST_POLARITY".to_string(), json!(arst_pol));
    parameters.insert("ARST_VALUE".to_string(), arst_value);
    make_cell("$adff", parameters, connections)
}

fn dffe_cell(clk_pol: bool, en_pol: bool, connections: &[(&str, &[i64])]) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("WIDTH".to_string(), json!(2));
    parameters.insert("CLK_POLARITY".to_string(), json!(clk_pol));
    parameters.insert("EN_POLARITY".to_string(), json!(en_pol));
    make_cell("$dffe", parameters, connections)
}

fn adffe_cell(
    clk_pol: bool,
    en_pol: bool,
    arst_pol: bool,
    arst_value: Value,
    connections: &[(&str, &[i64])],
) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("WIDTH".to_string(), json!(2));
    parameters.insert("CLK_POLARITY".to_string(), json!(clk_pol));
    parameters.insert("EN_POLARITY".to_string(), json!(en_pol));
    parameters.insert("ARST_POLARITY".to_string(), json!(arst_pol));
    parameters.insert("ARST_VALUE".to_string(), arst_value);
    make_cell("$adffe", parameters, connections)
}

fn make_cell(
    kind: &str,
    parameters: BTreeMap<String, Value>,
    connections: &[(&str, &[i64])],
) -> Cell {
    let mut connections_map = BTreeMap::new();
    for (pin, bits) in connections {
        connections_map.insert((*pin).to_string(), json!(bits));
    }

    Cell {
        kind: kind.to_string(),
        parameters,
        attributes: BTreeMap::new(),
        connections: connections_map,
    }
}

fn design_with_cell(cell_name: &str, cell: Cell, nets: &[(&str, &[i64])]) -> RtlilJson {
    let mut netnames = BTreeMap::new();
    for (name, bits) in nets {
        netnames.insert((*name).to_string(), net(bits));
    }

    let mut cells = BTreeMap::new();
    cells.insert(cell_name.to_string(), cell);

    let module = Module {
        attributes: BTreeMap::new(),
        ports: BTreeMap::new(),
        cells,
        netnames,
    };

    let module_name = "top".to_string();
    RtlilJson {
        top: module_name.clone(),
        modules: BTreeMap::from([(module_name, module)]),
    }
}

fn net(bits: &[i64]) -> Value {
    json!({ "bits": bits })
}

fn net_bit(name: &str, lsb: u32, msb: u32) -> BitRef {
    BitRef::Net(BitRefNet {
        net: name.to_string(),
        lsb,
        msb,
    })
}
