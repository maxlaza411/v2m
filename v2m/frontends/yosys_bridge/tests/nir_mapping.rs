use std::collections::BTreeMap;

use anyhow::Result;
use frontends_yosys_bridge::rtlil_to_nir;
use insta::assert_json_snapshot;
use serde_json::{json, Value};
use yosys_bridge::loader::{Cell, Module, RtlilJson};

#[test]
fn alu4_sample_to_nir_snapshot() -> Result<()> {
    let design = alu4_design();
    let nir = rtlil_to_nir(&design)?;
    let value = serde_json::to_value(&nir)?;
    assert_json_snapshot!("alu4_sample", value);
    Ok(())
}

#[test]
fn mux_tree_sample_to_nir_snapshot() -> Result<()> {
    let design = mux_tree_design();
    let nir = rtlil_to_nir(&design)?;
    let value = serde_json::to_value(&nir)?;
    assert_json_snapshot!("mux_tree_sample", value);
    Ok(())
}

fn alu4_design() -> RtlilJson {
    let mut ports = BTreeMap::new();
    ports.insert("a".to_string(), port(&[1, 2, 3, 4], "input"));
    ports.insert("b".to_string(), port(&[5, 6, 7, 8], "input"));
    ports.insert("op".to_string(), port(&[9, 10], "input"));
    ports.insert("y".to_string(), port(&[35, 36, 37, 38], "output"));

    let mut netnames = BTreeMap::new();
    netnames.insert("a".to_string(), net(&[1, 2, 3, 4]));
    netnames.insert("b".to_string(), net(&[5, 6, 7, 8]));
    netnames.insert("op".to_string(), net(&[9, 10]));
    netnames.insert("sum".to_string(), net(&[11, 12, 13, 14]));
    netnames.insert("and_ab".to_string(), net(&[15, 16, 17, 18]));
    netnames.insert("or_ab".to_string(), net(&[19, 20, 21, 22]));
    netnames.insert("xor_ab".to_string(), net(&[23, 24, 25, 26]));
    netnames.insert("lo_sel".to_string(), net(&[27, 28, 29, 30]));
    netnames.insert("hi_sel".to_string(), net(&[31, 32, 33, 34]));
    netnames.insert("y".to_string(), net(&[35, 36, 37, 38]));

    let mut cells = BTreeMap::new();
    cells.insert(
        "add_sum".to_string(),
        add_cell(
            &[
                ("A", &[1, 2, 3, 4]),
                ("B", &[5, 6, 7, 8]),
                ("Y", &[11, 12, 13, 14]),
            ],
            4,
            4,
            4,
        ),
    );
    cells.insert(
        "and_node".to_string(),
        bitwise_cell(
            "$and",
            &[
                ("A", &[1, 2, 3, 4]),
                ("B", &[5, 6, 7, 8]),
                ("Y", &[15, 16, 17, 18]),
            ],
            4,
        ),
    );
    cells.insert(
        "or_node".to_string(),
        bitwise_cell(
            "$or",
            &[
                ("A", &[1, 2, 3, 4]),
                ("B", &[5, 6, 7, 8]),
                ("Y", &[19, 20, 21, 22]),
            ],
            4,
        ),
    );
    cells.insert(
        "xor_node".to_string(),
        bitwise_cell(
            "$xor",
            &[
                ("A", &[1, 2, 3, 4]),
                ("B", &[5, 6, 7, 8]),
                ("Y", &[23, 24, 25, 26]),
            ],
            4,
        ),
    );
    cells.insert(
        "mux_lo".to_string(),
        mux_cell(
            &[
                ("A", &[11, 12, 13, 14]),
                ("B", &[15, 16, 17, 18]),
                ("S", &[9]),
                ("Y", &[27, 28, 29, 30]),
            ],
            4,
            1,
        ),
    );
    cells.insert(
        "mux_hi".to_string(),
        mux_cell(
            &[
                ("A", &[19, 20, 21, 22]),
                ("B", &[23, 24, 25, 26]),
                ("S", &[9]),
                ("Y", &[31, 32, 33, 34]),
            ],
            4,
            1,
        ),
    );
    cells.insert(
        "mux_final".to_string(),
        mux_cell(
            &[
                ("A", &[27, 28, 29, 30]),
                ("B", &[31, 32, 33, 34]),
                ("S", &[10]),
                ("Y", &[35, 36, 37, 38]),
            ],
            4,
            1,
        ),
    );

    let module = Module {
        attributes: BTreeMap::new(),
        ports,
        cells,
        netnames,
    };

    RtlilJson {
        top: "alu4".to_string(),
        modules: BTreeMap::from([(module_name("alu4"), module)]),
    }
}

fn mux_tree_design() -> RtlilJson {
    let mut ports = BTreeMap::new();
    ports.insert("in0".to_string(), port(&[100, 101, 102, 103], "input"));
    ports.insert("in1".to_string(), port(&[104, 105, 106, 107], "input"));
    ports.insert("in2".to_string(), port(&[108, 109, 110, 111], "input"));
    ports.insert("in3".to_string(), port(&[112, 113, 114, 115], "input"));
    ports.insert("sel".to_string(), port(&[116, 117], "input"));
    ports.insert("y".to_string(), port(&[140, 141, 142, 143], "output"));

    let mut netnames = BTreeMap::new();
    netnames.insert("in0".to_string(), net(&[100, 101, 102, 103]));
    netnames.insert("in1".to_string(), net(&[104, 105, 106, 107]));
    netnames.insert("in2".to_string(), net(&[108, 109, 110, 111]));
    netnames.insert("in3".to_string(), net(&[112, 113, 114, 115]));
    netnames.insert("sel".to_string(), net(&[116, 117]));
    netnames.insert("not_in0".to_string(), net(&[118, 119, 120, 121]));
    netnames.insert("xnor_in1_in2".to_string(), net(&[122, 123, 124, 125]));
    netnames.insert("diff".to_string(), net(&[126, 127, 128, 129]));
    netnames.insert("mux_lo".to_string(), net(&[130, 131, 132, 133]));
    netnames.insert("mux_hi".to_string(), net(&[134, 135, 136, 137]));
    netnames.insert("y".to_string(), net(&[140, 141, 142, 143]));

    let mut cells = BTreeMap::new();
    cells.insert(
        "not_in0".to_string(),
        not_cell(
            &[("A", &[100, 101, 102, 103]), ("Y", &[118, 119, 120, 121])],
            4,
        ),
    );
    cells.insert(
        "xnor_inputs".to_string(),
        bitwise_cell(
            "$xnor",
            &[
                ("A", &[104, 105, 106, 107]),
                ("B", &[108, 109, 110, 111]),
                ("Y", &[122, 123, 124, 125]),
            ],
            4,
        ),
    );
    cells.insert(
        "sub_inputs".to_string(),
        sub_cell(
            &[
                ("A", &[112, 113, 114, 115]),
                ("B", &[108, 109, 110, 111]),
                ("Y", &[126, 127, 128, 129]),
            ],
            4,
            4,
            4,
        ),
    );
    cells.insert(
        "mux_level0".to_string(),
        mux_cell(
            &[
                ("A", &[118, 119, 120, 121]),
                ("B", &[122, 123, 124, 125]),
                ("S", &[116]),
                ("Y", &[130, 131, 132, 133]),
            ],
            4,
            1,
        ),
    );
    cells.insert(
        "mux_level1".to_string(),
        mux_cell(
            &[
                ("A", &[126, 127, 128, 129]),
                ("B", &[112, 113, 114, 115]),
                ("S", &[116]),
                ("Y", &[134, 135, 136, 137]),
            ],
            4,
            1,
        ),
    );
    cells.insert(
        "mux_final".to_string(),
        mux_cell(
            &[
                ("A", &[130, 131, 132, 133]),
                ("B", &[134, 135, 136, 137]),
                ("S", &[117]),
                ("Y", &[140, 141, 142, 143]),
            ],
            4,
            1,
        ),
    );

    let module = Module {
        attributes: BTreeMap::new(),
        ports,
        cells,
        netnames,
    };

    RtlilJson {
        top: "mux_tree".to_string(),
        modules: BTreeMap::from([(module_name("mux_tree"), module)]),
    }
}

fn add_cell(connections: &[(&str, &[i64])], a_width: u32, b_width: u32, y_width: u32) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("A_WIDTH".to_string(), json!(a_width));
    parameters.insert("B_WIDTH".to_string(), json!(b_width));
    parameters.insert("Y_WIDTH".to_string(), json!(y_width));
    parameters.insert("A_SIGNED".to_string(), json!(false));
    parameters.insert("B_SIGNED".to_string(), json!(false));
    parameters.insert("Y_SIGNED".to_string(), json!(false));
    make_cell("$add", parameters, connections)
}

fn sub_cell(connections: &[(&str, &[i64])], a_width: u32, b_width: u32, y_width: u32) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("A_WIDTH".to_string(), json!(a_width));
    parameters.insert("B_WIDTH".to_string(), json!(b_width));
    parameters.insert("Y_WIDTH".to_string(), json!(y_width));
    parameters.insert("A_SIGNED".to_string(), json!(false));
    parameters.insert("B_SIGNED".to_string(), json!(false));
    parameters.insert("Y_SIGNED".to_string(), json!(false));
    make_cell("$sub", parameters, connections)
}

fn bitwise_cell(kind: &str, connections: &[(&str, &[i64])], width: u32) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("A_WIDTH".to_string(), json!(width));
    parameters.insert("B_WIDTH".to_string(), json!(width));
    parameters.insert("Y_WIDTH".to_string(), json!(width));
    make_cell(kind, parameters, connections)
}

fn not_cell(connections: &[(&str, &[i64])], width: u32) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("A_WIDTH".to_string(), json!(width));
    parameters.insert("Y_WIDTH".to_string(), json!(width));
    make_cell("$not", parameters, connections)
}

fn mux_cell(connections: &[(&str, &[i64])], width: u32, select_width: u32) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("WIDTH".to_string(), json!(width));
    parameters.insert("S_WIDTH".to_string(), json!(select_width));
    make_cell("$mux", parameters, connections)
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

fn port(bits: &[i64], direction: &str) -> Value {
    json!({ "direction": direction, "bits": bits })
}

fn net(bits: &[i64]) -> Value {
    json!({ "bits": bits })
}

fn module_name(name: &str) -> String {
    name.to_string()
}
