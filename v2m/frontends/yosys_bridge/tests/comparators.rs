use std::collections::BTreeMap;

use anyhow::Result;
use frontends_yosys_bridge::rtlil_to_nir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use v2m_evaluator::{Evaluator, Packed, PackedIndex, SimOptions};
use v2m_formats::nir::PortDirection;
use yosys_bridge::loader::{Cell, Module, RtlilJson};

const RANDOM_ITERS: usize = 100;

#[test]
fn comparator_random_vectors() -> Result<()> {
    let cases = [
        ("$eq", false, false),
        ("$ne", false, false),
        ("$lt", false, false),
        ("$lt", true, false),
        ("$le", false, false),
        ("$le", true, true),
        ("$gt", false, false),
        ("$gt", true, true),
        ("$ge", false, false),
        ("$ge", true, false),
    ];
    let widths = [(8, 8), (13, 17), (32, 24)];

    for &(kind, a_signed, b_signed) in &cases {
        for &(a_width, b_width) in &widths {
            run_compare_case(kind, a_width, b_width, a_signed, b_signed)?;
        }
    }

    Ok(())
}

fn run_compare_case(
    kind: &str,
    a_width: u32,
    b_width: u32,
    a_signed: bool,
    b_signed: bool,
) -> Result<()> {
    let design = make_comparator_design(kind, a_width, b_width, a_signed, b_signed);
    let nir = rtlil_to_nir(&design)?;
    let module = nir
        .modules
        .get(design.top.as_str())
        .expect("module present");

    let mut evaluator = Evaluator::new(&nir, 1, SimOptions::default())?;
    let (mut inputs, input_indices) = build_input_layout(module);
    let output_indices = build_output_layout(module);

    let mut rng = StdRng::seed_from_u64(0x4d5a2d19fe77beef);
    for _ in 0..RANDOM_ITERS {
        let a_value = rng.gen::<u64>() & mask_value(a_width);
        let b_value = rng.gen::<u64>() & mask_value(b_width);

        set_input_value(&mut inputs, input_indices["a"], a_width, a_value);
        set_input_value(&mut inputs, input_indices["b"], b_width, b_value);

        evaluator.set_inputs(&inputs)?;
        evaluator.comb_eval()?;

        let outputs = evaluator.get_outputs();
        let y_index = output_indices["y"];
        let observed = (outputs.lane(y_index, 0)[0] & 1) != 0;
        let expected =
            expected_result(kind, a_value, b_value, a_width, b_width, a_signed, b_signed);
        assert_eq!(
            observed, expected,
            "mismatch for {kind} with a_width={a_width} b_width={b_width} \
             a_signed={a_signed} b_signed={b_signed} a=0x{a_value:x} b=0x{b_value:x}"
        );
    }

    Ok(())
}

fn build_input_layout(
    module: &v2m_formats::nir::Module,
) -> (Packed, BTreeMap<String, PackedIndex>) {
    let mut inputs = Packed::new(1);
    let mut indices = BTreeMap::new();
    for (name, port) in &module.ports {
        if matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
            let index = inputs.allocate(port.bits as usize);
            indices.insert(name.clone(), index);
        }
    }
    (inputs, indices)
}

fn build_output_layout(module: &v2m_formats::nir::Module) -> BTreeMap<String, PackedIndex> {
    let mut outputs = Packed::new(1);
    let mut indices = BTreeMap::new();
    for (name, port) in &module.ports {
        if matches!(port.dir, PortDirection::Output | PortDirection::Inout) {
            let index = outputs.allocate(port.bits as usize);
            indices.insert(name.clone(), index);
        }
    }
    indices
}

fn set_input_value(buffer: &mut Packed, index: PackedIndex, width: u32, value: u64) {
    for bit in 0..width as usize {
        let lane = buffer.lane_mut(index, bit);
        lane[0] = if (value >> bit) & 1 == 1 { u64::MAX } else { 0 };
    }
}

fn mask_value(width: u32) -> u64 {
    if width == 0 {
        0
    } else if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn expected_result(
    kind: &str,
    a_raw: u64,
    b_raw: u64,
    a_width: u32,
    b_width: u32,
    a_signed: bool,
    b_signed: bool,
) -> bool {
    let a_signed_value = if a_signed {
        sign_extend_to_i128(a_raw, a_width)
    } else {
        zero_extend_to_u128(a_raw, a_width) as i128
    };
    let b_signed_value = if b_signed {
        sign_extend_to_i128(b_raw, b_width)
    } else {
        zero_extend_to_u128(b_raw, b_width) as i128
    };
    let a_unsigned_value = zero_extend_to_u128(a_raw, a_width);
    let b_unsigned_value = zero_extend_to_u128(b_raw, b_width);

    match kind {
        "$eq" => a_unsigned_value == b_unsigned_value && a_signed_value == b_signed_value,
        "$ne" => a_unsigned_value != b_unsigned_value || a_signed_value != b_signed_value,
        "$lt" => {
            if a_signed || b_signed {
                a_signed_value < b_signed_value
            } else {
                a_unsigned_value < b_unsigned_value
            }
        }
        "$le" => {
            if a_signed || b_signed {
                a_signed_value <= b_signed_value
            } else {
                a_unsigned_value <= b_unsigned_value
            }
        }
        "$gt" => {
            if a_signed || b_signed {
                a_signed_value > b_signed_value
            } else {
                a_unsigned_value > b_unsigned_value
            }
        }
        "$ge" => {
            if a_signed || b_signed {
                a_signed_value >= b_signed_value
            } else {
                a_unsigned_value >= b_unsigned_value
            }
        }
        other => panic!("unsupported comparator `{other}`"),
    }
}

fn zero_extend_to_u128(value: u64, width: u32) -> u128 {
    if width == 0 {
        0
    } else if width >= 128 {
        value as u128
    } else {
        let mask = (1u128 << width) - 1;
        (value as u128) & mask
    }
}

fn sign_extend_to_i128(value: u64, width: u32) -> i128 {
    if width == 0 {
        0
    } else if width >= 128 {
        value as i128
    } else {
        let masked = zero_extend_to_u128(value, width);
        let shift = 128 - width;
        ((masked << shift) as i128) >> shift
    }
}

fn make_comparator_design(
    kind: &str,
    a_width: u32,
    b_width: u32,
    a_signed: bool,
    b_signed: bool,
) -> RtlilJson {
    let mut next_bit: i64 = 1;
    let a_bits = bit_ids(&mut next_bit, a_width);
    let b_bits = bit_ids(&mut next_bit, b_width);
    let y_bits = bit_ids(&mut next_bit, 1);

    let mut ports = BTreeMap::new();
    ports.insert("a".to_string(), port(&a_bits, "input"));
    ports.insert("b".to_string(), port(&b_bits, "input"));
    ports.insert("y".to_string(), port(&y_bits, "output"));

    let mut netnames = BTreeMap::new();
    netnames.insert("a".to_string(), net(&a_bits));
    netnames.insert("b".to_string(), net(&b_bits));
    netnames.insert("y".to_string(), net(&y_bits));

    let mut cells = BTreeMap::new();
    cells.insert(
        "cmp".to_string(),
        comparator_cell(
            kind, &a_bits, &b_bits, &y_bits, a_width, b_width, a_signed, b_signed,
        ),
    );

    let mut attributes = BTreeMap::new();
    attributes.insert("top".to_string(), json!(1));

    let module = Module {
        attributes,
        ports,
        cells,
        netnames,
    };

    RtlilJson {
        top: "cmp".to_string(),
        modules: BTreeMap::from([(module_name("cmp"), module)]),
    }
}

fn bit_ids(next: &mut i64, width: u32) -> Vec<i64> {
    let mut bits = Vec::with_capacity(width as usize);
    for _ in 0..width {
        bits.push(*next);
        *next += 1;
    }
    bits
}

fn module_name(name: &str) -> String {
    name.to_string()
}

fn port(bits: &[i64], direction: &str) -> Value {
    json!({ "direction": direction, "bits": bits })
}

fn net(bits: &[i64]) -> Value {
    json!({ "bits": bits })
}

fn comparator_cell(
    kind: &str,
    a_bits: &[i64],
    b_bits: &[i64],
    y_bits: &[i64],
    a_width: u32,
    b_width: u32,
    a_signed: bool,
    b_signed: bool,
) -> Cell {
    let mut parameters = BTreeMap::new();
    parameters.insert("A_WIDTH".to_string(), json!(a_width));
    parameters.insert("B_WIDTH".to_string(), json!(b_width));
    parameters.insert("Y_WIDTH".to_string(), json!(1u32));
    parameters.insert("A_SIGNED".to_string(), json!(a_signed));
    parameters.insert("B_SIGNED".to_string(), json!(b_signed));

    let mut connections = BTreeMap::new();
    connections.insert("A".to_string(), json!(a_bits));
    connections.insert("B".to_string(), json!(b_bits));
    connections.insert("Y".to_string(), json!(y_bits));

    Cell {
        kind: kind.to_string(),
        parameters,
        attributes: BTreeMap::new(),
        connections,
    }
}
