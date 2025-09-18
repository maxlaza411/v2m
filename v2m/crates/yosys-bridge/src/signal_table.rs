use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};

use crate::loader::Module;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetEntry {
    pub name: String,
    pub bits: usize,
    pub signed: bool,
}

pub fn build_signal_table(module: &Module) -> Result<Vec<NetEntry>> {
    let mut nets_by_name: BTreeMap<String, NetEntry> = BTreeMap::new();

    for (raw_name, raw_entry) in &module.netnames {
        let normalized_name = normalize_net_name(raw_name);
        if normalized_name.is_empty() {
            bail!("net name `{raw_name}` resolved to an empty string");
        }

        let entry_obj = raw_entry
            .as_object()
            .ok_or_else(|| anyhow!("net `{raw_name}` must be a JSON object"))?;

        let bits_value = entry_obj
            .get("bits")
            .ok_or_else(|| anyhow!("net `{raw_name}` is missing the `bits` field"))?;
        let bit_width = count_bits(bits_value)
            .with_context(|| format!("net `{raw_name}` has an invalid `bits` field"))?;

        if bit_width == 0 {
            bail!("net `{raw_name}` does not contain any bits");
        }

        let signed = net_is_signed(entry_obj);

        let entry = NetEntry {
            name: normalized_name.clone(),
            bits: bit_width,
            signed,
        };

        if nets_by_name.insert(normalized_name, entry).is_some() {
            bail!("duplicate net name encountered while building signal table");
        }
    }

    Ok(nets_by_name.into_values().collect())
}

fn count_bits(value: &Value) -> Result<usize> {
    let array = value
        .as_array()
        .ok_or_else(|| anyhow!("expected `bits` to be an array"))?;

    let mut total = 0usize;
    for entry in array {
        total += count_bits_entry(entry)?;
    }
    Ok(total)
}

fn count_bits_entry(value: &Value) -> Result<usize> {
    match value {
        Value::Array(items) => {
            let mut subtotal = 0usize;
            for item in items {
                subtotal += count_bits_entry(item)?;
            }
            Ok(subtotal)
        }
        Value::Number(_) | Value::String(_) => Ok(1),
        other => Err(anyhow!(
            "unsupported bit entry encountered in `bits`: {}",
            other
        )),
    }
}

fn net_is_signed(entry: &Map<String, Value>) -> bool {
    if let Some(flag) = entry.get("signed") {
        if value_is_truthy(flag) {
            return true;
        }
    }

    if let Some(attributes) = entry.get("attributes").and_then(|value| value.as_object()) {
        if let Some(flag) = attributes.get("signed") {
            if value_is_truthy(flag) {
                return true;
            }
        }
        if let Some(flag) = attributes.get("\\signed") {
            if value_is_truthy(flag) {
                return true;
            }
        }
    }

    false
}

fn value_is_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Number(num) => {
            if let Some(int) = num.as_i64() {
                int != 0
            } else if let Some(uint) = num.as_u64() {
                uint != 0
            } else if let Some(f) = num.as_f64() {
                f != 0.0
            } else {
                false
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                false
            } else {
                let lower = trimmed.to_ascii_lowercase();
                lower != "0" && lower != "false"
            }
        }
        _ => false,
    }
}

fn normalize_net_name(raw: &str) -> String {
    raw.strip_prefix('\\').unwrap_or(raw).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    use std::collections::BTreeMap;

    fn module_from_pairs(pairs: Vec<(&str, Value)>) -> Module {
        let mut netnames = BTreeMap::new();
        for (name, value) in pairs {
            netnames.insert(name.to_string(), value);
        }
        Module {
            attributes: BTreeMap::new(),
            ports: BTreeMap::new(),
            cells: BTreeMap::new(),
            netnames,
        }
    }

    #[test]
    fn collects_width_and_signedness() {
        let module = module_from_pairs(vec![
            (
                "a",
                json!({
                    "bits": [1, 2, 3],
                    "attributes": {},
                }),
            ),
            (
                "b",
                json!({
                    "bits": [4],
                    "signed": 1,
                    "attributes": {},
                }),
            ),
        ]);

        let nets = build_signal_table(&module).expect("build signal table");
        assert_eq!(
            nets,
            vec![
                NetEntry {
                    name: "a".into(),
                    bits: 3,
                    signed: false,
                },
                NetEntry {
                    name: "b".into(),
                    bits: 1,
                    signed: true,
                },
            ]
        );
    }

    #[test]
    fn handles_escaped_names_and_arrays() {
        let module = module_from_pairs(vec![
            (
                "\\foo[0]",
                json!({
                    "bits": [[10], [11, 12]],
                    "attributes": {},
                }),
            ),
            (
                "array[1]",
                json!({
                    "bits": [13, 14],
                    "signed": true,
                    "attributes": {},
                }),
            ),
            (
                "\\1foo",
                json!({
                    "bits": [30],
                    "attributes": {},
                }),
            ),
        ]);

        let nets = build_signal_table(&module).expect("build signal table");
        assert_eq!(
            nets,
            vec![
                NetEntry {
                    name: "1foo".into(),
                    bits: 1,
                    signed: false,
                },
                NetEntry {
                    name: "array[1]".into(),
                    bits: 2,
                    signed: true,
                },
                NetEntry {
                    name: "foo[0]".into(),
                    bits: 3,
                    signed: false,
                },
            ]
        );
    }

    #[test]
    fn reads_signedness_from_attributes() {
        let module = module_from_pairs(vec![(
            "bus",
            json!({
                "bits": [20, 21],
                "attributes": {"signed": 1},
            }),
        )]);

        let nets = build_signal_table(&module).expect("build signal table");
        assert_eq!(
            nets,
            vec![NetEntry {
                name: "bus".into(),
                bits: 2,
                signed: true,
            }]
        );
    }

    #[test]
    fn rejects_duplicate_names_after_normalization() {
        let module = module_from_pairs(vec![
            (
                "foo",
                json!({
                    "bits": [1],
                    "attributes": {},
                }),
            ),
            (
                "\\foo",
                json!({
                    "bits": [2],
                    "attributes": {},
                }),
            ),
        ]);

        let err = build_signal_table(&module).expect_err("expected duplicate name error");
        assert!(err.to_string().contains("duplicate net name"));
    }

    #[test]
    fn errors_when_bits_is_not_array() {
        let module = module_from_pairs(vec![(
            "broken",
            json!({
                "bits": 5,
                "attributes": {},
            }),
        )]);

        let err = build_signal_table(&module).expect_err("expected invalid bits error");
        assert!(err.to_string().contains("bits"));
    }

    #[test]
    fn counts_string_bit_entries() {
        let module = module_from_pairs(vec![(
            "const_wire",
            json!({
                "bits": ["0", "1", "x"],
                "attributes": {},
            }),
        )]);

        let nets = build_signal_table(&module).expect("build signal table");
        assert_eq!(
            nets,
            vec![NetEntry {
                name: "const_wire".into(),
                bits: 3,
                signed: false,
            }]
        );
    }
}
