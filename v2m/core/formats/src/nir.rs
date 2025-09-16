use crate::{
    format_validation_errors, read_and_validate, to_value_and_validate, write_validated, AttrValue,
    Error,
};
use jsonschema::{Draft, JSONSchema};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::LazyLock;

static SCHEMA_JSON: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../schemas/nir.schema.json"))
        .expect("invalid nir schema")
});

static COMPILED_SCHEMA: LazyLock<JSONSchema> = LazyLock::new(|| {
    JSONSchema::options()
        .with_draft(Draft::Draft202012)
        .compile(&SCHEMA_JSON)
        .expect("nir schema compilation failed")
});

fn validate_json(value: &Value) -> Result<(), Error> {
    if let Err(errors) = COMPILED_SCHEMA.validate(value) {
        return Err(format_validation_errors(errors));
    }
    Ok(())
}

pub fn schema_json() -> &'static Value {
    &SCHEMA_JSON
}

pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Nir, Error> {
    read_and_validate(reader, validate_json)
}

pub fn from_value(value: Value) -> Result<Nir, Error> {
    validate_json(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn to_value(nir: &Nir) -> Result<Value, Error> {
    to_value_and_validate(nir, validate_json)
}

pub fn to_writer<W: std::io::Write>(nir: &Nir, writer: W) -> Result<(), Error> {
    write_validated(nir, writer, validate_json)
}

pub fn load_nir(path: impl AsRef<Path>) -> Result<Nir, Error> {
    let file = File::open(path)?;
    from_reader(file)
}

pub fn save_nir(nir: &Nir, path: impl AsRef<Path>) -> Result<(), Error> {
    let path = path.as_ref();
    let file = File::create(path)?;
    to_writer(nir, file)?;

    #[cfg(debug_assertions)]
    {
        let file = File::open(path)?;
        from_reader(file)?;
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nir {
    pub v: String,
    pub design: String,
    pub top: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<BTreeMap<String, AttrValue>>,
    pub modules: BTreeMap<String, Module>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cmdline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_digest_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub ports: BTreeMap<String, Port>,
    pub nets: BTreeMap<String, Net>,
    pub nodes: BTreeMap<String, Node>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub dir: PortDirection,
    pub bits: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PortDirection {
    Input,
    Output,
    Inout,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Net {
    pub bits: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub uid: String,
    pub op: NodeOp,
    pub width: u32,
    pub pin_map: BTreeMap<String, BitRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<BTreeMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum NodeOp {
    And,
    Or,
    Xor,
    Xnor,
    Not,
    Mux,
    Add,
    Sub,
    Slice,
    Cat,
    Const,
    Dff,
    Latch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BitRef {
    Net(BitRefNet),
    Const(BitRefConst),
    Concat(BitRefConcat),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BitRefNet {
    pub net: String,
    pub lsb: u32,
    pub msb: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BitRefConst {
    #[serde(rename = "const")]
    pub value: String,
    pub width: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BitRefConcat {
    pub concat: Vec<BitRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedNetBit {
    pub net: String,
    pub bit: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedBit {
    Net(ResolvedNetBit),
    Const(bool),
}

pub fn resolve_bitref(module: &Module, bitref: &BitRef) -> Result<Vec<ResolvedBit>, Error> {
    match bitref {
        BitRef::Net(net) => resolve_bitref_net(module, net),
        BitRef::Const(constant) => resolve_bitref_const(constant),
        BitRef::Concat(concat) => {
            let mut resolved = Vec::new();
            for part in &concat.concat {
                resolved.extend(resolve_bitref(module, part)?);
            }
            Ok(resolved)
        }
    }
}

fn resolve_bitref_net(module: &Module, net: &BitRefNet) -> Result<Vec<ResolvedBit>, Error> {
    let definition = module.nets.get(&net.net).ok_or_else(|| Error::UnknownNet {
        net: net.net.clone(),
    })?;

    if net.lsb > net.msb {
        return Err(Error::InvalidBitRange {
            net: net.net.clone(),
            lsb: net.lsb,
            msb: net.msb,
        });
    }

    if net.msb >= definition.bits {
        return Err(Error::BitRangeOutOfBounds {
            net: net.net.clone(),
            lsb: net.lsb,
            msb: net.msb,
            width: definition.bits,
        });
    }

    let mut resolved = Vec::with_capacity((net.msb - net.lsb + 1) as usize);
    for bit in net.lsb..=net.msb {
        resolved.push(ResolvedBit::Net(ResolvedNetBit {
            net: net.net.clone(),
            bit,
        }));
    }

    Ok(resolved)
}

fn resolve_bitref_const(constant: &BitRefConst) -> Result<Vec<ResolvedBit>, Error> {
    let width = constant.width as usize;
    let literal = constant.value.as_str();

    let (base, digits) = match literal.strip_prefix("0b") {
        Some(rest) => (2_u32, rest),
        None => match literal.strip_prefix("0B") {
            Some(rest) => (2, rest),
            None => match literal.strip_prefix("0x") {
                Some(rest) => (16, rest),
                None => match literal.strip_prefix("0X") {
                    Some(rest) => (16, rest),
                    None => (10, literal),
                },
            },
        },
    };

    let digits: String = digits.chars().filter(|c| *c != '_').collect();
    if digits.is_empty() {
        return Err(Error::InvalidConstant {
            literal: literal.to_string(),
            reason: "literal has no digits".to_string(),
        });
    }

    match base {
        2 => resolve_binary_constant(&digits, width, literal),
        16 => resolve_hex_constant(&digits, width, literal),
        10 => resolve_decimal_constant(&digits, width, literal),
        _ => unreachable!(),
    }
}

fn resolve_binary_constant(
    digits: &str,
    width: usize,
    literal: &str,
) -> Result<Vec<ResolvedBit>, Error> {
    if digits.len() != width {
        return Err(Error::ConstantWidthMismatch {
            literal: literal.to_string(),
            width: width as u32,
            actual: digits.len(),
        });
    }

    let mut resolved = Vec::with_capacity(width);
    for ch in digits.chars().rev() {
        let bit = match ch {
            '0' => false,
            '1' => true,
            other => {
                return Err(Error::InvalidConstant {
                    literal: literal.to_string(),
                    reason: format!("`{other}` is not a binary digit"),
                })
            }
        };
        resolved.push(ResolvedBit::Const(bit));
    }
    Ok(resolved)
}

fn resolve_hex_constant(
    digits: &str,
    width: usize,
    literal: &str,
) -> Result<Vec<ResolvedBit>, Error> {
    let actual = digits.len() * 4;
    if actual != width {
        return Err(Error::ConstantWidthMismatch {
            literal: literal.to_string(),
            width: width as u32,
            actual,
        });
    }

    let mut resolved = Vec::with_capacity(width);
    for ch in digits.chars().rev() {
        let nibble = ch.to_digit(16).ok_or_else(|| Error::InvalidConstant {
            literal: literal.to_string(),
            reason: format!("`{ch}` is not a hexadecimal digit"),
        })?;

        for offset in 0..4 {
            resolved.push(ResolvedBit::Const(((nibble >> offset) & 1) == 1));
        }
    }
    Ok(resolved)
}

fn resolve_decimal_constant(
    digits: &str,
    width: usize,
    literal: &str,
) -> Result<Vec<ResolvedBit>, Error> {
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::InvalidConstant {
            literal: literal.to_string(),
            reason: "contains non-decimal digits".to_string(),
        });
    }

    let value = BigUint::parse_bytes(digits.as_bytes(), 10).ok_or_else(|| Error::InvalidConstant {
        literal: literal.to_string(),
        reason: "failed to parse decimal literal".to_string(),
    })?;

    let binary = value.to_str_radix(2);
    if binary.len() != width {
        return Err(Error::ConstantWidthMismatch {
            literal: literal.to_string(),
            width: width as u32,
            actual: binary.len(),
        });
    }

    let mut resolved = Vec::with_capacity(width);
    for ch in binary.chars().rev() {
        resolved.push(ResolvedBit::Const(ch == '1'));
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn module_with_nets(nets: &[(&str, u32)]) -> Module {
        let nets_map = nets
            .iter()
            .map(|(name, bits)| {
                (
                    (*name).to_string(),
                    Net {
                        bits: *bits,
                        attrs: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        Module {
            ports: BTreeMap::new(),
            nets: nets_map,
            nodes: BTreeMap::new(),
        }
    }

    #[test]
    fn resolves_net_slice_in_bit_order() {
        let module = module_with_nets(&[("data", 8)]);
        let bitref = BitRef::Net(BitRefNet {
            net: "data".to_string(),
            lsb: 2,
            msb: 4,
        });

        let resolved = resolve_bitref(&module, &bitref).expect("net slice should resolve");
        assert_eq!(
            resolved,
            vec![
                ResolvedBit::Net(ResolvedNetBit {
                    net: "data".to_string(),
                    bit: 2,
                }),
                ResolvedBit::Net(ResolvedNetBit {
                    net: "data".to_string(),
                    bit: 3,
                }),
                ResolvedBit::Net(ResolvedNetBit {
                    net: "data".to_string(),
                    bit: 4,
                }),
            ]
        );
    }

    #[test]
    fn resolves_binary_constant_bits() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "0b1010".to_string(),
            width: 4,
        });

        let resolved = resolve_bitref(&module, &bitref).expect("constant should resolve");
        assert_eq!(
            resolved,
            vec![
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
            ]
        );
    }

    #[test]
    fn resolves_hex_constant_bits() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "0xA5".to_string(),
            width: 8,
        });

        let resolved = resolve_bitref(&module, &bitref).expect("constant should resolve");
        assert_eq!(
            resolved,
            vec![
                ResolvedBit::Const(true),
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
                ResolvedBit::Const(false),
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
            ]
        );
    }

    #[test]
    fn resolves_decimal_constant_bits() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "10".to_string(),
            width: 4,
        });

        let resolved = resolve_bitref(&module, &bitref).expect("constant should resolve");
        assert_eq!(
            resolved,
            vec![
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
                ResolvedBit::Const(false),
                ResolvedBit::Const(true),
            ]
        );
    }

    #[test]
    fn resolves_concat_recursively() {
        let module = module_with_nets(&[("flag", 1), ("data", 4)]);
        let bitref = BitRef::Concat(BitRefConcat {
            concat: vec![
                BitRef::Net(BitRefNet {
                    net: "flag".to_string(),
                    lsb: 0,
                    msb: 0,
                }),
                BitRef::Const(BitRefConst {
                    value: "0b01".to_string(),
                    width: 2,
                }),
                BitRef::Net(BitRefNet {
                    net: "data".to_string(),
                    lsb: 1,
                    msb: 2,
                }),
            ],
        });

        let resolved = resolve_bitref(&module, &bitref).expect("concat should resolve");
        assert_eq!(
            resolved,
            vec![
                ResolvedBit::Net(ResolvedNetBit {
                    net: "flag".to_string(),
                    bit: 0,
                }),
                ResolvedBit::Const(true),
                ResolvedBit::Const(false),
                ResolvedBit::Net(ResolvedNetBit {
                    net: "data".to_string(),
                    bit: 1,
                }),
                ResolvedBit::Net(ResolvedNetBit {
                    net: "data".to_string(),
                    bit: 2,
                }),
            ]
        );
    }

    #[test]
    fn unknown_net_reports_name() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Net(BitRefNet {
            net: "missing".to_string(),
            lsb: 0,
            msb: 0,
        });

        let error = resolve_bitref(&module, &bitref).expect_err("unknown net must fail");
        assert!(matches!(error, Error::UnknownNet { net } if net == "missing"));
    }

    #[test]
    fn invalid_bit_range_reports_bounds() {
        let module = module_with_nets(&[("data", 4)]);
        let bitref = BitRef::Net(BitRefNet {
            net: "data".to_string(),
            lsb: 3,
            msb: 1,
        });

        let error = resolve_bitref(&module, &bitref).expect_err("invalid range must fail");
        assert!(matches!(
            error,
            Error::InvalidBitRange { net, lsb: 3, msb: 1 } if net == "data"
        ));
    }

    #[test]
    fn out_of_bounds_slice_reports_width() {
        let module = module_with_nets(&[("data", 4)]);
        let bitref = BitRef::Net(BitRefNet {
            net: "data".to_string(),
            lsb: 1,
            msb: 4,
        });

        let error = resolve_bitref(&module, &bitref).expect_err("oob must fail");
        assert!(matches!(
            error,
            Error::BitRangeOutOfBounds {
                net,
                lsb: 1,
                msb: 4,
                width: 4
            } if net == "data"
        ));
    }

    #[test]
    fn constant_width_mismatch_for_binary() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "0b01".to_string(),
            width: 3,
        });

        let error = resolve_bitref(&module, &bitref).expect_err("width mismatch must fail");
        assert!(matches!(
            error,
            Error::ConstantWidthMismatch {
                literal,
                width: 3,
                actual: 2
            } if literal == "0b01"
        ));
    }

    #[test]
    fn constant_width_mismatch_for_hex() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "0x1F".to_string(),
            width: 4,
        });

        let error = resolve_bitref(&module, &bitref).expect_err("width mismatch must fail");
        assert!(matches!(
            error,
            Error::ConstantWidthMismatch {
                literal,
                width: 4,
                actual: 8
            } if literal == "0x1F"
        ));
    }

    #[test]
    fn constant_width_mismatch_for_decimal() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "10".to_string(),
            width: 3,
        });

        let error = resolve_bitref(&module, &bitref).expect_err("width mismatch must fail");
        assert!(matches!(
            error,
            Error::ConstantWidthMismatch {
                literal,
                width: 3,
                actual: 4
            } if literal == "10"
        ));
    }

    #[test]
    fn decimal_with_underscores_is_supported() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Const(BitRefConst {
            value: "1_5".to_string(),
            width: 4,
        });

        let resolved = resolve_bitref(&module, &bitref).expect("constant should resolve");
        assert_eq!(
            resolved,
            vec![
                ResolvedBit::Const(true),
                ResolvedBit::Const(true),
                ResolvedBit::Const(true),
                ResolvedBit::Const(true),
            ]
        );
    }

    #[test]
    fn concat_bubbles_up_errors() {
        let module = module_with_nets(&[]);
        let bitref = BitRef::Concat(BitRefConcat {
            concat: vec![BitRef::Const(BitRefConst {
                value: "0b1".to_string(),
                width: 2,
            })],
        });

        let error = resolve_bitref(&module, &bitref).expect_err("error should bubble up");
        assert!(matches!(
            error,
            Error::ConstantWidthMismatch {
                literal,
                width: 2,
                actual: 1
            } if literal == "0b1"
        ));
    }
}
