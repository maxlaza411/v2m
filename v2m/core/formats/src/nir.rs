use crate::{
    format_validation_errors, read_and_validate, to_value_and_validate, write_validated, AttrValue,
    Error,
};
use jsonschema::{Draft, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
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
