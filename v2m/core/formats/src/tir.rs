use crate::{
    format_validation_errors, read_and_validate, to_value_and_validate, write_validated, Error,
};
use jsonschema::{Draft, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::nir::BitRef;

static SCHEMA_JSON: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../schemas/tir.schema.json"))
        .expect("invalid tir schema")
});

static COMPILED_SCHEMA: LazyLock<JSONSchema> = LazyLock::new(|| {
    JSONSchema::options()
        .with_draft(Draft::Draft202012)
        .compile(&SCHEMA_JSON)
        .expect("tir schema compilation failed")
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

pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Tir, Error> {
    read_and_validate(reader, validate_json)
}

pub fn from_value(value: Value) -> Result<Tir, Error> {
    validate_json(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn to_value(tir: &Tir) -> Result<Value, Error> {
    to_value_and_validate(tir, validate_json)
}

pub fn to_writer<W: std::io::Write>(tir: &Tir, writer: W) -> Result<(), Error> {
    write_validated(tir, writer, validate_json)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tir {
    pub v: String,
    pub lib: String,
    pub top: String,
    pub modules: BTreeMap<String, Module>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub ports: BTreeMap<String, Port>,
    pub nets: BTreeMap<String, Net>,
    pub instances: BTreeMap<String, Instance>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub dir: PortDirection,
    pub bits: u32,
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
pub struct Instance {
    pub uid: String,
    pub cell: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<BTreeMap<String, Value>>,
    pub connections: BTreeMap<String, BitRef>,
}
