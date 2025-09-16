use crate::{
    format_validation_errors, read_and_validate, to_value_and_validate, write_validated, Error,
};
use jsonschema::{Draft, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::LazyLock;

static SCHEMA_JSON: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../schemas/pir.schema.json"))
        .expect("invalid pir schema")
});

static COMPILED_SCHEMA: LazyLock<JSONSchema> = LazyLock::new(|| {
    JSONSchema::options()
        .with_draft(Draft::Draft202012)
        .compile(&SCHEMA_JSON)
        .expect("pir schema compilation failed")
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

pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Pir, Error> {
    read_and_validate(reader, validate_json)
}

pub fn from_value(value: Value) -> Result<Pir, Error> {
    validate_json(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn to_value(pir: &Pir) -> Result<Value, Error> {
    to_value_and_validate(pir, validate_json)
}

pub fn to_writer<W: std::io::Write>(pir: &Pir, writer: W) -> Result<(), Error> {
    write_validated(pir, writer, validate_json)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pir {
    pub v: String,
    pub top: String,
    pub placements: Vec<Placement>,
    pub routes: Vec<Route>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub uid: String,
    pub inst: String,
    pub cell: String,
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub rot: Rotation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Rotation {
    R0,
    R90,
    R180,
    R270,
    Mx,
    My,
    Mz,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub uid: String,
    pub net: String,
    pub segments: Vec<RouteSegment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteSegment {
    pub kind: SegmentKind,
    pub points: Vec<Point>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<BTreeMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vias: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SegmentKind {
    Dust,
    Repeater,
    Comparator,
    TorchTower,
    Elevator,
    Bridge,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}
