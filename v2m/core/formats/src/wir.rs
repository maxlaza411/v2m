use crate::{
    format_validation_errors, read_and_validate, to_value_and_validate, write_validated, Error,
};
use jsonschema::{Draft, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::LazyLock;

static SCHEMA_JSON: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../schemas/wir.schema.json"))
        .expect("invalid wir schema")
});

static COMPILED_SCHEMA: LazyLock<JSONSchema> = LazyLock::new(|| {
    JSONSchema::options()
        .with_draft(Draft::Draft202012)
        .compile(&SCHEMA_JSON)
        .expect("wir schema compilation failed")
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

pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Wir, Error> {
    read_and_validate(reader, validate_json)
}

pub fn from_value(value: Value) -> Result<Wir, Error> {
    validate_json(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn to_value(wir: &Wir) -> Result<Value, Error> {
    to_value_and_validate(wir, validate_json)
}

pub fn to_writer<W: std::io::Write>(wir: &Wir, writer: W) -> Result<(), Error> {
    write_validated(wir, writer, validate_json)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wir {
    pub v: String,
    pub target: Target,
    pub ops: Vec<Op>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    RedstoneWorld,
    Datapack,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Op {
    Setblock(SetBlockOp),
    Placestructure(PlaceStructureOp),
    Writefile(WriteFileOp),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetBlockOp {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub block: String,
    pub state: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaceStructureOp {
    pub structure: String,
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WriteFileOp {
    pub path: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<WriteStage>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteStage {
    Load,
    TickRead,
    TickComb,
    TickWrite,
}
