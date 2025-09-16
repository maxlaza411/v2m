use crate::{
    format_validation_errors, read_and_validate, to_value_and_validate, write_validated, Error,
};
use jsonschema::{Draft, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::LazyLock;

static SCHEMA_JSON: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../schemas/constraints.schema.json"))
        .expect("invalid constraints schema")
});

static COMPILED_SCHEMA: LazyLock<JSONSchema> = LazyLock::new(|| {
    JSONSchema::options()
        .with_draft(Draft::Draft202012)
        .compile(&SCHEMA_JSON)
        .expect("constraints schema compilation failed")
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

pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Constraints, Error> {
    read_and_validate(reader, validate_json)
}

pub fn from_value(value: Value) -> Result<Constraints, Error> {
    validate_json(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn to_value(constraints: &Constraints) -> Result<Value, Error> {
    to_value_and_validate(constraints, validate_json)
}

pub fn to_writer<W: std::io::Write>(constraints: &Constraints, writer: W) -> Result<(), Error> {
    write_validated(constraints, writer, validate_json)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constraints {
    pub v: String,
    pub timing: Timing,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset: Option<Reset>,
    pub retime: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<Routing>,
    pub backend: Backend,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub clock_name: String,
    pub clock_period_ticks: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty_ticks: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fanout: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reset {
    pub name: String,
    pub polarity: ResetPolarity,
    pub kind: ResetKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_zero: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResetPolarity {
    ActiveHigh,
    ActiveLow,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResetKind {
    Sync,
    Async,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_pitch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via_cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buffer_every: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Redstone,
    Datapack,
}
