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
    serde_json::from_str(include_str!("../../../schemas/techlib.schema.json"))
        .expect("invalid techlib schema")
});

static COMPILED_SCHEMA: LazyLock<JSONSchema> = LazyLock::new(|| {
    JSONSchema::options()
        .with_draft(Draft::Draft202012)
        .compile(&SCHEMA_JSON)
        .expect("techlib schema compilation failed")
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

pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Techlib, Error> {
    read_and_validate(reader, validate_json)
}

pub fn from_value(value: Value) -> Result<Techlib, Error> {
    validate_json(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub fn to_value(techlib: &Techlib) -> Result<Value, Error> {
    to_value_and_validate(techlib, validate_json)
}

pub fn to_writer<W: std::io::Write>(techlib: &Techlib, writer: W) -> Result<(), Error> {
    write_validated(techlib, writer, validate_json)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Techlib {
    pub v: String,
    pub name: String,
    pub minecraft: Minecraft,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics: Option<Physics>,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Minecraft {
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edition: Option<MinecraftEdition>,
    pub coords: MinecraftCoords,
    pub orientations: Vec<MinecraftOrientation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MinecraftEdition {
    Java,
    Bedrock,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MinecraftCoords {
    #[serde(rename = "x-east_y-up_z-south")]
    EastUpSouth,
    #[serde(rename = "x-east_y-up_z-north")]
    EastUpNorth,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MinecraftOrientation {
    North,
    South,
    East,
    West,
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Physics {
    pub tick: f64,
    pub power_levels: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_ticks: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via_ticks: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub name: String,
    pub kind: CellKind,
    pub pins: Vec<CellPin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<BTreeMap<String, Value>>,
    pub delays: Vec<DelayArc>,
    pub footprint: Footprint,
    pub rotations: Vec<CellRotation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CellKind {
    Comb,
    Seq,
    Wire,
    Via,
    Io,
    Macro,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellPin {
    pub name: String,
    pub dir: PinDirection,
    pub bitwidth: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PinDirection {
    In,
    Out,
    Inout,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayArc {
    pub from: String,
    pub to: String,
    pub ticks: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Footprint {
    pub size: FootprintSize,
    pub voxels: Vec<Voxel>,
    pub connectors: Vec<Connector>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keepouts: Vec<Keepout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rot_overrides: Option<BTreeMap<String, FootprintOverride>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FootprintSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voxel {
    pub block: String,
    pub state: BTreeMap<String, AttrValue>,
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connector {
    pub pin: String,
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub facing: Facing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Facing {
    North,
    South,
    East,
    West,
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Keepout {
    pub x1: i64,
    pub y1: i64,
    pub z1: i64,
    pub x2: i64,
    pub y2: i64,
    pub z2: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inclusive: Option<KeepoutInclusive>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KeepoutInclusive {
    Minmax,
    MinOnly,
    MaxOnly,
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CellRotation {
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
pub struct FootprintOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connectors: Option<Vec<Connector>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voxels: Option<Vec<Voxel>>,
}
