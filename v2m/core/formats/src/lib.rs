use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::fmt;
use std::io::{Read, Write};

pub mod constraints;
pub mod nir;
pub mod pir;
pub mod techlib;
pub mod tir;
pub mod wir;

pub use constraints::Constraints;
pub use nir::{BitRef, Nir, load_nir, save_nir};
pub use pir::Pir;
pub use techlib::Techlib;
pub use tir::Tir;
pub use wir::Wir;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("schema validation failed:\n{0}")]
    Schema(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum AttrValue {
    String(String),
    Number(serde_json::Number),
    Bool(bool),
}

pub(crate) fn read_and_validate<T, R, F>(mut reader: R, validate: F) -> Result<T, Error>
where
    T: DeserializeOwned,
    R: Read,
    F: Fn(&Value) -> Result<(), Error>,
{
    let value: Value = serde_json::from_reader(&mut reader)?;
    validate(&value)?;
    Ok(serde_json::from_value(value)?)
}

pub(crate) fn to_value_and_validate<T, F>(value: &T, validate: F) -> Result<Value, Error>
where
    T: Serialize,
    F: Fn(&Value) -> Result<(), Error>,
{
    let json = serde_json::to_value(value)?;
    validate(&json)?;
    Ok(json)
}

pub(crate) fn write_validated<T, W, F>(value: &T, mut writer: W, validate: F) -> Result<(), Error>
where
    T: Serialize,
    W: Write,
    F: Fn(&Value) -> Result<(), Error>,
{
    let json = to_value_and_validate(value, validate)?;
    serde_json::to_writer_pretty(&mut writer, &json)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub(crate) fn format_validation_errors<'a, E>(errors: E) -> Error
where
    E: IntoIterator,
    E::Item: fmt::Display,
{
    let message = errors
        .into_iter()
        .map(|err| err.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    Error::Schema(message)
}
