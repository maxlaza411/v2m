use std::fs::File;
use std::path::PathBuf;

use serde_json::Value;
use v2m_formats as formats;
use walkdir::WalkDir;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn validate_example(format: &str, value: Value) -> Result<(), formats::Error> {
    match format {
        "constraints" => formats::constraints::from_value(value).map(|_| ()),
        "nir" => formats::nir::from_value(value).map(|_| ()),
        "tir" => formats::tir::from_value(value).map(|_| ()),
        "pir" => formats::pir::from_value(value).map(|_| ()),
        "techlib" => formats::techlib::from_value(value).map(|_| ()),
        "wir" => formats::wir::from_value(value).map(|_| ()),
        other => panic!("unexpected format directory: {other}"),
    }
}

#[test]
fn examples_reject_unknown_root_fields() {
    for entry in WalkDir::new(examples_dir())
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .map(|ext| ext == "json")
                    .unwrap_or(false)
        })
    {
        let path = entry.path();
        let format = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .expect("example path format");

        let value: Value = serde_json::from_reader(File::open(path).expect("open example"))
            .expect("parse example");
        validate_example(format, value.clone()).expect("examples must validate");

        let Value::Object(mut obj) = value else {
            panic!("example {path:?} must be an object");
        };
        let valid_clone = Value::Object(obj.clone());
        validate_example(format, valid_clone).expect("validation should succeed");

        obj.insert("__typo__".to_string(), Value::Bool(true));
        let invalid = Value::Object(obj);
        assert!(
            validate_example(format, invalid).is_err(),
            "format {format} accepted unknown fields"
        );
    }
}
