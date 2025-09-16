use std::fs::File;
use std::path::PathBuf;

use insta::assert_json_snapshot;
use serde_json::Value;
use v2m_formats as formats;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

#[test]
fn constraints_roundtrip_snapshot() {
    let path = examples_dir().join("constraints/minimal.json");
    let doc = formats::constraints::from_reader(File::open(&path).expect("open constraints"))
        .expect("load constraints");
    let mut buf = Vec::new();
    formats::constraints::to_writer(&doc, &mut buf).expect("write constraints");
    let doc2 = formats::constraints::from_reader(buf.as_slice()).expect("reload constraints");
    assert_eq!(doc, doc2);
    let value: Value = serde_json::from_slice(&buf).expect("roundtrip value");
    assert_json_snapshot!("constraints_roundtrip", value);
}

#[test]
fn nir_roundtrip_snapshot() {
    let fixtures = ["minimal.json", "full_adder.json", "counter.json"];
    let base_dir = examples_dir().join("nir");

    for fixture in fixtures {
        let path = base_dir.join(fixture);
        let doc =
            formats::nir::from_reader(File::open(&path).expect("open nir")).expect("load nir");
        let mut buf = Vec::new();
        formats::nir::to_writer(&doc, &mut buf).expect("write nir");
        let doc2 = formats::nir::from_reader(buf.as_slice()).expect("reload nir");
        assert_eq!(doc, doc2);
        let value: Value = serde_json::from_slice(&buf).expect("roundtrip value");
        let snapshot_name = format!(
            "nir_roundtrip__{}",
            fixture.strip_suffix(".json").unwrap_or(fixture)
        );
        assert_json_snapshot!(snapshot_name, value);
    }
}

#[test]
fn tir_roundtrip_snapshot() {
    let path = examples_dir().join("tir/minimal.json");
    let doc = formats::tir::from_reader(File::open(&path).expect("open tir")).expect("load tir");
    let mut buf = Vec::new();
    formats::tir::to_writer(&doc, &mut buf).expect("write tir");
    let doc2 = formats::tir::from_reader(buf.as_slice()).expect("reload tir");
    assert_eq!(doc, doc2);
    let value: Value = serde_json::from_slice(&buf).expect("roundtrip value");
    assert_json_snapshot!("tir_roundtrip", value);
}

#[test]
fn pir_roundtrip_snapshot() {
    let path = examples_dir().join("pir/minimal.json");
    let doc = formats::pir::from_reader(File::open(&path).expect("open pir")).expect("load pir");
    let mut buf = Vec::new();
    formats::pir::to_writer(&doc, &mut buf).expect("write pir");
    let doc2 = formats::pir::from_reader(buf.as_slice()).expect("reload pir");
    assert_eq!(doc, doc2);
    let value: Value = serde_json::from_slice(&buf).expect("roundtrip value");
    assert_json_snapshot!("pir_roundtrip", value);
}

#[test]
fn techlib_roundtrip_snapshot() {
    let path = examples_dir().join("techlib/minimal.json");
    let doc = formats::techlib::from_reader(File::open(&path).expect("open techlib"))
        .expect("load techlib");
    let mut buf = Vec::new();
    formats::techlib::to_writer(&doc, &mut buf).expect("write techlib");
    let doc2 = formats::techlib::from_reader(buf.as_slice()).expect("reload techlib");
    assert_eq!(doc, doc2);
    let value: Value = serde_json::from_slice(&buf).expect("roundtrip value");
    assert_json_snapshot!("techlib_roundtrip", value);
}

#[test]
fn wir_roundtrip_snapshot() {
    let path = examples_dir().join("wir/minimal.json");
    let doc = formats::wir::from_reader(File::open(&path).expect("open wir")).expect("load wir");
    let mut buf = Vec::new();
    formats::wir::to_writer(&doc, &mut buf).expect("write wir");
    let doc2 = formats::wir::from_reader(buf.as_slice()).expect("reload wir");
    assert_eq!(doc, doc2);
    let value: Value = serde_json::from_slice(&buf).expect("roundtrip value");
    assert_json_snapshot!("wir_roundtrip", value);
}
