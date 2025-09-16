use std::path::PathBuf;

use v2m_formats::load_nir;
use v2m_nir::normalize_nir;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/nir")
}

fn load_normalized(name: &str) -> v2m_nir::NormalizedNir {
    let path = data_dir().join(format!("{name}.nir.json"));
    let nir = load_nir(&path).expect("load nir design");
    normalize_nir(&nir).expect("normalize design")
}

fn assert_snapshot(name: &str) {
    let mut settings = insta::Settings::new();
    settings.set_snapshot_path(data_dir());
    settings.set_snapshot_suffix("snap.json");
    settings.set_omit_expression(true);
    settings.set_prepend_module_to_snapshot(false);
    let _guard = settings.bind_to_scope();

    let design = load_normalized(name);
    insta::assert_json_snapshot!(name, design);
}

#[test]
fn golden_fa1() {
    assert_snapshot("fa1");
}

#[test]
fn golden_alu4() {
    assert_snapshot("alu4");
}

#[test]
fn golden_reg1x8() {
    assert_snapshot("reg1x8");
}
