use std::{fs, path::PathBuf};

use yosys_bridge::{load_rtlil_json, LoaderOptions};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn loads_full_adder_fixture() {
    let rtlil = load_rtlil_json(fixture_path("full_adder.json"), &LoaderOptions::default())
        .expect("load full_adder.json");

    assert_eq!(rtlil.top, "full_adder");
    assert!(rtlil.modules.contains_key("full_adder"));
}

#[test]
fn errors_when_modules_missing() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("missing_modules.json");
    fs::write(&path, "{\"creator\": \"yosys\"}").expect("write temp json");

    let error = load_rtlil_json(&path, &LoaderOptions::default()).expect_err("expect error");
    assert!(
        error.to_string().contains("modules"),
        "error did not mention missing modules: {error:?}"
    );
}

#[test]
fn detects_remaining_memory_cells() {
    let error = load_rtlil_json(fixture_path("has_mem.json"), &LoaderOptions::default())
        .expect_err("expect error when memory cells remain");

    let message = error.to_string();
    assert!(
        message.contains("mem_inst"),
        "error missing cell name: {message}"
    );
}

#[test]
fn allow_mem_blackbox_bypasses_memory_check() {
    let mut options = LoaderOptions::default();
    options.allow_mem_blackbox = true;

    let rtlil = load_rtlil_json(fixture_path("has_mem.json"), &options)
        .expect("allow_mem_blackbox should permit memory cells");
    assert_eq!(rtlil.top, "memory_block");
}

#[test]
fn errors_when_top_module_missing() {
    let error = load_rtlil_json(fixture_path("missing_top.json"), &LoaderOptions::default())
        .expect_err("expect top module error");

    assert!(
        error.to_string().to_lowercase().contains("top"),
        "error should mention top module"
    );
}
