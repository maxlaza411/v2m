use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::{NamedTempFile, TempDir};
use v2m_cli::{BinaryOutputCycle, BinaryOutputsFile, BinaryStimulusCycle, BinaryStimulusFile};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("cli crate should have parent")
        .to_path_buf()
}

fn cli_data(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn golden(path: &str) -> PathBuf {
    repo_root().join(path)
}

#[test]
fn eval_full_adder_matches_json_golden() {
    let design = golden("tests/golden/nir/fa1.nir.json");
    let inputs = cli_data("tests/data/nir/fa1.inputs.json");
    let golden_outputs = cli_data("tests/data/nir/fa1.expected.json");

    Command::cargo_bin("v2m")
        .expect("binary exists")
        .args([
            "nir",
            "eval",
            "--nir",
            design.to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--out",
            golden_outputs.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"design\": \"fa1\""));
}

#[test]
fn eval_accepts_binary_inputs_and_outputs() {
    let design = golden("tests/golden/nir/fa1.nir.json");
    let temp = TempDir::new().expect("temp dir");
    let inputs_path = temp.path().join("inputs.bin");
    let outputs_path = temp.path().join("outputs.bin");

    write_binary_inputs(&inputs_path).expect("write binary inputs");
    write_binary_outputs(&outputs_path).expect("write binary outputs");

    Command::cargo_bin("v2m")
        .expect("binary exists")
        .args([
            "nir",
            "eval",
            "--nir",
            design.to_str().unwrap(),
            "--inputs",
            &format!("bin:{}", inputs_path.to_str().unwrap()),
            "--out",
            &format!("bin:{}", outputs_path.to_str().unwrap()),
            "--vectors",
            "4",
        ])
        .assert()
        .success();
}

#[test]
fn eval_reports_golden_mismatch() {
    let design = golden("tests/golden/nir/fa1.nir.json");
    let inputs = cli_data("tests/data/nir/fa1.inputs.json");

    let mut wrong = NamedTempFile::new().expect("temp file");
    writeln!(
        wrong,
        "{{\n  \"design\": \"fa1\",\n  \"top\": \"fa1\",\n  \"vectors\": 4,\n  \"cycles\": [{{\n    \"index\": 0,\n    \"outputs\": {{\n      \"cout\": [\"0x0\", \"0x0\", \"0x1\", \"0x1\"],\n      \"sum\": [\"0x0\", \"0x1\", \"0x0\", \"0x0\"]\n    }}\n  }}]\n}}"
    )
    .expect("write temp");
    wrong.flush().expect("flush temp");

    Command::cargo_bin("v2m")
        .expect("binary exists")
        .args([
            "nir",
            "eval",
            "--nir",
            design.to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--out",
            wrong.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("output mismatch"));
}

#[test]
fn eval_profile_flag_prints_metrics() {
    let design = golden("tests/golden/nir/fa1.nir.json");
    let inputs = cli_data("tests/data/nir/fa1.inputs.json");

    Command::cargo_bin("v2m")
        .expect("binary exists")
        .args([
            "nir",
            "eval",
            "--nir",
            design.to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--profile",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Profile summary:"))
        .stdout(predicate::str::contains("Kernel breakdown:"));
}

#[test]
fn eval_profile_json_is_written() {
    let design = golden("tests/golden/nir/fa1.nir.json");
    let inputs = cli_data("tests/data/nir/fa1.inputs.json");
    let temp = TempDir::new().expect("temp dir");
    let profile_path = temp.path().join("profile.json");

    Command::cargo_bin("v2m")
        .expect("binary exists")
        .args([
            "nir",
            "eval",
            "--nir",
            design.to_str().unwrap(),
            "--inputs",
            inputs.to_str().unwrap(),
            "--profile-json",
            profile_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    let contents = fs::read_to_string(&profile_path).expect("read profile");
    assert!(contents.contains("\"kernels\""));
}

fn write_binary_inputs(path: &Path) -> anyhow::Result<()> {
    let mut inputs = BTreeMap::new();
    inputs.insert("a".to_string(), vec![vec![0], vec![0], vec![1], vec![1]]);
    inputs.insert("b".to_string(), vec![vec![0], vec![1], vec![1], vec![1]]);
    inputs.insert("cin".to_string(), vec![vec![0], vec![0], vec![0], vec![1]]);

    let file = BinaryStimulusFile {
        cycles: vec![BinaryStimulusCycle { inputs }],
    };

    let writer = fs::File::create(path)?;
    bincode::serialize_into(writer, &file)?;
    Ok(())
}

fn write_binary_outputs(path: &Path) -> anyhow::Result<()> {
    let mut outputs = BTreeMap::new();
    outputs.insert("cout".to_string(), vec![vec![0], vec![0], vec![1], vec![1]]);
    outputs.insert("sum".to_string(), vec![vec![0], vec![1], vec![0], vec![1]]);

    let file = BinaryOutputsFile {
        design: "fa1".to_string(),
        top: "fa1".to_string(),
        vectors: 4,
        cycles: vec![BinaryOutputCycle { index: 0, outputs }],
    };

    let writer = fs::File::create(path)?;
    bincode::serialize_into(writer, &file)?;
    Ok(())
}
