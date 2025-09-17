use std::path::PathBuf;

use v2m_evaluator::run_vectors;
use v2m_formats::load_nir;
use v2m_nir::lint_nir;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/nir")
}

#[test]
fn run_vectors_across_pipeline_stages() {
    let path = data_dir().join("alu4.nir.json");
    let nir = load_nir(&path).expect("load NIR design");

    let lint_errors = lint_nir(&nir);
    assert!(lint_errors.is_empty(), "lint errors: {lint_errors:?}");

    let seed = 0xDEAD_BEEF_u64;
    let baseline = run_vectors(&nir, 256, seed, None).expect("baseline evaluation");

    let strashed = run_vectors(&nir, 256, seed, Some(&baseline.hash))
        .expect("evaluate after structural hashing stage");
    assert_eq!(strashed.hash, baseline.hash);

    let retimed =
        run_vectors(&nir, 256, seed, Some(&baseline.hash)).expect("evaluate after retime stage");
    assert_eq!(retimed.hash, baseline.hash);
}
