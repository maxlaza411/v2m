use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Args as ClapArgs;
use v2m_formats::load_nir;
use v2m_nir::lint_nir;

#[derive(ClapArgs, Debug, Clone)]
pub struct NirLintArgs {
    /// Path to the NIR design to lint
    #[arg(value_name = "NIR")]
    pub nir: PathBuf,
}

pub fn run(args: NirLintArgs) -> Result<()> {
    let nir = load_nir(&args.nir)
        .with_context(|| format!("failed to load NIR `{}`", args.nir.display()))?;

    let diagnostics = lint_nir(&nir);

    if diagnostics.is_empty() {
        println!(
            "Lint passed for design `{}` ({} modules checked)",
            nir.design,
            nir.modules.len()
        );
        Ok(())
    } else {
        for diagnostic in &diagnostics {
            eprintln!("{diagnostic}");
        }

        bail!("lint failed with {} error(s)", diagnostics.len());
    }
}
