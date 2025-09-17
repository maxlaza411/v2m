use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use v2m_formats::load_nir;
use v2m_nir::{export_dot, DotExportOptions};

#[derive(ClapArgs, Debug, Clone)]
pub struct NirDotArgs {
    /// Path to the NIR design to export
    #[arg(value_name = "NIR")]
    pub nir: PathBuf,

    /// Module to export (defaults to the design top)
    #[arg(long, value_name = "MODULE")]
    pub module: Option<String>,

    /// Output path for the GraphViz dot file
    #[arg(short = 'o', long = "output", value_name = "PATH")]
    pub output: PathBuf,

    /// Include node identifiers in the dot labels
    #[arg(long)]
    pub show_ids: bool,

    /// Include node level information in the dot labels
    #[arg(long)]
    pub show_levels: bool,
}

pub fn run(args: NirDotArgs) -> Result<()> {
    let nir = load_nir(&args.nir)
        .with_context(|| format!("failed to load NIR `{}`", args.nir.display()))?;

    let module_name = args.module.clone().unwrap_or_else(|| nir.top.clone());
    let module = nir.modules.get(&module_name).with_context(|| {
        format!(
            "module `{}` not found in design `{}`",
            module_name,
            args.nir.display()
        )
    })?;

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create parent directory `{}`", parent.display())
            })?;
        }
    }

    let options = DotExportOptions {
        show_ids: args.show_ids,
        show_levels: args.show_levels,
    };

    export_dot(module, &args.output, options).with_context(|| {
        format!(
            "failed to export module `{}` to `{}`",
            module_name,
            args.output.display()
        )
    })?;

    println!(
        "Wrote dot for module `{}` to {}",
        module_name,
        args.output.display()
    );

    Ok(())
}
