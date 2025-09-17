use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Args as ClapArgs;
use serde::Serialize;
use v2m_formats::{load_nir, nir::Module};
use v2m_nir::ModuleGraph;

#[derive(ClapArgs, Debug, Clone)]
pub struct NirStatsArgs {
    /// Path to the NIR design to analyze
    #[arg(value_name = "NIR")]
    pub nir: PathBuf,

    /// Restrict statistics to a single module
    #[arg(long, value_name = "MODULE")]
    pub module: Option<String>,

    /// Emit statistics in JSON format
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct ModuleStats {
    node_count: usize,
    combinational_node_count: usize,
    sequential_node_count: usize,
    net_count: usize,
    max_depth: usize,
    depth_histogram: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct NirStatsReport {
    design: String,
    top: String,
    modules: BTreeMap<String, ModuleStats>,
}

pub fn run(args: NirStatsArgs) -> Result<()> {
    let nir = load_nir(&args.nir)
        .with_context(|| format!("failed to load NIR `{}`", args.nir.display()))?;

    let module_entries: Vec<(String, &Module)> = if let Some(ref name) = args.module {
        let module = nir
            .modules
            .get(name)
            .with_context(|| format!("module `{}` not found in `{}`", name, args.nir.display()))?;
        vec![(name.clone(), module)]
    } else {
        nir.modules
            .iter()
            .map(|(name, module)| (name.clone(), module))
            .collect()
    };

    if module_entries.is_empty() {
        bail!("no modules available in design `{}`", args.nir.display());
    }

    let mut modules = BTreeMap::new();
    for (module_name, module) in module_entries {
        let metrics = ModuleGraph::from_module(module)
            .with_context(|| format!("failed to build module `{module_name}`"))?
            .metrics()
            .with_context(|| format!("failed to analyze module `{module_name}`"))?;

        let combinational = metrics.combinational_node_count();
        let sequential = metrics.node_count.saturating_sub(combinational);

        modules.insert(
            module_name.clone(),
            ModuleStats {
                node_count: metrics.node_count,
                combinational_node_count: combinational,
                sequential_node_count: sequential,
                net_count: metrics.net_count,
                max_depth: metrics.max_depth,
                depth_histogram: metrics.depth_histogram.clone(),
            },
        );
    }

    let report = NirStatsReport {
        design: nir.design.clone(),
        top: nir.top.clone(),
        modules,
    };

    if args.json {
        let mut stdout = io::BufWriter::new(io::stdout().lock());
        serde_json::to_writer_pretty(&mut stdout, &report)?;
        writeln!(stdout)?;
        Ok(())
    } else {
        print_text_report(&report)
    }
}

fn print_text_report(report: &NirStatsReport) -> Result<()> {
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    writeln!(stdout, "Design: {} (top: {})", report.design, report.top)?;

    for (module_name, stats) in &report.modules {
        writeln!(stdout)?;
        writeln!(stdout, "Module `{module_name}`:")?;
        writeln!(
            stdout,
            "  nodes: {} ({} combinational, {} sequential)",
            stats.node_count, stats.combinational_node_count, stats.sequential_node_count
        )?;
        writeln!(stdout, "  nets: {}", stats.net_count)?;
        writeln!(stdout, "  max_depth: {}", stats.max_depth)?;
        writeln!(stdout, "  depth_histogram: {:?}", stats.depth_histogram)?;
    }

    stdout.flush()?;
    Ok(())
}
