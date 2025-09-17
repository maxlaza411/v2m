use clap::{Parser, Subcommand};

mod eval;

pub use eval::{BinaryOutputCycle, BinaryOutputsFile, BinaryStimulusCycle, BinaryStimulusFile};

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Nir(command) => match command {
            NirCommand::Eval(args) => eval::run(args),
        },
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "v2m",
    bin_name = "v2m",
    version,
    about = "V2M toolkit",
    long_about = None,
    arg_required_else_help = true,
    disable_help_subcommand = true,
    subcommand_required = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(subcommand)]
    Nir(NirCommand),
}

#[derive(Subcommand, Debug)]
enum NirCommand {
    #[command(about = "Evaluate a NIR design")]
    Eval(eval::NirEvalArgs),
}
