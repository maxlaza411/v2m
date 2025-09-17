fn main() {
    if let Err(error) = v2m_cli::run() {
        eprintln!("{error}");
        for source in error.chain().skip(1) {
            eprintln!("  caused by: {source}");
        }
        std::process::exit(1);
    }
}
