use anyhow::{anyhow, ensure, Context, Result};
use std::{
    fmt::Write as _,
    fs::{self, File},
    io::Write as IoWrite,
    path::{Path, PathBuf},
    process::Command,
};

pub struct YosysCfg {
    pub top: String,
    pub srcs: Vec<String>,
    pub incs: Vec<String>,
    pub defines: Vec<(String, String)>,
    pub out_json: String,
    pub log_path: String,
    pub use_container: bool,
    pub container_ref: Option<String>,
}

pub fn run_yosys(cfg: &YosysCfg) -> Result<()> {
    let workspace = fs::canonicalize(std::env::current_dir()?)
        .context("failed to resolve workspace directory")?;

    let script_path = workspace.join("build/yosys/run.ys");
    if let Some(parent) = script_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create Yosys script directory {}",
                parent.display()
            )
        })?;
    }

    let log_path = make_absolute(&workspace, &cfg.log_path);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create Yosys log directory {}", parent.display())
        })?;
    }

    let out_json_path = make_absolute(&workspace, &cfg.out_json);
    if let Some(parent) = out_json_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create Yosys output directory {}",
                parent.display()
            )
        })?;
    }

    let srcs = cfg
        .srcs
        .iter()
        .map(|src| make_existing_absolute(&workspace, src))
        .collect::<Result<Vec<_>>>()?;
    let incs = cfg
        .incs
        .iter()
        .map(|inc| make_existing_absolute(&workspace, inc))
        .collect::<Result<Vec<_>>>()?;

    let script = render_script(&cfg.top, &srcs, &incs, &cfg.defines, &out_json_path)?;
    fs::write(&script_path, script)
        .with_context(|| format!("failed to write Yosys script to {}", script_path.display()))?;

    let workspace_str = path_to_string(&workspace)?;
    let script_str = path_to_string(&script_path)?;

    let mut command = if cfg.use_container {
        let image = cfg
            .container_ref
            .as_deref()
            .ok_or_else(|| anyhow!("container_ref must be provided when use_container is true"))?;

        let mut cmd = Command::new("docker");
        cmd.arg("run")
            .arg("--rm")
            .arg("--workdir")
            .arg(&workspace_str)
            .arg("-v")
            .arg(format!("{0}:{0}", workspace_str))
            .arg(image)
            .arg("yosys")
            .arg("-s")
            .arg(&script_str);
        cmd
    } else {
        let mut cmd = Command::new("yosys");
        cmd.arg("-s").arg(&script_str);
        cmd.current_dir(&workspace);
        cmd
    };

    let output = command
        .output()
        .with_context(|| "failed to execute Yosys command".to_string())?;

    let mut log = File::create(&log_path)
        .with_context(|| format!("failed to create Yosys log at {}", log_path.display()))?;
    log.write_all(&output.stdout)
        .with_context(|| format!("failed to write Yosys stdout to {}", log_path.display()))?;
    log.write_all(&output.stderr)
        .with_context(|| format!("failed to write Yosys stderr to {}", log_path.display()))?;
    log.flush()
        .with_context(|| format!("failed to flush Yosys log {}", log_path.display()))?;

    ensure!(
        output.status.success(),
        "Yosys failed, see {}",
        log_path.display()
    );

    Ok(())
}

fn render_script(
    top: &str,
    srcs: &[PathBuf],
    incs: &[PathBuf],
    defines: &[(String, String)],
    out_json: &Path,
) -> Result<String> {
    let mut read_cmd = vec![
        "read_verilog".to_string(),
        "-sv".to_string(),
        "-defer".to_string(),
    ];

    for inc in incs {
        read_cmd.push("-I".to_string());
        read_cmd.push(quote_path(inc)?);
    }

    for (name, value) in defines {
        if value.is_empty() {
            read_cmd.push(format!("-D{}", name));
        } else {
            read_cmd.push(format!("-D{}={}", name, value));
        }
    }

    for src in srcs {
        read_cmd.push(quote_path(src)?);
    }

    let mut script = String::new();
    writeln!(&mut script, "{}", read_cmd.join(" "))?;
    writeln!(&mut script, "hierarchy -check -top {}", top)?;
    writeln!(&mut script, "proc")?;
    writeln!(&mut script, "opt")?;
    writeln!(&mut script, "wreduce")?;
    writeln!(&mut script, "alumacc")?;
    writeln!(&mut script, "opt")?;
    writeln!(&mut script, "memory -nomap")?;
    writeln!(&mut script, "memory_dff")?;
    writeln!(&mut script, "memory_map")?;
    writeln!(&mut script, "opt")?;
    writeln!(&mut script, "techmap -map +/cmp2.v")?;
    writeln!(&mut script, "opt_clean")?;
    writeln!(&mut script, "write_json -o {}", quote_path(out_json)?)?;

    Ok(script)
}

fn quote_path(path: &Path) -> Result<String> {
    let raw = path_to_string(path)?;
    Ok(quote_string(&raw))
}

fn quote_string(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => quoted.push_str("\\\\"),
            '"' => quoted.push_str("\\\""),
            _ => quoted.push(ch),
        }
    }
    quoted.push('"');
    quoted
}

fn path_to_string(path: &Path) -> Result<String> {
    path.to_str()
        .map(|s| s.to_owned())
        .ok_or_else(|| anyhow!("path contains invalid unicode: {}", path.display()))
}

fn make_existing_absolute(root: &Path, raw: &str) -> Result<PathBuf> {
    let path = make_absolute(root, raw);
    fs::canonicalize(&path).with_context(|| {
        format!(
            "failed to canonicalize path {} (from input {})",
            path.display(),
            raw
        )
    })
}

fn make_absolute(root: &Path, raw: &str) -> PathBuf {
    let candidate = Path::new(raw);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    }
}
