use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args as ClapArgs;
use num_bigint::BigUint;
use num_traits::{Num, Zero};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use v2m_evaluator::{div_ceil, Evaluator, PackedBitMask, ProfileReport, SimOptions};
use v2m_formats::load_nir;
use v2m_formats::nir::{Module, PortDirection};

#[derive(ClapArgs, Debug, Clone)]
pub struct NirEvalArgs {
    /// Path to the NIR design to evaluate
    #[arg(value_name = "NIR")]
    pub nir: PathBuf,

    /// Stimulus input file (json:path or bin:path)
    #[arg(long = "vec", alias = "inputs", value_name = "SPEC")]
    pub inputs: Option<FormatPath>,

    /// Number of stimulus vectors per cycle
    #[arg(long, value_name = "N")]
    pub vectors: Option<usize>,

    /// Seed used for random stimulus or reset generation
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,

    /// Number of cycles to evaluate
    #[arg(long, value_name = "CYCLES")]
    pub cycles: Option<usize>,

    /// Reset pattern applied per cycle (e.g. 100 or 1,0,0)
    #[arg(long, value_name = "PATTERN")]
    pub reset_pattern: Option<String>,

    /// Path used to compare or capture outputs (json:path or bin:path)
    #[arg(long, value_name = "SPEC")]
    pub out: Option<FormatPath>,

    /// Enable kernel profiling and print metrics to stdout
    #[arg(long)]
    pub profile: bool,

    /// Export profile metrics to a JSON file
    #[arg(long, value_name = "PATH")]
    pub profile_json: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Json,
    Bin,
}

impl DataFormat {
    fn from_extension(path: &Path) -> Option<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("json") => Some(DataFormat::Json),
            Some(ext) if ext.eq_ignore_ascii_case("bin") || ext.eq_ignore_ascii_case("bincode") => {
                Some(DataFormat::Bin)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FormatPath {
    pub format: Option<DataFormat>,
    pub path: PathBuf,
}

impl std::str::FromStr for FormatPath {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (format, path) = if let Some((prefix, rest)) = value.split_once(':') {
            let format = prefix.parse()?;
            (Some(format), PathBuf::from(rest))
        } else if let Some((prefix, rest)) = value.split_once('=') {
            let format = prefix.parse()?;
            (Some(format), PathBuf::from(rest))
        } else {
            (None, PathBuf::from(value))
        };

        Ok(Self { format, path })
    }
}

impl std::str::FromStr for DataFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Ok(DataFormat::Json),
            "bin" | "binary" | "bincode" => Ok(DataFormat::Bin),
            other => bail!("unsupported format `{other}`"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalOutputs {
    pub design: String,
    pub top: String,
    pub vectors: usize,
    pub cycles: Vec<OutputCycle>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputCycle {
    pub index: usize,
    pub outputs: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryStimulusFile {
    pub cycles: Vec<BinaryStimulusCycle>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryStimulusCycle {
    pub inputs: BTreeMap<String, Vec<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryOutputsFile {
    pub design: String,
    pub top: String,
    pub vectors: usize,
    pub cycles: Vec<BinaryOutputCycle>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryOutputCycle {
    pub index: usize,
    pub outputs: BTreeMap<String, Vec<Vec<u8>>>,
}

pub fn run(args: NirEvalArgs) -> Result<()> {
    let nir = load_nir(&args.nir)
        .with_context(|| format!("failed to load NIR `{}`", args.nir.display()))?;

    let profile_flag = args.profile;
    let profile_json_path = args.profile_json.clone();
    let profiling_enabled = profile_flag || profile_json_path.is_some();

    let design_module = nir.modules.get(&nir.top).with_context(|| {
        format!(
            "top module `{}` not found in `{}`",
            nir.top,
            args.nir.display()
        )
    })?;

    let mut rng = StdRng::seed_from_u64(args.seed.unwrap_or(0));

    let (mut input_cycles, inferred_vectors) = if let Some(ref spec) = args.inputs {
        let (format, path) = resolve_format_path(spec, DataFormat::Json)?;
        let cycles = load_input_cycles(&path, format)?;
        let vectors = infer_vectors(&cycles);
        (cycles, vectors)
    } else {
        (Vec::new(), None)
    };

    let num_vectors = match (args.vectors, inferred_vectors) {
        (Some(value), Some(inferred)) if value != inferred => {
            bail!("requested {value} vectors but input file provides {inferred} vectors")
        }
        (Some(value), _) => value,
        (None, Some(inferred)) => inferred,
        (None, None) => 1,
    };

    if num_vectors == 0 {
        bail!("vector count must be greater than zero");
    }

    let requested_cycles = args.cycles.or_else(|| {
        if !input_cycles.is_empty() {
            Some(input_cycles.len())
        } else {
            None
        }
    });
    let total_cycles = requested_cycles.unwrap_or(1);

    if input_cycles.is_empty() {
        input_cycles = generate_random_inputs(design_module, total_cycles, num_vectors, &mut rng);
    } else {
        adjust_cycle_count(&mut input_cycles, total_cycles)?;
    }

    validate_input_vectors(&input_cycles, design_module, num_vectors)?;

    let reset_masks = build_reset_masks(
        args.reset_pattern.as_deref(),
        total_cycles,
        num_vectors,
        &mut rng,
    )?;

    let mut evaluator = Evaluator::new(&nir, num_vectors, SimOptions::default())
        .context("failed to build evaluator")?;

    if profiling_enabled {
        evaluator.enable_profiling();
    }

    let mut output_cycles = Vec::with_capacity(total_cycles);

    for (cycle_idx, cycle_inputs) in input_cycles.iter().enumerate().take(total_cycles) {
        let packed_inputs = evaluator
            .pack_inputs_from_biguints(cycle_inputs)
            .context("failed to pack input values")?;
        let outputs = evaluator
            .tick(&packed_inputs, &reset_masks[cycle_idx])
            .context("evaluator tick failed")?;
        let outputs_map = evaluator
            .unpack_outputs_to_biguints(&outputs)
            .context("failed to unpack outputs")?;

        output_cycles.push(OutputCycle {
            index: cycle_idx,
            outputs: format_output_values(design_module, outputs_map)?,
        });
    }

    let eval_outputs = EvalOutputs {
        design: nir.design.clone(),
        top: nir.top.clone(),
        vectors: num_vectors,
        cycles: output_cycles,
    };

    print_outputs(&eval_outputs)?;

    if let Some(spec) = args.out {
        verify_or_write_outputs(&spec, &eval_outputs)?;
    }

    if profiling_enabled {
        if let Some(report) = evaluator.profile_report() {
            if profile_flag {
                print_profile_report(&report);
            }
            if let Some(path) = profile_json_path.as_ref() {
                write_profile_json(path, &report)?;
            }
        }
    }

    Ok(())
}

fn resolve_format_path(spec: &FormatPath, default: DataFormat) -> Result<(DataFormat, PathBuf)> {
    let format = spec
        .format
        .or_else(|| DataFormat::from_extension(&spec.path))
        .unwrap_or(default);
    Ok((format, spec.path.clone()))
}

fn load_input_cycles(
    path: &Path,
    format: DataFormat,
) -> Result<Vec<HashMap<String, Vec<BigUint>>>> {
    match format {
        DataFormat::Json => load_input_cycles_json(path),
        DataFormat::Bin => load_input_cycles_bin(path),
    }
}

fn load_input_cycles_json(path: &Path) -> Result<Vec<HashMap<String, Vec<BigUint>>>> {
    let reader =
        File::open(path).with_context(|| format!("failed to open inputs `{}`", path.display()))?;
    let value: Value = serde_json::from_reader(reader)
        .with_context(|| format!("failed to parse inputs `{}`", path.display()))?;
    parse_inputs_value(&value)
}

fn load_input_cycles_bin(path: &Path) -> Result<Vec<HashMap<String, Vec<BigUint>>>> {
    let reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open inputs `{}`", path.display()))?,
    );
    let file: BinaryStimulusFile = bincode::deserialize_from(reader)
        .with_context(|| format!("failed to parse binary inputs `{}`", path.display()))?;
    let mut result = Vec::with_capacity(file.cycles.len());
    for cycle in file.cycles {
        let mut map = HashMap::new();
        for (name, vectors) in cycle.inputs {
            let values = vectors
                .into_iter()
                .map(|bytes| BigUint::from_bytes_le(&bytes))
                .collect();
            map.insert(name, values);
        }
        result.push(map);
    }
    Ok(result)
}

fn parse_inputs_value(value: &Value) -> Result<Vec<HashMap<String, Vec<BigUint>>>> {
    match value {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                parse_cycle_value(item).with_context(|| format!("invalid inputs for cycle {index}"))
            })
            .collect(),
        Value::Object(map) => {
            if let Some(cycles) = map.get("cycles") {
                parse_inputs_value(cycles)
            } else {
                let cycle = parse_cycle_map(map)?;
                Ok(vec![cycle])
            }
        }
        other => bail!("expected inputs to be an array or object, found {other:?}"),
    }
}

fn parse_cycle_value(value: &Value) -> Result<HashMap<String, Vec<BigUint>>> {
    let map = value
        .as_object()
        .ok_or_else(|| anyhow!("cycle entries must be objects"))?;
    parse_cycle_map(map)
}

fn parse_cycle_map(map: &Map<String, Value>) -> Result<HashMap<String, Vec<BigUint>>> {
    let mut result = HashMap::new();
    for (port, entry) in map {
        let values = parse_port_values(entry)
            .with_context(|| format!("invalid values for port `{port}`"))?;
        result.insert(port.clone(), values);
    }
    Ok(result)
}

fn parse_port_values(value: &Value) -> Result<Vec<BigUint>> {
    match value {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                parse_biguint(item).with_context(|| format!("invalid value at index {index}"))
            })
            .collect(),
        other => bail!("port values must be arrays, found {other:?}"),
    }
}

fn parse_biguint(value: &Value) -> Result<BigUint> {
    match value {
        Value::String(s) => parse_biguint_str(s),
        Value::Number(num) => parse_biguint_str(&num.to_string()),
        other => bail!("values must be numbers or strings, found {other:?}"),
    }
}

fn parse_biguint_str(text: &str) -> Result<BigUint> {
    let cleaned: String = text
        .chars()
        .filter(|c| *c != '_' && !c.is_whitespace())
        .collect();
    let (radix, digits) = if let Some(rest) = cleaned.strip_prefix("0x") {
        (16, rest)
    } else if let Some(rest) = cleaned.strip_prefix("0b") {
        (2, rest)
    } else if let Some(rest) = cleaned.strip_prefix("0o") {
        (8, rest)
    } else {
        (10, cleaned.as_str())
    };

    BigUint::from_str_radix(digits, radix)
        .with_context(|| format!("invalid integer literal `{text}`"))
}

fn infer_vectors(cycles: &[HashMap<String, Vec<BigUint>>]) -> Option<usize> {
    for cycle in cycles {
        if let Some(values) = cycle.values().next() {
            return Some(values.len());
        }
    }
    None
}

fn adjust_cycle_count(
    cycles: &mut Vec<HashMap<String, Vec<BigUint>>>,
    total_cycles: usize,
) -> Result<()> {
    if cycles.is_empty() {
        return Ok(());
    }

    if total_cycles == 0 {
        cycles.clear();
        return Ok(());
    }

    if cycles.len() == total_cycles {
        return Ok(());
    }

    if cycles.len() == 1 {
        let single = cycles[0].clone();
        cycles.resize_with(total_cycles, || single.clone());
        return Ok(());
    }

    bail!(
        "input file contains {} cycles but {total_cycles} cycles were requested",
        cycles.len()
    );
}

fn generate_random_inputs(
    module: &Module,
    total_cycles: usize,
    num_vectors: usize,
    rng: &mut StdRng,
) -> Vec<HashMap<String, Vec<BigUint>>> {
    let mut cycles = Vec::with_capacity(total_cycles);
    for _ in 0..total_cycles {
        let mut map = HashMap::new();
        for (name, port) in &module.ports {
            if !matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
                continue;
            }
            let width = port.bits as usize;
            let mut values = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                values.push(random_biguint(width, rng));
            }
            map.insert(name.clone(), values);
        }
        cycles.push(map);
    }
    cycles
}

fn random_biguint(width_bits: usize, rng: &mut StdRng) -> BigUint {
    if width_bits == 0 {
        return BigUint::default();
    }

    let byte_len = div_ceil(width_bits, 8);
    let mut bytes = vec![0u8; byte_len];
    rng.fill_bytes(&mut bytes);

    let excess_bits = byte_len * 8 - width_bits;
    if excess_bits > 0 {
        let keep = 8 - excess_bits;
        let mask = if keep == 0 {
            0
        } else {
            (1u16 << keep) as u8 - 1
        };
        if let Some(last) = bytes.last_mut() {
            *last &= mask;
        }
    }

    BigUint::from_bytes_le(&bytes)
}

fn validate_input_vectors(
    cycles: &[HashMap<String, Vec<BigUint>>],
    module: &Module,
    num_vectors: usize,
) -> Result<()> {
    for (cycle_idx, cycle) in cycles.iter().enumerate() {
        for (name, values) in cycle {
            if values.len() != num_vectors {
                bail!(
                    "port `{name}` cycle {cycle_idx} expects {num_vectors} vectors but found {}",
                    values.len()
                );
            }
            let port = module
                .ports
                .get(name)
                .with_context(|| format!("unknown port `{name}` in stimulus"))?;
            if !matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
                bail!("port `{name}` is not an input");
            }
            let width = port.bits as usize;
            for (vec_idx, value) in values.iter().enumerate() {
                if value.bits() > width as u64 {
                    bail!(
                        "value for port `{name}` cycle {cycle_idx} vector {vec_idx} exceeds width {width} bits"
                    );
                }
            }
        }
    }
    Ok(())
}

fn build_reset_masks(
    pattern: Option<&str>,
    cycles: usize,
    num_vectors: usize,
    rng: &mut StdRng,
) -> Result<Vec<PackedBitMask>> {
    if cycles == 0 {
        return Ok(Vec::new());
    }

    let tokens = parse_reset_pattern(pattern)?;
    let mut masks = Vec::with_capacity(cycles);
    let token_count = tokens.len();
    for cycle in 0..cycles {
        let token = tokens[cycle % token_count];

        let mut mask = PackedBitMask::new(num_vectors);
        match token {
            ResetToken::Zero => {}
            ResetToken::One => fill_mask_all(&mut mask, num_vectors),
            ResetToken::Random => fill_mask_random(&mut mask, num_vectors, rng),
        }
        masks.push(mask);
    }
    Ok(masks)
}

#[derive(Clone, Copy, Debug)]
enum ResetToken {
    Zero,
    One,
    Random,
}

fn parse_reset_pattern(pattern: Option<&str>) -> Result<Vec<ResetToken>> {
    let Some(pattern) = pattern else {
        return Ok(vec![ResetToken::Zero]);
    };

    let mut tokens = Vec::new();
    for ch in pattern.chars() {
        match ch {
            '0' => tokens.push(ResetToken::Zero),
            '1' => tokens.push(ResetToken::One),
            'r' | 'R' => tokens.push(ResetToken::Random),
            '_' | ',' | ' ' | '\t' | '\n' => continue,
            other => {
                return Err(anyhow!("invalid character `{other}` in reset pattern"));
            }
        }
    }
    if tokens.is_empty() {
        Ok(vec![ResetToken::Zero])
    } else {
        Ok(tokens)
    }
}

fn fill_mask_all(mask: &mut PackedBitMask, num_vectors: usize) {
    let words = mask.words_mut();
    if words.is_empty() {
        return;
    }

    let word_count = words.len();
    for (idx, word) in words.iter_mut().enumerate() {
        *word = if idx == word_count - 1 {
            final_word_mask(num_vectors % 64)
        } else {
            u64::MAX
        };
    }
}

fn fill_mask_random(mask: &mut PackedBitMask, num_vectors: usize, rng: &mut StdRng) {
    let total_words = mask.words().len();
    if total_words == 0 {
        return;
    }

    for (idx, word) in mask.words_mut().iter_mut().enumerate() {
        let mut value = rng.next_u64();
        if idx == total_words - 1 {
            value &= final_word_mask(num_vectors % 64);
        }
        *word = value;
    }
}

fn final_word_mask(remainder: usize) -> u64 {
    if remainder == 0 {
        u64::MAX
    } else {
        (1u64 << remainder) - 1
    }
}

fn format_output_values(
    module: &Module,
    outputs: HashMap<String, Vec<BigUint>>,
) -> Result<BTreeMap<String, Vec<String>>> {
    let mut map = BTreeMap::new();
    for (name, values) in outputs {
        let port = module
            .ports
            .get(&name)
            .with_context(|| format!("unknown output port `{name}`"))?;
        if !matches!(port.dir, PortDirection::Output | PortDirection::Inout) {
            bail!("port `{name}` is not an output");
        }
        let formatted: Vec<String> = values.iter().map(|value| format_biguint(value)).collect();
        map.insert(name, formatted);
    }
    Ok(map)
}

fn format_biguint(value: &BigUint) -> String {
    if value.is_zero() {
        "0x0".to_string()
    } else {
        format!("0x{}", value.to_str_radix(16))
    }
}

fn print_outputs(outputs: &EvalOutputs) -> Result<()> {
    let json = serde_json::to_string_pretty(outputs)?;
    println!("{json}");
    Ok(())
}

fn print_profile_report(report: &ProfileReport) {
    println!();
    println!("Profile summary:");
    println!("  total kernels: {}", report.total_ops);
    println!("  total bytes moved: {}", format_bytes(report.total_bytes));
    println!(
        "  total kernel time: {}",
        format_duration(report.total_time_ns)
    );

    if report.kernels.is_empty() {
        println!("  no combinational kernels executed");
        return;
    }

    println!();
    println!("Kernel breakdown:");
    for entry in &report.kernels {
        let share = if report.total_time_ns > 0 {
            (entry.total_time_ns as f64 / report.total_time_ns as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  - {:<6} ops={:<5} bytes={} time={} avg={} share={:.1}%",
            entry.kind,
            entry.ops,
            format_bytes(entry.bytes_moved),
            format_duration(entry.total_time_ns),
            format_duration(entry.average_time_ns),
            share
        );
    }
}

fn write_profile_json(path: &Path, report: &ProfileReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }
    }

    let writer = BufWriter::new(
        File::create(path).with_context(|| format!("failed to create `{}`", path.display()))?,
    );
    serde_json::to_writer_pretty(writer, report)?;
    Ok(())
}

fn format_duration(ns: u128) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.3} s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.3} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.3} µs", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn verify_or_write_outputs(spec: &FormatPath, outputs: &EvalOutputs) -> Result<()> {
    let (format, path) = resolve_format_path(spec, DataFormat::Json)?;
    if path.exists() {
        let expected = load_outputs(&path, format)?;
        if let Err(diff) = compare_outputs(&expected, outputs) {
            bail!("output mismatch vs `{}`: {diff}", path.display());
        }
        Ok(())
    } else {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create `{}`", parent.display()))?;
            }
        }
        write_outputs(&path, format, outputs)
    }
}

fn load_outputs(path: &Path, format: DataFormat) -> Result<EvalOutputs> {
    match format {
        DataFormat::Json => {
            let reader = File::open(path)
                .with_context(|| format!("failed to open golden `{}`", path.display()))?;
            serde_json::from_reader(reader)
                .with_context(|| format!("failed to parse golden `{}`", path.display()))
        }
        DataFormat::Bin => {
            let reader = BufReader::new(
                File::open(path)
                    .with_context(|| format!("failed to open golden `{}`", path.display()))?,
            );
            let file: BinaryOutputsFile = bincode::deserialize_from(reader)
                .with_context(|| format!("failed to parse binary golden `{}`", path.display()))?;
            Ok(convert_binary_outputs(file))
        }
    }
}

fn write_outputs(path: &Path, format: DataFormat, outputs: &EvalOutputs) -> Result<()> {
    match format {
        DataFormat::Json => {
            let mut writer = BufWriter::new(
                File::create(path)
                    .with_context(|| format!("failed to create `{}`", path.display()))?,
            );
            serde_json::to_writer_pretty(&mut writer, outputs)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            Ok(())
        }
        DataFormat::Bin => {
            let file = convert_to_binary_outputs(outputs);
            let writer = BufWriter::new(
                File::create(path)
                    .with_context(|| format!("failed to create `{}`", path.display()))?,
            );
            bincode::serialize_into(writer, &file)?;
            Ok(())
        }
    }
}

fn convert_binary_outputs(outputs: BinaryOutputsFile) -> EvalOutputs {
    let cycles = outputs
        .cycles
        .into_iter()
        .map(|cycle| OutputCycle {
            index: cycle.index,
            outputs: cycle
                .outputs
                .into_iter()
                .map(|(name, vectors)| {
                    let formatted = vectors
                        .into_iter()
                        .map(|bytes| {
                            let value = BigUint::from_bytes_le(&bytes);
                            format_biguint(&value)
                        })
                        .collect();
                    (name, formatted)
                })
                .collect(),
        })
        .collect();

    EvalOutputs {
        design: outputs.design,
        top: outputs.top,
        vectors: outputs.vectors,
        cycles,
    }
}

fn convert_to_binary_outputs(outputs: &EvalOutputs) -> BinaryOutputsFile {
    let cycles = outputs
        .cycles
        .iter()
        .map(|cycle| BinaryOutputCycle {
            index: cycle.index,
            outputs: cycle
                .outputs
                .iter()
                .map(|(name, values)| {
                    let vectors = values
                        .iter()
                        .map(|value| {
                            let value =
                                parse_biguint_str(value).expect("output value must be valid hex");
                            value.to_bytes_le()
                        })
                        .collect();
                    (name.clone(), vectors)
                })
                .collect(),
        })
        .collect();

    BinaryOutputsFile {
        design: outputs.design.clone(),
        top: outputs.top.clone(),
        vectors: outputs.vectors,
        cycles,
    }
}

fn compare_outputs(
    expected: &EvalOutputs,
    actual: &EvalOutputs,
) -> std::result::Result<(), String> {
    if expected.design != actual.design {
        return Err(format!(
            "design name mismatch (expected `{}`, got `{}`)",
            expected.design, actual.design
        ));
    }
    if expected.top != actual.top {
        return Err(format!(
            "top module mismatch (expected `{}`, got `{}`)",
            expected.top, actual.top
        ));
    }
    if expected.vectors != actual.vectors {
        return Err(format!(
            "vector count mismatch (expected {}, got {})",
            expected.vectors, actual.vectors
        ));
    }
    if expected.cycles.len() != actual.cycles.len() {
        return Err(format!(
            "cycle count mismatch (expected {}, got {})",
            expected.cycles.len(),
            actual.cycles.len()
        ));
    }

    for (cycle_idx, (expected_cycle, actual_cycle)) in
        expected.cycles.iter().zip(&actual.cycles).enumerate()
    {
        if expected_cycle.outputs.len() != actual_cycle.outputs.len() {
            return Err(format!(
                "cycle {cycle_idx} output port count mismatch (expected {}, got {})",
                expected_cycle.outputs.len(),
                actual_cycle.outputs.len()
            ));
        }
        for (name, expected_values) in &expected_cycle.outputs {
            let Some(actual_values) = actual_cycle.outputs.get(name) else {
                return Err(format!(
                    "cycle {cycle_idx} missing port `{name}` in outputs"
                ));
            };
            if expected_values.len() != actual_values.len() {
                return Err(format!(
                    "cycle {cycle_idx} port `{name}` vector count mismatch (expected {}, got {})",
                    expected_values.len(),
                    actual_values.len()
                ));
            }
            for (vec_idx, (expected_value, actual_value)) in
                expected_values.iter().zip(actual_values).enumerate()
            {
                if expected_value != actual_value {
                    return Err(format!(
                        "cycle {cycle_idx} port `{name}` vector {vec_idx} mismatch (expected {}, got {})",
                        expected_value, actual_value
                    ));
                }
            }
        }
    }

    Ok(())
}
