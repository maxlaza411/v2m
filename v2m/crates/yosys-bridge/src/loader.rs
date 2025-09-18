use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Default, Clone)]
pub struct LoaderOptions {
    pub allow_mem_blackbox: bool,
}

#[derive(Debug)]
pub struct RtlilJson {
    pub top: String,
    pub modules: BTreeMap<String, Module>,
}

#[derive(Debug)]
pub struct Module {
    pub attributes: BTreeMap<String, Value>,
    pub ports: BTreeMap<String, Value>,
    pub cells: BTreeMap<String, Cell>,
    pub netnames: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct Cell {
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Deserialize)]
struct RawRtlilJson {
    modules: Option<BTreeMap<String, RawModule>>,
}

#[derive(Debug, Deserialize)]
struct RawModule {
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
    ports: Option<BTreeMap<String, Value>>,
    cells: Option<BTreeMap<String, Cell>>,
    netnames: Option<BTreeMap<String, Value>>,
}

impl Module {
    fn from_raw(name: &str, raw: RawModule) -> Result<Self> {
        let ports = raw
            .ports
            .ok_or_else(|| anyhow!("RTLIL module `{}` is missing `ports`", name))?;
        let cells = raw
            .cells
            .ok_or_else(|| anyhow!("RTLIL module `{}` is missing `cells`", name))?;
        let netnames = raw
            .netnames
            .ok_or_else(|| anyhow!("RTLIL module `{}` is missing `netnames`", name))?;

        Ok(Self {
            attributes: raw.attributes,
            ports,
            cells,
            netnames,
        })
    }

    fn is_top(&self) -> bool {
        const TOP_KEYS: [&str; 2] = ["top", "\\top"];
        self.attributes
            .iter()
            .filter(|(key, _)| TOP_KEYS.contains(&key.as_str()))
            .any(|(_, value)| value_is_truthy(value))
    }
}

pub fn load_rtlil_json(path: impl AsRef<Path>, options: &LoaderOptions) -> Result<RtlilJson> {
    let path = path.as_ref();
    let raw_data = fs::read_to_string(path)
        .with_context(|| format!("failed to read RTLIL JSON from {}", display_path(path)))?;

    let raw: RawRtlilJson = serde_json::from_str(&raw_data)
        .with_context(|| format!("failed to parse RTLIL JSON from {}", display_path(path)))?;

    let modules_map = raw
        .modules
        .ok_or_else(|| anyhow!("RTLIL JSON is missing the `modules` map"))?;
    ensure!(
        !modules_map.is_empty(),
        "RTLIL JSON does not contain any modules"
    );

    let mut modules = BTreeMap::new();
    let mut top_modules = Vec::new();
    let mut memory_cells = Vec::new();

    for (name, raw_module) in modules_map {
        let module = Module::from_raw(&name, raw_module)?;

        if module.is_top() {
            top_modules.push(name.clone());
        }

        for (cell_name, cell) in &module.cells {
            if is_memory_cell(&cell.kind) {
                memory_cells.push(format!("{name}.{cell_name}"));
            }
        }

        modules.insert(name, module);
    }

    let top = match top_modules.len() {
        0 => {
            bail!(
                "RTLIL JSON does not identify a top module. Add the `(* top *)` attribute before exporting."
            )
        }
        1 => top_modules.remove(0),
        _ => bail!(
            "multiple RTLIL modules are marked as top: {}",
            top_modules.join(", ")
        ),
    };

    if !memory_cells.is_empty() && !options.allow_mem_blackbox {
        bail!(
            "RTLIL JSON still contains memory cells that must be lowered: {}",
            memory_cells.join(", ")
        );
    }

    Ok(RtlilJson { top, modules })
}

fn is_memory_cell(cell_type: &str) -> bool {
    matches!(cell_type, "$mem" | "$memrd" | "$memwr")
}

fn value_is_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(true) => true,
        Value::Number(num) => num.as_u64() == Some(1),
        Value::String(s) => s == "1" || s.eq_ignore_ascii_case("true"),
        _ => false,
    }
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_attribute_variants() {
        assert!(value_is_truthy(&Value::Bool(true)));
        assert!(value_is_truthy(&Value::String("1".to_string())));
        assert!(value_is_truthy(&Value::String("TRUE".to_string())));
        assert!(value_is_truthy(&serde_json::json!(1)));
        assert!(!value_is_truthy(&Value::Bool(false)));
        assert!(!value_is_truthy(&serde_json::json!(0)));
        assert!(!value_is_truthy(&Value::String("no".to_string())));
    }

    #[test]
    fn memory_cell_detection() {
        assert!(is_memory_cell("$mem"));
        assert!(is_memory_cell("$memrd"));
        assert!(is_memory_cell("$memwr"));
        assert!(!is_memory_cell("$dff"));
    }
}
