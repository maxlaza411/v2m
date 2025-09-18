use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};
use v2m_formats::nir::{Module as NirModule, Net as NirNet, Nir, Port as NirPort, PortDirection};

use crate::cells;
use crate::context::{BitLookupMap, Ctx, NetBitInfo};
use crate::nir_node::NirNode;

pub struct ModuleBuilder<'a> {
    module_name: &'a str,
    module: &'a yosys_bridge::loader::Module,
    bit_lookup: BitLookupMap,
    nets: BTreeMap<String, NirNet>,
}

impl<'a> ModuleBuilder<'a> {
    pub fn new(module_name: &'a str, module: &'a yosys_bridge::loader::Module) -> Result<Self> {
        let (nets, bit_lookup) = build_nets(module_name, &module.netnames)?;
        Ok(Self {
            module_name,
            module,
            bit_lookup,
            nets,
        })
    }

    pub fn build(self) -> Result<NirModule> {
        let ports = build_ports(self.module_name, &self.module.ports)?;
        let nodes = self.build_nodes()?;
        Ok(NirModule {
            ports,
            nets: self.nets,
            nodes,
        })
    }

    fn build_nodes(&self) -> Result<BTreeMap<String, v2m_formats::nir::Node>> {
        let mut nodes = BTreeMap::new();
        for (cell_name, cell) in &self.module.cells {
            let nir_node = map_cell(self.module_name, cell_name, cell, &self.bit_lookup)?;
            nodes.insert(cell_name.clone(), nir_node.into_node(cell_name.clone()));
        }
        Ok(nodes)
    }
}

pub fn rtlil_to_nir(design: &yosys_bridge::loader::RtlilJson) -> Result<Nir> {
    let mut modules = BTreeMap::new();
    for (name, module) in &design.modules {
        let builder = ModuleBuilder::new(name, module)?;
        let nir_module = builder.build()?;
        modules.insert(name.clone(), nir_module);
    }

    Ok(Nir {
        v: "nir-1.1".to_string(),
        design: design.top.clone(),
        top: design.top.clone(),
        attrs: None,
        modules,
        generator: None,
        cmdline: None,
        source_digest_sha256: None,
    })
}

fn map_cell(
    module_name: &str,
    cell_name: &str,
    cell: &yosys_bridge::loader::Cell,
    bit_lookup: &BitLookupMap,
) -> Result<NirNode> {
    let mut ctx = Ctx::new(module_name, cell_name, bit_lookup);
    match cell.kind.as_str() {
        "$and" => cells::logic::map_and(cell, &mut ctx),
        "$or" => cells::logic::map_or(cell, &mut ctx),
        "$xor" => cells::logic::map_xor(cell, &mut ctx),
        "$xnor" => cells::logic::map_xnor(cell, &mut ctx),
        "$not" => cells::logic::map_not(cell, &mut ctx),
        "$mux" => cells::logic::map_mux(cell, &mut ctx),
        "$add" => cells::arith::map_add(cell, &mut ctx),
        "$sub" => cells::arith::map_sub(cell, &mut ctx),
        other => bail!("unsupported cell `{other}` in module `{module_name}` while mapping to NIR"),
    }
}

fn build_ports(
    module_name: &str,
    ports: &BTreeMap<String, Value>,
) -> Result<BTreeMap<String, NirPort>> {
    let mut result = BTreeMap::new();
    for (name, value) in ports {
        let obj = value
            .as_object()
            .ok_or_else(|| anyhow!("port `{name}` in module `{module_name}` must be an object"))?;
        let direction_value = obj
            .get("direction")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow!("port `{name}` in module `{module_name}` is missing `direction`")
            })?;
        let bits_value = obj
            .get("bits")
            .ok_or_else(|| anyhow!("port `{name}` in module `{module_name}` is missing `bits`"))?;

        let bits = collect_bit_ids(bits_value).with_context(|| {
            format!("port `{name}` in module `{module_name}` has invalid `bits`")
        })?;
        if bits.is_empty() {
            bail!("port `{name}` in module `{module_name}` does not reference any bits");
        }
        let width = u32::try_from(bits.len()).context("port width exceeds supported range")?;
        let dir = match direction_value {
            "input" => PortDirection::Input,
            "output" => PortDirection::Output,
            "inout" => PortDirection::Inout,
            other => {
                bail!("port `{name}` in module `{module_name}` has unsupported direction `{other}`")
            }
        };

        result.insert(
            normalize_name(name),
            NirPort {
                dir,
                bits: width,
                attrs: None,
            },
        );
    }
    Ok(result)
}

fn build_nets(
    module_name: &str,
    netnames: &BTreeMap<String, Value>,
) -> Result<(BTreeMap<String, NirNet>, BitLookupMap)> {
    let mut nets = BTreeMap::new();
    let mut lookup: HashMap<i64, NetBitInfo> = HashMap::new();

    for (raw_name, value) in netnames {
        let obj = value.as_object().ok_or_else(|| {
            anyhow!("net `{raw_name}` in module `{module_name}` must be an object")
        })?;
        let bits_value = obj.get("bits").ok_or_else(|| {
            anyhow!("net `{raw_name}` in module `{module_name}` is missing `bits`")
        })?;
        let bit_ids = collect_bit_ids(bits_value).with_context(|| {
            format!("net `{raw_name}` in module `{module_name}` has invalid `bits`")
        })?;
        if bit_ids.is_empty() {
            bail!("net `{raw_name}` in module `{module_name}` contains no bits");
        }
        let width = u32::try_from(bit_ids.len()).context("net width exceeds supported range")?;
        let normalized = normalize_name(raw_name);

        for (offset, bit_id) in bit_ids.into_iter().enumerate() {
            let offset_u32 = u32::try_from(offset).context("net bit offset exceeds range")?;
            if lookup
                .insert(
                    bit_id,
                    NetBitInfo {
                        net: normalized.clone(),
                        offset: offset_u32,
                    },
                )
                .is_some()
            {
                bail!("bit id {bit_id} referenced by multiple nets in module `{module_name}`");
            }
        }

        nets.insert(
            normalized,
            NirNet {
                bits: width,
                attrs: None,
            },
        );
    }

    Ok((nets, lookup))
}

fn collect_bit_ids(value: &Value) -> Result<Vec<i64>> {
    let mut bits = Vec::new();
    collect_bit_ids_inner(value, &mut bits)?;
    Ok(bits)
}

fn collect_bit_ids_inner(value: &Value, bits: &mut Vec<i64>) -> Result<()> {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_bit_ids_inner(item, bits)?;
            }
            Ok(())
        }
        Value::Number(num) => {
            let bit_id = num
                .as_i64()
                .ok_or_else(|| anyhow!("connection bit index `{num}` is not an integer"))?;
            bits.push(bit_id);
            Ok(())
        }
        other => bail!("unsupported bit entry `{other}`"),
    }
}

fn normalize_name(raw: &str) -> String {
    raw.strip_prefix('\\').unwrap_or(raw).to_string()
}

#[allow(dead_code)]
fn value_is_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                i != 0
            } else if let Some(u) = num.as_u64() {
                u != 0
            } else if let Some(f) = num.as_f64() {
                f != 0.0
            } else {
                false
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                false
            } else {
                let lower = trimmed.to_ascii_lowercase();
                lower != "0" && lower != "false"
            }
        }
        _ => false,
    }
}

fn net_is_signed(obj: &Map<String, Value>) -> bool {
    if let Some(flag) = obj.get("signed") {
        if value_is_truthy(flag) {
            return true;
        }
    }
    if let Some(attributes) = obj.get("attributes").and_then(Value::as_object) {
        if let Some(flag) = attributes.get("signed") {
            if value_is_truthy(flag) {
                return true;
            }
        }
        if let Some(flag) = attributes.get("\\signed") {
            if value_is_truthy(flag) {
                return true;
            }
        }
    }
    false
}

// Placeholder to silence unused warning until signedness is used.
#[allow(dead_code)]
fn _maybe_use_signedness(obj: &Map<String, Value>) {
    let _ = net_is_signed(obj);
}
