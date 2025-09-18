use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use v2m_formats::nir::{BitRef, BitRefConcat, BitRefConst, BitRefNet};

type BitLookup = HashMap<i64, NetBitInfo>;

#[derive(Debug, Clone)]
pub struct NetBitInfo {
    pub net: String,
    pub offset: u32,
}

#[derive(Debug)]
pub struct Ctx<'a> {
    module_name: &'a str,
    cell_name: &'a str,
    bit_lookup: &'a BitLookup,
    cache: BTreeMap<String, BitRef>,
}

impl<'a> Ctx<'a> {
    pub fn new(module_name: &'a str, cell_name: &'a str, bit_lookup: &'a BitLookup) -> Self {
        Self {
            module_name,
            cell_name,
            bit_lookup,
            cache: BTreeMap::new(),
        }
    }

    pub fn module_name(&self) -> &str {
        self.module_name
    }

    pub fn cell_name(&self) -> &str {
        self.cell_name
    }

    pub fn bitref(&mut self, cell: &yosys_bridge::loader::Cell, pin: &str) -> Result<BitRef> {
        if let Some(bitref) = self.cache.get(pin) {
            return Ok(bitref.clone());
        }

        let value = cell
            .connections
            .get(pin)
            .ok_or_else(|| self.missing_pin_error(pin))?;

        let bitref = self.extract_bitref(pin, value).with_context(|| {
            format!(
                "failed to extract pin `{pin}` on cell `{}` in module `{}`",
                self.cell_name, self.module_name
            )
        })?;
        self.cache.insert(pin.to_string(), bitref.clone());
        Ok(bitref)
    }

    pub fn bitref_width(bitref: &BitRef) -> u32 {
        match bitref {
            BitRef::Net(BitRefNet { lsb, msb, .. }) => msb - lsb + 1,
            BitRef::Const(BitRefConst { width, .. }) => *width,
            BitRef::Concat(BitRefConcat { concat }) => concat.iter().map(Self::bitref_width).sum(),
        }
    }

    pub fn ensure_equal_width<'b>(&self, entries: &[(&'b str, &'b BitRef)]) -> Result<u32> {
        let mut width: Option<u32> = None;
        for (pin, bitref) in entries {
            let bit_width = Self::bitref_width(bitref);
            if let Some(current) = width {
                if current != bit_width {
                    bail!(
                        "pin `{}` on cell `{}` in module `{}` has width {} but expected {current}",
                        pin,
                        self.cell_name,
                        self.module_name,
                        bit_width
                    );
                }
            } else {
                width = Some(bit_width);
            }
        }

        width.ok_or_else(|| {
            anyhow!(
                "expected at least one pin to determine width for cell `{}` in module `{}`",
                self.cell_name,
                self.module_name
            )
        })
    }

    pub fn expect_width(&self, pin: &str, bitref: &BitRef, expected: u32) -> Result<()> {
        let actual = Self::bitref_width(bitref);
        if actual != expected {
            bail!(
                "pin `{}` on cell `{}` in module `{}` has width {} but expected {}",
                pin,
                self.cell_name,
                self.module_name,
                actual,
                expected
            );
        }
        Ok(())
    }

    fn extract_bitref(&self, pin: &str, value: &Value) -> Result<BitRef> {
        let mut bits = Vec::new();
        self.collect_bits(pin, value, &mut bits)?;
        let extracted = yosys_bridge::bitref::to_bitref(&bits)?;
        Ok(convert_bitref(&extracted))
    }

    fn collect_bits(
        &self,
        pin: &str,
        value: &Value,
        bits: &mut Vec<yosys_bridge::bitref::RtlilBit>,
    ) -> Result<()> {
        match value {
            Value::Array(items) => {
                for item in items {
                    self.collect_bits(pin, item, bits)?;
                }
                Ok(())
            }
            Value::Number(num) => {
                let bit_id = num
                    .as_i64()
                    .ok_or_else(|| anyhow!("connection bit for pin `{pin}` is not an integer"))?;
                let info = self
                    .bit_lookup
                    .get(&bit_id)
                    .ok_or_else(|| {
                        anyhow!(
                            "pin `{pin}` on cell `{}` in module `{}` references unknown net bit {}",
                            self.cell_name, self.module_name, bit_id
                        )
                    })?;
                bits.push(yosys_bridge::bitref::RtlilBit::Net {
                    net: info.net.clone(),
                    bit_index: info.offset,
                });
                Ok(())
            }
            Value::String(text) => {
                if text.is_empty() {
                    bail!(
                        "pin `{pin}` on cell `{}` in module `{}` contains empty constant",
                        self.cell_name, self.module_name
                    );
                }
                for ch in text.chars() {
                    bits.push(yosys_bridge::bitref::RtlilBit::Const(ch));
                }
                Ok(())
            }
            other => bail!(
                "unsupported connection element `{other}` for pin `{pin}` on cell `{}` in module `{}`",
                self.cell_name,
                self.module_name
            ),
        }
    }

    fn missing_pin_error(&self, pin: &str) -> anyhow::Error {
        anyhow!(
            "cell `{}` in module `{}` is missing connection for pin `{}`",
            self.cell_name,
            self.module_name,
            pin
        )
    }
}

pub fn convert_bitref(source: &yosys_bridge::bitref::BitRef) -> BitRef {
    match source {
        yosys_bridge::bitref::BitRef::Slice { net, lsb, msb } => BitRef::Net(BitRefNet {
            net: net.clone(),
            lsb: *lsb,
            msb: *msb,
        }),
        yosys_bridge::bitref::BitRef::Const { value, width } => BitRef::Const(BitRefConst {
            value: value.clone(),
            width: *width,
        }),
        yosys_bridge::bitref::BitRef::Concat { parts } => BitRef::Concat(BitRefConcat {
            concat: parts.iter().map(convert_bitref).collect(),
        }),
    }
}

pub type BitLookupMap = BitLookup;
