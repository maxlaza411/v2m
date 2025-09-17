use std::cmp::min;
use std::collections::HashMap;

use thiserror::Error;
use v2m_formats::nir::{BitRef, Module};
use v2m_formats::{resolve_bitref, ResolvedBit};
use v2m_nir::ModuleGraph;
use v2m_nir::NetId;

pub(crate) const LANE_BITS: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ConstId(pub(crate) usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConstValue {
    pub(crate) words: Vec<u64>,
    pub(crate) width: u32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ConstPool {
    values: Vec<ConstValue>,
    lookup: HashMap<ConstKey, ConstId>,
}

impl ConstPool {
    pub(crate) fn allocate(&mut self, bits: &[bool]) -> ConstId {
        let width = bits.len() as u32;
        let lanes = (bits.len() + (LANE_BITS as usize - 1)) / LANE_BITS as usize;
        let mut words = vec![0u64; lanes];
        for (index, &bit) in bits.iter().enumerate() {
            if bit {
                let lane = index / LANE_BITS as usize;
                let offset = index % LANE_BITS as usize;
                words[lane] |= 1u64 << offset;
            }
        }

        let key = ConstKey {
            width,
            words: words.clone(),
        };

        if let Some(&id) = self.lookup.get(&key) {
            return id;
        }

        let id = ConstId(self.values.len());
        self.lookup.insert(key, id);
        self.values.push(ConstValue { words, width });
        id
    }

    #[allow(dead_code)]
    pub(crate) fn get(&self, id: ConstId) -> &ConstValue {
        &self.values[id.0]
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn values(&self) -> &[ConstValue] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConstKey {
    width: u32,
    words: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SignalId {
    Net(NetId),
    Const(ConstId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BitDescriptor {
    pub(crate) source: SignalId,
    pub(crate) lane_offset: u32,
    pub(crate) bit_offset: u8,
    pub(crate) width: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BitBinding {
    descriptors: Vec<BitDescriptor>,
}

impl BitBinding {
    pub(crate) fn new(descriptors: Vec<BitDescriptor>) -> Self {
        Self { descriptors }
    }

    #[allow(dead_code)]
    pub(crate) fn descriptors(&self) -> &[BitDescriptor] {
        &self.descriptors
    }

    #[allow(dead_code)]
    pub(crate) fn total_width(&self) -> u32 {
        self.descriptors.iter().map(|desc| desc.width as u32).sum()
    }
}

#[derive(Debug, Error)]
pub enum PinBindingError {
    #[error(transparent)]
    Resolve(#[from] v2m_formats::Error),
    #[error("net `{net}` referenced by bitref not found")]
    UnknownNet { net: String },
}

pub(crate) fn bind_bitref(
    module: &Module,
    graph: &ModuleGraph,
    bitref: &BitRef,
    const_pool: &mut ConstPool,
) -> Result<BitBinding, PinBindingError> {
    let resolved = resolve_bitref(module, bitref)?;
    let mut descriptors = Vec::new();
    let mut index = 0usize;

    while index < resolved.len() {
        match &resolved[index] {
            ResolvedBit::Net(net_bit) => {
                let net_id = graph.net_id(net_bit.net.as_str()).ok_or_else(|| {
                    PinBindingError::UnknownNet {
                        net: net_bit.net.clone(),
                    }
                })?;
                let start_bit = net_bit.bit;
                let mut end_bit = net_bit.bit;
                index += 1;
                while index < resolved.len() {
                    match &resolved[index] {
                        ResolvedBit::Net(next)
                            if next.net == net_bit.net && next.bit == end_bit + 1 =>
                        {
                            end_bit = next.bit;
                            index += 1;
                        }
                        _ => break,
                    }
                }

                let mut current = start_bit;
                loop {
                    let lane_offset = current / LANE_BITS;
                    let bit_offset = (current % LANE_BITS) as u8;
                    let lane_end = lane_offset * LANE_BITS + (LANE_BITS - 1);
                    let chunk_end = min(end_bit, lane_end);
                    let width = (chunk_end - current + 1) as u8;
                    descriptors.push(BitDescriptor {
                        source: SignalId::Net(net_id),
                        lane_offset,
                        bit_offset,
                        width,
                    });
                    if chunk_end == end_bit {
                        break;
                    }
                    current = chunk_end + 1;
                }
            }
            ResolvedBit::Const(value) => {
                let mut bits = Vec::new();
                bits.push(*value);
                index += 1;
                while index < resolved.len() {
                    match resolved[index] {
                        ResolvedBit::Const(next) => {
                            bits.push(next);
                            index += 1;
                        }
                        _ => break,
                    }
                }

                let const_id = const_pool.allocate(&bits);
                let mut remaining = bits.len();
                let mut lane_offset = 0u32;
                while remaining > 0 {
                    let chunk = min(remaining, LANE_BITS as usize);
                    descriptors.push(BitDescriptor {
                        source: SignalId::Const(const_id),
                        lane_offset,
                        bit_offset: 0,
                        width: chunk as u8,
                    });
                    remaining -= chunk;
                    lane_offset += 1;
                }
            }
        }
    }

    Ok(BitBinding::new(descriptors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use v2m_formats::nir::{
        BitRefConcat, BitRefConst, BitRefNet, Module as NirModule, Net as NirNet,
    };

    fn module_with_nets(nets: &[(&str, u32)]) -> NirModule {
        let nets_map = nets
            .iter()
            .map(|(name, bits)| {
                (
                    (*name).to_string(),
                    NirNet {
                        bits: *bits,
                        attrs: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        NirModule {
            ports: BTreeMap::new(),
            nets: nets_map,
            nodes: BTreeMap::new(),
        }
    }

    #[test]
    fn binds_slice_across_lane_boundary() {
        let module = module_with_nets(&[("wide", 130)]);
        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let mut const_pool = ConstPool::default();
        let bitref = BitRef::Net(BitRefNet {
            net: "wide".to_string(),
            lsb: 60,
            msb: 75,
        });

        let binding = bind_bitref(&module, &graph, &bitref, &mut const_pool).expect("bind bitref");
        let descriptors = binding.descriptors();
        assert_eq!(descriptors.len(), 2);
        assert!(const_pool.is_empty());

        let net_id = graph.net_id("wide").expect("net id");
        assert_eq!(
            descriptors[0],
            BitDescriptor {
                source: SignalId::Net(net_id),
                lane_offset: 0,
                bit_offset: 60,
                width: 4,
            }
        );
        assert_eq!(
            descriptors[1],
            BitDescriptor {
                source: SignalId::Net(net_id),
                lane_offset: 1,
                bit_offset: 0,
                width: 12,
            }
        );
    }

    #[test]
    fn binds_concat_spanning_multiple_lanes() {
        let module = module_with_nets(&[("left", 96), ("right", 96)]);
        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let mut const_pool = ConstPool::default();
        let bitref = BitRef::Concat(BitRefConcat {
            concat: vec![
                BitRef::Net(BitRefNet {
                    net: "left".to_string(),
                    lsb: 48,
                    msb: 95,
                }),
                BitRef::Net(BitRefNet {
                    net: "right".to_string(),
                    lsb: 0,
                    msb: 70,
                }),
            ],
        });

        let binding = bind_bitref(&module, &graph, &bitref, &mut const_pool).expect("bind bitref");
        let descriptors = binding.descriptors();
        assert_eq!(descriptors.len(), 4);
        assert!(const_pool.is_empty());

        let left_id = graph.net_id("left").expect("left id");
        let right_id = graph.net_id("right").expect("right id");

        assert_eq!(
            descriptors,
            [
                BitDescriptor {
                    source: SignalId::Net(left_id),
                    lane_offset: 0,
                    bit_offset: 48,
                    width: 16,
                },
                BitDescriptor {
                    source: SignalId::Net(left_id),
                    lane_offset: 1,
                    bit_offset: 0,
                    width: 32,
                },
                BitDescriptor {
                    source: SignalId::Net(right_id),
                    lane_offset: 0,
                    bit_offset: 0,
                    width: 64,
                },
                BitDescriptor {
                    source: SignalId::Net(right_id),
                    lane_offset: 1,
                    bit_offset: 0,
                    width: 7,
                },
            ]
        );
    }

    #[test]
    fn binds_constant_bits_into_pool() {
        let module = module_with_nets(&[]);
        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let mut const_pool = ConstPool::default();
        let bitref = BitRef::Const(BitRefConst {
            value: "0b10101100".to_string(),
            width: 8,
        });

        let binding = bind_bitref(&module, &graph, &bitref, &mut const_pool).expect("bind const");
        let descriptors = binding.descriptors();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(const_pool.len(), 1);

        let const_id = match descriptors[0].source {
            SignalId::Const(id) => id,
            other => panic!("expected const descriptor, got {other:?}"),
        };
        assert_eq!(descriptors[0].lane_offset, 0);
        assert_eq!(descriptors[0].bit_offset, 0);
        assert_eq!(descriptors[0].width, 8);

        let value = const_pool.get(const_id);
        assert_eq!(value.width, 8);
        assert_eq!(value.words.len(), 1);
        assert_eq!(value.words[0], 0b10101100);
    }

    #[test]
    fn reuses_existing_constant_values() {
        let module = module_with_nets(&[]);
        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let mut const_pool = ConstPool::default();
        let bitref = BitRef::Const(BitRefConst {
            value: "0b1100".to_string(),
            width: 4,
        });

        let first = bind_bitref(&module, &graph, &bitref, &mut const_pool).expect("bind const");
        let second = bind_bitref(&module, &graph, &bitref, &mut const_pool).expect("bind const");

        assert_eq!(const_pool.len(), 1);

        let first_id = match first.descriptors()[0].source {
            SignalId::Const(id) => id,
            other => panic!("expected const descriptor, got {other:?}"),
        };
        let second_id = match second.descriptors()[0].source {
            SignalId::Const(id) => id,
            other => panic!("expected const descriptor, got {other:?}"),
        };

        assert_eq!(first_id, second_id);
    }

    #[test]
    fn rejects_out_of_bounds_slice() {
        let module = module_with_nets(&[("wide", 32)]);
        let graph = ModuleGraph::from_module(&module).expect("build graph");
        let mut const_pool = ConstPool::default();
        let bitref = BitRef::Net(BitRefNet {
            net: "wide".to_string(),
            lsb: 0,
            msb: 64,
        });

        let err =
            bind_bitref(&module, &graph, &bitref, &mut const_pool).expect_err("expected error");
        match err {
            PinBindingError::Resolve(v2m_formats::Error::BitRangeOutOfBounds { net, .. }) => {
                assert_eq!(net, "wide");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
}
