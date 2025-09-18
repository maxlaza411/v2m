use std::convert::TryFrom;

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtlilBit {
    Net { net: String, bit_index: u32 },
    Const(char),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitRef {
    Slice { net: String, lsb: u32, msb: u32 },
    Const { value: String, width: u32 },
    Concat { parts: Vec<BitRef> },
}

pub fn to_bitref(bits: &[RtlilBit]) -> Result<BitRef> {
    if bits.is_empty() {
        bail!("RTLIL connection does not contain any bits");
    }

    let mut parts: Vec<BitRef> = Vec::new();
    let mut index = 0usize;

    while index < bits.len() {
        match &bits[index] {
            RtlilBit::Net { net, bit_index } => {
                let net_name = net.clone();
                let lsb = *bit_index;
                let mut msb = *bit_index;
                index += 1;

                while index < bits.len() {
                    match &bits[index] {
                        RtlilBit::Net {
                            net: next_net,
                            bit_index: next_bit,
                        } => {
                            if *next_net == net_name {
                                if let Some(expected) = msb.checked_add(1) {
                                    if *next_bit == expected {
                                        msb = *next_bit;
                                        index += 1;
                                        continue;
                                    }
                                }
                            }
                            break;
                        }
                        _ => break,
                    }
                }

                parts.push(BitRef::Slice {
                    net: net_name,
                    lsb,
                    msb,
                });
            }
            RtlilBit::Const(_) => {
                let mut const_bits: Vec<bool> = Vec::new();

                while index < bits.len() {
                    match bits[index] {
                        RtlilBit::Const(ch) => {
                            let lower = ch.to_ascii_lowercase();
                            match lower {
                                '0' => const_bits.push(false),
                                '1' => const_bits.push(true),
                                'x' | 'z' => {
                                    bail!(
                                        "RTLIL constant contains `{}`; re-run with --allow-x to permit X/Z bits",
                                        lower
                                    );
                                }
                                other => {
                                    bail!(
                                        "unsupported RTLIL constant bit `{}` encountered in connection",
                                        other
                                    );
                                }
                            }
                            index += 1;
                        }
                        _ => break,
                    }
                }

                let width = u32::try_from(const_bits.len())
                    .context("constant width exceeds maximum supported size")?;

                let mut literal = String::with_capacity(const_bits.len() + 2);
                literal.push_str("0b");
                for bit in const_bits.iter().rev() {
                    literal.push(if *bit { '1' } else { '0' });
                }

                parts.push(BitRef::Const {
                    value: literal,
                    width,
                });
            }
        }
    }

    if parts.len() == 1 {
        Ok(parts.into_iter().next().unwrap())
    } else {
        Ok(BitRef::Concat { parts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net_bit(net: &str, bit: u32) -> RtlilBit {
        RtlilBit::Net {
            net: net.to_string(),
            bit_index: bit,
        }
    }

    fn const_bit(bit: char) -> RtlilBit {
        RtlilBit::Const(bit)
    }

    #[test]
    fn single_contiguous_slice_becomes_single_bitref() {
        let bits = vec![net_bit("a", 0), net_bit("a", 1), net_bit("a", 2)];
        let bitref = to_bitref(&bits).expect("to_bitref");
        assert_eq!(
            bitref,
            BitRef::Slice {
                net: "a".to_string(),
                lsb: 0,
                msb: 2,
            }
        );
    }

    #[test]
    fn non_contiguous_slice_creates_concat() {
        let bits = vec![net_bit("a", 0), net_bit("a", 2)];
        let bitref = to_bitref(&bits).expect("to_bitref");
        assert_eq!(
            bitref,
            BitRef::Concat {
                parts: vec![
                    BitRef::Slice {
                        net: "a".to_string(),
                        lsb: 0,
                        msb: 0,
                    },
                    BitRef::Slice {
                        net: "a".to_string(),
                        lsb: 2,
                        msb: 2,
                    },
                ],
            }
        );
    }

    #[test]
    fn constants_turn_into_const_bitref() {
        let bits = vec![const_bit('0'), const_bit('1')];
        let bitref = to_bitref(&bits).expect("to_bitref");
        assert_eq!(
            bitref,
            BitRef::Const {
                value: "0b10".to_string(),
                width: 2,
            }
        );
    }

    #[test]
    fn mixes_of_nets_and_consts_become_concat() {
        let bits = vec![
            net_bit("a", 0),
            const_bit('0'),
            net_bit("b", 3),
            net_bit("b", 4),
        ];
        let bitref = to_bitref(&bits).expect("to_bitref");
        assert_eq!(
            bitref,
            BitRef::Concat {
                parts: vec![
                    BitRef::Slice {
                        net: "a".to_string(),
                        lsb: 0,
                        msb: 0,
                    },
                    BitRef::Const {
                        value: "0b0".to_string(),
                        width: 1,
                    },
                    BitRef::Slice {
                        net: "b".to_string(),
                        lsb: 3,
                        msb: 4,
                    },
                ],
            }
        );
    }

    #[test]
    fn cross_boundary_slice_builds_concat() {
        let bits = vec![
            net_bit("data", 4),
            net_bit("data", 5),
            net_bit("data", 6),
            net_bit("data", 7),
            net_bit("data", 0),
            net_bit("data", 1),
            net_bit("data", 2),
            net_bit("data", 3),
        ];

        let bitref = to_bitref(&bits).expect("to_bitref");
        assert_eq!(
            bitref,
            BitRef::Concat {
                parts: vec![
                    BitRef::Slice {
                        net: "data".to_string(),
                        lsb: 4,
                        msb: 7,
                    },
                    BitRef::Slice {
                        net: "data".to_string(),
                        lsb: 0,
                        msb: 3,
                    },
                ],
            }
        );
    }

    #[test]
    fn rejects_x_constants() {
        let bits = vec![const_bit('x')];
        let err = to_bitref(&bits).expect_err("expect x rejection");
        assert!(err.to_string().contains("--allow-x"));
    }

    #[test]
    fn rejects_z_constants() {
        let bits = vec![const_bit('z')];
        let err = to_bitref(&bits).expect_err("expect z rejection");
        assert!(err.to_string().contains("--allow-x"));
    }
}
