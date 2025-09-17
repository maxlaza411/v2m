use num_bigint::BigUint;
use num_traits::{Num, Zero};
use serde_json::Value;

use crate::packed::{mask_for_word, Packed, PackedIndex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResetKind {
    None,
    Sync,
    Async,
}

pub(crate) fn parse_reset_kind(node: &v2m_formats::nir::Node) -> ResetKind {
    if let Some(attrs) = &node.attrs {
        if let Some(kind_value) = attrs.get("reset_kind") {
            if let Some(kind) = reset_kind_from_value(kind_value) {
                return kind;
            }
        }

        if let Some(reset_value) = attrs.get("reset") {
            if let Some(kind) = reset_kind_from_value(reset_value) {
                return kind;
            }

            if let Value::Object(map) = reset_value {
                if let Some(kind) = map
                    .get("init")
                    .and_then(|value| reset_kind_from_value(value))
                    .or_else(|| {
                        map.get("value")
                            .and_then(|value| reset_kind_from_value(value))
                    })
                {
                    return kind;
                }
            }
        }

        if let Some(flag) = attrs.get("async_reset").and_then(Value::as_bool) {
            return if flag {
                ResetKind::Async
            } else {
                ResetKind::Sync
            };
        }
    }

    ResetKind::Sync
}

pub(crate) fn parse_init_bits(node: &v2m_formats::nir::Node) -> Vec<bool> {
    let width = node.width as usize;
    if width == 0 {
        return Vec::new();
    }

    if let Some(attrs) = &node.attrs {
        if let Some(bits) = attrs
            .get("init")
            .and_then(|value| parse_init_bits_value(value, width))
        {
            return bits;
        }

        if let Some(reset_value) = attrs.get("reset") {
            if let Some(bits) = parse_init_bits_value(reset_value, width) {
                return bits;
            }

            if let Value::Object(map) = reset_value {
                if let Some(bits) = map
                    .get("init")
                    .and_then(|value| parse_init_bits_value(value, width))
                    .or_else(|| {
                        map.get("value")
                            .and_then(|value| parse_init_bits_value(value, width))
                    })
                {
                    return bits;
                }
            }
        }
    }

    vec![false; width]
}

pub(crate) fn apply_register_init_bits(
    packed: &mut Packed,
    index: PackedIndex,
    bits: &[bool],
    num_vectors: usize,
) {
    let words_per_lane = packed.words_per_lane();
    if words_per_lane == 0 {
        return;
    }

    for (lane, bit) in bits.iter().copied().enumerate().take(index.lanes()) {
        let lane_offset = index.lane_offset(lane, words_per_lane);
        for word_idx in 0..words_per_lane {
            let mask = mask_for_word(word_idx, words_per_lane, num_vectors);
            packed.storage_mut()[lane_offset + word_idx] = if bit { mask } else { 0 };
        }
    }
}

fn reset_kind_from_value(value: &Value) -> Option<ResetKind> {
    match value {
        Value::String(kind) => reset_kind_from_str(kind),
        Value::Bool(flag) => Some(if *flag {
            ResetKind::Async
        } else {
            ResetKind::Sync
        }),
        Value::Object(map) => {
            if let Some(kind_value) = map.get("kind") {
                reset_kind_from_value(kind_value)
            } else if let Some(kind_value) = map.get("type") {
                reset_kind_from_value(kind_value)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn reset_kind_from_str(kind: &str) -> Option<ResetKind> {
    if kind.eq_ignore_ascii_case("sync") {
        Some(ResetKind::Sync)
    } else if kind.eq_ignore_ascii_case("async") {
        Some(ResetKind::Async)
    } else if kind.eq_ignore_ascii_case("none") || kind.eq_ignore_ascii_case("disabled") {
        Some(ResetKind::None)
    } else {
        None
    }
}

fn parse_init_bits_value(value: &Value, width: usize) -> Option<Vec<bool>> {
    match value {
        Value::String(text) => parse_biguint_string(text).map(|big| biguint_to_bits(&big, width)),
        Value::Number(number) => number
            .as_u64()
            .map(|raw| biguint_to_bits(&BigUint::from(raw), width)),
        Value::Bool(flag) => {
            let big = if *flag {
                BigUint::from(1u8)
            } else {
                BigUint::zero()
            };
            Some(biguint_to_bits(&big, width))
        }
        Value::Object(map) => {
            if let Some(inner) = map.get("init") {
                return parse_init_bits_value(inner, width);
            }
            if let Some(inner) = map.get("value") {
                return parse_init_bits_value(inner, width);
            }
            None
        }
        _ => None,
    }
}

fn parse_biguint_string(value: &str) -> Option<BigUint> {
    let cleaned: String = value.chars().filter(|ch| *ch != '_').collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return Some(BigUint::zero());
    }

    let (radix, digits) = if let Some(rest) = trimmed.strip_prefix("0x") {
        (16, rest)
    } else if let Some(rest) = trimmed.strip_prefix("0X") {
        (16, rest)
    } else if let Some(rest) = trimmed.strip_prefix("0b") {
        (2, rest)
    } else if let Some(rest) = trimmed.strip_prefix("0B") {
        (2, rest)
    } else if let Some(rest) = trimmed.strip_prefix("0o") {
        (8, rest)
    } else if let Some(rest) = trimmed.strip_prefix("0O") {
        (8, rest)
    } else {
        (10, trimmed)
    };

    let digits = digits.trim_start_matches('+');
    if digits.is_empty() {
        return Some(BigUint::zero());
    }

    BigUint::from_str_radix(digits, radix).ok()
}

fn biguint_to_bits(value: &BigUint, width: usize) -> Vec<bool> {
    let mut bits = vec![false; width];
    let digits = value.to_u64_digits();
    for bit in 0..width {
        let word_index = bit / 64;
        let bit_index = bit % 64;
        if let Some(word) = digits.get(word_index) {
            bits[bit] = ((*word >> bit_index) & 1) != 0;
        }
    }
    bits
}
