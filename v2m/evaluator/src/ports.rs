use num_bigint::BigUint;
use num_traits::Zero;
use thiserror::Error;
use v2m_formats::nir::PortDirection;

use crate::packed::{Packed, PackedIndex, WORD_BITS};

#[derive(Debug, Error)]
pub enum PortValueError {
    #[error("missing data for port `{name}`")]
    MissingPort { name: String },
    #[error("unexpected port `{name}`")]
    UnexpectedPort { name: String },
    #[error("port `{name}` expects {expected} vectors, got {actual}")]
    VectorCountMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("value for port `{name}` vector {vector} exceeds width {width_bits} bits")]
    ValueTooWide {
        name: String,
        width_bits: usize,
        vector: usize,
    },
    #[error("packed buffer expects {expected} vectors, got {actual}")]
    PackedVectorMismatch { expected: usize, actual: usize },
    #[error("packed buffer words-per-lane mismatch (expected {expected}, got {actual})")]
    WordsPerLaneMismatch { expected: usize, actual: usize },
    #[error("packed buffer layout mismatch for port `{name}`")]
    PackedLayoutMismatch { name: String },
    #[error("port `{name}` has unsupported direction `{actual:?}` for this operation")]
    DirectionMismatch { name: String, actual: PortDirection },
}

pub(crate) fn pack_port_biguints(
    target: &mut Packed,
    index: PackedIndex,
    width_bits: usize,
    values: &[BigUint],
    name: &str,
) -> Result<(), PortValueError> {
    if width_bits > index.lanes() {
        return Err(PortValueError::PackedLayoutMismatch {
            name: name.to_string(),
        });
    }

    if values.len() != target.num_vectors() {
        return Err(PortValueError::VectorCountMismatch {
            name: name.to_string(),
            expected: target.num_vectors(),
            actual: values.len(),
        });
    }

    let words_per_lane = target.words_per_lane();
    let slice = target.slice_mut(index);
    slice.fill(0);

    if width_bits == 0 {
        for (vec_idx, value) in values.iter().enumerate() {
            if !value.is_zero() {
                return Err(PortValueError::ValueTooWide {
                    name: name.to_string(),
                    width_bits,
                    vector: vec_idx,
                });
            }
        }
        return Ok(());
    }

    for (vec_idx, value) in values.iter().enumerate() {
        if value.bits() > width_bits as u64 {
            return Err(PortValueError::ValueTooWide {
                name: name.to_string(),
                width_bits,
                vector: vec_idx,
            });
        }

        if value.is_zero() {
            continue;
        }

        let word_idx = vec_idx / WORD_BITS;
        let bit_in_word = vec_idx % WORD_BITS;
        let bit_mask = 1u64 << bit_in_word;

        for (chunk_idx, mut chunk) in value.to_u64_digits().into_iter().enumerate() {
            if chunk == 0 {
                continue;
            }

            let base_lane = chunk_idx * WORD_BITS;
            while chunk != 0 {
                let bit = chunk.trailing_zeros() as usize;
                let lane = base_lane + bit;
                if lane >= width_bits {
                    break;
                }

                let offset = index.offset() + lane * words_per_lane + word_idx;
                target.storage_mut()[offset] |= bit_mask;
                chunk &= chunk - 1;
            }
        }
    }

    Ok(())
}

pub(crate) fn unpack_port_biguints(
    source: &Packed,
    index: PackedIndex,
    width_bits: usize,
    name: &str,
) -> Result<Vec<BigUint>, PortValueError> {
    if width_bits > index.lanes() {
        return Err(PortValueError::PackedLayoutMismatch {
            name: name.to_string(),
        });
    }

    let words_per_lane = source.words_per_lane();
    let end = index.offset() + index.lanes() * words_per_lane;
    if end > source.storage().len() {
        return Err(PortValueError::PackedLayoutMismatch {
            name: name.to_string(),
        });
    }

    let num_vectors = source.num_vectors();
    let mut result = vec![BigUint::default(); num_vectors];

    if width_bits == 0 {
        return Ok(result);
    }

    let slice = source.slice(index);
    for lane in 0..width_bits {
        let lane_offset = lane * words_per_lane;
        for word_idx in 0..words_per_lane {
            let word = slice[lane_offset + word_idx];
            if word == 0 {
                continue;
            }

            let base_vector = word_idx * WORD_BITS;
            let mut mask = word;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let vector_idx = base_vector + bit;
                if vector_idx >= num_vectors {
                    break;
                }

                result[vector_idx].set_bit(lane as u64, true);
                mask &= mask - 1;
            }
        }
    }

    Ok(result)
}
