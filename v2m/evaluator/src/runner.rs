use std::collections::HashMap;

use num_bigint::BigUint;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sha2::{Digest, Sha256};
use thiserror::Error;
use v2m_formats::nir::{Module, Nir, PortDirection};

use crate::{Evaluator, Packed, PackedBitMask, PortValueError, SimOptions};

#[derive(Debug, Clone)]
pub struct VectorRun {
    pub outputs: Packed,
    pub hash: [u8; 32],
}

#[derive(Debug, Error)]
pub enum RunVectorsError {
    #[error(transparent)]
    Evaluator(#[from] crate::Error),
    #[error(transparent)]
    Port(#[from] PortValueError),
    #[error("output mismatch vs oracle (expected {expected}, got {actual})")]
    Mismatch { expected: String, actual: String },
}

pub fn run_vectors(
    nir: &Nir,
    num_vectors: usize,
    seed: u64,
    oracle: Option<&[u8; 32]>,
) -> Result<VectorRun, RunVectorsError> {
    run_vectors_with_options(nir, num_vectors, seed, SimOptions::default(), oracle)
}

pub fn run_vectors_with_options(
    nir: &Nir,
    num_vectors: usize,
    seed: u64,
    options: SimOptions,
    oracle: Option<&[u8; 32]>,
) -> Result<VectorRun, RunVectorsError> {
    let mut evaluator = Evaluator::new(nir, num_vectors, options)?;
    let module = nir
        .modules
        .get(nir.top.as_str())
        .expect("top module must exist when evaluator is constructed");

    let mut rng = StdRng::seed_from_u64(seed);
    let inputs = generate_random_inputs(module, num_vectors, &mut rng);
    let packed_inputs = evaluator.pack_inputs_from_biguints(&inputs)?;
    let reset_mask = PackedBitMask::new(num_vectors);
    let outputs = evaluator.tick(&packed_inputs, &reset_mask)?;
    let hash = hash_packed_outputs(&outputs);

    if let Some(expected) = oracle {
        if &hash != expected {
            return Err(RunVectorsError::Mismatch {
                expected: hex_digest(expected),
                actual: hex_digest(&hash),
            });
        }
    }

    Ok(VectorRun { outputs, hash })
}

pub fn hash_packed_outputs(packed: &Packed) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((packed.num_vectors() as u64).to_le_bytes());
    hasher.update((packed.words_per_lane() as u64).to_le_bytes());
    hasher.update((packed.total_lanes() as u64).to_le_bytes());
    for word in packed.storage() {
        hasher.update(word.to_le_bytes());
    }
    hasher.finalize().into()
}

fn generate_random_inputs(
    module: &Module,
    num_vectors: usize,
    rng: &mut StdRng,
) -> HashMap<String, Vec<BigUint>> {
    let mut map = HashMap::new();
    for (name, port) in &module.ports {
        if matches!(port.dir, PortDirection::Input | PortDirection::Inout) {
            let width = port.bits as usize;
            let mut values = Vec::with_capacity(num_vectors);
            for _ in 0..num_vectors {
                values.push(random_biguint(width, rng));
            }
            map.insert(name.clone(), values);
        }
    }
    map
}

fn random_biguint(width_bits: usize, rng: &mut StdRng) -> BigUint {
    if width_bits == 0 {
        return BigUint::default();
    }

    let byte_len = (width_bits + 7) / 8;
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

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut text, "{:02x}", byte);
    }
    text
}
