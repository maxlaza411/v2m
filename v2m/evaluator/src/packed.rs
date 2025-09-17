use thiserror::Error;

pub(crate) const WORD_BITS: usize = 64;

#[inline]
pub fn div_ceil(value: usize, divisor: usize) -> usize {
    debug_assert!(divisor > 0);
    if value == 0 {
        0
    } else {
        1 + (value - 1) / divisor
    }
}

#[inline]
pub(crate) fn lanes_for_width(width_bits: usize) -> usize {
    width_bits
}

#[inline]
pub(crate) fn words_for_vectors(num_vectors: usize) -> usize {
    div_ceil(num_vectors, WORD_BITS)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    num_vectors: usize,
    words_per_lane: usize,
    storage: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedIndex {
    offset: usize,
    lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBitMask {
    words: Vec<u64>,
}

#[derive(Debug, Error)]
pub enum PackedError {
    #[error(
        "packed buffers must have the same shape (lanes {expected_lanes}x{expected_words}, got {actual_lanes}x{actual_words})"
    )]
    ShapeMismatch {
        expected_lanes: usize,
        expected_words: usize,
        actual_lanes: usize,
        actual_words: usize,
    },
}

impl Packed {
    pub fn new(num_vectors: usize) -> Self {
        Self {
            num_vectors,
            words_per_lane: words_for_vectors(num_vectors),
            storage: Vec::new(),
        }
    }

    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    #[inline]
    pub fn words_per_lane(&self) -> usize {
        self.words_per_lane
    }

    #[inline]
    pub fn total_lanes(&self) -> usize {
        if self.words_per_lane == 0 {
            0
        } else {
            self.storage.len() / self.words_per_lane
        }
    }

    pub fn allocate(&mut self, width_bits: usize) -> PackedIndex {
        let lanes = lanes_for_width(width_bits);
        let offset = self.storage.len();
        self.storage.resize(offset + lanes * self.words_per_lane, 0);
        PackedIndex { offset, lanes }
    }

    pub fn duplicate_layout(&self) -> Self {
        Self {
            num_vectors: self.num_vectors,
            words_per_lane: self.words_per_lane,
            storage: vec![0; self.storage.len()],
        }
    }

    #[inline]
    fn lane_offset(&self, index: PackedIndex, lane: usize) -> usize {
        debug_assert!(lane < index.lanes);
        index.offset + lane * self.words_per_lane
    }

    pub fn lane(&self, index: PackedIndex, lane: usize) -> &[u64] {
        let start = self.lane_offset(index, lane);
        let end = start + self.words_per_lane;
        &self.storage[start..end]
    }

    pub fn lane_mut(&mut self, index: PackedIndex, lane: usize) -> &mut [u64] {
        let start = self.lane_offset(index, lane);
        let end = start + self.words_per_lane;
        &mut self.storage[start..end]
    }

    #[inline]
    pub fn slice(&self, index: PackedIndex) -> &[u64] {
        let end = index.offset + index.lanes * self.words_per_lane;
        &self.storage[index.offset..end]
    }

    #[inline]
    pub fn slice_mut(&mut self, index: PackedIndex) -> &mut [u64] {
        let end = index.offset + index.lanes * self.words_per_lane;
        &mut self.storage[index.offset..end]
    }

    pub fn copy_from(&mut self, other: &Packed) -> Result<(), PackedError> {
        if self.words_per_lane != other.words_per_lane || self.storage.len() != other.storage.len()
        {
            return Err(PackedError::ShapeMismatch {
                expected_lanes: self.total_lanes(),
                expected_words: self.words_per_lane,
                actual_lanes: other.total_lanes(),
                actual_words: other.words_per_lane,
            });
        }

        self.storage.copy_from_slice(&other.storage);
        Ok(())
    }

    #[inline]
    pub(crate) fn storage(&self) -> &[u64] {
        &self.storage
    }

    #[inline]
    pub(crate) fn storage_mut(&mut self) -> &mut [u64] {
        &mut self.storage
    }
}

impl PackedIndex {
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    #[inline]
    pub fn lane_offset(&self, lane: usize, words_per_lane: usize) -> usize {
        debug_assert!(lane < self.lanes);
        self.offset + lane * words_per_lane
    }

    #[inline]
    pub fn word_offset(&self, lane: usize, word: usize, words_per_lane: usize) -> usize {
        debug_assert!(words_per_lane > 0);
        debug_assert!(word < words_per_lane);
        self.lane_offset(lane, words_per_lane) + word
    }
}

impl PackedBitMask {
    pub fn new(num_vectors: usize) -> Self {
        Self {
            words: vec![0; words_for_vectors(num_vectors)],
        }
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

#[inline]
pub(crate) fn mask_for_word(word_index: usize, words_per_lane: usize, num_vectors: usize) -> u64 {
    if words_per_lane == 0 {
        return 0;
    }

    if word_index + 1 < words_per_lane {
        return u64::MAX;
    }

    let remainder = num_vectors % WORD_BITS;
    if remainder == 0 {
        if num_vectors == 0 {
            0
        } else {
            u64::MAX
        }
    } else {
        (1u64 << remainder) - 1
    }
}
