// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(missing_docs)]

/*!
fffft is a finite-field fast Fourier transform implementation
for sequences of values that implement the [ff::PrimeField] trait.
*/

use err_derive::Error;
use itertools::iterate;

/// Err variant from FFT functions
#[derive(Debug, Error)]
pub enum FFTError {
    /// Input length is not a power of 2
    #[error(display = "Non--power-of-two input length")]
    NotPowerOfTwo,
    /// Unsupported FFT length for this field
    #[error(display = "Input length greater than 2-adicity of field")]
    TooBig,
    /// Unknown error
    #[error(display = "unknown")]
    Unknown,
}

/// a field that supports an FFT
pub trait FieldFFT: ff::Field {
    /// For a field of characteristic q, S is the unique
    /// value such that 2^S * t = q - 1, with t odd.
    const S: u32;

    /// Returns a 2^S'th root of unity.
    fn root_of_unity() -> Self;

    /// in-place fft, out of order result
    fn fft_i_o<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        // basic checks
        let xi = xi.as_mut();
        if !xi.len().is_power_of_two() {
            return Err(FFTError::NotPowerOfTwo);
        }
        let log_len = (xi.len() as f64).log2().round() as u32;
        if (1 << log_len) != xi.len() {
            return Err(FFTError::Unknown);
        }
        if log_len > <Self as FieldFFT>::S {
            return Err(FFTError::TooBig);
        }

        // compute the needed roots of unity
        let mut gap = xi.len() / 2;
        let roots_of_unity: Vec<Self> =
            iterate(Self::one(), |&v| v * <Self as FieldFFT>::root_of_unity())
                .take(gap)
                .collect();

        while gap > 0 {
            let nchunks = xi.len() / (2 * gap);
            for cidx in 0..nchunks {
                let offset = 2 * cidx * gap;
                for idx in 0..gap {
                    let neg = xi[offset + idx] - xi[offset + idx + gap];
                    xi[offset + idx] += xi[offset + idx + gap];
                    xi[offset + idx + gap] = neg * roots_of_unity[nchunks * idx];
                }
            }
            gap /= 2;
        }

        Ok(())
    }
}
