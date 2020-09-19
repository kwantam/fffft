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
use ff::Field;
use itertools::iterate;

#[cfg(test)]
mod tests;

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
pub trait FieldFFT: Field {
    /// For a field of characteristic q, S is the unique
    /// value such that 2^S * t = q - 1, with t odd.
    const S: u32;

    /// Returns a 2^S'th root of unity.
    fn root_of_unity() -> Self;

    /// in-place fft, out-of-order result
    fn fft_i_o<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        let xi = xi.as_mut();
        let log_len = get_log_len(xi, <Self as FieldFFT>::S)?;
        fft_help(xi, log_len, <Self as FieldFFT>::root_of_unity());
        Ok(())
    }

    /// in-place fft, in-order result
    fn fft_i<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        let xi = xi.as_mut();
        let log_len = get_log_len(xi, <Self as FieldFFT>::S)?;
        fft_help(xi, log_len, <Self as FieldFFT>::root_of_unity());
        derange(xi, log_len);
        Ok(())
    }

    /// in-place ifft, out-of-order result
    fn ifft_i_o<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        let xi = xi.as_mut();
        let log_len = get_log_len(xi, <Self as FieldFFT>::S)?;
        fft_help(
            xi,
            log_len,
            <Self as FieldFFT>::root_of_unity().invert().unwrap(),
        );
        divide_by_n(xi, log_len);
        Ok(())
    }

    /// in-place ifft, in-order result
    fn ifft_i<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        let xi = xi.as_mut();
        let log_len = get_log_len(xi, <Self as FieldFFT>::S)?;
        fft_help(
            xi,
            log_len,
            <Self as FieldFFT>::root_of_unity().invert().unwrap(),
        );
        divide_by_n(xi, log_len);
        derange(xi, log_len);
        Ok(())
    }
}

fn bitrev(a: u64, log_len: u32) -> u64 {
    a.reverse_bits() >> (64 - log_len)
}

fn get_log_len<T>(xi: &[T], s: u32) -> Result<u32, FFTError> {
    use FFTError::*;

    if !xi.len().is_power_of_two() {
        return Err(NotPowerOfTwo);
    }

    let log_len = 63 - (xi.len() as u64).leading_zeros();
    if (1 << log_len) != xi.len() {
        return Err(Unknown);
    }

    if log_len > s {
        return Err(TooBig);
    }

    Ok(log_len)
}

fn fft_help<T: FieldFFT>(xi: &mut [T], log_len: u32, mut root: T) {
    // compute the needed roots of unity
    let mut gap = xi.len() / 2;
    let roots_of_unity: Vec<T> = {
        for _ in 0..(<T as FieldFFT>::S - log_len) {
            root *= root;
        }
        iterate(T::one(), |&v| v * root).take(gap).collect()
    };

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
}

fn derange<T>(xi: &mut [T], log_len: u32) {
    for idx in 1..(xi.len() as u64 - 1) {
        let ridx = bitrev(idx, log_len);
        if idx < ridx {
            xi.swap(idx as usize, ridx as usize);
        }
    }
}

fn divide_by_n<T: Field>(xi: &mut [T], log_len: u32) {
    let n_inv = {
        let mut tmp = <T as Field>::one();
        for _ in 0..log_len {
            tmp = tmp.double();
        }
        tmp.invert().unwrap()
    };
    for x in xi {
        x.mul_assign(&n_inv);
    }
}

impl<T: ff::PrimeField> FieldFFT for T {
    const S: u32 = <Self as ff::PrimeField>::S;

    fn root_of_unity() -> Self {
        <Self as ff::PrimeField>::root_of_unity()
    }
}
