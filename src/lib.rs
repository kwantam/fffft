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
use rayon::prelude::*;

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

    /// fft: in-order input, in-order result
    fn fft_ii<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        fft_help(xi.as_mut(), FFTOrder::II)
    }

    /// fft: in-order input, out-of-order result
    fn fft_io<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        fft_help(xi.as_mut(), FFTOrder::IO)
    }

    /// fft: out-of-order input, in-order result
    fn fft_oi<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        fft_help(xi.as_mut(), FFTOrder::OI)
    }

    /// ifft: in-order input, in-order result
    fn ifft_ii<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        ifft_help(xi.as_mut(), FFTOrder::II)
    }

    /// ifft: in-order input, out-of-order result
    fn ifft_io<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        ifft_help(xi.as_mut(), FFTOrder::IO)
    }

    /// ifft: out-of-order input, in-order result
    fn ifft_oi<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        ifft_help(xi.as_mut(), FFTOrder::OI)
    }

    /// turn in-order into out-of-order, or vice-versa
    fn derange<T: AsMut<[Self]>>(mut xi: T) -> Result<(), FFTError> {
        let log_len = get_log_len(xi.as_mut(), Self::S)?;
        derange(xi.as_mut(), log_len);
        Ok(())
    }
}

impl<T: ff::PrimeField> FieldFFT for T {
    const S: u32 = <Self as ff::PrimeField>::S;

    fn root_of_unity() -> Self {
        <Self as ff::PrimeField>::root_of_unity()
    }
}

#[derive(PartialEq, Eq, Debug)]
enum FFTOrder {
    II,
    IO,
    OI,
}

fn fft_help<T: FieldFFT>(xi: &mut [T], ord: FFTOrder) -> Result<(), FFTError> {
    use FFTOrder::*;

    let log_len = get_log_len(xi, T::S)?;
    let root_of_unity = T::root_of_unity();

    if ord == OI {
        oi_help(xi, root_of_unity, log_len, T::S);
    } else {
        io_help(xi, root_of_unity, log_len, T::S);
    }

    if ord == II {
        derange(xi, log_len);
    }

    Ok(())
}

fn ifft_help<T: FieldFFT>(xi: &mut [T], ord: FFTOrder) -> Result<(), FFTError> {
    use FFTOrder::*;

    let log_len = get_log_len(xi, T::S)?;
    let root_of_unity = T::root_of_unity().invert().unwrap();

    if ord == II {
        derange(xi, log_len);
    }

    if ord == IO {
        io_help(xi, root_of_unity, log_len, T::S);
    } else {
        oi_help(xi, root_of_unity, log_len, T::S);
    }

    divide_by_n(xi, log_len);
    Ok(())
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

fn rou_base<T: Field>(mut root: T, log_len: u32, log_max: u32, rdeg: u32) -> (T, Vec<T>) {
    // at most log_len - 1
    assert!(log_max < log_len);

    // compute log_len'th root of unity from rdeg'th root of unity, rdeg >= log_len
    for _ in 0..(rdeg - log_len) {
        root *= root;
    }

    // compute remaining roots of unity
    (root, iterate(T::one(), |&v| v * root).take(1 << log_max).collect())
}

fn roots_of_unity<T: Field>(root: T, log_len: u32, rdeg: u32) -> Vec<T> {
    use std::cmp::min;
    const LOG_MAX_SMPOW: u32 = 8; // XXX(how big???)
    const MAX_SMPOW: usize = 1 << LOG_MAX_SMPOW;

    // compute 2^log_len'th root of unity and small powers table
    let log_max = min(log_len - 1, LOG_MAX_SMPOW);
    let (root, small_powers) = rou_base(root, log_len, log_max, rdeg);
    if log_max == log_len - 1 {
        // early return for the easy case
        return small_powers;
    }

    // this is the next root after the ones above
    let base_root = *small_powers.last().unwrap() * root;
    // precompute power-of-two powers of the root of unity
    let max_pow = log_len - 1 - LOG_MAX_SMPOW;
    let log_roots: Vec<T> = iterate(base_root, |&r| r * r).take(max_pow as usize).collect();
    let mut ret = vec![T::one(); 1 << (log_len - 1)];
    ret.par_chunks_mut(MAX_SMPOW).enumerate().for_each(|(idx, rt)| {
        for off in 0..max_pow as usize {
            let bit = idx & (1 << off);
            if bit != 0 {
                rt[0] *= log_roots[off];
            }
        }

        for jdx in 1..MAX_SMPOW {
            rt[jdx] = rt[0] * small_powers[jdx];
        }
    });
    ret
}

fn io_help<T: Field>(xi: &mut [T], root: T, log_len: u32, rdeg: u32) {
    let roots = roots_of_unity(root, log_len, rdeg);

    let mut gap = xi.len() / 2;
    while gap > 0 {
        let nchunks = xi.len() / (2 * gap);
        for cidx in 0..nchunks {
            let offset = 2 * cidx * gap;
            for idx in 0..gap {
                let neg = xi[offset + idx] - xi[offset + idx + gap];
                xi[offset + idx] += xi[offset + idx + gap];
                xi[offset + idx + gap] = neg * roots[nchunks * idx];
            }
        }
        gap /= 2;
    }
}

fn oi_help<T: Field>(xi: &mut [T], root: T, log_len: u32, rdeg: u32) {
    // needed roots of unity
    let roots = roots_of_unity(root, log_len, rdeg);

    let mut gap = 1;
    while gap < xi.len() {
        let nchunks = xi.len() / (2 * gap);
        for cidx in 0..nchunks {
            let offset = 2 * cidx * gap;
            for idx in 0..gap {
                xi[offset + idx + gap] *= roots[nchunks * idx];
                let neg = xi[offset + idx] - xi[offset + idx + gap];
                xi[offset + idx] += xi[offset + idx + gap];
                xi[offset + idx + gap] = neg;
            }
        }
        gap *= 2;
    }
}

fn bitrev(a: u64, log_len: u32) -> u64 {
    a.reverse_bits() >> (64 - log_len)
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
        // hack: compute n :: T
        for _ in 0..log_len {
            tmp = tmp.double();
        }
        tmp.invert().unwrap()
    };
    for x in xi {
        x.mul_assign(&n_inv);
    }
}
