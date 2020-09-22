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

const LOG_MAX_SMPOW: u32 = 6; // XXX(how big?)
fn roots_of_unity<T: Field>(mut root: T, log_len: u32, rdeg: u32) -> Vec<T> {
    // 2^log_len'th root of unity
    for _ in 0..(rdeg - log_len) {
        root *= root;
    }

    // early exit for short inputs
    if log_len - 1 <= LOG_MAX_SMPOW {
        return iterate(T::one(), |&v| v * root)
            .take(1 << (log_len - 1))
            .collect();
    }

    // w, w^2, w^4, w^8, ..., w^(2^(log_len - 1))
    let log_roots: Vec<T> = iterate(root, |&r| r * r)
        .take(log_len as usize - 1)
        .collect();

    // allocate the return array and start the recursion
    let mut ret = vec![T::default(); 1 << (log_len - 1)];
    rou_rec(ret.as_mut(), log_roots.as_ref());

    ret
}

fn rou_rec<T: Field>(out: &mut [T], log_roots: &[T]) {
    assert_eq!(out.len(), 1 << log_roots.len());

    // base case: just compute the roots sequentially
    if log_roots.len() <= LOG_MAX_SMPOW as usize {
        out[0] = T::one();
        for idx in 1..out.len() {
            out[idx] = out[idx - 1] * log_roots[0];
        }
        return;
    }

    // recursive case:
    // 1. split log_roots in half
    let (lr_lo, lr_hi) = log_roots.split_at(log_roots.len() / 2);
    let mut scr_lo = vec![T::default(); 1 << lr_lo.len()];
    let mut scr_hi = vec![T::default(); 1 << lr_hi.len()];
    // 2. compute each half individually
    rayon::join(
        || rou_rec(scr_lo.as_mut(), lr_lo),
        || rou_rec(scr_hi.as_mut(), lr_hi),
    );
    // 3. recombine halves
    out.par_chunks_mut(scr_lo.len())
        .enumerate()
        .for_each(|(idx, rt)| {
            for jdx in 0..rt.len() {
                rt[jdx] = scr_hi[idx] * scr_lo[jdx];
            }
        });
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
