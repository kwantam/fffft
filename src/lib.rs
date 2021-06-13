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

#![feature(test)]
#[cfg(feature = "bench")]
extern crate test;

#[cfg(all(test, feature = "bench"))]
mod bench;
#[cfg(any(test, feature = "bench"))]
mod tests;

use err_derive::Error;
use ff::Field;
use itertools::iterate;
use rayon::prelude::*;

/// Err variant from FFT functions
#[derive(Debug, Error)]
pub enum FFTError {
    /// Input length is not a power of 2
    #[error(display = "Non--power-of-two input length")]
    NotPowerOfTwo,
    /// Unsupported FFT length for this field
    #[error(display = "Input length greater than 2-adicity of field")]
    TooBig,
    /// Precomputed data has wrong size for this input
    #[error(display = "Precomputed data has wrong size for this input")]
    WrongSizePrecomp,
    /// Unknown error
    #[error(display = "unknown")]
    Unknown,
}

/// Precomputed FFT data (for use with fft_*_pc variants)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FFTPrecomp<T> {
    log_len: u32,
    rou: Vec<T>,
}

impl<T> FFTPrecomp<T> {
    /// Return log_len for this data
    pub fn get_log_len(&self) -> u32 {
        self.log_len
    }
}

/// Precomputed IFFT data (for use with ifft_*_pc variants)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IFFTPrecomp<T> {
    log_len: u32,
    irou: Vec<T>,
    ninv: T,
}

impl<T> IFFTPrecomp<T> {
    /// Return log_len for this data
    pub fn get_log_len(&self) -> u32 {
        self.log_len
    }
}

/// a field that supports an FFT
pub trait FieldFFT: Field {
    /// For a field of characteristic q, S is the unique
    /// value such that 2^S * t = q - 1, with t odd.
    const S: u32;

    /// Returns a 2^S'th root of unity.
    fn root_of_unity() -> Self;

    /// Generate precomputed data for FFT of length `len`
    fn precomp_fft(len: usize) -> Result<FFTPrecomp<Self>, FFTError> {
        fft_precomp(len)
    }

    /// Generate precomputed data for IFFT of length `len`
    fn precomp_ifft(len: usize) -> Result<IFFTPrecomp<Self>, FFTError> {
        ifft_precomp(len)
    }

    /// fft: in-order input, in-order result using precomputed roots of unity
    fn fft_ii_pc<T: AsMut<[Self]>>(mut xi: T, pc: &FFTPrecomp<Self>) -> Result<(), FFTError> {
        fft_help_pc(xi.as_mut(), pc.log_len, &pc.rou[..], FFTOrder::II)
    }

    /// fft: in-order input, out-of-order result using precomputed roots of unity
    fn fft_io_pc<T: AsMut<[Self]>>(mut xi: T, pc: &FFTPrecomp<Self>) -> Result<(), FFTError> {
        fft_help_pc(xi.as_mut(), pc.log_len, &pc.rou[..], FFTOrder::IO)
    }

    /// fft: out-of-order input, in-order result using precomputed roots of unity
    fn fft_oi_pc<T: AsMut<[Self]>>(mut xi: T, pc: &FFTPrecomp<Self>) -> Result<(), FFTError> {
        fft_help_pc(xi.as_mut(), pc.log_len, &pc.rou[..], FFTOrder::OI)
    }

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

    /// ifft: in-order input, in-order result using precomputed roots of unity
    fn ifft_ii_pc<T: AsMut<[Self]>>(mut xi: T, pc: &IFFTPrecomp<Self>) -> Result<(), FFTError> {
        ifft_help_pc(
            xi.as_mut(),
            pc.log_len,
            &pc.irou[..],
            &pc.ninv,
            FFTOrder::II,
        )
    }

    /// ifft: in-order input, out-of-order result using precomputed roots of unity
    fn ifft_io_pc<T: AsMut<[Self]>>(mut xi: T, pc: &IFFTPrecomp<Self>) -> Result<(), FFTError> {
        ifft_help_pc(
            xi.as_mut(),
            pc.log_len,
            &pc.irou[..],
            &pc.ninv,
            FFTOrder::IO,
        )
    }

    /// ifft: out-of-order input, in-order result using precomputed roots of unity
    fn ifft_oi_pc<T: AsMut<[Self]>>(mut xi: T, pc: &IFFTPrecomp<Self>) -> Result<(), FFTError> {
        ifft_help_pc(
            xi.as_mut(),
            pc.log_len,
            &pc.irou[..],
            &pc.ninv,
            FFTOrder::OI,
        )
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
        let log_len = get_log_len(xi.as_mut().len(), Self::S)?;
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

fn fft_precomp<T: FieldFFT>(len: usize) -> Result<FFTPrecomp<T>, FFTError> {
    let log_len = get_log_len(len, T::S)?;
    let rou = roots_of_unity(T::root_of_unity(), log_len, T::S);
    Ok(FFTPrecomp { log_len, rou })
}

fn ifft_precomp<T: FieldFFT>(len: usize) -> Result<IFFTPrecomp<T>, FFTError> {
    let log_len = get_log_len(len, T::S)?;
    let irou = roots_of_unity(T::root_of_unity().invert().unwrap(), log_len, T::S);
    let ninv = n_inv(log_len);
    Ok(IFFTPrecomp {
        log_len,
        irou,
        ninv,
    })
}

#[derive(PartialEq, Eq, Debug)]
enum FFTOrder {
    II,
    IO,
    OI,
}

fn fft_help_pc<T: FieldFFT>(
    xi: &mut [T],
    log_len: u32,
    rou: &[T],
    ord: FFTOrder,
) -> Result<(), FFTError> {
    use FFTOrder::*;

    if xi.len() != (1 << log_len) {
        return Err(FFTError::WrongSizePrecomp);
    }

    if ord == OI {
        oi_help(xi, rou);
    } else {
        io_help(xi, rou);
    }

    if ord == II {
        derange(xi, log_len);
    }

    Ok(())
}

fn fft_help<T: FieldFFT>(xi: &mut [T], ord: FFTOrder) -> Result<(), FFTError> {
    let pc = T::precomp_fft(xi.len())?;
    fft_help_pc(xi, pc.log_len, &pc.rou[..], ord)
}

fn ifft_help_pc<T: FieldFFT>(
    xi: &mut [T],
    log_len: u32,
    irou: &[T],
    ninv: &T,
    ord: FFTOrder,
) -> Result<(), FFTError> {
    use FFTOrder::*;

    if xi.len() != (1 << log_len) {
        return Err(FFTError::WrongSizePrecomp);
    }

    if ord == II {
        derange(xi, log_len);
    }

    if ord == IO {
        io_help(xi, irou);
    } else {
        oi_help(xi, irou);
    }

    for x in xi {
        x.mul_assign(ninv);
    }
    Ok(())
}

fn ifft_help<T: FieldFFT>(xi: &mut [T], ord: FFTOrder) -> Result<(), FFTError> {
    let pc = T::precomp_ifft(xi.len())?;
    ifft_help_pc(xi, pc.log_len, &pc.irou[..], &pc.ninv, ord)
}

fn get_log_len(len: usize, s: u32) -> Result<u32, FFTError> {
    use FFTError::*;

    if !len.is_power_of_two() {
        return Err(NotPowerOfTwo);
    }

    let log_len = 63 - (len as u64).leading_zeros();
    if (1 << log_len) != len {
        return Err(Unknown);
    }

    if log_len > s {
        return Err(TooBig);
    }

    Ok(log_len)
}

// minimum size at which to parallelize.
const LOG_PAR_LIMIT: u32 = 6;
fn roots_of_unity<T: Field>(mut root: T, log_len: u32, rdeg: u32) -> Vec<T> {
    // 2^log_len'th root of unity
    for _ in 0..(rdeg - log_len) {
        root *= root;
    }

    // early exit for short inputs
    if log_len - 1 <= LOG_PAR_LIMIT {
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
    if log_roots.len() <= LOG_PAR_LIMIT as usize {
        out[0] = T::one();
        for idx in 1..out.len() {
            out[idx] = out[idx - 1] * log_roots[0];
        }
        return;
    }

    // recursive case:
    // 1. split log_roots in half
    let (lr_lo, lr_hi) = log_roots.split_at((1 + log_roots.len()) / 2);
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

#[cfg(any(test, feature = "bench"))]
fn roots_of_unity_ser<T: Field>(mut root: T, log_len: u32, rdeg: u32) -> Vec<T> {
    for _ in 0..(rdeg - log_len) {
        root *= root;
    }

    iterate(T::one(), |&v| v * root)
        .take(1 << (log_len - 1))
        .collect()
}

#[cfg(feature = "bench")]
fn io_help_ser<T: Field>(xi: &mut [T], roots: &[T]) {
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

fn io_help<T: Field>(xi: &mut [T], roots: &[T]) {
    let mut gap = xi.len() / 2;
    while gap > 0 {
        // each butterfly cluster uses 2*gap positions
        let nchunks = xi.len() / (2 * gap);
        xi.par_chunks_mut(2 * gap).for_each(|cxi| {
            let (lo, hi) = cxi.split_at_mut(gap);
            lo.par_iter_mut()
                .zip(hi)
                .enumerate()
                .for_each(|(idx, (lo, hi))| {
                    let neg = *lo - *hi;
                    *lo += *hi;
                    *hi = neg * roots[nchunks * idx];
                });
        });
        gap /= 2;
    }
}

#[cfg(feature = "bench")]
fn oi_help_ser<T: Field>(xi: &mut [T], roots: &[T]) {
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

fn oi_help<T: Field>(xi: &mut [T], roots: &[T]) {
    let mut gap = 1;
    while gap < xi.len() {
        let nchunks = xi.len() / (2 * gap);
        xi.par_chunks_mut(2 * gap).for_each(|cxi| {
            let (lo, hi) = cxi.split_at_mut(gap);
            lo.par_iter_mut()
                .zip(hi)
                .enumerate()
                .for_each(|(idx, (lo, hi))| {
                    *hi *= roots[nchunks * idx];
                    let neg = *lo - *hi;
                    *lo += *hi;
                    *hi = neg;
                });
        });
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

fn n_inv<T: Field>(log_len: u32) -> T {
    let mut tmp = <T as Field>::one();
    for _ in 0..log_len {
        tmp = tmp.double();
    }
    tmp.invert().unwrap()
}
