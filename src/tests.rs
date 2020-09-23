// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::FieldFFT;
use ft::*;

use ff::{Field, PrimeField};
use std::iter::repeat_with;

pub(crate) mod ft {
    use ff::PrimeField;
    #[derive(PrimeField)]
    #[PrimeFieldModulus = "70386805592835581672624750593"]
    #[PrimeFieldGenerator = "17"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 2]);
}

#[test]
fn rug_fft() {
    use rand::Rng;
    use rug::Integer;
    let mut rng = rand::thread_rng();
    for _ in 0..16 {
        let len = 8 + rand::random::<u32>() % 8;
        let input: Vec<u64> = repeat_with(|| rng.gen()).take(1 << len).collect();
        let mut rug_input: Vec<Integer> = input.into_iter().map(Integer::from).collect();
        let mut input: Vec<Ft> = rug_input
            .iter()
            .map(|x| Ft::from_str(&x.to_string_radix(10)).unwrap())
            .collect();

        let p = Integer::from_str_radix("70386805592835581672624750593", 10).unwrap();
        let w = {
            // ```sage
            // p = 70386805592835581672624750593
            // assert 16388205250919699127 * 2^32 + 1 == p
            // F = GF(p)
            // pr = F.primitive_element()
            // w = pr ^ 16388205250919699127
            // assert w == F(48006014286678626680047775496)
            // ```
            let mut tmp = Integer::from_str_radix("48006014286678626680047775496", 10).unwrap();
            let mut lnd = (1u64 << 32) / (rug_input.len() as u64);
            while lnd > 1 {
                tmp.square_mut();
                tmp %= &p;
                lnd /= 2;
            }
            tmp
        };

        rug_fft::bit_rev_radix_2_ntt(rug_input.as_mut(), &p, &w);
        Ft::fft_ii(&mut input).unwrap();

        let rug_output: Vec<Ft> = rug_input
            .iter()
            .map(|x| Ft::from_str(&x.to_string_radix(10)).unwrap())
            .collect();
        assert_eq!(rug_output, input);
    }
}

#[test]
fn rug_ifft() {
    use rand::Rng;
    use rug::Integer;
    let mut rng = rand::thread_rng();
    for _ in 0..16 {
        let len = 8 + rand::random::<u32>() % 8;
        let input: Vec<u64> = repeat_with(|| rng.gen()).take(1 << len).collect();
        let mut rug_input: Vec<Integer> = input.into_iter().map(Integer::from).collect();
        let mut input: Vec<Ft> = rug_input
            .iter()
            .map(|x| Ft::from_str(&x.to_string_radix(10)).unwrap())
            .collect();

        let p = Integer::from_str_radix("70386805592835581672624750593", 10).unwrap();
        let w = {
            // ```sage
            // p = 70386805592835581672624750593
            // assert 16388205250919699127 * 2^32 + 1 == p
            // F = GF(p)
            // pr = F.primitive_element()
            // w = pr ^ 16388205250919699127
            // assert w == F(48006014286678626680047775496)
            // ```
            let mut tmp = Integer::from_str_radix("48006014286678626680047775496", 10).unwrap();
            let mut lnd = (1u64 << 32) / (rug_input.len() as u64);
            while lnd > 1 {
                tmp.square_mut();
                tmp %= &p;
                lnd /= 2;
            }
            tmp
        };

        rug_fft::bit_rev_radix_2_intt(rug_input.as_mut(), &p, &w);
        Ft::ifft_ii(&mut input).unwrap();

        let rug_output: Vec<Ft> = rug_input
            .iter()
            .map(|x| Ft::from_str(&x.to_string_radix(10)).unwrap())
            .collect();
        assert_eq!(rug_output, input);
    }
}

#[test]
fn roundtrip() {
    let mut rng = rand::thread_rng();
    for _ in 0..16 {
        let len = 8 + rand::random::<u32>() % 8;
        let mut fi: Vec<Ft> = repeat_with(|| Ft::random(&mut rng))
            .take(1 << len)
            .collect();
        let fi2 = fi.clone();

        // fft_ii tests
        Ft::fft_ii(&mut fi).unwrap();
        Ft::ifft_ii(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::fft_ii(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::ifft_oi(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::fft_ii(&mut fi).unwrap();
        Ft::ifft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        // fft_io tests
        Ft::fft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::ifft_ii(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::fft_io(&mut fi).unwrap();
        Ft::ifft_oi(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::fft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::ifft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        // fft_oi tests
        Ft::derange(&mut fi).unwrap();
        Ft::fft_oi(&mut fi).unwrap();
        Ft::ifft_ii(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::derange(&mut fi).unwrap();
        Ft::fft_oi(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::ifft_oi(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::derange(&mut fi).unwrap();
        Ft::fft_oi(&mut fi).unwrap();
        Ft::ifft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        assert_eq!(fi, fi2);
    }
}

#[test]
fn rev_roundtrip() {
    let mut rng = rand::thread_rng();
    for _ in 0..16 {
        let len = 8 + rand::random::<u32>() % 8;
        let mut fi: Vec<Ft> = repeat_with(|| Ft::random(&mut rng))
            .take(1 << len)
            .collect();
        let fi2 = fi.clone();

        // ifft_ii tests
        Ft::ifft_ii(&mut fi).unwrap();
        Ft::fft_ii(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::ifft_ii(&mut fi).unwrap();
        Ft::fft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::ifft_ii(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::fft_oi(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        // ifft_io tests
        Ft::ifft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::fft_ii(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::ifft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::fft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::ifft_io(&mut fi).unwrap();
        Ft::fft_oi(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        // ifft_oi tests
        Ft::derange(&mut fi).unwrap();
        Ft::ifft_oi(&mut fi).unwrap();
        Ft::fft_ii(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::derange(&mut fi).unwrap();
        Ft::ifft_oi(&mut fi).unwrap();
        Ft::fft_io(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        assert_eq!(fi, fi2);

        Ft::derange(&mut fi).unwrap();
        Ft::ifft_oi(&mut fi).unwrap();
        Ft::derange(&mut fi).unwrap();
        Ft::fft_oi(&mut fi).unwrap();
        assert_eq!(fi, fi2);
    }
}

#[test]
fn roots_of_unity() {
    for _ in 0..16 {
        let len = 10 + rand::random::<u32>() % 12;

        // use parallel roots of unity computation
        let root = <Ft as FieldFFT>::root_of_unity();
        let s = <Ft as FieldFFT>::S;
        let ret = super::roots_of_unity(root, len, s);

        // compute naively
        let rfn = super::roots_of_unity_ser(root, len, s);

        assert_eq!(ret, rfn);
    }
}
