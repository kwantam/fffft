// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::FieldFFT;

use ff::{Field, PrimeField};
#[cfg(feature = "bench")]
use test::Bencher;

mod ft {
    use ff::PrimeField;
    #[derive(PrimeField)]
    #[PrimeFieldModulus = "70386805592835581672624750593"]
    #[PrimeFieldGenerator = "17"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 2]);
}

#[test]
fn rug_check() {
    use ft::*;
    use rand::seq::SliceRandom;
    use rug::Integer;

    let mut input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    input.shuffle(&mut rand::thread_rng());
    input.truncate(1 << (1 + (rand::random::<u8>() % 4)));

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

    rug_fft::naive_ntt(rug_input.as_mut(), &p, &w);
    Ft::fft_ii(&mut input).unwrap();

    let rug_output: Vec<Ft> = rug_input
        .iter()
        .map(|x| Ft::from_str(&x.to_string_radix(10)).unwrap())
        .collect();

    assert_eq!(rug_output, input);
}

#[test]
fn sm_roundtrip() {
    use ft::*;
    use rand::seq::SliceRandom;

    let mut input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    input.shuffle(&mut rand::thread_rng());
    input.truncate(1 << (1 + (rand::random::<u8>() % 4)));

    let mut fi: Vec<Ft> = input
        .iter()
        .map(|x| {
            let sx = format!("{}", x);
            Ft::from_str(&sx).unwrap()
        })
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

#[test]
fn sm_rev_roundtrip() {
    use ft::*;
    use rand::seq::SliceRandom;

    let mut input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    input.shuffle(&mut rand::thread_rng());
    input.truncate(1 << (1 + (rand::random::<u8>() % 4)));

    let mut fi: Vec<Ft> = input
        .iter()
        .map(|x| {
            let sx = format!("{}", x);
            Ft::from_str(&sx).unwrap()
        })
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

#[test]
fn roots_of_unity() {
    use ft::*;
    use itertools::iterate;

    for _ in 0..16 {
        let len = 10 + rand::random::<u32>() % 12;

        // use parallel roots of unity computation
        let mut root = <Ft as FieldFFT>::root_of_unity();
        let s = <Ft as FieldFFT>::S;
        let ret = super::roots_of_unity(root, len, s);

        // compute naively
        for _ in 0..(s - len) {
            root *= root;
        }
        let rfn: Vec<Ft> = iterate(Ft::one(), |&v| v * root)
            .take(1 << (len - 1))
            .collect();

        assert_eq!(ret, rfn);
    }
}

#[cfg(feature = "bench")]
#[bench]
fn roots_of_unity_serial(b: &mut Bencher) {
    use ft::*;
    use itertools::iterate;
    use test::black_box;

    let mut root = <Ft as FieldFFT>::root_of_unity();
    let s = <Ft as FieldFFT>::S;
    let len = 20;
    for _ in 0..(s - len) {
        root *= root;
    }

    b.iter(|| {
        black_box(iterate(Ft::one(), |&v| v * root)
                  .take(1 << (len - 1))
                  .collect::<Vec<Ft>>());
    });
}

#[cfg(feature = "bench")]
#[bench]
fn roots_of_unity_parallel(b: &mut Bencher) {
    use ft::*;
    use test::black_box;

    let root = <Ft as FieldFFT>::root_of_unity();
    let s = <Ft as FieldFFT>::S;
    let len = 20;

    b.iter(|| {
        black_box(super::roots_of_unity(root, len, s));
    });
}
