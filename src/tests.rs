// Copyright 2020 Riad S. Wahby <rsw@cs.stanford.edu>
//
// This file is part of fffft.
//
// Licensed under the Apache License, Version 2.0 (see
// LICENSE or https://www.apache.org/licenses/LICENSE-2.0).
// This file may not be copied, modified, or distributed
// except according to those terms.

use super::FieldFFT;

use ff::PrimeField;

mod ft {
    use ff::PrimeField;
    #[derive(PrimeField)]
    #[PrimeFieldModulus = "17"]
    #[PrimeFieldGenerator = "3"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 1]);
}

/*
mod fr {
    use ff::PrimeField;
    #[derive(PrimeField)]
    #[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
    #[PrimeFieldGenerator = "7"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Fr([u64; 4]);
}

mod ft2 {
    use ff::PrimeField;
    #[derive(PrimeField)]
    #[PrimeFieldModulus = "70386805592835581672624750593"]
    #[PrimeFieldGenerator = "17"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft([u64; 2]);
}
*/

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

    let p = 17.into();
    let w = {
        let mut tmp: Integer = 3.into();
        let mut lnd = 16 / rug_input.len();
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
fn roundtrip() {
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
    let log_len = super::get_log_len(&fi, <Ft as FieldFFT>::S).unwrap();

    // fft_ii tests
    Ft::fft_ii(&mut fi).unwrap();
    Ft::ifft_ii(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    Ft::fft_ii(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::ifft_oi(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    Ft::fft_ii(&mut fi).unwrap();
    Ft::ifft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    assert_eq!(fi, fi2);

    // fft_io tests
    Ft::fft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::ifft_ii(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    Ft::fft_io(&mut fi).unwrap();
    Ft::ifft_oi(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    Ft::fft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::ifft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    assert_eq!(fi, fi2);

    // fft_oi tests
    super::derange(&mut fi, log_len);
    Ft::fft_oi(&mut fi).unwrap();
    Ft::ifft_ii(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    super::derange(&mut fi, log_len);
    Ft::fft_oi(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::ifft_oi(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    super::derange(&mut fi, log_len);
    Ft::fft_oi(&mut fi).unwrap();
    Ft::ifft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    assert_eq!(fi, fi2);
}

#[test]
fn rev_roundtrip() {
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
    let log_len = super::get_log_len(&fi, <Ft as FieldFFT>::S).unwrap();

    // ifft_ii tests
    Ft::ifft_ii(&mut fi).unwrap();
    Ft::fft_ii(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    Ft::ifft_ii(&mut fi).unwrap();
    Ft::fft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    assert_eq!(fi, fi2);

    Ft::ifft_ii(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::fft_oi(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    // ifft_io tests
    Ft::ifft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::fft_ii(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    Ft::ifft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::fft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    assert_eq!(fi, fi2);

    Ft::ifft_io(&mut fi).unwrap();
    Ft::fft_oi(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    // ifft_oi tests
    super::derange(&mut fi, log_len);
    Ft::ifft_oi(&mut fi).unwrap();
    Ft::fft_ii(&mut fi).unwrap();
    assert_eq!(fi, fi2);

    super::derange(&mut fi, log_len);
    Ft::ifft_oi(&mut fi).unwrap();
    Ft::fft_io(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    assert_eq!(fi, fi2);

    super::derange(&mut fi, log_len);
    Ft::ifft_oi(&mut fi).unwrap();
    super::derange(&mut fi, log_len);
    Ft::fft_oi(&mut fi).unwrap();
    assert_eq!(fi, fi2);
}
