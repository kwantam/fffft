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
fn simple() {
    use ft::*;
    use rug::Integer;

    let mut rug_input: Vec<Integer> = vec![1, 9, 13, 2, 7, 5, 4, 8]
        .into_iter()
        .map(Integer::from)
        .collect();

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
    Ft::fft_i(&mut input).unwrap();

    let rug_output: Vec<Ft> = rug_input
        .iter()
        .map(|x| Ft::from_str(&x.to_string_radix(10)).unwrap())
        .collect();

    assert_eq!(rug_output, input);
}

#[test]
fn roundtrip() {
    use ft::*;

    let input = vec![1, 7, 4, 8, 9, 16, 2, 11];
    let mut fi: Vec<Ft> = input
        .iter()
        .map(|x| {
            let sx = format!("{}", x);
            Ft::from_str(&sx).unwrap()
        })
        .collect();
    let fi2 = fi.clone();

    Ft::fft_i(&mut fi).unwrap();
    Ft::ifft_i(&mut fi).unwrap();

    assert_eq!(fi, fi2);
}
