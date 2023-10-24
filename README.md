# fffft

[![Documentation](https://docs.rs/fffft/badge.svg)](https://docs.rs/fffft/)
[![Crates.io](https://img.shields.io/crates/v/fffft.svg)](https://crates.io/crates/fffft)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

Parallel FFT computation for [ff::Field] types via [rayon].

Implementing the trait for other [ff::Field] types is very simple:
define a constant `S` and a function `root_of_unity()` that returns a
field element that is a 2^`S`th primitive roof of unity.
This crate contains a blanket trait impl for [ff::PrimeField].

[ff::Field]: https://docs.rs/ff
[ff::PrimeField]: https://docs.rs/ff
[rayon]: https://docs.rs/rayon

## changelog

- 0.2.0: Dependency updates only. Bumps `ff` to 0.9, `rand` to 0.8, `rand_core` to 0.6, and `bitvec` to 0.20.

- 0.3.0: Dependency updates only. Bumps `ff` to 0.10, `bitvec` to 0.22.

- 0.4.0: Update deps. Add new functions that use precomputed roots of unity.

- 0.5.0: Update deps (`ff` 0.10 => 0.12; `bitvec` 0.22 => 1.0.1)

- 0.6.0: Update `ff` to 0.13, no API changes.

## license

    Copyright 2020 Riad S. Wahby

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
