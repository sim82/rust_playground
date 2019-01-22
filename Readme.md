[![Build Status][s1]][tc] [![MIT/Apache][s3]][li] 

[s1]: https://api.travis-ci.org/sim82/rust_playground.svg?branch=master
[s3]: https://img.shields.io/badge/license-MIT%2FApache-blue.svg
[tc]: https://travis-ci.org/sim82/rust_playground
[li]: COPYING

start with:
```
 RUSTFLAGS="-C target-feature=+avx" cargo run --release --bin crystal_planes
```
** without release mode it will be quite slow **

this should open a render window. 
* Move around with WASD + mouse
* Move light source with IJKL keys.
* press Q or E for party-mode
* press F3 to exit

First goal is to have this look like the [c++ version](https://github.comsim82/shooter2), i.e. like [this](https://youtu.be/wNDT1-M3570).