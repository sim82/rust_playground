#![feature(drain_filter)]

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate itertools;

#[cfg(feature = "capnp_bits")]
pub mod capnp_bits;
pub mod crystal;
pub mod render_bits;

#[cfg(feature = "capnp_bits")]
pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}
#[cfg(feature = "capnp_bits")]
pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}
