#[cfg(feature = "capnp_bits")]
extern crate capnpc;

fn main() {
    #[cfg(feature = "capnp_bits")]
    ::capnpc::CompilerCommand::new()
        .edition(capnpc::RustEdition::Rust2018)
        .file("capnp/asset.capnp")
        .file("capnp/scene.capnp")
        .run()
        .expect("compiling schema");
}
