extern crate capnpc;

fn main() {
    ::capnpc::CompilerCommand::new()
        .file("capnp/asset.capnp")
        .file("capnp/scene.capnp")
        .run()
        .expect("compiling schema");
}
