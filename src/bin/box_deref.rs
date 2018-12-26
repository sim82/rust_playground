extern crate capnp_test;


use capnp_test::deref::{create_outer, Factory, InnerTrait, OuterTrait};

fn main() {
    println!("Hello, world!");

    //let f = Factory::new();

    //let o = f.get();
    let v : u32 = 666;
    let o = create_outer(&v);
    let i = o.get();
    i.bla();

}