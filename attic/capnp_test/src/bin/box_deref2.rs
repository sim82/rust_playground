extern crate capnp_test;

pub trait InnerTrait<'a> {
    fn bla(&self);
}

pub trait OuterTrait<'a> {
    fn get(&'a self) -> Box<InnerTrait<'a> + 'a>;
}

struct Inner<'a> {
    u: &'a u32,
}

pub struct Outer<'a> {
    x: &'a u32,
}

impl<'a> OuterTrait<'a> for Outer<'a> {
    fn get(&'a self) -> Box<InnerTrait + 'a> {
        Box::new(Inner { u: &self.x }) //, u : &self.x })
    }
}

impl<'a> InnerTrait<'a> for Inner<'a> {
    fn bla(&self) {
        println!("bla {}", self.u);
    }
}

pub fn create_outer<'a>(i: &'a u32) -> Box<Outer<'a>> {
    Box::new(Outer { x: i })
}

fn main() {
    println!("Hello, world!");

    let v: u32 = 123;
    let o = create_outer(&v);
    let i = o.get();
    i.bla();
}
