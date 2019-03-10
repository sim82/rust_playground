use rust_playground::script;
use rust_playground::script::{BindingDispatcher, Environment, ToValue};
use std::string::ToString;
use std::sync::mpsc::channel;

type Point3 = cgmath::Point3<f32>;

fn main() {
    let mut e = Environment::new();

    let (tx, rx) = channel();
    let mut b = BindingDispatcher::new(rx);

    e.subscribe(tx.clone());

    b.add_callback("test", |name, value, old| {
        println!("set1 {} {} {:?}", name, value, old)
    });
    b.add_callback("test2", |name, value, old| {
        println!("set2 {} {} {:?}", name, value, old)
    });

    e.set("test", "v1".to_value());

    e.set("test2", "v2".to_value());

    e.set("test", Point3::new(1f32, 2f32, 3f32).to_value());
    e.set("test", Point3::new(666f32, 42f32, 1234f32).to_value());

    // b.tx.send(BindingAction::Update("test".into(), "x".into()));
    // b.tx.send(BindingAction::Update("test2".into(), "y".into()));

    b.dispatch();

    // println!(
    //     "{}",
    //     // (1f32, 2f32, 3f32).to_string() //
    //     Point3::new(1f32, 2f32, 3f32).to_string()
    // );

    println!("{:?}", Point3::new(1f32, 2f32, 3f32));
}

// impl std::fmt::Display for Point3 {}
