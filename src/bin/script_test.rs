use rust_playground::script::{BindingDispatcher, Environment};
use std::sync::mpsc::channel;

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

    e.set("test", "v1");

    e.set("test2", "v2");

    e.set("test", "v3");

    // b.tx.send(BindingAction::Update("test".into(), "x".into()));
    // b.tx.send(BindingAction::Update("test2".into(), "y".into()));

    b.dispatch();
}
