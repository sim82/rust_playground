use rust_playground::script::{BindingAction, BindingDispatcher, Environment};

fn main() {
    let mut e = Environment::new();

    let mut b = BindingDispatcher::new();

    e.subscribe(b.tx.clone());

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
