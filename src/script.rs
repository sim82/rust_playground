use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Clone)]
pub enum BindingAction {
    Update(String, String, Option<String>),
}

pub struct Environment {
    subscriptions: Vec<Sender<BindingAction>>,
    pub variables: HashMap<String, String>,
}

#[derive(Default)]
pub struct Value {
    v: String,
}

impl Value {
    fn new_internal(v: &str) -> Value {
        Value { v: v.into() }
    }
    pub fn new<T: std::fmt::Display>(v: T) -> Value {
        Value {
            v: format!("{}", v),
        }
    }

    pub fn get<T: std::str::FromStr + std::default::Default>(&self) -> T {
        match self.v.parse::<T>() {
            Ok(v) => v,
            _ => std::default::Default::default(), //panic!("conversion failed"), // todo: default value
        }
    }
}

pub struct ValueWatch {
    value: Value,
    updated: bool,
}

impl ValueWatch {
    pub fn new() -> Rc<RefCell<ValueWatch>> {
        Rc::new(RefCell::new(ValueWatch {
            value: std::default::Default::default(),
            updated: true,
        }))
    }

    pub fn set(&mut self, v: Value) {
        self.value = v;
        self.updated = true;
    }

    pub fn get_update<T: std::str::FromStr + std::default::Default>(&mut self) -> Option<T> {
        if self.updated {
            self.updated = false;
            Some(self.value.get::<T>())
        } else {
            None
        }
    }
}

pub struct BindingDispatcher {
    rx: Receiver<BindingAction>,

    callbacks: HashMap<String, Box<FnMut(String, String, Option<String>)>>,
    i32_bindings: HashMap<String, Rc<RefCell<i32>>>,

    value_bindings: HashMap<String, Rc<RefCell<ValueWatch>>>,
}

impl BindingDispatcher {
    pub fn new(rx: Receiver<BindingAction>) -> Self {
        BindingDispatcher {
            rx: rx,
            callbacks: HashMap::new(),
            i32_bindings: HashMap::new(),
            value_bindings: HashMap::new(),
        }
    }

    pub fn add_callback<CB: FnMut(String, String, Option<String>) + 'static>(
        &mut self,
        name: &str,
        c: CB,
    ) {
        self.callbacks.insert(name.into(), Box::new(c));
    }
    pub fn bind_i32(&mut self, name: &str, i: Rc<RefCell<i32>>) {
        self.i32_bindings.insert(name.into(), i);
    }

    pub fn bind_value(&mut self, name: &str, v: Rc<RefCell<ValueWatch>>) {
        self.value_bindings.insert(name.into(), v);
    }
    pub fn dispatch(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(BindingAction::Update(name, value, old)) => {
                    match self.callbacks.get_mut(&name) {
                        Some(cb) => (*cb)(name.clone(), value.clone(), old),
                        _ => (),
                    }
                    match self.i32_bindings.get_mut(&name) {
                        Some(binding) => {
                            binding.replace(value.parse::<i32>().unwrap());
                        }
                        _ => (),
                    }

                    match self.value_bindings.get_mut(&name) {
                        Some(binding) => {
                            // binding.replace(value.parse::<i32>().unwrap());
                            binding.borrow_mut().set(Value::new_internal(&*value));
                        }
                        _ => (),
                    }
                }
                _ => break,
            }
        }
    }
}

impl Environment {
    pub fn new() -> Self {
        Environment {
            subscriptions: Vec::new(),
            variables: HashMap::new(),
        }
    }

    pub fn get(&self, str: &str) -> String {
        match self.variables.get(str) {
            Some(v) => v.clone(),
            None => "".into(),
        }
    }

    pub fn set(&mut self, name: &str, value: &str) {
        let old = self.variables.insert(name.into(), value.into());

        self.send_all(BindingAction::Update(name.into(), value.into(), old));
    }

    pub fn subscribe(&mut self, sender: Sender<BindingAction>) {
        self.subscriptions.push(sender);
    }

    fn send_all(&mut self, action: BindingAction) {
        self.subscriptions
            .drain_filter(|sub| sub.send(action.clone()).is_err());
    }
}

pub fn parse(line: &str, env: &mut Environment) {
    let token = line.split_whitespace().collect::<Vec<_>>();

    if token.len() >= 3 && token[0] == "set" {
        env.set(token[1], token[2])
    }
}
