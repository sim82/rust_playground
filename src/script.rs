use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};

#[derive(Clone)]
pub enum BindingAction {
    Update(String, String, Option<String>),
}

pub struct Environment {
    subscriptions: Vec<Sender<BindingAction>>,
    variables: HashMap<String, String>,
}

pub struct BindingDispatcher {
    rx: Receiver<BindingAction>,
    pub tx: Sender<BindingAction>,

    callbacks: HashMap<String, Box<FnMut(String, String, Option<String>)>>,
}

impl BindingDispatcher {
    pub fn new() -> Self {
        let (tx, rx) = channel();

        BindingDispatcher {
            tx: tx,
            rx: rx,
            callbacks: HashMap::new(),
        }
    }

    pub fn add_callback<CB: FnMut(String, String, Option<String>) + 'static>(
        &mut self,
        name: &str,
        c: CB,
    ) {
        self.callbacks.insert(name.into(), Box::new(c));
    }

    pub fn dispatch(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(BindingAction::Update(name, value, old)) => {
                    match self.callbacks.get_mut(&name) {
                        Some(cb) => (*cb)(name, value, old),
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
