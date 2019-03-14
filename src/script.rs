use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Default, Clone)]
pub struct Value {
    v: String,
}

#[derive(Clone)]
pub enum BindingAction {
    Update(String, Value, Option<Value>),
}

pub struct Environment {
    subscriptions: Vec<Sender<BindingAction>>,
    pub variables: HashMap<String, Value>,
}

impl Value {
    fn new_internal(v: &str) -> Value {
        Value { v: v.into() }
    }
    pub fn new<T: ToValue>(v: T) -> Value {
        v.to_value()
    }

    pub fn get<T: FromValue>(&self) -> T {
        T::from_value(self)
    }
}

pub trait ToValue {
    fn to_value(&self) -> Value;
}

pub trait FromValue {
    fn from_value(value: &Value) -> Self;
}

impl ToValue for i32 {
    fn to_value(&self) -> Value {
        Value {
            v: self.to_string(),
        }
    }
}

impl FromValue for i32 {
    fn from_value(value: &Value) -> Self {
        match Self::from_str(&value.v) {
            Ok(v) => v,
            _ => Self::default(),
        }
    }
}

impl<T: std::fmt::Display> ToValue for cgmath::Point3<T> {
    fn to_value(&self) -> Value {
        Value {
            v: format!("({} {} {})", self.x, self.y, self.z),
        }
    }
}

impl<T: std::str::FromStr + std::default::Default> FromValue for cgmath::Point3<T> {
    fn from_value(value: &Value) -> Self {
        let v = &value.v;

        if !(v.starts_with("(") && v.ends_with(")")) {
            return Self::new(T::default(), T::default(), T::default());
        }

        let v = &v[1..v.len() - 1];
        let comps = v.split_whitespace().collect::<Vec<_>>();

        if comps.len() != 3 {
            return Self::new(T::default(), T::default(), T::default());
        }

        match (
            T::from_str(comps[0]),
            T::from_str(comps[1]),
            T::from_str(comps[2]),
        ) {
            (Ok(c0), Ok(c1), Ok(c2)) => Self::new(c0, c1, c2),
            _ => Self::new(T::default(), T::default(), T::default()),
        }
    }
}

impl ToValue for str {
    fn to_value(&self) -> Value {
        Value { v: self.into() }
    }
}

impl FromValue for String {
    fn from_value(value: &Value) -> Self {
        match Self::from_str(&value.v) {
            Ok(v) => v.clone(),
            _ => Self::default(),
        }
    }
}

impl<T: ToValue> From<T> for Value {
    fn from(v: T) -> Self {
        v.to_value()
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Display::fmt(&self.v, fmt)
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Debug::fmt(&self.v, fmt)
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

    pub fn get_update<T: FromValue>(&mut self) -> Option<T> {
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

    callbacks: HashMap<String, Box<FnMut(String, Value, Option<Value>)>>,
    // i32_bindings: HashMap<String, Rc<RefCell<i32>>>,
    value_bindings: HashMap<String, Rc<RefCell<ValueWatch>>>,
}

impl BindingDispatcher {
    pub fn new(rx: Receiver<BindingAction>) -> Self {
        BindingDispatcher {
            rx: rx,
            callbacks: HashMap::new(),
            // i32_bindings: HashMap::new(),
            value_bindings: HashMap::new(),
        }
    }

    pub fn add_callback<CB: FnMut(String, Value, Option<Value>) + 'static>(
        &mut self,
        name: &str,
        c: CB,
    ) {
        self.callbacks.insert(name.into(), Box::new(c));
    }
    // pub fn bind_i32(&mut self, name: &str, i: Rc<RefCell<i32>>) {
    //     self.i32_bindings.insert(name.into(), i);
    // }

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
                    // match self.i32_bindings.get_mut(&name) {
                    //     Some(binding) => {
                    //         binding.replace(value.parse::<i32>().unwrap());
                    //     }
                    //     _ => (),
                    // }

                    match self.value_bindings.get_mut(&name) {
                        Some(binding) => {
                            // binding.replace(value.parse::<i32>().unwrap());
                            binding.borrow_mut().set(value);
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

    pub fn get(&self, str: &str) -> Value {
        match self.variables.get(str) {
            Some(v) => v.clone(),
            None => Value::new_internal(""),
        }
    }

    // pub fn set(&mut self, name: &str, value: &str) {
    //     let old = self.variables.insert(name.into(), value.into());

    //     self.send_all(BindingAction::Update(name.into(), value.into(), old));
    // }

    pub fn set(&mut self, name: &str, v: Value) {
        let old = self.variables.insert(name.into(), v.clone());
        self.send_all(BindingAction::Update(name.into(), v.clone(), old));
    }

    pub fn subscribe(&mut self, sender: Sender<BindingAction>) {
        self.subscriptions.push(sender);
    }

    fn send_all(&mut self, action: BindingAction) {
        self.subscriptions
            .drain_filter(|sub| sub.send(action.clone()).is_err());
    }
}

pub enum ScriptToken {
    Set,
    Get,
    Variable(String),
    Value(String),
}

pub enum CompletionQuery {
    None,
    Variable(String),
}
// pub struct CompletionDesc(CompletionQuery, usize);

pub fn tokenize(line: &str) -> Option<VecDeque<ScriptToken>> {
    let mut out = VecDeque::new();
    let mut tokens = line.split_whitespace().collect::<VecDeque<_>>();
    if tokens.is_empty() {
        return None;
    }

    let token = tokens.pop_front().unwrap();
    if token == "set" {
        out.push_back(ScriptToken::Set);

        if !tokens.is_empty() {
            out.push_back(ScriptToken::Variable(tokens.pop_front().unwrap().into()));

            if !tokens.is_empty() {
                out.push_back(ScriptToken::Value(tokens.pop_front().unwrap().into()));
            }
        }
    } else if token == "get" {
        out.push_back(ScriptToken::Get);

        if !tokens.is_empty() {
            out.push_back(ScriptToken::Variable(tokens.pop_front().unwrap().into()));
        }
    }

    Some(out)
}

pub fn parse(line: &str, env: &mut Environment) {
    // let token = line.split_whitespace().collect::<Vec<_>>();

    // if token.len() >= 3 && token[0] == "set" {
    //     env.set(token[1], token[2].to_value())
    // } else if token.len() >= 2 && token[0] == "print" {
    //     println!("{}: {}", token[1], env.get(token[1]));
    // }

    let mut tokens = tokenize(line);

    if tokens.is_none() {
        return;
    }
    let mut tokens = tokens.unwrap();

    match tokens.pop_front() {
        Some(ScriptToken::Set) => match tokens.pop_front() {
            Some(ScriptToken::Variable(variable)) => match tokens.pop_front() {
                Some(ScriptToken::Value(value)) => {
                    env.set(&variable, value.to_value());
                }
                _ => (),
            },
            _ => (),
        },
        _ => (),
    }
}

pub fn complete(line: &str, env: &Environment) -> Vec<String> {
    vec![line.into()]
}
