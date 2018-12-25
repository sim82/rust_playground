trait TestTrait {
	fn get<'a>(&'a self) -> &'a u32;
}


struct Struct1 { x : u32 }

impl TestTrait for Struct1 {
	fn get<'a>(&'a self) -> &'a u32 {
		&self.x
	}
}

fn main() {
    println!("Hello, world!");


    let tt = Box::new(Struct1{ x : 666}) as Box<TestTrait>;


    tt.get();
}