trait TestTrait {
    fn get<'a>(&'a self) -> &'a u32;
}

struct Test<'a> {
    x: &'a u32,
}

impl TestTrait for Test<'_> {
    fn get<'a>(&'a self) -> &'a u32 {
        self.x
    }
}

fn create_test<'a>(i: &'a u32) -> Box<TestTrait + 'a> {
    Box::new(Test { x: i })
}

// // same error:
// fn create_test<'a>( i : &'a u32) -> impl TestTrait<'a> {
// 	Test{ x : i }
// }

// // works:
// fn create_test<'a>( i : &'a u32) -> Test<'a> {
// 	Test{ x : i }
// }

// // works too:
// fn create_test<'a>( i : &'a u32) -> Box<Test<'a>> {
// 	Box::new(Test{ x : i })
// }

fn main() {
    println!("Hello, world!");

    let v: u32 = 123;
    let o = create_test(&v);
    let i = o.get();
    println!("{}", i);
}
