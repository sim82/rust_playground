//extern crate asset_bundle;

extern crate capnp_test;
use capnp_test::asset_bundle::AssetBundleAccess;

fn test( _switch : bool) -> capnp_test::asset_bundle::Result<()> {


	let bundle = capnp_test::asset_bundle::AssetBundleFile::open("test2.bin")?;
    let access = bundle.access()?;
    // let access = capnp_test::asset_bundle::owned_access("test.bin")?;
    {
        for name in access.get_names() {
            println!("{}", name);
        }
        let r = access.get_by_name("model/test/map_color_2.png");
        // access.bla();
        println!("{}", r.get_header().unwrap().get_name().unwrap());
    }

    Ok(())
}

fn main() {
    match test(true) {
    	Err(r) => { println!("error: {}", r)}
    	_ => {}
    }
    // drop(r);
}
