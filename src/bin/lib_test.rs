//extern crate asset_bundle;

extern crate capnp_test;
use capnp_test::asset_bundle::AssetBundleAccess;
use capnp_test::asset_bundle::AssetRef;


fn main() {
	let bundle = capnp_test::asset_bundle::AssetBundleFile::open("test.bin"); 
	let access = bundle.access();
	//let bundle = capnp_test::asset_bundle::MappedBundle::open("test.bin");

	{
		// let r = access.get_by_name2("model/test/map_color_2.png");
		access.bla();
	// println!("{}", r.get_header().unwrap().get_name().unwrap());
	}
	// drop(r);

}