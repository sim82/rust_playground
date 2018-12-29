//extern crate asset_bundle;

extern crate capnp;
extern crate capnp_test;

use capnp_test::asset_bundle::AssetBundleAccess;
use capnp_test::asset_capnp::{asset, asset_bundle, asset_pixel_data};

pub fn print_asset(asset: asset::Reader) -> capnp::Result<()> {
    println!("name: {}", asset.get_header()?.get_name()?);

    match asset.which() {
        Ok(asset::PixelData(Ok(pixel_data))) => {
            println!("pixel data");
            match pixel_data.which() {
                Ok(asset_pixel_data::Stored(_stored)) => {
                    println!("stored");
                }
                Ok(asset_pixel_data::Cooked(_cooked)) => {
                    println!("cooked");
                }
                Err(::capnp::NotInSchema(_)) => {}
            }

            // println!("pixel data")
        }
        // Ok(asset::PixelData(Ok(asset_pixel_data::Stored(Ok(stored))))) => {

        // 	println!("stored pixel data")
        // }
        _ => {} // Err(::capnp::NotInSchema(_)) => { }
    }
    Ok(())
}

fn test(_switch: bool) -> capnp_test::asset_bundle::Result<()> {
    let bundle = capnp_test::asset_bundle::AssetBundleFile::open("test.bin")?;
    let access = bundle.access()?;
    // let access = capnp_test::asset_bundle::owned_access("test.bin")?;
    {
        for (name, i) in access.get_names() {
            println!("{}:", name);
            let r = access.get(i)?;
            print_asset(r);
        }
        let r = access.get_by_name("model/test/map_color_2.png")?;
        // access.bla();
        println!("{}", r.get_header()?.get_name()?);
    }

    Ok(())
}

fn main() {
    match test(true) {
        Err(r) => println!("error: {}", r),
        _ => {}
    }
    // drop(r);
}
