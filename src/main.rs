extern crate capnp;
extern crate memmap;

pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}

pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}

mod test {
    use std::fs::File;
    // use std::io::Read;
    use memmap::MmapOptions;

    use asset_capnp::{asset, asset_bundle, asset_pixel_data};
    use capnp::serialize;
    use capnp::Word;

    pub fn with_reader<F: Fn(asset_bundle::Reader) -> capnp::Result<()>>(
        f: F,
    ) -> capnp::Result<()> {
        let mut file = File::open("test.bin").unwrap();
        let message_reader =
            serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new()).unwrap();
        f(message_reader.get_root::<asset_bundle::Reader>()?)
    }

    pub fn with_reader_mmap<F: Fn(asset_bundle::Reader) -> capnp::Result<()>>( f: F,) -> capnp::Result<()> {
        let file = File::open("test.bin").unwrap();

        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };

        let message = serialize::read_message_from_words(
            unsafe { Word::bytes_to_words(&mmap[..]) },
            ::capnp::message::ReaderOptions::new(),
        )
        .unwrap();
        f(message.get_root::<asset_bundle::Reader>().unwrap())
        // Ok(())
    }

    pub fn print_asset_bundle(asset_bundle: asset_bundle::Reader) -> capnp::Result<()> {
        for asset in asset_bundle.get_assets()?.iter() {
            println!("{}", asset.get_header()?.get_name()?);

            match asset.which() {
                Ok(asset::PixelData(Ok(pixel_data))) => {
                    match pixel_data.which() {
                        Ok(asset_pixel_data::Stored(_stored)) => {
                            // println!("stored");
                        }
                        Ok(asset_pixel_data::Cooked(_cooked)) => {
                            // println!("cooked");
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
        }
        Ok(())
    }
}
fn main() {
    println!("Hello, world!");

    for _ in 0..100 {
        //test::read_asset_bundle3().unwrap();
        test::with_reader_mmap(test::print_asset_bundle).unwrap();
    }
}
