extern crate capnp;
extern crate memmap;

pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}

pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}

mod test {
    use memmap::MmapOptions;
    use std::fs::File;

    use asset_capnp::{asset, asset_bundle, asset_pixel_data};
    use capnp::serialize;
    use capnp::Word;

    pub enum CapnpMessageData {
        Owned(capnp::message::Reader<capnp::serialize::OwnedSegments>),
        Mapped(memmap::Mmap),
    }

    pub fn with_reader_owned_asset_bundle<F: Fn(asset_bundle::Reader) -> capnp::Result<()>>(
        f: F,
    ) -> capnp::Result<()> {
        with_reader_asset_bundle(&read_owned(), f)
    }

    // pub fn with_reader_owned<'a, T, F: Fn(T) -> capnp::Result<()>>(
    //        f: F,
    //    ) -> capnp::Result<()>
    //    	where T : capnp::traits::FromPointerReader<'a>
    //    {
    //        with_reader::<T, F>(&read_owned(), f)
    //    }

    pub fn with_reader<
        'a,
        T: for<'b> capnp::traits::FromPointerReader<'b>,
        // T: capnp::traits::FromPointerReader<'a>,
        
        F: Fn(T) -> capnp::Result<()>,
    >(
        data: &'a CapnpMessageData,
        f: F,
    ) -> capnp::Result<()> {
        match data {
            CapnpMessageData::Owned(reader) => f(reader.get_root::<T>()?),
            CapnpMessageData::Mapped(mmap) => {
                let message = serialize::read_message_from_words(
                    unsafe { Word::bytes_to_words(&mmap[..]) },
                    ::capnp::message::ReaderOptions::new(),
                )
                .unwrap();

                f(message.get_root::<T>()?)
            }
        }
    }

	pub fn get_reader<
        'a,
        T: capnp::traits::FromPointerReader<'a>,
        // T: capnp::traits::FromPointerReader<'a>,
    >(
        data: &'a CapnpMessageData,
       
    ) -> capnp::Result<T> {
        match data {
            CapnpMessageData::Owned(reader) => reader.get_root::<T>(),
            CapnpMessageData::Mapped(mmap) => {
                let message = serialize::read_message_from_words(
                    unsafe { Word::bytes_to_words(&mmap[..]) },
                    ::capnp::message::ReaderOptions::new(),
                )
                .unwrap();

                message.get_root::<T>()
            }
        }
    }    

    pub fn with_reader_mmap_asset_bundle<F: Fn(asset_bundle::Reader) -> capnp::Result<()>>(
        f: F,
    ) -> capnp::Result<()> {
        with_reader_asset_bundle(&read_mmap(), f)
    }

    pub fn with_reader_asset_bundle<F: Fn(asset_bundle::Reader) -> capnp::Result<()>>(
        data: &CapnpMessageData,
        f: F,
    ) -> capnp::Result<()> {
        match data {
            CapnpMessageData::Owned(reader) => f(reader.get_root::<asset_bundle::Reader>()?),
            CapnpMessageData::Mapped(mmap) => {
                let message = serialize::read_message_from_words(
                    unsafe { Word::bytes_to_words(&mmap[..]) },
                    ::capnp::message::ReaderOptions::new(),
                )
                .unwrap();

                f(message.get_root::<asset_bundle::Reader>()?)
            }
        }
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

    pub fn read_owned() -> CapnpMessageData {
        let mut file = File::open("test.bin").unwrap();
        let message_reader =
            serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new()).unwrap();
        CapnpMessageData::Owned(message_reader)
    }
    pub fn read_mmap() -> CapnpMessageData {
        let file = File::open("test.bin").unwrap();

        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        CapnpMessageData::Mapped(mmap)
    }

    pub fn test_asset_bundle_owned() {
        // with_reader(&read_owned(), print_asset_bundle).unwrap();
     	with_reader_asset_bundle(&read_owned(), print_asset_bundle).unwrap();
        let reader = get_reader::<asset_bundle::Reader>(&read_owned()).unwrap();
        print_asset_bundle(reader);
    }

}
fn main() {
    println!("Hello, world!");

    for _ in 0..100 {
        // test::with_reader_owned_asset_bundle(test::print_asset_bundle).unwrap();
        test::test_asset_bundle_owned();
    }

    for _ in 0..100 {
        test::with_reader_mmap_asset_bundle(test::print_asset_bundle).unwrap();
    }
}
