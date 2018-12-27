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

    pub enum CapnpReader<'a> {
        Owned(capnp::message::Reader<capnp::serialize::OwnedSegments>),
        Sliced(capnp::message::Reader<capnp::serialize::SliceSegments<'a>>),
    }

    impl<'a> CapnpReader<'a> {
        fn get_root<T: capnp::traits::FromPointerReader<'a>>(&'a self) -> Box<T> {
            match self {
                CapnpReader::Owned(r) => Box::new(r.get_root::<T>().unwrap()) as Box<T>,
                CapnpReader::Sliced(r) => Box::new(r.get_root::<T>().unwrap()),
            }
        }
    }

    pub trait ReaderCreator {
        // fn get_reader<'a>(&'a self) -> CapnpReader<'a>;
        fn get_reader(&self) -> CapnpReader<'_>;

        // fn get_root<'a, T : capnp::traits::FromPointerReader<'a> >(&'a self) -> Box<T>;
    }

    struct MappedReaderCreator {
        mmap: memmap::Mmap,
    }

    impl MappedReaderCreator {
        fn new() -> MappedReaderCreator {
            let file = File::open("test.bin").unwrap();
            MappedReaderCreator {
                mmap: unsafe { MmapOptions::new().map(&file).unwrap() },
            }
        }
    }

    struct OwnedReaderCreator {}

    impl OwnedReaderCreator {
        fn new() -> OwnedReaderCreator {
            OwnedReaderCreator {}
        }
    }

    impl ReaderCreator for MappedReaderCreator {
        // fn get_reader<'a>(&'a self) -> CapnpReader<'a> {
        fn get_reader(&self) -> CapnpReader<'_> {
            CapnpReader::Sliced(
                serialize::read_message_from_words(
                    unsafe { Word::bytes_to_words(&self.mmap[..]) },
                    ::capnp::message::ReaderOptions::new(),
                )
                .unwrap(),
            )
        }
    }

    impl ReaderCreator for OwnedReaderCreator {
        fn get_reader(&self) -> CapnpReader<'_> {
            let mut file = File::open("test.bin").unwrap();
            CapnpReader::Owned(
                serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new()).unwrap(),
            )
        }
    }

    pub fn test_reader2(reader: &CapnpReader<'_>) -> capnp::Result<()> {
        print_asset_bundle(*reader.get_root::<asset_bundle::Reader>())
    }

    pub fn test_new_try_with_traits(switch: bool) -> capnp::Result<()> {
        let creator = if switch {
            Box::new(MappedReaderCreator::new()) as Box<ReaderCreator>
        } else {
            Box::new(OwnedReaderCreator::new())
        };

        test_reader2(&creator.get_reader())
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

    for _i in 0..100 {
        //test::test_new_try( i % 2 == 0).unwrap();
        // test::test_new_try_with_traits( i % 2 == 0).unwrap();
        test::test_new_try_with_traits(false).unwrap();
    }

    // for _ in 0..100 {
    //     // test::with_reader_owned_asset_bundle(test::print_asset_bundle).unwrap();
    //     test::test_asset_bundle_owned();
    // }

    // for _ in 0..100 {
    //     test::with_reader_mmap_asset_bundle(test::print_asset_bundle).unwrap();
    // }
}
