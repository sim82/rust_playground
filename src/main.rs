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

    pub trait ReaderCreator {
    	fn get_reader<'a>( &'a self ) -> CapnpReader<'a>;
    	fn bla1(&self);
    	fn bla<'a>(&'a self) -> &'a u32;
    }

    struct MappedReaderCreator {
    	mmap : memmap::Mmap,
    	x : u32,
    }

    impl MappedReaderCreator {
    	fn new() -> MappedReaderCreator
    	{
    		let file = File::open("test.bin").unwrap();
    		MappedReaderCreator{mmap : unsafe { MmapOptions::new().map(&file).unwrap() }, x : 666}
    	}
    	
    }

    struct OwnedReaderCreator { x : u32 }

    impl OwnedReaderCreator {
    	fn new() -> OwnedReaderCreator 
    	{
    		OwnedReaderCreator{ x : 1234 }
    	}
    }

    impl ReaderCreator for MappedReaderCreator {
    	
    	fn get_reader<'a>( &'a self ) -> CapnpReader<'a> {
			CapnpReader::Sliced(serialize::read_message_from_words(
            	unsafe { Word::bytes_to_words(&self.mmap[..]) },
            	::capnp::message::ReaderOptions::new(),
        	).unwrap())    		
    	}

    	fn bla<'a>(&'a self) -> &'a u32{
    		&self.x
    	}
    	fn bla1(&self) {
    		println!("OwnedReaderCreator");
    		
    	}
    }

    impl ReaderCreator for OwnedReaderCreator {
    	fn get_reader<'a>(&'a self ) -> CapnpReader<'a> {
			let mut file = File::open("test.bin").unwrap();
			CapnpReader::Owned(serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new()).unwrap())
		}
		fn bla<'a>(&'a self) -> &'a u32 {
    		&self.x
    	}
    	fn bla1(&self) {
    		println!("MappedReaderCreator");
    		
    	}
    }

    pub fn test_reader2<'a>( reader : &'a CapnpReader<'a> )
     -> capnp::Result<()>
    {
    	let asset_bundle = match reader {
    		CapnpReader::Owned(r) => r.get_root::<asset_bundle::Reader>(),
    		CapnpReader::Sliced(r) => r.get_root::<asset_bundle::Reader>()
    	};

		print_asset_bundle(asset_bundle?)    	
    }


    pub fn test_new_try( switch : bool ) -> capnp::Result<()>
    {
    	let mut file = File::open("test.bin").unwrap();

		let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    	let reader = if switch {

    		CapnpReader::Sliced(serialize::read_message_from_words(
            	unsafe { Word::bytes_to_words(&mmap[..]) },
            	::capnp::message::ReaderOptions::new(),
        	)?)
    	} else {
			CapnpReader::Owned(serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new())?)    		
    	};

     //    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    	// let reader = CapnpReader::Sliced(serialize::read_message_from_words(
     //        unsafe { Word::bytes_to_words(&mmap[..]) },
     //        ::capnp::message::ReaderOptions::new(),
     //    )?);

     	// let reader = CapnpReader::Owned(serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new())?);

    	test_reader2(&reader)
    }


    pub fn test_new_try_with_traits( switch : bool ) -> capnp::Result<()>{
  		let creator;
  		if switch {
  			creator = Box::new(MappedReaderCreator::new()) as Box<ReaderCreator>;
  		} else {
  			creator = Box::new(OwnedReaderCreator::new());
  		}

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

	for i in 0..100 {
    	//test::test_new_try( i % 2 == 0).unwrap();
    	// test::test_new_try_with_traits( i % 2 == 0).unwrap();
		test::test_new_try_with_traits( false ).unwrap();
    
    }

    // for _ in 0..100 {
    //     // test::with_reader_owned_asset_bundle(test::print_asset_bundle).unwrap();
    //     test::test_asset_bundle_owned();
    // }

    // for _ in 0..100 {
    //     test::with_reader_mmap_asset_bundle(test::print_asset_bundle).unwrap();
    // }
}
