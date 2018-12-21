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

	use capnp::Word;
	use capnp::serialize;
	use capnp::message::Reader;
	use capnp::serialize::{OwnedSegments};
	use asset_capnp::{asset_bundle, asset, asset_pixel_data};


	// struct MappedMessage<'a> {
	// 	map : Box<memmap::Mmap>,

	// 	message : Cell<capnp::message::Reader<capnp::serialize::SliceSegments<'a>>>
	// }

	// impl MappedMessage<'a> {
	// 	fn get_root<T>() -> capnp::Result<T> {

	// 	}
	// }

	pub enum CapnpMessage {
		Owned (Reader<OwnedSegments> ),
		Mapped (memmap::Mmap)
	}


	pub fn get_root<'a, T : capnp::traits::FromPointerReader<'a> >( msg : &'a CapnpMessage ) -> capnp::Result<T> {
		match msg {
			CapnpMessage::Owned( r ) => {
				r.get_root::<'a, T>()
			}
			CapnpMessage::Mapped( mmap ) => {
			//	r.get_root::<'a, T>()
				//let words = unsafe { Word::bytes_to_words(&mmap[..]) };
	    
				serialize::read_message_from_words( unsafe { Word::bytes_to_words(&mmap[..]) },::capnp::message::ReaderOptions::new()).unwrap().get_root::<T>()
	    
			}
		}
	}

	fn load_file<'a>() -> CapnpMessage {
		let mut file = File::open("test.bin").unwrap();
	    let message_reader = serialize::read_message(&mut file,::capnp::message::ReaderOptions::new()).unwrap();

	    CapnpMessage::Owned(message_reader)
	}

	fn load_file_mmap<'a>() -> CapnpMessage {
		let file = File::open("test.bin").unwrap();

		let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
		// let words = unsafe { capnp::Word::bytes_to_words(&mmap[..]) };
	    // let message_reader = serialize::read_message_from_words( & words,::capnp::message::ReaderOptions::new()).unwrap();
	    
	    CapnpMessage::Mapped(mmap)
	
		// CapnpMessage::Mapped(mmap, message_reader)
	
	}



	pub fn read_asset_bundle() -> capnp::Result<()>
	{
		let stdin = ::std::io::stdin();
	    let message_reader = serialize::read_message(&mut stdin.lock(),::capnp::message::ReaderOptions::new())?;
	
		let asset_bundle = message_reader.get_root::<asset_bundle::Reader>()?;
		//let index = message_reader.get_root::<asset_index::Reader>()?;
		// let index = asset_bundle.get_index()?;

		// for asset in index.get_headers()?.iter() {
		// 	println!("{}", asset.get_name()?);
		// }

		for asset in asset_bundle.get_assets()?.iter() {
			println!("{}", asset.get_header()?.get_name()?);
		
			match asset.which() {
				Ok(asset::PixelData(Ok(pixel_data))) => {
					match pixel_data.which() {
						Ok(asset_pixel_data::Stored(_stored)) => {
							println!("stored");
						}
						Ok(asset_pixel_data::Cooked(_cooked)) => {
							println!("cooked");
						}
						Err(::capnp::NotInSchema(_)) => { }
					}

					println!("pixel data")
				}
				// Ok(asset::PixelData(Ok(asset_pixel_data::Stored(Ok(stored))))) => {
					

				// 	println!("stored pixel data")
				// }
				_ => {}
				// Err(::capnp::NotInSchema(_)) => { }
			}
		}

		Ok(())
	}

	pub fn with_mmap_reader<F: Fn(&capnp::message::Reader<capnp::message::ReaderSegments>) -> ()>( _f : F ) {

		// let file = File::open("test.bin").unwrap();
		// let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
		// let words = unsafe { capnp::Word::bytes_to_words(&mmap[..]) };
	    // let message_reader = serialize::read_message_from_words( & words,::capnp::message::ReaderOptions::new()).unwrap();
	    // f(&message_reader);
	}

	pub fn read_asset_bundle_mmap() -> capnp::Result<()> {
		
		let file = File::open("test.bin").unwrap();
		let mmap = unsafe { MmapOptions::new().map(&file)? };
		let words = unsafe { capnp::Word::bytes_to_words(&mmap[..]) };
	    let message_reader = serialize::read_message_from_words( & words,::capnp::message::ReaderOptions::new())?;

		let asset_bundle = message_reader.get_root::<asset_bundle::Reader>()?;
		//let index = message_reader.get_root::<asset_index::Reader>()?;
		let _index = asset_bundle.get_index()?;

		// for asset in index.get_headers()?.iter() {
		// 	println!("{}", asset.get_name()?);
		// }

		for asset in asset_bundle.get_assets()?.iter() {
			println!("{}", asset.get_header()?.get_name()?);
		
			match asset.which() {
				Ok(asset::PixelData(Ok(pixel_data))) => {
					match pixel_data.which() {
						Ok(asset_pixel_data::Stored(_stored)) => {
							println!("stored");
						}
						Ok(asset_pixel_data::Cooked(_cooked)) => {
							println!("cooked");
						}
						Err(::capnp::NotInSchema(_)) => { }
					}

					println!("pixel data")
				}
				// Ok(asset::PixelData(Ok(asset_pixel_data::Stored(Ok(stored))))) => {
					

				// 	println!("stored pixel data")
				// }
				_ => {}
				// Err(::capnp::NotInSchema(_)) => { }
			}
		}

		return Ok(())
	}

}
fn main() {
    println!("Hello, world!");

    test::read_asset_bundle_mmap().unwrap();
}

