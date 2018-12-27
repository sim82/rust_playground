extern crate capnp;
extern crate memmap;

pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}

pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}


pub mod asset_bundle {
	use asset_capnp::{asset_bundle, asset};
	use std::path::Path;
	use std::fs::File;
	use memmap::MmapOptions;	
	use capnp::{serialize, Word};

	pub trait AssetBundleAccess {
		fn get_by_name<'a>( &'a self, name : &str ) -> asset::Reader<'a>;
	}

	struct IndexCache<T : capnp::message::ReaderSegments> {
		message : capnp::message::Reader<T>,
		map : std::collections::HashMap<String,u32>,
	}

	impl<T : capnp::message::ReaderSegments> AssetBundleAccess for capnp::message::Reader<T> {
		fn get_by_name<'a>( &'a self, name : &str ) -> asset::Reader<'a> {
            let mut index = -1;
            let asset_bundle = self.get_root::<asset_bundle::Reader>().unwrap();
            
            let headers = asset_bundle.get_index().unwrap().get_headers().unwrap();

            for (i, header) in headers.iter().enumerate() {
            	if header.get_name().unwrap() == name {
            		index = i as isize;
            	}
            }

            if index == -1 {
            	panic!("not found");
            }

            let assets = asset_bundle.get_assets().unwrap();
			assets.get(index as u32)            

		}
	}

	// impl<T : capnp::message::ReaderSegments> AssetBundleAccess for capnp::message::Reader<T> {
	// 	fn get_by_name<'a>( &'a self, name : &str ) -> asset::Reader<'a> {
 //            let mut index = -1;
 //            let asset_bundle = self.get_root::<asset_bundle::Reader>().unwrap();
            
 //            let headers = asset_bundle.get_index().unwrap().get_headers().unwrap();

 //            for (i, header) in headers.iter().enumerate() {
 //            	if header.get_name().unwrap() == name {
 //            		index = i as isize;
 //            	}
 //            }

 //            if index == -1 {
 //            	panic!("not found");
 //            }

 //            let assets = asset_bundle.get_assets().unwrap();
	// 		assets.get(index as u32)            

	// 	}
	// }

	pub struct AssetBundleFile {
		map : memmap::Mmap,
	}

	impl AssetBundleFile {
		pub fn open<P : AsRef<Path>>(path : P) -> AssetBundleFile {
			let file = File::open(path).unwrap();
			let map = unsafe { MmapOptions::new().map(&file).unwrap() };
            AssetBundleFile{ map : map }
		}

		pub fn access<'a>(&'a self) -> impl AssetBundleAccess + 'a {
			let msg = serialize::read_message_from_words(
                unsafe { Word::bytes_to_words(&self.map[..]) },
                ::capnp::message::ReaderOptions::new(),
            )
            .unwrap();
			msg
		}
	}

	pub fn owned_access<P : AsRef<Path>>(path : P) -> impl AssetBundleAccess {
		let mut file = File::open(path).unwrap();

		//OwnedAssetBundle{message : capnp::serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new()).unwrap()}
		capnp::serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new()).unwrap()
	}
}