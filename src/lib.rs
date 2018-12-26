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
	use std::cell::RefCell;

	pub trait AssetRef {
		fn get<'a>(&'a self) -> asset::Reader<'a>;
	}

	pub trait AssetBundleAccess<'a> {
		fn get_by_name( &'a self, name : &str ) -> asset::Reader<'a>;
		fn get_by_name2(&'a self, name : &str) -> Box<AssetRef + 'a>;
		fn bla(&'a self);
	}

	struct MappedBundleAssetRef<'a> {
		reader : &'a capnp::message::Reader<capnp::serialize::SliceSegments<'a>>,
		index : u32,
	}

	impl AssetRef for MappedBundleAssetRef<'_> {
		fn get<'a>(&'a self) -> asset::Reader<'a> {
			self.reader.get_root::<asset_bundle::Reader>().unwrap().get_assets().unwrap().get(self.index)	
		}
	}

	pub struct MappedBundle<'a> {
		message : capnp::message::Reader<capnp::serialize::SliceSegments<'a>>,
	}

	
	impl<'a> AssetBundleAccess<'a> for MappedBundle<'a> {
		fn get_by_name( &'a self, name : &str ) -> asset::Reader<'a> {
            let mut index = -1;
            let asset_bundle = self.message.get_root::<asset_bundle::Reader>().unwrap();
            
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

		fn get_by_name2( &'a self, _name : &str ) -> Box<AssetRef + 'a> {
    		Box::new(MappedBundleAssetRef { reader : &self.message, index : 0 } )         
		}
		fn bla( &'a self ) {
			
		}
	}

	pub struct AssetBundleFile {
		map : memmap::Mmap,
	}
	impl AssetBundleFile {
		pub fn open<P : AsRef<Path>>(path : P) -> AssetBundleFile {
			let file = File::open(path).unwrap();
			let map = unsafe { MmapOptions::new().map(&file).unwrap() };
            AssetBundleFile{ map : map }
		}

		pub fn access<'a>(&'a self) -> impl AssetBundleAccess<'a> {
			let msg = serialize::read_message_from_words(
                unsafe { Word::bytes_to_words(&self.map[..]) },
                ::capnp::message::ReaderOptions::new(),
            )
            .unwrap();
			MappedBundle{message : msg}
		}
	}

}

pub mod deref {
	pub struct Factory {
		i : u32
	}

	impl Factory {
		pub fn new() -> Factory {
			Factory{ i : 666 }
		}

		pub fn get<'a>(&'a self) -> impl OuterTrait<'a> {
			Outer{ x : &self.i }
		}  
	}

	pub trait InnerTrait {
		fn bla(&self);
	}

	pub trait OuterTrait<'a> {
		fn get(&'a self) -> Box<InnerTrait + 'a>;
	}

	struct Inner<'a> {
		o : &'a Outer<'a>,
		u : &'a u32,
	}

	struct Outer<'a> {
		x : &'a u32
	}

	impl<'a> OuterTrait<'a> for Outer<'a> {
		fn get(&'a self) -> Box<InnerTrait + 'a> {
			Box::new(Inner{ o : self, u : &self.x })
		}
	}

	impl InnerTrait for Inner<'_> {
		fn bla(&self) {
			println!("bla {}", self.o.x);
		}
	}

	pub fn create_outer<'a>( i : &'a u32) -> impl OuterTrait<'a> {
		Outer{ x : i }
	}

}