extern crate capnp;
extern crate memmap;

pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}

pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}

pub mod asset_bundle {
    use asset_capnp::{asset, asset_bundle};
    use capnp::{serialize, Word};
    use memmap::MmapOptions;
    use std::fs::File;
    use std::path::Path;

    pub trait AssetBundleAccess {
        fn get_names<'a>(&'a self) -> Vec<&'a str>;
        fn get_by_name<'a>(&'a self, name: &str) -> asset::Reader<'a>;
    }

    struct IndexCache<T: capnp::message::ReaderSegments> {
        message: capnp::message::Reader<T>,
        map: std::collections::HashMap<String, u32>,
    }

    impl<T: capnp::message::ReaderSegments> AssetBundleAccess for IndexCache<T> {
        fn get_names<'a>(&'a self) -> Vec<&'a str> {
            self.map.keys().map(|x| &x[..]).collect()
        }

        fn get_by_name<'a>(&'a self, name: &str) -> asset::Reader<'a> {
            let asset_bundle = self.message.get_root::<asset_bundle::Reader>().unwrap();
            let assets = asset_bundle.get_assets().unwrap();

            let index = self.map.get(name).unwrap();
            assets.get(*index)
        }
    }

    pub struct AssetBundleFile {
        map: memmap::Mmap,
    }

    fn index_cache<T: capnp::message::ReaderSegments>(
        reader: capnp::message::Reader<T>,
    ) -> IndexCache<T> {
        let map = reader
            .get_root::<asset_bundle::Reader>()
            .unwrap()
            .get_index()
            .unwrap()
            .get_headers()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, h)| (String::from(h.get_name().unwrap()), i as u32))
            .collect();

        IndexCache {
            message: reader,
            map: map,
        }
    }

    impl AssetBundleFile {
        pub fn open<P: AsRef<Path>>(path: P) -> AssetBundleFile {
            let file = File::open(path).unwrap();
            let map = unsafe { MmapOptions::new().map(&file).unwrap() };
            AssetBundleFile { map: map }
        }

        pub fn access<'a>(&'a self) -> impl AssetBundleAccess + 'a {
            let msg = serialize::read_message_from_words(
                unsafe { Word::bytes_to_words(&self.map[..]) },
                ::capnp::message::ReaderOptions::new(),
            )
            .unwrap();
            index_cache(msg)
        }
    }

    pub fn owned_access<P: AsRef<Path>>(path: P) -> impl AssetBundleAccess {
        let mut file = File::open(path).unwrap();
        index_cache(
            capnp::serialize::read_message(&mut file, ::capnp::message::ReaderOptions::new())
                .unwrap(),
        )
    }
}
