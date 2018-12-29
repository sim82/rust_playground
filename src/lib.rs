extern crate capnp;
extern crate custom_error;
extern crate memmap;

pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}

pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}

pub mod asset_bundle {
    use crate::asset_capnp::{asset, asset_bundle};
    use capnp::{serialize, Word};
    use memmap::MmapOptions;
    use std::fs::File;
    use std::path::Path;

    custom_error::custom_error! { pub Error
        Io{ source : std::io::Error } = "unable to read file",
        Capnp{ source : capnp::Error } = "capnp error",
        NotFound = "Element not found error",
    }

    pub trait AssetBundleAccess {
        fn get_names<'a>(&'a self) -> Box<Iterator<Item = (&str, usize)> + 'a>;
        fn get_names2<'a>(&'a self) -> Box<Iterator<Item = (&str, &u32)> + 'a>;

        fn get<'a>(&'a self, index: usize) -> Result<asset::Reader<'a>>;
        fn get_by_name<'a>(&'a self, name: &str) -> Result<asset::Reader<'a>>;
    }

    struct IndexCache<T: capnp::message::ReaderSegments> {
        message: capnp::message::Reader<T>,
        map: std::collections::HashMap<String, u32>,
    }

    pub type Result<T> = std::result::Result<T, Error>;

    impl<T: capnp::message::ReaderSegments> AssetBundleAccess for IndexCache<T> {
        fn get_names<'a>(&'a self) -> Box<Iterator<Item = (&str, usize)> + 'a> {
            Box::new(self.map.iter().map(|(x, i)| (&x[..], *i as usize)))
        }
        fn get_names2<'a>(&'a self) -> Box<Iterator<Item = (&str, &u32)> + 'a> {
            Box::new(self.map.iter().map(|(x, i)| (&x[..], i)))
        }
        fn get<'a>(&'a self, index: usize) -> Result<asset::Reader<'a>> {
            // TODO: truncate check?

            let asset_bundle = self.message.get_root::<asset_bundle::Reader>()?;
            let assets = asset_bundle.get_assets()?;
            Ok(assets.get(index as u32))
        }

        fn get_by_name<'a>(&'a self, name: &str) -> Result<asset::Reader<'a>> {
            match self.map.get(name) {
                Some(index) => self.get(*index as usize),
                None => Err(Error::NotFound),
            }
        }
    }

    pub struct AssetBundleFile {
        map: memmap::Mmap,
    }

    fn index_cache<T: capnp::message::ReaderSegments>(
        reader: capnp::message::Reader<T>,
    ) -> Result<IndexCache<T>> {
        let map = reader
            .get_root::<asset_bundle::Reader>()?
            .get_index()?
            .get_headers()?
            .iter()
            .enumerate()
            .map(|(i, h)| (String::from(h.get_name().unwrap()), i as u32))
            .collect();

        Ok(IndexCache {
            message: reader,
            map: map,
        })
    }

    impl AssetBundleFile {
        pub fn open<P: AsRef<Path>>(path: P) -> Result<AssetBundleFile> {
            let file = File::open(path)?;
            let map = unsafe { MmapOptions::new().map(&file)? };
            Ok(AssetBundleFile { map: map })
        }

        pub fn access<'a>(&'a self) -> Result<impl AssetBundleAccess + 'a> {
            let msg = serialize::read_message_from_words(
                unsafe { Word::bytes_to_words(&self.map[..]) },
                ::capnp::message::ReaderOptions::new(),
            )?;

            index_cache(msg)
        }
    }

    pub fn owned_access<P: AsRef<Path>>(path: P) -> Result<impl AssetBundleAccess> {
        let mut file = File::open(path)?;
        index_cache(capnp::serialize::read_message(
            &mut file,
            ::capnp::message::ReaderOptions::new(),
        )?)
    }
}
