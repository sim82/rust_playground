extern crate capnp;
extern crate custom_error;
extern crate memmap;
extern crate uuid;

pub mod asset_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/asset_capnp.rs"));
}

pub mod scene_capnp {
    include!(concat!(env!("OUT_DIR"), "/capnp/scene_capnp.rs"));
}

pub mod asset_bundle {
    use crate::asset_capnp::{asset, asset_bundle, asset_index};
    use capnp::{message, serialize, Word};
    use memmap::{Mmap, MmapOptions};
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use uuid::Uuid;

    custom_error::custom_error! { pub Error
        Io{ source : std::io::Error } = "unable to read file",
        Capnp{ source : capnp::Error } = "capnp error",
        NotFound = "Element not found error",
        NotImplemented = "Not implemented",
    }

    #[derive(PartialEq, Clone)]
    pub enum AssetType {
        PixelData,
        PcmData,
        MaterialDesc,
        MeshData,
    }

    /// acces to an abstract asset bundle
    pub trait AssetBundleAccess {
        /// Iterate over name -> uuid mapping
        fn get_names<'a>(&'a self) -> Box<Iterator<Item = (&str, Uuid)> + 'a>;

        /// get asser reader by uuid
        fn get<'a>(&'a self, id: &Uuid) -> Result<asset::Reader<'a>>;

        /// get asset reader by name
        fn get_by_name<'a>(&'a self, name: &str) -> Result<asset::Reader<'a>>;

        /// iterate over all assets of a specific type
        fn iter_by_type<'a>(
            &'a self,
            asset_type: AssetType,
        ) -> Result<Box<Iterator<Item = asset::Reader<'a>> + 'a>>;
    }

    struct IndexCache<T: message::ReaderSegments> {
        message: message::Reader<T>,
        map: std::collections::HashMap<String, u32>,

        id_map: std::collections::HashMap<Uuid, u32>,
        // name_map : std::collections::HashMap<String, u32>,
    }

    pub type Result<T> = std::result::Result<T, Error>;

    impl<T: message::ReaderSegments> AssetBundleAccess for IndexCache<T> {
        fn get_names<'a>(&'a self) -> Box<Iterator<Item = (&str, Uuid)> + 'a> {
            let asset_bundle = self.message.get_root::<asset_bundle::Reader>().unwrap();
            let headers = asset_bundle.get_index().unwrap().get_headers().unwrap();

            // panic!("not implemented" )
            Box::new(self.map.iter().map(move |(x, i)| {
                (
                    &x[..],
                    Uuid::parse_str(headers.get(*i).get_uuid().unwrap()).unwrap(),
                )
            }))
        }
        fn get<'a>(&'a self, index: &Uuid) -> Result<asset::Reader<'a>> {
            let asset_bundle = self.message.get_root::<asset_bundle::Reader>().unwrap();
            let assets = asset_bundle.get_assets()?;
            match self.id_map.get(index) {
                Some(i) => Ok(assets.get(*i)),
                None => Err(Error::NotFound),
            }
        }

        fn get_by_name<'a>(&'a self, name: &str) -> Result<asset::Reader<'a>> {
            let asset_bundle = self.message.get_root::<asset_bundle::Reader>()?;
            let assets = asset_bundle.get_assets()?;

            match self.map.get(name) {
                Some(index) => Ok(assets.get(*index)),
                None => Err(Error::NotFound),
            }
        }

        fn iter_by_type<'a>(
            &'a self,
            asset_type: AssetType,
        ) -> Result<Box<Iterator<Item = asset::Reader<'a>> + 'a>> {
            let asset_bundle = self.message.get_root::<asset_bundle::Reader>()?;

            Ok(Box::new(asset_bundle.get_assets()?.iter().filter(
                move |asset| match asset.which() {
                    Ok(asset::PixelData(_)) => asset_type == AssetType::PixelData,
                    Ok(asset::PcmData(_)) => asset_type == AssetType::PcmData,
                    Ok(asset::MaterialDesc(_)) => asset_type == AssetType::MaterialDesc,
                    Ok(asset::MeshData(_)) => asset_type == AssetType::MeshData,
                    _ => false,
                },
            )))
        }
    }

    pub struct AssetBundleFile {
        map: Mmap,
    }

    fn index_cache<T: message::ReaderSegments>(
        reader: message::Reader<T>,
    ) -> Result<IndexCache<T>> {
        let map = reader
            .get_root::<asset_bundle::Reader>()?
            .get_index()?
            .get_headers()?
            .iter()
            .enumerate()
            .map(|(i, h)| (String::from(h.get_name().unwrap()), i as u32))
            .collect();

        let id_map = reader
            .get_root::<asset_bundle::Reader>()?
            .get_index()?
            .get_headers()?
            .iter()
            .enumerate()
            .map(|(i, h)| (Uuid::parse_str(h.get_uuid().unwrap()).unwrap(), i as u32))
            .collect();

        Ok(IndexCache {
            message: reader,
            map: map,
            id_map: id_map,
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
                message::ReaderOptions::new(),
            )?;

            index_cache(msg)
        }
    }

    pub fn owned_access<P: AsRef<Path>>(path: P) -> Result<impl AssetBundleAccess> {
        let mut file = File::open(path)?;
        index_cache(serialize::read_message(
            &mut file,
            message::ReaderOptions::new(),
        )?)
    }

    struct AssetBundleDirectory {
        // path: PathBuf,
        id_map: std::collections::HashMap<Uuid, message::Reader<serialize::OwnedSegments>>,
        name_map: std::collections::HashMap<String, Uuid>,
    }

    impl AssetBundleAccess for AssetBundleDirectory {
        fn get_names<'a>(&'a self) -> Box<Iterator<Item = (&str, Uuid)> + 'a> {
            Box::new(self.name_map.iter().map(|(name, id)| (&name[..], *id)))
            // Box::new(self.map.iter().map(|(x, i)| (&x[..], *i as usize)))
        }
        fn get<'a>(&'a self, id: &Uuid) -> Result<asset::Reader<'a>> {
            match self.id_map.get(id) {
                Some(r) => Ok(r.get_root::<asset::Reader>()?),
                None => Err(Error::NotFound),
            }
        }

        fn get_by_name<'a>(&'a self, name: &str) -> Result<asset::Reader<'a>> {
            match self.name_map.get(name) {
                Some(uuid) => {
                    match self.id_map.get(uuid) {
                        Some(reader) => Ok(reader.get_root::<asset::Reader>()?),
                        None => Err(Error::NotFound),
                    }

                    // let mut filename = self.path;
                    // filename.push(uuid);
                    // let mut file = File::open(filename)?;
                }
                None => Err(Error::NotFound),
            }
        }

        fn iter_by_type<'a>(
            &'a self,
            asset_type: AssetType,
        ) -> Result<Box<Iterator<Item = asset::Reader<'a>> + 'a>> {
            Ok(Box::new(self.id_map.iter().filter_map(
                move |(_, reader)| {
                    let asset = reader.get_root::<asset::Reader>().unwrap();
                    let b = match asset.which() {
                        Ok(asset::PixelData(_)) => asset_type == AssetType::PixelData,
                        Ok(asset::PcmData(_)) => asset_type == AssetType::PcmData,
                        Ok(asset::MaterialDesc(_)) => asset_type == AssetType::MaterialDesc,
                        Ok(asset::MeshData(_)) => asset_type == AssetType::MeshData,
                        _ => false,
                    };
                    if b {
                        Some(asset)
                    } else {
                        None
                    }
                },
            )))
            // Err(Error::NotImplemented)
        }
    }

    pub fn directory_access<P: AsRef<Path>>(path: P) -> Result<impl AssetBundleAccess> {
        let mut index = PathBuf::from(path.as_ref());

        index.push("index");

        let mut index_file = File::open(index)?;
        let index_message =
            serialize::read_message(&mut index_file, message::ReaderOptions::new())?;
        let index = index_message.get_root::<asset_index::Reader>()?;

        let read_asset = |uuid| {
            let mut filename = PathBuf::from(path.as_ref());

            filename.push(format!("{:x}", uuid));

            match File::open(filename) {
                Ok(mut file) => {
                    match serialize::read_message(&mut file, message::ReaderOptions::new()) {
                        Ok(reader) => Some((uuid, reader)),
                        _ => None,
                    }
                }
                _ => None,
            }
        };

        let id_map = index
            .get_headers()?
            .iter()
            .filter_map(|r| read_asset(Uuid::parse_str(r.get_uuid().unwrap()).unwrap()))
            .collect();
        let name_map = index
            .get_headers()?
            .iter()
            .map(|r| {
                (
                    String::from(r.get_name().unwrap()),
                    Uuid::parse_str(r.get_uuid().unwrap()).unwrap(),
                )
            })
            .collect();

        Ok(AssetBundleDirectory {
            // path: PathBuf::from(path.as_ref()),
            id_map: id_map,
            name_map: name_map,
        })
    }

    struct AssetBundleMulti<'a> {
        bundles: Vec<Box<AssetBundleAccess + 'a>>,
    }

    pub fn multi_access<'a>(
        v: Vec<Box<AssetBundleAccess + 'a>>,
    ) -> Result<impl AssetBundleAccess + 'a> {
        Ok(AssetBundleMulti { bundles: v })
    }

    impl AssetBundleAccess for AssetBundleMulti<'_> {
        fn get_names<'a>(&'a self) -> Box<Iterator<Item = (&str, Uuid)> + 'a> {
            Box::new(self.bundles.iter().flat_map(|it| it.get_names()))
        }
        fn get<'a>(&'a self, id: &Uuid) -> Result<asset::Reader<'a>> {
            for bundle in &self.bundles {
                match bundle.get(id) {
                    Ok(r) => return Ok(r),
                    _ => continue,
                }
            }
            Err(Error::NotFound)
        }

        fn get_by_name<'a>(&'a self, name: &str) -> Result<asset::Reader<'a>> {
            for bundle in &self.bundles {
                match bundle.get_by_name(name) {
                    Ok(r) => return Ok(r),
                    _ => continue,
                }
            }
            Err(Error::NotFound)
        }
        fn iter_by_type<'a>(
            &'a self,
            asset_type: AssetType,
        ) -> Result<Box<Iterator<Item = asset::Reader<'a>> + 'a>> {
            Ok(Box::new(self.bundles.iter().flat_map(move |it| {
                it.iter_by_type(asset_type.clone()).unwrap()
            })))
        }
    }
}
