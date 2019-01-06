@0xb6ac9e37128f3f5f;

using Cxx = import "/capnp/c++.capnp";
using Scene = import "scene.capnp";
$Cxx.namespace("cp::asset");

#using Java = import "/capnp/java.capnp";
#$Java.package("com.dev3dyne.cp");
#$Java.outerClassname("AssetCp");

##########################################
# Pixel Data
##########################################
struct MipMapLevel {
    #data @0 :Data;
    width @0 :UInt32;
    height @1 :UInt32;
}


struct AssetPixelDataCooked {
    pixelFormat @0 : UInt32;
    levels @1 : List(MipMapLevel);
    levelData @2 : List(Data);
}

struct AssetPixelDataStored {
    srcFilename @0 :Text;
    mimeType @1 :Text;

    data @2 :Data;
}

struct AssetPixelData {
    union {
        stored @0 :AssetPixelDataStored;
        cooked @1 :AssetPixelDataCooked;
    }

}

##########################################
# Pcm Data
##########################################

struct AssetPcmDataCooked {
    sampleFormat @0 :UInt32;

    offset @1 : List(Float32);
    data @2 : List(Data);
}

struct AssetPcmDataStored {
    srcFilename @0 :Text;
    mimeType @1 :Text;

    data @2 :Data;
}


struct AssetPcmData {
    union {
        stored @0 :AssetPcmDataStored;
        cooked @1 :AssetPcmDataCooked;
    }
}


##########################################
# Material
##########################################

struct AssetMaterialDesc {
    primaryImage @0 :Text;
}


##########################################
# Mesh Data
##########################################


struct AssetMeshData
{
    attributeArrayInterleavedList @0 : List(Scene.AttributeArrayInterleaved);
    appearanceList @1 : List(Text);
}

struct AssetHeader
{
    name @0 :Text;
    uuid @1 :Text;
}

struct AssetIndex
{
    headers @0 : List(AssetHeader);
}

struct Asset {
    header @0 : AssetHeader;

    union {
        pixelData @1 :AssetPixelData;
        pcmData @2 :AssetPcmData;
        materialDesc @3 :AssetMaterialDesc;
        meshData @4 :AssetMeshData;
    }
}

struct AssetBundle {
    assets @0: List(Asset);
    #names @1 : List(Text); # redundant copy of Asset::name (allow for compact storage in front of capnp file)
    #uuids @2 : List(Text); # redundant copy of Asset::uuid (dto.)
    index @1 : AssetIndex;
}


struct FramedAssetBundle
{
    struct IdIndex
    {
        struct Uuid
        {
            idLow @0 : UInt64;
            idHigh @ 1 : UInt64;
        }
        sortedIds @0 : List(Uuid);
        index @1 : List(UInt64);
    }

    struct NameIndex
    {
        sortedNames @0 : List(Text);
        index @1 : List(UInt64);
    }

    struct OffsetTable
    {
        offsets @0 : List(UInt64);
        sizes @1 : List(UInt64);
        uncompressedSizes @2 : List(UInt64);
    }

    struct IndexSection
    {
        idIndex @0 : IdIndex;
        nameIndex @1 : NameIndex;
        offsetTable @2 : OffsetTable;
    }

    struct HeaderSection
    {
        magick @ 0 : UInt64;
        dataSectionOffset @1 : UInt64;
        dataSectionSize @2 : UInt64;
        indexSectionOffset @3 : UInt64;
        indexSectionSize @4 : UInt64;
    }
}

struct GUID {
    v0 @0 : UInt32;
    v4 @1 : UInt32;
    v8 @2 : UInt32;
    v12 @3 : UInt32;
}


struct GUIDHashTable {
    dir @0 : List(Entry);

    struct Entry {
        key @0 : GUID;
        size @1 : UInt64;
        offset @2 : UInt64;
    }
}


interface AssetProvider
{
    interface Handle {
        isValid @3 () -> (valid : Bool);
        getId @0 () -> (uuid :Text);
        getAsset @1 () -> (asset :Asset);
        getBaked @2 () -> (baked :Asset);
    }


    get @0 ( uuid: Text ) -> (handle :Handle);
    getByName @1( name: Text ) -> (handle :Handle);

    nameList @2() -> (list :List(Text));
    mapNameToUuid @3 (name :Text) -> (uuid :Text);


}

struct Bundle {
    names @0 : List(Text);
    providers @1: List(AssetPixelDataCooked);
    xxx @2: Text;
}
