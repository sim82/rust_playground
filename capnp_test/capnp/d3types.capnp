@0xc8acfe67a229c913;


using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("cp::d3types");

struct StaticBitmap
{
    bitmap @0 : Data;
    nextFree @1 : UInt64;
}


struct DynamicBitmap
{
    bbs @0 : List(StaticBitmap);
    bbNumAlloc @1 : List(UInt64);
    curBlock @2 : UInt64;
}
