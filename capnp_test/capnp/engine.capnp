@0xe78749693b939951;

using Cxx = import "/capnp/c++.capnp";
using Scene = import "scene.capnp";
using Asset = import "asset.capnp";
$Cxx.namespace("cp::engine");

interface EntityQuery
{
    num @0 () -> ( num: UInt64 );
    getJson @1 ( index: UInt64 ) -> (output: Text);
    id @2 (index: UInt64) -> (id : Text);
}

interface Engine
{
    test @0 ( input: Text ) -> ( output: Text );
    entityQuery @1 () -> ( query: EntityQuery );
}
