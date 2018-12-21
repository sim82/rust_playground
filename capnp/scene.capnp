@0x9382c358ada605c9;


using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("cp::scene");

struct Vector2 {
    x @0 : Float32;
    y @1 : Float32;
}

struct Vector3 {
    x @0 : Float32;
    y @1 : Float32;
    z @2 : Float32;
}
struct AABB3 {
    min @0 : Vector3;
    max @1 : Vector3;
}

struct RTree {

    struct Branch {
        box @0 : AABB3;
        id @1  : Int32;
    }

    struct Node {
        level @0 : Int32;
        branches @1 : List(Branch);
    }

    maxBranches @0 : Int32;
    nodes @1 : List(Node);
}

struct Plane {
    normal @0 : Vector3;
    distance @1 : Float32;
}

struct Polygon {
    points @0 : List(Vector3);
}

struct TexturedTriangle {
    p0 @0 : Vector3;
    p1 @1 : Vector3;
    p2 @2 : Vector3;

    t0 @3 : Vector2;
    t1 @4 : Vector2;
    t2 @5 : Vector2;
}

struct Portal {
    polygon @0 : Polygon;
    backLeafId @1 : Int32;
    frontLeafId @2 : Int32;
}
struct Visleaf {
    bounds @0 : AABB3;
    planes @1 : List(Plane);
    portalIds @2 : List(Int32);
}

struct Spatial {
    visleafs @0 : List(Visleaf);
    portals @1 : List(Portal);

}

struct AttributeArrayInterleaved
{
    enum Type {
        float32 @0;
        int32 @1;
        uint32 @2;
        int16 @3;
        uint16 @4;

    }
    enum PrimitiveType
    {
        trianglesCcw @0;
        trianglesCw @1;
    }
    struct Attribute {
        name @0 : Text;
        type @1 : Type;
        width @2 : Int32;
        offset @ 3 : Int32;
    }

    attributes @0 : List(Attribute);
    attributeStride @1 : Int32;
    numVertex @2 : Int32;
    attributeArray @3 : Data;

    numIndex @4 : Int32;
    indexType @5 : Type;
    indexArray @6 : Data;

    primitiveType @7 : PrimitiveType;
}

struct ShaderMesh {
    struct Stored {
        enum Type {
            bm1 @0;
            pm3 @1;
            tm4 @2;
        }
        type @0 : Type;
        data @1 : Data;
    }

    struct Brush {
        struct Face {
            appearance @0 : Text;
            normal @1 : Vector3;
            polygon @2 : Polygon;

            texTris @3 : List(TexturedTriangle);
        }

        faces @0 : List(Face);
    }

    struct TexturedPatch {
        dummy @0 : Int32;
    }

    struct Lattice {
        struct Mesh {
            appearance @0 : Text;
            array @1 : AttributeArrayInterleaved;
        }



        meshes @0 : List(Mesh);

    }


    union {
        stored @0 : Stored;
        brush @1 : Brush;
        texturedPatch @2 : TexturedPatch;
        lattice @3 : Lattice;
    }

}


struct ShaderMeshDB {
    ids @0 : List(UInt64);
    meshes @1 : List(ShaderMesh);

}

struct TestMesh {
    tris @0 : List(TexturedTriangle);
}
