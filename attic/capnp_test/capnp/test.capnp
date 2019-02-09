@0x93f0d31f7d02aab1;
# struct Person {
#   id @0 :UInt32;
#   name @1 :Text;
#   email @2 :Text;
#   phones @3 :List(PhoneNumber);
#
#   struct PhoneNumber {
#     number @0 :Text;
#     type @1 :Type;
#
#     enum Type {
#       mobile @0;
#       home @1;
#       work @2;
#     }
#   }
#
#   employment :union {
#     unemployed @4 :Void;
#     employer @5 :Text;
#     school @6 :Text;
#     selfEmployed @7 :Void;
#     # We assume that a person is only one of these.
#   }
# }
#
# struct AddressBook {
#   people @0 :List(Person);
# }

#struct MipMapLevel {
#    #data @0 :Data;
#    width @0 :UInt32;
#    height @1 :UInt32;
#}
#
#
#struct MipMapProvider {
#    pixelFormat @0 : UInt32;
#    levels @1 : List(MipMapLevel);
#    levelData @2 : List(Data);
#    #numLevels @0 :UInt8;
#}



#struct TestData {
#    test1 @0 : UInt32;
#    data @1: Data;
#}
#
#struct TestList {
#   names @0 : List(Text);
#   data @1 : List(TestData);
#}

#struct Bundle {
#    names @0 : List(Text);
#    providers @1: List(import "asset.capnp".AssetPixelDataCooked);
#    xxx @2: Text;
#}

