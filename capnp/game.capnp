@0xace601020add72c7;


using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("cp::game");

struct SerializedActor
{
    id @0 : UInt64;
    url @1 : Text;
    data @2 : Data;
}

struct ActorGeneration
{
    actors @0 : List(SerializedActor);
    struct WakeupQueueEntry
    {
        time @0 : UInt64;
        id @1 : UInt64;
    }

    wakeupQueue @1 : List(WakeupQueueEntry);
    genCount @2 : UInt64;
    monotonicTime @3 : UInt64;
}

struct ActorGame
{
    actorGeneration @0 : ActorGeneration;
    idBitmap @1 : Data;
}

struct SinglePlayerGame
{

}

struct ScriptValue
{
    union
    {
        table @0 : List(ScriptKeyValue);
        array @1 : List(ScriptValue);
        intValue @2 : Int64;
        floatValue @3 : Float64;
        boolValue @4 : Bool;
        stringValue @5 : Text;
        closure @6 : Data;
        unhandled @7 : Int32;
        instance @8 : List(ScriptKeyValue);
    }
}

struct ScriptKeyValue
{
    key @0 : Text;
    value @1 : ScriptValue;
}


struct DebugRequest
{
    token @0 : Int64;


    struct ScriptInfo
    {

    }

    struct AddBreakpoint
    {
        id @0 : Int32;
        line @1 : Int32;
    }

    struct ScriptGet
    {
        id @0 : Int32;
    }

    struct Execute
    {
        script @0 : Text;
        immediate @1 : Bool;
    }
    union
    {
        scriptInfo @1 : ScriptInfo;
        addBreakpoint @2 : AddBreakpoint;
        scriptGet @3 : ScriptGet;
        execute @4 : Execute;
    }
}


struct DebugReply
{
    token @0 : Int64;
    struct ScriptInfo
    {
        id @0 : Int32;
        sourceName @1 : Text;
    }
    struct AddBreakpoint
    {
        breakpointId @0 : Int32;
    }

    struct ScriptGet
    {
        sourceLines @0 : List(Text);
    }

    struct EventStopped
    {
        union
        {
            breakpointId @0 : Int32;
            signal @1 : Int32;
        }
        scriptId @2 : Int32;
        line @3 : Int32;
    }

    struct EventWatchpoint
    {
        watchpointId @0 : Int32;
        scriptId @1 : Int32;
        line @2 : Int32;

        localNames @3 : List(Text);
        localValues @4 : List(ScriptValue);
    }

    struct Execute
    {
        consoleOutput @0 : Text;
        error @1 : Bool;
    }

    union
    {
        error @1 : Text;
        scriptInfo @2 : List(ScriptInfo);
        addBreakpoint @3 : AddBreakpoint;
        scriptGet @4 : ScriptGet;
        eventStopped @5 : EventStopped;
        eventWatchpoint @6 : EventWatchpoint;
        execute @7 : Execute;
    }
}

struct DebugEvent
{

}
