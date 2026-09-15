from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from pkg.dsp import OnePole

struct PyOnePole(Defaultable, Movable, Writable):
    var f: OnePole
    def __init__(out self):
        self.f = OnePole()

    @staticmethod
    def set_cutoff(self_ptr: Pointer[Self, MutAnyOrigin], hz: PythonObject, sr: PythonObject) raises -> PythonObject:
        self_ptr[].f.set_cutoff(Float64(py=hz), Float64(py=sr))
        return PythonObject(None)

    @staticmethod
    def process(self_ptr: Pointer[Self, MutAnyOrigin], src: PythonObject, dst: PythonObject, n: PythonObject) raises -> PythonObject:
        var s = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=src))
        var d = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=dst))
        var count = Int(py=n)
        ref f = self_ptr[].f
        for i in range(count):
            d[unsafe_offset=i] = f.next(s[unsafe_offset=i])
        return PythonObject(None)

@export
def PyInit_onepole_ext() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("onepole_ext")
        _ = (m.add_type[PyOnePole]("OnePole")
            .def_init_defaultable[PyOnePole]()
            .def_method[PyOnePole.set_cutoff]("set_cutoff")
            .def_method[PyOnePole.process]("process"))
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
