from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

def process(src: PythonObject, dst: PythonObject, n: PythonObject) raises -> PythonObject:
    var s = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=src))
    var d = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=dst))
    var count = Int(py=n)
    ref cpy = Python().cpython()
    var ts = cpy.PyEval_SaveThread()
    var z: Float32 = 0.0
    for i in range(count):
        z += 0.1 * (s[unsafe_offset=i] - z)
        d[unsafe_offset=i] = z
    cpy.PyEval_RestoreThread(ts)
    return PythonObject(None)

@export
def PyInit_gil2() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("gil2")
        m.def_function[process]("process")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
