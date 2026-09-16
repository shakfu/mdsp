"""E5: the E4 graph engine as a Python type.

Build: mojo build --emit shared-lib --fp-mode contract=off -I ../../../../src/mdsp/_mojo -I . graph_ext.mojo -o graph_ext.so
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from engine import FPtr, Graph


struct PyGraph(Movable, Writable):
    var graph: Graph

    def __init__(out self, sample_rate: Float64, block: Int):
        self.graph = Graph(sample_rate, block)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Graph(nodes=", len(self.graph.nodes), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        self = Self(Float64(py=args[0]), Int(py=args[1]))

    @staticmethod
    def add(self_ptr: Pointer[Self, MutAnyOrigin], kind: PythonObject) raises -> PythonObject:
        return PythonObject(self_ptr[].graph.add_kind(String(py=kind)))

    @staticmethod
    def connect(
        self_ptr: Pointer[Self, MutAnyOrigin], src: PythonObject, dst: PythonObject, port: PythonObject
    ) raises -> PythonObject:
        self_ptr[].graph.connect(Int(py=src), Int(py=dst), Int(py=port))
        return PythonObject(None)

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin], node: PythonObject, param: PythonObject, value: PythonObject
    ) raises -> PythonObject:
        self_ptr[].graph.set(Int(py=node), Int(py=param), Float64(py=value))
        return PythonObject(None)

    @staticmethod
    def process(
        self_ptr: Pointer[Self, MutAnyOrigin], dst: PythonObject, frames: PythonObject
    ) raises -> PythonObject:
        var address = Int(py=dst)
        var n = Int(py=frames)
        ref cpython = Python().cpython()
        var state = cpython.PyEval_SaveThread()
        self_ptr[].graph.process(FPtr(unsafe_from_address=address), n)
        cpython.PyEval_RestoreThread(state)
        return PythonObject(None)


@export
def PyInit_graph_ext() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("graph_ext")
        _ = (
            m.add_type[PyGraph]("Graph")
            .def_py_init[PyGraph.py_init]()
            .def_method[PyGraph.add]("add")
            .def_method[PyGraph.connect]("connect")
            .def_method[PyGraph.set]("set")
            .def_method[PyGraph.process]("process")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module graph_ext: ", e))
