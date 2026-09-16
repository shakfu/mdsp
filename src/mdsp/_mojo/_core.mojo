"""Python extension module `mdsp._core`.

Each kernel is exposed as `Bank[Kernel]`: one kernel instance per channel.
Buffers cross as addresses of planar `[channels, frames]` float32 arrays:
the audio input, the output, and one optional buffer per modulation input.
`mdsp._base` validates dtype, layout and shape before passing an address;
nothing here can check them.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import size_of

from dsp import (
    MAX_INPUTS,
    Biquad,
    Delay,
    Gain,
    OnePole,
    Phasor,
    Ports,
    Processor,
    SamplePtr,
    Saw,
    Sine,
    Square,
    Svf,
)


struct Bank[P: Processor](Movable, Writable):
    var units: List[Self.P]

    def __init__(out self, sample_rate: Float64, channels: Int):
        self.units = List[Self.P](capacity=channels)
        for _ in range(channels):
            self.units.append(Self.P(sample_rate))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Bank(channels=", len(self.units), ")")

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 2:
            raise Error("expected (sample_rate, channels)")
        var sample_rate = Float64(py=args[0])
        var channels = Int(py=args[1])
        if not sample_rate > 0.0:
            raise Error("sample_rate must be positive")
        if channels < 1:
            raise Error("channels must be >= 1")
        self = Self(sample_rate, channels)

    @staticmethod
    def param_names(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var names = Python.list()
        for name in Self.P.param_names():
            names.append(PythonObject(name))
        return names

    @staticmethod
    def input_names(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        var names = Python.list()
        for name in Self.P.input_names():
            names.append(PythonObject(name))
        return names

    @staticmethod
    def set(
        self_ptr: Pointer[Self, MutAnyOrigin], param: PythonObject, value: PythonObject
    ) raises -> PythonObject:
        var p = Int(py=param)
        var v = Float64(py=value)
        for ref unit in self_ptr[].units:
            unit.set(p, v)
        return PythonObject(None)

    @staticmethod
    def reset(self_ptr: Pointer[Self, MutAnyOrigin]) raises -> PythonObject:
        for ref unit in self_ptr[].units:
            unit.reset()
        return PythonObject(None)

    @staticmethod
    def process(
        self_ptr: Pointer[Self, MutAnyOrigin],
        src: PythonObject,
        dst: PythonObject,
        frames: PythonObject,
        mods: PythonObject,
    ) raises -> PythonObject:
        """Process `len(units)` planar channels of `frames` samples each.

        `mods` holds one address per modulation input, in `input_names()` order
        after the audio input; 0 means unconnected.
        """
        var src_addr = Int(py=src)
        var dst_addr = Int(py=dst)
        var n = Int(py=frames)
        var stride = n * size_of[Float32]()
        var mod_addrs = InlineArray[Int, MAX_INPUTS](fill=0)
        var num_mods = Int(py=len(mods))
        if num_mods > MAX_INPUTS - 1:
            raise Error("too many modulation inputs")
        for k in range(num_mods):
            mod_addrs[k + 1] = Int(py=mods[k])
        var ports = InlineArray[Int, MAX_INPUTS](fill=0)
        var ports_ptr = Ports(unsafe_from_address=Int(Pointer(to=ports)))
        ref units = self_ptr[].units
        # No PythonObject may be touched while the GIL is released.
        ref cpython = Python().cpython()
        var thread_state = cpython.PyEval_SaveThread()
        for c in range(len(units)):
            ports[0] = src_addr + c * stride
            for k in range(1, num_mods + 1):
                ports[k] = mod_addrs[k] + c * stride if mod_addrs[k] != 0 else 0
            units[c].process(
                ports_ptr, SamplePtr(unsafe_from_address=dst_addr + c * stride), n
            )
        cpython.PyEval_RestoreThread(thread_state)
        return PythonObject(None)


def _add_bank[P: Processor](mut m: PythonModuleBuilder, name: StaticString) raises:
    _ = (
        m.add_type[Bank[P]](name)
        .def_py_init[Bank[P].py_init]()
        .def_method[Bank[P].param_names]("param_names")
        .def_method[Bank[P].input_names]("input_names")
        .def_method[Bank[P].set]("set")
        .def_method[Bank[P].reset]("reset")
        .def_method[Bank[P].process]("process")
    )


def _short_type_names(module: PythonObject) raises:
    """Rebind `mdsp._core.X` attributes to `X`.

    Types are registered under dotted names so CPython sets `__module__`.
    Without it, importlib warns on import and doctest collection fails.
    """
    var builtins = Python.import_module("builtins")
    for name in builtins.list(builtins.vars(module)):
        var parts = name.rsplit(".", 1)
        if len(parts) == 2:
            builtins.setattr(module, parts[1], builtins.getattr(module, name))
            builtins.delattr(module, name)


@export
def PyInit__core() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mdsp._core")
        _add_bank[Phasor](m, "mdsp._core.Phasor")
        _add_bank[Sine](m, "mdsp._core.Sine")
        _add_bank[Saw](m, "mdsp._core.Saw")
        _add_bank[Square](m, "mdsp._core.Square")
        _add_bank[OnePole](m, "mdsp._core.OnePole")
        _add_bank[Biquad](m, "mdsp._core.Biquad")
        _add_bank[Svf](m, "mdsp._core.Svf")
        _add_bank[Gain](m, "mdsp._core.Gain")
        _add_bank[Delay](m, "mdsp._core.Delay")
        var module = m.finalize()
        _short_type_names(module)
        return module
    except e:
        abort(String("failed to create module mdsp._core: ", e))
