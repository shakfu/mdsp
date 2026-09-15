import time, sys
sys.path.insert(0, ".")
import numpy as np, onepole_ext as spk2
from math import exp, pi
f = spk2.OnePole(); print(f)
f.set_cutoff(1000.0, 48000.0)
x = np.random.default_rng(0).standard_normal(48000*10).astype(np.float32)
y = np.empty_like(x)
t=time.perf_counter(); f.process(x.ctypes.data, y.ctypes.data, x.size); dt=time.perf_counter()-t
print(f"mojo 10s@48k: {dt*1e3:.2f} ms, {x.size/dt/1e6:.0f} M samp/s")
# reference
a = np.float32(1-exp(-2*pi*1000/48000)); z=np.float32(0); r=np.empty(4800,np.float32)
for i in range(4800):
    z += a*(x[i]-z); r[i]=z
print("max err vs ref (first 4800):", np.abs(r-y[:4800]).max())
t=time.perf_counter()
for _ in range(10000): f.process(x.ctypes.data, y.ctypes.data, 64)
print(f"call overhead per 64-sample call: {(time.perf_counter()-t)/10000*1e6:.1f} us")
