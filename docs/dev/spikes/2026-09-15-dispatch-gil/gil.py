import sys, time, threading
sys.path.insert(0, ".")
import numpy as np, onepole_ext as spk2
N = 48000 * 200
xs = [np.random.default_rng(i).standard_normal(N).astype(np.float32) for i in range(4)]
ys = [np.empty_like(x) for x in xs]
fs = [spk2.OnePole() for _ in xs]
for f in fs: f.set_cutoff(1000.0, 48000.0)
def work(i): fs[i].process(xs[i].ctypes.data, ys[i].ctypes.data, N)
t = time.perf_counter(); work(0); one = time.perf_counter() - t
t = time.perf_counter()
th = [threading.Thread(target=work, args=(i,)) for i in range(4)]
[x.start() for x in th]; [x.join() for x in th]
four = time.perf_counter() - t
print(f"1 call: {one*1e3:.1f} ms; 4 threads: {four*1e3:.1f} ms; speedup vs 4 sequential: {4*one/four:.2f}x")
