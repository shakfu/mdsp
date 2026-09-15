import sys, time, threading
sys.path.insert(0, ".")
import numpy as np, gil2
N = 48000 * 200
xs = [np.random.default_rng(i).standard_normal(N).astype(np.float32) for i in range(4)]
ys = [np.empty_like(x) for x in xs]
def work(i): gil2.process(xs[i].ctypes.data, ys[i].ctypes.data, N)
t = time.perf_counter(); work(0); one = time.perf_counter() - t
t = time.perf_counter()
th = [threading.Thread(target=work, args=(i,)) for i in range(4)]
[x.start() for x in th]; [x.join() for x in th]
four = time.perf_counter() - t
print(f"1 call: {one*1e3:.1f} ms; 4 threads: {four*1e3:.1f} ms; speedup: {4*one/four:.2f}x")
ref = np.empty_like(ys[0]); 
print("ok" if np.all(np.isfinite(ys[3])) else "bad")
