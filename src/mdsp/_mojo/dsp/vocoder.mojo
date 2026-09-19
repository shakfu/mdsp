"""Phase vocoder time stretching, built on `dsp.fft`.

WSOLA splices waveform segments; this resynthesises them. Each bin's phase is
advanced by the frequency it actually holds rather than the one its bin stands
for, so overlapping frames stay coherent at any stretch. Tonal material keeps
its pitch and does not phase-cancel; transients smear, which is the trade
against `dsp.stretch`.
"""

from std.math import atan2, cos, floor, pi, sin, sqrt

from dsp.fft import RealFft, round_up_pow2
from dsp.processor import SamplePtr

comptime OVERLAP = 4  # hop is a quarter window: Hann squared then sums flat


@always_inline
def _mirror(index: Int, length: Int) -> Int:
    """Reflect an out-of-range index back into [0, length).

    Edge frames reach past both ends of the source. Reading zeros there tapers
    the first and last `window * factor` output samples, because the frames
    that cover them carry less signal than the window they are divided by;
    reflecting keeps them full. It never affects output at factor 1, where
    only in-range indices are ever read.
    """
    if length < 2:
        return 0
    var i = index
    while i < 0 or i >= length:
        if i < 0:
            i = -i
        if i >= length:
            i = 2 * length - 2 - i
    return i


@always_inline
def _principal(angle: Float64) -> Float64:
    """Wrap to (-pi, pi]: the phase step a bin's own frequency cannot explain."""
    return angle - 2.0 * pi * floor(angle / (2.0 * pi) + 0.5)


def vocoder_stretch(
    source: Int,
    destination: Int,
    channels: Int,
    frames_in: Int,
    frames_out: Int,
    window: Int,
):
    """Write `frames_out` frames of the `frames_in` planar frames at *source*.

    Channel 0 measures the phase advance and every channel is rotated by the
    same amount, so inter-channel phase, and with it the stereo image, comes
    through unchanged.
    """
    var src = SamplePtr(unsafe_from_address=source)
    var dst = SamplePtr(unsafe_from_address=destination)
    if frames_out < 1 or frames_in < 1 or channels < 1:
        return
    for i in range(frames_out * channels):
        dst[unsafe_offset=i] = 0.0

    var n = round_up_pow2(window)
    # A bin's phase step is only unambiguous while the analysis hop stays at
    # a quarter window, so compression shortens the synthesis hop instead of
    # lengthening the analysis one. Below 0.75 it cost up to 2.1 dB.
    var factor = Float64(frames_out) / Float64(frames_in)
    var hop = n // OVERLAP
    if factor < 1.0:
        hop = max(1, Int(Float64(n) / Float64(OVERLAP) * factor))
    var fft = RealFft(n)
    var bins = fft.bins()

    var shape = List[Float32](length=n, fill=0.0)
    for j in range(n):
        shape[j] = Float32(0.5 - 0.5 * cos(2.0 * pi * Float64(j) / Float64(n)))
    var frame = List[Float32](length=n, fill=0.0)
    var re = List[Float32](length=bins * channels, fill=0.0)
    var im = List[Float32](length=bins * channels, fill=0.0)
    var previous = List[Float64](length=bins, fill=0.0)  # last analysis phase
    var running = List[Float64](length=bins, fill=0.0)  # synthesis phase
    var summed = List[Float32](length=frames_out, fill=0.0)

    var step = Float64(hop) * Float64(frames_in) / Float64(frames_out)
    var expected_scale = 2.0 * pi * step / Float64(n)
    var advance = Float64(hop) / step if step > 0.0 else 1.0
    # Start OVERLAP - 1 frames early, reading zeros before the signal, so the
    # summed window is already at full height by output sample 0. Without it
    # the first samples divide by nearly nothing and are dropped.
    var frame_index = -(OVERLAP - 1)
    var out_pos = frame_index * hop
    while out_pos < frames_out:
        var start = Int(floor(Float64(frame_index) * step + 0.5))
        var first = max(0, -start)
        var last = min(n, frames_in - start)
        for c in range(channels):
            var base = c * frames_in
            for j in range(first):
                var index = _mirror(start + j, frames_in)
                frame[j] = src[unsafe_offset=base + index] * shape[j]
            for j in range(first, last):
                frame[j] = src[unsafe_offset=base + start + j] * shape[j]
            for j in range(last, n):
                var index = _mirror(start + j, frames_in)
                frame[j] = src[unsafe_offset=base + index] * shape[j]
            fft.forward(
                SamplePtr(unsafe_from_address=Int(frame.unsafe_ptr())),
                SamplePtr(unsafe_from_address=Int(re.unsafe_ptr()) + c * bins * 4),
                SamplePtr(unsafe_from_address=Int(im.unsafe_ptr()) + c * bins * 4),
            )
        for k in range(bins):
            var phase = atan2(Float64(im[k]), Float64(re[k]))
            if frame_index == 0:
                running[k] = phase
            else:
                var expected = expected_scale * Float64(k)
                var drift = _principal(phase - previous[k] - expected)
                running[k] += (expected + drift) * advance
            previous[k] = phase
            # One rotation, applied to every channel: it carries the phase
            # correction without touching any channel's magnitude.
            var turn = running[k] - phase
            var cr = Float32(cos(turn))
            var ci = Float32(sin(turn))
            for c in range(channels):
                var index = c * bins + k
                var xr = re[index]
                var xi = im[index]
                re[index] = xr * cr - xi * ci
                im[index] = xr * ci + xi * cr
        var from_j = max(0, -out_pos)
        var to_j = min(n, frames_out - out_pos)
        for c in range(channels):
            fft.inverse(
                SamplePtr(unsafe_from_address=Int(re.unsafe_ptr()) + c * bins * 4),
                SamplePtr(unsafe_from_address=Int(im.unsafe_ptr()) + c * bins * 4),
                SamplePtr(unsafe_from_address=Int(frame.unsafe_ptr())),
            )
            var out_base = c * frames_out + out_pos
            for j in range(from_j, to_j):
                dst[unsafe_offset=out_base + j] += frame[j] * shape[j]
        for j in range(from_j, to_j):
            summed[out_pos + j] += shape[j] * shape[j]
        frame_index += 1
        out_pos += hop

    for i in range(frames_out):
        var weight = summed[i]
        if weight > 1.0e-3:
            for c in range(channels):
                dst[unsafe_offset=c * frames_out + i] /= weight
