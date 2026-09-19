"""Offline time stretching: WSOLA.

Unlike every kernel here this is not a `Processor`: it reads a whole buffer
and writes a longer or shorter one, so it has no per-sample step and cannot
sit in a `Graph`.
"""

from std.math import cos, floor, pi, sqrt

from dsp.processor import SamplePtr

comptime TINY = 1.0e-12


@always_inline
def _clamp_int(value: Int, low: Int, high: Int) -> Int:
    return low if value < low else (high if value > high else value)


def _best_offset(
    src: SamplePtr,
    template: Int,
    ideal: Int,
    window: Int,
    search: Int,
    last: Int,
) -> Int:
    """Start within `search` of `ideal` whose window best follows `template`.

    Normalised cross-correlation, so a loud candidate does not win on level
    alone. Channel 0 decides for every channel: the splice has to be the same
    everywhere or the image moves.
    """
    var low = _clamp_int(ideal - search, 0, last)
    var high = _clamp_int(ideal + search, 0, last)
    var best = _clamp_int(ideal, 0, last)
    var best_score = -1.0e300
    var energy = 0.0
    for j in range(window):
        var v = Float64(src[unsafe_offset=low + j])
        energy += v * v
    for candidate in range(low, high + 1):
        var dot = 0.0
        for j in range(window):
            dot += Float64(src[unsafe_offset=template + j]) * Float64(
                src[unsafe_offset=candidate + j]
            )
        var score = dot / sqrt(energy + TINY)
        if score > best_score:
            best_score = score
            best = candidate
        if candidate < high:  # slide the energy rather than resumming it
            var leaving = Float64(src[unsafe_offset=candidate])
            var entering = Float64(src[unsafe_offset=candidate + window])
            energy += entering * entering - leaving * leaving
    return best


def timestretch(
    source: Int,
    destination: Int,
    channels: Int,
    frames_in: Int,
    frames_out: Int,
    window: Int,
    search: Int,
):
    """Write `frames_out` frames of the `frames_in` planar frames at *source*.

    Segments of `window` samples are overlap-added a half window apart in the
    output and a half window scaled by the stretch in the input; each is taken
    from wherever it continues the previous one best, which is what keeps
    periodic material in phase. The summed window is divided out at the end,
    so the ends are as loud as the middle.
    """
    var src = SamplePtr(unsafe_from_address=source)
    var dst = SamplePtr(unsafe_from_address=destination)
    if frames_out < 1 or frames_in < 1 or channels < 1:
        return
    var count = frames_out * channels
    for i in range(count):
        dst[unsafe_offset=i] = 0.0
    # A source shorter than one window has nothing to splice: copy what fits.
    if frames_in <= window:
        for c in range(channels):
            for i in range(min(frames_in, frames_out)):
                dst[unsafe_offset=c * frames_out + i] = src[
                    unsafe_offset=c * frames_in + i
                ]
        return

    var hop = window // 2
    var shape = List[Float32](length=window, fill=0.0)
    for j in range(window):
        # Periodic Hann: at 50% overlap the shifted copies sum to 1.
        shape[j] = Float32(0.5 - 0.5 * cos(2.0 * pi * Float64(j) / Float64(window)))
    # Nothing precedes the first segment, so its rising half stays flat.
    # Fading it in instead would divide the first samples by nearly nothing.
    var head = shape.copy()
    for j in range(hop):
        head[j] = 1.0
    var summed = List[Float32](length=frames_out, fill=0.0)

    var last = frames_in - window  # latest start that still holds a window
    var analysis = 0.0
    var step = Float64(hop) * Float64(frames_in) / Float64(frames_out)
    var template = -1  # no previous segment to follow yet
    var out_pos = 0
    while out_pos < frames_out:
        var ideal = _clamp_int(Int(analysis), 0, frames_in - 1)
        var start = ideal
        if template >= 0 and search > 0 and ideal <= last:
            start = _best_offset(src, template, ideal, window, search, last)
        # The last segments run off the end of the source; they add what is
        # left rather than repeating the last whole window.
        var span = min(window, min(frames_out - out_pos, frames_in - start))
        var weights = (head if out_pos == 0 else shape).unsafe_ptr()
        for c in range(channels):
            var out_base = c * frames_out + out_pos
            var in_base = c * frames_in + start
            for j in range(span):
                dst[unsafe_offset=out_base + j] += (
                    src[unsafe_offset=in_base + j] * weights[unsafe_offset=j]
                )
        for j in range(span):
            summed[out_pos + j] += weights[unsafe_offset=j]
        template = _clamp_int(start + hop, 0, last)
        analysis += step
        out_pos += hop

    for i in range(frames_out):
        var weight = summed[i]
        if weight > 1.0e-3:
            for c in range(channels):
                dst[unsafe_offset=c * frames_out + i] /= weight
