"""Reference model of the AudioStreamWriter playback path, in numpy.

    uv run python stream_engine_check.py <captured.f32> audio.rsc2

The Lua engine in rsc_decoder.lua synthesizes with a continuous oscillator per bin and an
exponential amplitude ramp, instead of the windowed overlap-add that rsc_decoder.py uses. This
script runs the identical algorithm in numpy so the Lua code can be checked off-line, without
Roblox and without having to trust what it printed.

The captured file is float32 little-endian, channel-major, exactly the bytes that a real
AudioStreamWriter receives from Write(). If the model and the capture agree, the Lua engine is
implemented the way it is documented. If they agree and the sound is still wrong, then the
algorithm is the thing to argue about, not the code.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import rsc2_codec as codec

AMP_EPS = 1e-5


def read_frames(data):
    """Header plus per-channel (bins, amps, phases) arrays, read straight off the frame layout."""
    h = codec.decode_header(data)
    stride = codec.frame_stride(h["max_partials"])
    start  = codec.data_start()

    part = np.dtype([("bin", "<u2"), ("amp", "<u2"), ("phase", "<u2")])
    rec  = np.dtype([("peak", "<f4"), ("npart", "<u2"), ("parts", part, (h["max_partials"],))])
    assert rec.itemsize == stride, (rec.itemsize, stride)

    nyquist = h["fft_size"] // 2
    chans = []
    for ch in range(h["channels"]):
        block = np.frombuffer(data[start + ch * h["n_frames"] * stride:
                                   start + (ch + 1) * h["n_frames"] * stride], dtype=rec)
        peak  = block["peak"].astype(np.float64)[:, None]
        bins  = block["parts"]["bin"].astype(np.int32)
        rel   = block["parts"]["amp"].astype(np.float64) / 65535.0
        phase = block["parts"]["phase"].astype(np.float64)

        scale = np.where((bins > 0) & (bins < nyquist), 2.0, 1.0)
        amp   = rel * peak * scale / h["window_sum"]
        ph    = (phase / 65535.0) * 2 * np.pi - np.pi
        chans.append((bins, amp, ph, h))
    return chans


def model(bins, amp, phase, h, n_frames, out=None, record=None, work=None):
    """The stream engine's algorithm: exact bin frequency, phase accumulated, gain ramped.
    Pass a list as `record` to collect the slot state at the end of every hop, for cross-checking
    against the Lua engine's own state.

    Past the last frame the Lua engine keeps playing the final frame, so this mirrors that with
    `work`: the number of samples to fill, whether or not the file has frames for them."""
    fft, hop, sr = h["fft_size"], h["hop_size"], h["sample_rate"]
    rampK = 1.0 - np.exp(-(1.0 / sr) / ((hop / 2.0) / sr))
    decay = 1.0 - rampK
    if work is None:
        work = n_frames * hop
    if out is None:
        out = np.zeros(work, dtype=np.float64)
    else:
        out[:] = 0
    len_out = min(work, len(out))

    k      = np.arange(hop)
    slots  = {}                                # bin -> [amp, target, phase]
    n_total = int(np.ceil(len_out / hop))
    for i in range(n_total):
        fi = i if i < n_frames else n_frames - 1
        present = set()
        for b, a, p in zip(bins[fi], amp[fi], phase[fi]):
            if a <= AMP_EPS or b <= 0 or b >= h["fft_size"] // 2:
                continue
            s = slots.get(int(b))
            if s is None:
                slots[int(b)] = [0.0, float(a), (float(p) / (2 * np.pi)) % 1.0]
            else:
                s[1] = float(a)
            present.add(int(b))
        for b in list(slots):
            if b not in present:
                slots[b][1] = 0.0

        base = i * hop
        step = min(base + hop, len_out)
        seg  = step - base
        k = np.arange(seg) if seg != hop else np.arange(hop)

        for b, (a0, target, ph0) in list(slots.items()):
            inc = b / fft
            # The Lua engine ramps gain and advances phase before emitting each sample, so the
            # first sample of a hop is already k=1, not k=0. Mirroring that exactly.
            j = k + 1
            amp_ramp = target + (a0 - target) * (decay ** j)
            ph = ph0 + inc * j
            ph -= np.floor(ph)
            out[base:step] += amp_ramp * np.sin(2 * np.pi * ph)
            slots[b][0] = float(target + (a0 - target) * decay ** seg)
            slots[b][2] = float((ph0 + inc * seg) % 1.0)     # phase must carry into the next hop
            if record is not None:
                record.append((i, ch_of(h), b, slots[b][0], slots[b][1],
                               float((ph0 + inc * seg) % 1.0)))
            if slots[b][1] <= 0 and slots[b][0] <= AMP_EPS:
                del slots[b]
    return np.clip(out, -1.0, 1.0)


_CHANNEL = [0]


def ch_of(_):
    return _CHANNEL[0]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cap_path, rsc_path = sys.argv[1], sys.argv[2]
    data = open(rsc_path, "rb").read()
    chans = read_frames(data)
    h = chans[0][3]

    raw = np.frombuffer(open(cap_path, "rb").read(), dtype="<f4")
    per = len(raw) // max(h["channels"], 1)
    captured = raw.reshape(h["channels"], per)
    print(f"capture {per:,} frames per channel, {per / h['sample_rate']:.2f} s, "
          f"{h['channels']} ch, fft={h['fft_size']} hop={h['hop_size']}")

    work = min(per, 1_000_000)
    buf = np.zeros(work, dtype=np.float64)
    for ch in range(h["channels"]):
        bins, amp, phase, _ = chans[ch]
        ref = model(bins, amp, phase, h, h["n_frames"], buf, work=work)
        got = captured[ch, :work].astype(np.float64)
        err = got - ref
        print(f"  ch{ch}: model rms {ref.std():.5f}  capture rms {got.std():.5f}  "
              f"max err {np.abs(err).max():.2e}  es-signal {20*np.log10(ref.std()/err.std()):.1f} dB")


if __name__ == "__main__":
    main()
