#!/usr/bin/env python3
"""RSC2 decoder: .rsc2 back to a WAV file, v2.

Reconstruction is additive synthesis with overlap-add. Each frame becomes one tone per partial,
windowed by the analysis window and laid down at frame_index * hop_size, so consecutive frames
cross-fade instead of butting together. That is the whole click fix on this side: a hard frame
join is a one-sample step, and at 48 kHz with hop = 2048 that step landed 23 times a second.

Two things the v1 decoder got wrong and both show up as artefacts rather than errors:
  * magnitudes were divided by window_sum only, leaving the output 6 dB quiet everywhere except
    where a frame chose DC or Nyquist, which got the opposite treatment. rfft bins are half a
    tone, DC and Nyquist are whole tones, and the correction now comes from rsc2_codec.
  * the synthesiser advanced the phase before emitting the first sample of each partial, so
    every tone sat one sample late and Nyquist came out inverted.

  uv run python rsc_decoder.py input.rsc2 output.wav
"""

import argparse
import struct
import time
import wave

import numpy as np

import rsc2_codec as codec

try:
    from numba import njit
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap

_TWO_PI = 2.0 * np.pi


@njit(cache=True, fastmath=True)
def _synth(out, win, bins, amps, phases, hop_size, fft_size):
    """Lay every frame down over the output buffer. Phase is advanced after each emitted
    sample, which keeps frame 0 of a partial at the phase the analyser reported."""
    half = bins.shape[1]
    step_unit = _TWO_PI / fft_size
    for i in range(bins.shape[0]):
        base = i * hop_size
        for j in range(half):
            amp = amps[i, j]
            if amp <= 0.0:
                continue
            b = bins[i, j]
            step = step_unit * b
            p = phases[i, j]
            if p < 0.0:
                p += _TWO_PI
            for n in range(fft_size):
                out[base + n] += amp * win[n] * np.cos(p)
                p += step
                if p >= _TWO_PI:
                    p -= _TWO_PI
    return out


def _synth_numpy(out, win, bins, amps, phases, hop_size, fft_size):
    m   = np.arange(fft_size)
    w   = win
    for i in range(bins.shape[0]):
        base = i * hop_size
        keep = amps[i] > 0.0
        if not np.any(keep):
            continue
        b  = bins[i][keep].astype(np.float64)
        a  = amps[i][keep]
        ph = phases[i][keep]
        frame = (a[:, None] * w[None, :] * np.cos(2.0 * np.pi * b[:, None] * m[None, :] / fft_size
                                                  + ph[:, None]))
        out[base:base + fft_size] += frame.sum(axis=0)


def overlap_normalisation(win: np.ndarray, fft_size: int, hop_size: int) -> tuple:
    """The constant that overlap-add leaves behind, plus a warning if it is not flat.

    For hop = fft_size / 2 and a periodic Hann this comes out as exactly 1.0, which is why that
    pairing is the default. Any other whole-number overlap ratio still reconstructs, but the
    window sum wobbles within a hop and you hear it as a level ripple.
    """
    ratio = fft_size / hop_size
    if ratio != int(ratio):
        return 1.0, f"overlap ratio {ratio} is not a whole number"
    r = int(ratio)
    if r == 1:
        return 1.0, "no overlap, frames butt together"

    one = np.zeros(hop_size)
    for j in range(r):
        one += win[j * hop_size:(j + 1) * hop_size]
    flat = float(np.mean(one))
    if one.max() - one.min() > 1e-9:
        return flat, (f"overlap sum wobbles between {one.min():.3f} and {one.max():.3f}, "
                      "expect level ripple")
    return flat, "flat"


def decode(data: bytes):
    h = codec.decode_header(data)
    print(f"   RSC2 v{h['version']} | {h['sample_rate']} Hz | {h['channels']} ch | "
          f"fft={h['fft_size']} hop={h['hop_size']} partials={h['max_partials']} "
          f"frames={h['n_frames']}")

    fft_size = h["fft_size"]
    hop_size = h["hop_size"]
    if hop_size <= 0 or hop_size > fft_size:
        raise ValueError(f"implausible hop_size {hop_size} for fft_size {fft_size}")
    if h["n_frames"] < 1:
        raise ValueError("file carries no frames")
    overlap  = fft_size / hop_size
    if overlap != int(overlap):
        raise ValueError(f"hop_size {hop_size} must divide fft_size {fft_size} evenly")
    ratio = int(overlap)

    win = codec.periodic_hann(fft_size) if ratio > 1 else np.ones(fft_size)
    norm, note = overlap_normalisation(win, fft_size, hop_size)
    if note != "flat":
        print(f"   ⚠️  {note}")

    stride  = codec.frame_stride(h["max_partials"])
    start   = codec.data_start()
    total   = (h["n_frames"] - 1) * hop_size + fft_size
    pcm_out = np.zeros((total, h["channels"]), dtype=np.float32)

    for ch in range(h["channels"]):
        block = data[start + ch * h["n_frames"] * stride:]
        rec = np.frombuffer(block, dtype="<u2",
                            count=h["n_frames"] * (stride // 2)).reshape(h["n_frames"],
                                                                        stride // 2)

        peak   = np.ascontiguousarray(rec[:, 0:2]).view("<f4").astype(np.float64).reshape(-1)
        bins   = rec[:, 3::3].astype(np.int64)
        amp_u  = rec[:, 4::3].astype(np.float64) / codec._U16_MAX
        phase  = rec[:, 5::3].astype(np.float64) * (_TWO_PI / codec._U16_MAX) - np.pi

        # peak * amp_u16/65535 is the frame's rfft magnitude at that bin; amplitude_scale
        # unwraps the half-tone convention and window_sum undoes the analysis window gain.
        scale = codec.amplitude_scale(bins, fft_size)
        amps  = amp_u * peak[:, None] * scale / h["window_sum"]

        frame_out = np.zeros(total, dtype=np.float64)     # accumulator stays 64-bit
        if _USE_NUMBA:
            _synth(frame_out, win, bins.astype(np.int32), amps, phase, hop_size, fft_size)
        else:
            _synth_numpy(frame_out, win, bins, amps, phase, hop_size, fft_size)
        pcm_out[:, ch] = frame_out
        pcm_out[:, ch] /= norm
        del frame_out

    return pcm_out, h["sample_rate"], h["channels"], h["n_samples"]


def save_wav(path: str, pcm: np.ndarray, sr: int):
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    clipped = np.clip(pcm, -1.0, 1.0)
    raw = (clipped * 32767.0).astype(np.int16).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def main():
    ap = argparse.ArgumentParser(description="RSC2 v2 decoder")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--trim", action="store_true",
                    help="cut the synthesis window tail instead of writing extra silence")
    a = ap.parse_args()

    print(f"📂 {a.input}")
    with open(a.input, "rb") as f:
        data = f.read()
    print(f"   {len(data):,} bytes | numba {'on' if _USE_NUMBA else 'off'}")

    t0   = time.perf_counter()
    pcm, sr, ch, n_samples = decode(data)
    dt   = time.perf_counter() - t0

    if a.trim:
        pcm = pcm[:n_samples]
    save_wav(a.output, pcm, sr)
    print(f"✅ {a.output}: {pcm.shape[0]} samples {sr} Hz {ch} ch "
          f"({pcm.shape[0]/sr:.2f} s) decoded in {dt * 1000:.0f} ms")


if __name__ == "__main__":
    main()
