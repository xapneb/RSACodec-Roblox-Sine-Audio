#!/usr/bin/env python3
"""RSC2 analysis → Desmos JSON.

Same analysis as rsc_encoder.py, same frames, same bins. Both go through
rsc2_codec.analyse_channel so a .rsc2 file and a .rscd file made from the same wav can never
agree on some partials and not others again, which is what the two parallel scoring kernels
used to guarantee.

Schema, and the one convention worth knowing:

  "peak"        the frame's magnitude times 2 / window_sum, i.e. already in units of tone
                amplitude, and "amp_norm" is relative to that. Multiplying the two gives an
                absolute amplitude, so nothing downstream has to know about rfft conventions.
                Folding the factor into the frame peak is exact here because the selector never
                picks bin 0 or the Nyquist bin, which are the only bins rfft treats specially.

  "freq_hz"     bin * sample_rate / fft_size. Bin-quantised: accurate enough to hit the right
                note, not accurate enough to hold a vibrato in tune. See NOTES.md.

  "time_sec"    frame start = frame_index * hop_size / sample_rate.

  uv run python rscd_encoder.py input.wav output.rscd
"""

import argparse
import json
import time
import wave

import numpy as np

import rsc2_codec as codec


def encode_json(pcm: np.ndarray, sample_rate: int, fft_size: int, hop_size: int,
                max_partials: int, block: int = codec.ANALYSIS_BLOCK,
                max_frames: int = None, gain: float = 1.0) -> dict:
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if max_frames is not None:                       # stop before analysing what we will drop
        pcm = pcm[:max_frames * hop_size + fft_size]
    n_samples, n_channels = pcm.shape

    win        = codec.periodic_hann(fft_size)
    window_sum = float(win.sum())
    hop_sec    = hop_size / sample_rate

    channels, n_frames = [], None
    for ch in range(n_channels):
        bins, mags, peak, phases = codec.analyse_channel(pcm[:, ch], sample_rate, fft_size,
                                                        hop_size, max_partials, win, block)
        n_frames = bins.shape[0] if n_frames is None else min(n_frames, bins.shape[0])
        peak_safe = np.where(peak > 1e-12, peak, 1.0)
        rel = np.where(peak[:, None] > 1e-12, mags / peak_safe[:, None], 0.0)
        peak_abs = peak * 2.0 / window_sum * gain   # headroom, applied after selection so the
        # partial list and phases stay identical to a gain=1 encode (synthesis is linear)

        frames = []
        for i in range(n_frames):
            partials = [{"bin": int(b),
                         "freq_hz": round(int(b) * sample_rate / fft_size, 4),
                         "amp_norm": round(float(a), 6),
                         "phase_rad": round(float(p), 6)}
                        for b, a, p in zip(bins[i], rel[i], phases[i]) if a > 0.0]
            frames.append({"frame_idx": i,
                           "time_sec": round(i * hop_sec, 6),
                           "peak": round(float(peak_abs[i]), 8),
                           "partials": partials})
        channels.append({"frames": frames})
        print(f"   ch {ch}: {n_frames} frames")

    return {"sample_rate": sample_rate, "n_channels": n_channels, "n_frames": n_frames,
            "fft_size": fft_size, "hop_size": hop_size, "max_partials": max_partials,
            "window_sum": window_sum, "duration_sec": round(n_samples / sample_rate, 6),
            "channels": channels}


def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        s = np.frombuffer(raw, np.int16).astype(np.float32)
        s /= 32768.0
    elif width == 4:
        s = np.frombuffer(raw, np.int32).astype(np.float32)
        s /= 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    return s.reshape(-1, ch), sr, ch


def main():
    ap = argparse.ArgumentParser(description="RSC2 → Desmos JSON")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--fft",          type=int, default=codec.FFT_SIZE)
    ap.add_argument("--hop",          type=int, default=codec.HOP_SIZE)
    ap.add_argument("--max-partials", type=int, default=codec.MAX_PARTIALS)
    ap.add_argument("--max-frames",   type=int, default=None,
                    help="analyse only the first N frames (Desmos expression lists are finite)")
    ap.add_argument("--analysis-block", type=int, default=codec.ANALYSIS_BLOCK,
                    help="frames in flight; lower it if you are memory constrained")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="linear headroom gain folded into the frame peak; applied after "
                         "partial selection so nothing is dropped or moved, and the "
                         "measured value for a file can be read off rsc_encoder.py's two-pass "
                         "(the two encoders share the analysis)")
    a = ap.parse_args()

    print(f"🎵 {a.input}")
    pcm, sr, ch = load_wav(a.input)
    print(f"   {sr} Hz | {ch} ch | {len(pcm)} samples | {len(pcm)/sr:.2f} s")
    print(f"⚙️  fft={a.fft} hop={a.hop} partials={a.max_partials} window=periodic hann")

    t0   = time.perf_counter()
    out  = encode_json(pcm, sr, a.fft, a.hop, a.max_partials, a.analysis_block, a.max_frames)
    dt   = time.perf_counter() - t0

    print(f"💾 {a.output}")
    with open(a.output, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"✅ {out['n_frames']} frames × {ch} ch in {dt*1000:.0f} ms "
          f"→ pass to rscd_prep.py")


if __name__ == "__main__":
    main()
