#!/usr/bin/env python3
"""RSC2 encoder: WAV to .rsc2, v2.

Every frame is analysed through a periodic Hann window at hop = fft_size / 2, and the frame
record keeps the exact same byte layout as v1, so the finished Roblox decoder still parses it.
See rsc2_codec.py for what changed and the reasoning behind it.

  uv run python rsc_encoder.py input.wav output.rsc2 [--max-partials 24]
"""

import argparse
import time
import wave

import numpy as np

import rsc2_codec as codec


def analyse_channel(sig: np.ndarray, sample_rate: int, fft_size: int, hop_size: int,
                    max_partials: int, win: np.ndarray, block: int = codec.ANALYSIS_BLOCK):
    """Thin wrapper kept so this file reads as a pipeline; the analysis itself is shared with
    rscd_encoder.py through rsc2_codec.analyse_channel."""
    bins, mags, peak, phases = codec.analyse_channel(sig, sample_rate, fft_size, hop_size,
                                                    max_partials, win, block)
    peak_safe = np.where(peak > 1e-12, peak, 1.0)
    rel = np.where(peak[:, None] > 1e-12, mags / peak_safe[:, None], 0.0)
    return bins, rel, peak, phases


def pack_frames(bins: np.ndarray, rel: np.ndarray, peak: np.ndarray, phases: np.ndarray,
                max_partials: int) -> bytes:
    """One fixed-stride record per frame, so frame i is a single multiplication away.

    A record is (peak f32 as two words, n_partials u16, then max_partials triples of
    bin / amplitude / phase), all u16 words, all little-endian.
    """
    n_frames = bins.shape[0]
    bins        = np.where(rel > 0.0, bins, 1)     # silent slots point at a subsonic placeholder
    rec = np.zeros((n_frames, 3 + max_partials * 3), dtype="<u2")

    rec[:, 0:2] = peak.astype(np.float32).reshape(n_frames, 1).view(np.uint16)
    rec[:, 2]   = max_partials
    rec[:, 3::3] = bins.astype(np.uint16)
    rec[:, 4::3] = np.rint(rel * codec._U16_MAX).astype(np.uint16)
    rec[:, 5::3] = np.rint((phases + np.pi) * (codec._U16_MAX / (2.0 * np.pi))).astype(np.uint16)
    return rec.tobytes()


def encode(pcm: np.ndarray, sample_rate: int, fft_size: int = codec.FFT_SIZE,
           hop_size: int = codec.HOP_SIZE, max_partials: int = codec.MAX_PARTIALS,
           verbose: bool = True, block: int = codec.ANALYSIS_BLOCK) -> bytes:
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples, n_channels = pcm.shape

    win       = codec.periodic_hann(fft_size)
    window_sum = float(win.sum())
    blocks    = []

    for ch in range(n_channels):
        bins, rel, peak, phases = analyse_channel(pcm[:, ch], sample_rate,
                                                  fft_size, hop_size, max_partials, win, block)
        if verbose:
            print(f"   ch {ch}: {bins.shape[0]} frames, "
                  f"{int((rel > 1e-6).sum())} partials kept")
        blocks.append(pack_frames(bins, rel, peak, phases, max_partials))

    header = codec.encode_header(sample_rate, n_channels, n_channels,
                                 bins.shape[0], n_samples, window_sum, fft_size, hop_size,
                                 max_partials)
    return header + b"".join(blocks)


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
    ap = argparse.ArgumentParser(description="RSC2 v2 encoder")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--fft",          type=int,   default=codec.FFT_SIZE)
    ap.add_argument("--hop",          type=int,   default=codec.HOP_SIZE)
    ap.add_argument("--max-partials", type=int,   default=codec.MAX_PARTIALS)
    ap.add_argument("--analysis-block", type=int, default=codec.ANALYSIS_BLOCK,
                    help="frames in flight; lower it if you are memory constrained")
    a = ap.parse_args()

    if a.fft % a.hop != 0:
        ap.error("--hop must divide --fft; only whole-number overlap ratios reconstruct cleanly")
    if a.fft < 128 or a.hop < 32:
        ap.error("--fft and --hop are too small to be useful")

    print(f"🎵 Loading {a.input}…")
    pcm, sr, ch = load_wav(a.input)
    print(f"   {sr} Hz | {ch} ch | {len(pcm)} samples | {len(pcm)/sr:.2f} s")
    print(f"⚙️  Encoding fft={a.fft} hop={a.hop} max_partials={a.max_partials} "
          f"window=periodic hann")

    t0   = time.perf_counter()
    data = encode(pcm, sr, a.fft, a.hop, a.max_partials, block=a.analysis_block)
    dt   = time.perf_counter() - t0

    with open(a.output, "wb") as f:
        f.write(data)
    print(f"✅ {len(data):,} bytes → {a.output}  ({dt * 1000:.0f} ms, "
          f"{(len(pcm)/sr)/dt:.1f}× real time)")


if __name__ == "__main__":
    main()
