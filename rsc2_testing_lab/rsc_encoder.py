#!/usr/bin/env python3
"""RSC2 encoder: WAV to .rsc2.

Every frame is analysed through a periodic Hann window at hop = fft_size / 2. Version 2 keeps
the v1 byte layout; version 3 (the default) packs each partial into one 32-bit word, which
cuts about a third off the file. See rsc2_codec.py for what changed and the reasoning behind it.

  uv run python rsc_encoder.py input.wav output.rsc2 [--max-partials 24] [--version 3]
"""

import argparse
import time
import wave

import numpy as np

import rsc2_codec as codec

import rsc_decoder  # only to decode the first pass and measure what the codec actually emits


def analyse_channel(sig: np.ndarray, sample_rate: int, fft_size: int, hop_size: int,
                    max_partials: int, win: np.ndarray, block: int = codec.ANALYSIS_BLOCK,
                    psycho: bool = True):
    """Thin wrapper kept so this file reads as a pipeline; the analysis itself is shared with
    rscd_encoder.py through rsc2_codec.analyse_channel."""
    bins, mags, peak, phases = codec.analyse_channel(sig, sample_rate, fft_size, hop_size,
                                                    max_partials, win, block, psycho=psycho)
    peak_safe = np.where(peak > 1e-12, peak, 1.0)
    rel = np.where(peak[:, None] > 1e-12, mags / peak_safe[:, None], 0.0)
    return bins, rel, peak, phases


def pack_frames(bins: np.ndarray, rel: np.ndarray, peak: np.ndarray, phases: np.ndarray,
                max_partials: int, fft_size: int = codec.FFT_SIZE,
                version: int = 3) -> bytes:
    """One fixed-stride record per frame, so frame i is a single multiplication away.

    v2 (version < 3) writes three u16 words per partial. v3 packs bin and amplitude into a
    24-bit little-endian triple followed by the same u16 phase v2 used — see
    rsc2_codec.pack_bits — and writes the all-zero record for silent slots, replacing the
    v1/v2 placeholder-bin trick. The frame header (peak f32, then n_partials u16) is identical.
    """
    n_frames = bins.shape[0]

    if version < 3:
        rec = np.zeros((n_frames, 3 + max_partials * 3), dtype="<u2")
        head = np.zeros((n_frames, 3), dtype="<u2")
        head[:, 0:2] = peak.astype(np.float32).reshape(n_frames, 1).view(np.uint16)
        head[:, 2]   = max_partials
        rec[:, 0:3] = head
        bins         = np.where(rel > 0.0, bins, 1)  # silent slots keep v1's placeholder bin
        rec[:, 3::3] = bins.astype(np.uint16)
        rec[:, 4::3] = np.rint(rel * codec._U16_MAX).astype(np.uint16)
        rec[:, 5::3] = np.rint((phases + np.pi) * (codec._U16_MAX / (2.0 * np.pi))).astype(np.uint16)
        return rec.tobytes()

    bin_bits, amp_bits = codec.pack_bits(fft_size)
    amp_mask = (1 << amp_bits) - 1

    bins = np.where(rel > 0.0, bins, 0)              # silent slot = word 0, bin 0 is skipped
    bins = np.where(bins < (1 << bin_bits), bins, 0) # safety: never emit a bin that truncates
    amps = np.clip(np.rint(rel * amp_mask), 0, amp_mask)
    packed = (bins.astype(np.uint32)
              | (amps.astype(np.uint32) << bin_bits))  # 24 bits used
    # phase keeps v2's exact (-pi, pi] u16 grid, so frame overlap loses no continuity
    ph_u = (np.rint((phases + np.pi) * (codec._U16_MAX / (2.0 * np.pi)))
            .astype(np.uint32) & 0xFFFF)

    buf = np.zeros((n_frames, 6 + max_partials * 5), dtype=np.uint8)
    buf[:, 0:4] = peak.astype(np.float32).reshape(n_frames, 1).view(np.uint8)
    buf[:, 4] = max_partials & 0xFF
    buf[:, 5] = max_partials >> 8
    rec = np.empty((n_frames, max_partials, 5), dtype=np.uint8)
    rec[:, :, 0] = packed & 0xFF
    rec[:, :, 1] = (packed >> 8) & 0xFF
    rec[:, :, 2] = (packed >> 16) & 0xFF
    rec[:, :, 3] = ph_u & 0xFF
    rec[:, :, 4] = ph_u >> 8
    buf[:, 6:] = rec.reshape(n_frames, max_partials * 5)
    return buf.tobytes()


def encode(pcm: np.ndarray, sample_rate: int, fft_size: int = codec.FFT_SIZE,
           hop_size: int = codec.HOP_SIZE, max_partials: int = codec.MAX_PARTIALS,
           verbose: bool = True, block: int = codec.ANALYSIS_BLOCK,
           version: int = 3, psycho: bool = True, gain: float = 1.0) -> bytes:
    """Encode one file. `gain` is applied to the per-frame peak AFTER analysis, not to the
    samples before it — synthesis is linear in amplitude, so scaling the peak scales the output
    exactly, while selection and bit allocation stay byte-identical to a gain=1 encode. Scaling
    the waveform beforehand would move every partial against the absolute dBFS thresholds and
    silently drop the quiet ones the psycho gate would otherwise keep.
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples, n_channels = pcm.shape

    win       = codec.periodic_hann(fft_size)
    window_sum = float(win.sum())
    if version >= 3:
        codec.pack_bits(fft_size)   # fail fast if this fft_size cannot be packed
    blocks    = []

    for ch in range(n_channels):
        bins, rel, peak, phases = analyse_channel(pcm[:, ch], sample_rate,
                                                  fft_size, hop_size, max_partials, win, block,
                                                  psycho=psycho)
        if verbose:
            print(f"   ch {ch}: {bins.shape[0]} frames, "
                  f"{int((rel > 1e-6).sum())} partials kept")
        if gain != 1.0:
            peak = peak * gain
        blocks.append(pack_frames(bins, rel, peak, phases, max_partials, fft_size, version))

    header = codec.encode_header(sample_rate, n_channels, n_channels,
                                 bins.shape[0], n_samples, window_sum, fft_size, hop_size,
                                 max_partials, version=version)
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
    ap = argparse.ArgumentParser(description="RSC2 encoder")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--fft",          type=int,   default=codec.FFT_SIZE)
    ap.add_argument("--hop",          type=int,   default=codec.HOP_SIZE)
    ap.add_argument("--max-partials", type=int,   default=codec.MAX_PARTIALS)
    ap.add_argument("--version",      type=int,   default=3, choices=(2, 3),
                    help="frame layout: 2 is the wide v1-compatible triple, 3 packs 33%% smaller")
    ap.add_argument("--no-psy",       action="store_true",
                    help="disable the psychoacoustic gate and select on raw salience, as v2 did")
    ap.add_argument("--ceiling",      type=float, default=1.0,
                    help="decode the first pass, and if the output peak sits above this, re-encode"
                         " with a gain that brings it down; 0 disables (default 1.0)")
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
          f"window=periodic hann psycho={'off' if a.no_psy else 'on'}")

    t0   = time.perf_counter()
    data = encode(pcm, sr, a.fft, a.hop, a.max_partials, block=a.analysis_block,
                  version=a.version, psycho=not a.no_psy)
    dt   = time.perf_counter() - t0

    if a.ceiling > 0:
        # Overshoot is a property of the material as much as the codec — hard-limited input rings
        # much harder — so the gain is measured from the first pass, not assumed. The decoder
        # returns un-clamped float, which is what makes this exact.
        pk = float(np.abs(rsc_decoder.decode(data)[0]).max())
        if pk > a.ceiling:
            g = a.ceiling / pk
            print(f"   output peak {20 * np.log10(pk):+.2f} dBFS -> re-encoding at "
                  f"{20 * np.log10(g):+.2f} dB gain to sit under {20 * np.log10(a.ceiling):+.1f}")
            data = encode(pcm, sr, a.fft, a.hop, a.max_partials, block=a.analysis_block,
                          version=a.version, psycho=not a.no_psy, gain=g)
        dt = time.perf_counter() - t0

    with open(a.output, "wb") as f:
        f.write(data)
    print(f"✅ {len(data):,} bytes → {a.output}  ({dt * 1000:.0f} ms, "
          f"{(len(pcm)/sr)/dt:.1f}× real time)")


if __name__ == "__main__":
    main()
