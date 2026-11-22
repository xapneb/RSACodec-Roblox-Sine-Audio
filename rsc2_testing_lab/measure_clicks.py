#!/usr/bin/env python3
"""Click harness for RSC2: does this build put steps into the audio?

A click is broadband, so the test is: band-limit the input, run it through the codec, then look
only at the range the input cannot possibly contain. Anything above the input's band was
manufactured by the codec, and a click makes a spike there rather than a hiss. Counting those
spikes answers "does it click". The level and error-to-signal columns explain what else the
codec is doing on the way.

  uv run python measure_clicks.py                     # bundled synthetic material
  uv run python measure_clicks.py track.wav
  uv run python measure_clicks.py --impl-dir /tmp/v1  # measure the old pipeline instead

The synthetic signal is deliberate: band-limited so the residual is honest, hard note attacks
and silences so partials appear and vanish, and one partial that sits between bins so its phase
keeps slipping. Those three are exactly the conditions a frame-boundary step needs to be heard.

Pass --impl-dir a directory holding an impl.py that exposes encode()/decode() the same way, to
measure any other build with the identical test.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

import rsc_decoder as rdec
import rsc_encoder as renc
import rsc2_codec as codec

GUARD = 3 * codec.FFT_SIZE      # skip the codec's own start and end ramp

IMPL = None


def v2_pair():
    return (lambda pcm, sr, fft, hop, mp, verbose=False: renc.encode(pcm, sr, fft, hop, mp, verbose)), \
           (lambda data: rdec.decode(data))


def band_limited_reference(seconds: float, sr: int, bin_hz: float = None) -> np.ndarray:
    """Four notes with sharp attacks, two silences, and a vibrato that lands off every bin.

    The note carriers are snapped to bin centres so the only thing this can show is a frame
    join. The last tone is deliberately left off-bin, because a partial whose phase keeps
    slipping is what an unwindowed frame boundary turns into a click on.
    """
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    snap = (lambda f: f) if bin_hz is None else (lambda f: round(f / bin_hz) * bin_hz)
    x = np.zeros_like(t)
    for freq, amp, start, length in ((snap(220.0), 0.30, 0.00, 0.55), (snap(330.0), 0.45, 0.90, 0.55),
                                     (snap(440.0), 0.22, 1.80, 0.55), (snap(587.0), 0.35, 2.70, 0.55)):
        held = np.clip(t - start, 0.0, None)
        mask = (t >= start) & (t < start + length)
        env = np.where(mask, np.exp(-6.0 * held), 0.0)
        vib = 1.0 + 0.006 * np.sin(2.0 * np.pi * 5.3 * held)
        x += amp * env * np.sin(2.0 * np.pi * freq * vib * held)
    x += 0.12 * np.sin(2.0 * np.pi * snap(900.0) * t) * (t > 3.4)   # partial that switches on
    x += 0.10 * np.sin(2.0 * np.pi * 1371.3 * t)                    # never lands on a bin
    return 0.75 * x / np.max(np.abs(x))


def residual_spikes(out: np.ndarray, sr: int, hop: int, cutoff_khz: float) -> dict:
    sos = butter(8, 2.0 * (cutoff_khz * 1000.0) / sr, btype="highpass", output="sos")
    hf = sosfilt(sos, out)
    body = hf[GUARD:-GUARD]
    hf_rms = float(np.sqrt(np.mean(body ** 2)))
    if 20.0 * np.log10(hf_rms + 1e-30) < -100.0:
        return {"hf_rms": hf_rms, "spikes": 0, "on_boundary": 0, "per_second": 0.0,
                "clean": True}

    med = float(np.median(np.abs(body))) or 1e-12
    spikes = np.flatnonzero(np.abs(hf) > 8.0 * med)
    spikes = spikes[(spikes > GUARD) & (spikes < len(hf) - GUARD)]
    rel = spikes % hop
    return {"hf_rms": hf_rms, "spikes": int(spikes.size),
            "on_boundary": int(np.sum(np.minimum(rel, hop - rel) <= 1)) if spikes.size else 0,
            "per_second": spikes.size / (len(out) / sr), "clean": False}


def score(label: str, original: np.ndarray, out: np.ndarray, sr: int, hop: int,
          cutoff_khz: float) -> None:
    if original.ndim > 1:
        original = original[:, 0]
    if out.ndim > 1:
        out = out[:, 0]
    n = min(len(original), len(out))
    original, out = original[:n], out[:n]
    lo, hi = GUARD, n - GUARD
    rms_in = float(np.sqrt(np.mean(original[lo:hi] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[lo:hi] ** 2)))
    es = 20.0 * np.log10(rms_in / (float(np.std(out[lo:hi] - original[lo:hi])) + 1e-12))
    s = residual_spikes(out, sr, hop, cutoff_khz)
    tail = ("residual below -100 dBFS, nothing to count" if s.get("clean") else
            f"residual {s['hf_rms'] * 1e6:8.1f} u   spikes {s['spikes']:5d} "
            f"({s['per_second']:5.1f}/s), {s['on_boundary']:4d} on frame joins")
    print(f"  {label:22s} level {rms_out / rms_in:6.3f}   es-signal {es:6.2f} dB   {tail}")


def read_wav_mono(path: Path, seconds: float):
    with wave.open(str(path), "rb") as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, np.int16).astype(np.float64).reshape(-1, ch) / 32768.0
    take = min(pcm.shape[0], int(seconds * sr))
    return pcm[:take], sr


def main():
    ap = argparse.ArgumentParser(description="measure RSC2 click artefacts")
    ap.add_argument("input", nargs="?", default=None,
                    help="wav to test in addition to the synthetic material")
    ap.add_argument("--fft",          type=int,   default=codec.FFT_SIZE)
    ap.add_argument("--hop",          type=int,   default=codec.HOP_SIZE)
    ap.add_argument("--max-partials", type=int,   default=codec.MAX_PARTIALS)
    ap.add_argument("--seconds",      type=float, default=6.0)
    ap.add_argument("--cutoff-khz",   type=float, default=7.0,
                    help="measure residual above this; raise it for material that reaches higher")
    ap.add_argument("--impl-dir",     default=None,
                    help="directory with impl.py to measure, for example the pre-fix pipeline")
    ap.add_argument("--compare-hops", action="store_true",
                    help="also run hop = fft and fft/4 so the overlap tradeoff is visible")
    a = ap.parse_args()

    global IMPL
    if a.impl_dir:
        sys.path.insert(0, str(Path(a.impl_dir).resolve()))
        for stale in ("rsc_encoder", "rsc_decoder", "rsc2_codec"):
            sys.modules.pop(stale, None)      # the v2 copies came in with this module
        import impl
        IMPL = (impl.encode, impl.decode)
        impl_label = f"{a.impl_dir}/impl.py"
    else:
        IMPL = v2_pair()
        impl_label = str(Path(__file__).resolve().parent)

    hops = [a.hop] if not a.compare_hops else [a.fft, a.fft // 2, a.fft // 4]
    print(f"\nimpl {impl_label}")

    sr_ref = 48000
    ref = band_limited_reference(a.seconds, sr_ref, sr_ref / a.fft)
    print(f"synthetic material {a.seconds} s @ {sr_ref} Hz, carriers snapped to "
          f"{sr_ref / a.fft:.2f} Hz bins, residual above {a.cutoff_khz} kHz")
    for hop in hops:
        try:
            data = IMPL[0](ref[:, None], sr_ref, a.fft, hop, a.max_partials)
            out = IMPL[1](data)[0]
        except Exception as exc:
            print(f"  hop {hop:5d}: not supported by this impl ({exc})")
            continue
        score(f"hop {hop:5d} (overlap {a.fft // hop})", ref, out, sr_ref, hop, a.cutoff_khz)

    if a.input:
        pcm, sr = read_wav_mono(Path(a.input), a.seconds)
        print(f"\n{Path(a.input).name} {pcm.shape[0]} samples @ {sr} Hz, "
              f"{pcm.shape[1]} ch")
        print("  above the cutoff this material has real signal, so the spike column reads as "
              "transient density rather than clicks")
        for hop in hops:
            data = IMPL[0](pcm, sr, a.fft, hop, a.max_partials)
            out = IMPL[1](data)[0]
            score(f"hop {hop:5d} (overlap {a.fft // hop})", pcm, out, sr, hop, a.cutoff_khz)


if __name__ == "__main__":
    main()
