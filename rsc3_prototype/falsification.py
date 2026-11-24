"""
Falsification pass. Any "big win over the shipping codec" is a reason to look for
the bug, not to celebrate, so this measures the things an ES number hides:

* is the output the signal, or just quiet? (energy ratio, not error ratio)
* is the ES window honest? (skip the un-overlapped first/last blocks)
* does a single partial survive, and do transients?
* is the un-coded transform the bound it claims to be?

Expected readings for this codec, kept here so a regression is obvious:
    audio.wav   psycho  512 / 256 / 128 b/f  ->  12.95 / 12.77 / 10.97 dB
    1 kHz tone  budget 400                  ->  14.52 dB
    click train budget 256                  ->  19.06 dB
    un-coded transform                      ->  ~307 dB  (the sanity bound)
If a number here moves but the ES elsewhere doesn't, trust this one.
"""
import sys, wave, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
import numpy as np
import rsc3_codec as r3

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LAB, "audio.wav")
with wave.open(path, "rb") as wf:
    sr, nch = wf.getframerate(), wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
pcm = np.frombuffer(raw, np.int16).astype(np.float64)
pcm = (pcm.reshape(-1, nch) if nch > 1 else pcm[:, None])[:, 0] / 32768.0
N2, hop, inv = 4096, 2048, 1024.0


def report(name, x, y, lo, hi):
    e_in = (x[lo:hi] ** 2).mean()
    e_out = (y[lo:hi] ** 2).mean()
    e_err = ((y[lo:hi] - x[lo:hi]) ** 2).mean()
    print(f"{name:26s} ES {10 * np.log10(e_in / e_err):6.2f} dB   "
          f"output/input energy {10 * np.log10(e_out / e_in):+6.2f} dB   "
          f"rms {np.sqrt(e_out):.4f} vs {np.sqrt(e_in):.4f}")


print("== 1 kHz tone, full scale, 4 s ==")
t = np.arange(4 * sr) / sr
tone = 0.9 * np.sin(2 * np.pi * 1000.0 * t)
X, _ = r3.analyse(tone, N2)
A = r3.amplitude(X, N2)
bits = r3.allocate(A, sr, 400, psycho=True)
y = r3.synthesise(r3.quantise(A, bits) * inv, N2, len(tone))
report("   coded", tone, y, 2 * N2, len(tone) - N2)
report("   un-coded transform", tone, r3.synthesise(X, N2, len(tone)), 2 * N2, len(tone) - N2)
print(f"   bits/frame {int(bits.sum() / X.shape[0])}  nonzero coeffs "
      f"{float((bits[5:X.shape[0] - 5] > 0).mean()) * 100:.2f}%")

print(f"\n== {os.path.basename(path)} (channel 0) ==")
X, _ = r3.analyse(pcm, N2)
A = r3.amplitude(X, N2)
for budget in (512, 256, 128):
    bits = r3.allocate(A, sr, budget, psycho=True)
    y = r3.synthesise(r3.quantise(A, bits) * inv, N2, len(pcm))
    report(f"   psycho budget {budget}", pcm, y, 3 * N2, len(pcm) - N2)
    k = min(500, X.shape[0] - 1) * hop
    ref = np.abs(np.fft.rfft(pcm[k:k + N2] * r3.window(N2)))
    got = np.abs(np.fft.rfft(y[k:k + N2] * r3.window(N2)))
    top = np.argsort(-ref)[:5]
    print("      loudest bins (Hz, ref->coded dBFS):  "
          + "  ".join(f"{int(b * sr / (2 * N2))}Hz {20 * np.log10(max(ref[b], 1e-12)):5.1f}"
                      f"->{20 * np.log10(max(got[b], 1e-12)):5.1f}" for b in top))

print("\n== click train ==")
clicks = np.zeros(3 * sr)
for i in range(150):
    p = int(i * sr / 150)
    clicks[p:p + 160] += np.hanning(160) * 0.9
X, _ = r3.analyse(clicks, N2)
A = r3.amplitude(X, N2)
y = r3.synthesise(r3.quantise(A, r3.allocate(A, sr, 256, psycho=True)) * inv, N2, len(clicks))
report("   coded clicks", clicks, y, 3 * N2, len(clicks) - N2)
print(f"      max |diff| {np.abs(y - clicks).max():.4f}   near an onset "
      f"{np.abs(y[hop:2 * hop] - clicks[hop:2 * hop]).max():.4f}")
