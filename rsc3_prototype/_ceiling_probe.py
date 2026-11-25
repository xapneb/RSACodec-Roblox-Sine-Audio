"""
What the mask actually allows, band by band, above 9 kHz - and how far the shipped flat run word
sits under that.

The shape field is justified by one sentence: "a flat run is pinned by its tightest bin, so the
region decodes under what its own mask allows." That is either true or it is a story, and the only
way to tell is to print, on real frames, each dead band's own ceiling next to what the decoder
plays there, and see which end of the run is tight.
"""
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
sys.path.insert(0, HERE)
import rsc3_codec as r3
import rsc3_stream as st

N2 = 4096
W = 16
src = sys.argv[1] if len(sys.argv) > 1 else "cam15.wav"
fr = int(sys.argv[2]) if len(sys.argv) > 2 else 100
br = int(sys.argv[3]) if len(sys.argv) > 3 else 37

wf = wave.open(os.path.join(LAB, src), "rb")
sr = wf.getframerate()
raw = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768.0
wf.close()
A = r3.amplitude(r3.analyse(raw.reshape(-1, 2)[:, 0], N2)[0], N2)
nfr, nc = A.shape
nb = st.band_grid(nc, W)
A3 = A.reshape(nfr, nb, W)
thr = r3.threshold_dbfs(20 * np.log10(np.maximum(np.abs(A), 1e-30)), sr, nc,
                        psycho=True).reshape(nfr, nb, W)
cap = st.band_fill_cap(A3[fr], thr[fr])
b = np.clip(r3.allocate(A[fr:fr + 1], sr, br, psycho=True).reshape(nb, W).max(axis=1)
            .astype(np.int64), 0, 16)
hz = (np.arange(nc) * (sr / N2)).reshape(nb, W).mean(axis=1)
runs = st.dead_runs(b)
words = st.run_fill_words(cap, b, "safe")
Xf = st.reconstruct_bands(st.band_exponents(A3[fr]), b, np.zeros((nb, W), np.int64),
                          words, fr, W)
lvl = 10 * np.log10(np.maximum((Xf.reshape(nb, W) ** 2).mean(axis=-1), 1e-300))

print(f"{src} frame {fr}   dead bands above 9 kHz   ({int(b.size)} bands, "
      f"{int((b == 0).sum())} dead)")
print("  band   kHz   own ceiling   decoder plays   under by")
for r, (code, tilt) in zip(runs, words):
    if hz[r].max() < 9000:
        continue
    print(f"  run {r[0]:3d}-{r[-1]:3d}  {hz[r[0]] / 1e3:5.1f}-{hz[r[-1]] / 1e3:5.1f} kHz"
          f"   code {code:2d} level {st.fill_level(code):7.1f} tilt {st.FILL_TILT[tilt]:6.1f}"
          f"   tightest {int(np.argmin(cap[r]))}")
    for j in r:
        if hz[j] < 9000:
            continue
        print(f"  {j:5d} {hz[j] / 1e3:6.1f} {cap[j]:13.1f} {lvl[j]:14.1f}"
              f" {lvl[j] - cap[j]:10.1f}")

# and the aggregate, over many frames: how far under its own ceiling does the shipped word put
# each region. Only bands the word actually fills count here - a run that sends code 0 is silence,
# and fill_level() reports -1e9 for it, which averages to nonsense if you let it in.
print("\nshipped flat word, distance from each dead band's own ceiling, frames 20..70:")
print("   region      n    median gap   mean gap   % > 10 dB under")
for lo, hi in ((9e3, 12e3), (12e3, 14e3), (14e3, 18e3), (18e3, 20e3), (20e3, 24e3)):
    gaps, n = [], 0
    for f in range(20, min(70, nfr)):
        bf = np.clip(r3.allocate(A[f:f + 1], sr, br, psycho=True).reshape(nb, W).max(axis=1)
                     .astype(np.int64), 0, 16)
        capf = st.band_fill_cap(A3[f], thr[f])
        wfs = st.run_fill_words(capf, bf, "safe")
        lvf = np.full(nb, -1e9)
        for r, (code, tilt) in zip(st.dead_runs(bf), wfs):
            if not code:
                continue
            lo_, hi_ = float(r[0]), float(r[-1])
            half = max((hi_ - lo_) * 0.5, 1.0)
            lvf[r] = (st.fill_level(code)
                      + st.FILL_TILT[tilt] * (r.astype(np.float64) - (lo_ + hi_) * 0.5) / half)
        m = (hz >= lo) & (hz < hi) & (bf == 0) & (lvf > -1e8)
        if m.any():
            g = lvf[m] - capf[m]
            gaps.append(g)
            n += int(m.sum())
    g = np.concatenate(gaps)
    print(f"   {lo / 1e3:4.0f}-{hi / 1e3:4.0f} kHz {n:8d}   {np.median(g):9.1f}   {g.mean():8.1f}"
          f"   {float((g < -10).mean() * 100):8.0f}")
