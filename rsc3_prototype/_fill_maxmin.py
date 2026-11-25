"""
Fill the dead bands at the loudest level that is still masked, by construction.

The audit before this put the fill in front of the ruler that matters and the answer was not
good: with a flat floor per run, about 1 in 5 filled bands sits ABOVE its own masking threshold,
and the worst overshoot is 27 dB. That is not a bug, it is the shape of the problem. A band that
was given zero bits is a band whose bins were quiet, and a quiet bin hides nothing. So "fill it
at the band's own energy" is always louder than what that band can hide, and no amount of
level-tuning fixes that - the geometry has to change.

What does fix it is a max-min, and it costs nothing extra to state:

    L = largest value with  L + tilt * p[j] <= mask[j]   for every bin j in the run

where p is the bin's position in the run, -1 to +1. The transmitted level then IS the binding
constraint, so overshoot is impossible rather than rare. The tilt is what gives the fill its low
end back: the mask of a run of dead bands almost always falls with frequency, so a flat floor is
held down by its highest bin, while a falling floor can be loud where something hides it and
quiet where nothing does.

It is all transmittable, which is the bar the fill has to clear to be a codec rather than a
demo. The level word is quantised DOWN, never to nearest, so rounding cannot make it illegal.
The tilt is 2 bits. A decoder needs neither the mask nor the source - b gives it the runs, 8
bits per run give it the envelope, and a hash of (frame, band) gives it the values.

What it costs is the honest part. Masked fill is quieter than the loud obvious fill that made
this trick sound impressive to begin with, so the HF density gain shrinks and ES still dips,
because noise in place of silence is still error power. If that is not worth the bits, the right
answer is --no-fill, and this file is the evidence for saying so either way.
"""
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.join(os.path.dirname(HERE), "rsc2_testing_lab"))
sys.path.insert(0, HERE)
import rsc3_codec as r3
import rsc3_stream as st

N2 = 4096
W = r3.BAND_WIDTH
# 2-bit tilt alphabet: dB of total fall ACROSS the run, not per band. A run here is routinely
# 100 bands wide, so "half a dB per band" would be a 50 dB ramp.
TILT = np.array([0.0, -12.0, -28.0, -60.0])
TILT_BITS = 2


def run_fill(A3f, thr_f, b):
    """One (level code, tilt index) pair per dead run, legal at every bin in the run."""
    src = 10.0 * np.log10(np.maximum((A3f ** 2).mean(axis=1), 10 ** st.FILL_FLOOR))
    cap = np.minimum(src, np.maximum(thr_f.min(axis=-1), st.FILL_FLOOR))
    out = []
    for r in st.dead_runs(b):
        j = np.arange(int(r[0]), int(r[-1]) + 1, dtype=np.float64)
        mid = (j[0] + j[-1]) * 0.5
        half = max((j[-1] - j[0]) * 0.5, 1.0)
        p = (j - mid) / half
        best_lv, best_q = st.FILL_FLOOR, 0
        for q, t in enumerate(TILT):
            lv = float(np.min(cap[r] - t * p))
            if lv > best_lv:
                best_lv, best_q = lv, q
        code = st.fill_code(best_lv)
        # round DOWN, so the level the decoder reads can never be illegal
        while code and st.fill_level(code) > best_lv + 1e-9:
            code -= 1
        out.append((code, best_q, st.fill_level(code) if code else None, mid, half))
    return out


def apply_fill(X, runs, got, frame, width):
    """Exactly what a decoder would do with these words, so nothing here is encoder-side magic."""
    for r, (code, q, legal, mid, half) in zip(runs, got):
        if legal is None:
            continue
        lvl = legal + TILT[q] * (r - mid) / half
        amp = 10.0 ** (0.05 * lvl)
        for jj, j in enumerate(r):
            X[int(j)] = st.fill_noise(int(j), int(frame), width) * amp[jj]
    return X


for name in ("cam15.wav", "ij15.wav"):
    wf = wave.open(os.path.join(LAB, name), "rb")
    sr = wf.getframerate()
    x = (np.frombuffer(wf.readframes(int(15 * sr)), np.int16).astype(np.float64)
         / 32768.0).reshape(-1, 2)
    A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
    nfr, nc = A.shape
    nb = st.band_grid(nc, W)
    A3 = A.reshape(nfr, nb, W)
    thr = r3.threshold_dbfs(20 * np.log10(np.maximum(np.abs(A3.reshape(nfr, -1)), 1e-300)),
                            sr, nc, psycho=True).reshape(nfr, nb, W)
    hz = np.arange(nc) * sr / N2
    grp = ((0, 5e3), (5e3, 10e3), (10e3, 14e3), (14e3, 20e3))
    grp_b = [((hz >= lo) & (hz < hi)).reshape(nb, W) for lo, hi in grp]
    Eall = np.array([st.band_exponents(A3[f]) for f in range(nfr)])
    print(f"\n{name}   mask mean {float(thr.mean()):.1f} dBFS   quiet bins {float(thr.min()):.0f}")
    print(f"  {'budget':>6} {'mode':>7} {'kbit/s':>8} {'ES':>7} {'fill b/f':>9} "
          f"{'over mask':>11} {'worst dB':>9} {'tilts':>18}   "
          + "".join(f"{f'{lo/1e3:.0f}-{hi/1e3:.0f}k':>8}" for lo, hi in grp))
    for budget in (16, 40, 256):
        bits = np.clip(r3.allocate(A, sr, budget, psycho=True).reshape(nfr, nb, W)
                       .max(axis=2).astype(np.int64), 0, st.MAX_BITS)
        Mall = np.array([st.quantise_bands(A3[f], Eall[f], bits[f]) for f in range(nfr)])
        base = st.frame_costs(Eall, bits, Mall, W)
        for mode in ("none", "maxmin"):
            X = (Mall * np.array([st.band_steps(Eall[f], bits[f]) for f in range(nfr)])[:, :, None])
            L = np.full((nfr, nb), -400.0)
            nbits = 0
            tilts = []
            for f in range(nfr):
                if mode == "none":
                    continue
                got = run_fill(A3[f], thr[f], bits[f])
                runs = st.dead_runs(bits[f])
                # every bit the fill can cost, charged here: the flag and both words per run
                nbits += 1 + (st.FILL_BITS + TILT_BITS) * len(got) if any(g[0] for g in got) \
                    else 1
                tilts += [int(g[1]) for g in got if g[2] is not None]
                for (code, q, legal, mid, half), r in zip(got, runs):
                    if legal is None:
                        continue
                    L[f, r] = legal + TILT[q] * (r - mid) / half
                apply_fill(X[f], runs, got, f, W)
            nbits = int(base.sum()) + nbits
            y = r3.synthesise(X.reshape(nfr, nb * W) * (N2 / 4.0), N2, x.shape[0])
            lo_, hi_ = 3 * N2, x.shape[0] - N2
            g = (float(np.dot(x[lo_:hi_, 0], y[lo_:hi_]))
                 / max(float(np.dot(y[lo_:hi_], y[lo_:hi_])), 1e-300))
            es = 10 * np.log10(float(np.mean(x[lo_:hi_, 0] ** 2))
                               / max(float(np.mean((x[lo_:hi_, 0] - g * y[lo_:hi_]) ** 2)), 1e-300))
            dead = bits == 0
            m = dead & (L > -100)
            exc = np.where(m[:, :, None], L[:, :, None] - thr, 0.0)
            nbad = int(((exc.max(axis=2) > 0.01) & m).sum())
            print(f"  {budget:6d} {mode:>7} {float(nbits) / nfr * (sr / (N2 / 2)) / 1000.0:8.2f}"
                  f" {es:7.2f} {float(nbits) / nfr:9.1f}   {nbad:>6}/{int(m.sum()):>6}"
                  f" {float(exc.max()) if m.any() else 0.0:9.1f}   "
                  f"{str([int(v) for v in np.bincount(np.array(tilts, np.int64), minlength=4)]) if tilts else '-':>16}"
                  + "".join(f"{10 * np.log10(max(float((X[:, mm] ** 2).mean()), 1e-300)):8.1f}"
                            for mm in grp_b))
    src = [10 * np.log10(max(float((A3[:, mm] ** 2).mean()), 1e-300)) for mm in grp_b]
    print(f"  {'':>6} {'src':>7} {'':>8} {'':>7} {'':>9}   {'-':>11} {'-':>9} {'':>18}   "
          + "".join(f"{d:8.1f}" for d in src))
