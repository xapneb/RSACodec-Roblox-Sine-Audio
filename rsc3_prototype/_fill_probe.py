"""
Fill the dead bands, and say it cheaply: one level per run, plus a tilt across it.

What is wanted is not noise shaping, it is noise *fill*: wherever the allocation gave a band zero
bits the decoder produces silence, so a 10 kbit/s stream leaves a hole from roughly 7 kHz upward.
fill_experiment.py showed the trick is affordable, but it never put the level in a file, and it
drew noise from numpy's generator per band in iteration order, so no streaming decoder and no
Luau port could ever have rebuilt it.

The cheap part is geometric: dead bands arrive in runs, so one 6-bit word per run covers all of
them. The bad part is exactly what the README calls unsolved: a flat floor across a run that
spans a source roll-off lands high on the quiet members, low on the loud ones. Printed here, a
flat word puts 10-14 kHz and 14-20 kHz at *identical* density - the tell that one scalar is
covering 10 kHz of spectrum.

The repair is to let the run have a tilt, paid with 2 bits. It is priced in dB of total fall
ACROSS THE RUN rather than per band, and that is not an implementation detail: a run on this
material is 127 bands long, so a "half a dB per band" tilt would be a 64 dB ramp, and an earlier
draft of this file did exactly that and produced noise that measured 60 dB above the source.

The fit is per-run least squares on the run's own normalised position p, which puts p at -1 at
the bottom of the run and +1 at the top regardless of how many bands the run covers:

    level(p) = mean + slope * p,   SSE(s) = Syy - 2s*Spy + s^2*Spp

and the encoder picks whichever candidate slope in the alphabet minimises that, per run.

Everything is priced with its bits included. The noise is a hash of (frame, band, bin),
normalised by a value both sides compute themselves, so the decoder's side costs nothing but the
words. ES is expected to fall while HF density rises: matched-energy noise carries about twice
the error power of zeroing. This is a trade, and it is printed as one.
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
W = r3.BAND_WIDTH
HASHMASK = (1 << 32) - 1
FLOOR = -120.0
LVBITS = 6
# 2-bit tilt alphabet: dB of total fall across the run. Flat is 0, so a run that was already flat
# pays nothing for the option.
SLOPE = np.array([0.0, -10.0, -25.0, -50.0])


def hash_noise(n, frame, band):
    """mulberry32, vectorised: two integers in, n deterministic values out. No generator state
    for a decoder to fall out of step with, and short enough to port to Luau verbatim."""
    s = (int(frame) * 0x9E3779B9 + int(band) * 0x85EBCA6B) & HASHMASK
    u = np.arange(n, dtype=np.int64)
    x = (s + u * 0x9E3779B1) & HASHMASK
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & HASHMASK
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & HASHMASK
    x = x ^ (x >> 16)
    return (x.astype(np.float64) / 2 ** 31 - 1.0)


def noise_field(nfr, nb):
    z = np.empty((nfr, nb, W))
    for f in range(nfr):
        for j in range(nb):
            v = hash_noise(W, f, j)
            z[f, j] = v / float(np.sqrt((v * v).mean()))
    return z


def run_ids(dead):
    """run id per (frame, band), 1-based; 0 means "not dead", so a live band never reads a word."""
    ids = np.zeros(dead.shape, np.int64)
    start = dead.copy()
    start[:, 1:] = dead[:, 1:] & ~dead[:, :-1]
    ids[:, 1:] = np.cumsum(start[:, 1:], axis=1)
    ids[:, 0] = dead[:, 0].astype(np.int64)
    return ids


def to_code(db):
    step = -FLOOR / ((1 << LVBITS) - 1)
    return np.clip(np.round((np.maximum(db, FLOOR) - FLOOR) / step), 0,
                   (1 << LVBITS) - 1).astype(np.int64)


def from_code(code):
    step = -FLOOR / ((1 << LVBITS) - 1)
    return np.where(code > 0, FLOOR + code * step, -400.0)


def run_positions(ids, dead, jidx, nrunmax):
    """Normalised position within the run: -1 at its first band, +1 at its last. This is what
    makes the tilt independent of the run's length, and runs here vary from 1 to 127 bands."""
    nfr, nb = ids.shape
    rows = np.repeat(np.arange(nfr)[:, None], nb, axis=1)
    lo = np.full((nfr, nrunmax), np.inf)
    hi = np.full((nfr, nrunmax), -np.inf)
    nd = np.where(dead, True, False)
    np.minimum.at(lo, (rows[nd], ids[nd]), jidx[nd])
    np.maximum.at(hi, (rows[nd], ids[nd]), jidx[nd])
    mid = np.where(np.isfinite(lo), (lo + hi) * 0.5, 0.0)
    half = np.where(np.isfinite(hi), np.maximum((hi - lo) * 0.5, 1.0), 1.0)
    p = (jidx - np.take_along_axis(mid, ids, 1)) / np.take_along_axis(half, ids, 1)
    return np.clip(p, -1.0, 1.0)


for name in ("cam15.wav", "ij15.wav"):
    wf = wave.open(os.path.join(LAB, name), "rb")
    sr = wf.getframerate()
    x = (np.frombuffer(wf.readframes(int(15 * sr)), np.int16).astype(np.float64)
         / 32768.0).reshape(-1, 2)
    print(f"\n{name}   {x.shape[0]} samples  {sr} Hz")
    A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
    nfr, nc = A.shape
    nb = st.band_grid(nc, W)
    A3 = A.reshape(nfr, nb, W)
    hz = np.arange(nc) * sr / N2
    grp = ((0, 5e3), (5e3, 10e3), (10e3, 14e3), (14e3, 20e3))
    grp_b = [((hz >= lo) & (hz < hi)).reshape(nb, W) for lo, hi in grp]
    z = noise_field(nfr, nb)
    print(f"  {'budget':>6} {'arm':>7} {'fill b/f':>9} {'ES':>7}"
          + "".join(f" {f'{lo/1e3:.0f}-{hi/1e3:.0f}k':>9}" for lo, hi in grp))
    for budget in (16, 40, 256):
        bits = np.clip(r3.allocate(A, sr, budget, psycho=True).reshape(nfr, nb, W)
                       .max(axis=2).astype(np.int64), 0, st.MAX_BITS)
        E = np.array([st.band_exponents(A3[f]) for f in range(nfr)])
        M = np.array([st.quantise_bands(A3[f], E[f], bits[f]) for f in range(nfr)])
        dead = bits == 0
        want = 10 * np.log10(np.maximum((A3 ** 2).mean(axis=2), 10 ** FLOOR))
        ids = run_ids(dead)
        nrunmax = int(ids.max()) + 1
        nruns = float(sum(len(np.unique(r[r > 0])) for r in ids)) / nfr
        step = np.where(bits > 0, np.power(2.0, (E - np.maximum(bits, 1)).astype(float)), 1.0)
        base = M * step[:, :, None]
        jidx = np.broadcast_to(np.arange(nb)[None, :], (nfr, nb))
        rows = np.repeat(np.arange(nfr)[:, None], nb, axis=1)
        nd = np.where(dead, 1.0, 0.0)
        # run statistics, in the run's own normalised coordinate
        p = run_positions(ids, dead, jidx, nrunmax)
        n = np.zeros((nfr, nrunmax))
        sy = np.zeros((nfr, nrunmax))
        sp = np.zeros((nfr, nrunmax))
        spp = np.zeros((nfr, nrunmax))
        spy = np.zeros((nfr, nrunmax))
        for acc, val in ((n, nd), (sy, nd * want), (sp, nd * p), (spp, nd * p * p),
                         (spy, nd * p * want)):
            np.add.at(acc, (rows, ids), val)
        cnt = np.maximum(n, 1.0)
        mean = sy / cnt
        code = to_code(mean)
        code[n <= 0] = 0
        # the run's own residual about its mean, to price each candidate slope
        syy = None
        yy = np.zeros((nfr, nrunmax))
        np.add.at(yy, (rows, ids), nd * want * want)
        syy = yy - cnt * mean ** 2
        sse = syy[:, None, :] - 2.0 * SLOPE[None, :, None] * (spy[:, None, :]
                                                             - mean[:, None, :] * sp[:, None, :])
        sse = sse + (SLOPE ** 2)[None, :, None] * (spp[:, None, :] - sp[:, None, :] ** 2 / cnt[:, None, :])
        pick = np.argmin(np.where(n[:, None, :] > 0, sse, 1e30), axis=1)
        best_slope = SLOPE[pick]
        print(f"    runs/frame {nruns:5.2f}   longest {int(n.max()):3d} bands   "
              f"tilt picks flat/-10/-25/-50 dB {np.bincount(pick[n > 0], minlength=4)}")
        for arm, bill in (("none", 0.0), ("flat", nruns * LVBITS),
                          ("tilt", nruns * (LVBITS + 2))):
            if arm == "none":
                X = base
            else:
                lvl = np.take_along_axis(from_code(code), ids, 1)
                if arm == "tilt":
                    lvl = lvl + np.take_along_axis(best_slope, ids, 1) * p
                lvl = np.where(dead & (lvl > -100), lvl, -400.0)
                X = np.where(lvl[:, :, None] > -100,
                             z * np.power(10.0, 0.05 * lvl)[:, :, None], base)
            y = r3.synthesise(X.reshape(nfr, nc) * (N2 / 4.0), N2, x.shape[0])
            lo_, hi_ = 3 * N2, x.shape[0] - N2
            y1 = y if y.ndim == 1 else y[:, 0]
            g = (float(np.dot(x[lo_:hi_, 0], y1[lo_:hi_]))
                 / max(float(np.dot(y1[lo_:hi_], y1[lo_:hi_])), 1e-300))
            es = 10 * np.log10(float(np.mean(x[lo_:hi_, 0] ** 2))
                               / max(float(np.mean((x[lo_:hi_, 0] - g * y1[lo_:hi_]) ** 2)), 1e-300))
            dens = [10 * np.log10(max(float((X[:, mm] ** 2).mean()), 1e-300)) for mm in grp_b]
            print(f"  {budget:6d} {arm:>7} {bill:9.1f} {es:7.2f}"
                  + "".join(f" {d:9.1f}" for d in dens))
    src = [10 * np.log10(max(float((A3[:, mm] ** 2).mean()), 1e-300)) for mm in grp_b]
    print(f"  {'':>6} {'source':>7} {'':>9} {'':>7}" + "".join(f" {d:9.1f}" for d in src))
