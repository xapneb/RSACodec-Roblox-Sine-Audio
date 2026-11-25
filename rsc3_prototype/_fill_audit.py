"""
Fill the dead bands, but not louder than the band that has to hide it.

The first version of this filled each run of dead bands with noise at the run's own mean energy,
and an ES test said it cost about 1.3 dB. That number was useless: error power always prefers
silence to matched noise, so ES cannot tell a masked floor from an audible one. The ruler that can
is the original's masking threshold, measured bin by bin, and when this codec was checked against
it the answer was not encouraging. Measured on cam15 at 10 kbit/s, 24% of the bands the fill
touches land ABOVE their own mask, and the worst overshoot is 45 dB. That is not air, that is a
band of noise louder than the signal that was supposed to be hiding it, and the reason is the
mean: a run's mean is set by its loudest member, so the quiet members get a floor tens of dB over
what they can hide.

The fix is a cap, and it is not expensive, because the cap is computed from data the encoder
already has:

    level per band  =  min(source energy of that band, the quietest mask in that band)

A flat floor across a band is only masked if its LOWEST bin can hide it, so the band's share of
the cap is the minimum threshold over its W bins, not the sum and not the mean. An earlier draft
of this experiment used the summed masking capacity and let the quiet bins through, which is
exactly the mistake above, written differently.

Then the run level is the mean of the capped band levels, so a quiet band pulls the run down
instead of a loud band pushing it up. The cost is the same 6 bits per run, because the cap is not
transmitted - only the level that survived it is.
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


def run_levels(A3f, b, thr, mode):
    """One code per dead run, under one of the three policies this arm is testing."""
    src = 10.0 * np.log10(np.maximum((A3f ** 2).mean(axis=1), 10 ** st.FILL_FLOOR))
    if mode == "mean":
        want = src
    elif mode == "cap":
        # the band can only hide as much as its WEAKEST bin, because the floor is flat
        ceil = thr.min(axis=-1)
        want = np.minimum(src, np.maximum(ceil, st.FILL_FLOOR))
    else:
        return None
    return [st.fill_code(float(np.mean(want[r]))) for r in st.dead_runs(b)]


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
    Mall = np.array([st.quantise_bands(A3[f], Eall[f], np.clip(
        r3.allocate(A, sr, 40, psycho=True).reshape(nfr, nb, W).max(axis=2).astype(np.int64),
        0, st.MAX_BITS)[f]) for f in range(nfr)])
    print(f"\n{name}   mask mean {float(thr.mean()):.1f} dBFS   "
          f"quietest bin {float(thr.min()):.1f} dBFS")
    print(f"  {'budget':>6} {'mode':>5} {'kbit/s':>8} {'ES':>7} {'fill':>6} "
          f"{'bands over mask':>16} {'worst dB':>9} {'energy over':>12}   "
          + "".join(f"{f'{lo/1e3:.0f}-{hi/1e3:.0f}k':>8}" for lo, hi in grp))
    for budget in (16, 40, 256):
        bits = np.clip(r3.allocate(A, sr, budget, psycho=True).reshape(nfr, nb, W)
                       .max(axis=2).astype(np.int64), 0, st.MAX_BITS)
        Mall = np.array([st.quantise_bands(A3[f], Eall[f], bits[f]) for f in range(nfr)])
        for mode in ("none", "mean", "cap"):
            X = np.zeros((nfr, nb * W))
            L = np.full((nfr, nb), -400.0)
            nf = 0
            for f in range(nfr):
                codes = run_levels(A3[f], bits[f], thr[f], mode)
                if codes is not None:
                    nf += sum(st.fill_cost(bits[f], codes) for _ in [0])
                    for r_i, c in enumerate(st.dead_runs(bits[f])):
                        L[f, c] = st.fill_level(codes[r_i])
                X[f] = st.reconstruct_bands(Eall[f], bits[f], Mall[f], codes, f, W).reshape(-1)
            y = r3.synthesise(X * (N2 / 4.0), N2, x.shape[0])
            lo_, hi_ = 3 * N2, x.shape[0] - N2
            g = (float(np.dot(x[lo_:hi_, 0], y[lo_:hi_]))
                 / max(float(np.dot(y[lo_:hi_], y[lo_:hi_])), 1e-300))
            es = 10 * np.log10(float(np.mean(x[lo_:hi_, 0] ** 2))
                               / max(float(np.mean((x[lo_:hi_, 0] - g * y[lo_:hi_]) ** 2)), 1e-300))
            cost = st.frame_costs(Eall, bits, Mall, W, fill_levels=None if mode == "none" else
                                  [run_levels(A3[f], bits[f], thr[f], mode) for f in range(nfr)])
            dead = bits == 0
            over = np.where(dead[:, :, None] & ((L[:, :, None] - thr) > 0), L[:, :, None] - thr, 0.0)
            nfill = int(dead.sum())
            nover = int(((over.max(axis=2) > 0) & dead).sum())
            # how much of the noise a listener would actually notice: the share of the FILLED
            # energy that sits above the mask. Restricted to filled bands, or the live bands
            # (L = -400, power ~1e-12) vanish from the denominator and the fraction reads huge.
            filled = dead & (L > -100)
            e_pow = np.where(L > -100, 10 ** (0.1 * L), 0.0)
            above = np.where(L[:, :, None] > thr, e_pow[:, :, None], 0.0)
            frac = float(above.sum()) / max(float(e_pow.sum() * thr.shape[2]), 1e-300)
            print(f"  {budget:6d} {mode:>5} {float(cost.sum()) / nfr * (sr / (N2 / 2)) / 1000.0:8.2f}"
                  f" {es:7.2f} {nf / nfr:6.1f}"
                  f"   {nover:6d}/{nfill:6d} {nover / max(nfill, 1) * 100:5.1f}%"
                  f" {float(over.max()):9.1f} {frac * 100:11.1f}%   "
                  + "".join(f"{10 * np.log10(max(float((X.reshape(nfr, nb, W)[:, m] ** 2).mean()), 1e-300)):8.1f}"
                            for m in grp_b))
    src = [10 * np.log10(max(float((A3[:, m] ** 2).mean()), 1e-300)) for m in grp_b]
    print(f"  {'':>6} {'src':>5} {'':>8} {'':>7} {'':>6}   {'-':>15} {'-':>9} {'-':>12}   "
          + "".join(f"{d:8.1f}" for d in src))
