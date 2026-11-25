"""
What does it cost to buy 20 kHz, per kilobit?

At 10 kbit/s the spectrum above about 14 kHz is not quiet, it is ABSENT - the waterfill runs out
of bits and everything higher decodes to zero, and the fix shipped so far replaces that hole with
a noise floor that sits under the mask. Honest, inaudible, and not the audio.

The vector-code idea is that the grammar pays for high-frequency bins in the wrong unit. To put
one sine at 17 kHz you must revive its band (count field), point at the bin (gap), give it a sign
and a magnitude - roughly 10 bits to say "one sine, over there". A code that enumerates the whole
region at once prices it once per HIT and shares the position cost across all of them:

    hits everywhere in an L-slot region:      log2 C(L, R) + R (1 + m)
    K pulses spread over the same L slots:    log2 N(L, K),   N = sum_R 2^R C(L,R) C(K-1,R-1)

Both are exact integer counts evaluated here, not approximations, because rate has to be
arithmetic for this to mean anything. With L = 1024 slots (10-24 kHz at this block size), the
price per hit comes out near 6-7 bits, against ~10 for the per-bin way.

Everything else is held equal so the comparison is only about who pays:

  * the region is masked out of the scalar waterfill, so the bits this buys are new bits and not
    the same bits spent twice;
  * the low part of the spectrum is identical in every arm;
  * reconstruction is the encoder's own - scalar steps, then placed hits, then the masked fill on
    whatever stayed dead, all through reconstruct_bands;
  * the side info the scheme needs (a region scale, and for the pyramid a pulse count) is charged
    with the container's own Rice machinery, not waved away.

The control variable is how many hits you buy, not the bitrate, because that is the question
being asked: at a given rate, how much *real* content above 10 kHz is purchasable, and does it
beat a masked floor on the two rulers that decide quality.
"""
import math
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
FRATE = 48000 / (N2 / 2)
GROUPS = ((0, 5e3), (5e3, 10e3), (10e3, 14e3), (14e3, 18e3), (18e3, 24e3))
_NC = {}


def log2_N(M, K):
    if K <= 0:
        return 0.0
    key = (M, K)
    if key not in _NC:
        _NC[key] = sum((1 << r) * math.comb(M, r) * math.comb(K - 1, r - 1)
                       for r in range(1, min(M, K) + 1))
    n = _NC[key]
    return math.log2(n) if n > 1 else 0.0


def log2_C(M, R):
    if R <= 0 or R > M:
        return 0.0
    key = ("C", M, R)
    if key not in _NC:
        _NC[key] = math.comb(M, R)
    n = _NC[key]
    return math.log2(n) if n > 1 else 0.0


def read_wav(p, secs=15.0):
    wf = wave.open(p, "rb")
    sr = wf.getframerate()
    d = wf.readframes(min(wf.getnframes(), int(secs * sr)))
    wf.close()
    return np.frombuffer(d, np.int16).astype(np.float64).reshape(-1, 2) / 32768.0, sr


def rice_bits(prev, cur):
    """A delta stream at the price the container pays for its other deltas."""
    z = st.zz_vec(np.asarray(cur, np.int64) - np.asarray(prev, np.int64))
    k = st.pick_k(z)
    _, wd = st.rice_fields(z, k)
    return 5 + int(wd.sum())


for name in ("cam15.wav", "ij15.wav"):
    x, sr = read_wav(os.path.join(LAB, name))
    A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
    nfr, nc = A.shape
    nb = st.band_grid(nc, W)
    A3 = A.reshape(nfr, nb, W)
    Ad = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
    thr = r3.threshold_dbfs(Ad, sr, nc, psycho=True).reshape(nfr, nb, W)
    Eall = np.array([st.band_exponents(A3[f]) for f in range(nfr)])
    hz = np.arange(nc) * sr / N2
    band_hz = hz.reshape(nb, W).mean(axis=1)
    print(f"\n{name}   {nfr} frames  {nb} bands")
    print(f"  {'region':>8} {'arm':>8} {'buy':>6}   {'kbit/s':>7} {'ES':>6}   "
          f"{'hits/f':>7} {'bits/f':>7}   " + " ".join(f"{f'{a/1e3:.0f}-{b/1e3:.0f}':>7}"
                                                       for a, b in GROUPS) + "   source")
    for f0 in (10e3, 14e3):
        reg = band_hz >= f0
        L = int(reg.sum()) * W
        # region masked out of the waterfill: a high bin can no longer steal bits from the frame
        thr2 = thr.copy()
        thr2[:, reg] = 991.0
        bl2 = np.maximum(r3._band_level_db(Ad, r3.band_edges(nc)), thr2.reshape(nfr, nc))
        for arm, grid in (("hitmap", (4, 8, 16, 32, 64, 128, 256, 512)),
                          ("pyramid", (4, 8, 16, 32, 64, 128, 256, 512))):
            for buy in grid:
                # the scalar part is fixed at one budget so every point in this row differs from
                # every other only in what the region code cost
                SB = 32
                bits = np.clip(r3.waterfill_select(Ad, thr2.reshape(nfr, nc), bl2, SB)
                               .reshape(nfr, nb, W).max(axis=2), 0, st.MAX_BITS).astype(np.int64)
                prev_E = np.zeros(nb, np.int64)
                prev_b = np.zeros(nb, np.int64)
                prev_Er = np.zeros(nfr, np.int64)
                prev_K = np.zeros(nfr, np.int64)
                X = np.zeros((nfr, nb, W))
                tot = 0
                nhit = 0
                for f in range(nfr):
                    E = Eall[f]
                    mant = st.quantise_bands(A3[f], E, bits[f])
                    Xf = st.band_steps(E, bits[f])[:, None] * mant.astype(np.float64)
                    nf = 0
                    # --- the low part, priced by the container's own accounting
                    live = bits[f] > 0
                    if live.any():
                        nf += rice_bits(prev_E[live], E[live])
                    nf += rice_bits(prev_b, bits[f])
                    nz = mant != 0
                    gb, cb = st.gap_bits(nb), st.count_bits(nb)
                    n_cnt = nz.sum(axis=1).astype(np.int64)
                    b1 = np.maximum(bits[f], 1)
                    cg = cb + n_cnt * (gb + 1 + b1)
                    cf = W * (1 + b1)
                    nf += int((1 + np.where(live, np.minimum(cf, cg), 0)).sum())
                    # --- the region, bought as content
                    rev = np.zeros(nb, bool)
                    if reg.any():
                        pk = float(np.abs(A3[f][reg]).max())
                        Er = int(math.ceil(math.log2(max(pk, 2.0 ** -60))))
                        nf += rice_bits(prev_Er, np.array([Er]))
                        prev_Er = np.array([Er])
                        aud = (np.abs(A3[f]) > 10 ** (0.05 * thr[f])) & reg[:, None]
                        idx = np.flatnonzero(aud.reshape(-1))
                        mag = np.abs(A3[f].reshape(-1)[idx])
                        if idx.size > buy:
                            idx = idx[np.argsort(-mag)[:buy]]
                        mag = np.abs(A3[f].reshape(-1)[idx])
                        if idx.size:
                            if arm == "hitmap":
                                s = 2.0 ** (Er - 2)
                                q = np.clip(np.round(mag / s), 1, 3)
                                nf += int(log2_C(L, idx.size)) + idx.size * 3
                            else:
                                # a pyramid vector cannot have fewer pulses than nonzero slots,
                                # so buying more positions than pulses is not a thing you can
                                # encode. R is the floor on K, and it is what the codebook says.
                                K = max(int(buy), idx.size)
                                p = max(int(math.ceil(math.log2(K))), 1)
                                s = 2.0 ** (Er - p)
                                q = np.maximum(np.round(mag / s).astype(np.int64), 1)
                                while int(q.sum()) > K:
                                    j = int(np.argmax(q))
                                    if q[j] <= 1:
                                        break
                                    q[j] -= 1
                                while int(q.sum()) < K:
                                    q[int(np.argmax(np.abs(mag) * q + 1e-30))] += 1
                                nf += int(log2_N(L, int(q.sum()))) + rice_bits(prev_K,
                                                                               np.array([int(q.sum())]))
                                prev_K = np.array([int(q.sum())])
                            Xf.reshape(-1)[idx] = np.sign(A3[f].reshape(-1)[idx]) * q * s
                            nhit += idx.size
                            rev = np.zeros(nb, bool)
                            rev[idx // W] = True
                    # --- fill, on dead bands that are not in the region and hold no hits
                    dead = (bits[f] == 0) & ~rev
                    runs = st.dead_runs(dead.astype(np.int64))
                    if runs:
                        lvl = 10 * np.log10(np.maximum((A3[f] ** 2).mean(axis=1), 1e-30))
                        cap = np.minimum(lvl, np.maximum(thr[f].min(axis=1), st.FILL_FLOOR))
                        any_ = False
                        for r in runs:
                            code = st.fill_code(float(np.min(cap[r])))
                            if code:
                                any_ = True
                                mid = (r[0] + r[-1]) * 0.5
                                half = max((r[-1] - r[0]) * 0.5, 1.0)
                                for j in r:
                                    Xf[j] = (st.fill_noise(int(j), f, W)
                                             * 10 ** (0.05 * st.fill_level(code)))
                        nf += 1 + (8 * len(runs) if any_ else 0)
                    X[f] = Xf
                    prev_E = E
                    prev_b = bits[f]
                    tot += nf
                y = r3.synthesise(X.reshape(nfr, nc) * (N2 / 4.0), N2, x.shape[0])
                lo, hi = 3 * N2, x.shape[0] - N2
                g = (float(np.dot(x[lo:hi, 0], y[lo:hi]))
                     / max(float(np.dot(y[lo:hi], y[lo:hi])), 1e-300))
                es = 10 * math.log10(float(np.mean(x[lo:hi, 0] ** 2))
                                     / max(float(np.mean((x[lo:hi, 0] - g * y[lo:hi]) ** 2)),
                                           1e-300))
                dens = [10 * math.log10(max(float((X.reshape(nfr, nc) ** 2)[:, (hz >= a)
                                                     & (hz < b)].mean()), 1e-300))
                        for a, b in GROUPS]
                print(f"  {f0 / 1e3:6.0f}k {arm:>8} {buy:>6}   {tot / nfr * FRATE / 1000.0:7.2f}"
                      f" {es:6.2f}   {nhit / nfr:7.1f} {tot / nfr:7.0f}   "
                      + " ".join(f"{d:7.1f}" for d in dens))
        dens = [10 * math.log10(max(float((A.reshape(nfr, nc) ** 2)[:, (hz >= a) & (hz < b)]
                                     .mean()), 1e-300)) for a, b in GROUPS]
        print(f"  {'':>8} {'source':>8} {'':>6}   {'':>7} {'':>6}   {'':>7} {'':>7}   "
              + " ".join(f"{d:7.1f}" for d in dens))
