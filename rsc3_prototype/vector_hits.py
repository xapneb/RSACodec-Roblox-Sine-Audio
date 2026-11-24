"""
Can a vector code buy 20 kHz where the scalar codec buys silence?

The shipping codec at ~10 kbit/s stops being real audio somewhere near 14 kHz: the waterfill
runs out of bits, those bands decode dead, and the fill replaces them with a masked noise floor.
Honest, inaudible, and not the source. The user's claim is that the bits are there to spend, and
that PVQ is the way to spend them where the scalar grammar charges too much.

What the scalar grammar actually charges for one sine at 17 kHz:

  revive its band  ~5-8 bits   the count field says this band is alive this frame
  point at the bin 8 bits      gap field, gap_bits(16)
  say what it is   1 + b       sign and magnitude

so between about 15 and 20 bits, most of it *about the position*, and that price is paid per
band even when the band exists only to hold one sine. A vector code over the whole region prices
the same event once and shares the position cost:

  R hits anywhere in L slots:      log2 C(L, R) + R (1 + m)
  K pulses spread over L slots:    log2 N(L, K),  N = sum_R 2^R C(L,R) C(K-1,R-1)

Both are exact integer counts, evaluated as integers here, because a rate claim that is not
arithmetic is not a rate claim. For L = 848 slots (14-24 kHz at this block size) the hits code
lands near 6 bits a hit, so 16 real sines cost about a hundred bits.

Everything except who pays is held equal:

  * the region is masked out of the scalar waterfill, so these are new bits, not the same bits
    spent twice, and the low spectrum is whatever the scalar budget bought;
  * positions are chosen where they are audible - bins above the mask, loudest first - and bins
    under the mask stay the fill's job, because buying masked content is the one way to make a
    cheap bit useless;
  * the hit bins *overwrite* the fill in the same band rather than adding to it, which is what a
    decoder can actually do (it knows the bin, it knows the band was dead, it writes the tone);
  * the region's own side info, one scale per frame delta-coded with the container's own Rice
    rules, is charged. If that makes the scheme expensive, that is the finding.

The axes reported together on purpose: rate, gain-fitted ES, audible error against the mask, and
sines per frame. Bits and oscillators are two different budgets, and the whole point of this
question is that they trade against each other.

WHAT IT MEASURED (cam15, 15 s, 48 kHz)

The layer buys 14-18 kHz, and only 14-18 kHz. At a matched rate:

  scalar 48 b/f          11.98 kbit/s   ES 5.12   63.5 sines/f   14-18k = -71.7 dB
  hits 14k, m=1, 14.6/f  11.84 kbit/s   ES 4.46   57.6 sines/f   14-18k = -46.5 dB
  hits 14k, m=2, 14.6/f  12.18 kbit/s   ES 4.56   57.6 sines/f   14-18k = -52.3 dB
  masked, no hits, 32 b/f 8.79 kbit/s   ES 4.54   42.9 sines/f   14-18k = -72.2 dB
                                                       (source: -49.5 dB)

so 25 dB of real bandwidth for 0.66 dB of gain-fitted ES and 6 oscillators. At equal *quality*
instead of equal rate the same bandwidth costs +31% rate (ES ~4.17: scalar 7.23 kbit/s, hits
9.50). Both statements are true, "it's free" is not. The masked-no-hits row is what makes the
attribution sound: above 14 kHz scalar and masking decode the same thing (-72.2 both), so every
decibel of the m=1 row's advantage is a hit that was actually bought.

The m=1 row lands 3 dB HOTTER than the source's own 14-18 kHz density, not under it: with one
magnitude bit every hit decodes to the same level, so the level is systematic rather than
fitted. Audible error is unchanged, so it does no harm, but it is a quantiser artifact and not
evidence that the band is better reproduced than it should be.

Three things that keep it honest:

  * One magnitude bit beats two, on rate and density both (11.95 kbit/s / -46.5 dB against
    12.30 / -52.3). The combinatorial position code is where the money is; magnitude bits are
    where the waste is. That is the actual "abuse".
  * Above 18 kHz it buys nothing, in every arm, at every budget: the mask is right that there
    is nothing audible there. The region must stop around 14-18 kHz or it is paying a scale for
    empty air.
  * On ij15 there are 0.1 hits/frame above 14 kHz, because that material's HF sits under its
    own mask. A layer like this is per-frame material, not a codec feature you can always-on.

The claim the experiment was built around - that this leaves slack for *more* sines to run - is
false at a matched rate: scalar plays more oscillators at the same bitrate, they just decode to
noise nobody hears. The layer buys bandwidth, not headroom.
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
import rsc3_encode as enc
import rsc3_stream as st

N2 = 4096
W = 16
GROUPS = ((0, 5e3), (5e3, 10e3), (10e3, 14e3), (14e3, 18e3), (18e3, 24e3))
_NC = {}


def log2_C(M, R):
    """bits to say WHICH R of M slots carry a hit. Exact integer, then converted."""
    if R <= 0 or R > M:
        return 0.0
    key = ("C", M, R)
    if key not in _NC:
        _NC[key] = math.comb(M, R)
    n = _NC[key]
    return math.log2(n) if n > 1 else 0.0


def log2_N(M, K):
    """Pyramid cardinality. Not used by the hits arm, but it is the other price for the same
    region and the arithmetic is one line, so keeping it costs nothing and an evaluation of the
    pulse form then needs no new code."""
    if K <= 0:
        return 0.0
    key = ("N", M, K)
    if key not in _NC:
        _NC[key] = sum((1 << r) * math.comb(M, r) * math.comb(K - 1, r - 1)
                       for r in range(1, min(M, K) + 1))
    n = _NC[key]
    return math.log2(n) if n > 1 else 0.0


def read_wav(p, secs=15.0):
    wf = wave.open(p, "rb")
    sr = wf.getframerate()
    d = wf.readframes(min(wf.getnframes(), int(secs * sr)))
    wf.close()
    return np.frombuffer(d, np.int16).astype(np.float64).reshape(-1, 2) / 32768.0, sr


def rice_bits(prev, cur):
    """A delta stream at the price the container pays for its other deltas, so nothing this
    scheme transmits goes uncounted."""
    z = st.zz_vec(np.asarray(cur, np.int64) - np.asarray(prev, np.int64))
    k = st.pick_k(z)
    _, wd = st.rice_fields(z, k)
    return 5 + int(wd.sum())


def score(x, sr, X3all, nfr, ball, mant, Eall, thr3, hz, extra=0.0):
    """Rate through frame_costs (bit-exact against write_frame, and that equivalence is a test,
    not a belief) plus `extra` bits/frame that some other code pays, plus the quality rulers and
    the oscillator count."""
    tot = st.frame_costs(Eall, ball, mant, W,
                         [st.run_fill_words(st.band_fill_cap(X3all[f], thr3[f]), ball[f], "safe")
                          for f in range(nfr)])
    nfr_total = float(tot.sum())
    frate = sr / (N2 / 2)
    kbps = (nfr_total + extra) / nfr * frate / 1000.0
    X = X3all.reshape(nfr, -1)
    y = r3.synthesise(X * (N2 / 4.0), N2, x.shape[0])
    lo, hi = 3 * N2, x.shape[0] - N2
    g = float(np.dot(x[lo:hi, 0], y[lo:hi])) / max(float(np.dot(y[lo:hi], y[lo:hi])), 1e-300)
    es = 10 * math.log10(float(np.mean(x[lo:hi, 0] ** 2))
                          / max(float(np.mean((x[lo:hi, 0] - g * y[lo:hi]) ** 2)), 1e-300))
    # Audible error has to live where the mask lives, which is the analysis domain, not the time
    # domain: an error signal reshaped from samples into bins is a different quantity entirely.
    Ae = r3.amplitude(r3.analyse(x[:, 0] - g * y, N2)[0], N2)
    Ae = Ae[:nfr]
    ed = 20 * np.log10(np.maximum(np.abs(Ae), 1e-300)).reshape(nfr, -1)[:, :thr3.size // nfr]
    aud = float(np.sum(np.maximum(ed[:, :Ae.shape[1]] - thr3.reshape(nfr, -1)[:, :Ae.shape[1]],
                                  0.0)))
    aud_db = -10 * math.log10(max(aud / (nfr * Ae.shape[1]), 1e-30))
    dens = [10 * math.log10(max(float((X ** 2)[:, (hz >= a) & (hz < b)].mean()), 1e-30))
            for a, b in GROUPS]
    # Tonal count, not lit-up-bin count. A bin carrying fill noise is not an oscillator the
    # player schedules, so counting X != 0 makes the dead bands look expensive and the honest
    # part of the trade invisible.
    tonal = float((mant != 0).sum()) / nfr
    return kbps, es, aud_db, tonal, dens


def arm_scalar(x, sr, budgets):
    """The control is the shipping encoder itself, so the left end of every curve below is a
    number this repo already publishes rather than a model of it."""
    A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
    nfr, nc = A.shape
    nb = st.band_grid(nc, W)
    Ad = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
    thr3 = r3.threshold_dbfs(Ad, sr, nc, psycho=True).reshape(nfr, nb, W)
    A3 = A.reshape(nfr, nb, W)
    hz = np.arange(nc) * sr / N2
    rows = []
    for B in budgets:
        bits = enc.band_bits(A, sr, B, psycho=True)
        Eall = np.array([st.band_exponents(A3[f]) for f in range(nfr)])
        mant = np.array([st.quantise_bands(A3[f], Eall[f], bits[f]) for f in range(nfr)])
        rows.append((B,) + score(x, sr, np.array([st.reconstruct_bands(Eall[f], bits[f], mant[f],
                                                                     st.run_fill_words(
                                                                         st.band_fill_cap(A3[f],
                                                                                          thr3[f]),
                                                                         bits[f], "safe"), f, W)
                                                  for f in range(nfr)]),
                                 nfr, bits, mant, Eall, thr3, hz))
    return rows, (A3, nfr, nb, Ad, thr3, Eall, bits, mant, hz)


def arm_hits(x, sr, base, f0, R, m, budgets):
    """Scalar below f0 with the region masked out, plus R real sines above it."""
    A3, nfr, nb, Ad, thr3, Eall, _, _, hz = base
    thr2 = thr3.copy()
    band_hz = hz.reshape(nb, W).mean(axis=1)
    reg = (band_hz >= f0)
    thr2[:, reg] = 991.0
    bl2 = np.maximum(r3._band_level_db(Ad, r3.band_edges(nb * W)), thr2.reshape(nfr, -1))
    out = []
    for SB in budgets:
        bits = np.clip(r3.waterfill_select(Ad, thr2.reshape(nfr, -1), bl2, SB)
                       .reshape(nfr, nb, W).max(axis=2), 0, st.MAX_BITS).astype(np.int64)
        mant = np.array([st.quantise_bands(A3[f], Eall[f], bits[f]) for f in range(nfr)])
        X = np.zeros((nfr, nb, W))
        regbits = 0
        nhit = 0
        prev_Er = np.zeros(1, np.int64)
        for f in range(nfr):
            X[f] = st.reconstruct_bands(Eall[f], bits[f], mant[f],
                                        st.run_fill_words(st.band_fill_cap(A3[f], thr3[f]),
                                                          bits[f], "safe"), f, W)
            if not reg.any():
                continue
            pk = float(np.abs(A3[f][reg]).max())
            if pk <= 0.0:
                continue
            Er = int(math.ceil(math.log2(max(pk, 2.0 ** -60))))
            sel = reg[:, None] & (np.abs(A3[f]) > 10.0 ** (0.05 * thr3[f]))
            idx = np.flatnonzero(sel.reshape(-1))
            if idx.size == 0:
                continue
            mag = np.abs(A3[f].reshape(-1)[idx])
            if idx.size > R:
                idx = idx[np.argsort(-mag)[:R]]
                mag = np.abs(A3[f].reshape(-1)[idx])
            r_act = int(idx.size)
            # The step comes from the region's own peak so a hit can never be quantised to zero,
            # and bins that would land there are simply not bought. A hit that decodes to nothing
            # is not a cheap bit, it is a wasted one, and it also breaks the rate arithmetic:
            # log2 C(L, R) counts the positions the decoder has to unpack, so R here must be the
            # hits that are actually placed, not the hits the grid asked for.
            step = 2.0 ** (Er - (m - 1))
            q = np.clip(np.round(mag / step), 1, (1 << m) - 1).astype(np.int64)
            keep = q >= 1
            idx, q = idx[keep], q[keep]
            r_act = int(idx.size)
            if r_act == 0:
                continue
            sgn = np.sign(A3[f].reshape(-1)[idx])
            X[f].reshape(-1)[idx] = sgn * q * step
            regbits += int(math.ceil(log2_C(int(reg.sum()) * W, r_act))) \
                + r_act * (1 + m) + rice_bits(prev_Er, np.array([Er]))
            prev_Er = np.array([Er])
            nhit += r_act
        out.append((SB, score(x, sr, X, nfr, bits, mant, Eall, thr3, hz, extra=regbits),
                    regbits / nfr, nhit / nfr))
    return out


if __name__ == "__main__":
    budgets_sc = (16, 24, 32, 40, 48, 64, 96)
    budgets_hb = (24, 28, 32, 36, 40, 48)
    # R = 0 is the control that matters: same masking, NO hits, so the gap between its density
    # columns and the scalar arm's is pure fill, and anything above that is what hits bought.
    hit_grids = [(14e3, 2, (0, 16, 64, 256)), (14e3, 1, (0, 16, 64, 256)),
                 (10e3, 2, (0, 16, 64, 256)), (10e3, 1, (16, 64))]
    for name in ("cam15.wav", "ij15.wav"):
        x, sr = read_wav(os.path.join(LAB, name))
        print(f"\n{name}   {x.shape[0]} samples  {sr} Hz")
        print(f"  {'arm':>10} {'budget':>6} {'hits':>5} {'m':>2}  {'kbit/s':>7} {'ESf':>6} "
              f"{'aud':>6} {'sines/f':>7} {'reg/f':>6}   "
              + " ".join(f"{a/1e3:>5.0f}-{b/1e3:<4.0f}" for a, b in GROUPS))
        rows, base = arm_scalar(x, sr, budgets_sc)
        for B, kb, es, au, sn, dens in rows:
            print(f"  {'scalar':>10} {B:6d} {'-':>5} {'-':>2}  {kb:7.2f} {es:6.2f} {au:6.1f} "
                  f"{sn:7.1f} {'-':>6}   " + " ".join(f"{d:9.1f}" for d in dens))
        for f0, m, grid in hit_grids:
            for R in grid:
                res = arm_hits(x, sr, base, f0, R, m, budgets_hb)
                for SB, (kb, es, au, sn, dens), rb, nh in res:
                    print(f"  {('hits%dk' % (f0 / 1e3)):>10} {SB:6d} {nh:5.1f} {m:2d}  "
                          f"{kb:7.2f} {es:6.2f} {au:6.1f} {sn + nh:7.1f} {rb:6.0f}   "
                          + " ".join(f"{d:9.1f}" for d in dens))
        sys.stdout.flush()
