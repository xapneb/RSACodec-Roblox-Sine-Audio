"""Shape the dead-run noise after the mask instead of laying it flat.

The shipping fill is deliberately the quietest thing that is legal: one level word per dead run,
and the "safe" rule picks the level so that no band inside the run can ever sit above its own
mask. On a 14 kHz-and-up run that is ~136 bands, the binding bin is the one with the least
masking capacity - which after 16 kHz is near the floor - so the whole run is filled at the
level legal for its HIGHEST frequency, and the part where the mask actually had room (14-18 kHz,
where the source still has something) gets silence with a certificate. That certificate costs
nothing, but so does the density it refuses: measured, masked flat fill and dead scalar land
within a dB of each other above 14 kHz, and both are ~22 dB under the source.

The question is how much SHAPE is worth buying, because shape is what turns "a floor" back into
"the treble". A price ladder, all of it mask-safe band-by-band (the level on band j is always at
or below band j's own ceiling, so no arm can make itself inaudible by lying), ordered by what
transmitting the shape costs:

  flat        8 b/run      what ships: level + 2-bit tilt
  tilt16     10 b/run      same run words, a 16-slope table instead of 4 slopes
  blocks      +Rice/stream  one word per G-band block, delta-coded (G = 16, 4)
  bins        +~1 b/band   every dead band's own level word, delta-coded across the run

The block and bin forms are new grammar (the run structure comes from b, so positions are free;
the levels are a delta stream like every other field). Charge model, stated once: the flat word
for every dead run is still written, exactly as it is today, and the shape rides ON TOP of it -
a shape code of zero means "this band's flat word stands", which is both cheaper and legal,
because the flat word is band-safe by the safe rule. What is charged is only what is new: a
per-run flag bit, the first block's absolute 6-bit word, and a 5-bit Rice parameter per delta
stream.

Rulers: rate, gain-fitted ES, audible error against the mask, density by group. Audible error
cannot get worse than the control by construction - every filled band sits at or under its own
threshold minimum - so if a shaped arm prints more audible error, that is a bug, not a trade.
What decides it is the density column against the price column, and then listening.

Run:  python _shaped_hf.py
"""
import math
import os
import sys

import numpy as np

import vector_hits as vh
import rsc3_codec as r3
import rsc3_stream as st

N2, W = vh.N2, vh.W
GROUPS = vh.GROUPS

# A wider slope table than the container's 4 entries: 16 log-spaced slopes, transmitted as a
# 4-bit index. Quadratic spacing because masking capacity falls with frequency super-linearly,
# and only the fine end of the range needs the precision.
TILT = tuple(np.round(-200.0 * (np.linspace(0.0, 1.0, 16) ** 2), 1))
TILT_BITS = 4


def score_nofill(x, sr, X, nfr, ball, mant, Eall, thr3, hz, extra, fillw=None):
    """vh.score, with an explicit fill-word list (charged exactly as the shipping grammar does)
    plus `extra` bits/frame for whatever shape grammar this arm adds."""
    tot = st.frame_costs(Eall, ball, mant, W, fillw)
    frate = sr / (N2 / 2.0)
    kbps = (float(tot.sum()) + float(extra)) / nfr * frate / 1000.0
    XX = np.asarray(X).reshape(nfr, -1)
    y = r3.synthesise(XX * (N2 / 4.0), N2, x.shape[0])
    lo, hi = 3 * N2, x.shape[0] - N2
    g = float(np.dot(x[lo:hi, 0], y[lo:hi])) / max(float(np.dot(y[lo:hi], y[lo:hi])), 1e-300)
    es = 10 * math.log10(float(np.mean(x[lo:hi, 0] ** 2))
                          / max(float(np.mean((x[lo:hi, 0] - g * y[lo:hi]) ** 2)), 1e-300))
    Ae = r3.amplitude(r3.analyse(x[:, 0] - g * y, N2)[0], N2)[:nfr]
    ed = 20 * np.log10(np.maximum(np.abs(Ae), 1e-300)).reshape(nfr, -1)[:, :thr3.size // nfr]
    aud = float(np.sum(np.maximum(ed[:, :Ae.shape[1]] - thr3.reshape(nfr, -1)[:, :Ae.shape[1]],
                                  0.0)))
    aud_db = -10 * math.log10(max(aud / (nfr * Ae.shape[1]), 1e-30))
    dens = [10 * math.log10(max(float((XX ** 2)[:, (hz >= a) & (hz < b)].mean()), 1e-30))
            for a, b in GROUPS]
    return kbps, es, aud_db, float((mant != 0).sum()) / nfr, dens


def dither_codes(cap_db, b):
    """Each band's own ceiling, floored DOWN onto the fill word grid. Floor, not round: a code
    that rounds up would be the decoder playing noise the encoder never certified."""
    step = -st.FILL_FLOOR / ((1 << st.FILL_BITS) - 1)
    c = np.floor((np.maximum(np.asarray(cap_db, np.float64), st.FILL_FLOOR)
                  - st.FILL_FLOOR) / step + 1e-9)
    c = np.clip(c, 0, (1 << st.FILL_BITS) - 1).astype(np.int64)
    c[np.asarray(b) > 0] = 0
    return c


def block_codes(codes, G):
    """Per-block level = MIN of the block's codes, so the block decodes at the level that is
    legal for its tightest bin even after the word grid floored each one."""
    m = codes.size
    if m == 0:
        return codes
    return np.array([int(codes[i * G:min(i * G + G, m)].min())
                     for i in range((m + G - 1) // G)], np.int64)


def delta_rice(lv):
    """The price of one level stream: the first block's absolute word, and k + Rice for the rest.
    A one-block run needs no stream, so its grammar is shorter, and it is charged shorter."""
    if lv.size <= 1:
        return 6
    z = st.zz_vec(np.diff(lv).astype(np.int64))
    k = st.pick_k(z)
    _, wd = st.rice_fields(z, k)
    return 6 + 5 + int(wd.sum())


def shape_levels(cap, b, reg, mode, G):
    """Per-band dB levels for the dead bands of the region, plus the exact bits to ship them.
    Where a code is zero no level is returned, which means the flat word already charged for
    that run stands there - legal, because the safe rule floored that word under this run's
    tightest bin, so under any individual band's ceiling."""
    runs = st.dead_runs(b)
    codes = dither_codes(cap, b) if mode in ("block", "bins") else None
    levels = {}
    bits = 0
    for r in runs:
        # Shape the run-intersection with the region, not the run. Runs straddle f0 (in quiet
        # frames everything above ~12 kHz is dead at once), and writing the below-region bands
        # too is legal-per-band but changes what this experiment claims. The decoder derives the
        # same intersection from b and the codec's f0, so charging per intersected run is exact.
        j = np.asarray(r)[reg[np.asarray(r)]]
        if j.size == 0:
            continue
        bits += 1  # run flag: this run's shape field is on
        if mode == "tilt16":
            cap_r = cap[j]
            mid = (j[0] + j[-1]) * 0.5
            half = max((j[-1] - j[0]) * 0.5, 1.0)
            p = (j - mid) / half
            best_lv, best_q = st.FILL_FLOOR, 0
            for q, s in enumerate(TILT):
                lv = float(np.min(cap_r - s * p))
                if lv > best_lv:
                    best_lv, best_q = lv, q
            code = st.fill_code(best_lv)
            while code and st.fill_level(code) > best_lv + 1e-9:
                code -= 1
            if code:
                bits += TILT_BITS
                for jf in j:
                    levels[int(jf)] = (st.fill_level(code)
                                       + TILT[best_q] * (jf - mid) / half)
            continue
        c_r = codes[j]
        lv = block_codes(c_r, G) if mode == "block" else c_r
        if not lv.any():
            continue
        bits += delta_rice(lv)
        for t, band_j in enumerate(j):
            blk = t if mode == "bins" else min(t // G, lv.size - 1)
            if lv[blk]:
                levels[int(band_j)] = st.fill_level(int(lv[blk]))
    return levels, bits


def arm_shape(x, sr, base, f0, mode, G, budgets, unmasked_cap=False):
    """Scalar with the region masked out, flat fill shipped everywhere, shaped levels overlaid
    on the dead region bands. mode: 'flat' (no overlay, the control), 'tilt16', 'block', 'bins'.
    unmasked_cap takes the mask OUT of the ceiling, which is the arm that shows whether the
    shape is safe because of the mask or only claims to be."""
    A3, nfr, nb, Ad, thr3, Eall, _, _, hz = base
    band_hz = hz.reshape(nb, W).mean(axis=1)
    reg = band_hz >= f0
    thr2 = thr3.copy()
    thr2[:, reg] = 991.0
    bl2 = np.maximum(r3._band_level_db(Ad, r3.band_edges(nb * W)), thr2.reshape(nfr, -1))
    out = []
    for SB in budgets:
        bits = np.clip(r3.waterfill_select(Ad, thr2.reshape(nfr, -1), bl2, SB)
                       .reshape(nfr, nb, W).max(axis=2), 0, st.MAX_BITS).astype(np.int64)
        mant = np.array([st.quantise_bands(A3[f], Eall[f], bits[f]) for f in range(nfr)])
        X = np.zeros((nfr, nb, W))
        fillw = []
        extra = 0
        for f in range(nfr):
            # The flat words always ship mask-safe, exactly as they do today; unmasked_cap only
            # changes the SHAPE ceiling, so the comparison isolates what the mask is doing.
            fl = st.run_fill_words(st.band_fill_cap(A3[f], thr3[f]), bits[f], "safe")
            fillw.append(fl)
            X[f] = st.reconstruct_bands(Eall[f], bits[f], mant[f], fl, f, W)
            if mode != "flat" and reg.any():
                cap = (st.band_fill_cap(A3[f], None) if unmasked_cap
                       else st.band_fill_cap(A3[f], thr3[f]))
                lv, cost = shape_levels(cap, bits[f], reg, mode, G)
                extra += cost
                for j, lvl in lv.items():
                    X[f][int(j)] = st.fill_noise(int(j), f, W) * 10.0 ** (0.05 * lvl)
        out.append((SB, score_nofill(x, sr, X, nfr, bits, mant, Eall, thr3, hz, extra, fillw),
                    extra / nfr))
    return out


if __name__ == "__main__":
    budgets = (24, 28, 32, 36, 40, 48)
    for name in ("cam15.wav", "ij15.wav"):
        x, sr = vh.read_wav(os.path.join(vh.LAB, name))
        print(f"\n{name}   15 s   {x.shape[0]} samples")
        A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
        nfr, nc = A.shape
        hz = np.arange(nc) * sr / N2
        sd = [10 * math.log10(max(float((A ** 2)[:, (hz >= a) & (hz < b)].mean()), 1e-30))
              for a, b in GROUPS]
        print(f"  {'arm':>14}  {'kb/s':>6} {'ESf':>6} {'aud':>5} {'sn':>5} {'b/f':>5}   "
              + " ".join(f"{a / 1e3:>3.0f}-{b / 1e3:<3.0f}" for a, b in GROUPS) + "   (source)")
        print("  " + " " * 47 + " ".join(f"{d:6.1f}" for d in sd))
        rows, base = vh.arm_scalar(x, sr, (32, 48))
        for B, kb, es, au, sn, dens in rows:
            print(f"  {'scalar':>14}  {kb:6.2f} {es:6.2f} {au:5.1f} {sn:5.1f} {'-':>5}   "
                  + " ".join(f"{d:6.1f}" for d in dens))
        for f0 in (14e3,):
            ctrl = vh.arm_hits(x, sr, base, f0, 0, 1, budgets)
            for SB, (kb, es, au, sn, dens), rb, nh in ctrl:
                print(f"  {'flat':>14}  {kb:6.2f} {es:6.2f} {au:5.1f} {sn:5.1f} {rb:5.0f}   "
                      + " ".join(f"{d:6.1f}" for d in dens))
            for mode, G, tag in (("tilt16", 1, "tilt16"), ("block", 16, "blk16"),
                                 ("block", 4, "blk4"), ("bins", 1, "bins")):
                for SB, (kb, es, au, sn, dens), xb in arm_shape(x, sr, base, f0, mode, G,
                                                                budgets):
                    print(f"  {tag:>14}  {kb:6.2f} {es:6.2f} {au:5.1f} {sn:5.1f} {xb:5.0f}   "
                          + " ".join(f"{d:6.1f}" for d in dens))
            for SB, (kb, es, au, sn, dens), xb in arm_shape(x, sr, base, f0, "block", 16,
                                                            budgets, unmasked_cap=True):
                print(f"  {'blk16-nomask':>14}  {kb:6.2f} {es:6.2f} {au:5.1f} {sn:5.1f} "
                      f"{xb:5.0f}   " + " ".join(f"{d:6.1f}" for d in dens))
        sys.stdout.flush()
