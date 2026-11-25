"""
A band can say which bins are on for 16 flat bits. Does that beat both shapes now?

The mantissa field now has two shapes and the crossover between them is measurable: gap coding
wins when a band is sparse, flat coding wins when it is full, and between the two the encoder is
choosing the lesser of two mediocre prices. A 16-bit occupancy mask sits exactly in that gap.
It costs 16 bits, flat, to name every bin in the band - the same price as the flat shape's
first bit for every bin - and then it pays (1 + b) only for the bins that are actually on, so
unlike flat it never pays for a zero, and unlike gaps it never pays 4 bits to point.

        gap    cb + n * (gb + 1 + b)     wins for n <= 2 on this material
        mask   W       + n * (1 + b)     should win for  3 <= n < 16b/(1+b)
        flat   W * (1 + b)              wins only when the band is really full

The crossover against flat is n < W*b/(1+b), which for a 5-bit band is n < 13: the mask is
cheaper than flat on any band that is not completely full, which is most of them. So if this
is right it does not add a third option so much as replace two of them.

The price has to be honest, and there are two prices to pay: the selector that tells the
decoder which shape it is reading, and the fact that a mask is 16 bits even when the band has
one bin in it. Both shapes are priced with the same selector width, so the comparison is not
rigged, and the decoded values cannot move - same grid, same step, same clamp.
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
FRATE = 48000 / (N2 / 2)

for name in ("cam15.wav", "ij15.wav"):
    wf = wave.open(os.path.join(LAB, name), "rb")
    sr = wf.getframerate()
    x = (np.frombuffer(wf.readframes(int(15 * sr)), np.int16).astype(np.float64)
         / 32768.0).reshape(-1, 2)
    A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
    nfr, nc = A.shape
    nb = st.band_grid(nc, W)
    A3 = A.reshape(nfr, nb, W)
    gb, cb = st.gap_bits(W), st.count_bits(W)
    print(f"\n{name}   {nfr} frames   {nb} bands of {W}   gap {gb} b, count {cb} b")
    print(f"  {'budget':>6} {'2 shapes':>9} {'+ mask':>9} {'saved':>8} {'kbit/s':>8}"
          f" {'gap%':>6} {'mask%':>6} {'flat%':>6}")
    for budget in (16, 33, 64, 384, 1024):
        bits = np.clip(r3.allocate(A, sr, budget, psycho=True).reshape(nfr, nb, W)
                       .max(axis=2).astype(np.int64), 0, st.MAX_BITS)
        two = three = 0
        cnt = np.zeros(3)
        for f in range(nfr):
            E = st.band_exponents(A3[f])
            m = st.quantise_bands(A3[f], E, bits[f])
            live = bits[f] > 0
            n = (m != 0).sum(axis=1)
            b1 = np.maximum(bits[f], 1)
            c_gap = cb + n * (gb + 1 + b1)
            c_mask = W + n * (1 + b1)
            c_flat = W * (1 + b1)
            # two-shape world, what ships now: 1 flag bit per live band
            two += int(np.where(live, 1 + np.minimum(c_gap, c_flat), 0).sum())
            # three shapes: a band pays 1 bit if it is the cheap gap case, else 2.
            # Ordering the selector by cost keeps it 1-2 bits rather than always 2.
            pick3 = np.minimum(np.minimum(c_gap, c_mask), c_flat)
            # three shapes, priced honestly: 1 bit says "sparse" (gap), and if not sparse a
            # second bit distinguishes mask from flat. So a sparse band pays 1 selector bit
            # and nothing else pays more than it has to.
            sel = 1 + np.where(live & (pick3 != c_gap), 1, 0)
            three += int(np.where(live, sel + pick3, 0).sum())
            best = np.where(live, np.argmin(np.stack([c_gap, c_mask, c_flat], axis=0)), 3)
            cnt += np.bincount(best, minlength=4)[:3]
        print(f"  {budget:6d} {two / nfr:9.1f} {three / nfr:9.1f} {(two - three) / nfr:8.1f}"
              f" {(two - three) / nfr * FRATE / 1000:8.2f}"
              f" {cnt[0] / max(cnt.sum(), 1) * 100:5.0f}% {cnt[1] / max(cnt.sum(), 1) * 100:5.0f}%"
              f" {cnt[2] / max(cnt.sum(), 1) * 100:5.0f}%")
