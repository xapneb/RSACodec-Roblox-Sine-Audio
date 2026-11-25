"""
How often does the shape field actually say something, in the encoder that writes files?

The rate number (21.8 / 27.5 b/f) is easy to read off an encode. The distribution behind it is
not, and it matters: a field that costs 21 b/f because one frame in ten pays 200 is a different
proposition than one that pays 21 every frame, because only the second one can be gated off.

This runs the encoder's own decision function, st.shape_field, frame by frame and counts lit
blocks per frame. No harness, no re-derived thresholds: same calls, same order.
"""
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

W = 16
for src, br in ((sys.argv[1], int(sys.argv[2])), (sys.argv[3], int(sys.argv[4]))):
    wf = wave.open(os.path.join(LAB, src), "rb")
    sr = wf.getframerate()
    raw = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768.0
    wf.close()
    x = raw.reshape(-1, 2)[:, 0]
    A = r3.amplitude(r3.analyse(x, 4096)[0], 4096)
    nfr, nc = A.shape
    nb = st.band_grid(nc, W)
    A3 = A.reshape(nfr, nb, W)
    bits = enc.band_bits(A, sr, br, psycho=True)
    thr = r3.threshold_dbfs(20 * np.log10(np.maximum(np.abs(A), 1e-30)), sr, nc,
                            psycho=True).reshape(nfr, nb, W)
    reg = st.shape_reg(sr, 2 * nc, W)
    per = np.zeros(nfr, np.int64)
    cost = np.zeros(nfr, np.int64)
    for f in range(nfr):
        field = st.shape_field(st.band_fill_cap(A3[f], thr[f]), bits[f], reg)
        per[f] = sum(int(v.sum()) for v in field if v is not None)
        cost[f] = st.shape_cost(field)
    lit = int((per > 0).sum())
    print(f"{src:12s} {nfr} frames   {lit} with at least one lit block   "
          f"mean {cost.mean():5.1f} b/f   p50 {int(np.median(cost)):2d}   "
          f"p90 {int(np.percentile(cost, 90)):2d}   max {int(cost.max()):3d}   "
          f"zero in {int((cost == 0).sum()):3d}")
