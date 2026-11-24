"""
Level-fit every measurement, because a coherent gain offset is not coding error.

cam's lag check caught RSC2 decoding 6.3 dB under the source while still
peaking at 1.0. Since error power of a scaled signal is ((1-g)^2 * P_x), a pure
6 dB level loss caps ES at +6.0 dB regardless of codec quality - so my table
has been charging RSC2 for a level difference it did not create with
quantisation, and that bias is not even in the direction of my conclusions.

Fix, stated once: fit the least-squares gain g = <x,y> / <y,y>, then measure
error against g*y. That isolates the part of the error a codec can't fix by
being turned up. Report both, so nothing is quietly improved.
"""
import sys, os
import numpy as np
from scipy.io import wavfile

lab = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rsc2_testing_lab")
N2, hop = 4096, 1024


def load(p):
    sr, x = wavfile.read(p)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / 32768.0
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    return sr, x


for src_name, tag, files in ((
        "cam.wav", "cam", ("cam_rsc3_psy", "cam_rsc3_blind", "cam_rsc3_psy_nf", "cam_rsc2_mp8")),
    (
        "ij.wav", "ij", ("ij_rsc3_psy", "ij_rsc3_blind", "ij_rsc2_mp8"))):
    sr, x = load(os.path.join(lab, src_name))
    print(f"\n{tag}  (source rms {np.sqrt((x[100000:600000]**2).mean()):.4f})")
    print(f"{'file':16s} {'gain dB':>8} {'ES naive':>9} {'ES fitted':>10}   "
          f"{'what changed':>14}")
    for f in files:
        p = os.path.join(lab, f + ".wav")
        if not os.path.exists(p):
            continue
        _, y = load(p)
        n = min(len(x), len(y))
        lo, hi = 3 * N2, n - N2
        a, b = x[lo:hi], y[lo:hi]
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        naive = 10 * np.log10((a ** 2).mean() / max(((a - b) ** 2).mean(), 1e-300))
        g = float((a * b).mean() / max((b * b).mean(), 1e-300))
        fit = 10 * np.log10((a ** 2).mean() / max(((a - g * b) ** 2).mean(), 1e-300))
        print(f"{f:16s} {20*np.log10(max(g, 1e-9)):+8.2f} {naive:9.2f} {fit:10.2f}   "
              f"{fit - naive:+14.2f}")
