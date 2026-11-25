"""Is the HF hit layer worth its price when there are almost no bits to spend?

At 10 kbit/s the layer costs ~150 b/f and buys 25 dB of 14-18 kHz (see vector_hits.py). That
price is fixed: hits per frame is set by what the source has above the mask, not by the scalar
budget, so at very low rates the layer is a *bigger* slice of the frame, not a smaller one. At
budget 16 a whole frame is ~236 b/f, and 15 hits would be nearly 70% of it, so the only honest
way to spend them is to gut the scalar arm to near-silence. This measures whether that is
actually worth hearing, as matched-rate pairs:

    scalar budget B        ->  kbit/s, ES, audible, density
    masked budget B' + R hits, total rate lands as close to B as integers allow

The worry stated up front so a good number can kill it: if the pair is equal-rate and the ES
column says "worse" while the density column says "fuller", that is exactly the trade shipping
RSC2 already makes with its HF bins, and the honest answer is no.

Run:  python _hits_lowrate.py
"""
import os
import sys

import vector_hits as vh

HERE = os.path.dirname(os.path.abspath(__file__))

for name in ("cam15.wav", "ij15.wav"):
    x, sr = vh.read_wav(os.path.join(vh.LAB, name), secs=15.0)
    print(f"\n{name}   15 s   (frame = 23.44 fps, so 1 kbit/s = 42.7 b/f)")
    rows, base = vh.arm_scalar(x, sr, (4, 8, 16, 24))
    for r in rows:
        B, kb, es, au, sn, dens = r
        print(f"  scalar  {B:4d} b     {kb:6.2f} kbit/s  ESf {es:6.2f}  aud {au:5.1f} "
              f"sines {sn:6.1f}  10-14 {dens[2]:6.1f}  14-18 {dens[3]:6.1f}  18-24 {dens[4]:6.1f}")
    for R in (16, 64):
        for r in vh.arm_hits(x, sr, base, 14e3, R, 1, (2, 4, 8, 12, 16, 20)):
            SB, (kb, es, au, sn, dens), rb, nh = r
            print(f"  hits    {SB:4d} b+{nh:4.1f}h {kb:6.2f} kbit/s  ESf {es:6.2f}  aud {au:5.1f} "
                  f"sines {sn + nh:6.1f}  10-14 {dens[2]:6.1f}  14-18 {dens[3]:6.1f} "
                  f"18-24 {dens[4]:6.1f}   reg {rb:5.0f} b/f")
    sys.stdout.flush()
