"""
rate_curve.py - what this codec actually costs, in kbit/s, and what it actually sounds like.

Budgets are not rates. Everything in this directory is quoted per frame, and the number a
listener hears is bits/second, which adds the frame rate and - the part that got away from us -
everything the container writes around the mantissas. At budget 512 the file costs 3611 b/f,
not 512, because the allocation buys bit *planes* per coefficient while the container pays
(1 + b) for every nonzero bin of every live band. A band whose peak needs 8 bits and whose
neighbours are merely audible costs 16 x 9, not 8.

That is why the old "+5.2 dB at 1.4x lower rate" line cannot survive as written. It compared a
RSC2 file rate against an RSC3 *model* rate. Both sides now have to be measured the same way,
and this tool exists to do that: for each budget, the real payload on disk and both rulers -
plain ES, and gain-fitted ES, the ruler this repo actually listens with, because it does not
charge a coherent level offset as error.

The RSC2 half of that comparison is not this file's output. When it is quoted again it should
come from decoding an actual .rsc through rsc_decoder.py and counting its bytes, not from a
budget. Until someone does that, the honest statement is in the table: rates are per channel,
measured, and the claim against RSC2 is unproven.

Run:  python rate_curve.py [wav ...]        (default: this repo's two reference tracks)
      python rate_curve.py --budgets 1,16,64,256,1024
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rsc3_codec as r3
import rsc3_encode as enc
import rsc3_decode as dec

N2 = 4096
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
WIDTH = r3.BAND_WIDTH


def read_wav(path, secs=None):
    wf = wave.open(path, "rb")
    sr = wf.getframerate()
    nch = wf.getnchannels()
    n = wf.getnframes()
    if secs:
        n = min(n, int(secs * sr))
    raw = np.frombuffer(wf.readframes(n), np.int16).astype(np.float64) / 32768.0
    wf.close()
    return raw.reshape(-1, nch), sr


def fit_es(x, y, n2):
    """Pooled over channels, on the settled part of the file. Returns BOTH rulers: plain ES,
    which charges a coherent level offset as error, and gain-fitted ES, which is what this repo
    listens with. This used to return the fitted number twice - the plain column in the table
    was the fitted one all along, so nothing could see a level loss. Keep them distinct."""
    lo, hi = 3 * n2, min(len(x), len(y)) - n2
    num = den = den_fit = 0.0
    for c in range(x.shape[1]):
        a, b = x[lo:hi, c], y[lo:hi, c]
        g = float(np.dot(a, b)) / max(float(np.dot(b, b)), 1e-300)
        num += float(np.mean(a * a))
        den += float(np.mean((a - b) ** 2))
        den_fit += float(np.mean((a - g * b) ** 2))
    base = max(float(num), 1e-300)
    return (10 * math.log10(base / max(den, 1e-300)),
            10 * math.log10(base / max(den_fit, 1e-300)),
            num)


def curve(x, sr, budgets, fill=True, mode="safe"):
    """Every budget measured through the container, which is the only thing worth quoting.
    `fill` is an argument for the same reason it is one in the encoder: the shipping default
    sends a noise floor, so a curve measured without it describes a file nobody encodes."""
    dur = x.shape[0] / sr
    nfr = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2).shape[0]
    rows = []
    A_all = [r3.amplitude(r3.analyse(x[:, c], N2)[0], N2) for c in range(x.shape[1])]
    for b in budgets:
        bits_total = 0
        err = err_fit = sig = 0.0
        for c, A in enumerate(A_all):
            p, rc = enc.encode_channel(A, sr, b, psycho=True, fill=fill, mode=mode)
            bits_total += len(p) * 8
            y = r3.synthesise(rc * (N2 / 4.0), N2, x.shape[0])
            lo = 3 * N2
            a = x[lo:-N2, c]
            d = y[lo:-N2]
            g = float(np.dot(a, d)) / max(float(np.dot(d, d)), 1e-300)
            sig += float(np.mean(a * a))
            err += float(np.mean((a - d) ** 2))
            err_fit += float(np.mean((a - g * d) ** 2))
        kbps = bits_total / dur / 1000.0
        rows.append((b, bits_total / nfr, kbps / x.shape[1],
                     10 * math.log10(sig / max(err, 1e-300)),
                     10 * math.log10(sig / max(err_fit, 1e-300))))
    return A_all, nfr, rows


def main():
    ap = argparse.ArgumentParser(description="measured rate/quality curve of RSC3")
    ap.add_argument("files", nargs="*", default=[os.path.join(LAB, f) for f in
                                                ("cam15.wav", "ij15.wav")])
    ap.add_argument("--budgets", default="1,4,16,64,256,1024,4096")
    ap.add_argument("--secs", type=float, default=15.0)
    ap.add_argument("--no-fill", dest="fill", action="store_false", default=True,
                    help="leave dead bands silent, i.e. measure the previous bit map")
    ap.add_argument("--fill-mode", choices=("safe", "mean"), default="safe",
                    help="which fill this curve is describing (see rsc3_encode.py)")
    args = ap.parse_args()

    budgets = [int(t) for t in args.budgets.split(",") if t.strip()]
    print(f"fill: {'off' if not args.fill else args.fill_mode}")
    for path in args.files:
        x, sr = read_wav(path, args.secs)
        print(f"\n{os.path.basename(path)}   {x.shape[0]} samples  {sr} Hz  {x.shape[1]} ch")
        print(f"   {'budget':>8} {'b/f file':>10} {'kbit/s/ch':>11}   {'ES':>7}   "
              f"{'gain-fitted ES':>15}   {'bits are':>9}")
        A_all, nfr, rows = curve(x, sr, budgets, fill=args.fill, mode=args.fill_mode)
        for b, bpf, kbps, es, esf in rows:
            print(f"   {b:8d} {bpf:10.1f} {kbps:11.2f}   {es:7.2f}   {esf:15.2f}")
        # where the bits actually go, once, so nobody has to trust a model of the file again
        _, _, cost = enc.encode_channel(A_all[0], sr, budgets[-1], psycho=True, want_cost=True,
                                        fill=args.fill, mode=args.fill_mode)
        tot = float(cost.sum())
        print(f"   at budget {budgets[-1]}: scale {cost[0] / nfr:.1f} b/f, "
              f"counts {cost[1] / nfr:.1f}, mantissa {cost[2] / nfr:.1f}, "
              f"fill {cost[3] / nfr:.1f}   "
              f"side info is {(cost[0] + cost[1] + cost[3]) / max(tot, 1) * 100:.0f}% of the file")


if __name__ == "__main__":
    main()
