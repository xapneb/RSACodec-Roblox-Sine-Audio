"""
rsc3_encode.py - RSC3 encoder CLI.

Writes a real RSC3 bitstream, so for the first time the prototype's rate can be measured
instead of quoted. That matters: 512 bits/frame is what the waterfill spends, and a file that
pays for scalefactors and mantissa signs inside that budget costs a different number. This
tool charges both, inside the budget, and reports what it actually wrote.

Usage:
    python rsc3_encode.py input.wav -o out.rsc3 --bitrate 128
    python rsc3_encode.py input.wav -o out.rsc3 --budget 512 --no-search
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsc3_codec as r3
import rsc3_stream as st

N2 = 4096
WIDTH = r3.BAND_WIDTH


def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    x = raw.astype(np.float64) / 32768.0
    return (x.reshape(-1, ch) if ch > 1 else x.reshape(-1, 1)), sr


def band_bits(A, sr, budget, psycho=True):
    """
    The model's per-coefficient allocation, collapsed to one integer per band: a band pays
    for its loudest bin. Bins below that pay nothing extra, because they land on the band's
    step and round toward zero rather than being skipped by a separate rule.
    """
    nfr, nc = A.shape
    nb = st.band_grid(nc, WIDTH)
    b = r3.allocate(A, sr, budget, psycho=psycho).reshape(nfr, nb, WIDTH).max(axis=2)
    return np.clip(b.astype(np.int64), 0, st.MAX_BITS)


def encode_channel(A, sr, budget, psycho, want_cost=False):
    """
    Returns (payload, recon), where recon is built by the shared reconstruct_bands - the same
    code the decoder will run - so the encoder's quality number is the decoder's, not the
    encoder's opinion of it.
    """
    nfr, nc = A.shape
    nb = st.band_grid(nc, WIDTH)
    bits = band_bits(A, sr, budget, psycho)
    A3 = A.reshape(nfr, nb, WIDTH)
    recon = np.zeros((nfr, nc))
    w = st.Writer()
    prev_E = np.zeros(nb, np.int64)
    prev_b = np.zeros(nb, np.int64)
    cost = np.zeros(3, np.int64) if want_cost else None

    for f in range(nfr):
        b = bits[f]
        E = st.band_exponents(A3[f])
        mant = st.quantise_bands(A3[f], E, b)
        recon[f] = st.reconstruct_bands(E, b, mant).reshape(-1)

        E, field = st.write_frame(w, E, b, mant, prev_E, prev_b, WIDTH)
        if want_cost:
            cost += np.array(field, np.int64)
        prev_E, prev_b = E, b

    if want_cost:
        return w.done(), recon, cost
    return w.done(), recon


def main():
    ap = argparse.ArgumentParser(description="RSC3 encoder (MDCT block transform + waterfill)")
    ap.add_argument("input", help="input .wav")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--bitrate", type=float, default=128.0,
                    help="target kbit/s per channel; the mantissa budget is searched to hit it")
    ap.add_argument("--probe-frames", type=int, default=48,
                    help="frames the rate search samples (fewer = faster, less accurate)")
    ap.add_argument("--budget", type=int, default=512,
                    help="bits/frame spent on mantissas, used as given with --no-search")
    ap.add_argument("--no-search", action="store_true",
                    help="encode at --budget as given and report the measured rate")
    ap.add_argument("--blind", action="store_true",
                    help="disable the psychoacoustic threshold (ATH only)")
    args = ap.parse_args()

    x, sr = read_wav(args.input)
    nch = x.shape[1]
    n_samp = x.shape[0]
    nfr = st.frame_count(n_samp, N2)
    dur = n_samp / sr
    out = args.output or (os.path.splitext(args.input)[0] + ".rsc3")
    print(f"RSC3 encode   {args.input}")
    print(f"   {n_samp} samples  {dur:.2f}s  {sr} Hz  {nch}ch   {nfr} frames   "
          f"block {N2}  {WIDTH}-bin bands  {N2 // 2 // WIDTH} bands")

    A_all = [r3.amplitude(r3.analyse(x[:, c], N2)[0], N2) for c in range(nch)]
    nb = st.band_grid(A_all[0].shape[1], WIDTH)
    stride = max(1, nfr // args.probe_frames)
    A_sub = [A[::stride] for A in A_all]
    nsub = A_sub[0].shape[0]
    frate = sr / (N2 // 2)

    def run(budget):
        payloads, recons, costs = [], [], []
        for A in A_all:
            p, rc, cost = encode_channel(A, sr, budget, psycho=not args.blind,
                                         want_cost=True)
            payloads.append(p)
            recons.append(rc)
            costs.append(cost)
        kbps = sum(len(p) for p in payloads) * 8.0 / dur / 1000.0
        return payloads, recons, kbps, np.sum(costs, axis=0) / nch, [
            abs(int(c.sum()) - len(p) * 8) for c, p in zip(costs, payloads)]

    def estimate_kbps(budget, frames=None):
        """
        kbit/s per channel for a candidate budget, from the geometry of the fields rather
        than from a packed file. bits/frame is monotone in the budget, and each field's size
        is arithmetic, so encoding is not needed to answer this - which is what the 15-try
        search was paying for.
        """
        frames = A_sub if frames is None else frames
        tot = 0.0
        for A in frames:
            n = A.shape[0]
            bits = band_bits(A, sr, budget, psycho=not args.blind)
            A3 = A.reshape(n, nb, WIDTH)
            E = np.array([st.band_exponents(A3[f]) for f in range(n)])
            M = np.array([st.quantise_bands(A3[f], E[f], bits[f]) for f in range(n)])
            tot += float(st.frame_costs(E, bits, M, WIDTH).sum())
        return tot / max(nsub, 1) * frate / 1000.0

    if args.no_search:
        chosen = args.budget
        payloads, recons, kbps, cost, drift = run(chosen)
    else:
        # The target is a rate, so the mantissa budget has to be found. Bisect using the
        # estimator over frames spread across the file, then let the real file have the last
        # word and correct along the measured slope if the estimate was off.
        lo, hi = 1, 1536
        for _ in range(11):
            mid = (lo + hi) // 2
            if estimate_kbps(mid) / nch > args.bitrate:
                hi = mid
            else:
                lo = mid
        chosen = lo
        est = None
        run_for = None
        for _ in range(4):
            if run_for != chosen:
                payloads, recons, kbps, cost, drift = run(chosen)
                run_for = chosen
            got = kbps / nch
            if abs(got - args.bitrate) <= max(0.4, 0.01 * args.bitrate) or chosen <= 1:
                break
            chosen = max(1, int(round(chosen * args.bitrate / max(got, 1e-9))))
        if run_for != chosen:
            payloads, recons, kbps, cost, drift = run(chosen)
        est = estimate_kbps(chosen) / nch

    # quality of the decoded file, measured on the reconstruction the decoder will produce
    lo_i = 3 * N2
    err = 0.0
    sig = 0.0
    for c, rc in enumerate(recons):
        y = r3.synthesise(rc * (N2 / 4.0), N2, n_samp)
        ref = x[lo_i:-N2, c]
        err += float(np.mean((y[lo_i:-N2] - ref) ** 2))
        sig += float(np.mean(ref ** 2))
    es = 10 * math.log10(sig / max(err, 1e-300))

    print(f"   budget {chosen} b/f   scale {cost[0] / nfr:7.1f} b/f   "
          f"bits {cost[1] / nfr:6.1f} b/f   mantissa {cost[2] / nfr:7.1f} b/f   "
          f"(per channel)")
    print(f"   side info is {(cost[0] + cost[1]) / max(cost.sum(), 1) * 100:.0f}% of the file")
    # The fields must add up to the bytes. Attributed bits can only differ from the file by
    # the final byte-flush padding, so a drift over 7 bits means a field went unwritten, or
    # read twice, and no rate number printed below is worth quoting.
    bad = [d for d in drift if d > 7]
    print(f"   field accounting vs bytes on disk: drift {max(drift)} bits "
          f"{'(flush padding only)' if not bad else '*** UNACCOUNTED BITS ***'}")
    if not args.no_search:
        # The estimate is only there to save encodes. When it disagrees with the file, the
        # file wins and the disagreement gets printed, because a silent estimator bias would
        # move every rate number in the same direction and nobody would ever see it.
        bias = (kbps / nch) / max(est, 1e-9) - 1.0
        print(f"   probe-frame estimate {est:7.2f} kbit/s   measured {kbps / nch:7.2f}"
              f"   bias {bias * 100:+5.1f}%")
    h = st.pack_header(sr, nch, N2, WIDTH, n_samp, chosen, [len(p) for p in payloads])
    with open(out, "wb") as f:
        f.write(h)
        for p in payloads:
            f.write(p)
    size = len(h) + sum(len(p) for p in payloads)
    print(f"   {kbps / nch:.2f} kbit/s per channel   {kbps:.2f} total   "
          f"file {size:,} B   ES {es:.3f} dB")
    print(f"   wrote {out}")
    print(f"   decode with: python rsc3_decode.py {out} -o decoded.wav")


if __name__ == "__main__":
    main()
