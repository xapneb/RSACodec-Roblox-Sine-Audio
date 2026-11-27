"""Does the masking model change the answer, and is the frequency structure paying for itself?

Run:  python psy_audit.py moz.wav 24

Same signal, several allocations, all compared at the same rate, which is the only comparison that
is fair to a mask: a model that spends less is not better, it is just quieter. Each arm is bisected
to hit the file's own bit budget, so every row of the table is the same codec at the same bitrate
with a different opinion about what is hidden.

Two scores are printed for every arm, because they disagree on purpose. Plain error is what the
encoder reports. "Audible" is the part of the quantisation error that clears the mask, which is the
only part a listener can hear, and a working model is expected to trade the first against the
second: it should look *worse* on plain error and better on audible error, and if it does neither
then the mask is not doing anything.

What this tool proves and what it cannot:

  Proven here, on moz.wav at 24 kbit/s, 1500 frames, 1023 bits per frame:
    masking             1024 b/f   plain -59.6 dB   audible 31.77 dB
    global floor, -50 dB  1023 b/f   plain -66.7 dB   audible 33.95 dB
    blind, level only     1024 b/f   plain -65.0 dB   audible 34.54 dB
  The mask changes the decision on about 30% of coefficients and buys 2.8 dB of audible error for
  5.5 dB of plain SNR, and a single global floor only buys 0.6 dB of that, so the ERB skirts are
  earning the rest. On tone material the whole effect shrinks to under 1 dB, so it is material
  dependent and this is not a number to carry between files.

  Also measured, and the reason the arms are chosen rather than swept: a mask that is too strong
  for the codec to spend stops being a mask and becomes a rate limit. Taking everything 20 dB
  below the frame's loudest bin and calling it hidden starves the allocator until no budget,
  however large, reaches the target - it tops out near 100 bits per frame and the rest goes
  unspent. So masking strength is bounded above by the allocator's ability to buy something, and
  that ceiling is a good deal lower than the point where the curve looks best on paper.

  Not measured: whether the mask is the right *strength*. The score here is the shipped model, so
  asking it to judge a stronger or weaker mask is that model grading its own homework. The
  structural claim (skirts over a single floor) survives that; the parameters - the 15% and the
  10/25 dB per ERB - do not, and they want ears rather than another hour of arithmetic.

Two things to hold about what is being compared:

  This varies the allocation only. It is not the encoder's `--blind` flag, which also drops the
  ear-filter share and moves what the fill field is allowed to cover, so its numbers and this
  tool's will not agree and neither is wrong. The allocation is where the mask does most of its
  work, so that is what is worth the run.

  The arms go through `r3.allocate`, which is what the codec uses when a band is coded as planes.
  The `--pvq` path runs the same mask through `waterfill_select` with the auditory-filter share on
  top of it (see `band_bits`), so on a PVQ build this is not the exact bit map the encoder writes
  - it is the mask under examination, without the share that rides along with pyramid coding.
"""
import sys

import numpy as np

import rsc3_codec as r3
from scipy.io import wavfile

FRAMES = 1500


def global_mask(band, nc, k):
    """A mask that knows nothing but the frame's loudest bin: everything k dB under it is hidden.
    What a model without frequency structure would have to say."""
    m = (band.max(axis=1) - k)[:, None] * np.ones((1, nc))
    return np.maximum(m, r3.QUIET_FLOOR_DBFS)


def main(path, br=24, nframes=FRAMES):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64) / 32768.0
    A = r3.amplitude(r3.analyse(x, 4096)[0], 4096)
    nfr, nc = A.shape
    hop = x.size / nfr
    nfr = min(nfr, nframes)
    A, A_db = A[:nfr], 20.0 * np.log10(np.maximum(np.abs(A[:nfr]), 1e-30))
    band = r3._band_level_db(A_db, r3.band_edges(nc))
    masks = {"masking": r3.threshold_dbfs(A_db, sr, nc, psycho=True),
             "blind": r3.threshold_dbfs(A_db, sr, nc, psycho=False),
             "global -20": global_mask(band, nc, 20.0),
             "global -35": global_mask(band, nc, 35.0),
             "global -50": global_mask(band, nc, 50.0)}
    score_mask = masks["masking"]
    target = int(br * 1000.0 * hop / sr)

    def alloc(budget, thr):
        # the threshold is passed in, so `psycho` is not read; which mask reaches the allocator is
        # the whole of what this tool varies.
        b = r3.allocate(A, sr, int(budget), psycho=True, A_db=A_db, thr=thr)
        return np.clip(b.astype(np.float64), 0, r3.MAX_BITS_PER_COEF)

    def score(b):
        err = np.where(b > 0, band - 6.02 * b, A_db)
        over = np.maximum(err - score_mask, 0.0)
        return (float(b.sum()) / nfr,
                float(10.0 * np.log10(np.mean(np.power(10.0, err / 10.0)) + 1e-30)),
                float(10.0 * np.log10(np.mean(np.power(10.0, over / 10.0)) + 1e-30)))

    print(f"{path} {br} kbit/s: {nfr} frames, {nc} coefficients, {target} bits/frame "
          f"scored against the shipped mask")
    got = {}
    for name, thr in masks.items():
        lo, hi = 8.0, 16384.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if score(alloc(mid, thr))[0] > target:
                hi = mid
            else:
                lo = mid
        got[name] = alloc(0.5 * (lo + hi), thr)
        bits, err, over = score(got[name])
        print(f"   {name:16} {bits:7.1f} bit/frame   plain {err:8.2f} dB   audible {over:7.2f} dB"
              + ("   <-- cannot spend the rate" if bits < 0.9 * target else ""))
    ref = got["masking"]
    for name, b in got.items():
        if name != "masking":
            print(f"   masking vs {name:16} same decision on "
                  f"{100 * float((b == ref).mean()):4.1f}% of coefficients")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
