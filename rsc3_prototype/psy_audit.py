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

Read the scale before the numbers. Lower "audible" is better, and the bottom of what is available
is 43.7 dB, which is what this codec scores if it codes nothing at all. A row read without that
anchor says "4 dB" and turns out to be a third of the range.

On moz.wav at 24 kbit/s, 1023 bits per frame:

    coding nothing       0 b/f   plain -50.0 dB   audible 43.71   <- nothing coded, the ceiling
    masking           1024 b/f   plain -59.6 dB   audible 31.77   <- what ships
    blind             1024 b/f   plain -65.0 dB   audible 34.54
    floor -75         1023 b/f   plain -67.6 dB   audible 33.82
    peak -20           104 b/f   plain -58.8 dB   audible 34.78   <- cannot spend the rate

The mask changes the decision on about a third of the coefficients and buys 2.8 dB of audible error
for 5.5 dB of plain SNR, which is the trade a mask exists to make. Tone material shrinks all of it
below a decibel, so this is not a number to carry from one file to another.

Two corrections are written into the arms themselves.

The arms first shipped under the name "global" were peak-relative, which is the wrong way to write
a mask with no frequency structure: one loud bin in a frame buys a threshold that covers the frame's
own floor and nothing else, so that arm coded about two percent of what it was asked to and landed
closer to silence than to the codec. They are called "peak" now, for what they are. The honest
arm without frequency structure is an absolute floor, and that one is better than it first looks.

The other one is about rate, and it is the reason this tool exists. "Cannot spend the rate" is not
a quirk of one over-aggressive mask. An absolute floor saturates too, for a reason that has nothing
to do with how loud the mask is set: the price ladder only holds so many prices above that floor,
so at 96 kbit/s a -75 arm hands back 1778 of 4094 bits and stops improving on its own, while the
shipped mask spends every bit and reaches 14.02 dB against the same ceiling. What lets the shipped
mask scale is the part of it that falls away - the quiet tail where no masker reaches and the
threshold drops toward -120 dBFS, giving the ladder somewhere left to buy. A model with a tidy flat
floor would look fine at 24 kbit/s and quietly give up on every rate above it.

Not measured, and not measurable this way: whether the mask is the right *strength*. The score is
the shipped model, so any stronger or weaker arm is graded by the thing under test. The structural
claim, that skirts beat any single level, stands; the 15 percent and the 10 and 25 dB per ERB want
listening rather than another hour of arithmetic.

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


def peak_mask(band, nc, k):
    """Everything k dB under the frame's loudest bin is hidden. This is what a model without
    frequency structure looks like when it is written the obvious way, and the obvious way is
    wrong: it scales with the loudest sample, so a single loud bin buys the frame a mask that
    covers nothing the frame actually contains, and it is the reason the -20 and -35 rows below
    mute the file rather than coding it. A model that cannot spend the rate is not a mask, it is
    an off switch."""
    m = (band.max(axis=1) - k)[:, None] * np.ones((1, nc))
    return np.maximum(m, r3.QUIET_FLOOR_DBFS)


def floor_mask(band, k):
    """A global model that is not absurd: a level that does not move with the frame's peak, which
    is what a quiet-audio floor actually is. This one can spend, so it is the fair thing to compare
    the skirts against - a mask that knows about loudness and nothing about frequency."""
    return np.full(band.shape, float(k))


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
             "peak -20": peak_mask(band, nc, 20.0),
             "peak -50": peak_mask(band, nc, 50.0),
             "floor -60": floor_mask(band, -60.0),
             "floor -75": floor_mask(band, -75.0)}
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
    mute = np.zeros_like(A_db)
    bits, err, over = score(mute)
    print(f"   coding nothing     {bits:7.1f} bit/frame   plain {err:8.2f} dB   audible "
          f"{over:7.2f} dB   <- the whole signal is the error, this is the top of the scale")
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
