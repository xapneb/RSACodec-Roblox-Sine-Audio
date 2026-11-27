"""Measure what a codec does to a transient, which is a time question, not an error-score one.

Run:  python transient_audit.py source.wav decoded.wav

Every other number in this repository is computed per frame, in the frequency domain, from the
frame's own spectrum. That is fine until something arrives, because the whole point of an onset is
that it is not spread out over 43 ms, and a per-frame score cannot tell the difference between
"wrong value" and "right value, wrong time". This measures the second thing: where the decoded
energy sits in time, relative to where the source put it.

Three windows, expressed as a fraction of the hit's own energy, so they are readable on any
material. "Before" is the window immediately preceding the onset, which is where pre-echo shows
up; the ear is most exposed to it there, because nothing is masking anything yet. "After" is the
30 to 140 ms that follow, which is where ringing and a smeared release live. Both want to be as
far below zero as the material allows.

Reference numbers, so the readings below mean something. A 6-second click train at 96 kbit/s,
58 attacks found, with the mask as shipped:

    ES 10.70   at the hit -16.9   before -15.1   after -10.5

with --temporal-mask 12 at the same rate:

    ES 16.64   at the hit -27.9   before -21.3   after -16.4

and with --temporal-mask 12 at the same rate. A synthetic click is the hard case: a Dirac has every bin in the file to itself, which is
exactly the frame a spectral mask declares fully masked. Real percussive material reads a few dB
better than these. If "before" rises while "at the hit" falls, the fix is spending its bits in the
wrong place.

Click material is not lying around the repository, so make some:

    python - <<'PY'
    import numpy as np, scipy.io.wavfile as w
    sr, x = 48000, np.zeros(6 * 48000)
    for n in range(1, 60):                 # a hit every 100 ms, with a touch of ring
        i = int(n * 0.1 * sr)
        x[i], x[i + 1] = 0.60, -0.08
    w.write("/tmp/click.wav", sr, (np.tile(x, (2, 1)) * 32767).T.astype(np.int16))
    PY
"""
import sys

import numpy as np
from scipy.io.wavfile import read

PRE, POST = (int(0.040 * 48000), int(0.001 * 48000)), (int(0.030 * 48000), int(0.140 * 48000))


def main(src_path, dec_path, step_s=0.1):
    sr, s = read(src_path)
    _, d = read(dec_path)
    n = min(s.shape[0], d.shape[0])
    s = s[:n].astype(np.float64) / 32768.0
    d = d[:n].astype(np.float64) / 32768.0
    if s.ndim > 1:
        s, d = s.mean(1), d.mean(1)
    e = s - d
    step = int(step_s * sr)
    # An attack is where the signal jumps out of what it was doing, so find it in the source and
    # nothing else: a sample well above what the same moment looked like one millisecond ago, and
    # above a fraction of the file's own level so the noise floor does not qualify as an onset.
    k = max(int(0.001 * sr), 1)
    a = np.abs(s)
    held = np.concatenate([np.zeros(k), a[:-k]])
    cand = np.nonzero((a > 8.0 * held) & (a > 0.05 * a.max()))[0]
    hits, last = [], -step
    for i in cand:
        if i - last > max(int(0.010 * sr), 1) and PRE[0] <= i <= n - POST[1]:
            hits.append(int(i))
            last = i
    spaced = len(hits) >= 8
    if not spaced:
        # Continuous material has no isolated attacks to measure, and pretending otherwise would
        # print a number that looks like a transient result and is really just the steady state.
        hits = list(range(PRE[0], n - POST[1], step))
        print("   no isolated attacks found; these rows describe the steady state, not an onset")
    tot = np.zeros(3)
    tot = np.zeros(3)
    for i in hits:
        t = (s[i - PRE[1]:i + step] ** 2).sum() + 1e-30
        tot[0] += (e[i - PRE[1]:i + PRE[1]] ** 2).sum() / t
        tot[1] += (e[i - PRE[0]:i - PRE[1]] ** 2).sum() / t
        tot[2] += (e[i + POST[0]:i + POST[1]] ** 2).sum() / t
    k = max(len(hits), 1)
    es = 10 * np.log10((s ** 2).sum() / ((e ** 2).sum() + 1e-30))
    print(f"{src_path} -> {dec_path}: {len(hits)} hits measured   {"(spaced attacks)" if spaced else "(interval sweep)"}   ES {es:6.2f} dB")
    print(f"   at the hit {10*np.log10(tot[0]/k+1e-30):7.1f} dB   "
          f"in the 40 ms BEFORE {10*np.log10(tot[1]/k+1e-30):7.1f} dB   "
          f"in the 30-140 ms AFTER {10*np.log10(tot[2]/k+1e-30):7.1f} dB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
