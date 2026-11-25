"""
What RSC2 actually costs, measured from the bytes a decoder has to read.

Everything in this repo's README used to quote RSC2 from the encoder's model of its own frame.
That is fine for debugging and worthless for a comparison, because the model leaves out the
parts the container has to write anyway. This runs the shipping encoder over 15 s fixtures,
decodes the files it writes, counts the bytes on disk, and scores the decoded audio with the
same ruler rate_curve.py uses for RSC3. So both sides of "5.2 dB better at 1.4x lower rate" come
from the same kind of measurement, and if that claim is wrong this will say so.

Run:  python _rsc2_curve.py
"""
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LAB, "rsc3_prototype"))
import rsc3_codec as r3

PY = sys.executable
N2 = 4096


def read_wav(p):
    w = wave.open(p, "rb")
    sr = w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
    w.close()
    return raw.reshape(-1, 2), sr


def run(*cmd):
    # the RSC2 encoder prints emoji, which dies on a cp1252 pipe, so force UTF-8 on the child's
    # stdout. Without this the encode "fails" with a UnicodeEncodeError and no codec ran.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=os.path.join(LAB, "rsc2_testing_lab"), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr)[-800:])
    return p.stdout


def es(x, y, n2, fit=True):
    lo, hi = 3 * n2, min(len(x), len(y)) - n2
    num = den = 0.0
    for c in range(x.shape[1]):
        a, b = x[lo:hi, c], y[lo:hi, c]
        num += float(np.mean(a * a))
        if fit:
            g = float(np.dot(a, b)) / max(float(np.dot(b, b)), 1e-300)
            den += float(np.mean((a - g * b) ** 2))
        else:
            den += float(np.mean((a - b) ** 2))
    return 10 * np.log10(max(float(num), 1e-300) / max(float(den), 1e-300))


print(f"{'file':>10} {'mp':>4} {'kbit/s/ch':>10}   {'ES':>7} {'gain-fitted':>12}   "
      f"{'bytes':>9}   {'peak':>7}")
for stem in ("cam15", "ij15"):
    src = os.path.join(LAB, "rsc2_testing_lab", f"{stem}.wav")
    x, sr = read_wav(src)
    dur = x.shape[0] / sr
    for mp in (4, 6, 8, 12, 16):
        rsc = f"_rc_{stem}_{mp}.rsc"
        wav = f"_rc_{stem}_{mp}.wav"
        # both children run in rsc2_testing_lab, so their relative outputs land there and must be
        # read from there too, whatever this script's own cwd is.
        ab = {f: os.path.join(LAB, "rsc2_testing_lab", f) for f in (rsc, wav)}
        run(PY, "rsc_encoder.py", os.path.relpath(src, os.path.join(LAB, "rsc2_testing_lab")),
            rsc, "--version", "3", "--max-partials", str(mp))
        run(PY, "rsc_decoder.py", rsc, wav)
        y, _ = read_wav(ab[wav])
        size = os.path.getsize(ab[rsc])
        print(f"{stem:>10} {mp:>4} {size * 8 / dur / 1000.0 / 2:10.2f}   "
              f"{es(x, y, N2, False):7.2f} {es(x, y, N2, True):12.2f}   "
              f"{size:>9,}   {float(np.abs(y).max()):7.4f}")
        for f in (rsc, wav):
            os.remove(os.path.join(LAB, "rsc2_testing_lab", f))
    print()
