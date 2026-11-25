"""
The one ruler that agrees with listening, measured through files, for the two fill arms.

Flat fill versus shaped fill is a question about a noise floor above 14 kHz, and gain-fitted ES
structurally prefers silence, so it is not evidence (see README, "Rulers"). What is left is the
audible ledger: sum max(0, E_bin - THR_bin) over the ERROR spectrum, where THR is the mask of the
ORIGINAL, shared by both arms so neither is graded against its own curve.

This goes through the CLIs, because a harness that re-implements the decoder can be self-consistent
and still wrong. Same file, same target, one bit apart.
"""
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
PY = sys.executable
sys.path.insert(0, HERE)
import rsc3_codec as r3

N2 = 4096


def read_wav(p):
    w = wave.open(p, "rb")
    sr = w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
    w.close()
    return raw.reshape(-1, 2), sr


ARMS = (("flat", []), ("shape", ["--fill-shape"]))
print(f"{'file':>10} {'arm':>6} {'kbit/s':>8} {'fit ES':>8} {'audible dB':>11}   "
      f"{'audible frac':>12}")
for src in ("cam15.wav", "ij15.wav"):
    for target in (10.0,):
        for arm, extra in ARMS:
            tmp, wav = "_sa.rsc3", "_sa.wav"
            subprocess.run([PY, "-u", "-B", os.path.join(".", "rsc3_encode.py"),
                            os.path.join(LAB, src), "-o", tmp, "--bitrate", str(target), *extra],
                           cwd=HERE, capture_output=True, text=True, check=True)
            subprocess.run([PY, "-u", "-B", os.path.join(".", "rsc3_decode.py"), tmp, "-o", wav],
                           cwd=HERE, capture_output=True, text=True, check=True)
            x, sr = read_wav(os.path.join(LAB, src))
            y, _ = read_wav(wav)
            size = os.path.getsize(tmp)
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            lo = 3 * N2
            g = (float(np.dot(x[lo:-N2, 0], y[lo:-N2, 0]))
                 / max(float(np.dot(y[lo:-N2, 0], y[lo:-N2, 0])), 1e-300))
            es = 10 * np.log10(float(np.mean(x[lo:-N2, 0] ** 2))
                               / max(float(np.mean((x[lo:-N2, 0] - g * y[lo:-N2, 0]) ** 2)),
                                     1e-300))
            A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
            nf, nc = A.shape
            THR = np.array([10 ** (0.1 * r3.threshold_dbfs(
                20 * np.log10(np.maximum(np.abs(A[i]), 1e-30)), sr, nc, True))
                for i in range(nf)])
            E = r3.amplitude(r3.analyse(y[:, 0] - x[:, 0], N2)[0], N2) ** 2
            sl = slice(3, nf - 1)
            aud = float(np.maximum(0.0, E[sl] - THR[sl]).sum())
            print(f"{src:>10} {arm:>6} "
                  f"{size * 8 / (n / sr) / 1000.0 / 2:8.2f} {es:8.2f} "
                  f"{10 * np.log10(aud / ((nf - 4) * nc) + 1e-300):11.1f}   "
                  f"{100 * aud / max(float(E[sl].sum()), 1e-300):11.1f}%")
            for f in (tmp, wav):
                if os.path.exists(f):
                    os.remove(f)
        print()
