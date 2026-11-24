"""
The bitrate sweep, rescored in audible-error units.

Same file (audio.wav mono), same budgets, same two modes. Only the ruler changes:
per frame, sum max(0, E_bin - THR_bin) over the ERROR spectrum E, where THR is
the mask of the ORIGINAL signal with psycho=True (shared model, so neither mode
is graded on its own allocation curve). Audible dB = log10 of that sum per
coefficient-frame, like an error noise floor you could actually hear.

Raw ES stays in the table for the record: the point is to show what the old
table was hiding, not to pretend it's meaningless.
"""
import sys, wave, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
import numpy as np, rsc3_codec as r3

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LAB, "audio.wav")
with wave.open(path, "rb") as wf:
    sr, nch = wf.getframerate(), wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
pcm = np.frombuffer(raw, np.int16).astype(np.float64)
pcm = (pcm.reshape(-1, nch) if nch > 1 else pcm[:, None])[:, 0] / 32768.0
N2, hop, inv = 4096, 2048, 1024.0
X, _ = r3.analyse(pcm, N2)
A = r3.amplitude(X, N2)
A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
edges = r3.band_edges(X.shape[1])
nf = A.shape[0]

# shared threshold, computed once: mask of the original, psycho gates on
THR = np.zeros_like(A_db)
for i in range(nf):
    THR[i] = 10 ** (0.1 * r3.threshold_dbfs(A_db[i], sr, A.shape[1], True))

lo = 3 * N2
print(f"audio.wav mono {nf} frames  audible-error sweep (THR = mask of original)")
print(f"{'budget':>7}   {'raw ES':>13}   {'audible dB':>13}   {'audible frac':>13}   mode")
for budget in (2048, 1024, 512, 384, 256, 192, 128, 96, 64):
    for psycho in (True, False):
        bits = r3.allocate(A, sr, budget, psycho=psycho, edges=edges)
        q = r3.quantise(A, bits, edges)
        y = r3.synthesise(q * inv, N2, len(pcm))
        E = r3.amplitude(r3.analyse(y - pcm, N2)[0], N2) ** 2
        sl = slice(3, nf - 1)
        aud = float(np.maximum(0.0, E[sl] - THR[sl]).sum())
        tot = float(E[sl].sum())
        es = 10 * np.log10((pcm[lo:-N2] ** 2).mean() / ((y[lo:-N2] - pcm[lo:-N2]) ** 2).mean())
        ad = 10 * np.log10(aud / ((nf - 4) * E.shape[1]) + 1e-300)
        print(f"{budget:7d}   {es:6.2f} dB   {ad:10.1f} dB   {100*aud/max(tot,1e-300):10.1f}%   "
              f"psycho={'on' if psycho else 'off '}")
