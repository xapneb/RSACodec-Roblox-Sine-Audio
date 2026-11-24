"""
Where do the bits actually go, and where does the audible error? Frequency
table per band GROUP, banded psycho mode, cursedseptember @ 512 b/f - the case
whose spectrogram you just read.

If high bands get ~fair bits but error says otherwise, the problem is steps or
thresholds, not allocation. If bits-per-Hz collapses with frequency, it's the
waterfill bias and the fix lives in the bit model. Numbers per group:

  bits %      share of paid mantissa bits landing in this group
  sig %       share of signal power (the fair share to compare bits against)
  err %       share of TOTAL error
  auderr %    share of error that sits ABOVE threshold (what you hear)
  eff         auderr%/bits% - 1.0 = paying exactly your share of damage,
              2.0 = double damage per bit, 0.5 = half.

The fix is whatever column says: bits% low -> allocation; eff high -> mask
shape or band width. One table decides the investment.
"""
import sys, wave, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
import numpy as np
import rsc3_codec as r3

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LAB, "cursedseptember.wav")
N2, hop, inv = 4096, 2048, 1024.0
BUDGET = 512

info = wave.open(src, "rb")
sr, nch = info.getframerate(), info.getnchannels()
pcm = np.frombuffer(info.readframes(info.getnframes()), np.int16).astype(np.float64)
pcm = pcm.reshape(-1, nch) / 32768.0
info.close()

# group coefficients by coarse regions, bin k -> k*sr/N2 Hz (A rows = coeffs)
groups = [("< 0.5k", 0, 0.5e3), ("0.5-2k", 0.5e3, 2e3), ("2-5k", 2e3, 5e3),
          ("5-10k", 5e3, 10e3), ("10-14k", 10e3, 14e3), ("14-18k", 14e3, 18e3),
          ("> 18k", 18e3, 24e3)]

acc = {g: np.zeros(4) for g, _, _ in groups}   # bits, sig, err, auderr
for ch in range(nch):
    X, _ = r3.analyse(pcm[:, ch], N2)
    A = r3.amplitude(X, N2)
    A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
    bits = r3.allocate(A, sr, BUDGET, psycho=True)
    q = r3.quantise(A, bits)
    y = r3.synthesise(q * inv, N2, len(pcm))
    E = r3.amplitude(r3.analyse(y - pcm[:, ch], N2)[0], N2) ** 2
    THR = np.array([10 ** (0.1 * r3.threshold_dbfs(A_db[i], sr, A.shape[1], True))
                    for i in range(A.shape[0])])
    sl = slice(3, A.shape[0] - 1)
    freqs = np.arange(A.shape[1]) * sr / N2
    for g, lo, hi in groups:
        cols = (freqs >= lo) & (freqs < hi)      # 1-D over coefficients
        b_, s_, e_, t_ = bits[sl], A[sl], E[sl], THR[sl]
        acc[g][0] += float(b_[:, cols].sum())
        acc[g][1] += float((s_[:, cols] ** 2).sum())
        acc[g][2] += float(e_[:, cols].sum())
        acc[g][3] += float(np.maximum(0.0, e_[:, cols] - t_[:, cols]).sum())

tot = [sum(a[i] for a in acc.values()) for i in range(4)]
print(f"cursedseptember {nch}ch, psycho banded, {BUDGET} b/f   (23.4 Hz/bin)")
print(f"{'group':8s} {'bits %':>7} {'sig %':>7} {'err %':>7} {'auderr %':>9}   {'eff':>5}   "
      f"{'bits/bin':>9}")
for g, _, _ in groups:
    bits_p, sig_p, err_p, aud_p = (100 * acc[g][i] / max(tot[i], 1e-300) for i in range(4))
    eff = aud_p / max(bits_p, 1e-9)
    print(f"{g:8s} {bits_p:6.1f} {sig_p:6.1f} {err_p:6.1f} {aud_p:9.1f}   {eff:5.2f}")
