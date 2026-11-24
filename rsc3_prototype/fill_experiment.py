"""
Fill at the SAME bitrate: what does the air cost, and which word pays for it?

The version you loved spends ~858 b/f because every dead band pays a 6-bit
energy word and the fill is free side info. This is the honest version:
fill is billed FIRST, the mantissa gets 512 minus the bill, and the spectrum
has to survive on the scraps. That is the only comparison that could ever
justify shipping the trick.

Two representations, same budget:
  band  - one energy word per dead band.  Accurate, expensive: bill = 6*bands.
  run   - one energy word per contiguous run of dead bands, flat floor across
          the run.  Cheap: bill = 6*runs.  The risk is that bands inside a run
          disagree about their own energy, so the floor lands wrong on part of
          the run.  The internal spread number says whether that's a risk.

Everything else is fixed: same file, same 512 b/f, same psycho thresholds,
same metric. Three arms, and the lie detector is the 16-19 kHz column:
  nofill  -> the hole you spotted            (~ -158 dBFS)
  band    -> dense, but mantissa-starved     (this is what free fill was doing)
  run     -> dense AND 512 b/f               (the candidate)
"""
import sys, wave, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
import rsc3_codec as r3

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LAB, "cam.wav")
tag = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(src))[0]
out_dir = LAB
N2, hop, inv = 4096, 2048, 1024.0
BUDGET = 512
MIN_E = 10 ** (-11.0)
FB = 6
ed = r3.band_edges(2048)
NB = len(ed) - 1

info = wave.open(src, "rb")
sr, nch = info.getframerate(), info.getnchannels()
pcm = np.frombuffer(info.readframes(info.getnframes()), np.int16).astype(np.float64)
pcm = pcm.reshape(-1, nch) / 32768.0
info.close()
hz = np.arange(2048) * sr / N2


def analyse_ch(ch):
    X, _ = r3.analyse(pcm[:, ch], N2)
    return r3.amplitude(X, N2)


def band_energy(A):
    """mean power per (frame, band) - and since bands are equal width, this is
    also the mean power over the run's bin range, which is what makes the
    run-based floor exact to compute."""
    return np.stack([(A[:, int(ed[j]):int(ed[j + 1])] ** 2).mean(axis=1) for j in range(NB)], 1)


def dead_map(A, q):
    d = np.zeros((A.shape[0], NB), bool)
    for j in range(NB):
        lo_, hi_ = int(ed[j]), int(ed[j + 1])
        d[:, j] = (np.abs(q[:, lo_:hi_]) == 0).all(axis=1) & (band_energy(A)[:, j] > MIN_E)
    return d


def run_ids(d):
    """run id per (frame, band); 0 means 'not dead'."""
    starts = d.copy()
    starts[:, 1:] = d[:, 1:] & ~d[:, :-1]
    starts[:, 0] = d[:, 0]
    return np.cumsum(starts, axis=1)


def spread_of_runs(d, be):
    """how much the dead bands inside one run disagree, in dB (max/min power)."""
    rid = run_ids(d)
    mx = np.zeros(d.shape)
    mn = np.full(d.shape, np.inf)
    for j in range(NB):
        m = d[:, j]
        if not m.any():
            continue
        r = rid[m, j]
        np.maximum.at(mx, (np.arange(d.shape[0])[m], r), be[m, j])
        np.minimum.at(mn, (np.arange(d.shape[0])[m], r), be[m, j])
    v = mx[rid > 0] / np.maximum(mn[rid > 0], 1e-30)
    return 20 * np.log10(np.maximum(v, 1.0))


def arm(name, mant, fill):
    chans, aud_tot, frames, paid = [], 0.0, 0, 0.0
    for ch in range(nch):
        rng = np.random.default_rng(0xC0FFEE + ch)
        A = analyse_ch(ch)
        A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
        be = band_energy(A)
        b = BUDGET
        if fill is not None:
            for _ in range(2):
                d = dead_map(A, r3.quantise(A, r3.allocate(A, sr, b, psycho=True)))
                rid = run_ids(d)
                nf = float(d.sum(axis=1).mean()) if fill == "band" else \
                     float(rid.max(axis=1)[rid.max(axis=1) > 0].mean())
                b = max(64, BUDGET - FB * nf)
            bill = FB * nf
        else:
            bill = 0.0
        bits = r3.allocate(A, sr, b, psycho=True)
        q = r3.quantise(A, bits)
        if fill is not None:
            d = dead_map(A, q)
            rid = run_ids(d)
            if fill == "band":
                for j in range(NB):
                    m = d[:, j]
                    if not m.any():
                        continue
                    w = int(ed[j + 1]) - int(ed[j])
                    z = rng.standard_normal((int(m.sum()), w))
                    z /= np.sqrt((z * z).mean(axis=1, keepdims=True))
                    q[m, int(ed[j]):int(ed[j + 1])] = z * np.sqrt(be[m, j])[:, None]
                    paid += m.sum() * FB / d.shape[0]
            else:
                tot = np.zeros((A.shape[0], int(rid.max()) + 1))
                np.add.at(tot, (np.arange(A.shape[0])[:, None], rid), be)
                cnt = np.zeros_like(tot)
                np.add.at(cnt, (np.arange(A.shape[0])[:, None], rid), 1.0)
                run_e = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
                flat = np.take_along_axis(run_e, rid, 1)          # per (frame,band)
                for j in range(NB):
                    m = d[:, j]
                    if not m.any():
                        continue
                    w = int(ed[j + 1]) - int(ed[j])
                    z = rng.standard_normal((int(m.sum()), w))
                    z /= np.sqrt((z * z).mean(axis=1, keepdims=True))
                    q[m, int(ed[j]):int(ed[j + 1])] = z * np.sqrt(flat[m, j])[:, None]
                    paid += FB / d.shape[0] if False else 0
        paid = bill if fill else 0.0
        y = r3.synthesise(q * inv, N2, len(pcm))
        chans.append(y)
        E = r3.amplitude(r3.analyse(y - pcm[:, ch], N2)[0], N2) ** 2
        THR = np.array([10 ** (0.1 * r3.threshold_dbfs(A_db[i], sr, 2048, True))
                        for i in range(A.shape[0])])
        sl = slice(3, A.shape[0] - 1)
        aud_tot += float(np.maximum(0.0, E[sl] - THR[sl]).sum())
        frames += sl.stop - 3
    y = np.stack(chans, axis=1)
    pk = float(np.abs(y).max())
    g = min(1.0, 1.0 / pk)
    n = min(y.shape[0], pcm.shape[0])
    lo = 3 * N2
    es = 10 * np.log10((pcm[lo:n - N2, 0] ** 2).mean() /
                       max(((y[lo:n - N2, 0] - pcm[lo:n - N2, 0]) ** 2).mean(), 1e-300))
    aud = 10 * np.log10(aud_tot / frames / 2048 / nch + 1e-300)
    _, P = welch(y[:2 ** 24, 0], fs=sr, nperseg=65536, detrend=False)
    f = np.arange(len(P)) * sr / 65536

    def cell(lo_, hi_):
        m = (f >= lo_) & (f < hi_)
        return 10 * np.log10(max(P[m].mean(), 1e-300))
    total = b + paid
    print(f"{name:11s} mantissa {b:4.0f} + fill {paid:3.0f} = {total:4.0f} b/f   "
          f"ES {es:6.2f}   audible {aud:7.1f}   16-17k {cell(16e3,17e3):7.1f}   "
          f"18-19k {cell(18e3,19e3):7.1f}   (level-match {20*np.log10(g):+.1f} dB)")
    wavfile.write(os.path.join(out_dir, f"{tag}_{name}.wav"), sr, (y * g).astype(np.float32))
    print(f"            -> {tag}_{name}.wav")


print(f"{tag}: {nch} ch, {pcm.shape[0] / sr:.0f} s, 512 b/f budget, "
      f"{NB} bands of 16 bins\n")
A0 = analyse_ch(0)
q0 = r3.quantise(A0, r3.allocate(A0, sr, BUDGET, psycho=True))
d0 = dead_map(A0, q0)
rid0 = run_ids(d0)
sp = spread_of_runs(d0, band_energy(A0))

# Run lengths, counted by hand from the map. This exists because a flat spread
# number is exactly the shape a broken accumulator makes, and the whole price
# of the run-based word depends on whether multi-band runs even exist.
hist = {}
for i in range(0, d0.shape[0], 5):
    j = 0
    while j < NB:
        if d0[i, j]:
            j2 = j
            while j2 + 1 < NB and d0[i, j2 + 1]:
                j2 += 1
            hist[j2 - j + 1] = hist.get(j2 - j + 1, 0) + 1
            j = j2 + 1
        else:
            j += 1
multi = sum(v for k, v in hist.items() if k > 1)
print(f"dead bands {float(d0.sum(axis=1).mean()):.1f}/frame  "
      f"runs {float(rid0.max(axis=1)[rid0.max(axis=1) > 0].mean()):.1f}/frame   "
      f"run lengths (sampled 1/{5}): " + "  ".join(f"{k}x:{v}" for k, v in sorted(hist.items())[:10])
      + f"\n  runs with >1 band: {100 * multi / max(sum(hist.values()), 1):.0f}%"
      f"   singletons: {100 - 100 * multi / max(sum(hist.values()), 1):.0f}%")
print(f"energy spread inside a run (all runs): p50 {np.percentile(sp, 50):4.1f} dB   "
      f"p90 {np.percentile(sp, 90):4.1f} dB   p99 {np.percentile(sp, 99):4.1f} dB   "
      f"max {sp.max():4.1f}\n")
_, P0 = welch(pcm[:2 ** 24, 0], fs=sr, nperseg=65536, detrend=False)
f = np.arange(len(P0)) * sr / 65536
m = (f >= 16e3) & (f < 19e3)
print(f"{'source':11s}                                              "
      f"16-19k {10*np.log10(P0[m].mean()):7.1f}\n")
arm("nofill", None, None)
arm("bandfill", None, "band")
arm("runfill", None, "run")
