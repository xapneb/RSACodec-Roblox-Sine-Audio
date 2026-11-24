"""
A/B sample generator - parameterized successor of cs_ab.py.

Usage: python ab_codec.py <input.wav> [tag]
  <tag> defaults to the file's stem; outputs land next to the codec test lab
  as <tag>_rsc3_psy.wav / <tag>_rsc3_blind.wav / <tag>_rsc2_mp8.wav

Everything this session learned is baked in: float32 writers (never clip),
ceiling-folded headroom gain, budget-matched twins for the mask comparison,
and raw-ES + audible-error both reported so the numbers come with their units.
"""
import sys, wave, os, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
sys.path.insert(0, LAB)
import numpy as np
from scipy.io import wavfile
import rsc3_codec as r3

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LAB, "ij.wav")
tag = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(src))[0]
out_dir = LAB
N2, hop, inv = 4096, 2048, 1024.0
BUDGET = 512
py = sys.executable

info = wave.open(src, "rb")
sr, nch = info.getframerate(), info.getnchannels()
pcm = np.frombuffer(info.readframes(info.getnframes()), np.int16).astype(np.float64)
pcm = pcm.reshape(-1, nch) / 32768.0
info.close()
print(f"{os.path.basename(src)} {sr} Hz {nch} ch {pcm.shape[0]/sr:.1f} s  "
      f"peak {np.abs(pcm).max():.4f}")


MIN_E = 10 ** (-11.0)   # -110 dBFS floor for noise-fill eligibility


def stats_and_write(name, y_raw):
    """ES/audible measured on the UN-GAINED synthesis (the headroom gain belongs
    in the file, never in the error - that bug made twins with different peaks
    comparable by artifact, once).

    Both ES flavours are printed. Naive charges the codec for any coherent level
    offset, which caps ES at whatever that offset allows (a pure 6 dB level loss
    caps ES at 6 dB even for perfect coding) and made the twins comparable by
    artifact. Fitted solves for the least-squares gain first and measures what
    is left, so it isolates the error the codec cannot fix by being turned up.
    """
    n = min(y_raw.shape[0], pcm.shape[0])
    y = y_raw[:n]
    pk = float(np.abs(y).max())
    g = min(1.0, 1.0 / pk)
    lo = 3 * N2
    x = pcm[lo:n - N2, 0]
    e = y[lo:n - N2, 0]
    naive = 10 * np.log10((x ** 2).mean() / max(((x - e) ** 2).mean(), 1e-300))
    k = float((x * e).mean() / max((e * e).mean(), 1e-300))
    fit = 10 * np.log10((x ** 2).mean() / max(((x - k * e) ** 2).mean(), 1e-300))
    wavfile.write(os.path.join(out_dir, f"{tag}_{name}.wav"),
                  sr, (y * g).astype(np.float32))
    return pk, g, naive, fit


for name, psy, fill in (("rsc3_psy", True, False), ("rsc3_blind", False, False),
                        ("rsc3_psy_nf", True, True)):
    aud_tot, frames, bpf_tot = 0.0, 0, 0.0
    chans = []
    ed = r3.band_edges(2048)
    for ch in range(nch):
        rng = np.random.default_rng(0xC0FFEE + ch)
        X, _ = r3.analyse(pcm[:, ch], N2)
        A = r3.amplitude(X, N2)
        A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
        bits = r3.allocate(A, sr, BUDGET, psycho=psy)
        q = r3.quantise(A, bits)
        if fill:
            for j in range(len(ed) - 1):
                lo_, hi_ = int(ed[j]), int(ed[j + 1])
                dead = (np.abs(q[:, lo_:hi_]) == 0).all(axis=1) & \
                       ((A[:, lo_:hi_] ** 2).mean(axis=1) > MIN_E)
                bpf_tot += int(dead.sum()) / A.shape[0]
                if dead.any():
                    n = hi_ - lo_
                    z = rng.standard_normal((int(dead.sum()), n))
                    z /= np.sqrt((z * z).mean(axis=1, keepdims=True))
                    q[dead, lo_:hi_] = z * np.sqrt((A[dead][:, lo_:hi_] ** 2).mean(axis=1))[:, None]
        y = r3.synthesise(q * inv, N2, len(pcm))
        chans.append(y)
        E = r3.amplitude(r3.analyse(y - pcm[:, ch], N2)[0], N2) ** 2
        THR = np.array([10 ** (0.1 * r3.threshold_dbfs(A_db[i], sr, A.shape[1], True))
                        for i in range(A.shape[0])])
        sl = slice(3, A.shape[0] - 1)
        aud_tot += float(np.maximum(0.0, E[sl] - THR[sl]).sum())
        frames += sl.stop - 3
    pk, g, naive, fit = stats_and_write(name, np.stack(chans, axis=1))
    aud = 10 * np.log10(aud_tot / frames / A.shape[1] / nch + 1e-300)
    fb = 6 * bpf_tot / nch if fill else 0.0
    print(f"  {name:12s} ES {naive:6.2f} naive / {fit:6.2f} fitted   "
          f"audible {aud:7.1f} dBFS   "
          f"raw peak {pk:.3f} (level-match {20*np.log10(g):+.1f} dB)   "
          f"~{BUDGET + fb:.0f} b/f{f' incl fill +{fb:.0f}' if fill else ''}   "
          f"{out_dir}\\{tag}_{name}.wav")

# production codec, real CLI, two-pass ceiling
tmp = os.path.join(tempfile.gettempdir(), f"{tag}_rsc2.rsc2")
env = dict(os.environ, PYTHONIOENCODING="utf-8")
subprocess.run([py, os.path.join(LAB, "rsc_encoder.py"),
                src, tmp, "--max-partials", "8"], check=True, env=env,
               stdout=subprocess.DEVNULL)
import rsc_decoder
res = rsc_decoder.decode(open(tmp, "rb").read())
y2 = res[0] if res[0].ndim == 2 else res[0][:, None]
pk, g, naive, fit = stats_and_write("rsc2_mp8", y2)
kbps = 8 * os.path.getsize(tmp) / (pcm.shape[0] / sr) / 1000
print(f"  {'rsc2_mp8':12s} ES {naive:6.2f} naive / {fit:6.2f} fitted   "
      f"raw peak {pk:.3f} (level-match {20*np.log10(g):+.1f} dB)   "
      f"{kbps:.0f} kbit/s   {out_dir}\\{tag}_rsc2_mp8.wav")
os.remove(tmp)
print("\nFloat files, ceiling-capped. rsc3 twins are bitrate-equal by budget; "
      "rsc2_mp8 is the shipping reference. Judge by the FITTED ES and the "
      "audible ledger; naive ES still charges for level.")
