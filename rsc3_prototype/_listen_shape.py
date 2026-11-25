"""Listening pairs for the shape ladder: flat, blk16, bins. The rulers are blind on this axis
(ES and audible error agree to a decimal across arms), so the decision has to be made by ear.
The audio is the arm's own decode path: reconstruct with flat fill, overlay shaped levels, then
synthesise the way the score does - same X the rate and density columns were measured on."""
import os
import sys
import wave

import numpy as np

import vector_hits as vh
import _shaped_hf as sh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "listening")
os.makedirs(OUT, exist_ok=True)

name = "cam15.wav"
x, sr = vh.read_wav(os.path.join(vh.LAB, name))
rows, base = vh.arm_scalar(x, sr, (32,))
print("control rates:")
for r in rows:
    print(f"  scalar {r[0]}  {r[1]:.2f} kbit/s")

for mode, G, tag in (("flat", 1, "flat"), ("block", 16, "blk16"), ("bins", 1, "bins")):
    # reuse arm_shape's X by running it and rebuilding X identically, cheaply, for the wav
    A3, nfr, nb, Ad, thr3, Eall = base[:6]
    band_hz = base[8].reshape(nb, vh.W).mean(axis=1)
    reg = band_hz >= 14e3
    thr2 = thr3.copy()
    thr2[:, reg] = 991.0
    bl2 = np.maximum(vh.r3._band_level_db(Ad, vh.r3.band_edges(nb * vh.W)), thr2.reshape(nfr, -1))
    bits = np.clip(vh.r3.waterfill_select(Ad, thr2.reshape(nfr, -1), bl2, 32)
                   .reshape(nfr, nb, vh.W).max(axis=2), 0, vh.st.MAX_BITS).astype(np.int64)
    mant = np.array([vh.st.quantise_bands(A3[f], Eall[f], bits[f]) for f in range(nfr)])
    X = np.zeros((nfr, nb, vh.W))
    for f in range(nfr):
        fl = vh.st.run_fill_words(vh.st.band_fill_cap(A3[f], thr3[f]), bits[f], "safe")
        X[f] = vh.st.reconstruct_bands(Eall[f], bits[f], mant[f], fl, f, vh.W)
        if mode != "flat":
            lv, _ = sh.shape_levels(vh.st.band_fill_cap(A3[f], thr3[f]), bits[f], reg, mode, G)
            for j, lvl in lv.items():
                X[f][int(j)] = vh.st.fill_noise(int(j), f, vh.W) * 10.0 ** (0.05 * lvl)
    y = vh.r3.synthesise(X.reshape(nfr, -1) * (vh.N2 / 4.0), vh.N2, x.shape[0])
    y = np.clip(y, -1.0, 1.0)
    out = os.path.join(OUT, f"{name[:-4]}_shape_{tag}.wav")
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((y * 32767.0).astype(np.int16).tobytes())
    print(f"  {tag:>6}  ->  {out}")
print("\nA/B by ear: does blk16 sound like treble returned, or like hiss? The rulers already gave up.")
