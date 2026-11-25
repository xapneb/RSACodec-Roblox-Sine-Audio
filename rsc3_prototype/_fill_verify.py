"""
Is the noise a listener hears, in the file that ships, below the mask?

Everything measured so far has been the encoder's own reconstruction, which is the right thing
to charge an ES number against and the wrong thing to trust about audibility. This goes through
the two CLIs: encode to a real .rsc3, read that file back the way a decoder would, and compare
what the decoder makes sound in against the ORIGINAL's masking threshold, bin by bin.

Three numbers, because this is a trade and each of them belongs to a different argument:

  rate     - kbit/s on disk, every field charged. The thing budgets never measured.
  ES       - gain-fitted error-signal on the decoded audio. Dips slightly, because noise in
             place of silence is still error power, and that dip is not the same as "worse".
  audibility - the share of filled bins that clear the mask, and the worst overshoot. The one
             that decides shipping: masked noise is free, unmasked noise is a wart.

It also catches the embarrassment case: a decoder filling bands the encoder left silent, or the
reverse, which would be a desync rather than a quality difference and would show up here as
noise where the audit expects none.
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
import rsc3_stream as st

N2 = 4096
W = r3.BAND_WIDTH
GROUPS = ((0, 5e3), (5e3, 10e3), (10e3, 14e3), (14e3, 18e3), (18e3, 24e3))
ARMS = (("off", ["--no-fill"]), ("safe", ["--fill-mode", "safe"]),
        ("loud", ["--fill-mode", "mean"]), ("shape", ["--fill-shape"]))


def read_wav(p):
    w = wave.open(p, "rb")
    sr = w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
    w.close()
    return raw.reshape(-1, 2), sr


def decode_stream(path):
    """
    A decoder's eye view of the file: payloads, then b and the coefficients, frame by frame.
    Nothing here is allowed to know the allocation model, because that is the whole claim.

    Channel 0 only, which is what every quality number in this project is measured on. The rate
    still comes from the whole file on disk.
    """
    with open(path, "rb") as f:
        blob = f.read()
    h = st.unpack_header(blob)
    nbands = st.band_grid(h["n2"] // 2, h["width"])
    nfr = st.frame_count(h["n_samples"], h["n2"])
    r = st.Reader(blob[h["body_off"]:h["body_off"] + h["lens"][0]])
    pe = np.zeros(nbands, np.int64)
    pb = np.zeros(nbands, np.int64)
    fill = bool(h["flags"] & st.FILL_FLAG)
    shape = bool(h["flags"] & st.SHAPE_FLAG)
    reg = st.shape_reg(h["sr"], h["n2"], h["width"]) if shape else None
    X, B = [], []
    for f in range(nfr):
        if fill or shape:
            got = st.read_frame(r, pe, pb, h["width"], fill=fill, shape=shape, reg=reg)
            if shape:
                e, b, m, lv, wd = got
            else:
                e, b, m, lv = got
                wd = None
            X.append(st.reconstruct_bands(e, b, m, lv, f, h["width"], shape=wd))
        else:
            e, b, m = st.read_frame(r, pe, pb, h["width"])
            X.append(st.reconstruct_bands(e, b, m))
        B.append(b)
        pe, pb = e, b
    return h, np.array(X), np.array(B)


print(f"{'file':>10} {'target':>7} {'arm':>5} {'kbit/s':>8} {'ES':>7}   "
      f"{'bins over mask':>17} {'worst dB':>9}   "
      + "".join(f"{f'{a/1e3:.0f}-{b/1e3:.0f}k':>8}" for a, b in GROUPS))
for src in ("cam15.wav", "ij15.wav"):
    for target in (10.0, 128.0):
        for arm, extra in ARMS:
            tmp, wav = "_fv.rsc3", "_fv.wav"
            cmd = [PY, "-u", "-B", os.path.join(".", "rsc3_encode.py"),
                   os.path.join(LAB, src), "-o", tmp, "--bitrate", str(target), *extra]
            p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
            if p.returncode:
                print((p.stdout + p.stderr)[-1200:])
                sys.exit(1)
            subprocess.run([PY, "-u", "-B", os.path.join(".", "rsc3_decode.py"), tmp, "-o", wav],
                           cwd=HERE, capture_output=True, text=True, check=True)
            x, sr = read_wav(os.path.join(LAB, src))
            y, _ = read_wav(wav)
            size = os.path.getsize(tmp)
            h, Xall, Ball = decode_stream(tmp)
            nfr = Xall.shape[0]
            A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
            nc = A.shape[1]
            nb = st.band_grid(nc, W)
            # reconstruct_bands is already amplitude-domain, so the decoder's spectrum is just
            # these coefficients. No transform, so no chance of applying one twice, which is how
            # a 6 dB error hides in this part of the code. The band grid pads to a whole number
            # of bands, so the real coefficients are the first nc.
            Y = Xall[:, :nc]
            A3, Y3 = A.reshape(nfr, nb, W), Y.reshape(nfr, nb, W)
            thr = r3.threshold_dbfs(20 * np.log10(np.maximum(np.abs(A), 1e-30)), sr, nc,
                                    psycho=True).reshape(nfr, nb, W)
            dead = Ball == 0                      # bands the stream says decode to nothing
            lvl_y = 10 * np.log10(np.maximum((Y3 ** 2).mean(axis=2), 1e-300))
            exc = lvl_y[:, :, None] - thr
            made = dead & ((Y3 ** 2).mean(axis=2) > 1e-24)
            bad = made[:, :, None] & (exc > 0)
            lo, hi = 3 * N2, y.shape[0] - N2
            g = (float(np.dot(x[lo:hi, 0], y[lo:hi, 0]))
                 / max(float(np.dot(y[lo:hi, 0], y[lo:hi, 0])), 1e-300))
            es = 10 * np.log10(float(np.mean(x[lo:hi, 0] ** 2))
                               / max(float(np.mean((x[lo:hi, 0] - g * y[lo:hi, 0]) ** 2)), 1e-300))
            dens = [10 * np.log10(max(float((Y3[:, ((np.arange(nc) * sr / N2 >= a)
                                                 & (np.arange(nc) * sr / N2 < b))
                                                .reshape(nb, W)] ** 2).mean()), 1e-300))
                    for a, b in GROUPS]
            # per channel, so it lines up with every other number in this repo. The file pays
            # for both channels; quoting the file rate here made a "10 kbit/s" run read 2x.
            print(f"{src:>10} {target:7.1f} {arm:>5} "
                  f"{size * 8 / (x.shape[0] / sr) / 1000.0 / y.shape[1]:8.2f}"
                  f" {es:7.2f}   {int(bad.any(axis=2).sum()):>8}/{int(made.sum()):<8}"
                  f" {float(exc[bad].max()) if bad.any() else 0.0:9.1f}   "
                  + "".join(f"{d:8.1f}" for d in dens))
            for f in (tmp, wav):
                os.remove(f)
    print()
