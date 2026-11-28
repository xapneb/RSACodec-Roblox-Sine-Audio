"""Where the bits go in frequency, and what that does to the top of the spectrum.

Three numbers, because "high frequencies are ignored" is three claims:

  bits      share of paid bits landing above 8 / 12 / 16 kHz, against their share of the signal
  error     share of total error, and of the error that is above the mask, in the same regions
  birdies   bins that survive on their own: loud enough to be heard, with neighbours 20 dB down,
            which is an isolated tone, not a spectrum, and the thing a listener calls a tweet

The birdie rule is deliberately blunt. It asks for a bin that is at least -80 dBFS, that is
isolated over +-4 bins above 8 kHz and +-8 below, and counts it once per frame. Quantising a
sparse spectrum hard leaves exactly this, and nothing in the format can say whether it happened
until someone counts it.
"""
import sys, os, io, re, contextlib, runpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import librosa
import rsc3_codec as r3
import rsc3_stream as st

N2 = 4096
HOP = N2 // 2
BIRD_FLOOR = -80.0
BIRD_DROP = 20.0


def analyse_file(path, sr):
    x, _ = librosa.load(path, sr=sr, mono=True)
    x = np.asarray(x, np.float64)
    A = r3.analyse(x, N2)[0]
    return x, A


def encode_capture(path, rate, extra):
    """Encode for real, capture the coefficients the decoder will read."""
    got = {"frames": []}
    saved = st.reconstruct_bands

    def rec(E, b, mant, levels=None, frame=0, width=16, shape=None, pulses=None):
        if np.ndim(E) == 1:
            got["frames"].append((int(frame), np.asarray(E, np.int64), np.asarray(b, np.int64),
                                  np.asarray(mant, np.int64), None if pulses is None
                                  else np.asarray(pulses, np.float64), int(width)))
        return saved(E, b, mant, levels, frame, width, shape=shape, pulses=pulses)

    st.reconstruct_bands = rec
    buf = io.StringIO()
    sys.argv = ["rsc3_encode.py", path, "--bitrate", str(rate), "--ms"] + list(extra)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(os.path.join(HERE, "rsc3_encode.py"), run_name="__main__")
    finally:
        st.reconstruct_bands = saved
    text = buf.getvalue()
    kbit = float(re.search(r"([\d.]+) kbit/s per channel", text).group(1))
    es = float(re.search(r"ES ([\d.]+) dB", text).group(1))
    return got, kbit, es


def report(path, rate, sr=48000, extra=("--pvq", "--fill")):
    x, A = analyse_file(path, sr)
    nfr, nc = A.shape
    got, kbit, es = encode_capture(path, rate, list(extra))
    frames = got["frames"]
    if not frames:
        return None
    width = frames[0][5]
    nb = nc // width
    band_hz = (np.arange(nb) + 0.5) * width * float(sr) / (2.0 * nc)
    hz = (np.arange(nc) + 0.5) * float(sr) / (2.0 * nc)

    # per coefficient: what the decoder plays, and what was there
    X = np.zeros((nfr, nc))
    B = np.zeros((nfr, nc), np.int64)
    for fr, E, b, mant, pulses, w in frames:
        if fr >= nfr:
            continue
        X[fr] = np.repeat(st.band_steps(E, b), w) * mant.reshape(-1)
        if pulses is not None:
            p = pulses.reshape(-1)
            hit = p != 0.0
            X[fr][hit] = p[hit]
        B[fr] = np.repeat(b, width)
    err = A - X
    err_db = 20.0 * np.log10(np.maximum(np.abs(err), 1e-30))
    sig_db = 20.0 * np.log10(np.maximum(np.abs(A), 1e-30))
    thr = r3.threshold_dbfs(sig_db, sr, nc, psycho=True)
    abs_aud = (err_db > thr)                    # error the model says is audible
    bits = np.zeros(nc)
    for f in range(nfr):
        bits += (B[f] > 0) * 1.0                # one paid slot per live coefficient
    tot_bits = max(bits.sum(), 1.0)
    pwr = (A ** 2)
    tot_pwr = max(pwr.sum(), 1e-30)
    print("  %-12s %6.2f kbit/s  ES %5.2f   %d frames, %d-bin bands" %
          (os.path.basename(path), kbit, es, nfr, width))
    for lo, hi, tag in ((8e3, 12e3, "8-12k"), (12e3, 16e3, "12-16k"), (16e3, 24e3, "16-24k")):
        m = (hz >= lo) & (hz < hi)
        all_m = np.ones(nc, bool)
        bshare = 100.0 * bits[m].sum() / tot_bits
        pshare = 100.0 * pwr[:, m].sum() / tot_pwr
        eshare = 100.0 * (err[:, m] ** 2).sum() / max((err ** 2).sum(), 1e-30)
        ashare = 100.0 * abs_aud[:, m].sum() / max(abs_aud.sum(), 1)
        print("     %-7s bits %5.2f%%  signal %5.2f%%  error %5.2f%%  audible-error %5.2f%%  "
              "eff %+.2f" % (tag, bshare, pshare, eshare, ashare,
                             (ashare / max(bshare, 0.01)) - 1.0))
    # Birdies: a bin that stands 20 dB above everything within +-4 bins of it is not a piece of
    # a spectrum, it is a tone. The count that matters is decoded against source, because the
    # source is allowed one, and the decoder is not.
    def lone_count(S, span, drop):
        d = 20.0 * np.log10(np.maximum(np.abs(S), 1e-30))
        d = np.where(d > BIRD_FLOOR, d, -200.0)
        pad = np.pad(d, ((0, 0), (span, span)), constant_values=-200.0)
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * span + 1, axis=1)
        mid = win[:, :, span]
        rest = np.delete(win, span, axis=2).max(axis=2)
        return int(((mid - rest) > drop).sum())

    span = 4
    for lo in (8e3, 0.0):
        m = hz >= lo
        dec = lone_count(X[:, m], span, BIRD_DROP)
        src = lone_count(A[:, m], span, BIRD_DROP)
        print("     isolated bins %-5s decoded %7d (%.2f/frame)  source %7d (%.2f/frame)  "
              "excess %+.2f/frame" % ("%.0fk" % (lo / 1e3), dec, dec / max(nfr, 1), src,
                                      src / max(nfr, 1), (dec - src) / max(nfr, 1)))

    return dict(kbit=kbit, es=es, bits=bits, hz=hz, pwr=pwr, err=err, abs=abs_aud)


if __name__ == "__main__":
    files = sys.argv[1:2] or ["moz.wav"]
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    # Whatever comes after the rate is the encoder's own command line, so scalar mode is
    # simply "don't pass --pvq". Passing --pvq plus --no-pvq used to be a silent no-op.
    extra = (sys.argv[3].split() if len(sys.argv) > 3 else ["--pvq", "--fill"])

    for f in files:
        for r in ([rate] if rate else [12, 24, 64]):
            print("%s @ %d kbit/s" % (f, r))
            report(f, r, extra=extra)
