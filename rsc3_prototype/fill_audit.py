"""fill_audit.py - what the dead-band filler does, priced in the file's own currency.

Three arms run against the same allocation, so the only difference between them is how the dead
bands are covered, and how much that coverage costs:

  off      silence, which is what a hole sounds like
  safe     one level word per run of dead bands, levelled by the mask
  mean     one level word per run, levelled by the run's own energy
  shape    the per-block words on top, which is where per-band detail lives

Three numbers, because they belong to three different arguments:

  cost   the fill field's bits as a share of the file, and the rate that gives up for it
  cover  how many of the dead bands get anything at all, and how far below the source's own
         band energy that floor sits
  law    filled bins against the source's masking threshold, and the worst overshoot. Zero is
         the contract for `safe`; the others are options precisely because they break it.

The levels it measures are the ones the reconstruction call was handed, so they are what a
decoder plays, not the encoder's opinion of them.
"""
import sys, os, io, re, wave, contextlib, runpy, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import librosa
import rsc3_codec as r3
import rsc3_stream as st

N2 = 4096
HOP = N2 // 2
EPS = 0.5                      # dB, the tolerance the level word's own grid leaves


def capture(src, rate, width, extra):
    """Encode for real, and keep what the reconstruction was handed per frame."""
    rec = {"frames": [], "caps": None, "bits": None}
    saved = {}

    def words_all(cap_all, b_all, mode="safe"):
        rec["caps"] = np.asarray(cap_all, np.float64)
        rec["bits"] = np.asarray(b_all, np.int64)
        return saved["words"](cap_all, b_all, mode)

    def rec_band(E, b, mant, levels=None, frame=0, width=16, shape=None, pulses=None):
        if np.ndim(E) == 1:
            rec["frames"].append((int(frame), np.asarray(b, np.int64), levels, int(width), shape))
        return saved["rec"](E, b, mant, levels, frame, width, shape=shape, pulses=pulses)

    saved["words"] = st.run_fill_words_all
    saved["rec"] = st.reconstruct_bands
    st.run_fill_words_all = words_all
    st.reconstruct_bands = rec_band
    buf = io.StringIO()
    sys.argv = ["rsc3_encode.py", src, "--bitrate", str(rate), "--band-width", str(width),
                "--pvq", "--ms"] + list(extra)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(os.path.join(HERE, "rsc3_encode.py"), run_name="__main__")
    finally:
        st.run_fill_words_all = saved["words"]
        st.reconstruct_bands = saved["rec"]
    text = buf.getvalue()

    def grab(pat, cast=float, default=0.0):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else default

    return rec, dict(kbps=grab(r"([\d.]+) kbit/s per channel"), es=grab(r"ES ([\d.]+) dB"),
                     budget=grab(r"budget (\d+) b/f", int),
                     side=grab(r"side info is (\d+)%", int))


def nfr_total(rec):
    return max(len(rec["frames"]), 1)


def measure(rec, x, sr, width):
    """Fill coverage, the hole it leaves, and its legality, on the recorded frames."""
    nfr = len(rec["frames"])
    if nfr == 0:
        return None
    width = rec["frames"][0][3]
    nb = (N2 // 2) // width
    bits = np.array([f[1] for f in rec["frames"]], np.int64)
    A = r3.analyse(x, N2)[0][:nfr]
    thr = r3.threshold_dbfs(20.0 * np.log10(np.maximum(np.abs(A), 1e-30)), sr, N2 // 2,
                            psycho=True)[:nfr]
    A3 = A.reshape(nfr, nb, width)
    out = dict(dead=0, runs=0, codes=0, lit_bands=0, filled=0, over=0, worst=0.0,
               hole=0.0, hole_n=0, shape_bands=0, fill_bits=0)
    for f in range(nfr):
        b = bits[f]
        runs = st.dead_runs(b)
        out["runs"] += len(runs)
        out["dead"] += int((b <= 0).sum())
        lit = sum(1 for c, _ in (rec["frames"][f][2] or []) if int(c))
        out["codes"] += lit
        if rec["frames"][f][4] is not None:
            sh = np.asarray(rec["frames"][f][4], np.int64)
            out["shape_bands"] += int((sh > 0).sum())
            out["fill_bits"] += st.FILL_BITS * int((sh > 0).sum()) + 1
        lv = np.full(nb, np.nan)
        if rec["frames"][f][2]:
            st._write_run_levels(lv, runs, rec["frames"][f][2])
        if rec["frames"][f][4] is not None:
            st._write_shape_levels(lv, rec["frames"][f][4])
        on = ~np.isnan(lv)
        if not on.any():
            continue
        out["lit_bands"] += int(on.sum())
        ob = np.repeat(on, width)
        lvl = np.repeat(np.where(on, lv, -200.0), width)
        out["filled"] += int(ob.sum())
        bad = lvl[ob] > thr[f][ob] + EPS
        out["over"] += int(bad.sum())
        if bad.any():
            out["worst"] = max(out["worst"], float((lvl[ob] - thr[f][ob]).max()))
        # what sits in the band, and what the floor replaces it with
        src_db = 10.0 * np.log10(np.maximum((A3[f] ** 2).sum(axis=-1), 1e-12))
        play_db = 10.0 * np.log10(np.maximum(10.0 ** (np.nan_to_num(lv, neginf=-200.0) * 0.1)
                                             * width, 1e-12))
        m = (b <= 0) & (src_db > -80.0) & (lv > -100.0)
        out["hole"] += float((src_db[m] - play_db[m]).sum())
        out["hole_n"] += int(m.sum())
    return out


def one(path_mono, rate, arm, width, sr, margin=0.0):
    x, _ = librosa.load(path_mono, sr=sr, mono=False)
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[None]
    ch = x[0]
    import tempfile
    fd, wav = tempfile.mkstemp(prefix="fill_audit_", suffix=".wav")
    os.close(fd)
    with wave.open(wav, "wb") as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(sr)
        wv.writeframes(np.clip(np.round(ch * 32767.0), -32768, 32767).astype("<i2").tobytes())
    extra = {"off": ["--no-fill"],
             "flat": ["--fill", "--fill-mode", "safe", "--no-fill-shape"],
             "mean": ["--fill", "--fill-mode", "mean", "--no-fill-shape"],
             "shape": ["--fill", "--fill-mode", "safe", "--fill-shape"],
             "both": ["--fill", "--fill-mode", "safe", "--fill-shape"]}[arm]
    extra = extra + ["--fill-margin", str(margin)]
    rec, got = capture(wav, rate, width, extra)
    m = measure(rec, ch, sr, width)
    if m is None:
        return None
    # The fill field's own share: the run words' bits are 1 + 8 per run whenever the run is lit.
    run_bits = 0
    for f, (fr, b, lvs, w, sh) in enumerate(rec["frames"]):
        runs = st.dead_runs(b)
        if not runs:
            run_bits += 1
        elif lvs and any(int(c) for c, _ in lvs):
            run_bits += 1 + 8 * len(runs)
    m["fill_bits"] += run_bits
    m["bits"] = got["kbps"] * 1000.0 * HOP * nfr_total(rec) / float(sr)
    return dict(got, **m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--rates", default="12,24,64")
    ap.add_argument("--band-width", dest="width", type=int, default=32)
    ap.add_argument("--arms", default="off,flat,shape")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="dB of mask the filler may spend, passed to the encoder")
    a = ap.parse_args()
    print("src %s   band width %d   one channel" % (a.src, a.width))
    for rate in [int(v) for v in a.rates.split(",")]:
        for arm in a.arms.split(","):
            d = one(a.src, rate, arm, a.width, a.sr, a.margin)
            if d is None:
                continue
            cover = 100.0 * d["lit_bands"] / max(d["dead"], 1)
            print("  %4dk %-5s m%4.0f  %6.2f kbit/s  ES %5.2f  dead %6d  lit %5.1f%%  "
                  "bins over %6d (%4.2f%%)  worst %+5.1f dB  hole %+5.1f dB  fill field %4.1f%%"
                  % (rate, arm, a.margin, d["kbps"], d["es"], d["dead"], cover, d["over"],
                     100.0 * d["over"] / max(d["filled"], 1), d["worst"],
                     d["hole"] / max(d["hole_n"], 1), 100.0 * d["fill_bits"] / max(d["bits"], 1)))
