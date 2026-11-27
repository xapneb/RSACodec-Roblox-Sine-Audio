"""Is the shape field cheaper when the word width is chosen by frequency instead of by file?

The honest form of the claim that a codec does better above the crossover than below it: the fine
shape words below it might not need to be fine. This prices the same region cut into blocks whose
width is a function of the band's own centre frequency, against the same field cut into uniform
blocks, so the two answers are on one curve and not two arguments.

    python _shape_split.py moz.wav --bitrate 24 --fill --fill-shape

It buys a couple of bits and no measurable error, which is the same curve one dial wider to the
left. The width is not the lever; see the summary above SHAPE_FLAG.
"""
import contextlib, io, sys
import numpy as np

sys.path.insert(0, "/home/benpax/RSACodec-Roblox-Sine-Audio/rsc3_prototype")
import rsc3_stream as st
import rsc3_encode
import rsc3_codec as r3
W = r3.BAND_WIDTH

CAP = {}
_real = st.shape_window
SRC = sys.argv[1]
ARGS = sys.argv[2:]


def spy(level_db, bits, reg, block=st.SHAPE_BLOCK):
    out = _real(level_db, bits, reg, block)
    CAP["a"] = (np.array(level_db, np.float64), np.array(bits), np.array(reg, bool))
    return out


st.shape_window = spy
sys.argv = ["rsc3_encode.py", SRC, *ARGS, "-o", "/tmp/sp.rsc3"]
with contextlib.redirect_stdout(io.StringIO()):
    rsc3_encode.main()
level, bits, reg = CAP["a"]
nfr = len(bits) if np.array(bits).ndim > 1 else 1
level = np.repeat(level[None], nfr, 0) if np.array(level).ndim == 1 else level
bits = np.atleast_2d(np.array(bits))
bits = np.repeat(bits[:1], nfr, 0) if bits.shape[0] != nfr else bits
reg = np.repeat(reg[None], nfr, 0) if np.array(reg).ndim == 1 else np.asarray(reg, bool)
if reg.shape[0] != nfr:
    reg = np.repeat(reg[:1], nfr, 0)
nb = int(bits.shape[1])
import wave
with wave.open(SRC) as w:
    sr = w.getframerate()
ctr = np.array([np.arange(j * W, (j + 1) * W) for j in range(nb)]).mean(axis=1)
ctr = ctr * (sr / (nb * 2 * W))


def rice_bits(z):
    if z.size == 0:
        return 0
    return int(st.rice_fields(z, int(st.pick_k(z)))[1].sum())


def price(sel, wfun, lev):
    """One run: its block codes per band, the sequence they are written as, and their price."""
    peak = float(np.max(lev))
    seq, per = [], []
    t = 0
    while t < int(sel.size):
        w = max(1, int(wfun(int(sel[t]))))
        grp = sel[t:t + w]
        c = int(st._block_codes(lev[grp][None, :], peak)[0])
        seq.append(c)
        per.extend([c] * int(grp.size))
        t += w
    c = np.asarray(seq, np.int64)
    b = 1
    if c.size and c.any():
        b += st.FILL_BITS
        if c.size > 1:
            b += 5 + rice_bits(st.zz_vec(np.diff(c).astype(np.int64)))
    return b, np.asarray(per, np.int64)


def run(wfun):
    """Cost, and the error on the bands the field actually puts a word on."""
    bits_sum, err = 0, []
    for f in range(nfr):
        lev = level[f]
        for r in st.dead_runs(bits[f]):
            sel = r[reg[f][r]]
            if sel.size == 0:
                continue
            b, per = price(sel, wfun, lev)
            bits_sum += b
            lit = per > 0
            if lit.any():
                played = st.FILL_FLOOR + per[lit].astype(np.float64) * st._FILL_STEP
                err.extend(np.abs(played - lev[sel[lit]]).tolist())
    return bits_sum / nfr, np.asarray(err)


print(f"{SRC} {' '.join(ARGS)}   {nfr} frames, {int(bits.shape[1])} bands")
for name, f in (("all 2 bands", lambda j: 2), ("all 4 bands", lambda j: 4),
                ("all 8 bands", lambda j: 8),
                ("2 below 5 kHz, 4 above", lambda j: 2 if ctr[j] < 5000 else 4),
                ("2 below 5 kHz, 8 above", lambda j: 2 if ctr[j] < 5000 else 8),
                ("2 below 3 kHz, 8 above", lambda j: 2 if ctr[j] < 3000 else 8),
                ("1 below 5 kHz, 8 above", lambda j: 1 if ctr[j] < 5000 else 8),
                ("2 below 8 kHz, 8 above", lambda j: 2 if ctr[j] < 8000 else 8)):
    b, d = run(f)
    if d.size == 0:
        print(f"  {name:24} {b:7.1f} bit/frame   (nothing lit)")
        continue
    print(f"  {name:24} {b:7.1f} bit/frame   p50 {float(np.percentile(d, 50)):5.2f} dB "
          f"p90 {float(np.percentile(d, 90)):5.2f} dB  max {float(d.max()):5.1f}")
