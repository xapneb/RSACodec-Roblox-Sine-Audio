"""Rate and distortion of the shape field if it were predicted by a line and sent as exceptions.

This is the tool that killed the idea. It is kept because it is the only place the numbers live:
a per-frame gain and slope, then an exception for every block that misses it by more than T code
units, priced exactly as the writer would price it and scored against the block codes it is
replacing. Run it with a source file and the settings you want priced.

    python _shape_line.py moz.wav --bitrate 24 --fill --fill-shape

Rejection, restated in the summary comment above SHAPE_FLAG: on three files and two rates a line
costs more than the delta stream it would replace at every threshold, so the field is not one the
decoder can be handed the shape of.
"""
import contextlib, io, math, sys
import numpy as np

sys.path.insert(0, "/home/benpax/RSACodec-Roblox-Sine-Audio/rsc3_prototype")
import rsc3_stream as st
import rsc3_encode

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench20.wav"
ARGS = sys.argv[2:] or ["--bitrate", "24", "--fill", "--fill-shape"]
CAP = {}
_real = st.shape_window


def spy(level_db, bits, reg, block=st.SHAPE_BLOCK):
    out = _real(level_db, bits, reg, block)
    CAP["a"] = (np.array(level_db, np.float64), np.array(bits), np.array(reg, bool), int(block))
    CAP["f"] = out
    return out


st.shape_window = spy
sys.argv = ["rsc3_encode.py", SRC, *ARGS, "-o", "/tmp/rd.rsc3"]
with contextlib.redirect_stdout(io.StringIO()):
    rsc3_encode.main()
level, bits, reg, block = CAP["a"]
field = CAP["f"]
nfr = len(field)
if np.array(bits).ndim == 1:
    bits = np.repeat(bits[None], nfr, 0)
if np.array(reg).ndim == 1:
    reg = np.repeat(reg[None], nfr, 0)
if level.ndim == 1:
    level = np.repeat(level[None], nfr, 0)



def fit(c):
    """Least-squares line over positions, or a flat line when there is only one word."""
    c = np.asarray(c, np.int64)
    m = c.size
    if m < 2:
        return 0.0, float(c[0]) if m else 0.0
    x = np.arange(m, dtype=np.float64)
    s, g = np.polyfit(x, c.astype(np.float64), 1)
    return float(s), float(g)


def predict(c):
    s, g = fit(c)
    return np.rint(g + s * np.arange(c.size, dtype=np.float64)).astype(np.int64)


def rice_bits(z):
    """What write_rice charges for the zigzagged row z, with pick_k on it."""
    if z.size == 0:
        return 0
    k = int(st.pick_k(z))
    return int(st.rice_fields(z, k)[1].sum())


def line_bits(codes, T):
    """Predict each block with a line, and pay for the blocks that miss it by more than T."""
    m = codes.size
    if m == 0:
        return 0
    pred = predict(codes)
    gap, val = [], []
    prev = -1
    for t, (c, p) in enumerate(zip(codes, pred)):
        if abs(int(c) - int(p)) > T:
            gap.append(c + t - prev - 1)
            val.append(st.zz(int(c - p)))
            prev = t
    b = 6 + 5 + (1 if not len(gap) else 0)
    if gap:
        b += 5 + rice_bits(np.asarray(gap, np.int64)) + 5 + rice_bits(np.asarray(val, np.int64))
    return int(b)


# what today costs, and what the same codes cost when the prediction is a line
now, fine = 0, 0
errs = {0: [], 1: [], 2: [], 4: []}
nb = 0
for f in range(nfr):
    row = field[f]
    if row is None:
        continue
    for lv in row:
        if lv is None:
            continue
        now += st.shape_cost([lv])
        nb += int(lv.size)
        c = np.asarray(lv, np.int64)
        if c.size == 0:
            continue
        for T in (0, 1, 2, 4):
            fine += 0
            pred = predict(c)
            rec = np.where(np.abs(c - pred) > T, c, pred)
            errs[T].extend(np.abs(c - rec).tolist())
        for T in (0, 1, 2, 4):
            fine += line_bits(c, T) if T == 0 else 0
tot_now = now / nfr
print(f"{SRC}: {nfr} frames, {nb / nfr:.1f} shape words/frame")
print(f"today                      {tot_now:7.1f} bit/frame")
# the line form, per threshold: bit cost measured with T=0 plus the distortion for that T
for T in (0, 1, 2, 4):
    bsum, dsum = 0, np.asarray(errs[T], np.float64)
    for f in range(nfr):
        for lv in (field[f] or []):
            if lv is None:
                continue
            c = np.asarray(lv, np.int64)
            pred = predict(c)
            rec = np.where(np.abs(c - pred) > T, c, pred)
            gap, val, prev = [], [], -1
            for t in range(c.size):
                if abs(int(c[t]) - int(pred[t])) > T:
                    gap.append(t - prev - 1)
                    val.append(st.zz(int(c[t] - pred[t])))
                    prev = t
            b = 1 + 6 + 5
            if gap:
                b += 5 + rice_bits(np.asarray(gap, np.int64)) + 5 + rice_bits(
                    np.asarray(val, np.int64))
            bsum += b
    d = dsum * st._FILL_STEP
    print(f"line + exceptions, T={T}   {bsum / nfr:7.1f} bit/frame   reconstruction error "
          f"p50 {float(np.percentile(d, 50)):.2f} dB  p90 {float(np.percentile(d, 90)):.2f} dB")
print(f"\n(and the same field with no predictor at all, today's delta stream, is what the "
      f"{tot_now:.0f} bit/frame above buys)")
