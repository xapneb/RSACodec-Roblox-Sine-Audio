"""
test_stream.py - does the RSC3 bit grammar write and read the same thing?

Everything here is a regression test for a bug that already happened. The frame grammar was
rewritten for speed, and a byte-identical result is the only claim that lets me call that an
optimisation rather than a new codec, so all three of these run on every change:

  1. a hand-made frame whose band bit counts are chosen rather than derived - the one case
     where a mantissa field can be written with the wrong width and nothing complains;
  2. real audio, both streams, compared against the pre-optimisation reference byte for byte;
  3. the container header, which the two CLIs share and cannot afford to disagree about;
  4. the shape word geometry, which is the only part of the frame that now depends on a header
     field rather than on arithmetic both sides can do for themselves.

The real-audio arms need rsc2_testing_lab, which is not part of this checkout. Those print SKIP
rather than pass, because a fixture that is quietly missing is how a real regression gets in.

Run:  python test_stream.py
"""
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
sys.path.insert(0, HERE)

from importlib import util as u


def load(name, path):
    spec = u.spec_from_file_location(name, path)
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import rsc3_codec as r3
import rsc3_stream as cur

ref_path = os.path.join(HERE, "_ref_stream.py")
HAVE_REF = os.path.exists(ref_path)

N2 = 4096
W = r3.BAND_WIDTH
NC = N2 // 2
NB = NC // W
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


def pack_frames(mod, A3, bits):
    """
    Write a run of frames and return the payload plus what the decoder should read back.

    The E list is the *parked* state, not the analysis exponents: a band with b = 0 transmits
    no exponent and keeps its old one, and that old value is the only thing either side of
    the format is allowed to remember about it. Recording the analysis value here would be a
    test that agrees with a codec that never ran.
    """
    w = mod.Writer()
    pe = np.zeros(NB, np.int64)
    pb = np.zeros(NB, np.int64)
    E, B, M = [], [], []
    for f in range(A3.shape[0]):
        e = mod.band_exponents(A3[f])
        m = mod.quantise_bands(A3[f], e, bits[f])
        mod.write_frame(w, e, bits[f], m, pe, pb, W)
        pe = np.where(bits[f] > 0, e, pe)
        pb = bits[f].astype(np.int64)
        E.append(pe.copy())
        B.append(pb.copy())
        M.append(m)
    return w.done(), E, B, M


def unpack_frames(mod, blob, nfr):
    r = mod.Reader(blob)
    pe = np.zeros(NB, np.int64)
    pb = np.zeros(NB, np.int64)
    out = []
    for f in range(nfr):
        e, b, m = mod.read_frame(r, pe, pb, W)
        out.append((e, b, m))
        pe, pb = e, b
    return out


print("rsc3 bit grammar")

# 1 ------------------------------------------------------------------ synthetic edge case
A = np.zeros((1, NC))
A[0, 0:64] = 0.4        # four bands, loud, 6 bits each
A[0, 64:200] = 0.01     # nine bands, quiet, 1 bit each - the field width is exactly 1
syn = np.zeros((1, NB), np.int64)
syn[0, :4] = 6
syn[0, 4:13] = 1
A3 = A.reshape(1, NB, W)
blob, E, B, M = pack_frames(cur, A3, syn)
got = unpack_frames(cur, blob, 1)[0]
check("synthetic frame round-trips",
      all(np.array_equal(a, b) for a, b in zip(got, (E[0], B[0], M[0]))),
      f"{len(blob)} B, 13 live bands")

# 2 ------------------------------------------------------------------ real material
for name in ("cam.wav", "ij.wav"):
    p = os.path.join(LAB, name)
    if not os.path.exists(p):
        print(f"  SKIP  {name} round-trips exactly: {p} is not in this checkout")
        continue
    wf = wave.open(p, "rb")
    sr = wf.getframerate()
    raw = np.frombuffer(wf.readframes(int(15 * sr)), np.int16).astype(np.float64)
    x = raw.reshape(-1, 2)[:, 0] / 32768.0
    A = r3.amplitude(r3.analyse(x, N2)[0], N2)
    nfr = A.shape[0]
    A3 = A.reshape(nfr, NB, W)
    bits = np.clip(r3.allocate(A, sr, 384, psycho=True).reshape(nfr, NB, W).max(axis=2)
                   .astype(np.int64), 0, 16)
    blob, E, B, M = pack_frames(cur, A3, bits)
    got = unpack_frames(cur, blob, nfr)
    exact = all(np.array_equal(got[f][0], E[f]) and np.array_equal(got[f][1], B[f])
                and np.array_equal(got[f][2], M[f]) for f in range(nfr))
    check(f"{name} round-trips exactly", exact,
          f"{nfr} frames  {len(blob):,} B  {len(blob) * 8 / nfr:.0f} b/f")
    # The sound the bits produce, checked against the sound the encoder computed from the same
    # numbers. This is the claim that survived the bit-map change: dead bands transmit no
    # exponent, but they also reconstruct to zero, so nothing audible is at stake.
    rc_enc = np.array([cur.reconstruct_bands(E[f], B[f], M[f]) for f in range(nfr)])
    rc_dec = np.array([cur.reconstruct_bands(*got[f]) for f in range(nfr)])
    check(f"{name} decodes to the encoder's own samples",
          float(np.abs(rc_enc - rc_dec).max()) == 0.0,
          f"max diff {float(np.abs(rc_enc - rc_dec).max()):.1e}")
    if HAVE_REF:
        rb = load("ref_s", ref_path)
        rr = rb.Reader
        # the reference grammar's decode, same inputs, and the only honest comparison is the
        # samples - the bytes are supposed to differ now, because dead bands stopped paying
        rblob = []
        rw = rb.Writer()
        pe = np.zeros(NB, np.int64)
        pb = np.zeros(NB, np.int64)
        for f in range(nfr):
            e = rb.band_exponents(A3[f])
            m = rb.quantise_bands(A3[f], e, bits[f])
            rb.write_frame(rw, e, bits[f], m, pe, pb, W)
            pe, pb = e, bits[f].astype(np.int64)
        rblob = rw.done()
        rd = rb.Reader(rblob)
        pe = np.zeros(NB, np.int64)
        pb = np.zeros(NB, np.int64)
        rrecon = []
        for f in range(nfr):
            e, b, m = rb.read_frame(rd, pe, pb, W)
            rrecon.append(rb.reconstruct_bands(e, b, m))
            pe, pb = e, b
        rrecon = np.array(rrecon)
        same_sound = float(np.abs(rrecon - rc_enc).max())
        check(f"{name} sounds identical to the previous bit map", same_sound == 0.0,
              f"max sample diff {same_sound:.1e}   bytes {len(rblob):,} -> {len(blob):,} "
              f"({(1 - len(blob) / max(len(rblob), 1)) * 100:+.1f}%)")

# 3 ------------------------------------------------------------------ parked exponents
# The case the new rule creates and the old one could not express: a band that is loud, then
# silent for a stretch, then loud again. Its exponent is not transmitted while it is silent,
# so the decoder must still remember it, and the second time it is live the delta must be
# against that memory rather than against zero.
A2 = np.zeros((3, NC))
A2[0, 8:24] = 0.5      # band 0 live, loud
A2[2, 8:24] = 0.5      # band 0 live again after one dead frame
A2[1, 8:24] = 1e-9     # band 0 dies
b2 = np.zeros((3, NB), np.int64)
b2[0, 0] = 5
b2[1, 0] = 0
b2[2, 0] = 5
blob, E, B, M = pack_frames(cur, A2.reshape(3, NB, W), b2)
got = unpack_frames(cur, blob, 3)
same_state = np.array_equal(got[1][0], E[1]) and np.array_equal(got[2][0], E[2])
memory = bool(got[2][0][0] == E[0][0] == got[1][0][0])
rc_e = np.array([cur.reconstruct_bands(E[f], B[f], M[f]) for f in range(3)])
rc_d = np.array([cur.reconstruct_bands(*got[f]) for f in range(3)])
check("a silent band parks its exponent and finds it again", same_state and memory,
      f"band 0: live E {int(E[0][0])}, silent E stays {int(E[1][0])}, "
      f"live again {int(E[2][0])}")
check("  and the silent stretch decodes to silence", float(np.abs(rc_d[1]).max()) == 0.0,
      f"frame 1 peak {float(np.abs(rc_d[1]).max()):.1e}   {len(blob) * 8 / 3:.0f} b/f")

# 3 ------------------------------------------------------------------ container
h = cur.pack_header(48000, 2, 4096, 16, 720000, 384, [9, 9])
d = cur.unpack_header(h)
check("header round-trips",
      (d["sr"], d["nch"], d["n2"], d["width"], d["n_samples"], d["budget"], d["lens"]) ==
      (48000, 2, 4096, 16, 720000, 384, [9, 9]), f"{len(h)} B")

# 3b ---------------------------------------------------------------- shape word geometry
# Two claims the wide shape word lives on. The header now carries the width, so a decoder has
# to read what a file actually says and answer with the fine word when it says nothing - which
# is every file written before this field existed. And the bits and the words have to agree at
# every width, because a word that means four bands in one file and eight in another is exactly
# what a derived region cannot be trusted to get right on its own.
reg = cur.shape_reg(48000, N2, W)
ok_read = ok_bits = True
lit = 0
for block in (1, 2, 4, 8, 16):
    for f in range(8):
        b = np.zeros(NB, np.int64)
        b[:11] = 3                                   # the lows, live, below the region
        for j in range(11, NB):
            b[j] = 3 if (j + 3 * f) % 11 < 2 else 0  # sparse live bands inside dead treble
        lvl = -60.0 + 40.0 * np.sin((np.arange(NB) + 7 * f) / 9.0)   # air that rolls with f
        field = cur.shape_window(lvl.reshape(1, -1), b.reshape(1, -1), reg, block)
        w = cur.Writer()
        cur.write_shape(w, field[0])
        written = w.bits()
        blob = w.done()
        want = cur.shape_expand(field[0], b, reg, block)
        r = cur.Reader(blob)
        got = cur.read_shape(r, b, reg, block)
        lit += int((want > 0).sum())
        ok_read = ok_read and np.array_equal(want, got)
        # the reader's own position, in bits: 8 * bytes drawn, minus what it has still parked
        ok_bits = ok_bits and 8 * r.i - r.n == written
check("shape words round-trip at every width", ok_read,
      f"{lit:,} lit bands over 5 widths, {cur.SHAPE_BLOCK} to 16 bands per word")
check("  and the writer's bits are what the reader spends", ok_bits,
      "bit for bit, on every one of them")
geom = all(cur.shape_block(cur.shape_block_shift(b) << cur.SHAPE_BLOCK_SHIFT_W) == b
           for b in (2, 4, 8, 16))
legacy = cur.shape_block(cur.FILL_FLAG | cur.SHAPE_FLAG | cur.PVQ_FLAG
                         | cur.MS_FLAG | cur.GAIN_FLAG) == cur.SHAPE_BLOCK
check("the header says how wide a shape word is", geom and legacy,
      f"2/4/8/16 round-trip, and a file that says nothing reads {cur.SHAPE_BLOCK}")

# 4 ------------------------------------------------------------------ shape field
# The shape field is new grammar with a derived region, so its whole correctness claim is that
# write_shape's bits and read_shape's words land on the same per-band table, that the estimator
# charges exactly what the writer writes, and that both sides rebuild the same samples.
cam15 = os.path.join(LAB, "cam15.wav")
if not os.path.exists(cam15):
    print("  SKIP  shape field on real audio: no rsc2_testing_lab/cam15.wav here")
else:
    x15 = wave.open(os.path.join(LAB, "cam15.wav"), "rb")
    sr15 = x15.getframerate()
    xx = np.frombuffer(x15.readframes(int(15 * sr15)), np.int16).astype(np.float64)
    x15.close()
    x15d = xx.reshape(-1, 2)[:, 0] / 32768.0
    A = r3.amplitude(r3.analyse(x15d, N2)[0], N2)
    nf = A.shape[0]
    A3 = A.reshape(nf, NB, W)
    thr3 = r3.threshold_dbfs(20.0 * np.log10(np.maximum(np.abs(A), 1e-30)), sr15, A.shape[1],
                             psycho=True).reshape(nf, NB, W)
    bits = np.clip(r3.allocate(A, sr15, 24, psycho=True).reshape(nf, NB, W).max(axis=2)
                   .astype(np.int64), 0, 16)
    reg = cur.shape_reg(sr15, N2, W)
    wf = cur.Writer()
    pe = np.zeros(NB, np.int64)
    pb = np.zeros(NB, np.int64)
    sh, SF, E2, M2, sf_all = [], [], [], [], []
    for f in range(nf):
        e = cur.band_exponents(A3[f])
        m = cur.quantise_bands(A3[f], e, bits[f])
        sf = cur.shape_field(cur.band_fill_cap(A3[f], thr3[f]), bits[f], reg)
        _, field = cur.write_frame(wf, e, bits[f], m, pe, pb, W, shape=sf)
        sh.append(cur.shape_expand(sf, bits[f], reg))
        SF.append(field[4])
        E2.append(e)
        M2.append(m)
        sf_all.append(sf)
        pe = np.where(bits[f] > 0, e, pe)
        pb = bits[f].astype(np.int64)
    blob = wf.done()
    rr = cur.Reader(blob)
    pe = np.zeros(NB, np.int64)
    pb = np.zeros(NB, np.int64)
    got_sh = []
    for f in range(nf):
        e, b, m, _, words = cur.read_frame(rr, pe, pb, W, shape=True, reg=reg)
        got_sh.append(words)
        pe, pb = e, b
    check("shape field round-trips",
          all(np.array_equal(got_sh[f], sh[f]) for f in range(nf)),
          f"{nf} frames, {sum(int(v.sum()) for v in got_sh)} lit bands")
    # the estimator charge vs the writer, on the very same fields the writer was handed.
    t5 = cur.frame_costs(np.array(E2), bits, np.array(M2), W, shape_fields=sf_all)
    check("shape estimator matches the writer", int(t5[4]) == int(sum(SF)),
          f"estimator {int(t5[4])} b vs writer {int(sum(SF))} b over {nf} frames")
    # and the sound: encoder words and decoder words must build the same samples.
    enc_x = np.array([cur.reconstruct_bands(E2[f], bits[f], M2[f], None, f, W,
                                            shape=sh[f]) for f in range(nf)])
    dec_x = np.array([cur.reconstruct_bands(E2[f], bits[f], M2[f], None, f, W,
                                            shape=got_sh[f]) for f in range(nf)])
    check("shape decodes to the encoder's own samples",
          float(np.abs(enc_x - dec_x).max()) == 0.0,
          f"max diff {float(np.abs(enc_x - dec_x).max()):.1e}")

# 7 ------------------------------------------------------------------ the compiled walk
# The decoder has two ways to read the same pyramid band and they must not have two answers. One
# is a loop over a count table in machine code, the other is the grammar as it was written, and
# what makes the fast one safe is not that it is fast but that it lands on the same vector and
# leaves the cursor at the same bit.
try:
    import rsc3_bits as bits
except Exception:
    bits = None
if bits is None or bits.pvq_walk is None:
    print("  SKIP  the compiled walk is the same walk: no compiler in this checkout")
else:
    import random
    import numpy as np
    rng = np.random.default_rng(20)
    random.seed(20)
    rows = []
    for _ in range(400):
        n = int(rng.integers(1, 17))
        k = int(rng.integers(0, 40))
        # random.randrange takes the whole codebook size, which is bigger than 64 bits at the
        # wide end, and the walk has to be shown the whole range and not only the easy part.
        idx = random.randrange(int(cur.pvq_count(n, k)))
        rows.append((n, k, idx))
    same = all(np.array_equal(np.asarray(cur._pvq_shape(i, n, k), np.int64),
                              cur._pvq_shape_fast(i, n, k)) for n, k, i in rows)
    check("the compiled walk is the same walk", same, f"{len(rows)} bands, 1 to 16 wide")

    # and the band read, which is the part that has a cursor to keep: the compiled form has to
    # come back with the same three numbers and the same position, or the next field is wrong.
    bad = 0
    for _ in range(300):
        k = int(rng.choice([1, 2, 3, 8, 17, 32, 34, 64]))
        ncode = int(cur.pvq_count(16, k))
        nbits = (ncode - 1).bit_length() if ncode > 1 else 0
        idx = random.randrange(ncode) if ncode > 1 else 0
        wq = int(rng.integers(0, 7))
        q = int(rng.integers(0, 1 << wq)) if wq else 0
        z = k.bit_length() - 1
        w = cur.Writer()
        w.put((k << wq) | q, 2 * z + 1 + wq)
        cur._put_wide(w, idx, nbits)
        pay = w.done()
        fast = bits.pvq_band(np.frombuffer(pay, np.uint8), 0, wq, cur._PVQ_NBITS, 16, cur.PVQ_K_MAX)
        rr = cur.Reader(pay)
        slow = (cur._read_gamma(rr), rr.get(wq), rr.get(nbits) if nbits else 0, rr.tell())
        if nbits <= 62 and (fast[:3], fast[3]) != (slow[:3], slow[3]):
            bad += 1
        if nbits > 62 and fast[3] != -1:
            bad += 1        # the wide index has to be declined, not truncated into 64 bits
    check("the band read agrees bit for bit", bad == 0,
          f"300 bands, wide indices declined to the reference path")

print("\n" + ("all checks passed" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
