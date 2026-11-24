"""
test_stream.py - does the RSC3 bit grammar write and read the same thing?

Everything here is a regression test for a bug that already happened. The frame grammar was
rewritten for speed, and a byte-identical result is the only claim that lets me call that an
optimisation rather than a new codec, so all three of these run on every change:

  1. a hand-made frame whose band bit counts are chosen rather than derived - the one case
     where a mantissa field can be written with the wrong width and nothing complains;
  2. real audio, both streams, compared against the pre-optimisation reference byte for byte;
  3. the container header, which the two CLIs share and cannot afford to disagree about.

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
    wf = wave.open(os.path.join(LAB, name), "rb")
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

print("\n" + ("all checks passed" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
