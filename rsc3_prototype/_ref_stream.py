"""
RSC3 container: what the model produces, packed with every bit counted.

The prototype measured 512 bits/frame and never shipped a file, so nothing had to ask what
512 bits/frame *means*. Answering that changed the design.

A naive dump (1 sign bit per coded coefficient, 1 byte of scale per live band, no entropy
coding) is 1207 b/f on cam - 2.4x the budget, and the mantissas are not what makes it. Two
things kill it:

  * A per-coefficient bit map is 2048 symbols/frame. Measured with the same Rice machinery
    the shipping packer uses: 3101 b/f for that table alone. No delta coding rescues it, and
    a band-granular alternative comes out the same (104 vs 107 kbit/s). The table, not the
    mantissas, is the most expensive field a file can carry.
  * The prototype's quantiser step is band_max * 2**-b, where band_max is a float the
    encoder happens to know and no decoder could ever be told. So the scalefactor RSC3 has
    was never transmittable, which is the real reason there was no decoder.

The format that follows is the classical one, and it is what a decoder needs to exist:

  scale exponent E   per band, integer, so the scale is 2**E. Delta in time, then Rice.
  bits per band  b   per band, delta in time, then Rice.
  mantissas          per live band: a count, then (gap, sign, magnitude) per nonzero, so a
                     bin that rounds to zero is free and a silent band costs 5 bits.

step = 2**(E_band - b_band), computed by the decoder from those two fields. The encoder
quantises against that same step, so the allocation's charge is what actually lands - the
property quantise() asserted in memory but could never demonstrate, because its scale lived
only inside the encoder.

Three costs, all measured rather than assumed:
  * a power-of-two scale instead of the band's exact max: -0.19 dB (cam), -0.25 dB (ij).
  * one bit count per band instead of per coefficient: makes the side info affordable, and
    buys the zero rule for free - a bin the model would have dropped now rounds to zero,
    and a zero is a gap, which costs nothing here.
  * a band pays its loudest bin's bit count, so quiet bins land on a coarser step than they
    would individually and simply round away.

This module owns the header, the bit packer and the frame grammar. Neither CLI may write a
bit by hand: encoder and decoder drift is the failure mode a codec format actually dies of,
and a shared implementation makes that structurally impossible.
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = b"RSC3"
VERSION = 1
MAX_BITS = 16

# Field widths. GAP_BITS carries a run of skipped bins inside a band, so it must reach
# width-1; a band holds `width` mantissas, so the count reaches width. Both are 4 and 5 for
# the 16-bin bands this codec uses, and deriving them beats hard-coding 16 like the first
# draft did, which spent 2.4 kbit/s advertising that a band had fewer than 32768 entries.


def gap_bits(width):
    return max(int(np.ceil(np.log2(max(width - 1, 1)))) or 1, 1)


def count_bits(width):
    return max(int(np.ceil(np.log2(width + 1))), 1)


# ------------------------------------------------------------------ bit packer

class Writer:
    """MSB-first. Same convention as rsc_encoder.py's Rice writer, so one reader logic
    serves both codecs and a Luau port looks like the one that already ships."""

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, value, w):
        v = int(value) & ((1 << w) - 1) if w else 0
        self.acc = (self.acc << w) | v
        self.n += w
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
            self.acc &= (1 << self.n) - 1

    def done(self):
        if self.n:
            self.put(0, 8 - self.n)
        return bytes(self.buf)

    def bits(self):
        """Bits emitted so far: what has been flushed, plus what is still in the accumulator.
        This is the only place that computes it, so a size report cannot disagree with the
        bytes on disk."""
        return 8 * len(self.buf) + self.n


class Reader:
    def __init__(self, data):
        self.d = data
        self.i = 0
        self.acc = 0
        self.n = 0

    def get(self, w):
        while self.n < w:
            b = self.d[self.i] if self.i < len(self.d) else 0
            self.i += 1
            self.acc = (self.acc << 8) | b
            self.n += 8
        self.n -= w
        v = (self.acc >> self.n) & ((1 << w) - 1) if w else 0
        self.acc &= (1 << self.n) - 1
        return v


# ------------------------------------------------------------------- codes

def zz(v):
    """signed -> unsigned, small magnitudes first"""
    v = int(v)
    return (v << 1) if v >= 0 else ((-v) << 1) - 1


def unz(z):
    z = int(z)
    return (z >> 1) if not (z & 1) else -((z + 1) >> 1)


def rice_len(v, k):
    return int((int(v) >> k) + k + 1)


def pick_k(vals, hi=17):
    if not vals:
        return 0
    best, k = 1 << 60, 0
    for k2 in range(hi):
        s = sum(rice_len(v, k2) for v in vals)
        if s < best:
            best, k = s, k2
    return k


def write_rice(w, vals, k):
    for v in vals:
        v = int(v)
        w.put(0, v >> k)
        w.put(1, 1)
        if k:
            w.put(v & ((1 << k) - 1), k)


def read_rice(r, n, k):
    """Rice(k) with no bound on the quotient: a corrupted stream reads on to the end of the
    buffer rather than failing early, so callers must not feed this arbitrary bytes."""
    out = []
    for _ in range(n):
        q = 0
        while r.get(1) == 0:
            q += 1
        out.append((q << k) | (r.get(k) if k else 0))
    return out


# ------------------------------------------------------------- quantiser grid

def band_grid(n_coefs, width):
    if n_coefs % width:
        raise ValueError(f"{n_coefs} coefficients do not divide into bands of {width}")
    return n_coefs // width


def band_exponents(A_band):
    """
    E = ceil(log2(peak)), per band. This is the whole scalefactor: an integer the decoder
    can be handed. The cost is that it is up to 2x coarser than the band's exact peak,
    measured at -0.19 dB (cam) / -0.25 dB (ij), and it is the price of a decodable codec.
    """
    peak = np.abs(A_band).max(axis=1).astype(np.float64)
    E = np.where(peak > 0.0, np.ceil(np.log2(np.maximum(peak, 2.0 ** -320))), 0.0)
    return E.astype(np.int64)


def band_steps(E_band, b_band):
    """2**(E-b). Dead bands get step 1 so a divide never touches an inf."""
    ex = (E_band.astype(np.float64) - np.where(b_band > 0, b_band, 1))
    step = np.power(2.0, ex)
    return np.where(b_band > 0, step, 1.0)


def quantise_bands(A_band, E_band, b_band):
    """
    Mantissas on the decoder's own grid, clamped to what the field holds: b magnitude bits
    reach 2**b - 1, and a bin sitting exactly on a power of two rounds onto 2**b. Clamping
    costs that bin one step of its own value, and keeps the field fixed width, which is
    better than spending an extra bit per bin on every band.
    """
    step = band_steps(E_band, b_band)
    m = np.round(A_band / step[:, None])
    lim = np.where(b_band > 0, (1 << np.maximum(b_band, 1)) - 1, 0)
    m = np.clip(m, -lim[:, None].astype(np.float64), lim[:, None].astype(np.float64))
    return np.where(b_band[:, None] > 0, m, 0.0).astype(np.int64)


def reconstruct_bands(E_band, b_band, mant):
    """The one reconstruction formula. Encoder and decoder both call this, so agreement is
    not a thing that can drift."""
    step = band_steps(E_band, b_band)
    return mant.astype(np.float64) * step[:, None]


# --------------------------------------------------------------- frame grammar

def write_frame(w, E_band, b_band, mant, prev_E, prev_bits, width):
    """
    Append one frame's bands to `w`. The writer is told E, b and the mantissas and invents
    nothing: the encoder derives them with quantise_bands, and what the writer serialises is
    exactly what the encoder reconstructed.

    Returns (E, field_bits), where field_bits counts what this frame cost in each of the
    three fields, read off the writer itself. The size report is then the truth about the
    file rather than a model of it, and it cannot drift when the grammar changes.
    """
    E = E_band.astype(np.int64)
    live = b_band > 0
    t0 = w.bits()
    dE = (E - prev_E).astype(np.int64)
    k = pick_k([zz(v) for v in dE])
    w.put(k, 5)
    write_rice(w, [zz(v) for v in dE], k)
    t1 = w.bits()

    db = (b_band.astype(np.int64) - prev_bits)
    k2 = pick_k([zz(v) for v in db])
    w.put(k2, 5)
    write_rice(w, [zz(v) for v in db], k2)
    t2 = w.bits()

    gb, cb = gap_bits(width), count_bits(width)
    for j in range(len(b_band)):
        if not live[j]:
            continue
        idx = np.nonzero(mant[j])[0]
        w.put(int(len(idx)), cb)
        base = -1
        for i in idx:
            w.put(int(i) - base - 1, gb)
            v = int(mant[j, i])
            w.put(1 if v < 0 else 0, 1)
            w.put(abs(v), int(b_band[j]))
            base = int(i)
    return E, (t1 - t0, t2 - t1, w.bits() - t2)


def read_frame(r, prev_E, prev_bits, width):
    """
    The mirror of write_frame, and the only path a decoder takes. It returns E, b and the
    mantissas; the caller reconstructs with reconstruct_bands, which is the same function the
    encoder used, so the decoded sound is the encoded sound by construction rather than by
    two tools agreeing twice.
    """
    k = r.get(5)
    dE = np.array([unz(v) for v in read_rice(r, len(prev_E), k)], np.int64)
    E = prev_E + dE
    k2 = r.get(5)
    db = np.array([unz(v) for v in read_rice(r, len(prev_bits), k2)], np.int64)
    b = np.clip(prev_bits + db, 0, MAX_BITS).astype(np.int64)

    mant = np.zeros((len(b), width), np.int64)
    gb, cb = gap_bits(width), count_bits(width)
    for j in range(len(b)):
        if b[j] <= 0:
            continue
        n = r.get(cb)
        base = -1
        for _ in range(n):
            i = base + 1 + r.get(gb)
            neg = r.get(1)
            mag = r.get(int(b[j]))
            mant[j, i] = -mag if neg else mag
            base = i
    return E, b, mant


# --------------------------------------------------------------- container
# The header is what the two tools must agree on without talking, so it lives here and no
# CLI packs it by hand. Little-endian, no padding, fixed size, then one length per payload.

# Little-endian, so `<` means no alignment padding and mixed field widths are safe; the size
# is just the sum of the fields. Fixed size, fixed order, then one length per payload.
HEADER_FMT = "<4sBIBHHIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def pack_header(sr, nch, n2, width, n_samples, budget, payload_lens):
    h = struct.pack(HEADER_FMT, MAGIC, VERSION, int(sr), int(nch), int(n2), int(width),
                    int(n_samples), int(budget), len(payload_lens))
    return h + b"".join(struct.pack("<I", int(L)) for L in payload_lens)


def unpack_header(h):
    magic, ver, sr, nch, n2, width, n_samp, budget, n_pl = struct.unpack(
        HEADER_FMT, h[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"not an RSC3 file: {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported RSC3 version {ver}")
    lens = [struct.unpack("<I", h[HEADER_SIZE + 4 * i:HEADER_SIZE + 4 * (i + 1)])[0]
            for i in range(n_pl)]
    return dict(sr=sr, nch=nch, n2=n2, width=width, n_samples=int(n_samp),
                budget=int(budget), lens=lens, body_off=HEADER_SIZE + 4 * n_pl)


def frame_count(n_samples, n2):
    """must match analyse(), which zero-pads rather than dropping the tail"""
    return int(np.ceil(n_samples / (n2 // 2))) + 1
