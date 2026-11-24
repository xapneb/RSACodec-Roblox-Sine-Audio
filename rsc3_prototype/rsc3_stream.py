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

def _bits_of(vals, widths):
    """Values expanded MSB-first into a flat bit vector. Field-major, row-major per value, so
    the order is exactly what a put() per field would have produced."""
    v = np.asarray(vals, np.int64)
    wd = np.asarray(widths, np.int64)
    if wd.size == 0:
        return np.zeros(0, np.uint8)
    mx = int(wd.max())
    if mx == 0:
        return np.zeros(0, np.uint8)
    col = np.arange(mx, dtype=np.int64)
    shift = np.maximum(wd[:, None] - 1 - col[None, :], 0)
    keep = col[None, :] < wd[:, None]
    return ((v[:, None] >> shift) & 1).astype(np.uint8)[keep]


class Writer:
    """MSB-first, same convention as rsc_encoder.py's Rice writer and the same bit order a
    field-by-field loop emits - which is verified against _ref_stream.py rather than assumed.

    Bits accumulate as a vector and are packed once at the end. Packing per value was the
    encoder's hot path for no reason: the bytes do not care what order they were told in.
    """

    def __init__(self):
        self._chunks = []
        self._n = 0

    def put(self, value, w):
        if w:
            self._chunks.append(_bits_of([int(value)], [w]))
            self._n += w

    def put_fields(self, vals, widths):
        """Append many fields at once. `vals` and `widths` are parallel arrays."""
        v = np.asarray(vals, np.int64)
        wd = np.asarray(widths, np.int64)
        if v.size == 0:
            return
        bad = v < 0
        if np.any(bad):
            raise ValueError(f"field {int(v[bad][0])} is negative; sign belongs elsewhere")
        over = v >= (np.int64(1) << np.where(wd < 63, wd, 63))
        if np.any(over):
            j = int(np.argmax(over))
            raise ValueError(f"value {int(v[j])} does not fit {int(wd[j])} bits")
        self._chunks.append(_bits_of(v, wd))
        self._n += int(wd.sum())

    def put_bits(self, bits):
        self._chunks.append(np.asarray(bits, np.uint8))
        self._n += len(bits)

    def bits(self):
        return self._n

    def done(self):
        pad = (-self._n) % 8
        if pad:
            self._chunks.append(np.zeros(pad, np.uint8))
            self._n += pad
        if not self._chunks:
            return b""
        return np.packbits(np.concatenate(self._chunks)).tobytes()


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


def zz_vec(v):
    """The same map, vectorised. (v<<1) ^ (v>>63) is the branch-free form of exactly that,
    so it agrees with zz() on every value and the two are interchangeable."""
    v = np.asarray(v, np.int64)
    return (v << 1) ^ (v >> 63)


def unz(z):
    z = int(z)
    return (z >> 1) if not (z & 1) else -((z + 1) >> 1)


def rice_len(v, k):
    return int((int(v) >> k) + k + 1)


def rice_fields(zvals, k):
    """
    The value and width of every Rice(k) codeword, as arrays: `q` zero bits, a one, then the
    k-bit remainder. As one integer per symbol that is (1 << k) | rem in q + 1 + k bits,
    which is the same bit pattern the loop emitted.
    """
    v = np.asarray(zvals, np.int64)
    q = v >> k
    rem = np.where(k, v & ((1 << k) - 1), 0) if k else np.zeros_like(v)
    return np.asarray((1 << k) | rem, np.int64), q + 1 + k


def pick_k(vals, hi=17):
    """
    The k that minimises the total codeword length. Vectorised, and identical to the loop it
    replaces: argmin returns the first minimum, and the loop only moved on a strict
    improvement, so ties resolve to the same k.
    """
    v = np.asarray(vals, np.int64)
    if v.size == 0:
        return 0
    ks = np.arange(hi, dtype=np.int64)
    tot = ((v[:, None] >> ks[None, :]) + ks[None, :] + 1).sum(axis=0)
    return int(np.argmin(tot))


def write_rice(w, vals, k):
    v, wd = rice_fields(vals, k)
    w.put_fields(v, wd)


def scatter(buf, vals, widths, at):
    """
    Place each value's bits into buf at an absolute bit offset, MSB first.

    The mantissa field has a shape that is not uniform - flags, counts, gaps and magnitudes all
    sit at their own offsets, and two different band shapes - so one function that can put a
    field anywhere is what keeps assembly vectorised. Without it the packer degenerates to one
    put() per value, which is the encoder's hot path for no good reason.
    """
    vals = np.asarray(vals, np.int64)
    widths = np.asarray(widths, np.int64)
    at = np.asarray(at, np.int64)
    if vals.size == 0:
        return
    col = np.arange(int(widths.max()), dtype=np.int64)
    sh = np.maximum(widths[:, None] - 1 - col[None, :], 0)
    keep = col[None, :] < widths[:, None]
    pos = at[:, None] + col[None, :]
    buf[pos[keep]] = ((vals[:, None] >> sh) & 1).astype(np.uint8)[keep]



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
    E = np.asarray(E_band, np.int64)
    b = np.asarray(b_band, np.int64)
    mant = np.asarray(mant, np.int64)
    live = b > 0
    gb, cb = gap_bits(width), count_bits(width)
    t0 = w.bits()

    # The bit counts come first, even though the exponents are what the quantiser was built
    # from: the decoder cannot read scalefactors until it knows which bands exist.
    #
    # This field then says how it is shaped. Below roughly a hundred bits per frame, almost
    # nothing changes allocation between frames, and the dense field still pays a Rice
    # terminator per band: 128 bits of "no" for a frame whose actual audio is 22 bits. Naming
    # only the bands that moved beats that by about 120 b/f, which is most of the remaining
    # floor. Above that the 7-bit gap fields make sparse the more expensive answer, so the
    # encoder measures both shapes and writes the cheaper one - it is not a preference.
    #
    # The shape is told with a prefix-free code, because "sparse, nothing moved" has to be
    # distinguishable from "sparse, here is the list" without reading a field the frame has
    # not got: 0 = dense, 10 = sparse and static, 11 = sparse with changes.
    prev_b = np.asarray(prev_bits, np.int64)
    nbands = len(b)
    gb_n, cb_n = gap_bits(nbands), count_bits(nbands)
    z_all = zz_vec(b - prev_b)
    k_all = pick_k(z_all)
    _, wd_all = rice_fields(z_all, k_all)
    dense_cost = 1 + 5 + int(wd_all.sum())
    ch = np.nonzero(b != prev_b)[0]
    if len(ch):
        z_s = zz_vec(b[ch] - prev_b[ch])
        k_s = pick_k(z_s)
        _, wd_s = rice_fields(z_s, k_s)
        sparse_cost = 2 + 5 + cb_n + int(len(ch)) * gb_n + int(wd_s.sum())
    else:
        z_s, k_s = None, 0
        sparse_cost = 2
    if sparse_cost <= dense_cost:
        if len(ch):
            w.put(0b11, 2)
            gaps = np.empty(len(ch), np.int64)
            gaps[0] = ch[0]
            if len(ch) > 1:
                gaps[1:] = ch[1:] - ch[:-1] - 1
            w.put(k_s, 5)
            w.put(int(len(ch)), cb_n)
            w.put_fields(gaps, np.full(len(ch), gb_n, np.int64))
            write_rice(w, z_s, k_s)
        else:
            w.put(0b10, 2)
    else:
        w.put(0, 1)
        w.put(k_all, 5)
        write_rice(w, z_all, k_all)
    t1 = w.bits()

    # Scalefactors only for live bands. A band with b = 0 decodes to silence - band_steps
    # forces its step to 1 and quantise_bands zeroes its mantissas - so its exponent is
    # information the decoder can never use, and it was costing ~2 bits x every band.
    # The exponent of a dead band stays parked at whatever it was, on both sides, so a band
    # that goes quiet and comes back is still delta-coded against its own history.
    if live.any():
        zE = zz_vec(E[live] - np.asarray(prev_E, np.int64)[live])
        k = pick_k(zE)
        w.put(k, 5)
        write_rice(w, zE, k)
    t2 = w.bits()
    E_next = np.where(live, E, np.asarray(prev_E, np.int64))

    # The mantissa field, and the only place a band gets to choose its own shape.
    #
    # A live band is written one of two ways, and the 1-bit flag that says which is the whole
    # cost of the choice:
    #
    #   gap    count, then (gap, sign, magnitude) per nonzero.  Cheapest when a band is sparse.
    #   flat   (sign, magnitude) for all W bins, no count, no positions.  Cheapest when dense.
    #
    # The gap shape is what this format used everywhere, and it is not free at low rate: the
    # encoder pays gb + 1 bits to point at a coefficient and then b bits to say what it is, so
    # once a band is mostly full it is spending more to describe *where* than *what*. Measured,
    # flat wins at 10 kbit/s on both reference files and everywhere above budget 64, and it
    # decodes faster as well, because there is no count to read and no running position to
    # maintain. Same grid, same step, same clamp, so no decoded value can move.
    nz = mant != 0
    if np.any(nz[~live]):
        raise ValueError("a dead band carries mantissas: the allocation and the quantiser "
                         "disagree, and no rate number from this frame would be meaningful")
    n_cnt = nz.sum(axis=1).astype(np.int64)
    b1 = np.maximum(b, 1)
    cost_gap = cb + n_cnt * (gb + 1 + b1)
    cost_flat = width * (1 + b1)
    flat = live & (cost_flat < cost_gap)
    sizes = np.where(live, 1 + np.where(flat, cost_flat, cost_gap), 0)
    start = np.zeros_like(sizes)
    start[1:] = np.cumsum(sizes)[:-1]
    total = int(sizes.sum())
    if total:
        out = np.zeros(total, np.uint8)
        scatter(out, flat[live].astype(np.int64), np.ones(int(live.sum()), np.int64), start[live])
        jg = np.flatnonzero(live & ~flat)
        jf = np.flatnonzero(flat)
        if jg.size:
            scatter(out, n_cnt[jg], np.full(jg.size, cb, np.int64), start[jg] + 1)
            sel = nz[jg]
            hit = np.flatnonzero(sel)              # band-major, then coefficient order
            bj = hit // width
            pos_e = hit % width
            first = np.zeros(jg.size, np.int64)
            first[1:] = np.cumsum(n_cnt[jg])[:-1]
            rank = np.arange(hit.size) - first[bj]
            prev = np.where(rank > 0, np.concatenate(([-1], pos_e[:-1])), -1)
            v_e = mant[jg][sel]
            b_e = b1[jg][bj]
            gaps = pos_e - prev - 1
            vals = (gaps << (1 + b_e)) | ((v_e < 0).astype(np.int64) << b_e) | np.abs(v_e)
            wd = gb + 1 + b_e
            scatter(out, vals, wd, start[jg][bj] + 1 + cb + rank * wd)
        if jf.size:
            m = mant[jf]
            wv = (1 + b1[jf])[:, None]
            c2 = np.arange(width, dtype=np.int64)
            vals = ((m < 0).astype(np.int64) << b1[jf][:, None]) | np.abs(m)
            scatter(out, vals.reshape(-1), np.repeat(wv, width, axis=1).reshape(-1),
                    (start[jf][:, None] + 1 + c2[None, :] * wv).reshape(-1))
        w.put_bits(out)
    # E is returned as the decoder's own state: live bands updated, dead ones parked. Handing
    # this back is what makes the retention rule the same in both directions.
    return E_next, (t2 - t1, t1 - t0, w.bits() - t2)


def frame_costs(Eall, ball, mant, width):
    """
    What one frame's three fields cost, in bits, computed without writing a single bit.

    This exists because a rate search has no business encoding the file 15 times to answer a
    question that is arithmetic: the mantissa field is a count plus (gap, sign, magnitude) per
    nonzero, and the two delta fields are Rice codewords whose lengths are known the moment k
    is chosen. Returns (scale, bits, mantissa) totals over the given frames.

    The numbers have to match write_frame's byte measurement. That is checked against a packed
    payload in the round trip, not assumed, because an estimator that quietly drifts would
    make every bitrate claim wrong in the same direction forever.
    """
    gb, cb = gap_bits(width), count_bits(width)
    nfr = len(Eall)
    tot = np.zeros(3, np.int64)
    prev_E = np.zeros(len(ball[0]), np.int64)
    prev_b = np.zeros(len(ball[0]), np.int64)
    for f in range(nfr):
        b = ball[f]
        live = b > 0
        nbands = len(b)
        gb_n, cb_n = gap_bits(nbands), count_bits(nbands)
        z_all = zz_vec(b - prev_b)
        k_all = pick_k(z_all)
        _, wda = rice_fields(z_all, k_all)
        dense_cost = 1 + 5 + int(wda.sum())
        ch = np.nonzero(b != prev_b)[0]
        if len(ch):
            z_s = zz_vec(b[ch] - prev_b[ch])
            k_s = pick_k(z_s)
            _, wds = rice_fields(z_s, k_s)
            sparse_cost = 2 + 5 + cb_n + int(len(ch)) * gb_n + int(wds.sum())
        else:
            sparse_cost = 2
        tot[1] += min(dense_cost, sparse_cost)
        if live.any():
            zE = zz_vec(Eall[f][live] - prev_E[live])
            k = pick_k(zE)
            _, wdE = rice_fields(zE, k)
            tot[0] += 5 + int(wdE.sum())
        n = (mant[f] != 0).sum(axis=1)
        b1 = np.maximum(b, 1)
        # the cheaper of the two band shapes, plus the flag that chooses between them
        per = 1 + np.minimum(cb + n * (gb + 1 + b1), width * (1 + b1))
        tot[2] += int(np.where(live, per, 0).sum())
        prev_E, prev_b = np.where(live, Eall[f], prev_E), b
    return tot


def read_frame(r, prev_E, prev_bits, width):
    """
    The mirror of write_frame, and the only path a decoder takes. It returns E, b and the
    mantissas; the caller reconstructs with reconstruct_bands, which is the same function the
    encoder used, so the decoded sound is the encoded sound by construction rather than by
    two tools agreeing twice.
    """
    # The counts field, both shapes, mirrored from write_frame. A prefix-free code decides the
    # rest of the frame, so a decoder never has to guess whether a list follows.
    prev_b = np.asarray(prev_bits, np.int64)
    nbands = len(prev_b)
    gb_n, cb_n = gap_bits(nbands), count_bits(nbands)
    if r.get(1) == 0:
        k_all = r.get(5)
        b = np.clip(prev_b + np.array([unz(v) for v in read_rice(r, nbands, k_all)], np.int64),
                    0, MAX_BITS).astype(np.int64)
    elif r.get(1) == 0:
        b = prev_b.copy()
    else:
        k_s = r.get(5)
        n_c = r.get(cb_n)
        b = prev_b.copy()
        idx = np.empty(n_c, np.int64)
        for i in range(n_c):
            g = r.get(gb_n)
            idx[i] = g if i == 0 else idx[i - 1] + g + 1
        # the writer emits every gap first and then every codeword, so the gaps are not
        # interleaved with the values here even though that would read more naturally
        if n_c:
            b[idx] = np.clip(b[idx] + np.array([unz(v) for v in read_rice(r, n_c, k_s)],
                                               np.int64), 0, MAX_BITS).astype(np.int64)
    live = b > 0
    E = np.asarray(prev_E, np.int64).copy()
    if live.any():
        k = r.get(5)
        dE = np.array([unz(v) for v in read_rice(r, int(live.sum()), k)], np.int64)
        E[live] = np.asarray(prev_E, np.int64)[live] + dE

    # Mantissas: one flag per live band, then that band in the shape it chose. The flat shape
    # is just the same (sign, magnitude) tuple for every bin, so the two branches share their
    # inner read and the only difference is whether positions are spelled out.
    mant = np.zeros((len(b), width), np.int64)
    gb, cb = gap_bits(width), count_bits(width)
    for j in range(len(b)):
        if b[j] <= 0:
            continue
        bw = max(int(b[j]), 1)
        lim = (1 << bw) - 1
        if r.get(1):
            base = -1
            for i in range(width):
                v = r.get(1 + bw)
                mant[j, i] = -(v & lim) if (v >> bw) else (v & lim)
            continue
        n = r.get(cb)
        base = -1
        for _ in range(n):
            i = base + 1 + r.get(gb)
            v = r.get(1 + bw)
            mant[j, i] = -(v & lim) if (v >> bw) else (v & lim)
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
