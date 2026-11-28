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

import rsc3_bits as _BITS     # the compiled form of the one sequential step, when it is available

MAGIC = b"RSC3"
VERSION = 1
MAX_BITS = 16
# _put_raw masks to w bits. Widths under 64 come from this table rather than from building
# 1<<w every call, which is a Python allocation on the end of every field the writer is handed.
_MASK = [(1 << w) - 1 for w in range(64)]
FILL_FLAG = 1  # header flag bit 0: this stream carries a dead-band noise fill field
# Header flag bit 2: a float32 playback gain follows the payload-length table. Quantizing a
# waveform that is already on the rail reconstructs hot, and by how much is a property of
# that waveform. A constant pad (the 6 dB one) still clips the next hotter file and quiets
# every file that never crossed. The gain is ceiling/peak of the synthesis the decoder
# plays, so the file sits on the rail and the quality figure is not charged for the offset.
GAIN_FLAG = 4
# Header flag bit 3: every live band is a pyramid vector instead of scalar mantissas.
# The bit budget of each band is whatever its scalar field would have spent. A dead band
# has no scalar field, so fill and shape still own it. A file with this bit clear has
# no pyramid data.
PVQ_FLAG = 8
# Header flag bit 4: stereo frames may code a band as mid/side instead of left/right.
# The mask is in channel 0 only, one decision for every MS_GROUP bands. A clear flag
# is the old independent channels, byte for byte.
MS_FLAG = 16
MS_GROUP = 4
_MS_SCALE = float(np.sqrt(0.5))
# Header flag bit 1: dead bands above SHAPE_F0 carry the shape field. One delta-coded level
# word per block of SHAPE_BLOCK dead bands, and the level is that block's own energy, not a
# share of the mantissa budget. A low rate never opens those bands — the waterfill spends
# every bit on the loud lows — and the flat fill cannot paint them either, because an
# unmasked band's ceiling is the -120 floor, which decodes as silence. The noise is what
# puts dense treble back. A code of 0 leaves the flat fill standing, and that is also how a
# sparse block or a block that is only a little energy is sent: the decoder has no gate of
# its own. The region is derived from the header, so moving SHAPE_F0 or SHAPE_BLOCK changes
# the bitstream of every file that sets the flag. The flatness and depth knees do too, but
# only the codes move; the grammar does not.
# Two things were tried against this field and both lost, so they are written down where the next
# person looking for a cheaper shape field will find them rather than in a commit message.
#
# The first is a predicted shape field: one gain and one slope per run, then an exception for every
# block that misses the line by more than T code units. The intuition was that the levels are a
# smooth curve, and they are, but a line does not beat what the field already does to that curve.
# Priced as the writer would price it (_shape_line.py): on beep.wav at 24 kbit/s it is 156 bit per
# frame against 136, at 10 kbit/s 133 against 120, and on a sine, where a run has almost nothing
# to say, 46 against 8. The exception form is worse at every threshold, including the one that
# reproduces the field exactly, because the delta stream it replaces already charges k+1 bits to
# say "same", which is close to free, while an exception has to say *where* it is first.
#
# The second is the frequency-dependent block width, which is the part of the same intuition that
# can be built: the shape above the crossover is nearly inaudible to be wrong about, so the fine
# words down here might not need to be fine. They cost what they cost either way (_shape_split.py:
# moz.wav 102 bit/frame and 8.72 dB at the 90th percentile for two bands below 8 kHz and eight
# above, against 104 and 8.77 for four everywhere) - a couple of bits and no error, so the same
# curve a dial wider to the left. The width was never the lever, and nothing above SHAPE_F0 is.
SHAPE_FLAG = 2
SHAPE_F0 = 2000.0
# Header bits 5 and 6 carry how wide a shape word is: the field is log2(block / SHAPE_BLOCK),
# so a file that says nothing is the fine two-band stream, byte for byte, and every file made
# before this field existed still decodes as it was made. The width is here rather than derived
# from the rate so that a decoder reading a file does not have to want what the encoder wanted:
# eight bands on a file that only meant two is a desync, not a difference of opinion, and the
# drift check catches that far too late to be cheap.
SHAPE_BLOCK_SHIFT = 32 | 64
SHAPE_BLOCK_SHIFT_W = 5
# The default above SHAPE_BLOCK, which is what a file gets unless --shape-block says otherwise.
# Chosen on beep.wav at two rates, and it won both: 145 b/f against 221 at 24 kbit/s per
# channel, ES 10.016 against 9.565, and 119 b/f against 206 at 10 kbit/s, ES 6.553 against
# 5.774 - where it is also 2% cheaper, because the fine words would otherwise be buying
# nothing. The trade it gives up is painting error, which ES cannot see: 1.15 dB median / 3.2 dB
# at the 90th percentile on two bands, against 1.40 / 4.26 here. It is a real trade, and the
# rate is not what decides it - the field is roughly a fifth of the file either way.
SHAPE_BLOCK_WIDE = 4
# Two bands, about 375 Hz. Eight bands was one power mean across 1.5 kHz, and that mean
# sits on the loud end of a falling block, so the top of the block plays hot. On the loud
# 8 s of beep.wav, at the same plane count, a painted band within 36 dB of the frame peak
# was off by 2.0 dB median and 5.2 dB at the 90th percentile. Two bands brings that to
# 1.3 and 3.3. One band is always flat, so the gate below would paint an isolated partial.
# A pair still codes 0 when one side is silence. The finer words cost about 130 bits/frame
# and the 12 kbit/s solve on that slice drops from 17 planes to 11 to stay on the rate.
# Moving this changes every SHAPE_FLAG file.
SHAPE_BLOCK = 2
# Geometric mean over arithmetic mean of the band powers in a block. A single partial sits
# near 0, because the empty neighbours crush the geometric mean, and the power mean would
# play those neighbours at the partial's level. Noise that actually fills the block sits
# near 1 (measured ≥ 0.5 on air, ≤ 0.17 on an isolated partial). Below LO the word is 0;
# above HI the level is the power mean; between them it fades, so a block on the line does
# not tick on and off every hop.
SHAPE_FLAT_LO = 0.20
SHAPE_FLAT_HI = 0.45
# dB of the block's power mean under the frame's loudest band. Full level within FULL,
# silence by OFF. Bright air on a loud fundamental sits about 35–45 dB down and a pure
# bass's leakage sits past 80, which is the split these two numbers are.
SHAPE_DEPTH_FULL = 42.0
SHAPE_DEPTH_OFF = 66.0

# Noise fill of dead bands. A band with b = 0 decodes to silence, and at 10 kbit/s that is
# most of everything above ~7 kHz, which is the hole this pays for. The decoder already knows
# exactly which bands are dead from b, so the run structure costs nothing: one 6-bit level per
# run is the whole of the side info, and code 0 means leave that run silent.
FILL_BITS = 6
FILL_FLOOR = -120.0
# The grid one level word sits on, and the largest word. fill_code is a scalar on this grid
# and _fill_codes is the same rounding on an array; both read these instead of recomputing.
_FILL_STEP = -FILL_FLOOR / ((1 << FILL_BITS) - 1)
_FILL_MAX = (1 << FILL_BITS) - 1
# 2-bit tilt: dB of total fall ACROSS a run, one-sided. A run on this material is routinely a
# hundred bands wide, so "dB per band" would be a 50 dB ramp, and the mask of a dead run only
# ever falls with frequency - a rising floor would be encoding a mistake.
FILL_TILT_BITS = 2
FILL_TILT = (0.0, -12.0, -28.0, -60.0)

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
    """MSB-first, same bit order a field-by-field loop emits.

    Bits collect in a small integer and spill to bytes every few kilobits. One numpy
    array per field was most of the cost of writing a frame: a loud pyramid frame
    puts three tiny fields per live band, and packing each one on its own dominated
    the encode. The bytes are the ones packbits used to produce.
    """

    _SPILL = 1 << 15

    def __init__(self):
        self._parts = []
        self._acc = 0
        self._n = 0
        self._total = 0

    def _put_raw(self, value, w):
        w = int(w)
        if w <= 0:
            return
        # A 64-bit mask is a Python int that costs more to build than the AND it is there for,
        # and a bulk put hands this one a whole run at once. Small widths get a table.
        self._acc = (self._acc << w) | ((int(value) & _MASK[w]) if w < 64
                                        else (int(value) & ((1 << w) - 1)))
        self._n += w
        self._total += w
        if self._n >= self._SPILL:
            self._spill()

    def _put_run(self, bits):
        """Append a run of bits, MSB first, as one integer instead of one call per bit.

        This is what makes a field a single transaction: pack the bits into whole bytes, read
        them back as an integer, and let _put_raw place it. The run is padded at the low end so
        the pack is byte aligned, and the padding is shifted away again so the value is exactly
        the bits that were given, which is the same result a loop of _put_raw would have left.
        """
        n = int(bits.size)
        if n == 0:
            return
        pad = (-n) & 7
        if pad:
            bits = np.concatenate((bits, np.zeros(pad, np.uint8)))
        word = int.from_bytes(np.packbits(bits).tobytes(), "big") >> pad
        self._put_raw(word, n)

    def _spill(self):
        spare = self._n & 7
        take = self._n - spare
        if take <= 0:
            return
        acc = self._acc >> spare if spare else self._acc
        self._parts.append(acc.to_bytes(take // 8, "big"))
        self._acc = self._acc & ((1 << spare) - 1) if spare else 0
        self._n = spare

    def put(self, value, w):
        self._put_raw(value, w)

    def put_fields(self, vals, widths):
        """Append many fields at once. `vals` and `widths` are parallel arrays.

        The checks run over the whole field at once, in one pass, because this is called a
        dozen times a frame and a frame is 1.5 ms of work. On a 20-second file that is about
        5,200 calls, and the older shape of it - one `any` per rule, and a bound built from
        three numpy operations to be compared against once - was costing more than the bits
        being packed, which is what a 4.91 s encode and a 4.58 s one differ by. What is kept is
        the whole of the rule: a negative value never belongs in a field, and no value may
        exceed the width it was given. Where a 63-bit field is reachable it is only ever a Rice
        quotient, and a non-negative int64 always fits one, which is why the bound is widened
        rather than the shift. A 64-bit field is only ever that quotient too, and the same
        reasoning holds. Output for valid input is the same bytes either way, so this is not a
        trade, it is just a check that stopped paying for itself.
        """
        v = np.asarray(vals, np.int64)
        wd = np.asarray(widths, np.int64)
        if v.size == 0:
            return
        lim = np.where(wd < 63, np.int64(1) << np.clip(wd, 0, 62), np.iinfo(np.int64).max)
        wrong = (v < 0) | (v >= lim)
        if wrong.any():
            j = int(np.argmax(wrong))
            if v[j] < 0:
                raise ValueError(f"field {int(v[j])} is negative; sign belongs elsewhere")
            raise ValueError(f"value {int(v[j])} does not fit {int(wd[j])} bits")
        if v.size > 2:
            self._put_run(_bits_of(v, wd))
            return
        for val, width in zip(v.tolist(), wd.tolist()):
            self._put_raw(val, width)

    def put_bits(self, bits):
        self._put_run(np.ascontiguousarray(np.asarray(bits, np.uint8).reshape(-1) & np.uint8(1)))

    def bits(self):
        return self._total

    def done(self):
        pad = (-self._total) % 8
        if pad:
            self._put_raw(0, pad)
        self._spill()
        if not self._parts:
            return b""
        return b"".join(self._parts)


class Reader:
    """MSB-first. The reservoir is a Python int holding the bits not yet consumed, refilled
    four bytes at a time. A per-byte refill was one Python call per coefficient, and a flat
    band is sixteen of those. `take` pulls a whole field in one shift; `quotient` counts the
    zeros of a Rice code from the highest set bit instead of reading them one by one.
    Past the end of the buffer the refill is zeros, which is what a short read has to do:
    a truncated stream runs on rather than raising, and callers must not hand this arbitrary
    bytes."""

    def __init__(self, data):
        self.d = data
        self.L = len(data)
        self.i = 0
        self.acc = 0
        self.n = 0
        self._arr = None

    def bitbuf(self):
        """The payload as a byte array, made once per reader.

        np.frombuffer does not copy, and the compiled walker wants an array rather than the
        bytes object itself, so this is built the first time a field is handed off and reused for
        every field after it.
        """
        if self._arr is None:
            self._arr = np.frombuffer(self.d, np.uint8)
        return self._arr

    def _fill(self, w):
        acc, n, i = self.acc, self.n, self.i
        d, L = self.d, self.L
        while n < w:
            if i + 4 <= L:
                acc = (acc << 32) | int.from_bytes(d[i:i + 4], "big")
                i += 4
                n += 32
            else:
                b = d[i] if i < L else 0
                i += 1
                acc = (acc << 8) | b
                n += 8
        self.acc, self.n, self.i = acc, n, i

    def get(self, w):
        if not w:
            return 0
        self._fill(w)
        n = self.n - w
        v = (self.acc >> n) & ((1 << w) - 1)
        self.n = n
        self.acc = self.acc & ((1 << n) - 1) if n else 0
        return v

    def tell(self):
        """The absolute bit offset of the next bit.

        The reservoir holds `n` bits of what starts at byte `i`, so the run it has not consumed
        begins at `i * 8 - n` wherever that happens to be, and a compiled walker handed the
        payload can start from exactly there.
        """
        return self.i * 8 - self.n

    def seek(self, pos):
        """Put the cursor at an absolute bit offset and rebuild the reservoir around it.

        Only whole bytes before `pos` count as consumed, so the tail of the byte `pos` lands in
        goes back into the accumulator rather than being read twice. It is the same state the
        reader would have reached one `get` at a time, which is the whole reason a field can be
        handed off and the field after it picked up normally.

        The tail is the *low* bits of that byte, because a byte is consumed from the top, and the
        reader's own invariant is that the next bit is the high bit of an accumulator holding
        exactly `n` bits. Reading that the other way round leaves the field's own symbols right
        and the field after it wrong, which is the worst failure this could have: the numbers in
        the middle look fine and the ones at the end do not.
        """
        i = (pos + 7) >> 3
        n = (i << 3) - pos
        self.i = i
        self.n = n
        if n:
            b = self.d[i - 1] if 0 < i <= self.L else 0
            self.acc = b & ((1 << n) - 1)
        else:
            self.acc = 0

    def take(self, w):
        """The next `w` bits as one integer, MSB first. Same bits `get` would return, one call."""
        return self.get(w)

    def quotient(self):
        """How many 0-bits precede the next 1, consuming that 1. The Rice quotient."""
        q = 0
        while True:
            if self.n == 0:
                self._fill(1)
            acc, n = self.acc, self.n
            if acc == 0:
                q += n
                self.n = 0
                self.acc = 0
                continue
            lead = (n - 1) - (acc.bit_length() - 1)
            q += lead
            n = n - lead - 1
            self.n = n
            self.acc = acc & ((1 << n) - 1) if n else 0
            return q


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


# pick_k's candidate list, by highest k needed. ks[m] is 0..m, so a field whose widest value
# fits in four bits gets four widths tested and not seventeen.
_PICK_K_TABLE = [np.arange(m + 1, dtype=np.int64) for m in range(64)]


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

    Only the widths that can hold the largest value need testing, and that is a bound rather
    than a heuristic. Once 2**k exceeds the biggest value, every quotient is zero and the
    length is n * (k + 1), which only grows, so the minimum over k >= bit_length(max) sits at
    bit_length(max) itself. The count and exponent fields are small deltas, so the usual
    candidate list is three or four wide rather than seventeen, and the table is built once.
    """
    v = np.asarray(vals, np.int64)
    if v.size == 0:
        return 0
    top = min(int(v.max()).bit_length(), hi - 1)
    ks = _PICK_K_TABLE[top]
    tot = ((v[:, None] >> ks[None, :]) + ks[None, :] + 1).sum(axis=0)
    return int(np.argmin(tot))


def write_rice(w, vals, k):
    v, wd = rice_fields(vals, k)
    w.put_fields(v, wd)


# The candidate Rice parameters. One array for the vectorised callers, so an estimator that
# works on rows and pick_k that works on one frame are searching the same set and picking the
# same k on a tie.
_RICE_KS = np.arange(17, dtype=np.int64)


def _rice_rows(Z, ks=_RICE_KS, counts=None):
    """The Rice parameter and the total codeword width of every row of Z, in one pass.

    Exactly what pick_k and rice_fields would have said about each row alone: the k that
    minimises the row's length, ties going to the smaller k, and the summed width that choice
    buys. `counts` is the number of symbols in a row when it is not the row's length, which
    happens to a masked row - only the entries that survived count, and that number is what
    multiplies (k + 1).
    """
    Z = np.asarray(Z, np.int64)
    cand = ks[:_k_stop(Z)]
    s = (Z[:, :, None] >> cand).sum(axis=1)
    n = Z.shape[1] if counts is None else np.asarray(counts, np.int64)[:, None]
    tot = s + n * (cand + 1)
    k = tot.argmin(axis=1)
    return k, np.take_along_axis(tot, k[:, None], axis=1).reshape(-1)


def _k_stop(Z):
    """How many Rice parameters a block of values can possibly want.

    Once 2**k is above the largest value every quotient is zero and the length is n * (k + 1),
    which only grows, so a row's shortest code sits at or below bit_length of that row's biggest
    value. A delta field of a frame is usually 0..3, so that is three columns tested instead of
    seventeen; on a stack of frames the same argument runs over the whole block, because a row's
    own bound cannot exceed the block's. Truncating the candidate list this way leaves the answer
    alone for that reason, not as a guess.
    """
    if Z.size == 0:
        return 1
    return max(1, int(Z.max()).bit_length())


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
    buffer rather than failing early, so callers must not feed this arbitrary bytes.

    The compiled walker takes the field when there is enough of it to be worth the handoff, and
    the per-symbol walk below is what runs without the JIT. Both paths read the same bits, and
    the bit offset the walker returns is where the reader would have ended up symbol by symbol,
    so the field that follows does not care which way this one was read.
    """
    if _BITS.rice_decode is not None and n >= 8:
        vals, pos = _BITS.rice_decode(r.bitbuf(), r.tell(), n, k)
        r.seek(pos)
        return vals
    out = np.empty(n, np.int64)
    if k:
        for i in range(n):
            out[i] = (r.quotient() << k) | r.get(k)
    else:
        for i in range(n):
            out[i] = r.quotient()
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
    # The last axis is the bins of a band. A stack of frames is the same call: the peak
    # does not depend on any other frame, so the encoder can hand the whole file at once.
    peak = np.abs(A_band).max(axis=-1).astype(np.float64)
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
    m = np.round(A_band / step[..., None])
    lim = np.where(b_band > 0, (1 << np.maximum(b_band, 1)) - 1, 0)
    m = np.clip(m, -lim[..., None].astype(np.float64), lim[..., None].astype(np.float64))
    return np.where(b_band[..., None] > 0, m, 0.0).astype(np.int64)


def reconstruct_bands(E_band, b_band, mant, levels=None, frame=0, width=16, shape=None,
                      pulses=None):
    """
    The one reconstruction formula. Encoder and decoder both call this, so agreement is not a
    thing that can drift. levels=None means no fill, which is the whole point of putting it in
    the signature: the encoder's ES and the decoder's samples come from the same call with the
    same arguments, so a fill that only the encoder hears is impossible to ship by accident.
    shape is the per-band words of the shape field, applied after the fill where they are lit.
    pulses is the pyramid replacement for live bands, band-major. Those bands are written
    with zero mantissas, so a nonzero pulse bin is the whole sample. A zero row is not a
    pyramid band: dead-band noise is left where it was drawn.
    """
    X = band_steps(E_band, b_band)[:, None] * mant.astype(np.float64)
    if levels is not None or shape is not None:
        # One draw per band. A lit shape word replaces the run's level, and the noise is a
        # function of (frame, band) only, so drawing the winning level once is the same sample
        # the previous order produced by drawing the run and then overwriting it.
        lvl = np.full(X.shape[0], np.nan)
        if levels is not None:
            _write_run_levels(lvl, dead_runs(b_band), levels)
        if shape is not None:
            _write_shape_levels(lvl, shape)
        X = _draw_floor(X, lvl, frame, width)
    if pulses is not None:
        p = np.asarray(pulses, np.float64).reshape(X.shape)
        hit = p != 0.0
        X[hit] = p[hit]
    return X


def shape_bands(X_band, words, frame, width):
    """Where a shape word is lit it replaces what the run word laid there - one band, its own
    word, the same deterministic noise draw - and where it is zero the fill stands untouched.
    A lit word is that block's energy on the fill grid, not a mask ceiling. Code 0 is the
    encoder declining to paint a sparse or quiet block, and this side does not second-guess it."""
    lvl = np.full(X_band.shape[0], np.nan)
    _write_shape_levels(lvl, words)
    return _draw_floor(X_band, lvl, frame, width)


# ------------------------------------------------------------- noise fill
# Dead bands decode to silence, which at low rate leaves a hole wherever the source had
# something. Filling it is affordable because the decoder already knows which bands are dead:
# b = 0, from the counts field, so the only thing to transmit is how loud the fill is.
#
# Two rules make this safe to put in a decoder.
#
# First, the noise is a pure function of (frame, band). No generator state, no sequence, no
# per-band draw in iteration order - a streaming decoder that skips a frame, or a Luau port
# that processes bands in a different order, still gets the same numbers. That is the property
# fill_experiment.py never had, and it is the reason that experiment could not ship.
#
# Second, the level the encoder synthesises with is the level the *code* carries, not the level
# it wanted. Both sides read the same code and reconstruct the same floor, so the encoder's own
# ES is measured against the audio the decoder will actually play. This is the discipline the
# parked-exponent work already established: return the decoder's state, never re-derive it.

_HASH = (1 << 32) - 1


def hash_noise(n, frame, band):
    """mulberry32. Two integers in, n deterministic values out: three multiplies, three
    shifts, no state, and short enough to port to Luau without trusting a float model.
    Uniform in [-1, 1), which is a fine input shape because it gets normalised anyway.

    `band` may be an array. One band returns shape (n,), a stack of bands returns (bands, n).
    The integer steps are the scalar formula, so a stack agrees with calling it once per band.
    """
    bands = np.asarray(band, np.int64)
    scalar = bands.ndim == 0
    bands = np.atleast_1d(bands)
    s = (np.int64(frame) * np.int64(0x9E3779B9) + bands * np.int64(0x85EBCA6B)) & np.int64(_HASH)
    u = np.arange(n, dtype=np.int64)
    x = (s[:, None] + u[None, :] * np.int64(0x9E3779B1)) & np.int64(_HASH)
    x = ((x ^ (x >> np.int64(16))) * np.int64(0x85EBCA6B)) & np.int64(_HASH)
    x = ((x ^ (x >> np.int64(13))) * np.int64(0xC2B2AE35)) & np.int64(_HASH)
    x = x ^ (x >> np.int64(16))
    out = x.astype(np.float64) / 2 ** 31 - 1.0
    return out[0] if scalar else out


def fill_noise(band, frame, width):
    """Unit-RMS noise for one band, or for a stack of bands. The normalisation is recomputed
    from the values themselves, on both sides, so nothing about it has to be transmitted and
    no rounding rule can disagree."""
    v = hash_noise(width, frame, band)
    if v.ndim == 1:
        return v / float(np.sqrt((v * v).mean()))
    rms = np.sqrt((v * v).mean(axis=-1))
    return v / rms[:, None]


def fill_code(level_db):
    """A dB level onto the 6-bit word. 0 is the floor, which is silence, so leaving a run
    empty is free rather than a special case the decoder has to be told about.

    This is arithmetic, not a numpy call. It sits inside the rate solve, where it runs once per
    run per frame per plane count it tries, and a handful of scalar numpy round/clip calls per
    one is more time than the value it produces is worth. round() on a float is the same
    round-half-to-even that np.round applies to a scalar, and the clip is a clamp.
    """
    c = round((max(float(level_db), FILL_FLOOR) - FILL_FLOOR) / _FILL_STEP)
    return 0 if c <= 0 else (_FILL_MAX if c >= _FILL_MAX else int(c))


def fill_level(code):
    """The inverse, and the only level either side is ever allowed to synthesise with."""
    return FILL_FLOOR + code * _FILL_STEP if code else -1e9


def _dead_bounds(b_band):
    """Inclusive-start, exclusive-end index of each contiguous run of dead bands."""
    dead = np.asarray(b_band) <= 0
    if dead.size == 0 or not dead.any():
        z = np.zeros(0, np.int64)
        return z, z
    d = np.empty(dead.size + 2, np.int8)
    d[0] = 0
    d[-1] = 0
    d[1:-1] = dead
    ch = np.diff(d)
    return np.flatnonzero(ch == 1), np.flatnonzero(ch == -1)


# dead_runs is asked for a lot, and usually about the same allocation: the rate solve walks the
# same frames for every plane count it tries, and a frame's runs only depend on which of its
# bands are dead. The key is that mask, not the whole allocation, so two budgets that close and
# open the same bands share one answer. The arrays in the cache are shared, never written - the
# callers below read runs to index into their own buffers, and nothing writes into a run.
_RUN_CACHE = {}
_RUN_CACHE_MAX = 20000


def dead_runs(b_band):
    """Contiguous runs of dead bands, band-major, as index arrays. The decoder can derive this
    from b alone, which is the entire reason the run is the unit of transmission."""
    dead = np.asarray(b_band) <= 0
    if dead.ndim != 1:
        dead = dead.reshape(-1)
    key = dead.tobytes()
    got = _RUN_CACHE.get(key)
    if got is not None:
        return list(got)
    starts, ends = _dead_bounds(b_band)
    runs = [np.arange(int(s), int(e)) for s, e in zip(starts, ends)]
    if len(_RUN_CACHE) > _RUN_CACHE_MAX:
        _RUN_CACHE.clear()
    _RUN_CACHE[key] = tuple(runs)
    return list(runs)


def run_tilt(code, tilt, j, lo, hi):
    """
    The decoder's own floor for band j of a run: a level word and a slope, no mask and no
    source. The position is normalised across the run, -1 at its first band and +1 at its last,
    which is what makes the tilt mean the same thing for a 2-band run and a 200-band one.
    """
    if not code:
        return -1e9
    mid = (lo + hi) * 0.5
    half = max((hi - lo) * 0.5, 1.0)
    return fill_level(code) + FILL_TILT[int(tilt)] * (j - mid) / half


def _write_run_levels(lvl, runs, levels):
    """Per-band floor of each live run, in dB, written into `lvl` (NaN elsewhere)."""
    for r, (code, tilt) in zip(runs, levels):
        if not int(code):
            continue
        j = np.asarray(r, np.int64)
        lo = float(int(j[0]))
        hi = float(int(j[-1]))
        mid = (lo + hi) * 0.5
        half = max((hi - lo) * 0.5, 1.0)
        pos = j.astype(np.float64)
        lvl[j] = fill_level(int(code)) + FILL_TILT[int(tilt)] * (pos - mid) / half


def _write_shape_levels(lvl, words):
    """A lit shape word replaces whatever the run wrote. Code 0 leaves the run alone."""
    words = np.asarray(words, np.int64)
    lit = np.flatnonzero(words > 0)
    if lit.size == 0:
        return
    lvl[lit] = FILL_FLOOR + words[lit].astype(np.float64) * _FILL_STEP


def _draw_floor(X_band, lvl, frame, width):
    """Unit-RMS noise times the linear gain of `lvl`, for every band that has one.
    Bands share one hash call. The draw for a single band is fill_noise's own formula."""
    idx = np.flatnonzero(~np.isnan(lvl))
    if idx.size == 0:
        return X_band
    v = hash_noise(width, int(frame), idx)
    rms = np.sqrt((v * v).mean(axis=-1))
    gain = 10.0 ** (0.05 * lvl[idx])
    X_band[idx] = (v / rms[:, None]) * gain[:, None]
    return X_band


def fill_bands(X_band, runs, levels, frame, width):
    """
    Replace silence with noise, band by band. Everything the decoder needs is here: b says
    which bands are dead, the level words say how loud, and (frame, band) seeds the same values
    the encoder drew. The encoder calls this too, so its ES is the decoder's sound.

    The seed is the band index, not the run. A run on this material is routinely 100 bands
    wide, and giving all of them the same W values would put a comb of identical bins across
    10 kHz, which is a filter, not a noise floor. Independent draws per band are what makes the
    thing read as air rather than as ringing.
    """
    lvl = np.full(X_band.shape[0], np.nan)
    _write_run_levels(lvl, runs, levels)
    return _draw_floor(X_band, lvl, frame, width)


def write_fill(w, b_band, levels):
    """
    One bit, then a level word and a slope per dead run. A frame with nothing to say pays 1
    bit, which is the whole reason the flag exists: silence between phrases should not cost
    eight bits a run.
    """
    runs = dead_runs(b_band)
    if not runs or not any(int(c) for c, _ in levels):
        w.put(0, 1)
        return
    w.put(1, 1)
    # One field array, not a put() per word. The bit order is the same MSB-first sequence
    # a loop of puts would have emitted: level, then slope, per run.
    lv = np.asarray(levels, np.int64).reshape(-1)
    wd = np.empty(lv.shape, np.int64)
    wd[0::2] = FILL_BITS
    wd[1::2] = FILL_TILT_BITS
    w.put_fields(lv, wd)


def read_fill(r, b_band):
    """The mirror. The run count comes from b, so this reads exactly what was written."""
    runs = dead_runs(b_band)
    if not r.get(1):
        return [(0, 0)] * len(runs)
    return [(r.get(FILL_BITS), r.get(FILL_TILT_BITS)) for _ in runs]


def fill_cost(b_band, levels):
    """What write_fill would charge, for the estimator."""
    runs = dead_runs(b_band)
    if not runs or not any(int(c) for c, _ in levels):
        return 1
    return 1 + (FILL_BITS + FILL_TILT_BITS) * len(runs)


# --------------------------------------------------------------- frame grammar

def run_fill_words_all(cap_all, b_all, mode="safe"):
    """run_fill_words for a whole stack of frames at once.

    Same answers, same order, because it is the same arithmetic done one time instead of one
    time per frame. Each dead run needs the tightest bin of each of the four candidate slopes,
    which is a segment minimum of cap - s * p over that run's bands. Done per frame that is
    about thirty-five numpy calls and four reduceats, and the rate solve runs it for every frame
    of every plane count it tries, so the call count is what takes the time, not the reduction.
    Laid end to end, the runs of a whole window are one flat array and a slope is a single
    reduceat over it, so the cost stops scaling with frames.

    Returns one list per frame of (level, slope) pairs in run order, which is what write_fill,
    fill_cost and _write_run_levels are handed.
    """
    cap = np.asarray(cap_all, np.float64)
    b = np.asarray(b_all, np.int64)
    if cap.ndim == 1:
        cap, b = cap[None], b[None]
    nfr, nb = cap.shape
    if mode != "safe":
        return [run_fill_words(cap[f], b[f], mode) for f in range(nfr)]
    dead = b <= 0
    if not dead.any():
        return [[] for _ in range(nfr)]
    # Run numbers that run across the whole stack: run 0, 1 ... of frame 0, then of frame 1.
    prev = np.zeros_like(dead)
    nxt = np.zeros_like(dead)
    prev[:, 1:] = dead[:, :-1]
    nxt[:, :-1] = dead[:, 1:]
    head = dead & ~prev
    tail = dead & ~nxt
    nrun = head.sum(axis=1)
    seg_base = np.zeros(nfr, np.int64)
    seg_base[1:] = np.cumsum(nrun)[:-1]
    seg = np.where(dead, seg_base[:, None] + np.cumsum(head, axis=1) - 1, -1)
    tot = int(nrun.sum())
    gstart = np.zeros(tot, np.int64)
    gend = np.zeros(tot, np.int64)
    gframe = np.zeros(tot, np.int64)
    fi, ji = np.nonzero(head)
    g = seg[fi, ji]
    gstart[g] = ji
    gframe[g] = fi
    fi, ji = np.nonzero(tail)
    gend[seg[fi, ji]] = ji + 1                    # exclusive, as _dead_bounds returns it
    lo = gstart.astype(np.float64)
    hi = (gend - 1).astype(np.float64)
    mid = (lo + hi) * 0.5
    half = np.maximum((hi - lo) * 0.5, 1.0)
    # The dead bands in frame-then-band order, which is the order np.flatnonzero returns, so a
    # run is a contiguous slice and its start is where the reduceat gets told to cut.
    flat = np.flatnonzero(dead)
    ff = np.floor_divide(flat, nb)
    fb = np.mod(flat, nb)
    vals = cap[ff, fb]
    pos = (fb.astype(np.float64) - mid[seg[ff, fb]]) / half[seg[ff, fb]]
    before = np.cumsum(dead, axis=1) - dead       # dead bands of its own frame before band j
    frame_base = np.zeros(nfr, np.int64)
    frame_base[1:] = np.cumsum(dead.sum(axis=1))[:-1]
    offs = frame_base[gframe] + before[gframe, gstart]
    best_lv = np.full(tot, FILL_FLOOR)
    best_q = np.zeros(tot, np.int64)
    for q, slope in enumerate(FILL_TILT):
        mn = np.minimum.reduceat(vals - slope * pos, offs)
        better = mn > best_lv
        best_lv = np.where(better, mn, best_lv)
        best_q = np.where(better, q, best_q)
    codes = _fill_codes(best_lv)
    # The per-frame path rounds onto the grid, then walks the code down while the level it names
    # is still above the run's own ceiling. Rounding to nearest overshoots by at most half a
    # step, so one pass is enough; the loop is here for the argument, not the data.
    for _ in range(2):
        lev = np.where(codes > 0, FILL_FLOOR + codes * _FILL_STEP, -1e9)
        over = lev > best_lv + 1e-9
        if not over.any():
            break
        codes = np.where(over, codes - 1, codes)
    out = []
    at = 0
    for f in range(nfr):
        n = int(nrun[f])
        out.append([(int(codes[k]), int(best_q[k])) for k in range(at, at + n)])
        at += n
    return out


def run_fill_words(cap_db, b_band, mode="safe"):
    """
    What the encoder sends: one (level, slope) pair per dead run.

    mode="safe" picks the run's level as loud as its own ceiling allows without exceeding it
    anywhere inside it. For a candidate slope s the loudest legal middle is the tightest bin,

        L(s) = min over bands j in the run of ( cap[j] - s * p[j] )

    and the encoder keeps the s that makes that minimum largest. The level word is rounded DOWN,
    because a code that rounds up would make the transmitted floor illegal, and the decoder
    would be playing noise the encoder never measured.

    mode="mean" fills at the run's own average energy and takes no slope. This is what the early
    experiment did, and it is kept because it is the option that SOUNDS most like air, for the
    one reason its level is set by the run rather than by its worst bin. Measured, that is the
    difference between inaudible and audible: about one filled band in five ends up above its
    own mask, and the worst of them by more than 40 dB.

    cap_db is per band. How that ceiling is worked out is a masking question and lives with the
    mask, not here.
    """
    starts, ends = _dead_bounds(b_band)
    if starts.size == 0:
        return []
    cap = np.asarray(cap_db, np.float64)
    if mode == "mean":
        # np.mean on the run, not a reduced sum: a long run's mean is what fill_code rounds,
        # and a different summation order can move that rounding.
        return [(fill_code(float(np.mean(cap[int(s):int(e)]))), 0)
                for s, e in zip(starts, ends)]
    lengths = ends - starts
    idx = np.flatnonzero(np.asarray(b_band) <= 0)
    group = np.cumsum(lengths) - lengths
    lo = starts.astype(np.float64)
    hi = (ends - 1).astype(np.float64)
    rid = np.repeat(np.arange(starts.size), lengths)
    mid = (lo[rid] + hi[rid]) * 0.5
    half = np.maximum((hi[rid] - lo[rid]) * 0.5, 1.0)
    p = (idx.astype(np.float64) - mid) / half
    cap_d = cap[idx]
    best_lv = np.full(starts.size, FILL_FLOOR)
    best_q = np.zeros(starts.size, np.int64)
    for q, slope in enumerate(FILL_TILT):
        mn = np.minimum.reduceat(cap_d - slope * p, group)
        better = mn > best_lv
        best_lv = np.where(better, mn, best_lv)
        best_q = np.where(better, q, best_q)
    out = []
    for lv, q in zip(best_lv, best_q):
        lv = float(lv)
        code = fill_code(lv)
        while code and fill_level(code) > lv + 1e-9:
            code -= 1
        out.append((code, int(q)))
    return out


def band_fill_cap(A_band, thr_band=None):
    """
    The ceiling a band may be filled to: its own energy, and no more than the quietest bin in it
    can hide. The second half is not decoration. A flat floor across a band is masked only if
    its WEAKEST bin masks it, so the band's share of the cap is the minimum threshold over its
    bins - using the band's summed masking capacity, or its mean, lets the quiet bins through,
    which is precisely the mistake that made the first version of this audible.

    Without thr_band this degrades to "fill at the band's energy", which is what the earlier
    experiment did, and is what this file exists to prove is not safe.
    """
    # Last axis is the bins. A stack of frames is the same reduction, one cap per band.
    lvl = 10.0 * np.log10(np.maximum((np.asarray(A_band) ** 2).mean(axis=-1), 10 ** FILL_FLOOR))
    if thr_band is None:
        return lvl
    floor = np.maximum(np.min(np.asarray(thr_band), axis=-1), FILL_FLOOR)
    return np.minimum(lvl, floor)


def shape_level_cap(A_band, thr_band, b_band):
    """What the shape field may paint: a live band keeps its own energy, a dead band takes the
    mask ceiling instead. The distinction is not pedantry. `shape_window` measures the frame's
    loudness off the same array it paints from, so capping every band would drag the frame peak
    down with it and then tell itself the frame had no treble to fill - which reads as a quieter
    file, not a safer one, and costs coverage for no legality. Dead bands are the only place a
    shape word is ever written, so that is the only place a ceiling has to be.
    """
    lvl = band_fill_cap(A_band, None)
    if thr_band is None:
        return lvl
    dead = np.asarray(b_band) <= 0
    return np.where(dead, np.minimum(lvl, band_fill_cap(A_band, thr_band)), lvl)


def shape_reg(sr, n2, width):
    """Which bands the shape field covers: band centres above SHAPE_F0. Derived, never sent -
    the decoder already knows sr, n2 and width from the header, and a field whose position is
    derived is a field that costs nothing to describe. Both sides call this one function, so
    the region cannot drift between them."""
    nc = n2 // 2
    hz = np.arange(nc) * (sr / n2)
    return hz.reshape(nc // width, width).mean(axis=1) >= SHAPE_F0


def shape_block(flags):
    """Bands per shape word, read off the header like the region is. Not sending a file's
    geometry is how a decoder gets to disagree with the encoder about it, and this one is
    worth two header bits rather than a per-frame field."""
    return SHAPE_BLOCK << ((int(flags) & SHAPE_BLOCK_SHIFT) >> SHAPE_BLOCK_SHIFT_W)


def shape_block_shift(block):
    """The header bits that say `block`, which is shape_block's inverse and the encoder's
    only way to be honest about what it wrote. An unrepresentable width is refused here
    rather than encoded as something a decoder will misread into a different stream."""
    k, b = 0, SHAPE_BLOCK
    while b < int(block):
        b, k = b << 1, k + 1
    if b != int(block):
        raise ValueError(f"shape block {int(block)} is not a power-of-two multiple of "
                         f"{SHAPE_BLOCK}")
    return k


def shape_codes(cap_db, b):
    """Each dead band's own level rounded onto the fill word grid. Live bands are 0: the
    shape field never describes a band the mantissas already carry."""
    c = np.round((np.maximum(np.asarray(cap_db, np.float64), FILL_FLOOR) - FILL_FLOOR)
                 / _FILL_STEP)
    c = np.clip(c, 0, _FILL_MAX).astype(np.int64)
    c[np.asarray(b) > 0] = 0
    return c


def _air_keep(level, peak_db):
    """Power gain for one block, and its power-mean level in dB.

    Zero means the word is silence. A partial plus empty neighbours has a flatness near 0,
    and painting the power mean there is a hiss across the whole block. A block far under
    the frame peak is treble the signal barely has, and the same hiss reads as an artifact
    there even when the bands are flat. Both fades are in power, and they multiply.
    """
    lvl = np.asarray(level, np.float64)
    pw = 10.0 ** (lvl / 10.0)
    am = float(pw.mean())
    floor_p = 10.0 ** (FILL_FLOOR / 10.0)
    if not np.isfinite(am) or am <= floor_p:
        return 0.0, FILL_FLOOR
    gm = float(np.exp(np.mean(np.log(np.maximum(pw, floor_p)))))
    flat = gm / am
    db = 10.0 * np.log10(am)
    if flat <= SHAPE_FLAT_LO:
        fg = 0.0
    elif flat >= SHAPE_FLAT_HI:
        fg = 1.0
    else:
        fg = (flat - SHAPE_FLAT_LO) / (SHAPE_FLAT_HI - SHAPE_FLAT_LO)
    below = float(peak_db) - db
    if below >= SHAPE_DEPTH_OFF:
        dg = 0.0
    elif below <= SHAPE_DEPTH_FULL:
        dg = 1.0
    else:
        dg = (SHAPE_DEPTH_OFF - below) / (SHAPE_DEPTH_OFF - SHAPE_DEPTH_FULL)
    return fg * dg, db


def _fill_codes(level_db):
    """`fill_code` on an array. Same round, same clip."""
    x = (np.maximum(np.asarray(level_db, np.float64), FILL_FLOOR) - FILL_FLOOR) / _FILL_STEP
    return np.clip(np.round(x), 0, _FILL_MAX).astype(np.int64)


def _block_codes(chunks, peak_db):
    """One code per row of `chunks`. The same gate as `_air_keep`, on every block at once.

    A frame of treble is about sixty of these, and calling the scalar gate on each one
    was a numpy launch per pair. The means are over the row, which for these widths is
    the same sum the scalar path reduces.
    """
    pw = 10.0 ** (np.asarray(chunks, np.float64) / 10.0)
    am = pw.mean(axis=1)
    floor_p = 10.0 ** (FILL_FLOOR / 10.0)
    dead = ~np.isfinite(am) | (am <= floor_p)
    logged = np.log(np.maximum(pw, floor_p))
    gm = np.exp(logged.mean(axis=1))
    flat = np.where(dead, 0.0, gm / am)
    db = np.where(dead, FILL_FLOOR, 10.0 * np.log10(np.maximum(am, floor_p)))
    span_f = SHAPE_FLAT_HI - SHAPE_FLAT_LO
    span_d = SHAPE_DEPTH_OFF - SHAPE_DEPTH_FULL
    fg = np.clip((flat - SHAPE_FLAT_LO) / span_f, 0.0, 1.0)
    # A scalar peak is the one frame block_shape has. A vector is one peak per
    # row, so a whole window of blocks can share this call. The subtraction is
    # the same one the scalar gate does.
    peak = np.asarray(peak_db, np.float64)
    below = (float(peak) if peak.ndim == 0 else peak) - db
    dg = np.clip((SHAPE_DEPTH_OFF - below) / span_d, 0.0, 1.0)
    gain = np.where(dead, 0.0, fg * dg)
    fade = (gain > 0.0) & (gain < 1.0)
    if fade.any():
        db = db.copy()
        db[fade] = db[fade] + 10.0 * np.log10(gain[fade])
    out = np.zeros(chunks.shape[0], np.int64)
    use = gain > 0.0
    if use.any():
        out[use] = _fill_codes(db[use])
    return out


def block_shape(level_db, block, peak_db):
    """One code per block of dead treble, or 0 when that block is not air.

    The level is the power mean of the band energies, not the mean of their decibels, and
    only once the block has passed the flatness and depth knees in `_air_keep`. Averaging
    decibels would sit a bright partial under a run of quiet neighbours. Averaging power
    across a partial and its empty neighbours is the opposite mistake, and it is the one
    this gate refuses: those neighbours would play at the partial's level. The code is what
    the decoder plays, so the rounding is the level, not a ceiling to stay under.
    """
    level = np.asarray(level_db, np.float64)
    m = int(level.size)
    if m == 0:
        return np.zeros(0, np.int64)
    block = int(block)
    nblk = (m + block - 1) // block
    out = np.zeros(nblk, np.int64)
    nfull = m // block
    if nfull:
        out[:nfull] = _block_codes(level[:nfull * block].reshape(nfull, block), peak_db)
    if nfull < nblk:
        gain, db = _air_keep(level[nfull * block:], peak_db)
        if gain > 0.0:
            if gain < 1.0:
                db = db + 10.0 * np.log10(gain)
            out[nfull] = fill_code(db)
    return out


def shape_window(level_db, bits, reg, block=SHAPE_BLOCK):
    """Shape field of every frame. The codes match `shape_field` one frame at a time.

    The gate does not care which run a block came from, only the block's bands
    and that frame's peak. Stacking the blocks is one launch instead of one
    per pair, which is what the rate solve was repeating for every plane count
    it tried.
    """
    level = np.asarray(level_db, np.float64)
    bits = np.asarray(bits)
    if level.ndim == 1:
        level = level.reshape(1, -1)
        bits = np.asarray(bits).reshape(1, -1)
    nfr = int(level.shape[0])
    block = int(block)
    reg = np.asarray(reg, bool)
    fields = []
    groups = {}
    peaks_g = {}
    where = {}
    for f in range(nfr):
        row_l = level[f]
        peak = float(row_l.max()) if row_l.size else FILL_FLOOR
        row = []
        runs = dead_runs(bits[f])
        for ri, r in enumerate(runs):
            sel = r[reg[r]]
            if sel.size == 0:
                row.append(None)
                continue
            nblk = (int(sel.size) + block - 1) // block
            dest = np.zeros(nblk, np.int64)
            row.append(dest)
            nfull = int(sel.size) // block
            if nfull:
                groups.setdefault(block, []).append(
                    row_l[sel[:nfull * block]].reshape(nfull, block))
                peaks_g.setdefault(block, []).append(np.full(nfull, peak, np.float64))
                where.setdefault(block, []).append((dest, 0, nfull))
            tail = int(sel.size) - nfull * block
            if tail:
                groups.setdefault(tail, []).append(row_l[sel[nfull * block:]].reshape(1, tail))
                peaks_g.setdefault(tail, []).append(np.full(1, peak, np.float64))
                where.setdefault(tail, []).append((dest, nfull, 1))
        fields.append(row)
    for width, parts in groups.items():
        codes = _block_codes(np.concatenate(parts, axis=0), np.concatenate(peaks_g[width]))
        at = 0
        for dest, start, n in where[width]:
            dest[start:start + n] = codes[at:at + n]
            at += n
    return fields


def shape_field(level_db, b, reg, block=SHAPE_BLOCK):
    """The shape grammar of one frame, as one block-code stream per dead run, parallel to
    dead_runs(b). `level_db` is each band's own energy, and the frame peak taken from it is
    what "not a lot of treble" is measured against. None where the region does not touch
    the run, which is derivable from b and costs nothing, and a run whose blocks round to
    silence sends a single flag bit."""
    level = np.asarray(level_db, np.float64)
    peak = float(level.max()) if level.size else FILL_FLOOR
    field = []
    for r in dead_runs(b):
        sel = r[reg[r]]
        if sel.size == 0:
            field.append(None)
            continue
        field.append(block_shape(level[sel], block, peak))
    return field


def shape_expand(field, b, reg, block=SHAPE_BLOCK):
    """The per-band words the reconstruct formula uses, derived from the per-run streams. Both
    sides take this exact path - the encoder before synthesising, the decoder after reading -
    so what the encoder measures is the decoder's sound rather than its intent."""
    out = np.zeros(len(b), np.int64)
    for r, lv in zip(dead_runs(b), field):
        if lv is None:
            continue
        sel = r[reg[r]]
        m = (sel.size + block - 1) // block
        for t, j in enumerate(sel):
            c = int(lv[min(t // block, m - 1)])
            if c > 0:
                out[int(j)] = c
    return out


def write_shape(w, field):
    """Per touched run: 1 flag bit; if anything is lit, the first block's absolute word, and
    for a multi-block run a 5-bit Rice parameter and the delta stream. Deltas rather than words
    because mask shapes are smooth, and a delta that says "same" costs k+1 bits."""
    for lv in field:
        if lv is None:
            continue
        if not lv.any():
            w.put(0, 1)
            continue
        w.put(1, 1)
        w.put(int(lv[0]), FILL_BITS)
        if lv.size > 1:
            z = zz_vec(np.diff(lv).astype(np.int64))
            k = pick_k(z)
            w.put(k, 5)
            write_rice(w, z, k)


def read_shape(r, b, reg, block=SHAPE_BLOCK):
    """The mirror: runs and block counts come from b, so this reads exactly what was written."""
    out = np.zeros(len(b), np.int64)
    for rv in dead_runs(b):
        sel = rv[reg[rv]]
        if sel.size == 0:
            continue
        if not r.get(1):
            continue
        lv = [r.get(FILL_BITS)]
        m = (sel.size + block - 1) // block
        if m > 1:
            k = r.get(5)
            d = [unz(v) for v in read_rice(r, m - 1, k)]
            lv.extend(lv[-1] + dt for dt in d)
        for t, j in enumerate(sel):
            c = int(lv[min(t // block, m - 1)])
            if c > 0:
                out[int(j)] = c
    return out


def shape_cost(field):
    """What write_shape would charge, for the estimator."""
    tot = 0
    for lv in field:
        if lv is None:
            continue
        tot += 1
        if lv.any():
            tot += FILL_BITS
            if lv.size > 1:
                z = zz_vec(np.diff(lv).astype(np.int64))
                k = pick_k(z)
                _, wd = rice_fields(z, k)
                tot += 5 + int(wd.sum())
    return tot


# ------------------------------------------------------------- pyramid pulses
# A live band is one gain and one shape. The gain is the band's energy. The shape is a
# signed integer vector on the L1 pyramid, length = every bin in the band, and the
# reconstruction is gain * y / ||y||. K is however many pulses the band's own bits can
# hold. Dead bands are not touched, so noise fill on a band the allocator left closed
# stays noise. The index is the combinatorial number of the vector, written in 16-bit
# chunks so a Luau port never holds the whole integer in one bit32 op. K stops at 64
# because that is already more pulses than a 16-bin band can usefully spend.
PVQ_K_MAX = 64
# How far the tables go, as distinct from how far a band is allowed to reach unless it is told
# to. The index of a K-pulse vector is the whole cost of K, so K is not free and the cap is a
# rate decision rather than a size limit - but the tables are what make a named K decodable at
# all, and every K an old file can name is already inside this, which is why raising the ceiling
# for a new encode cannot make an old file decode differently.
#
# It was worth asking whether 64 is the wrong cap, and it is not. Widening it to 128 and 256 was
# measured on moz.wav: nothing at all changes at 12 or 24 kbit/s, and at 64 kbit/s it buys
# 20.30 -> 20.57 dB for 1% more bytes, which is a wash. Taking it *down* is what moves, ES
# dropping from 20.30 to 18.91 at 32 pulses and to 15.68 at 16, so the marginal pulse is worth
# roughly what its index costs right up to about 64 and not much past that. That is what the
# rule in pvq_k_for_depth already says, which is the reason it is written as a budget and not a
# constant: the encoder is choosing K the way it chooses depth, and on this material both land
# on the same number.
PVQ_K_TABLE = 256
# How far the pyramid tables reach in bins, which is not the same question as how wide a band the
# encoder writes. A band wider than this is coded by the recursive count, and that is exact but it
# walks a Python stack one pulse deep: a 32-bin band at a low rate recursed nine hundred frames
# deep and died in the decoder rather than in the encoder, which is the worst place for it, since
# the encoder's own measure still said the file was good. The reach is therefore a capability the
# tables have to hold, not a preference about what looks tidy.
#
# Building to 64 bins costs 3 ms of import and nothing else, so the width an encoder is allowed to
# write is set by BAND_MAX in rsc3_encode, which reads this, rather than by what the tables happen
# to hold. What a wider band does *not* get for free is speed: past 62 bits of index the compiled
# walk stands aside and the reference walk answers, and an index that big arrives long before the
# band does anything interesting.
_PVQ_N = 64
_PVQ_COUNT = {}


def _pvq_tables(n_max, k_max):
    """V(n, k) for the band width this codec actually uses. The recurrence is the one
    pvq_count walks. A list beats a dict lookup on every pulse of every band."""
    tab = [[0] * (k_max + 1) for _ in range(n_max + 1)]
    for n in range(1, n_max + 1):
        tab[n][0] = 1
    for k in range(1, k_max + 1):
        tab[1][k] = 2
    for n in range(2, n_max + 1):
        row, prev = tab[n], tab[n - 1]
        acc = row[0]
        for k in range(1, k_max + 1):
            acc = acc + prev[k] + prev[k - 1]
            row[k] = acc
    bits = [[(v - 1).bit_length() if v > 1 else 0 for v in row] for row in tab]
    return tab, bits


_PVQ_TAB, _PVQ_BITS = _pvq_tables(_PVQ_N, PVQ_K_TABLE)

# The same table as one numpy integer array, clamped to the bound a walk's index is checked
# against, so a compiled walk can hold the whole search in int64. A count larger than the bound
# becomes the bound, and that is exact rather than close: an index below the bound is below the
# true count as well, so every comparison lands in the same place, and a count that large is
# never the one subtracted.
_PVQ_BOUND = 1 << 62
_PVQ_CNT = np.zeros((_PVQ_N + 1, PVQ_K_TABLE + 1), np.int64)
_PVQ_NBITS = np.zeros((_PVQ_N + 1, PVQ_K_TABLE + 1), np.int64)
for _n in range(_PVQ_N + 1):
    for _k in range(PVQ_K_TABLE + 1):
        _v = _PVQ_TAB[_n][_k]
        _PVQ_CNT[_n, _k] = _v if _v.bit_length() <= 62 else _PVQ_BOUND
        _PVQ_NBITS[_n, _k] = _PVQ_BITS[_n][_k]


def pvq_claims(b):
    """Every live band. Those are the ones whose scalar mantissas are not in the file.
    A dead band has no scalar field to replace."""
    return np.asarray(b) > 0


def pvq_count(n, k):
    """How many signed integer vectors of length n have L1 norm k. The zero vector is the
    one vector of norm 0. Length 1 is just +k and -k."""
    n, k = int(n), int(k)
    if k < 0 or n <= 0:
        return 0
    if n <= _PVQ_N and k <= PVQ_K_TABLE:
        return _PVQ_TAB[n][k]
    hit = _PVQ_COUNT.get((n, k))
    if hit is not None:
        return hit
    if k == 0:
        v = 1
    elif n == 1:
        v = 2
    else:
        # Last coordinate 0, or +m, or -m. This is the split the index walks.
        v = pvq_count(n, k - 1) + pvq_count(n - 1, k) + pvq_count(n - 1, k - 1)
    _PVQ_COUNT[(n, k)] = v
    return v


def _index_bits(n, k):
    v = pvq_count(n, k)
    if v <= 1:
        return 0
    return (v - 1).bit_length()


def _gamma_bits(k):
    return 2 * (int(k).bit_length() - 1) + 1


def _write_gamma(w, k):
    """Elias gamma. K = 1 is a single 1-bit, which is the common low-rate band."""
    k = int(k)
    z = k.bit_length() - 1
    if z:
        w.put(0, z)
    w.put(k, z + 1)


def _read_gamma(r):
    z = 0
    while r.get(1) == 0:
        z += 1
        if z > 16:
            break
    if z == 0:
        return 1
    return (1 << z) | r.get(z)


def _put_wide(w, value, width):
    """MSB-first, 16 bits at a time. Writer.put stores the value in an int64."""
    value = int(value)
    left = int(width)
    while left > 0:
        n = 16 if left > 16 else left
        left -= n
        w.put((value >> left) & ((1 << n) - 1), n)


def _enc_pvq(y, n, k):
    """Combinadic index. For a 16-bin band the counts are the table, and the walk
    stays on the same list: slicing a new array per coordinate was most of the cost.
    The split is unchanged, last coordinate 0, then +m, then -m."""
    n = int(n)
    k = int(k)
    if 0 < n <= _PVQ_N and 0 <= k <= PVQ_K_MAX:
        seq = y if isinstance(y, list) else [int(v) for v in y[:n]]
        return _enc_pvq_tab(seq, n, k)
    return _enc_pvq_slow(y, n, k)


def _enc_pvq_tab(y, n, k):
    tab = _PVQ_TAB
    total = 0
    while True:
        if k == 0 or n == 0:
            return total
        if n == 1:
            return total + (0 if y[0] > 0 else 1)
        last = y[n - 1]
        if last == 0:
            n -= 1
            continue
        base = tab[n - 1][k]
        m = -last if last < 0 else last
        if m > 1:
            row = tab[n - 1]
            acc = 0
            for t in range(k - 1, k - m, -1):
                acc += row[t]
            base += acc << 1
        if last < 0:
            base += tab[n - 1][k - m]
        total += base
        n -= 1
        k -= m


def _enc_pvq_slow(y, n, k):
    if k == 0 or n == 0:
        return 0
    if n == 1:
        return 0 if int(y[0]) > 0 else 1
    last = int(y[n - 1])
    if last == 0:
        return _enc_pvq(y[:n - 1], n - 1, k)
    base = pvq_count(n - 1, k)
    m = abs(last)
    for j in range(1, m):
        base += 2 * pvq_count(n - 1, k - j)
    if last < 0:
        base += pvq_count(n - 1, k - m)
    return base + _enc_pvq(y[:n - 1], n - 1, k - m)


def pvq_index(y):
    """Index and codebook size of one signed pulse vector."""
    y = np.asarray(y, np.int64).reshape(-1)
    k = int(np.abs(y).sum())
    n = int(y.size)
    return _enc_pvq(y, n, k), pvq_count(n, k)


def _pvq_shape(index, n, k):
    """The point of the pyramid `index` names, as coefficients from the top bin down.

    The recursion is the shape of the answer, and it wrote a numpy slice per level of itself:
    sixteen slice objects and an array to carry sixteen small integers, for every live band of
    every frame the decoder reads. This is the same walk with no recursion and no slicing.

    The arithmetic is a copy of the recursion's, including the two places where it looks like a
    slip: the +m branch is handed the index as it arrived, and the -m branch is handed the index
    after the one subtraction the check itself needed. Neither gets the subtraction that belongs
    to stepping to the next m. Reading that as symmetry and levelling it up produces a different
    vector for the same index, so this function is written to match, not to look tidy.
    """
    out = []
    while True:
        if k == 0 or n == 0:
            out.extend([0] * n)
            return out
        if n == 1:
            out.append(k if index == 0 else -k)
            return out
        left = pvq_count(n - 1, k)
        if index < left:
            out.append(0)
            n -= 1
            continue
        index -= left
        m = 1
        while m <= k:
            block = pvq_count(n - 1, k - m)
            if index < block:
                out.append(m)
                k -= m
                n -= 1
                break
            index -= block
            if index < block:
                out.append(-m)
                k -= m
                n -= 1
                break
            index -= block
            m += 1
        else:
            out.extend([0] * n)
            return out


def _pvq_shape_fast(index, n, k):
    """The pyramid walk, as int64 coordinates from the top bin down.

    A real stream asks for this a quarter of a million times per decode, and almost every ask is
    inside what the table and a 64-bit index cover, so the check is worth its cost: a width the
    table was built for, a pulse count it holds, an index under the bound. Beyond that the big
    integers are genuinely big and the reference walk is what answers.
    """
    index = int(index)
    n = int(n)
    k = int(k)
    if (_BITS.pvq_walk is not None and 1 <= n <= _PVQ_N and 0 <= k <= PVQ_K_TABLE
            and index.bit_length() <= 62):
        return _BITS.pvq_walk(_PVQ_CNT, index, n, k)
    return np.asarray(_pvq_shape(index, n, k), np.int64)


def pvq_from_index(index, n, k):
    return _pvq_shape_fast(int(index), int(n), int(k))[::-1]


def pvq_synth_band(y, q, E, b):
    """Band energy times the unit-energy shape.

    `q * 2**(E-b)` is the band's L2. E is the exponent stored for that gain, and y
    is the integer point on the pyramid. Dividing by ||y|| puts the gain on the
    band. The tallest bin is then whatever share of the shape it has, and the
    other bins keep theirs.
    """
    y = np.asarray(y, np.float64)
    if int(q) == 0:
        return np.zeros(y.shape, np.float64)
    n2 = float(np.dot(y, y))
    if n2 <= 0.0:
        return np.zeros(y.shape, np.float64)
    gain = int(q) * 2.0 ** (float(E) - float(b))
    return y * (gain / np.sqrt(n2))


def _pvq_place(x, k):
    """Greedy pyramid search. Pulses may stack, so one gain can follow a slope."""
    x = np.asarray(x, np.float64)
    n = int(x.size)
    y = np.zeros(n, np.int64)
    if k <= 0 or n == 0:
        return y, 0.0
    ax = np.abs(x)
    sx = np.sign(x).astype(np.int64)
    sx[sx == 0] = 1
    corr = 0.0
    ny = 0.0
    for _ in range(int(k)):
        score = (corr + ax) ** 2 / (ny + 2.0 * np.abs(y) + 1.0)
        i = int(np.argmax(score))
        if ax[i] <= 0.0:
            break
        ny += 2.0 * abs(int(y[i])) + 1.0
        corr += float(ax[i])
        y[i] += int(sx[i])
    if ny <= 0.0:
        return y, 0.0
    return y, corr / ny


def pvq_k_for_depth(b, width, k_cap=0):
    """Pulses the band's shape bits can hold.

    The shape budget is one bit per bin of the band's depth. The gain is sent
    on its own, in `b` bits, so it stays out of this. K is the largest pyramid
    that fits, which is the resolution the band was given.
    """
    b = int(b)
    width = int(width)
    if b <= 0 or width <= 0:
        return 0
    shape_bits = width * b
    k_hi = PVQ_K_MAX if not k_cap else min(PVQ_K_TABLE, max(int(k_cap), 1))
    bits_row = _PVQ_BITS[width] if 0 < width <= _PVQ_N else None
    best = 1
    for k in range(1, k_hi + 1):
        ib = bits_row[k] if bits_row is not None else _index_bits(width, k)
        if _gamma_bits(k) + ib <= shape_bits:
            best = k
        else:
            break
    return best


def _k_vec(b, width, k_cap):
    """`pvq_k_for_depth` for many bands. The cost grows with K, so the largest K
    that fits is the count of sizes that fit."""
    b = np.asarray(b, np.int64)
    if int(width) > _PVQ_N:
        return np.array([pvq_k_for_depth(int(v), width, k_cap) for v in b], np.int64)
    bits = np.asarray(_PVQ_BITS[int(width)], np.int64)
    gamma = np.array([_gamma_bits(k) for k in range(PVQ_K_TABLE + 1)], np.int64)
    cost = gamma + bits
    shape_bits = np.int64(width) * b
    ok = cost[None, :] <= shape_bits[:, None]
    ok[:, 0] = False
    ok[:, (PVQ_K_MAX if not k_cap else min(int(k_cap), PVQ_K_TABLE)) + 1:] = False
    return np.where(b > 0, ok.sum(axis=1), 0).astype(np.int64)


def _choose_bands(x, b, k_cap=0):
    """Gain and shape for many bands at once.

    One row is what `_choose_band` used to do alone: energy in bin order, then
    exactly K greedy pulses. The score is `(corr + |x|)^2 / (ny + 2*|y| + 1)`,
    and a tie keeps the earlier bin. Stacking the rows is the same arithmetic
    per band, which is what makes a loud file's placement cheap. On a machine
    with the JIT the pulses go through the compiled per-band form, which stops
    walking a band once it is done; the block below is the same rule for every
    other box, and `perf_check` is what says they agree.
    """
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    b = np.asarray(b, np.int64).reshape(-1)
    nband, width = x.shape
    y = np.zeros((nband, width), np.int64)
    energy = np.zeros(nband, np.float64)
    for i in range(width):
        energy += x[:, i] * x[:, i]
    live = (energy > 0.0) & (b > 0)
    q = np.zeros(nband, np.int64)
    E = np.zeros(nband, np.int64)
    if live.any():
        rms = np.zeros(nband, np.float64)
        rms[live] = energy[live] ** 0.5
        Ee = np.zeros(nband, np.float64)
        Ee[live] = np.ceil(np.log2(rms[live]))
        E = Ee.astype(np.int64)
        step = np.zeros(nband, np.float64)
        step[live] = 2.0 ** (E[live].astype(np.float64) - b[live].astype(np.float64))
        good = live & (step > 0.0)
        ratio = np.zeros(nband, np.float64)
        ratio[good] = rms[good] / step[good]
        lim = (np.int64(1) << np.maximum(b, 0)) - 1
        qq = np.clip(np.round(ratio), 0, lim.astype(np.float64)).astype(np.int64)
        q = np.where(good, qq, 0)
        kk = _k_vec(b, width, k_cap)
        if _BITS.pvq_place is not None:
            # One band at a time, so a band that stopped is not walked again. The rule and the
            # arithmetic are the ones just below, in the same order, which is why perf_check can
            # call it a byte-for-byte match rather than a close one.
            y = _BITS.pvq_place(np.ascontiguousarray(x, np.float64),
                                np.where(live, kk, 0).astype(np.int64))
            y[np.abs(y).sum(axis=1) == 0, 0] = 1
            return q, y, E
        ax = np.abs(x)
        # A silent bin has no direction. np.sign(0) is 0, which would plant a
        # later pulse with no sign, so a non-negative sample stays positive.
        sx = np.where(x >= 0.0, np.int64(1), np.int64(-1))
        ay = np.zeros((nband, width), np.float64)
        corr = np.zeros(nband, np.float64)
        ny = np.zeros(nband, np.float64)
        left = np.where(live, kk, 0)
        active = left > 0
        rows = np.arange(nband)
        while active.any():
            den = ny[:, None] + 2.0 * ay + 1.0
            num = corr[:, None] + ax
            score = (num * num) / den
            pick = np.argmax(score, axis=1)
            best_ax = ax[rows, pick]
            step_on = active & (best_ax > 0.0)
            if not step_on.any():
                break
            sel = rows[step_on]
            ii = pick[step_on]
            ay_i = ay[sel, ii]
            ny[sel] = ny[sel] + 2.0 * ay_i + 1.0
            corr[sel] = corr[sel] + best_ax[step_on]
            y[sel, ii] += sx[sel, ii]
            ay[sel, ii] += 1.0
            left[sel] -= 1
            active = left > 0
    y[np.abs(y).sum(axis=1) == 0, 0] = 1
    return q, y, E


def _synth_bands(y, q, E, b):
    """`pvq_synth_band` on many bands. The pulse vector is small integers, so
    its energy is an exact integer and the square root is the scalar one."""
    yf = np.asarray(y, np.float64)
    n2 = (yf * yf).sum(axis=1)
    gain = np.asarray(q, np.float64) * 2.0 ** (
        np.asarray(E, np.float64) - np.asarray(b, np.float64))
    scale = np.where((np.asarray(q) > 0) & (n2 > 0.0), gain / np.sqrt(n2), 0.0)
    return yf * scale[:, None]


def _choose_band(x, b, k_cap):
    """Gain and shape for one band. The batch placer does the work."""
    q, y, E = _choose_bands(np.asarray(x, np.float64).reshape(1, -1), [b], k_cap)
    return int(q[0]), y[0], int(E[0])


def plan_pvq_frames(A3, E, b, k_cap=0, synth=True):
    """Every frame's pyramid bands in one pass.

    `A3` is (frames, bands, bins). The returned records follow `plan_pvq`:
    None on a scalar band, and `(q, y)` on a pyramid band. Exponents on those
    bands are the band-energy exponents. `synth` is the samples reconstruct
    plays. A rate check that only needs the payload skips them.
    """
    A3 = np.asarray(A3, np.float64)
    E = np.array(E, copy=True)
    b = np.asarray(b)
    frames, nbands, width = A3.shape
    ff, jj = np.nonzero(b > 0)
    recs = [[None] * nbands for _ in range(frames)]
    if ff.size == 0:
        pulses = np.zeros(A3.shape, np.float64) if synth else None
        return recs, pulses, E
    q, y, Ee = _choose_bands(A3[ff, jj], b[ff, jj], k_cap)
    E[ff, jj] = Ee
    pulses = None
    if synth:
        pulses = np.zeros(A3.shape, np.float64)
        pulses[ff, jj] = _synth_bands(y, q, Ee, b[ff, jj])
    for n in range(int(ff.size)):
        recs[int(ff[n])][int(jj[n])] = (int(q[n]), y[n])
    return recs, pulses, E


def plan_pvq(A, E, b, mant, sr, n2, width, k_cap=0):
    """Per-band records for one frame, the samples to play, and the exponents to store.

    A record is None on a band that stays scalar. Claimed bands are (q, y). The
    returned exponents are the band-energy exponents on those bands and the
    caller's exponents everywhere else, which is what write_frame has to send:
    the decoder rebuilds the gain from that exponent and q.
    """
    A = np.asarray(A, np.float64)
    recs, pulses, E = plan_pvq_frames(
        A.reshape(1, A.shape[0], A.shape[1]),
        np.asarray(E).reshape(1, -1),
        np.asarray(b).reshape(1, -1),
        k_cap=k_cap)
    return recs[0], pulses[0], E[0]


def pvq_bands_cost(recs, b):
    """Bits write_frame will spend on the pyramid bands of one frame."""
    tot = 0
    b = np.asarray(b)
    for j, rec in enumerate(recs):
        if rec is None:
            continue
        _q, y = rec
        k = int(np.abs(y).sum())
        tot += _gamma_bits(k) + int(b[j]) + _index_bits(int(y.size), k)
    return tot


_MODEL_COST = {}


def _model_cost_table(width, hi):
    """gamma(K) + depth + index bits, for depth 0..hi. K is `pvq_k_for_depth`."""
    width = int(width)
    hi = max(int(hi), 0)
    hit = _MODEL_COST.get(width)
    if hit is not None and hit.size > hi:
        return hit
    top = max(hi, MAX_BITS)
    table = np.zeros(top + 1, np.int64)
    for depth in range(1, top + 1):
        k = pvq_k_for_depth(depth, width)
        table[depth] = _gamma_bits(k) + depth + _index_bits(width, k)
    _MODEL_COST[width] = table
    return table


def pvq_model_costs(band_bits, width):
    """Per-frame pyramid bits the rate model charges.

    Same total as `pvq_bands_cost` on a stand-in that puts all K pulses in the
    first bin. The real encoder places those pulses, and the count it writes is
    this K unless a band runs out of energy first. One payload sample corrects
    that. Depth 0 is not a pyramid.
    """
    b = np.asarray(band_bits, np.int64)
    if b.size == 0:
        return np.zeros(b.shape[0], np.int64)
    table = _model_cost_table(width, int(b.max()))
    return table[b].sum(axis=1)


def write_pvq_bands(w, recs, b):
    """Every pyramid band of one frame, as one field.

    A band's piece is short and fixed in shape: a gamma-coded pulse count, that band's gain, and
    the index written 16 bits at a time. With 128 live bands that is four put() calls per band,
    which was where a loud file spent the encode; collected into value and width arrays it is
    one field, and the writer packs the frame once.

    The gamma and the gain go together, because their combined width is well under 63 bits and
    the index cannot: a band with many pulses has an index wider than one numpy integer holds,
    which is the whole reason _put_wide exists.
    """
    vals = []
    wds = []
    for j, rec in enumerate(recs):
        if rec is None:
            continue
        q, y = rec
        y = np.asarray(y, np.int64)
        k = max(int(np.abs(y).sum()), 1)
        idx, ncode = pvq_index(y)
        bits = (int(ncode) - 1).bit_length() if ncode > 1 else 0
        z = k.bit_length() - 1
        wq = int(b[j])
        vals.append((k << wq) | int(q))
        wds.append(2 * z + 1 + wq)
        left = bits
        while left > 0:
            m = 16 if left > 16 else left
            left -= m
            vals.append((idx >> left) & ((1 << m) - 1))
            wds.append(m)
    if vals:
        w.put_fields(np.asarray(vals, np.int64), np.asarray(wds, np.int64))


def read_pvq_band(r, width, E, b):
    """One pyramid band, read and synthesised.

    read_frame does this for a whole frame's live bands in one go, so this is what is left for
    callers that want a single band.
    """
    k = _read_gamma(r)
    q = r.get(int(b))
    ncode = pvq_count(width, k)
    bits = (ncode - 1).bit_length() if ncode > 1 else 0
    idx = r.get(bits) if bits else 0
    return pvq_synth_band(pvq_from_index(idx, width, k), q, E, b)


def fill_codes_all(A3all, ball, thr_all=None, mode="safe"):
    """
    Per-frame transport words for a whole file, derived only from the analysis, the mask and the
    allocation - so the estimator that picks a bitrate charges exactly what the writer will
    write, without being told anything extra. One iteration of the zip is one frame.
    """
    out = []
    thr_all = [None] * len(A3all) if thr_all is None else thr_all
    for A3, b, thr in zip(A3all, ball, thr_all):
        out.append(run_fill_words(band_fill_cap(A3, thr), b, mode))
    return out


def write_frame(w, E_band, b_band, mant, prev_E, prev_bits, width, fill=None, shape=None,
                pvq=None):
    """
    Append one frame's bands to `w`. The writer is told E, b and the mantissas and invents
    nothing: the encoder derives them with quantise_bands, and what the writer serialises is
    exactly what the encoder reconstructed.

    fill is this frame's dead-run level codes, or None for no fill field. When present it goes
    last, after the mantissas, because the run structure is derived from b and the decoder can
    only read it once it has those counts.

    Returns (E, field_bits), counting this frame's cost in each of the four fields, read off
    the writer itself. The size report is then the truth about the file rather than a model of
    it, and it cannot drift when the grammar changes.
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
    # Pyramid bands are not in this field. With the flag clear, pack is every live band and
    # the bits are the ones this function has always written.
    pack = live
    if pvq is not None:
        claimed = np.array([rec is not None for rec in pvq], dtype=bool)
        pack = live & ~claimed
    flat = pack & (cost_flat < cost_gap)
    sizes = np.where(pack, 1 + np.where(flat, cost_flat, cost_gap), 0)
    start = np.zeros_like(sizes)
    start[1:] = np.cumsum(sizes)[:-1]
    total = int(sizes.sum())
    if total:
        out = np.zeros(total, np.uint8)
        scatter(out, flat[pack].astype(np.int64), np.ones(int(pack.sum()), np.int64), start[pack])
        jg = np.flatnonzero(pack & ~flat)
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
    # Live bands, in band order, where the scalar mantissas would have been.
    npvq = 0
    if pvq is not None:
        t_pvq = w.bits()
        write_pvq_bands(w, pvq, b)
        npvq = w.bits() - t_pvq
    # The fill field, last, because its run structure is derived from b and the decoder only
    # knows b after reading the counts. Nothing new is invented here: these are the very codes
    # the encoder synthesised with, so the bits and the sound cannot disagree.
    t3 = w.bits()
    nfill = 0
    if fill is not None:
        write_fill(w, b, fill)
        nfill = w.bits() - t3
    # The shape field, last of all: like the fill, its run structure is derived from b, so it
    # can only be read once the counts are known. Nothing here re-decides the levels - the same
    # per-run streams were synthesised with on both sides of reconstruct_bands.
    nshape = 0
    if shape is not None:
        t4 = w.bits()
        write_shape(w, shape)
        nshape = w.bits() - t4
    # E is returned as the decoder's own state: live bands updated, dead ones parked. Handing
    # this back is what makes the retention rule the same in both directions.
    return E_next, (t2 - t1, t1 - t0, w.bits() - t2 - nfill - nshape - npvq,
                    nfill, nshape, npvq)


def frame_costs(Eall, ball, mant, width, fill_levels=None, shape_fields=None,
                pvq_frames=None, pvq_costs=None):
    """
    What one frame's fields cost, in bits, computed without writing a single bit.

    This exists because a rate search has no business encoding the file 15 times to answer a
    question that is arithmetic: the mantissa field is a count plus (gap, sign, magnitude) per
    nonzero, and the two delta fields are Rice codewords whose lengths are known the moment k
    is chosen. Returns (scale, bits, mantissa, fill, shape, pvq) totals over the given frames.

    The numbers have to match write_frame's byte measurement. That is checked against a packed
    payload in the round trip, not assumed, because an estimator that quietly drifts would
    make every bitrate claim wrong in the same direction forever.
    """
    gb, cb = gap_bits(width), count_bits(width)
    ball = np.asarray(ball, np.int64)
    Eall = np.asarray(Eall, np.int64)
    mant = np.asarray(mant, np.int64)
    nfr, nb = ball.shape
    gb_n, cb_n = gap_bits(nb), count_bits(nb)
    tot = np.zeros(6, np.int64)
    # The two delta fields and the mantissa field are all reductions over a frame stack, and
    # each is one arithmetic question: given the codeword widths, what does the frame cost.
    # Asked frame by frame that is a few hundred small numpy calls per budget the search tries,
    # so the rows go through together and the loop is left to the fields that are ragged.
    prev_b = np.zeros_like(ball)
    if nfr > 1:
        prev_b[1:] = ball[:-1]
    Z = zz_vec(ball - prev_b)
    changed = ball != prev_b
    nch = changed.sum(axis=1)
    _, dense = _rice_rows(Z)
    _, sparse_w = _rice_rows(np.where(changed, Z, 0), counts=nch)
    sparse = np.where(nch > 0, 7 + cb_n + nch * gb_n + sparse_w, 2)
    tot[1] = int(np.minimum(dense + 6, sparse).sum())
    # The scalefield is delta-coded against the last frame that had this band live, which is
    # the same retention rule the writer runs, and a dead band's exponent parks rather than
    # moving. Forward-filled here, priced with the live bins only.
    Pe = np.empty_like(ball)
    prev_E = np.zeros(nb, np.int64)
    for f in range(nfr):
        Pe[f] = prev_E
        prev_E = np.where(ball[f] > 0, Eall[f], prev_E)
    live = ball > 0
    nlive = live.sum(axis=1)
    _, elen = _rice_rows(np.where(live, zz_vec(Eall - Pe), 0), counts=nlive)
    tot[0] = int(np.where(nlive > 0, 5 + elen, 0).sum())
    scalar = live
    if pvq_costs is not None:
        # A pyramid band writes no mantissas: its bits are the sixth field, priced from the
        # depth, and every live band is a pyramid, so this is the whole of the frame's cost
        # beyond the two delta fields.
        tot[5] = int(np.sum(pvq_costs))
        scalar = None
    if scalar is not None:
        n = (mant != 0).sum(axis=2)
        b1 = np.maximum(ball, 1)
        per = 1 + np.minimum(cb + n * (gb + 1 + b1), width * (1 + b1))
        tot[2] = int(np.where(scalar, per, 0).sum())
    if pvq_frames is not None:
        # The mask comes from the records the writer was handed, so which bands the scalar
        # field skipped is a per-frame list. This is the path that measures a real payload.
        for f in range(nfr):
            claimed = np.array([rec is not None for rec in pvq_frames[f]], dtype=bool)
            tot[2] += int(np.where(live[f] & ~claimed,
                                   1 + np.minimum(cb + (mant[f] != 0).sum(axis=1) *
                                                  (gb + 1 + np.maximum(ball[f], 1)),
                                                  width * (1 + np.maximum(ball[f], 1))),
                                   0).sum())
            tot[5] += pvq_bands_cost(pvq_frames[f], ball[f])
    if fill_levels is not None:
        for f in range(nfr):
            tot[3] += fill_cost(ball[f], fill_levels[f])
    if shape_fields is not None:
        for f in range(nfr):
            tot[4] += shape_cost(shape_fields[f])
    return tot
def read_frame(r, prev_E, prev_bits, width, fill=False, shape=False, reg=None,
               pvq=False, sr=0, n2=0, sblock=SHAPE_BLOCK):
    """
    The mirror of write_frame, and the only path a decoder takes. It returns E, b and the
    mantissas; the caller reconstructs with reconstruct_bands, which is the same function the
    encoder used, so the decoded sound is the encoded sound by construction rather than by
    two tools agreeing twice.

    fill=True says the frame carries the fourth field, and then it also returns that frame's
    level codes, which is what reconstruct_bands needs. Passing it from the header flag rather
    than deriving it is deliberate: the decoder is told whether to expect bits, it does not
    infer that a stream was made with a fill on.

    shape=True is the fifth field, read after the fill and derived from the same b. The region
    comes from reg = shape_reg(...), which is computed, not sent. How wide each of its words is
    comes from sblock, which is the header's business too - see shape_block.

    pvq=True means every live band is a pyramid vector, read in place of its scalar
    mantissas. sr and n2 are the header's. Dead-band noise after that is unchanged.
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
    claims = pvq_claims(b) if pvq else None
    for j in range(len(b)):
        if b[j] <= 0:
            continue
        if claims is not None and claims[j]:
            continue
        bw = max(int(b[j]), 1)
        lim = (1 << bw) - 1
        step = 1 + bw
        mask = (1 << step) - 1
        if r.get(1):
            # Flat: every bin is the same width, so the band is one integer and the split
            # stays in this loop. Sixteen get() calls were most of read_frame at a full band.
            blob = r.take(width * step)
            shift = width * step
            row = mant[j]
            for i in range(width):
                shift -= step
                v = (blob >> shift) & mask
                row[i] = -(v & lim) if (v >> bw) else (v & lim)
            continue
        n = r.get(cb)
        if _BITS.gap_decode is not None and n >= 4:
            vals, at, pos = _BITS.gap_decode(r.bitbuf(), r.tell(), n, gb, step, bw)
            r.seek(pos)
            mant[j, at] = vals
            continue
        base = -1
        for _ in range(n):
            i = base + 1 + r.get(gb)
            v = r.get(step)
            mant[j, i] = -(v & lim) if (v >> bw) else (v & lim)
            base = i
    pulses = None
    if pvq:
        # The pyramid bands in the order the writer put them, and their synthesis in one call
        # rather than one per band: the shape is small integers, so its energy is exact and the
        # square root is the same scalar either way.
        rows = []
        qs = []
        es = []
        bs = []
        for j in np.flatnonzero(claims):
            wq = int(b[j])
            got = -1
            if _BITS.pvq_band is not None:
                k_p, qv, idx, got = _BITS.pvq_band(r.bitbuf(), r.tell(), wq, _PVQ_NBITS,
                                                   width, PVQ_K_TABLE)
            if got < 0:
                # The band is outside what the compiled form reads: no compiler, a pulse count
                # the table does not hold, or an index wider than 64 bits. All rare, and the
                # cursor never moved, so this is the grammar read the way it always was.
                k_p = _read_gamma(r)
                qv = r.get(wq)
                ncode = pvq_count(width, k_p)
                nbits = (ncode - 1).bit_length() if ncode > 1 else 0
                idx = r.get(nbits) if nbits else 0
            else:
                r.seek(got)
            rows.append(_pvq_shape_fast(idx, width, k_p)[::-1])
            qs.append(qv)
            es.append(int(E[j]))
            bs.append(int(b[j]))
        pulses = np.zeros((len(b), width), np.float64)
        if rows:
            pulses[np.flatnonzero(claims)] = _synth_bands(np.asarray(rows, np.float64),
                                                          np.asarray(qs, np.float64),
                                                          np.asarray(es, np.float64),
                                                          np.asarray(bs, np.float64))
    if fill or shape or pvq:
        codes = read_fill(r, b) if fill else None
        words = read_shape(r, b, reg, sblock) if shape else None
        if pvq:
            return E, b, mant, codes, words, pulses
        if shape:
            return E, b, mant, codes, words
        return E, b, mant, codes
    return E, b, mant


# --------------------------------------------------------------- container
# The header is what the two tools must agree on without talking, so it lives here and no
# CLI packs it by hand. Little-endian, no padding, fixed size, then one length per payload.

# Little-endian, so `<` means no alignment padding and mixed field widths are safe; the size
# is just the sum of the fields. Fixed size, fixed order, then one length per payload. The
# flags word is what makes a fill stream self-describing: a decoder that ignores it can still
# read the frame, and one that heeds it can hear the difference, which is the whole reason
# "it is only 6 bits a frame" is not an excuse for an undeclared bit map.
HEADER_FMT = "<4sBIBHHIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def playback_gain(peak, ceiling=1.0):
    """The linear gain that puts `peak` on `ceiling`, or 1 when it is already under.

    Stored as the float32 the header will write. That rounding can land a hair over the
    ceiling, so the value is stepped down until peak times gain fits. A non-positive
    ceiling means the caller is leaving the level alone.
    """
    if ceiling <= 0 or not np.isfinite(peak) or peak <= ceiling:
        return 1.0
    g = np.float32(ceiling / peak)
    while peak * float(g) > ceiling and g > np.float32(0):
        g = np.nextafter(g, np.float32(0.0))
    return float(g)


def pack_header(sr, nch, n2, width, n_samples, budget, payload_lens, flags=0, gain=1.0):
    gain = float(gain)
    if gain < 1.0:
        flags |= GAIN_FLAG
    h = struct.pack(HEADER_FMT, MAGIC, VERSION, int(sr), int(nch), int(n2), int(width),
                    int(n_samples), int(budget), len(payload_lens), int(flags))
    h += b"".join(struct.pack("<I", int(L)) for L in payload_lens)
    if flags & GAIN_FLAG:
        h += struct.pack("<f", np.float32(gain))
    return h


def unpack_header(h):
    magic, ver, sr, nch, n2, width, n_samp, budget, n_pl, flags = struct.unpack(
        HEADER_FMT, h[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"not an RSC3 file: {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported RSC3 version {ver}")
    off = HEADER_SIZE
    lens = [struct.unpack("<I", h[off + 4 * i:off + 4 * (i + 1)])[0] for i in range(n_pl)]
    off += 4 * n_pl
    gain = 1.0
    if flags & GAIN_FLAG:
        gain = float(struct.unpack("<f", h[off:off + 4])[0])
        off += 4
    return dict(sr=sr, nch=nch, n2=n2, width=width, n_samples=int(n_samp),
                budget=int(budget), lens=lens, flags=int(flags), gain=gain,
                body_off=off)


def _ms_fold(row):
    """One decision per `MS_GROUP` bands. Each group in `row` has to agree."""
    row = np.asarray(row, np.uint8).reshape(-1) & np.uint8(1)
    g = MS_GROUP
    n = int(row.size)
    if g <= 1:
        return row
    if n % g:
        raise ValueError(f"mid/side mask length {n} is not a multiple of {g}")
    blocks = row.reshape(-1, g)
    if np.any(blocks != blocks[:, :1]):
        raise ValueError("mid/side mask changes inside a group of bands")
    return np.ascontiguousarray(blocks[:, 0])


def _ms_choice(row):
    """How `write_ms` stores one frame of decisions.

    0 is left/right and 2 is mid/side, each two bits. A mixed frame is whichever
    of the two lists is shorter: a bit per group (mode 1), or the smaller set of
    group numbers (mode 3). A tie in that set lists the mid/side groups.
    `row` is already one entry per group.
    """
    row = np.asarray(row, np.uint8).reshape(-1) & np.uint8(1)
    n = int(row.size)
    if n == 0 or not np.any(row):
        return 0, True, np.zeros(0, np.int64)
    if np.all(row):
        return 2, True, np.zeros(0, np.int64)
    ms_at = np.flatnonzero(row)
    k_ms = int(ms_at.size)
    if k_ms <= n - k_ms:
        listed = ms_at.astype(np.int64)
        list_is_ms = True
        k = k_ms
    else:
        listed = np.flatnonzero(row == 0).astype(np.int64)
        list_is_ms = False
        k = n - k_ms
    sparse = 2 + 1 + count_bits(n) + k * gap_bits(n)
    if sparse < 2 + n:
        return 3, list_is_ms, listed
    return 1, True, np.zeros(0, np.int64)


def _ms_row_bits(row):
    """Bits for one frame of decisions. `row` is already one entry per group."""
    row = np.asarray(row).reshape(-1)
    mode, _, listed = _ms_choice(row)
    if mode in (0, 2):
        return 2
    if mode == 1:
        return 2 + int(row.size)
    return 2 + 1 + count_bits(row.size) + int(listed.size) * gap_bits(row.size)


def ms_bits(row):
    """Bits `write_ms` spends on one frame's mask."""
    return _ms_row_bits(_ms_fold(row))


def write_ms(w, row):
    """The mask for one frame, one decision per `MS_GROUP` bands.

    See `_ms_choice` for which shape gets written. The row is the per-band
    mask, and every band in a group carries the same decision.
    """
    row = _ms_fold(row)
    mode, list_is_ms, listed = _ms_choice(row)
    w.put(mode, 2)
    if mode == 1:
        w.put_bits(row & np.uint8(1))
    elif mode == 3:
        n = int(row.size)
        w.put(0 if list_is_ms else 1, 1)
        w.put(int(listed.size), count_bits(n))
        if listed.size:
            gaps = np.empty(listed.size, np.int64)
            gaps[0] = listed[0]
            if listed.size > 1:
                gaps[1:] = listed[1:] - listed[:-1] - 1
            w.put_fields(gaps, np.full(listed.size, gap_bits(n), np.int64))
    return _ms_row_bits(row)


def read_ms(r, nbands):
    """The mask `write_ms` stored. 1 where that band was coded mid/side.

    The file has one decision per `MS_GROUP` bands. This repeats it so the
    caller still has one entry per band.
    """
    g = MS_GROUP
    n_full = int(nbands)
    if g > 1 and n_full % g:
        raise ValueError(f"mid/side band count {n_full} is not a multiple of {g}")
    n = n_full // g if g > 1 else n_full
    mode = r.get(2)
    if mode == 0:
        out = np.zeros(n, np.uint8)
    elif mode == 2:
        out = np.ones(n, np.uint8)
    elif mode == 1:
        out = np.empty(n, np.uint8)
        i = 0
        while i < n:
            take = 16 if n - i > 16 else n - i
            v = r.get(take)
            for k in range(take):
                out[i + k] = (v >> (take - 1 - k)) & 1
            i += take
    elif mode == 3:
        list_is_ms = r.get(1) == 0
        k = r.get(count_bits(n))
        out = np.zeros(n, np.uint8) if list_is_ms else np.ones(n, np.uint8)
        gb = gap_bits(n)
        pos = -1
        for _ in range(k):
            pos += 1 + r.get(gb)
            if pos < 0 or pos >= n:
                raise ValueError(f"mid/side group {pos} outside 0..{n - 1}")
            out[pos] = 1 if list_is_ms else 0
    else:
        raise ValueError(f"mid/side mode {mode}")
    if g > 1:
        return np.repeat(out, g)
    return out


def ms_rotate(L, R, use, width=16):
    """Mid and side on the bands `use` marks. Mid is (L+R)/sqrt(2).

    The scale keeps the band's total energy, so the waterfill sees the same
    loudness it would have split across left and right. A band left unmarked
    is copied through unchanged.
    """
    width = int(width)
    Ls = np.asarray(L, np.float64).reshape(L.shape[0], -1, width)
    Rs = np.asarray(R, np.float64).reshape(R.shape[0], -1, width)
    u = np.asarray(use, bool)[..., None]
    mid = (Ls + Rs) * _MS_SCALE
    side = (Ls - Rs) * _MS_SCALE
    return (np.where(u, mid, Ls).reshape(L.shape),
            np.where(u, side, Rs).reshape(R.shape))


def ms_unrotate(M, S, use, width=16):
    """Inverse of `ms_rotate`. Same scale, so an unquantized pair comes back exactly."""
    width = int(width)
    Ms = np.asarray(M, np.float64).reshape(M.shape[0], -1, width)
    Ss = np.asarray(S, np.float64).reshape(S.shape[0], -1, width)
    u = np.asarray(use, bool)[..., None]
    left = (Ms + Ss) * _MS_SCALE
    right = (Ms - Ss) * _MS_SCALE
    return (np.where(u, left, Ms).reshape(M.shape),
            np.where(u, right, Ss).reshape(S.shape))


def frame_count(n_samples, n2):
    """must match analyse(), which zero-pads rather than dropping the tail"""
    return int(np.ceil(n_samples / (n2 // 2))) + 1
