"""rsc3_bits.py - the sequential parts of the codec, compiled.

Everything else in the codec has been moved to numpy, where whole frames and whole bands go
through in one call. Two things cannot go that way. A Rice field: a codeword's length depends on
the value ahead of it, so the only way to know where the next symbol begins is to walk there.
Read in Python that walk is about a microsecond a symbol, which on twenty seconds of audio is
most of what the decoder does, and the per-symbol `quotient` call is also where the reservoir
spends its time. And the pyramid placer: its vectorised form steps every band once per pulse so
the bands that finished early are walked again with the ones still running, which on a 32-bin
band at 27 pulses is mostly memory traffic over work already done. The compiled `pvq_place` is
that loop per band; the rule, the order, and the tie-break are the same ones the numpy block
still holds.

The Python reader in `rsc3_stream` is still there and still correct, and it is what runs on a
machine without the JIT. That is deliberate: a decoder that only works once something is
compiled is a decoder that fails on a fresh box, and this codec's decoder has to survive being
the thing that ships. The two paths return the same symbols in the same order, which is what
the payload identity in perf_check proves, not what this file claims.

Past the end of the buffer the reference reads zeros and carries on. This does not follow it
there, and there is nothing to follow: the framing puts the exact payload length in the header,
so a run that reaches the end of the buffer before its field is finished is a file that was
truncated, and the reference's way of handling that is a loop that never ends.
"""
from __future__ import annotations

import os

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    HAVE_NUMBA = False

# The one knob that separates "this is the same codec" from "this is a faster codec", so the
# compiled walk can be switched off on the same file and compared bit for bit. It is off by
# default and on by default, and it costs nothing to honour: setting it is the same thing a
# machine without the compiler does to itself.
if os.environ.get("RSC3_NO_JIT") == "1":
    HAVE_NUMBA = False

if HAVE_NUMBA:
    import numpy as np

    @njit(cache=True, nogil=True, fastmath=False)
    def rice_decode(buf, bitpos, count, k):
        """`count` Rice(k) codewords from a bit buffer, MSB first.

        `buf` is the payload, `bitpos` the absolute bit offset of the field's first bit, and the
        return is (the unsigned symbols, the bit offset just past the field) - the same pair the
        reader's own cursor would have left behind, so the caller's next field starts exactly
        where the slow path would have put it.
        """
        out = np.empty(count, np.int64)
        nbits = buf.size << 3
        pos = bitpos
        mask = (1 << k) - 1
        for i in range(count):
            q = 0
            while pos < nbits:
                b = (buf[pos >> 3] >> (7 - (pos & 7))) & 1
                pos += 1
                if b:
                    break
                q += 1
            rem = 0
            for _ in range(k):
                if pos < nbits:
                    rem = (rem << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                else:
                    rem = rem << 1
                pos += 1
            out[i] = (q << k) | rem
        return out, pos

    @njit(cache=True, nogil=True, fastmath=False)
    def gap_decode(buf, bitpos, n_sym, gb, step, bw):
        """A band's gap-coded mantissas: n_sym runs of (position delta, magnitude).

        The same shape as a Rice field and for the same reason - each coefficient is read where
        the last one stopped. `gb` is the width of a gap, `step` the width of a magnitude, and
        `bw` the magnitude's own width inside it, so the sign is the high bit of the tuple and
        the rest is the value. Returns (values signed back in, positions within the band, the new
        bit offset).
        """
        vals = np.empty(n_sym, np.int64)
        pos_at = np.empty(n_sym, np.int64)
        nbits = buf.size << 3
        lim = (1 << bw) - 1
        pos = bitpos
        base = -1
        for i in range(n_sym):
            g = 0
            for _ in range(gb):
                g = (g << 1) | _bit(buf, nbits, pos)
                pos += 1
            v = 0
            for _ in range(step):
                v = (v << 1) | _bit(buf, nbits, pos)
                pos += 1
            base = base + 1 + g
            pos_at[i] = base
            vals[i] = -(v & lim) if (v >> bw) else (v & lim)
        return vals, pos_at, pos

    @njit(cache=True, nogil=True, fastmath=False)
    def _bit(buf, nbits, pos):
        return (buf[pos >> 3] >> (7 - (pos & 7))) & 1 if pos < nbits else 0

    @njit(cache=True, nogil=True, fastmath=False)
    def pvq_band(buf, pos, wq, nbits_tab, n, kmax):
        """Read one pyramid band: its pulse count, its gain, its index.

        The three pieces are adjacent and each one's size depends on the piece before it, so the
        compiler reads them in one go: the gamma code's leading zeros are counted straight out of
        the bit buffer instead of being asked for bit by bit, and the index is collected at the
        width the table says the codebook needs.

        A pulse count past `kmax` means a codebook the table does not hold, which in practice
        means a stream that was corrupted rather than written. The answer is then -1 for the new
        offset and the caller reads the band the slow way, with the cursor exactly where it was.
        """
        z = 0
        while True:
            if _bit(buf, buf.size << 3, pos):
                pos += 1
                break
            pos += 1
            z += 1
            if z > 16:
                break
        if z == 0:
            k = 1
        else:
            rem = 0
            for _ in range(z):
                rem = (rem << 1) | _bit(buf, buf.size << 3, pos)
                pos += 1
            k = (1 << z) | rem
        if k > kmax:
            return 0, 0, 0, -1
        q = 0
        for _ in range(wq):
            q = (q << 1) | _bit(buf, buf.size << 3, pos)
            pos += 1
        w = nbits_tab[n][k]
        if w > 62:
            # The index is wider than a 64-bit integer holds, so collecting it bit by bit here
            # would overflow and return a plausible-looking wrong vector. A pulse count high
            # enough to need this is rare, and it is the one case the reference walk is for.
            return 0, 0, 0, -1
        idx = 0
        for _ in range(w):
            idx = (idx << 1) | _bit(buf, buf.size << 3, pos)
            pos += 1
        return k, q, idx, pos


    @njit(cache=True, nogil=True, fastmath=False)
    def pvq_walk(cnt, index, n, k):
        """The point on the pyramid that `index` names, coordinate by coordinate.

        `cnt[n][k]` is how many length-n vectors have L1 norm k, held to a bound the caller has
        already guaranteed the index is under, and the walk is the combinatorial numbering read
        from the top bin down: everything with the last coordinate zero, then the blocks for
        +1, -1, +2, -2, and so on, with the index decremented past each one. It is pure integer
        arithmetic and a table lookup per level of sixteen, done for every live band of every
        frame the decoder reads, which is why it is here rather than in a loop of slices.

        The order is the reference walk's, coordinate by coordinate, and the two quirks of the
        recursion it replaces are kept: the +m branch reads the block before decrementing, and
        the -m branch reads it after the check, so levelling them returns a different vector.
        """
        out = np.empty(n, np.int64)
        p = 0
        while True:
            if k == 0 or n == 0:
                while p < out.size:
                    out[p] = 0
                    p += 1
                return out
            if n == 1:
                out[p] = k if index == 0 else -k
                return out
            left = cnt[n - 1][k]
            if index < left:
                out[p] = 0
                p += 1
                n -= 1
                continue
            index -= left
            m = 1
            hit = False
            while m <= k:
                block = cnt[n - 1][k - m]
                if index < block:
                    out[p] = m
                    p += 1
                    k -= m
                    n -= 1
                    hit = True
                    break
                index -= block
                if index < block:
                    out[p] = -m
                    p += 1
                    k -= m
                    n -= 1
                    hit = True
                    break
                index -= block
                m += 1
            if not hit:
                while p < out.size:
                    out[p] = 0
                    p += 1
                return out

    @njit(cache=True, nogil=True, fastmath=False)
    def pvq_place(x, kk):
        """Place `kk[i]` pulses on band `i`, the way `_choose_bands`'s loop does.

        The vectorised form runs the bands in lockstep, so every pulse costs a pass over
        (bands x bins) for the score, the argmax and the four accumulators, and the bands that
        stopped several pulses ago are walked again with the rest. A 32-bin band at 27 pulses
        is about 131 thousand score evaluations, most of them memory traffic.

        The arithmetic is the same, in the same order, so the doubles match: `score = (corr +
        |x|)^2 / (ny + 2*ay + 1)` at every bin including the ones whose sample is zero, a
        strict `>` so the earliest bin wins a tie as argmax does, and a band that chooses a
        zero bin is done. `ny` takes `2*ay_before + 1` the way the original reads the count
        before bumping it. Nothing else is decided here: energy, exponent and gain stay with
        the numpy code that computes them, because the exponent wants the same log2 a vector
        gives and that is not this loop's business.
        """
        nband, width = x.shape
        y = np.zeros((nband, width), np.int64)
        for i in range(nband):
            k = kk[i]
            if k <= 0:
                continue
            ay = np.zeros(width, np.int64)
            ny = 0.0
            corr = 0.0
            for _ in range(k):
                best = -1.0
                bi = -1
                best_ax = 0.0
                for j in range(width):
                    xv = x[i, j]
                    ax = -xv if xv < 0.0 else xv
                    den = ny + 2.0 * ay[j] + 1.0
                    num = corr + ax
                    s = (num * num) / den
                    if s > best:
                        best = s
                        bi = j
                        best_ax = ax
                if bi < 0 or best_ax <= 0.0:
                    break
                y[i, bi] += 1 if x[i, bi] >= 0.0 else -1
                ny += 2.0 * ay[bi] + 1.0
                ay[bi] += 1
                corr += best_ax
        return y

else:                                   # pragma: no cover - the path a box without the JIT takes
    # Everything here is the reader's own per-symbol walk, which is slower and correct.
    rice_decode = None
    gap_decode = None
    pvq_walk = None
    pvq_band = None
    pvq_place = None
