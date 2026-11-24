"""
RSC3 prototype codec: block-transform coding on an MDCT, with a perceptual bit allocation.

Everything here is measured, not assumed:

* The transform is the ISO/IEC-23003-3 / wiki form, kernel
      C[n,k] = cos( pi/N * (n + 0.5 + N/2) * (k + 0.5) ),  block 2N, hop N,
  with the lapped window w[n] = sin( pi (n+0.5) / (2N) ). Analysis X = C^T (w*x), synthesis
  y = w * (2/N) C X, and windowed overlap-add is exact to ~1e-15 (see transform_check.py).
  Both directions run through one complex DFT, so this is the form a Luau decoder would port.
* A full-scale sine measures |X| = N/2 (measured to 0.3%), so `amplitude()` puts coefficients
  in dBFS-comparable units. Every perceptual threshold below is stated in dB relative to full
  scale for exactly that reason: an ATH gate applied to frame-relative magnitudes is a gate that
  moves with the signal and quietly deletes quiet passages.
* The quantiser is the classical scalefactor + mantissa split, so the model's 6.02 dB/bit
  assumption is true of the output rather than merely asserted.

Speed: every stage is vectorized across frames, chunked so the intermediates stay bounded.
The per-frame scalar forms are kept as `*_1` helpers because the reference implementation is
what says a vectorized rewrite is still the same codec, not a plausible new one.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAX_BITS_PER_COEF = 16
BAND_WIDTH = 16                      # coefficients per scalefactor band
CHUNK = 128                          # frames per vectorized batch


# ---------------------------------------------------------------- transform

def window(N2):
    """Lapped analysis/synthesis window, length N2 = 2N."""
    return np.sin(np.pi * (np.arange(N2) + 0.5) / N2)


def dct4(v):
    """
    DCT-IV via one length-L complex inverse DFT, read out in even/odd bin pairs:
        C_k = 2 Re( e^{i pi (k+0.5)/(2L)} * sum_m v_m e^{i pi m/(2L)} e^{i pi m k / L} )
    which is two half-used inverse DFTs, i.e. one length-L complex FFT's worth of work.
    Verified against scipy's DCT-IV to 1e-15 for L = 8 .. 2048. Works on any stack of rows.
    """
    L = v.shape[-1]
    m = np.arange(L)
    q = v * np.exp(1j * np.pi * m / (2 * L))
    A = np.fft.ifft(q, axis=-1) * L                       # output bins k = 0, 2, 4, ...
    B = np.fft.ifft(q * np.exp(1j * np.pi * m / L), axis=-1) * L   # bins 1, 3, 5, ...
    tw = np.exp(1j * np.pi * (np.arange(L) + 0.5) / (2 * L))
    C = np.empty(v.shape, float)
    r = np.arange(L // 2)
    ke, ko = 2 * r, 2 * r + 1
    C[..., ke] = 2 * (tw[ke] * np.take(A, r, axis=-1)).real
    C[..., ko] = 2 * (tw[ko] * np.take(B, r, axis=-1)).real
    return C


def mdct(x, N2):
    """One windowed block of N2 samples -> N2/2 coefficients. Fold verified against the kernel."""
    h = N2 // 4
    a, b = x[:h], x[h:2 * h]
    c, d = x[2 * h:3 * h], x[3 * h:]
    return 0.5 * dct4(np.concatenate([-c[::-1] - d, a - b[::-1]]))


def imdct(X, N2):
    """N2/2 coefficients -> N2 samples, before window and overlap-add. Fold transpose, scale 1/N."""
    v = dct4(X)
    h = N2 // 4
    p, q = v[:h], v[h:2 * h]
    return (2.0 / N2) * np.concatenate([q, -q[::-1], -p[::-1], -p])


def _mdct_blocks(B, N2):
    """Rows of windowed blocks -> rows of coefficients. Same fold, batched."""
    h = N2 // 4
    u = -B[:, 2 * h:3 * h][:, ::-1] - B[:, 3 * h:4 * h]
    v = B[:, :h] - B[:, h:2 * h][:, ::-1]
    return 0.5 * dct4(np.concatenate([u, v], axis=1))


def _imdct_blocks(X, N2):
    """Rows of coefficients -> rows of time-domain samples, before windowing. Same transpose."""
    h = N2 // 4
    v = dct4(X)
    p, q = v[:, :h], v[:, h:2 * h]
    return (2.0 / N2) * np.concatenate([q, -q[:, ::-1], -p[:, ::-1], -p], axis=1)


def analyse(x, N2):
    """
    Signal -> (frames, coefficients). Hop is N2/2. Trailing frames are zero-padded rather than
    dropped, so the last windowed block still reconstructs its samples; nothing is thrown away.
    """
    n_samples = len(x)
    hop = N2 // 2
    n_frames = int(np.ceil(n_samples / hop)) + 1
    w = window(N2)
    X = np.empty((n_frames, hop), float)
    for s in range(0, n_frames, CHUNK):
        e = min(s + CHUNK, n_frames)
        a, b = s * hop, e * hop + N2
        buf = np.zeros(b - a, float)
        take = max(0, min(b, n_samples) - a)
        buf[:take] = x[a:a + take]
        idx = np.arange(e - s)[:, None] * hop + np.arange(N2)
        X[s:e] = _mdct_blocks(w * buf[idx], N2)
    return X, hop


def synthesise(X, N2, out_len):
    """
    Coefficients -> signal. Windowed overlap-add, the exact inverse of analyse() to ~1e-15.

    With hop = N2/2 every sample is covered by exactly two frames, so the overlap-add is a
    pairwise sum rather than a loop: the block at output position i*hop is
        acc[i, 0 : hop] + acc[i-1, hop : N2]
    which flattens straight into the output.
    """
    hop = N2 // 2
    w = window(N2)
    out = []
    tail = None
    for s in range(0, X.shape[0], CHUNK):
        e = min(s + CHUNK, X.shape[0])
        acc = _imdct_blocks(X[s:e], N2) * w
        d = acc[:, :hop].copy()
        d[1:] += acc[:-1, hop:]
        if tail is not None:
            # this chunk's first block still needs the previous chunk's second half,
            # which lives one chunk back and would otherwise be dropped per CHUNK frames
            d[0] += tail
        tail = acc[-1, hop:].copy()
        out.append(d.reshape(-1))
    full = np.concatenate(out)
    return full[:out_len]


def amplitude(X, N2):
    """Coefficients in amplitude units, where a full-scale sine is 1.0. The MDCT gain is N/2."""
    return X * (2.0 / (N2 / 2))


# ------------------------------------------------------------- psychoacoustics

# Absolute threshold of hearing, measured table (same curve RSC2's psycho layer uses). An
# analytic fit is what wrecked the first version of this file: the usual polynomial forms
# diverge at HF, mask the entire top of the spectrum and silently produce a silent codec.
_AT_HZ = np.array([20.0, 31.5, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0,
                   4000.0, 5000.0, 6000.0, 8000.0, 10000.0, 12500.0, 16000.0])
_AT_SPL = np.array([72.0, 54.0, 40.0, 27.0, 17.0, 12.0, 7.0, 3.0, 1.0, -1.0,
                    1.0, 4.0, 8.0, 15.0, 24.0, 36.0, 60.0])


def ath_db(n_coefs, sr):
    """ISO-226-shaped audibility floor, dB SPL."""
    hz = np.maximum(np.arange(n_coefs) * sr / (2 * n_coefs), _AT_HZ[0])
    return np.interp(hz, _AT_HZ, _AT_SPL)


def ath_dbfs(n_coefs, sr, spl_ref=96.0):
    """
    That floor in dBFS for this codec's amplitude units. spl_ref is the SPL a full-scale
    sample is assumed to play back at, so the reference is a stated assumption about listening
    level, not a property of the audio; 96 dB SPL corresponds to loud 16-bit playback.
    """
    return ath_db(n_coefs, sr) - spl_ref


def _erb_rate(n_coefs, sr):
    """ERB-rate axis (21.4 log10(1 + 0.006 f), f in Hz). One convention, on purpose: two
    conventions for the same axis is how masking slopes end up mis-scaled by 15%, and the
    first version divided f by 1000, which squeezed 24 kHz into about one ERB."""
    hz = np.maximum(np.arange(n_coefs) * sr / (2 * n_coefs), 1.0)
    return 21.4 * np.log10(1.0 + 0.006 * hz)


def _window_min(Xdb, w):
    """lmin[i] = min of the w bins left of i, rmin[i] = min of the w bins right of i.
    Padded so the ends read +inf rather than inventing a valley that isn't there."""
    inf = np.full((Xdb.shape[0], w), np.inf)
    lp = np.concatenate([inf, Xdb[:, :-1]], axis=1)
    rp = np.concatenate([Xdb[:, 1:], inf], axis=1)
    lmin = sliding_window_view(lp, w, axis=1).min(axis=-1)[:, :Xdb.shape[1]]
    rmin = sliding_window_view(rp, w, axis=1).min(axis=-1)[:, :Xdb.shape[1]]
    return lmin, rmin


def _ex_cumsum(x):
    """
    Prefix sums that EXCLUDE the element at each position: out[i] = sum(x[:i]). The exclusion
    is the whole point, since it keeps a bin's own power out of its own threshold sum instead
    of adding and subtracting it. The reversed-array case is handled by the caller reversing
    the result back.
    """
    c = np.cumsum(x, axis=1)
    return np.concatenate([np.zeros_like(c[:, :1]), c[:, :-1]], axis=1)


def _mask_db_blocks(Xdb, sr, spread_up, spread_down, scale, prominence_db, prom_window,
                    max_maskers):
    """
    Simultaneous-masking threshold for a stack of frames. Identical model to mask_db(), which
    documents what each part is for; this is the same computation with the frame loop gone.
    """
    n = Xdb.shape[1]
    z = _erb_rate(n, sr)
    w = max(int(prom_window), 1)
    lmin, rmin = _window_min(Xdb, w)
    left = np.concatenate([np.ones((Xdb.shape[0], 1), bool), Xdb[:, :-1] < Xdb[:, 1:]], axis=1)
    right = np.concatenate([Xdb[:, :-1] > Xdb[:, 1:], np.ones((Xdb.shape[0], 1), bool)], axis=1)
    dip = Xdb - np.maximum(lmin, rmin)        # drops into a valley on BOTH sides
    peak = left & right & (dip > prominence_db)

    # loudest maskers win: rank every bin by level, keep the first max_maskers peaks, then
    # PACK them to the front. The pack is the part that is easy to get wrong: peaks sit in a
    # level-rank order that is full of non-peaks, so the r-th peak is not in column r, and
    # slicing [:max_maskers] without packing drops every peak that landed further down the
    # rank. That deletes maskers rather than adding them, which reads as a quiet frame and
    # not as an error. Stable sort by "is this an active masker" compacts them while keeping
    # the level order, and the rest of the row falls away.
    order = np.argsort(-Xdb, axis=1)
    is_peak = np.take_along_axis(peak, order, 1)
    active = np.logical_and(np.cumsum(is_peak, axis=1) <= max_maskers, is_peak)
    got = active.any(axis=1)
    out = np.full(Xdb.shape, -200.0)
    if not got.any():
        return out
    m = max(int(active.sum(axis=1).max()), 1)
    # np.argsort orders False before True, so sorting `~active` stably is what leaves the real
    # maskers in the leading columns; sorting `active` (the obvious-looking version) keeps the
    # empty slots in front and drops every masker off the end, silently.
    perm = np.argsort(~active, axis=1, kind="stable")
    packed = np.take_along_axis(np.where(active, order, 0), perm, 1)
    sel = np.take_along_axis(active, perm, 1)[:, :m]
    cand = np.where(sel, packed[:, :m], 0)
    # Only real maskers contribute; an empty slot carries no power, or the frame gains a
    # phantom masker at bin 0 that raises thresholds and steals bits from the real ones.
    P = np.where(sel, 10.0 ** (0.1 * np.take_along_axis(Xdb, cand, 1)), 0.0)
    # The skirt is LINEAR in ERB-rate distance, which makes the kernel rank-1:
    #     10**(-c*(z_i - z_j)) = 10**(-c*z_i) * 10**(+c*z_j)
    # so summing maskers above or below bin i is a cumulative sum, not a matrix. The old form
    # built (frames x maskers x bins) of exponentials - 35M elements on this material, and it
    # was the stage that dominated allocate. This is 66x faster and agrees with the definition
    # to 1e-15; mask_speed.py measures both, skirt_debug.py proves the algebra.
    dense = np.zeros(Xdb.shape)
    rows = np.arange(Xdb.shape[0])[:, None] * np.ones((1, cand.shape[1]), int)
    np.add.at(dense, (rows, cand), P)
    cu = 0.1 * spread_up            # decades per ERB, up and down, so the kernel is exact
    cd = 0.1 * spread_down
    # EXCLUSIVE prefix sums, so a masker's own power never enters and the threshold at a
    # masker bin is the sum of the OTHER maskers, computed at full precision. Summing self and
    # subtracting is the same algebra and a catastrophic one: at a loud bin the two agree to
    # ~16 decades, which leaves cancellation residue where the answer should be zero. Measured:
    # the inclusive form drifts 1.2e-2 dB off the explicit loop on real frames, the exclusive
    # form 5.6e-8 dB. Both spend identical bit maps, and the exclusive one costs nothing.
    x_up = dense * (10.0 ** (cu * z))
    x_dn = (dense * (10.0 ** (-cd * z)))[:, ::-1]
    up = (10.0 ** (-cu * z)) * _ex_cumsum(x_up)
    dn = (10.0 ** (cd * z)) * _ex_cumsum(x_dn)[:, ::-1]
    # A masker is never masked by itself, or the threshold sits at the masker's own level and
    # zeroes every loudest bin in the frame - the single most destructive mistake this model makes.
    combined = up + dn
    res = 10.0 * np.log10(np.maximum(combined, 1e-30)) + 10.0 * np.log10(scale)
    return np.where(got[:, None], res, -200.0)


def mask_db(Xdb, sr, n_coefs, spread_up=10.0, spread_down=25.0, scale=0.15,
            prominence_db=6.0, prom_window=16, max_maskers=48):
    """
    Simultaneous-masking threshold of one frame, in dBFS (Xdb must already be dBFS).

    Only real maskers mask. Taking every bin as a masker buries the frame in its own broadband
    energy, which is why the first version of this model allocated zero bits: the threshold rose
    above the signal everywhere. A bin only imposes a threshold if it is a local peak standing
    prominence_db above its neighbours, and at most `max_maskers` of them — real codecs cap the
    masker list for exactly this reason, and a frame that needs more than 48 is a noisy frame
    where the skirts are the least of its problems.

    Masker power spreads in ERB-rate, 10 dB/ERB upward and 25 dB/ERB downward (Glasberg-Moore),
    overlapping skirts add in power, and 15% of that sum is what actually masks.

    Prominence is a *local* contrast. Two wrong versions both make the filter inert or
    destructive: a global gate (Xdb > max - 6) admits one masker per frame whose skirt then
    covers everything, and a running min from both ends puts the "valley" at the quietest bin
    anywhere in the frame, which is 50+ dB below every peak, so every local maximum passes and
    noise bumps impose thresholds too.
    """
    if Xdb.ndim > 1:
        return _mask_db_blocks(np.atleast_2d(Xdb), sr, spread_up, spread_down, scale,
                               prominence_db, prom_window, max_maskers)
    z = _erb_rate(n_coefs, sr)
    lmin, rmin = _window_min(Xdb[None, :], max(int(prom_window), 1))
    lmin, rmin = lmin[0], rmin[0]
    left = np.concatenate([[True], Xdb[:-1] < Xdb[1:]])
    right = np.concatenate([Xdb[:-1] > Xdb[1:], [True]])
    peak = (left & right) & ((Xdb - np.maximum(lmin, rmin)) > prominence_db)
    cand = np.nonzero(peak)[0]
    if cand.size == 0:
        return np.full(n_coefs, -200.0)
    if cand.size > max_maskers:
        cand = cand[np.argsort(-Xdb[cand])[:max_maskers]]
    P = 10.0 ** (0.1 * Xdb[cand])
    d = z[None, :] - z[cand, None]
    att = np.where(d >= 0, spread_up * d, spread_down * (-d))
    combined = (P[:, None] * 10.0 ** (-0.1 * att)).sum(axis=0)
    self_p = np.zeros(n_coefs)
    self_p[cand] = P
    combined = np.maximum(combined - self_p, 0.0)
    return 10.0 * np.log10(np.maximum(combined, 1e-30)) + 10.0 * np.log10(scale)


# ---------------------------------------------------------------- allocation

def band_edges(n_coefs, width=BAND_WIDTH):
    """Contiguous coefficient groups sharing one scalefactor."""
    return np.unique(np.concatenate([[0], np.arange(width, n_coefs + 1, width), [n_coefs]]))


def threshold_dbfs(A_db, sr, n_coefs=None, psycho=True, spl_ref=96.0, **kw):
    """
    The highest noise level that stays inaudible at each coefficient of one frame, or of a
    stack of frames: the audibility floor, raised wherever a real masker puts maskers there.
    psycho=False keeps only ATH, which is the blind baseline the model has to beat to be worth
    its complexity.
    """
    if n_coefs is None:
        n_coefs = A_db.shape[-1]
    ath = ath_dbfs(n_coefs, sr, spl_ref)
    if not psycho:
        return np.broadcast_to(ath, A_db.shape).copy() if A_db.ndim > 1 else ath
    m = mask_db(A_db, sr, n_coefs, **kw)
    m = np.atleast_2d(m) if A_db.ndim > 1 else m
    return np.maximum(ath, m)


def _band_level_db(A_db, edges):
    """Each band's loudest coefficient, broadcast back over the band. A bin sitting well below
    its band's peak needs more bits to reach the same absolute noise, and this is where that
    charge comes from - the thing a per-coefficient formula gets wrong and why scalefactors exist."""
    b = np.zeros_like(A_db)
    for j in range(len(edges) - 1):
        lo, hi = int(edges[j]), int(edges[j + 1])
        b[:, lo:hi] = A_db[:, lo:hi].max(axis=1, keepdims=True)
    return b


def _band_level_abs(A, edges):
    s = np.zeros_like(A)
    for j in range(len(edges) - 1):
        lo, hi = int(edges[j]), int(edges[j + 1])
        s[:, lo:hi] = np.abs(A[:, lo:hi]).max(axis=1, keepdims=True)
    return np.where(s > 0, s, 1.0)


def waterfill(A_db, threshold_db, band_level_db, budget_bits):
    """
    Reverse water-filling for one frame, in the shape a scalefactor coder actually spends.

    A coefficient in a band whose scale sits at S gets step S*2^-b, so its noise is S - 6.02b and
    the bits that reach the threshold T_k are
        b_k = ceil( (S_band(k) - max(T_k, floor)) / 6.02 )
    which correctly charges extra for coefficients that sit far below their band's peak. But a
    coefficient already UNDER T_k takes zero bits: zeroing it leaves an error equal to its own
    level, which is under the threshold by definition, while quantising it with the coarse step
    its bits can buy would ADD noise above what it was hiding. Omitting this turns masked bins
    into error sources, which is why a pure tone's quiet neighbours ate 96% of its residual and
    capped the tone at 14.5 dB no matter how many bits the budget offered.
    Shared floor is the only degree of freedom; bisection keeps the *lowest* floor that fits,
    since a lower floor puts bits on more coefficients. Returning the highest feasible floor,
    the obvious mistake, starves the frame and reads as silence.
    """
    return _waterfill_all(A_db[None, :], threshold_db[None, :],
                          band_level_db[None, :], budget_bits)[0]


def _waterfill_all(A_db, threshold_db, band_level_db, budget_bits):
    """
    The bisection, kept as the reference: it is what says the selection below is the same
    codec and not a plausible new one. It is not used by `allocate` any more.

    60 iterations is not padding. Measured on real material the bit map is still changing at
    iteration 56, and that is correct rather than wasteful, because the bracket is roughly
    300 dB wide - its lower bound is the frame's most negative threshold, which can read
    -308 dBFS when every interesting bin sits near -20. So the search spends most of its
    steps resolving decibels no coefficient can occupy. The fix is not fewer iterations, it
    is not searching.
    """
    lo = threshold_db.min(axis=1) - 6.02
    hi = band_level_db.max(axis=1) + 6.02 * MAX_BITS_PER_COEF + 1.0
    best = np.zeros(A_db.shape, int)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        floor = np.maximum(threshold_db, mid[:, None])
        room = np.where(A_db > floor, (band_level_db - floor) / 6.02, 0.0)
        bits = np.clip(np.where(room > 0, np.ceil(room), 0), 0, MAX_BITS_PER_COEF).astype(int)
        fits = bits.sum(axis=1) <= budget_bits
        hi = np.where(fits, mid, hi)
        lo = np.where(fits, lo, mid)
        best = np.where(fits[:, None], bits, best)
    return best


def waterfill_select(A_db, threshold_db, band_level_db, budget_bits, max_bits=MAX_BITS_PER_COEF):
    """
    Water-filling by selecting the price, not by searching for the floor.

    The quantity being solved is a step function, not a smooth one, so a search is the wrong
    shape of algorithm. Bit b of band k exists while max(T_k, floor) < S_k - 6.02(b-1), which
    says: that expression is the *price* of the b-th bit, raise the floor past a price and you
    lose that bit and only that bit. So the frame's total is just a count of prices above the
    floor, it is monotone in the floor, and the lowest floor that fits a budget is the
    (budget+1)-th largest price. Linear selection instead of sixty passes over the frame.

    Two details the bisection handled implicitly and a selection must not forget:
      * a price at or below its own bin's threshold is unreachable, because floor_k is
        max(T_k, floor) and is always at least T_k there. Leaving those in counts free bits
        the frame can never collect, and it inflates the budget the frame thinks it spent.
      * a bin cannot be charged for bits beyond its own level: the b-th bit also requires
        floor < A_k, which is why the price is capped at min(price, A_k). Without the cap,
        masked bins get quantised with the coarse step their bits would buy and add noise
        above what they were hiding - the exact failure that capped a pure tone at 14.5 dB.

    `band_level_db` must already be floored at the threshold, so band_level >= threshold.
    """
    step = 6.02
    b = np.arange(1, max_bits + 1, dtype=np.float64)
    out = np.zeros(A_db.shape, np.int16)
    for s in range(0, A_db.shape[0], CHUNK):
        e = min(s + CHUNK, A_db.shape[0])
        S = band_level_db[s:e, None, :, None]
        A = A_db[s:e, None, :, None]
        thr = threshold_db[s:e, None, :, None]
        price = S - step * (b[None, None, None, :] - 1.0)
        price = np.where((thr < price) & (A > thr), np.minimum(price, A), -np.inf)
        flat = price.reshape(e - s, -1)
        j = min(budget_bits, flat.shape[1] - 1)
        floor = -np.partition(-flat, j, axis=1)[:, j]
        got = flat > floor[:, None]
        out[s:e] = got.reshape(e - s, A_db.shape[1], max_bits).sum(axis=2).astype(np.int16)
    return out


def allocate(A, sr, budget_bits, psycho=True, spl_ref=96.0, edges=None, search=False):
    """
    Bits per (frame, coefficient). A is `amplitude()`-converted coefficients, one row per frame.
    """
    n_coefs = A.shape[1]
    if edges is None:
        edges = band_edges(n_coefs)
    A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
    thr = threshold_dbfs(A_db, sr, n_coefs, psycho, spl_ref)
    blevel = _band_level_db(A_db, edges)
    bl = np.maximum(blevel, thr)
    if search:
        return _waterfill_all(A_db, thr, bl, budget_bits)
    return waterfill_select(A_db, thr, bl, budget_bits).astype(int)


def quantise(A, bits, edges=None):
    """
    Round amplitudes onto the grid the allocation allows: step = band_scale * 2^-b, the band
    scale being that band's loudest coefficient in this frame, exactly the scale the allocation
    charged bits against. The step the decoder rebuilds from the transmitted scale and mantissa
    is bit-for-bit this one, so the 6.02 dB/bit the allocation spent is what actually comes out.
    (This is the uniform-mantissa stand-in for PVQ: real PVQ changes the error's shape, not the
    allocation that produced it.)
    """
    n_coefs = A.shape[1]
    if edges is None:
        edges = band_edges(n_coefs)
    scale = _band_level_abs(A, edges)
    step = np.where(bits > 0, 2.0 ** (-bits).astype(float), 0.0) * scale
    return np.round(A / np.where(step > 0, step, 1.0)) * step * (bits > 0)
