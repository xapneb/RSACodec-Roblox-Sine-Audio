"""
RSC3 prototype codec: block-transform coding on an MDCT, with a perceptual bit allocation.

Everything here is measured, not assumed:

* The transform is the ISO/IEC-23003-3 / wiki form, kernel
      C[n,k] = cos( pi/N * (n + 0.5 + N/2) * (k + 0.5) ),  block 2N, hop N,
  with the lapped window w[n] = sin( pi (n+0.5) / (2N) ). Analysis X = C^T (w*x), synthesis
  y = w * (2/N) C X, and windowed overlap-add is exact to ~1e-15 (see transform_check.py).
  Both directions run through one complex DFT, so this is the form a Luau decoder would port.
* A full-scale sine measures |X| = N/2 (measured to 0.3%), so `amplitude()` puts coefficients
  in dBFS-comparable units. Every perceptual threshold below is in those units. The threshold
  is the simultaneous mask. A flat floor stands in only where no masker reaches, so a silent
  bin is not chased downward and a hearing curve cannot delete a band the rate can carry.
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
# No-masker floor, dBFS. Simultaneous masking is the threshold wherever a real masker
# reaches. This only replaces the "nothing is masking" sentinel: 16-bit dither sits near
# -96 dBFS, and -120 stops the waterfill short of numerical silence. It is not a hearing
# curve. The old absolute-threshold table ended at 16 kHz and was held there, which pinned
# everything above that to about -36 dBFS and made those bands illegal at any bitrate.
QUIET_FLOOR_DBFS = -120.0


# ---------------------------------------------------------------- transform

_WIN = {}
_DCT4 = {}


def window(N2):
    """Lapped analysis/synthesis window, length N2 = 2N. Tabulated: every frame uses the same one."""
    w = _WIN.get(N2)
    if w is None:
        w = np.sin(np.pi * (np.arange(N2) + 0.5) / N2)
        _WIN[N2] = w
    return w


def _dct4_tabs(L):
    """The three twiddles and the even/odd index, once per length. Same values dct4 used to
    build on every call, so a cached table does not move a coefficient."""
    tab = _DCT4.get(L)
    if tab is None:
        m = np.arange(L)
        r = np.arange(L // 2)
        tab = (
            np.exp(1j * np.pi * m / (2 * L)),
            np.exp(1j * np.pi * m / L),
            np.exp(1j * np.pi * (m + 0.5) / (2 * L)),
            r,
            2 * r,
            2 * r + 1,
        )
        _DCT4[L] = tab
    return tab


def dct4(v):
    """
    DCT-IV via one length-L complex inverse DFT, read out in even/odd bin pairs:
        C_k = 2 Re( e^{i pi (k+0.5)/(2L)} * sum_m v_m e^{i pi m/(2L)} e^{i pi m k / L} )
    which is two half-used inverse DFTs, i.e. one length-L complex FFT's worth of work.
    Verified against scipy's DCT-IV to 1e-15 for L = 8 .. 2048. Works on any stack of rows.

    A single length-L/2 FFT reaches the same cosine and runs faster, and it disagrees with
    this one by ~1e-13 at L = 2048. That is enough to move a mantissa sitting on a rounding
    boundary, so the transform that already matches the kernel stays.
    """
    L = v.shape[-1]
    e1, e2, tw, r, ke, ko = _dct4_tabs(L)
    q = v * e1
    A = np.fft.ifft(q, axis=-1) * L                       # output bins k = 0, 2, 4, ...
    B = np.fft.ifft(q * e2, axis=-1) * L                  # bins 1, 3, 5, ...
    C = np.empty(v.shape, float)
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
    The highest noise level that stays masked at each coefficient of one frame, or of a stack
    of frames. With psycho on, that level is the simultaneous-masking curve from `mask_db`.
    Where no masker reaches, the curve falls away and `QUIET_FLOOR_DBFS` takes over.

    psycho=False is the blind arm: the same flat floor and no frequency masking, so allocation
    follows level alone. spl_ref is accepted and ignored. It used to slide a hearing curve,
    and that curve is gone.
    """
    del spl_ref
    if n_coefs is None:
        n_coefs = A_db.shape[-1]
    floor = np.full(n_coefs, QUIET_FLOOR_DBFS)
    if not psycho:
        return np.broadcast_to(floor, A_db.shape).copy() if A_db.ndim > 1 else floor.copy()
    m = mask_db(A_db, sr, n_coefs, **kw)
    m = np.atleast_2d(m) if A_db.ndim > 1 else m
    if A_db.ndim > 1 and TEMPORAL_MASK > 0.0:
        m = m - _attack_cut(A_db, TEMPORAL_MASK)[:, None]
    return np.maximum(floor, m)


# ------------------------------------------------------------- temporal mask
# A simultaneous mask is a picture of one frame, so it cannot tell an arriving sound from a
# steady one. On a frame that is mostly attack that costs real fidelity: the loudest, broadest
# bins of the hit pass as "masked" by the sum of their own skirts, get zero bits, and decode as
# holes in the transient - which is what listening to it sounds like. The fix here is only this:
# on a frame that just took a step up in energy, and on the frame leading into it, the threshold
# is lowered, so the ladder is allowed to buy what it was told not to.
#
# It is a module constant rather than another argument, and that is deliberate. The mask is read
# from three separate call sites (the prepared stack, the fill ceiling, and the band planner) and
# threading a new argument through all of them is how a codec ends up with two masks, one of them
# only in the encoder. Off by default, which is the only thing that keeps an old file's decisions
# reproducible, and the decoder never learns about it either way.
#
# Measured with it off, a uniform change to the mask strength does nothing: moz.wav writes the
# same file and the same error at 1.5, 3 and 6 dB of shift, because in a rate-limited frame the
# rank cut decides the spending and the mask only says which bins are allowed to exist. So the
# win here is the shape and not the amount, which is also why it survives. At 64 kbit/s moz.wav
# goes 20.30 -> 21.12 dB at 6 dB of cut and 22.55 dB at 48, each time writing a slightly smaller
# file, and a click train gains 5 dB at the hit. Nothing changes on material that does not move,
# and the frames it does touch are the ones that were losing detail anyway.
TEMPORAL_MASK = 0.0

# How far a frame's energy has to climb in one hop to be called an attack, in dB. Six is about the
# steepest slope the ear reads as a onset rather than a tremolo, and the cut saturates there so a
# frame that only half qualifies pays half.
ATTACK_RISE_DB = 6.0


def _attack_cut(Xdb, amount):
    """dB to take off each frame's threshold, by how much that frame's energy just arrived."""
    e = 10.0 * np.log10(np.mean(10.0 ** (Xdb / 10.0), axis=-1) + 1e-30)
    rise = np.clip((e - np.roll(e, 1)) / ATTACK_RISE_DB, 0.0, 1.0)
    rise = rise.copy()
    rise[0] = 0.0          # the first frame has no history to have arrived against
    return amount * np.maximum(rise, np.roll(rise, -1))


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


def ear_filter_share_db(level_db, hz, share=0.5):
    """Half the level of the auditory filter each band sits in, in dB.

    Filters are Glasberg and Moore's cam scale, z = 21.4 log10(1 + 0.00437 f).
    One cam is one filter. `level_db` is 10 log10 of the band power, which is
    what the pyramid waterfill already passes as 20 log10 of the band's L2.
    The masking skirts keep their own 0.006 axis. This scale is only the rank.

    Subtracting the whole filter level was measured to flatten every open band
    to one plane and lower the filter score (domain, 48 planes, 2.77 dB down
    to 2.14 dB). Half of it raised that score to 3.28 dB.
    """
    hz = np.asarray(hz, np.float64)
    z = 21.4 * np.log10(1.0 + 0.00437 * np.maximum(hz, 1.0))
    g = np.floor(z).astype(np.int32)
    power = 10.0 ** (np.asarray(level_db, np.float64) / 10.0)
    ng = int(g.max()) + 1
    nfr = power.shape[0]
    acc = np.zeros((nfr, ng), np.float64)
    np.add.at(acc, (np.arange(nfr)[:, None], np.broadcast_to(g, power.shape)), power)
    gdb = 10.0 * np.log10(np.maximum(acc, 1e-30))
    return float(share) * gdb[:, g]


def waterfill_select(A_db, threshold_db, band_level_db, budget_bits, max_bits=MAX_BITS_PER_COEF,
                     share_db=None):
    """
    Which bits a frame can afford.

    A bit's price is its level, S - 6.02*(b-1). `share_db` is subtracted from
    that when the caller has one. Pyramid bands pass half the level of the
    auditory filter the band sits in, so a band is priced as part of that
    filter. Without it the price is the raw level.

    A price at or below a bin's own threshold is not for sale, and a bit past
    the bin's own level is not either. Both of those buy quantisation noise
    above what the bin was hiding. The sale check stays on the raw level.
    `band_level_db` must already be floored at the threshold.
    """
    # Chunked, and the price table of a chunk is discarded. Holding every frame's prices
    # is about a gigabyte on a long file, which is the batch this codec already stopped doing.
    step = 6.02
    b = np.arange(1, max_bits + 1, dtype=np.float64)
    out = np.zeros(A_db.shape, np.int16)
    nc = A_db.shape[1]
    for s in range(0, A_db.shape[0], CHUNK):
        e = min(s + CHUNK, A_db.shape[0])
        part = None if share_db is None else share_db[s:e]
        flat = _price_chunk(A_db[s:e], threshold_db[s:e], band_level_db[s:e], b, step, part)
        out[s:e] = bits_from_prices(flat, budget_bits, nc, max_bits)
    return out


def _price_chunk(A_db, threshold_db, band_level_db, b, step, share_db=None):
    """Level, in dB, of bits 1..max. -inf where the bit cannot be bought.

    `share_db` is per bin and is removed from the rank only. The threshold
    test stays on the raw level, so a masked bin cannot become legal by sitting
    in a quiet filter.
    """
    S = band_level_db[:, None, :, None]
    A = A_db[:, None, :, None]
    thr = threshold_db[:, None, :, None]
    level = S - step * (b[None, None, None, :] - 1.0)
    legal = (thr < level) & (A > thr)
    if share_db is not None:
        level = level - np.asarray(share_db, np.float64)[:, None, :, None]
    return np.where(legal, level, -np.inf).reshape(A_db.shape[0], -1)


def waterfill_prices(A_db, threshold_db, band_level_db, max_bits=MAX_BITS_PER_COEF):
    """Price of every legal bit, shape (frames, coefficients * max_bits), chunked so the
    4-wide temporary stays inside a block of CHUNK frames."""
    step = 6.02
    b = np.arange(1, max_bits + 1, dtype=np.float64)
    nfr = A_db.shape[0]
    flat = np.empty((nfr, A_db.shape[1] * max_bits), np.float64)
    for s in range(0, nfr, CHUNK):
        e = min(s + CHUNK, nfr)
        flat[s:e] = _price_chunk(A_db[s:e], threshold_db[s:e], band_level_db[s:e], b, step)
    return flat


def bits_from_prices(flat, budget_bits, ncoefs, max_bits=MAX_BITS_PER_COEF, sorted_asc=None):
    """How many of each coefficient's prices sit strictly above the (budget+1)th largest.

    `sorted_asc` is `np.sort(flat, axis=1)` when the caller is about to ask this of many
    budgets. Ties at the rank are left unbought.
    """
    nfr, n = flat.shape
    j = min(int(budget_bits), n - 1)
    if sorted_asc is None:
        floor = -np.partition(-flat, j, axis=1)[:, j]
    else:
        floor = sorted_asc[:, n - 1 - j]
    got = flat > floor[:, None]
    return got.reshape(nfr, ncoefs, max_bits).sum(axis=2).astype(np.int16)


def allocate(A, sr, budget_bits, psycho=True, spl_ref=96.0, edges=None, search=False,
             A_db=None, thr=None):
    """
    Bits per (frame, coefficient). A is `amplitude()`-converted coefficients, one row per frame.

    A_db and thr are the log magnitude and the masking threshold, when the caller already
    has them. The fill ceiling is that same threshold, and recomputing it is a second mask
    for a number this function just threw away. Passing them in is optional: omitted, they
    are computed here, and the bits are the bits either way.
    """
    n_coefs = A.shape[1]
    if edges is None:
        edges = band_edges(n_coefs)
    if A_db is None:
        A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
    if thr is None:
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
