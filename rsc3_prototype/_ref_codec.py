"""
RSC3 prototype codec: block-transform coding on an MDCT, with a perceptual bit allocation.

Everything here is measured, not assumed:

* The transform is the ISO/IEC-23003-3 / wiki form, kernel
      C[n,k] = cos( pi/N * (n + 0.5 + N/2) * (k + 0.5) ),  block 2N, hop N,
  with the lapped window w[n] = sin( pi (n+0.5) / (2N) ). Analysis X = C^T (w*x), synthesis
  y = w * (2/N) C X, and windowed overlap-add is exact to ~1e-15 (see mdct_fast2.py). Both
  directions run through one complex DFT, so this is the form a Luau decoder would port.
* A full-scale sine measures |X| = N/2 (measured to 0.3%), so `amplitude()` puts coefficients
  in dBFS-comparable units. Every perceptual threshold below is stated in dB relative to full
  scale for exactly that reason: an ATH gate applied to frame-relative magnitudes is a gate that
  moves with the signal and quietly deletes quiet passages.
* The quantiser is the classical scalefactor + mantissa split, so the model's 6.02 dB/bit
  assumption is true of the output rather than merely asserted.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAX_BITS_PER_COEF = 16
BAND_WIDTH = 16                      # coefficients per scalefactor band


# ---------------------------------------------------------------- transform

def window(N2):
    """Lapped analysis/synthesis window, length N2 = 2N."""
    return np.sin(np.pi * (np.arange(N2) + 0.5) / N2)


def dct4(v):
    """
    DCT-IV via one length-L complex inverse DFT, read out in even/odd bin pairs:
        C_k = 2 Re( e^{i pi (k+0.5)/(2L)} * sum_m v_m e^{i pi m/(2L)} e^{i pi m k / L} )
    which is two half-used inverse DFTs, i.e. one length-L complex FFT's worth of work.
    Verified against scipy's DCT-IV to 1e-15 for L = 8 .. 2048.
    """
    L = len(v)
    m = np.arange(L)
    q = v * np.exp(1j * np.pi * m / (2 * L))
    A = np.fft.ifft(q) * L                                  # output bins k = 0, 2, 4, ...
    B = np.fft.ifft(q * np.exp(1j * np.pi * m / L)) * L     # output bins k = 1, 3, 5, ...
    C = np.empty(L)
    r = np.arange(L // 2)
    ke, ko = 2 * r, 2 * r + 1
    C[ke] = 2 * (np.exp(1j * np.pi * (ke + 0.5) / (2 * L)) * A[r]).real
    C[ko] = 2 * (np.exp(1j * np.pi * (ko + 0.5) / (2 * L)) * B[r]).real
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


def analyse(x, N2):
    """
    Signal -> (frames, coefficients). Hop is N2/2. Trailing frames are zero-padded rather than
    dropped, so the last windowed block still reconstructs its samples; nothing is thrown away.
    """
    n_samples = len(x)
    hop = N2 // 2
    n_frames = int(np.ceil(n_samples / hop)) + 1
    X = np.zeros((n_frames, hop))
    w = window(N2)
    for i in range(n_frames):
        s = i * hop
        take = x[s:min(s + N2, n_samples)]
        blk = take if take.shape[0] == N2 else np.concatenate([take, np.zeros(N2 - take.shape[0])])
        X[i] = mdct(w * blk, N2)
    return X, hop


def synthesise(X, N2, out_len):
    """Coefficients -> signal. Windowed overlap-add, the exact inverse of analyse() to ~1e-15."""
    hop = N2 // 2
    w = window(N2)
    out = np.zeros(out_len + 2 * N2)
    for i in range(X.shape[0]):
        s = i * hop
        if s >= len(out):
            break
        take = min(N2, len(out) - s)
        out[s:s + take] += (w * imdct(X[i], N2))[:take]
    return out[:out_len]


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
    """
    # ERB-rate axis, the same function RSC2's psycho layer uses (21.4 log10(1 + 0.006 f), f in
    # Hz). Two conventions for the same axis is how masking slopes end up mis-scaled by 15%, so
    # there is exactly one here. The first version divided f by 1000, which squeezed 24 kHz into
    # about one ERB: no pair of bins was ever far apart, skirts never attenuated, and the
    # threshold came out flat across the frame — which read as a uniform tax and spent nothing.
    hz = np.maximum(np.arange(n_coefs) * sr / (2 * n_coefs), 1.0)
    z = 21.4 * np.log10(1.0 + 0.006 * hz)
    left = np.concatenate([[True], Xdb[:-1] < Xdb[1:]])
    right = np.concatenate([Xdb[:-1] > Xdb[1:], [True]])

    # Prominence is a *local* contrast. Two wrong versions of this both make the filter inert or
    # destructive: a global gate (`Xdb > max - 6`) admits one masker per frame whose skirt then
    # covers everything, and a running min from both ends puts the "valley" at the quietest bin
    # anywhere in the frame, which is 50+ dB below every peak, so every local maximum passes and
    # noise bumps impose thresholds too. The real question is local: how far does it drop within
    # a window on each side? A bump on a slope drops very little upslope and is not a masker.
    w = max(int(prom_window), 1)
    inf = np.full(w, np.inf)
    # lmin[i] = min of the w bins left of i, rmin[i] = min of the w bins right of i. Padded so the
    # ends read as +inf rather than inventing a valley that is not there. A window of length L on
    # an array of length w+L-1 yields exactly one window per bin.
    lp = np.concatenate([inf, Xdb[:-1]])
    rp = np.concatenate([Xdb[1:], inf])
    assert lp.size >= n_coefs - 1 + w and rp.size >= n_coefs - 1 + w
    lmin = sliding_window_view(lp, w).min(axis=1)
    rmin = sliding_window_view(rp, w).min(axis=1)
    dip = Xdb - np.maximum(lmin, rmin)          # drops into a valley on BOTH sides
    peak = (left & right) & (dip > prominence_db)
    cand = np.nonzero(peak)[0]
    if cand.size == 0:
        return np.full(n_coefs, -200.0)
    if cand.size > max_maskers:                       # loudest maskers win
        cand = cand[np.argsort(-Xdb[cand])[:max_maskers]]
    P = 10.0 ** (0.1 * Xdb[cand])
    d = z[None, :] - z[cand, None]                    # masker -> probe, ERBs
    att = np.where(d >= 0, spread_up * d, spread_down * (-d))
    combined = (P[:, None] * 10.0 ** (-0.1 * att)).sum(axis=0)
    # A masker is never masked by itself. Leaving the diagonal in puts the threshold at the
    # masker's own level, which zeroes every loudest bin in the frame — the single most
    # destructive mistake this model can make.
    self_p = np.zeros(n_coefs)
    self_p[cand] = P
    combined = np.maximum(combined - self_p, 0.0)
    return 10.0 * np.log10(np.maximum(combined, 1e-30)) + 10.0 * np.log10(scale)


# ---------------------------------------------------------------- allocation

def band_edges(n_coefs, width=BAND_WIDTH):
    """Contiguous coefficient groups sharing one scalefactor."""
    return np.unique(np.concatenate([[0], np.arange(width, n_coefs + 1, width), [n_coefs]]))


def threshold_dbfs(A_db, sr, n_coefs, psycho=True, spl_ref=96.0, **kw):
    """
    The highest noise level that stays inaudible at each coefficient of one frame: the audibility
    floor, raised wherever a real masker puts maskers there. psycho=False keeps only ATH, which
    is the blind baseline the model has to beat to be worth its complexity.
    """
    ath = ath_dbfs(n_coefs, sr, spl_ref)
    if not psycho:
        return ath
    return np.maximum(ath, mask_db(A_db, sr, n_coefs, **kw))


def waterfill(A_db, threshold_db, band_level_db, budget_bits):
    """
    Reverse water-filling for one frame, in the shape a scalefactor coder actually spends.

    A coefficient in a band whose scale sits at S gets step S*2^-b, so its noise is S - 6.02b and
    the bits that reach the threshold T_k are
        b_k = ceil( (S_band(k) - max(T_k, floor)) / 6.02 )
    which correctly charges extra for coefficients that sit far below their band's peak — the
    thing a per-coefficient formula misses and the reason band scalefactors exist at all. But a
    coefficient already UNDER T_k takes zero bits: zeroing it leaves an error equal to its own
    level, which is under the threshold by definition, while quantising it with the coarse step
    its bits can buy would ADD noise above what it was hiding. Omitting this turns masked bins
    into error sources, which is why a pure tone's quiet neighbours ate 96% of its residual and
    capped the tone at 14.5 dB no matter how many bits the budget offered.
    Shared floor is the only degree of freedom; bisection keeps the *lowest* floor that fits,
    since a lower floor puts bits on more coefficients. Returning the highest feasible floor,
    the obvious mistake, starves the frame and reads as silence.
    """
    lo = float(threshold_db.min()) - 6.02
    hi = float(band_level_db.max()) + 6.02 * MAX_BITS_PER_COEF + 1.0
    best = np.zeros(A_db.shape, int)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        floor = np.maximum(threshold_db, mid)
        # bits only where the coefficient is above the floor: below it, zero is already audibly
        # clean and any step this band can afford is coarser than the bin's own level.
        room = np.where(A_db > floor, (band_level_db - floor) / 6.02, 0.0)
        bits = np.clip(np.where(room > 0, np.ceil(room), 0), 0, MAX_BITS_PER_COEF).astype(int)
        if bits.sum() > budget_bits:
            lo = mid
        else:
            hi = mid
            best = bits
    return best


def allocate(A, sr, budget_bits, psycho=True, spl_ref=96.0, edges=None):
    """
    Bits per (frame, coefficient). A is `amplitude()`-converted coefficients, one row per frame.
    """
    n_coefs = A.shape[1]
    if edges is None:
        edges = band_edges(n_coefs)
    A_db = 20 * np.log10(np.maximum(np.abs(A), 1e-30))
    bits = np.zeros(A.shape, int)
    for i in range(A.shape[0]):
        thr = threshold_dbfs(A_db[i], sr, n_coefs, psycho, spl_ref)
        # band scale = each band's loudest coefficient. A bin sitting well below its band's peak
        # needs more bits to reach the same absolute noise, and this is where that charge comes
        # from — the thing a per-coefficient formula gets wrong and why scalefactors exist.
        blevel = np.zeros(n_coefs)
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            blevel[lo:hi] = A_db[i, lo:hi].max()
        bits[i] = waterfill(A_db[i], thr, np.maximum(blevel, thr), budget_bits)
    return bits


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
    scale = np.zeros_like(A)
    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j + 1]
        scale[:, lo:hi] = np.abs(A[:, lo:hi]).max(axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    step = np.where(bits > 0, 2.0 ** (-bits).astype(float), 0.0) * scale
    q = np.round(A / np.where(step > 0, step, 1.0)) * step
    return q * (bits > 0)
