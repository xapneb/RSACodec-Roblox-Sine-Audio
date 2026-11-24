"""RSC2 v2 codec: analysis, partial selection, and binary layout.

Shared by rsc_encoder.py and rscd_encoder.py so the .rsc2 file and the Desmos JSON can
never drift apart again. The v1 copies of these two files each carried their own scoring
kernel, which is how the selection logic ended up broken in one and merely slow in the other.

What changed v1 -> v2, and why it kills the clicks:

  * Periodic Hann window replaces the rectangular window.
  * hop_size = fft_size / 2. Periodic Hann shifted by half its length sums to exactly 1.0,
    so overlap-add needs no normalisation term. A rectangular window with hop == fft_size
    butt-joined frames end to end, so any phase or amplitude error between two frames landed
    as a one-sample step every hop_size samples. At 48 kHz that is a click 23 times a second.
  * bin 0 (DC) is never selected. A frame that picks DC emits a level step, which is a click
    by construction and carries nothing we want to hear.
  * The salience score no longer rewrites bins while scanning them, and no longer indexes
    past the end of the array. It used to zero score[b-1] and score[b+1] inside the same loop
    that reads them, which made the selection flicker between neighbouring bins and made
    partials appear and vanish across frames.

Frequencies stay bin-quantised. That is the remaining accuracy limit of the format, not a
click source: it costs tuning, not continuity.
"""

import struct

import numpy as np
from numpy.fft import rfft

RSC2_MAGIC   = b"RSC2"
RSC2_VERSION = 2

# Header: version, channels, sample_rate, fft_size, hop_size, max_partials, n_frames,
# n_samples, window_sum. Little-endian, no padding, 24 bytes.
HEADER_FMT      = "<BBIHHHIIf"
FRAME_HDR_FMT   = "<fH"      # frame peak (f32) + partial count (u16)
PARTIAL_FMT     = "<HHH"     # bin, amplitude, phase, all u16

HEADER_SIZE    = struct.calcsize(HEADER_FMT)        # 24
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)     # 6
PARTIAL_SIZE   = struct.calcsize(PARTIAL_FMT)       # 6

FFT_SIZE     = 2048
HOP_SIZE     = FFT_SIZE // 2
MAX_PARTIALS = 24
SCORE_N      = 6
SCORE_HOLE   = 2
SUPPRESS     = 1
_U16_MAX     = 65535.0

# The analysis is batched for speed, but batching every frame of a long file at once means a
# few hundred megabytes of float64 per pass, and salience scoring keeps about ten of them alive
# at a time. This is the number of frames in flight; 256 keeps the whole pipeline around 50 MB
# and still batches the FFT. Lower it if you are running somewhere tight.
ANALYSIS_BLOCK = 256

# v3 (packed) frame layout. A frame is still one fixed-stride record so frame i stays one
# multiplication away, but each partial drops from three words to five bytes:
#
#   bytes 0..2    bin index in the low bits, amplitude quantised against the frame peak above it
#   bytes 3..4    phase u16, the same (-pi, pi] grid v2 used
#
# The bin+amplitude pair gets 24 bits: bin_bits is the smallest that holds this file's bins, so
# the rest goes to amplitude. 13 amplitude bits sit around -78 dB, under anything the analysis
# itself adds; a smaller phase field was tried and measured -44 dB on real audio, because frame
# overlap depends on frame-to-frame phase, so phase keeps all sixteen bits. A silent slot is the
# 5-byte word 0 (bin 0 is never selectable), replacing v1's placeholder-bin trick. Phase stays
# absolute per frame, so decoding a frame never depends on its neighbours and seek stays easy.
PACK_PHASE_BYTES = 2
PACK_MIN_AMP_BITS = 12


def pack_bits(fft_size: int) -> tuple:
    """(bin_bits, amp_bits) for the packed v3 record; phase is stored separately as u16."""
    nyq = fft_size // 2
    if nyq < 2:
        raise ValueError(f"fft_size {fft_size} too small to select any bin")
    bin_bits = (nyq - 1).bit_length()
    amp_bits = 3 * 8 - bin_bits
    if amp_bits < PACK_MIN_AMP_BITS:
        raise ValueError(
            f"fft_size {fft_size} leaves only {amp_bits} bits for amplitude, "
            f"the floor is {PACK_MIN_AMP_BITS}")
    return bin_bits, amp_bits


def frame_stride(max_partials: int, version: int = RSC2_VERSION) -> int:
    """Bytes per frame. v2 writes three words per partial; v3 packs the same triple into five
    bytes. Every frame still writes exactly max_partials records so a decoder can jump straight
    to frame i by multiplying."""
    return 6 + max_partials * (5 if version >= 3 else 6)


def data_start() -> int:
    """Byte offset where frame data starts: 4 magic bytes + 24-byte header."""
    return 4 + HEADER_SIZE


def periodic_hann(n: int) -> np.ndarray:
    """The Hann window whose half-length shifts sum to 1. The symmetric 'np.hanning' does not
    qualify: its edges taper twice at every crossover."""
    m = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * m / n)


def _shift_left(a: np.ndarray) -> np.ndarray:
    """Elementwise predecessor along axis 1, zero-filled at the left edge."""
    out = np.empty_like(a)
    out[:, 0]  = 0.0
    out[:, 1:] = a[:, :-1]
    return out


def _shift_right(a: np.ndarray) -> np.ndarray:
    """Elementwise successor along axis 1, zero-filled at the right edge."""
    out = np.empty_like(a)
    out[:, -1] = 0.0
    out[:, :-1] = a[:, 1:]
    return out


def salience_scores(mags: np.ndarray, score_n: int = SCORE_N,
                    score_hole: int = SCORE_HOLE, suppress: int = SUPPRESS) -> np.ndarray:
    """Salience per bin, batched over frames. mags is (n_frames, n_bins).

    A bin scores high when it stands above the local noise floor (t3) and above its own
    immediate neighbours (t5). Only bins that beat both neighbours keep their score, so the
    top-k picks land on real spectral peaks instead of a peak plus one of its skirts.
    """
    mags    = mags.astype(np.float64, copy=False)
    n_bins  = mags.shape[1]
    log_mag = np.log(mags + 1e-12)

    idx   = np.arange(n_bins)
    lo    = np.maximum(0, idx - score_n)
    hi    = np.minimum(n_bins - 1, idx + score_n)
    pad   = score_n
    cs    = np.concatenate([np.zeros(mags.shape[0])[:, None],
                            np.cumsum(np.pad(mags, ((0, 0), (pad, pad))), axis=1)], axis=1)

    win_sum  = cs[:, hi + pad + 1] - cs[:, lo + pad]
    win_cnt  = hi - lo + 1
    h_lo     = np.maximum(lo, idx - score_hole)
    h_hi     = np.minimum(hi, idx + score_hole)
    hole_sum = cs[:, h_hi + pad + 1] - cs[:, h_lo + pad]
    hole_cnt = h_hi - h_lo + 1

    floor = (win_sum - hole_sum) / np.maximum(win_cnt - hole_cnt, 1)
    t3 = np.maximum(0.0, log_mag - np.log(floor + 1e-12))
    t5 = np.maximum(0.0, log_mag - 0.5 * (_shift_left(log_mag) + _shift_right(log_mag)))

    score = (t3 + t5) * mags

    is_peak = ((score > _shift_left(score)) & (score >= _shift_right(score))
               & (score > 1e-12))
    near_peak = np.zeros_like(is_peak)
    for d in range(1, suppress + 1):          # neighbours of a peak lose, the peak itself keeps
        near_peak |= np.roll(is_peak,  d, axis=1)
        near_peak |= np.roll(is_peak, -d, axis=1)
    score[near_peak & ~is_peak] = 0.0

    score[:, 0] = 0.0                         # DC: a level step, not a partial
    if n_bins > 1:
        score[:, n_bins - 1] *= 0.0           # Nyquist, only ever half a cycle per sample
    return score


# ---------------------------------------------------------------------------
# Psychoacoustics: what is worth an encoder slot, measured in the ear's units.
#
# Salience alone is blind. It cannot tell an audible peak from one buried under a
# louder neighbour, and it has no idea that the ear stops hearing below roughly
# 20 dB of absolute threshold or above the slow high-frequency roll-off. This layer
# scores candidates against two gates:
#
#   * the absolute threshold of hearing (ATH) — a rough ISO-226 minimum audibility
#     curve, referenced to dBFS so that 0 dBFS is `spl_ref_db` dB SPL;
#   * simultaneous masking from the tonal peaks of the frame. Maskers spread
#     linearly on the ERB-rate scale — roughly 10 dB/ERB upward and 25 dB/ERB
#     downward, the Glasberg-Moore tone-masker shape — which is block-vectorised
#     by two cummax scans along the bin axis instead of a full masker-by-masker
#     matrix.
#
# A candidate is eligible if it is itself a masker peak (a peak masks its own slot
# but is still audible), clears the threshold plus `min_new_db`, or was audible
# within `keep_db` below threshold on the previous frame. That last clause is the
# hysteresis that keeps selections from flickering across frame joins, and is the
# whole reason a partially-masked note can fade in and out without clicking.
# ---------------------------------------------------------------------------

# Minimum audibility in air, dB SPL, roughly ISO 226. The exact standard does not
# matter here: this is the "you would not hear it anyway" gate, and its shape is
# what does the work. Interpolated linearly in Hz, so it is monotone and cheap.
_AT_HZ  = np.array([20.0, 31.5, 50, 100, 200, 300, 500, 1000, 2000, 3000,
                    4000, 5000, 6000, 8000, 10000, 12500, 16000])
_AT_SPL = np.array([72, 54, 40, 27, 17, 12, 7, 3, 1, -1, 1, 4, 8, 15, 24, 36, 60])

PSY_PARAMS = {
    "spl_ref_db": 96.0,           # 0 dBFS = 96 dB SPL; 16-bit material played loud
    "mask_up_db_per_erb": 10.0,   # Glasberg-Moore tone-masker slope, upward
    "mask_down_db_per_erb": 25.0, # ...and downward (shallower spread, steeper slope)
    "masker_margin_db": 3.0,      # a peak only masks if it is this far above ATH
    "min_new_db": 3.0,            # a new partial must clear threshold by this much
    "keep_db": 6.0,               # hysteresis: one already playing survives this far
    "rank": "level",              # "level" fills slots loud-first; "salience" peaks-first
}


def erb_axis(freqs: np.ndarray) -> np.ndarray:
    """Glasberg-Moore ERB-rate scale: auditory-filter rate, the axis masking spreads
    along. ERB(f) = 24.7*(4.37e-3*f + 1) Hz; the rate index integrates it so one ERB
    is one unit everywhere, which the Bark approximations get wrong in the corners."""
    f = np.asarray(freqs, dtype=np.float64)
    return 21.4 * np.log10(0.006 * f + 1.0)


def tone_level_db(mags: np.ndarray, fft_size: int) -> np.ndarray:
    """rfft magnitudes as a full-scale-referenced level. A pure tone of amplitude A lands
    an amplitude_scale-corrected A/1.0, so dBFS = 20 log10(A) with 1.0 full scale."""
    amp = amplitude_scale(np.arange(mags.shape[1]), fft_size) * mags * 2.0 / fft_size
    return 20.0 * np.log10(np.maximum(amp, 1e-12))


def masking_threshold_db(mags: np.ndarray, freqs: np.ndarray, fft_size: int,
                         params: dict = None) -> tuple:
    """Return (threshold_db, is_masker) for every bin of every frame, both (n_frames, n_bins).

    threshold_db is the loudest level that stays hidden at that bin: the ATH gate or
    the nearest masker's spread, whichever is higher.
    """
    params = params or PSY_PARAMS
    z   = erb_axis(np.asarray(freqs, dtype=np.float64))
    ath = np.interp(freqs, _AT_HZ, _AT_SPL) - params["spl_ref_db"]
    lvl = tone_level_db(mags, fft_size)

    is_masker = ((mags >= _shift_left(mags)) & (mags >= _shift_right(mags))
                 & (lvl > ath + params["masker_margin_db"]))
    is_masker[:, 0] = False                      # DC is not a tone, never a masker
    is_masker[:, -1] = False                     # nor is Nyquist

    # A masker at bin j raises bin i's threshold by L_j minus slope*(z_i - z_j), linearly
    # in ERB-rate, so the running max is exactly what two prefix scans over the bins give.
    inf = np.full_like(lvl, -np.inf)
    L = np.where(is_masker, lvl, inf)
    up_db   = params["mask_up_db_per_erb"]
    down_db = params["mask_down_db_per_erb"]
    up   = np.maximum.accumulate(L + up_db * z, axis=1) - up_db * z
    down = np.maximum.accumulate((L - down_db * z)[:, ::-1], axis=1)[:, ::-1] + down_db * z
    return np.maximum(ath, np.maximum(up, down)), is_masker


def select_partials(mags: np.ndarray, max_partials: int,
                    score_n: int = SCORE_N, score_hole: int = SCORE_HOLE):
    """Return (bins, mags_sel, peak) for every frame, each (n_frames, max_partials).

    mags is (n_frames, n_bins). Slots that carry nothing come back with a zero magnitude,
    so the frame stride stays constant.
    """
    k = min(max_partials, mags.shape[1])
    score = salience_scores(mags, score_n, score_hole)
    top   = np.argpartition(score, -k, axis=1)[:, -k:]
    order = np.argsort(np.take_along_axis(score, top, 1), axis=1)[:, ::-1]
    bins  = np.take_along_axis(top, order, 1)

    mags_flat = mags if mags.dtype == np.float64 else mags.astype(np.float64)
    picked = np.take_along_axis(mags_flat, bins, 1)
    peak   = picked.max(axis=1)
    return bins, picked, peak


def select_partials_psy(mags: np.ndarray, freqs: np.ndarray, fft_size: int,
                        max_partials: int, params: dict = None,
                        prev_sel: np.ndarray = None,
                        score_n: int = SCORE_N, score_hole: int = SCORE_HOLE):
    """Greedy masked-aware selection, one frame at a time.

    The mask threshold only ever comes from partials this selection actually *plays*.
    That is the whole design: if the model drops a partial under the shadow of a
    masker, the masker must be in the frame, or the shadow is fictional — an MP3-sized
    model pointed at a 24-slot budget would otherwise hide bins behind maskers that
    never survive the budget and quietly lose real signal. So the selection fills its
    slots by salience and, as it accepts each partial, raises the threshold every
    remaining candidate must clear.

    Pass 1 re-plays the previous frame's partials that are still audible within the
    hysteresis, so notes cross frame joins without flickering. Pass 2 fills the rest
    of the budget greedily. Slots that end up empty come back with zero magnitude.

    Returns (bins, mags_sel, peak, played_mask), the last handed back as prev_sel for
    the next block so hysteresis survives block boundaries.
    """
    params = dict(params or PSY_PARAMS)
    mags   = mags.astype(np.float64, copy=False)
    n_frames, n_bins = mags.shape

    score = salience_scores(mags, score_n, score_hole)
    lvl   = tone_level_db(mags, fft_size)
    z     = erb_axis(np.asarray(freqs, dtype=np.float64))
    ath   = np.interp(freqs, _AT_HZ, _AT_SPL) - params["spl_ref_db"]

    up_db   = params["mask_up_db_per_erb"]
    down_db = params["mask_down_db_per_erb"]
    keep_db = params["keep_db"]
    new_db  = params["min_new_db"]
    k       = min(max_partials, n_bins)

    bins   = np.zeros((n_frames, k), np.int64)
    picked = np.zeros((n_frames, k), np.float64)
    played = np.zeros((n_frames, n_bins), dtype=bool)

    prev = np.zeros(n_bins, dtype=bool) if prev_sel is None else prev_sel
    for f in range(n_frames):
        thr   = ath.copy()
        order = np.empty(k, np.int64)
        count = 0
        taken = np.zeros(n_bins, dtype=bool)     # bins this frame already voices

        def accept(b):
            nonlocal thr, count
            order[count] = b
            played[f, b] = True
            taken[b] = True
            L = lvl[f, b]
            spread = np.where(z >= z[b], up_db, down_db) * np.abs(z - z[b])
            thr = np.maximum(thr, L - spread)
            count += 1

        # Pass 1: fill the budget. rank="level" takes the loudest partials the frame
        # can spare and lets masking decide the rest; rank="salience" takes the
        # peakiest, which is closer to how the v2 selector thought.
        # Pass 1: fill the budget. Candidates are peaks (salience > 0 — that is what
        # keeps a flat noise-slope bin from getting voiced); rank="level" then takes
        # them loud-first, rank="salience" peaky-first.
        rank_salience = params.get("rank", "level") == "salience"
        cand = np.argsort(score[f] if rank_salience else lvl[f])[::-1]
        for b in cand:
            if count >= k:
                break
            if score[f][b] > 0.0 and lvl[f][b] > thr[b] + new_db:
                accept(b)

        # Pass 2: keep yesterday's notes, but only ones this frame did not already
        # voice under another bin. Voicing both sides of a note that drifted one bin
        # would double the amplitude and beat — the handoff has to pick the newer bin.
        if prev.any():
            hold = np.flatnonzero(prev)
            for b in hold[np.argsort(lvl[f][hold])[::-1]]:
                if count >= k:
                    break
                if taken[max(b - 1, 0):min(b + 2, n_bins)].any():
                    continue
                if lvl[f][b] > thr[b] - keep_db:
                    accept(b)

        bins[f, :count]   = order[:count]
        bins[f, count:]   = 0
        picked[f, :count] = mags[f][order[:count]]
        prev = played[f]

    peak = picked.max(axis=1)
    return bins, picked, peak, played


def encode_header(sample_rate: int, channels: int, n_channels: int, n_frames: int,
                  n_samples: int, window_sum: float, fft_size: int, hop_size: int,
                  max_partials: int, version: int = RSC2_VERSION) -> bytes:
    return struct.pack("<4sBBIHHHIIf", RSC2_MAGIC, version, n_channels, sample_rate,
                       fft_size, hop_size, max_partials, n_frames, n_samples, window_sum)


def decode_header(data: bytes) -> dict:
    if data[:4] != RSC2_MAGIC:
        raise ValueError(f"not an RSC2 file, magic was {data[:4]!r}")
    v, ch, sr, fft_size, hop_size, mp, nf, ns, ws = struct.unpack_from(HEADER_FMT, data, 4)
    return {"version": v, "channels": ch, "sample_rate": sr, "fft_size": fft_size,
            "hop_size": hop_size, "max_partials": mp, "n_frames": nf, "n_samples": ns,
            "window_sum": ws}


def bin_to_freq(bin_index, sample_rate: int, fft_size: int):
    """Real frequency of a bin. Kept in one place because the Lua decoder does the same math."""
    return np.asarray(bin_index, dtype=np.float64) * sample_rate / fft_size


def amplitude_scale(bin_index, fft_size: int):
    """rfft magnitudes are half the true tone amplitude, except at DC and Nyquist where the
    conjugate half has nowhere to leak. The same rule holds for both the rectangular (v1)
    and the Hann (v2) window."""
    b   = np.asarray(bin_index)
    two = (b > 0) & (b < fft_size // 2)
    return np.where(two, 2.0, 1.0)


def analyse_channel(sig: np.ndarray, sample_rate: int, fft_size: int, hop_size: int,
                    max_partials: int, win: np.ndarray, block: int = ANALYSIS_BLOCK,
                    psycho: bool = True, psy: dict = None):
    """Return (bins, magnitudes, frame peak, phases), the first three shaped so
    magnitudes is (n_frames, max_partials) and peak is (n_frames,).

    Frequencies are not stored: bin k is a tone of k * sample_rate / fft_size Hz. Phases are
    measured from the start of the frame, which is what the decoder needs for overlap-add,
    because two overlapping frames describing the same stationary partial then disagree by
    exactly the window shift and add back into one tone.

    The magnitudes are raw rfft magnitudes. Every consumer turns them into amplitudes with
    amplitude_scale and the window sum, so that convention lives in one place.

    With psycho=True, which is the default, selection is gated by the masking model above
    instead of pure salience: slots go to partials the ear can actually hear. psycho=False
    reproduces the v2 selection byte for byte. The hysteresis state is carried across blocks
    so a partial that survives one block boundary does not flicker at the next.

    Frames go through `block` at a time. Vectorised is worth a lot, but not at the cost of an
    OOM halfway through a two-hour file.
    """
    n_frames = max(0, (len(sig) - fft_size) // hop_size + 1)
    shape = (n_frames, max_partials)
    bins   = np.zeros(shape, np.int64)
    mags   = np.zeros(shape, np.float64)
    phases = np.zeros(shape, np.float64)
    peak   = np.zeros(n_frames, np.float64)
    freqs  = np.arange(fft_size // 2 + 1, dtype=np.float64) * (sample_rate / fft_size)

    prev_sel = None
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        view = np.lib.stride_tricks.as_strided(
            sig[start * hop_size:], shape=(stop - start, fft_size),
            strides=(sig.strides[0] * hop_size, sig.strides[0]), writeable=False)

        spec = rfft(view * win, axis=1)
        if psycho:
            picked, sel_mags, sel_peak, played = select_partials_psy(
                np.abs(spec), freqs, fft_size, max_partials, PSY_PARAMS, prev_sel)
            prev_sel = played[-1] if played.shape[0] else prev_sel
        else:
            picked, sel_mags, sel_peak = select_partials(np.abs(spec), max_partials)
        bins[start:stop]   = picked
        mags[start:stop]   = sel_mags
        peak[start:stop]   = sel_peak
        phases[start:stop] = np.angle(np.take_along_axis(spec, picked, 1))
    return bins, mags, peak, phases
