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


def data_start() -> int:
    """Byte offset where frame data starts: 4 magic bytes + 24-byte header."""
    return 4 + HEADER_SIZE


def frame_stride(max_partials: int) -> int:
    """Bytes per frame. Every frame writes exactly max_partials records so a decoder
    can jump straight to frame i by multiplying. Keep n_p == max_partials in the output."""
    return FRAME_HDR_SIZE + max_partials * PARTIAL_SIZE


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
                    max_partials: int, win: np.ndarray, block: int = ANALYSIS_BLOCK):
    """Return (bins, magnitudes, frame peak, phases), the first three shaped so
    magnitudes is (n_frames, max_partials) and peak is (n_frames,).

    Frequencies are not stored: bin k is a tone of k * sample_rate / fft_size Hz. Phases are
    measured from the start of the frame, which is what the decoder needs for overlap-add,
    because two overlapping frames describing the same stationary partial then disagree by
    exactly the window shift and add back into one tone.

    The magnitudes are raw rfft magnitudes. Every consumer turns them into amplitudes with
    amplitude_scale and the window sum, so that convention lives in one place.

    Frames go through `block` at a time. Vectorised is worth a lot, but not at the cost of an
    OOM halfway through a two-hour file.
    """
    n_frames = max(0, (len(sig) - fft_size) // hop_size + 1)
    shape = (n_frames, max_partials)
    bins   = np.zeros(shape, np.int64)
    mags   = np.zeros(shape, np.float64)
    phases = np.zeros(shape, np.float64)
    peak   = np.zeros(n_frames, np.float64)

    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        view = np.lib.stride_tricks.as_strided(
            sig[start * hop_size:], shape=(stop - start, fft_size),
            strides=(sig.strides[0] * hop_size, sig.strides[0]), writeable=False)

        spec = rfft(view * win, axis=1)
        picked, sel_mags, sel_peak = select_partials(np.abs(spec), max_partials)
        bins[start:stop]   = picked
        mags[start:stop]   = sel_mags
        peak[start:stop]   = sel_peak
        phases[start:stop] = np.angle(np.take_along_axis(spec, picked, 1))
    return bins, mags, peak, phases
