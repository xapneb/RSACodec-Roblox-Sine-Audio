"""
rsc3_encode.py - RSC3 encoder CLI.

Writes a real RSC3 bitstream, so for the first time the prototype's rate can be measured
instead of quoted. That matters: 512 bits/frame is what the waterfill spends, and a file that
pays for scalefactors and mantissa signs inside that budget costs a different number. This
tool charges both, inside the budget, and reports what it actually wrote.

Usage:
    python rsc3_encode.py input.wav -o out.rsc3 --bitrate 128
    python rsc3_encode.py input.wav -o out.rsc3 --budget 512
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsc3_codec as r3
import rsc3_progress as prog
import rsc3_stream as st

N2 = 4096
# How much mask the dead-band filler is allowed to spend above the quietest bin of the band it
# fills. 0 is the strict ceiling, where the whole band is pinned by its single least masked bin;
# on a 32-bin band that bin is one sample of a 375 Hz range, and it was costing most of the
# coverage the filler exists to provide. The number is not a guess about audibility: because the
# level is capped by min threshold + margin, no bin in the file can sit more than `margin` dB
# above the mask that was supposed to hide it, whatever the material does. Measure with
# fill_audit.py, which reports coverage, the share of bins over mask, and that worst case.
FILL_MARGIN = 0.0


WIDTH = r3.BAND_WIDTH
# The band width `auto` picks, and the widest one this codec can write. The second is a hard
# limit, not a preference: a pyramid shape is written over a whole band, and the count table only
# reaches so many bins, so a wider band is a file whose shape field has no codebook. 32 is where
# that table stops today, which is why the rule below cannot simply go wider.
BAND_AUTO = 32
BAND_MAX = st._PVQ_N if hasattr(st, "_PVQ_N") else WIDTH
# A frame so cheap that its own fields would be the file gets the wider band instead. This is
# measured rather than reasoned: moz.wav at 16 kbit/s reads 12.76 -> 13.71 dB for 2% more bytes,
# and phone.wav at 6 kbit/s 3.93 -> 4.51 dB. At 24 kbit/s the two widths are a wash, 16.36 against
# 16.34, which is why the threshold is allowed to sit below it: 32 is not worse there, it is the
# same sound and less walking. And 64 kbit/s comes out 4.3 dB under the same 32, so the rule dies
# somewhere under there. It is stated in bits per frame and not in kbit/s because that is the
# currency the solve spends, which is also what makes it behave the same on a 44.1 kHz file: 853
# b/f is 20 kbit/s at 48 kHz and 802 at 44.1.
BAND_NARROW = 64
BAND_NARROW_MAX_BPF = 853.0


def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    x = raw.astype(np.float64) / 32768.0
    return (x.reshape(-1, ch) if ch > 1 else x.reshape(-1, 1)), sr


def _mask_bits(ms):
    """Bits the mid/side mask adds across these frames. None is the old stream."""
    if ms is None:
        return 0
    return sum(st.ms_bits(ms[f]) for f in range(len(ms)))


def _field_bits(A, sr, budget, A_db, thr, fill, shape, mode, pvq=False, ms=None,
                sblock=st.SHAPE_BLOCK):
    """Bits write_frame will emit for these frames, plus the mid/side mask.

    With pyramids, the gain exponent is the band energy and the shape bits are
    the closed-form count from the depth. The scalar mantissa field is empty
    on those bands. The mask is only present on channel 0, and its size does
    not depend on the plane count.
    """
    nfr, nc = A.shape
    nb = st.band_grid(nc, WIDTH)
    A3 = A.reshape(nfr, nb, WIDTH)
    bits = band_bits(A, sr, budget, A_db=A_db, thr=thr, pvq=pvq)
    E = st.band_exponents(A3)
    if pvq:
        l2 = np.sqrt(np.maximum((A3 ** 2).sum(axis=-1), 0.0))
        live = bits > 0
        nz = live & (l2 > 0.0)
        Ee = np.zeros_like(E)
        Ee[nz] = np.ceil(np.log2(l2[nz])).astype(np.int64)
        E = np.where(live, Ee, E)
        mant = np.zeros(A3.shape, np.int64)
        pvq_costs = st.pvq_model_costs(bits, WIDTH)
    else:
        mant = st.quantise_bands(A3, E, bits)
        pvq_costs = None
    fills = None
    shapes = None
    # Same rule encode_channel uses for needing a mask: the run law wants it, and so does the
    # shape word's ceiling, and only when a field is actually being written. Where either side
    # got a mask the other did not, the drift check would be the only thing left telling the
    # truth, and it only reports a number, not which field forgot to ask.
    need_thr = (fill and mode == "safe") or (fill and shape)
    cap_thr = thr.reshape(nfr, nb, WIDTH) if (need_thr and thr is not None) else None
    if fill:
        caps = st.band_fill_cap(A3, cap_thr if mode == "safe" else None, FILL_MARGIN)
        fills = st.run_fill_words_all(caps, bits, mode)
    if shape:
        reg = st.shape_reg(sr, nc * 2, WIDTH)
        energy = st.shape_level_cap(A3, cap_thr, bits, FILL_MARGIN)
        shapes = st.shape_window(energy, bits, reg, sblock)
    return int(st.frame_costs(E, bits, mant, WIDTH, fill_levels=fills,
                              shape_fields=shapes, pvq_costs=pvq_costs).sum()) + _mask_bits(ms)


def _sample_starts(nfr, win=48, nwin=5):
    """A few runs spread through the file. Sixteen frames at the tail of beep.wav
    is quiet enough to understate the rate by a fifth, so the runs are longer.
    The Rice fields need the frames in order, so each run is contiguous and the
    runs are costed on their own."""
    if nfr <= win or nwin <= 1:
        return [0], nfr
    nwin = min(nwin, nfr // win)
    if nwin <= 1:
        return [0], min(win, nfr)
    span = nfr - win
    return [int(round(i * span / (nwin - 1))) for i in range(nwin)], win


def _sample_bpf(A, A_db, thr, sr, budget, fill, shape, mode, pvq=False, ms=None,
                progress=None, sblock=st.SHAPE_BLOCK):
    starts, win = _sample_starts(A.shape[0])
    bits = 0
    frames = 0
    for s in starts:
        if progress is not None:
            progress()
        w = min(win, A.shape[0] - s)
        bits += _field_bits(A[s:s + w], sr, budget, A_db[s:s + w], thr[s:s + w],
                            fill, shape, mode, pvq=pvq,
                            ms=None if ms is None else ms[s:s + w], sblock=sblock)
        frames += w
    return bits / frames


def plane_budget_for(A_all, prepared, sr, target_bpf, fill, shape, mode, pvq, psycho,
                     ms_all=None, progress=None, sblock=st.SHAPE_BLOCK):
    """Plane count whose written fields land on target_bpf file bits per frame.

    A waterfill plane is one coefficient bit, or one step of band depth when
    pyramids are on. The file then adds the side fields, so the plane count is
    not the file rate. The count here is the largest one whose field sizes,
    measured on a few runs of frames, stay within the target. The pyramid model
    counts the shape from the depth. One sample of the real payload catches the
    exponent field, which depends on the energy values themselves. Mask bits
    are in both the model and that sample, on the channel that carries them.
    """
    if ms_all is None:
        ms_all = [None] * len(A_all)
    elif len(ms_all) != len(A_all):
        raise ValueError("mid/side mask list does not match the channel count")
    cache = {}

    def bpf(budget):
        budget = int(budget)
        hit = cache.get(budget)
        if hit is None:
            vals = [_sample_bpf(A, Adb, thr, sr, budget, fill, shape, mode, pvq=pvq,
                                ms=ms, progress=progress, sblock=sblock)
                    for A, (Adb, thr), ms in zip(A_all, prepared, ms_all)]
            hit = sum(vals) / len(vals)
            cache[budget] = hit
        return hit

    cap = int(A_all[0].shape[1] * st.MAX_BITS)

    def solve(target):
        # Climb from one plane. The file's plane count sits far below the bit
        # target, and a probe up there prices every band at full depth. The
        # largest count whose fields still fit is unchanged: the cost grows
        # as planes are added, which is what the old bracket also assumed.
        if bpf(1) >= target:
            return 1
        lo, hi = 1, 1
        while hi < cap and bpf(hi) < target:
            lo = hi
            hi = min(cap, max(hi + 1, hi * 2))
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if bpf(mid) <= target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    chosen = solve(target_bpf)
    if not pvq:
        return chosen
    bits = 0
    frames = 0
    for A, (Adb, thr), ms in zip(A_all, prepared, ms_all):
        starts, win = _sample_starts(A.shape[0])
        for s in starts:
            w = min(win, A.shape[0] - s)
            payload, _ = encode_channel(
                A[s:s + w], sr, chosen, psycho=psycho, fill=fill, mode=mode,
                shape=shape, build_recon=False, A_db=Adb[s:s + w], thr=thr[s:s + w],
                pvq=True, ms=None if ms is None else ms[s:s + w], progress=progress,
                sblock=sblock)
            bits += len(payload) * 8
            frames += w
    model = bpf(chosen)
    got = bits / frames if frames else 0.0
    if model <= 0.0 or got <= 0.0:
        return chosen
    ratio = got / model
    if abs(ratio - 1.0) < 0.02:
        return chosen
    return solve(target_bpf / ratio)


def band_bits(A, sr, budget, psycho=True, A_db=None, thr=None, pvq=False):
    """
    Depth of each band.

    Scalar coding ranks coefficient bit-planes and a band pays for its loudest
    bin. Pyramid coding ranks each band as part of one auditory filter: the
    price of a step is the band's energy minus half that filter's energy, 6.02
    dB down per extra bit. With the mask off, that half is left out and the
    rank is the raw level. Masked bands stay closed.
    """
    nfr, nc = A.shape
    nb = st.band_grid(nc, WIDTH)
    if not pvq:
        b = r3.allocate(A, sr, budget, psycho=psycho, A_db=A_db, thr=thr)
        b = b.reshape(nfr, nb, WIDTH).max(axis=2)
        return np.clip(b.astype(np.int64), 0, st.MAX_BITS)
    if A_db is None:
        A_db = 20.0 * np.log10(np.maximum(np.abs(A), 1e-30))
    if thr is None:
        thr = r3.threshold_dbfs(A_db, sr, nc, psycho=psycho)
    A3 = A.reshape(nfr, nb, WIDTH)
    l2 = np.sqrt(np.maximum((A3 ** 2).sum(axis=-1), 1e-30))
    level = 20.0 * np.log10(l2)
    thr_b = np.min(thr.reshape(nfr, nb, WIDTH), axis=-1)
    share = None
    if psycho:
        hz = (np.arange(nb) + 0.5) * WIDTH * float(sr) / (2.0 * nc)
        share = r3.ear_filter_share_db(level, hz)
    b = r3.waterfill_select(level, thr_b, np.maximum(level, thr_b), budget, share_db=share)
    return np.clip(b.astype(np.int64), 0, st.MAX_BITS)


# Power ratio. 0.1 is 10 dB: the quieter of mid and side has to be that far
# under the quieter input before the group rotates. A group is MS_GROUP bands
# pooled, because a decision per band on a wide mix was 128 bits and the split
# did not earn them. A centered source still clears the whole frame, which is
# two bits, with the side gone. A near tie does not rotate.
MS_QUIET = 0.1


def ms_mask(L, R, width=None, quiet=MS_QUIET, group=None):
    """1 where mid/side leaves a clearly quieter channel than left/right.

    Mid is (L+R)/sqrt(2) and side is (L-R)/sqrt(2). The test pools `group`
    bands, then repeats that one decision across them. Identical channels
    rotate, and so do opposite ones. A silent channel stays left/right:
    splitting it would make two half-loud channels. `quiet` is the fraction
    the coded channel has to beat. 1 keeps any win.
    """
    width = int(WIDTH if width is None else width)
    group = st.MS_GROUP if group is None else int(group)
    if group < 1:
        raise ValueError(f"mid/side group {group} is not a positive count")
    Ls = np.asarray(L, np.float64).reshape(L.shape[0], -1, width)
    Rs = np.asarray(R, np.float64).reshape(R.shape[0], -1, width)
    nfr, nb, _ = Ls.shape
    if nb % group:
        raise ValueError(f"{nb} bands is not a multiple of mid/side group {group}")
    if group > 1:
        Ls = Ls.reshape(nfr, nb // group, group * width)
        Rs = Rs.reshape(nfr, nb // group, group * width)
    el = (Ls * Ls).sum(axis=-1)
    er = (Rs * Rs).sum(axis=-1)
    em = ((Ls + Rs) * (Ls + Rs)).sum(axis=-1) * 0.5
    es = ((Ls - Rs) * (Ls - Rs)).sum(axis=-1) * 0.5
    coarse = (np.minimum(em, es) < float(quiet) * np.minimum(el, er)).astype(np.uint8)
    if group == 1:
        return coarse
    return np.repeat(coarse, group, axis=1)


def encode_channel(A, sr, budget, psycho, want_cost=False, fill=True, mode="safe",
                   shape=False, build_recon=True, A_db=None, thr=None,
                   pvq=False, pvq_cap=0, progress=None, ms=None, sblock=st.SHAPE_BLOCK):
    """
    Returns (payload, recon), where recon is built by the shared reconstruct_bands - the same
    code the decoder will run - so the encoder's quality number is the decoder's, not the
    encoder's opinion of it.

    fill turns the dead-band noise fill on or off, and mode says which way it is levelled. It is
    an argument rather than a constant because it changes the sound: the encoder must measure the
    quality of the audio it will actually deliver, and an ES measured without the fill would be
    a claim about a file that does not exist. The words used for synthesis and the words written
    to the payload are the same object, so that cannot come apart.

    shape adds the fifth field on the dead bands above SHAPE_F0. A block of those bands is
    noise at the block's own energy when the bands are dense and the block is close to the
    rest of the frame, so treble a small mantissa budget never opens is still in the file.
    A partial, or a block that is only a little energy, is sent as silence rather than smeared
    across the block. The words are side info. They are not taken out of the mantissa budget,
    and a band the allocator did open is left as coefficients.

    pvq gives every live band one gain and one shape. The gain is the band's
    energy, and the shape is a pyramid over all of its bins, with as many pulses
    as that band's depth can hold. A band the allocator left closed is still
    fill or shape. pvq_cap limits pulses in one band. 0 leaves the count to the
    depth. The decoder never sees the cap.

    build_recon is what the quality number is for. The payload does not need the samples, so a
    rate search that is still moving the budget skips them and builds them once, on the budget
    it keeps. A_db and thr, when the caller already computed them, are the allocation's own
    numbers: the fill ceiling is that threshold, and a second mask would be the same work again.

    progress, when set, is called once per frame after that frame is written. The CLI uses it
    for the status line. Callers that do not care pass nothing and get no extra work.

    ms is the per-band mask for this channel, or None. Only channel 0 carries it, ahead of
    each frame. None leaves the payload identical to a file with the flag clear.
    """
    nfr, nc = A.shape
    nb = st.band_grid(nc, WIDTH)
    if ms is not None:
        ms = np.asarray(ms, np.uint8)
        if ms.shape != (nfr, nb):
            raise ValueError(f"mid/side mask is {ms.shape}, expected {(nfr, nb)}")
    # The fill ceiling is a masking question, and the allocation already knows the mask, so
    # taking the same threshold here is not extra theory - it is the same number, asked for.
    # "mean" fill does not need the run ceiling and getting it anyway would cost a mask per
    # frame for nothing. The shape field does need it: a shape word is the band's own energy,
    # and a band that was given no bits is precisely a band whose energy it could not hide.
    # Uncapped, that is not a small slip - measured through the real files, the shape field put
    # three bins in five above the mask it was filling, and the worst of them by 40 dB. See
    # fill_audit.py, which is what that sentence is standing on.
    need_thr = (fill and mode == "safe") or (fill and shape)
    run_cap = fill and mode == "safe"
    if A_db is None and (need_thr or thr is not None):
        A_db = 20.0 * np.log10(np.maximum(np.abs(A), 1e-30))
    if need_thr and thr is None:
        thr = r3.threshold_dbfs(A_db, sr, nc, psycho=psycho)
    bits = band_bits(A, sr, budget, psycho, A_db=A_db,
                     thr=thr if (pvq or run_cap or thr is not None) else None,
                     pvq=pvq)
    A3 = A.reshape(nfr, nb, WIDTH)
    E_all = st.band_exponents(A3)
    if pvq:
        # A pyramid band writes no mantissas. Placing every live band of the
        # channel in one pass is the same greedy score as one band at a time.
        # One band at a time was a fresh NumPy call per band, which is where a
        # loud file spent the encode.
        M_all = np.zeros(A3.shape, np.int64)
        recs_all, pulses_all, E_all = st.plan_pvq_frames(
            A3, E_all, bits, k_cap=pvq_cap, synth=build_recon)
    else:
        M_all = st.quantise_bands(A3, E_all, bits)
        recs_all = None
        pulses_all = None
    thr3 = thr.reshape(nfr, nb, WIDTH) if need_thr else None
    caps = st.band_fill_cap(A3, thr3 if mode == "safe" else None,
                            FILL_MARGIN) if fill else None
    energy = st.shape_level_cap(A3, thr3, bits, FILL_MARGIN) if shape else None
    reg = st.shape_reg(sr, 2 * nc, WIDTH) if shape else None
    shapes = st.shape_window(energy, bits, reg, sblock) if shape else None
    # The fill words are a pure function of this frame's dead runs and its own ceilings, so
    # they can be computed for every frame before the loop instead of inside it. Nothing here
    # decides anything: it is the same level and slope the per-frame call would have returned.
    words_all = st.run_fill_words_all(caps, bits, mode) if fill else None
    recon = np.zeros((nfr, nc)) if build_recon else None
    w = st.Writer()
    prev_E = np.zeros(nb, np.int64)
    prev_b = np.zeros(nb, np.int64)
    cost = np.zeros(6, np.int64) if want_cost else None

    for f in range(nfr):
        b = bits[f]
        E = E_all[f]
        mant = M_all[f]
        words = words_all[f] if fill else None
        sf = shapes[f] if shape else None
        recs = None
        pulses = None
        if pvq:
            # E on a pyramid band is already the band-energy exponent. That is
            # the exponent the decoder multiplies q by, so the stored scale and
            # the gain are the same number.
            recs = recs_all[f]
            pulses = None if pulses_all is None else pulses_all[f]
        if build_recon:
            sw = st.shape_expand(sf, b, reg, sblock) if shape else None
            recon[f] = st.reconstruct_bands(E, b, mant, words, f, WIDTH, shape=sw,
                                            pulses=pulses).reshape(-1)

        # Channel 0 only, ahead of the frame, so channel 1 keeps the old grammar.
        # The bits sit in the depth bucket: the drift check sums the buckets.
        spent = st.write_ms(w, ms[f]) if ms is not None else 0
        E, field = st.write_frame(w, E, b, mant, prev_E, prev_b, WIDTH,
                                  fill=words, shape=sf, pvq=recs)
        if want_cost:
            cost += np.array(field, np.int64)
            cost[1] += spent
        prev_E, prev_b = E, b
        if progress is not None:
            progress()

    if want_cost:
        return w.done(), recon, cost
    return w.done(), recon


def main():
    ap = argparse.ArgumentParser(description="RSC3 encoder (MDCT block transform + waterfill)")
    ap.add_argument("input", help="input .wav")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--bitrate", type=float, default=128.0,
                    help="kbit/s per channel of the file. The waterfill's plane count is "
                         "chosen so the written fields come out near this rate. "
                         "Ignored when --budget is set")
    ap.add_argument("--budget", type=int, default=None,
                    help="waterfill bit-planes per frame, used as given. One plane is not "
                         "one file bit: a band codes every bin at its loudest depth, and "
                         "side info is extra. Overrides --bitrate")
    ap.add_argument("--ceiling", type=float, default=1.0,
                    help="if the decoded peak sits above this, store the gain that puts it "
                         "there. Measured per file, not a fixed pad. 0 leaves the level alone")
    ap.add_argument("--blind", action="store_true",
                    help="disable frequency masking; bits follow level against a flat floor")
    ap.add_argument("--temporal-mask", type=float, default=0.0, metavar="DB", dest="temporal",
                    help="lower the mask DB on a frame that just took a step up in energy, and on "
                         "the frame leading into it, so a transient is allowed to buy bits the "
                         "frame's own masking would have refused it. 0 (default) is the shipped "
                         "mask, and an old file's decisions stay reproducible. Encoder only: "
                         "nothing about it reaches the stream")
    ap.add_argument("--fill", dest="fill", action="store_true", default=True,
                    help="fill dead bands with level-coded noise (default, and the default rate "
                         "the header says)")
    ap.add_argument("--no-fill", dest="fill", action="store_false",
                    help="leave dead bands silent: the previous bit map, byte for byte")
    ap.add_argument("--fill-mode", choices=("safe", "mean"), default="safe",
                    help="safe (default) levels each run as loud as ITS OWN MASK allows, so "
                         "nothing measured is above the audibility floor; mean levels the run "
                         "at its own energy, which is louder and airier and, measured, lands "
                         "about one band in five above the mask")
    shape = ap.add_mutually_exclusive_group()
    shape.add_argument("--fill-shape", dest="fill_shape", action="store_true",
                       help="noise-fill dense dead bands above the shape cutoff at their own "
                            "energy (the default whenever fill is on). A sparse partial or a "
                            "block far below the frame is left to the flat fill. Side info, "
                            "not mantissa bits")
    shape.add_argument("--no-fill-shape", dest="fill_shape", action="store_false",
                       help="leave dead high bands to the flat fill")
    ap.set_defaults(fill_shape=None)
    ap.add_argument("--fill-margin", dest="fill_margin", type=float, default=0.0, metavar="DB",
                    help="dB of mask the dead-band filler may spend above the quietest bin of "
                         "the band it fills. 0 is the strict ceiling, where one least-masked bin "
                         "pins the whole band; 6 to 12 buys back coverage and can never leave a "
                         "bin more than that far over its mask")
    ap.add_argument("--band-width", dest="band_width",
                    default="auto",
                    help="bins per scalefactor band. auto (default) is 32, or 64 when the frame "
                         "budget is so small the fields would be the file, which halves the scale "
                         "and mask fields against 16 and spends the saving on coefficients, where "
                         "it buys more than the frequency resolution it gives up. 16 is what every "
                         "file before this had, for A/B against an old stream")
    ap.add_argument("--pvq", action="store_true",
                    help="each live band is one energy and a pyramid shape over all of "
                         "its bins. Dead bands stay fill or silence. Off unless asked")
    ap.add_argument("--pvq-cap", type=int, default=0,
                    help="max pulses in one band. 0 spends only the bits that "
                         "band's scalar mantissas would have used")
    ap.add_argument("--shape-block", dest="shape_block", choices=("auto", "2", "4", "8"),
                    default="auto",
                    help="bands per shape word. auto (default) writes the wide word, which on "
                         "beep.wav costs a tenth of the shape field at 10 kbit/s and two thirds "
                         "of it at 24, and scores higher ES at both. 2 is the previous "
                         "bitstream, byte for byte, and is here for A/B against it. The width "
                         "goes in the header, so a decoder never has to agree with anything "
                         "the encoder decided")
    stereo = ap.add_mutually_exclusive_group()
    stereo.add_argument("--ms", dest="ms", action="store_true",
                        help="code a stereo band as mid/side when the quieter of mid and "
                             "side is about 10 dB below the quieter of left and right, "
                             "judged on four bands at a time. Default for two channels. "
                             "A frame that is all one way is two bits; a mixed frame "
                             "sends one decision per group of four")
    stereo.add_argument("--no-ms", dest="ms", action="store_false",
                        help="code left and right independently, the previous stereo bitstream")
    ap.set_defaults(ms=None)
    args = ap.parse_args()
    if args.fill_shape is None:
        args.fill_shape = args.fill
    global FILL_MARGIN
    FILL_MARGIN = args.fill_margin
    # One geometry for the whole file, chosen before the rate solve so the solve and the
    # payload agree, and written to the header so the decoder reads what was written. Band width
    # belongs to that choice, and is taken below because its rule needs the sample rate.
    if args.shape_block == "auto":
        sblock = st.SHAPE_BLOCK_WIDE
    elif args.shape_block == "2":
        sblock = st.SHAPE_BLOCK
    else:
        sblock = int(args.shape_block)

    x, sr = read_wav(args.input)
    nch = x.shape[1]
    n_samp = x.shape[0]
    nfr = st.frame_count(n_samp, N2)
    dur = n_samp / sr
    # The band width is the other half of that geometry, and it is already a header field, so the
    # decoder takes what is written and no format decision is hiding in it. Asked for explicitly
    # it is only checked; left to auto it is read off the frame budget, which is why it waits for
    # the sample rate.
    global WIDTH
    if args.band_width != "auto":
        w = int(args.band_width)
        if w < 1 or w > BAND_MAX:
            raise SystemExit(f"band width {w} is not between 1 and {BAND_MAX} bins, which is all "
                             f"the pyramid tables can name. The encoder would otherwise write a "
                             f"file whose shape field has no codebook")
        if (N2 // 2) % w:
            # A band is a slice of the coefficient row, so the grid has to divide it. This is the
            # reason the fitted optimum and the width actually used are different numbers: the
            # quality surface wants 25 bins at 64 kbit/s and 42 at 24 kbit/s, and 2048 = 2**11
            # coefficients divide only into powers of two, so the curve is sampled at the nearest
            # point it permits rather than followed. Measured, that costs 0.1 to 0.3 dB, which is
            # why nobody has rebuilt the grid to take an uneven last band.
            raise SystemExit(f"band width {w} does not divide {N2 // 2} coefficients into whole "
                             f"bands; this geometry takes widths that divide evenly")
        WIDTH = w
    else:
        bpf = args.budget if args.budget else args.bitrate * 1000.0 / (sr / (N2 // 2))
        WIDTH = BAND_NARROW if bpf <= BAND_NARROW_MAX_BPF else BAND_AUTO
        if WIDTH > BAND_MAX:
            WIDTH = BAND_AUTO
    r3.BAND_WIDTH = WIDTH
    out = args.output or (os.path.splitext(args.input)[0] + ".rsc3")
    print(f"RSC3 encode   {args.input}", flush=True)
    print(f"   {n_samp} samples  {dur:.2f}s  {sr} Hz  {nch}ch   {nfr} frames   "
          f"block {N2}  {WIDTH}-bin bands  {N2 // 2 // WIDTH} bands", flush=True)

    # One pass. The bar covers the frames plus one step per channel for the error score.
    n_pass = nfr * nch
    bar = prog.Bar(n_pass + nch, "transform")
    phase = ["encode"]

    psycho = not args.blind
    r3.TEMPORAL_MASK = float(args.temporal)
    if args.ms is None:
        args.ms = nch == 2
    if args.ms and nch != 2:
        print("   mid/side needs two channels; leaving them independent", flush=True)
        args.ms = False
    A_all = [r3.amplitude(r3.analyse(x[:, c], N2)[0], N2) for c in range(nch)]
    # Rotate before the mask and the rate solve. Waterfill, fill, shape, and
    # the pyramid then see the channels the file actually stores. An all-clear
    # mask leaves the spectra alone and the flag clear, so that file stays the
    # old independent bitstream.
    ms_rows = None
    ms_all = [None] * nch
    if args.ms:
        if A_all[0].shape != A_all[1].shape:
            raise ValueError("the two channels do not share a spectrum shape")
        ms_rows = ms_mask(A_all[0], A_all[1])
        if np.any(ms_rows):
            A_all[0], A_all[1] = st.ms_rotate(A_all[0], A_all[1], ms_rows, WIDTH)
            ms_all[0] = ms_rows
        else:
            ms_rows = None
            print("   mid/side left off: no band got quieter that way", flush=True)
    bar.set_label("mask")
    # One mask per channel. The fill ceiling is that threshold, not a second model.
    prepared = []
    for A in A_all:
        Adb = 20.0 * np.log10(np.maximum(np.abs(A), 1e-30))
        prepared.append((Adb, r3.threshold_dbfs(Adb, sr, A.shape[1], psycho=psycho)))
    # Hop is N2/2. --bitrate is file bits per second, and the plane count is whatever
    # makes the written fields add up to that. --budget skips the conversion.
    frate = sr / (N2 // 2)
    if args.budget is None:
        bar.set_label("rate")
        target_bpf = args.bitrate * 1000.0 / frate
        chosen = plane_budget_for(A_all, prepared, sr, target_bpf, args.fill,
                                  args.fill_shape, args.fill_mode, args.pvq, psycho,
                                  ms_all=ms_all, progress=bar.pump, sblock=sblock)
    else:
        chosen = args.budget

    def run(budget):
        payloads, recons, costs = [], [], []
        for ci, (A, (Adb, th)) in enumerate(zip(A_all, prepared)):
            if nch > 1:
                bar.set_label(f"{phase[0]}  ch {ci + 1}/{nch}")
            p, rc, cost = encode_channel(A, sr, budget, psycho=psycho,
                                         want_cost=True, fill=args.fill,
                                         mode=args.fill_mode, shape=args.fill_shape,
                                         A_db=Adb, thr=th,
                                         pvq=args.pvq, pvq_cap=args.pvq_cap,
                                         progress=bar.step, ms=ms_all[ci], sblock=sblock)
            payloads.append(p)
            recons.append(rc)
            costs.append(cost)
        kbps = sum(len(p) for p in payloads) * 8.0 / dur / 1000.0
        return payloads, recons, kbps, np.sum(costs, axis=0) / nch, [
            abs(int(c.sum()) - len(p) * 8) for c, p in zip(costs, payloads)]

    try:
        phase[0] = f"encode  {chosen} b/f"
        bar.set_label(phase[0])
        payloads, recons, kbps, cost, drift = run(chosen)
        if ms_rows is not None:
            recons[0], recons[1] = st.ms_unrotate(recons[0], recons[1], ms_rows, WIDTH)

        # quality of the decoded file, measured on the reconstruction the decoder will produce.
        # The playback gain is applied later, at the samples, and is kept out of this number:
        # a pure level offset caps ES at the size of the offset.
        lo_i = 3 * N2
        err = 0.0
        sig = 0.0
        peak = 0.0
        bar.set_label("measuring")
        for c, rc in enumerate(recons):
            y = r3.synthesise(rc * (N2 / 4.0), N2, n_samp)
            bar.step()
            peak = max(peak, float(np.max(np.abs(y))))
            ref = x[lo_i:-N2, c]
            err += float(np.mean((y[lo_i:-N2] - ref) ** 2))
            sig += float(np.mean(ref ** 2))
        es = 10 * math.log10(sig / max(err, 1e-300))
        gain = st.playback_gain(peak, args.ceiling)
        bar.close()
    finally:
        bar.close()

    print(f"   budget {chosen} b/f   scale {cost[0] / nfr:7.1f} b/f   "
          f"bits {cost[1] / nfr:6.1f} b/f   mantissa {cost[2] / nfr:7.1f} b/f   "
          f"fill {cost[3] / nfr:5.1f} b/f"
          + (f"   shape {cost[4] / nfr:5.1f} b/f" if args.fill_shape else "")
          + (f"   pvq {cost[5] / nfr:5.1f} b/f" if args.pvq else "")
          + "   (per channel)")
    coef_bits = cost[2] + cost[5]
    print(f"   side info is {(cost.sum() - coef_bits) / max(cost.sum(), 1) * 100:.0f}%"
          f" of the file   dead-band fill "
          f"{'on' if args.fill else 'OFF, silence where the source had something'}"
          + (f", signal-shaped noise above {st.SHAPE_F0 / 1e3:.0f} kHz" if args.fill_shape else "")
          + (", pyramid vectors on every live band" if args.pvq else "")
          + (", mid/side where that split is quieter" if ms_rows is not None else "")
          + (f", mask lifted {args.temporal:.0f} dB into an attack" if args.temporal else "")
          + (f", shape words span {sblock} bands"
             if args.fill_shape and sblock != st.SHAPE_BLOCK else ""))
    # The fields must add up to the bytes. Attributed bits can only differ from the file by
    # the final byte-flush padding, so a drift over 7 bits means a field went unwritten, or
    # read twice, and no rate number printed below is worth quoting.
    bad = [d for d in drift if d > 7]
    print(f"   field accounting vs bytes on disk: drift {max(drift)} bits "
          f"{'(flush padding only)' if not bad else '*** UNACCOUNTED BITS ***'}")
    if args.budget is None:
        print(f"   budget {chosen} b/f   set for {args.bitrate:g} kbit/s per channel")
    if ms_rows is not None:
        print(f"   mid/side on {100.0 * float(np.mean(ms_rows)):.0f}% of bands"
              f"   mask {_mask_bits(ms_rows) / nfr:.1f} b/f")
    h = st.pack_header(sr, nch, N2, WIDTH, n_samp, chosen, [len(p) for p in payloads],
                       flags=(st.FILL_FLAG if args.fill else 0)
                             | (st.SHAPE_FLAG if args.fill_shape else 0)
                             | (st.PVQ_FLAG if args.pvq else 0)
                             | (st.MS_FLAG if ms_rows is not None else 0)
                             | (st.shape_block_shift(sblock) << st.SHAPE_BLOCK_SHIFT_W
                                if args.fill_shape and sblock != st.SHAPE_BLOCK else 0),
                       gain=gain)
    with open(out, "wb") as f:
        f.write(h)
        for p in payloads:
            f.write(p)
    size = len(h) + sum(len(p) for p in payloads)
    over = 20 * math.log10(peak) if peak > 0 else -math.inf
    att = 20 * math.log10(gain) if gain > 0 else -math.inf
    print(f"   output peak {over:+.2f} dBFS   playback gain {att:+.2f} dB"
          + ("" if gain < 1.0 else "   (under the ceiling, left alone)"))
    print(f"   {kbps / nch:.2f} kbit/s per channel   {kbps:.2f} total   "
          f"file {size:,} B   ES {es:.3f} dB")
    print(f"   wrote {out}")
    print(f"   decode with: python rsc3_decode.py {out} -o decoded.wav")


if __name__ == "__main__":
    main()
