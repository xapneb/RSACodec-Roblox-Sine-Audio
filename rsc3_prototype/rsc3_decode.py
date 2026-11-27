"""
rsc3_decode.py - RSC3 decoder CLI.

The point of this file is that it must not need the encoder. It reads the container and the
transform, and nothing else: no allocation model, no thresholds, no band maxima, no
encoder-side state. Until this file can produce sound from a .rsc3 alone, RSC3 has no
decoder, and every quality number in the prototype describes a codec that cannot be heard.

Usage:
    python rsc3_decode.py out.rsc3 -o decoded.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsc3_codec as r3
import rsc3_progress as prog
import rsc3_stream as st


def decode_channel(payload, nfr, n2, width, fill=False, shape=False, sr=None, pvq=False,
                   progress=None, ms=False, sblock=st.SHAPE_BLOCK):
    """
    Rebuild the amplitude-domain coefficients using only what the stream says. The step comes
    from E and b in the file, and the amplitudes come out of reconstruct_bands, which is the
    encoder's own formula.

    fill comes from the header flag, not from guessing. Everything else the decoder needs for
    the noise - which bands are dead, and what values the noise takes - it can derive, and that
    is precisely why the fill is affordable. What it cannot derive is how loud, so that is the
    one thing the file has to say. shape is likewise the header's business: the region comes
    from sr and the block geometry, both of which the header already carries.

    pvq is the header flag for a pyramid on every live band: the band's energy and
    a shape over all of its bins, in place of the scalar mantissas. Dead bands,
    including shaped noise, are read the same way either way.

    progress, when set, is called once per frame. The CLI's status line uses it. Other
    callers leave it unset.

    ms reads the mid/side mask that channel 0 stores ahead of each frame. The file
    has one decision per four bands, and the mask comes back with that decision
    repeated per band. The spectra come back still rotated. The caller unmixes
    the pair. With ms clear the return is the coefficient array alone, which is
    what every older caller expects.
    """
    if pvq and not sr:
        raise ValueError("a pulse field needs the sample rate from the header")
    n_coef = n2 // 2
    nb = st.band_grid(n_coef, width)
    r = st.Reader(payload)
    reg = st.shape_reg(sr, n2, width) if shape else None
    X = np.zeros((nfr, n_coef))
    prev_E = np.zeros(nb, np.int64)
    prev_b = np.zeros(nb, np.int64)
    masks = [] if ms else None
    for f in range(nfr):
        if ms:
            masks.append(st.read_ms(r, nb))
        if fill or shape or pvq:
            got = st.read_frame(r, prev_E, prev_b, width, fill=fill, shape=shape, reg=reg,
                                pvq=pvq, sr=sr or 0, n2=n2, sblock=sblock)
            if pvq:
                E, b, mant, codes, words, pulses = got
            elif shape:
                E, b, mant, codes, words = got
                pulses = None
            else:
                E, b, mant, codes = got
                words = None
                pulses = None
            X[f] = st.reconstruct_bands(E, b, mant, codes, f, width, shape=words,
                                        pulses=pulses).reshape(-1)
        else:
            E, b, mant = st.read_frame(r, prev_E, prev_b, width)
            X[f] = st.reconstruct_bands(E, b, mant).reshape(-1)
        prev_E, prev_b = E, b
        if progress is not None:
            progress()
    if ms:
        return X, np.stack(masks)
    return X


def main():
    ap = argparse.ArgumentParser(description="RSC3 decoder")
    ap.add_argument("input", help="input .rsc3")
    ap.add_argument("-o", "--output", default="decoded.wav")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        blob = f.read()
    h = st.unpack_header(blob)
    n2 = h["n2"]
    hop = n2 // 2
    n_samp = h["n_samples"]
    nfr = st.frame_count(n_samp, n2)
    off = h["body_off"]
    chans = []
    for L in h["lens"]:
        chans.append(blob[off:off + L])
        off += L
    if len(chans) != h["nch"]:
        raise ValueError(f"header says {h['nch']} channels, found {len(chans)} payloads")
    ms = bool(h["flags"] & st.MS_FLAG)
    sblock = st.shape_block(h["flags"])
    if ms and h["nch"] != 2:
        raise ValueError(f"mid/side flag is set and the file has {h['nch']} channels")

    print(f"RSC3 decode   {args.input}", flush=True)
    print(f"   {n_samp} samples  {nfr} frames  {h['sr']} Hz  {h['nch']}ch   "
          f"payload {sum(h['lens']):,} B   budget {h['budget']} b/f", flush=True)
    # One step per frame, plus one per channel for the overlap-add, so the line is still
    # short of 100% while the samples are being rendered.
    bar = prog.Bar(nfr * h["nch"] + h["nch"], "decode")
    try:
        frames = []
        mask = None
        for i, p in enumerate(chans):
            if h["nch"] > 1:
                bar.set_label(f"decode  ch {i + 1}/{h['nch']}")
            got = decode_channel(p, nfr, n2, h["width"],
                                 fill=bool(h["flags"] & st.FILL_FLAG),
                                 shape=bool(h["flags"] & st.SHAPE_FLAG), sr=h["sr"],
                                 pvq=bool(h["flags"] & st.PVQ_FLAG),
                                 progress=bar.step, ms=ms and i == 0, sblock=sblock)
            if ms and i == 0:
                X, mask = got
                frames.append(X)
            else:
                frames.append(got)
        if ms:
            frames[0], frames[1] = st.ms_unrotate(frames[0], frames[1], mask, h["width"])
        # amplitude() is X * 4/N, so the inverse is X * N/4. N/4, not N/2: getting this wrong is
        # a silent 6 dB and the round trip is what catches it. The header gain is on top of that,
        # and only when the synthesis itself went past the ceiling.
        g = h["gain"]
        bar.set_label("decode  rendering")
        sig = []
        for f in frames:
            sig.append(r3.synthesise(f * (n2 / 4.0), n2, n_samp) * g)
            bar.step()
        pcm = np.clip(np.round(np.stack(sig, axis=1) * 32767.0), -32768.0, 32767.0)
        bar.set_label("decode  writing")
        with wave.open(args.output, "wb") as wf:
            wf.setnchannels(h["nch"])
            wf.setsampwidth(2)
            wf.setframerate(h["sr"])
            wf.writeframes(pcm.astype("<i2").tobytes())
        bar.close()
    finally:
        bar.close()
    # the fill is a property of the file, not of the person decoding it, so say which one was
    # read. A stream that lied about this would decode to silence or to noise in the wrong bands.
    print(f"   dead-band fill "
          f"{'on, mask-safe levels' if h['flags'] & st.FILL_FLAG else 'off'}"
          + (f", signal-shaped noise above {st.SHAPE_F0 / 1e3:.0f} kHz"
             if h['flags'] & st.SHAPE_FLAG else "")
          + (", pyramid vectors on every live band"
             if h['flags'] & st.PVQ_FLAG else "")
          + (", mid/side on the marked bands"
             if h['flags'] & st.MS_FLAG else "")
          + (f", shape words span {sblock} bands"
             if h['flags'] & st.SHAPE_FLAG and sblock != st.SHAPE_BLOCK else ""))
    if g < 1.0:
        print(f"   playback gain {20 * np.log10(g):+.2f} dB")
    print(f"   wrote {args.output}  ({os.path.getsize(args.output):,} B)")


if __name__ == "__main__":
    main()
