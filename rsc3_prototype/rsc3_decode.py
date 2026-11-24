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
import rsc3_stream as st


def decode_channel(payload, nfr, n2, width, fill=False):
    """
    Rebuild the amplitude-domain coefficients using only what the stream says. The step comes
    from E and b in the file, and the amplitudes come out of reconstruct_bands, which is the
    encoder's own formula.

    fill comes from the header flag, not from guessing. Everything else the decoder needs for
    the noise - which bands are dead, and what values the noise takes - it can derive, and that
    is precisely why the fill is affordable. What it cannot derive is how loud, so that is the
    one thing the file has to say.
    """
    n_coef = n2 // 2
    nb = st.band_grid(n_coef, width)
    r = st.Reader(payload)
    X = np.zeros((nfr, n_coef))
    prev_E = np.zeros(nb, np.int64)
    prev_b = np.zeros(nb, np.int64)
    for f in range(nfr):
        if fill:
            E, b, mant, codes = st.read_frame(r, prev_E, prev_b, width, fill=True)
            X[f] = st.reconstruct_bands(E, b, mant, codes, f, width).reshape(-1)
        else:
            E, b, mant = st.read_frame(r, prev_E, prev_b, width)
            X[f] = st.reconstruct_bands(E, b, mant).reshape(-1)
        prev_E, prev_b = E, b
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

    frames = [decode_channel(p, nfr, n2, h["width"], fill=bool(h["flags"] & st.FILL_FLAG))
              for p in chans]
    # amplitude() is X * 4/N, so the inverse is X * N/4. N/4, not N/2: getting this wrong is
    # a silent 6 dB and the round trip is what catches it.
    sig = [r3.synthesise(f * (n2 / 4.0), n2, n_samp) for f in frames]
    pcm = np.clip(np.round(np.stack(sig, axis=1) * 32767.0), -32768.0, 32767.0)
    with wave.open(args.output, "wb") as wf:
        wf.setnchannels(h["nch"])
        wf.setsampwidth(2)
        wf.setframerate(h["sr"])
        wf.writeframes(pcm.astype("<i2").tobytes())

    print(f"RSC3 decode   {args.input}")
    print(f"   {n_samp} samples  {nfr} frames  {h['sr']} Hz  {h['nch']}ch   "
          f"payload {sum(h['lens']):,} B   budget {h['budget']} b/f")
    # the fill is a property of the file, not of the person decoding it, so say which one was
    # read. A stream that lied about this would decode to silence or to noise in the wrong bands.
    print(f"   dead-band fill {'on, mask-safe levels' if h['flags'] & st.FILL_FLAG else 'off'}")
    print(f"   wrote {args.output}  ({os.path.getsize(args.output):,} B)")


if __name__ == "__main__":
    main()
