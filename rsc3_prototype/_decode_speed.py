"""
Decode is the side that ships, so it is the side that has to be fast.

The encoder is allowed to be slow: nobody runs it on a user's machine. The decoder runs inside
Roblox, per frame, and nothing measured so far says what it costs. This does, on a real file,
split into the parts that would have to be ported to Luau, because the answer decides which
conversation to have next:

  read_frame    - per-field bit extraction; becomes Luau bit-twiddling, and it is the part
                  whose shape changed when side info stopped being dense;
  reconstruct   - mantissa x step, per band, which the decoder cannot skip;
  synthesise    - the MDCT and overlap-add, already proved exact by transform_check.py.

If read_frame dominates, the port is cheap and mostly arithmetic. If synthesise dominates, the
port is a transform port and that is a different, larger conversation.
"""
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
sys.path.insert(0, HERE)
import rsc3_codec as r3
import rsc3_stream as st
import rsc3_encode as enc

N2 = 4096
W = r3.BAND_WIDTH

wf = wave.open(os.path.join(LAB, "cam15.wav"), "rb")
sr = wf.getframerate()
x = (np.frombuffer(wf.readframes(int(15 * sr)), np.int16).astype(np.float64)
     / 32768.0).reshape(-1, 2)
A = r3.amplitude(r3.analyse(x[:, 0], N2)[0], N2)
nfr, nc = A.shape
nb = st.band_grid(nc, W)

# a real payload, so the decode does real work rather than the empty-frame case
payload = b""
for b in (384, 16):
    payload, _ = enc.encode_channel(A, sr, b, psycho=True)
    X = np.zeros((nfr, nc))
    # stage 1: the grammar
    t = time.perf_counter()
    rd = st.Reader(payload)
    pe = np.zeros(nb, np.int64)
    pb = np.zeros(nb, np.int64)
    frames = []
    for _ in range(nfr):
        E, b_, m = st.read_frame(rd, pe, pb, W)
        frames.append((E, b_, m))
        pe, pb = E, b_
    t_read = time.perf_counter() - t

    # stage 2: the reconstruction the decoder owes
    t = time.perf_counter()
    for f in range(nfr):
        X[f] = st.reconstruct_bands(*frames[f]).reshape(-1)
    t_recon = time.perf_counter() - t

    # stage 3: synthesis, the part that is already exact
    t = time.perf_counter()
    y = r3.synthesise(X * (N2 / 4.0), N2, x.shape[0])
    t_synth = time.perf_counter() - t

    print(f"\n{len(payload):,} B payload, {nfr} frames, {nb} bands")
    print(f"  read_frame   {t_read:7.3f} s  {t_read / nfr * 1e3:6.3f} ms/frame   "
          f"{t_read / max(t_read + t_recon + t_synth, 1e-9) * 100:3.0f}%")
    print(f"  reconstruct  {t_recon:7.3f} s  {t_recon / nfr * 1e3:6.3f} ms/frame   "
          f"{t_recon / max(t_read + t_recon + t_synth, 1e-9) * 100:3.0f}%")
    print(f"  synthesise   {t_synth:7.3f} s  {t_synth / nfr * 1e3:6.3f} ms/frame   "
          f"{t_synth / max(t_read + t_recon + t_synth, 1e-9) * 100:3.0f}%")
    print(f"  full decode  {t_read + t_recon + t_synth:7.3f} s   "
          f"{15.0 / (t_read + t_recon + t_synth):5.1f}x realtime   "
          f"{(t_read + t_recon + t_synth) / 15 / x.shape[1] * 1000:.2f} ms per audio-second/ch")
    assert float(np.sqrt(np.mean(y ** 2))) > 0.01, "decoder produced silence"
