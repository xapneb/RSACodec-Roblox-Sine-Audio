"""
test_roundtrip.py - the two CLIs must disagree with each other by nothing at all.

The ES an encoder prints is only worth quoting if the decoder's output is the same signal.
Every earlier failure in this format was invisible to a test that looked at ES alone: the
decode crash showed up as audio quality, and the wrong delta reading showed up as a bigger
number, not an error. So this asserts two things:

  1. the samples the decoder writes are the samples the encoder's own reconstruction produced
     (to float rounding, not to a tolerance - if this is not ~1e-15 the decoder is doing its
     own arithmetic and the format is a lie);
  2. the decoder's ES against the original matches the encoder's reported ES. That is the
     number that goes in the README, and it means nothing until the other side proves it.

Both a comfortable and a desperate bitrate run here, because the desperate one is where the
grammar is most stressed and where a wrong field width survives longest.

Run:  python test_roundtrip.py
"""
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
PY = sys.executable
sys.path.insert(0, HERE)
import rsc3_codec as r3

FAILS = []


def run(*a):
    p = subprocess.run([PY, "-u", "-B", *a], cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + p.stderr)[-2000:])
    return p.stdout


def read_wav(p):
    w = wave.open(p, "rb")
    sr = w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    w.close()
    return raw.reshape(-1, 2), sr


def es(y, x, n2, sr):
    """Signal-to-error, with the ORIGINAL as the denominator. The encoder's number is
    mean(original^2) / mean(error^2); if you use the decoded signal instead, the number drifts
    whenever the codec discards energy, which is exactly the low-bitrate regime being tested -
    that is how a correct round trip got reported as a 5 dB failure."""
    lo = 3 * n2
    m = min(len(x), len(y)) - n2
    a, b = x[lo:m], y[lo:m]
    return 10 * np.log10(float(np.mean(a ** 2)) / max(float(np.mean((a - b) ** 2)), 1e-300))


def one(src, kbps, n2=4096, extra=(), mode="safe", tag=""):
    tmp = f"_rt_{kbps}{tag}.rsc3"
    wav = f"_rt_{kbps}{tag}.wav"
    out = run(os.path.join(".", "rsc3_encode.py"), os.path.join(LAB, src), "-o", tmp,
              "--bitrate", str(kbps), *extra)
    enc_es = float([t for t in out.split("ES ") if "dB" in t][0].split()[0])
    line = [t for t in out.splitlines() if "kbit/s per channel" in t]
    enc_kbps = float(line[0].split()[0]) if line else 0.0
    run(os.path.join(".", "rsc3_decode.py"), tmp, "-o", wav)
    x, sr = read_wav(os.path.join(LAB, src))
    y, _ = read_wav(wav)
    n = min(len(x), len(y))
    dec_es = es(y[:n], x[:n], n2, sr)

    # The invariant that actually gates shipping: the payload must reconstruct to the encoder's
    # own coefficients, exactly, before any of the file's rounding is considered. If this is
    # not zero the two CLIs do not share a format, whatever either ES number says.
    import rsc3_encode as enc
    import rsc3_decode as dec
    import rsc3_stream as st
    budget = int([t for t in out.splitlines() if "budget" in t][0].split("budget")[1].split()[0])
    # Whether this file carries the noise fill comes off the header, not off a guess. That is
    # the whole point of the flag: a decoder with no encoder beside it can tell, and a test
    # that assumed it would be testing a decoder that only works when someone tells it.
    with open(os.path.join(HERE, tmp), "rb") as f:
        # HEADER_SIZE alone is not enough: unpack_header also reads the per-payload length
        # table that follows it, so this reads a window, not the whole file.
        flags = st.unpack_header(f.read(st.HEADER_SIZE + 64))["flags"]
    fill = bool(flags & st.FILL_FLAG)
    shape = bool(flags & st.SHAPE_FLAG)
    n_samp = x.shape[0]
    enc_stack, q_stack = [], []
    for c in range(x.shape[1]):
        A = r3.amplitude(r3.analyse(x[:, c], n2)[0], n2)
        p, rc = enc.encode_channel(A, sr, budget, psycho=True, fill=fill, mode=mode,
                                   shape=shape)
        y_enc = r3.synthesise(rc * (n2 / 4.0), n2, n_samp)
        X = dec.decode_channel(p, A.shape[0], n2, 16, fill=fill, shape=shape, sr=sr)
        y_dec = r3.synthesise(X * (n2 / 4.0), n2, n_samp)
        if float(np.abs(y_enc - y_dec).max()) != 0.0:
            FAILS.append(f"{src}@{kbps} ch{c}: decoded coefficients differ by "
                         f"{float(np.abs(y_enc - y_dec).max()):.3e}")
        # and the file must be exactly the rounded reconstruction, so that the only difference
        # left is 16-bit, whose cost is then measured below rather than assumed
        q = np.clip(np.round(y_enc * 32767.0), -32768.0, 32767.0) / 32768.0
        if not np.array_equal(np.round(y[:n_samp, c] * 32768.0), np.round(q * 32768.0)):
            FAILS.append(f"{src}@{kbps} ch{c}: wav is not the rounded reconstruction")
        enc_stack.append(y_enc)
        q_stack.append(q)
    # the rounding shift, measured the same pooled way the CLI reports ES. Per-channel numbers
    # are not a bound: two channels can move in opposite directions and pool larger.
    enc_es_mine = es(np.stack(enc_stack, axis=1), x, n2, sr)
    rounded_es = es(np.stack(q_stack, axis=1), x, n2, sr)
    shift = rounded_es - enc_es_mine
    print(f"  {src}  target {kbps} kbit/s{'  ' + tag if tag else ''}   budget {budget} b/f"
          f"   encoded {enc_kbps:6.2f} kbit/s"
          f"   encoder ES {enc_es:5.2f} dB   decoder ES {dec_es:5.2f} dB")
    print(f"     int16 quantisation alone costs {shift:+.3f} dB of that")
    # The decoded file and the rounded reconstruction must be the same sound, to the last bit.
    # This is what "the decoder works" means; ES is only a description of it.
    if abs(dec_es - rounded_es) > 1e-6:
        FAILS.append(f"{src}@{kbps}: decoder ES {dec_es:.4f} disagrees with the rounded "
                     f"reconstruction {rounded_es:.4f}")
    # The encoder's own printed number is displayed to 0.01 dB, so it can only be checked to
    # 0.01 dB. Asking a 2-decimal report to match a full-precision measurement exactly is how
    # a test ends up with a fudge factor instead of a claim.
    if abs(enc_es - enc_es_mine) > 0.011:
        FAILS.append(f"{src}@{kbps}: encoder printed ES {enc_es} but its own signal says "
                     f"{enc_es_mine:.4f}")
    if len(y) < len(x) - n2:
        FAILS.append(f"{src}@{kbps}: decoded {len(y)} < input {len(x)}")
    if float(np.sqrt(np.mean(y[-6 * n2:] ** 2))) < 1e-9:
        FAILS.append(f"{src}@{kbps}: decoded output is all zeros")
    return enc_es


print("rsc3 encode -> decode through the files on disk")
# The lab holds no audio in git - *.wav is ignored on purpose - so the 15 s fixtures this test
# runs on are cut from the long files on demand. A fresh clone gets the same test for free;
# without this it fails on a missing file that was never meant to be committed.
for stem in ("cam", "ij"):
    src15 = os.path.join(LAB, stem + "15.wav")
    if os.path.exists(src15):
        continue
    long = os.path.join(LAB, stem + ".wav")
    if not os.path.exists(long):
        raise FileNotFoundError(f"neither {stem}15.wav nor {stem}.wav exists in {LAB}")
    w = wave.open(long, "rb")
    sr = w.getframerate()
    if w.getnchannels() != 2:
        raise RuntimeError(f"{stem}.wav is {w.getnchannels()}ch; this test is written for the "
                           f"stereo lab files")
    d = w.readframes(int(15 * sr))
    w.close()
    o = wave.open(src15, "wb")
    o.setnchannels(2)
    o.setsampwidth(2)
    o.setframerate(sr)
    o.writeframes(d)
    o.close()
for src, br in (("cam15.wav", 128), ("cam15.wav", 10), ("ij15.wav", 128), ("ij15.wav", 10)):
    try:
        one(src, br)
    except Exception as e:  # a crash is a failure, not a skip
        FAILS.append(f"{src}@{br}: {e}")
        print(f"  {src}  target {br}   *** {str(e)[:200]}")
# The loud fill levels each run by its own energy and sends no slope, so it writes different
# words than the default and shares nothing but the field widths. It gets its own round trip for
# that reason: a bug in how the two modes get written would otherwise only show up as audio.
try:
    one("cam15.wav", 10, extra=("--fill-mode", "mean"), mode="mean", tag="-loudfill")
except Exception as e:
    FAILS.append(f"cam15.wav@10 loudfill: {e}")
    print(f"  cam15.wav  target 10 loudfill   *** {str(e)[:200]}")
# The shape field is a new bit map with a header flag of its own, and its decode path runs
# through read_shape, which nothing else touches. It gets a round trip at the rate where the
# layer is a real share of the frame, on both reference tracks.
for src, br in (("cam15.wav", 10), ("ij15.wav", 10)):
    try:
        one(src, br, extra=("--fill-shape",), tag="-shape")
    except Exception as e:
        FAILS.append(f"{src}@{br} shape: {e}")
        print(f"  {src}  target {br} shape   *** {str(e)[:200]}")

for f in os.listdir(HERE):
    if f.startswith("_rt_"):
        os.remove(os.path.join(HERE, f))

print("\n" + (f"{len(FAILS)} FAILED\n" + "\n".join(FAILS) if FAILS else "round trip is clean"))
sys.exit(1 if FAILS else 0)
