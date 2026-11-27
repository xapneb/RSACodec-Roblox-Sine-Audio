"""
perf_check.py - does this change make it faster, and does it still make the same file?

Both claims are measured in one process, so import cost is not mistaken for speed:

  1. correctness. Each configuration is encoded and compared byte for byte against a reference
     payload, and the reference payload is decoded and compared sample for sample. A change
     that moves one bit is not a speed-up, it is a new codec, and the only way to tell is to
     look.
  2. cost. Encode and decode per configuration, in seconds.

The references live in /tmp/ref because they are machine-local. --bless makes them from the
current code; run it with no arguments to check against them.

  python perf_check.py            # check, and print timings
  python perf_check.py --bless    # (re)make the references
"""
import contextlib
import io
import os
import runpy
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REF = os.path.join("/tmp", "ref")

# Both mantissa grammars, both shape-word widths, a stereo and a mono file, and the two rate
# regimes where the fields change shape. The list is the coverage; dropping one is a hole.
RUNS = (
    ("/tmp/bench20.wav", ("--bitrate", "24", "--pvq", "--fill", "--ms"), "b24pvq"),
    ("/tmp/bench20.wav", ("--bitrate", "24", "--fill"), "b24sc"),
    ("/tmp/bench20.wav", ("--bitrate", "8", "--pvq", "--fill", "--ms"), "b08pvq"),
    ("/tmp/bench20.wav", ("--bitrate", "64", "--pvq", "--fill", "--ms"), "b64pvq"),
    (os.path.join(HERE, "sine.wav"), ("--bitrate", "24", "--pvq", "--fill", "--ms"), "s24pvq"),
    (os.path.join(HERE, "sine.wav"), ("--bitrate", "10", "--fill", "--shape-block", "2"),
     "s10fine"),
)


def cli(script, argv):
    """Run one of the two CLIs in-process, with argv as sys.argv would have it, and the status
    line swallowed: the point of this file is the seconds, not the progress bar."""
    old = sys.argv
    sys.argv = [script, *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        sys.argv = old


def encode(src, args, out):
    t = time.perf_counter()
    cli("rsc3_encode.py", [src, *args, "-o", out])
    return time.perf_counter() - t


def decode(src, out):
    t = time.perf_counter()
    cli("rsc3_decode.py", [src, "-o", out])
    return time.perf_counter() - t


def wav_equal(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    wa, wb = wave.open(a, "rb"), wave.open(b, "rb")
    try:
        return (wa.getnframes() == wb.getnframes()
                and np.array_equal(np.frombuffer(wa.readframes(wa.getnframes()), np.int16),
                                  np.frombuffer(wb.readframes(wb.getnframes()), np.int16)))
    finally:
        wa.close()
        wb.close()


def main():
    bless = "--bless" in sys.argv
    os.makedirs(REF, exist_ok=True)
    bad = []
    print(f"\n  {'config':9s} {'encode':>7s} {'decode':>7s} {'payload':>10s}   check")
    te = td = 0.0
    for src, args, tag in RUNS:
        ref_p, ref_w = f"{REF}/{tag}.rsc3", f"{REF}/{tag}.wav"
        cur_p, cur_w = f"{REF}/new_{tag}.rsc3", f"{REF}/new_{tag}.wav"
        et = encode(src, args, cur_p)
        size = os.path.getsize(cur_p)
        dt = decode(cur_p, cur_w)
        te, td = te + et, td + dt
        if bless:
            open(ref_p, "wb").write(open(cur_p, "rb").read())
            decode(ref_p, ref_w)
            note = "blessed"
        else:
            same_p = (os.path.exists(ref_p)
                      and open(cur_p, "rb").read() == open(ref_p, "rb").read())
            same_w = os.path.exists(ref_w) and wav_equal(cur_w, ref_w)
            if not (same_p and same_w):
                bad.append(tag)
            note = ("payload and samples identical" if same_p and same_w
                    else "*** " + " and ".join(c for c, ok in (("payload", same_p),
                                                               ("samples", same_w))
                                               if not ok) + " DIFFER ***")
        print(f"  {tag:9s} {et:7.2f} {dt:7.2f} {size:10,}   {note}")
    if bless:
        print("\n  references written. run again with no arguments to check.")
        return
    print(f"\n  totals   encode {te:.2f} s   decode {td:.2f} s")
    print("  " + ("every configuration matches" if not bad else f"MISMATCHED: {bad}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
