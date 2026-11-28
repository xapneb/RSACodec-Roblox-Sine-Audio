"""
perf_check.py - does this change make it faster, and does it still make the same file?

Both claims are measured in one process, so import cost is not mistaken for speed:

  1. correctness. Each configuration is encoded, and the payload is compared against a reference,
     then decoded, and the samples are compared against a reference. A change that moves one bit is
     not a speed-up, it is a new codec, and the only way to tell is to look.
  2. cost. Encode and decode per configuration, in seconds.

The references are not blobs and they are not machine-local, because both of those lost an argument
with a reboot. What is kept is `perf_manifest.json`, next to this file, in the repository: per
configuration, a hash of the payload, its size, and a hash of the decoded samples and their frame
count. It is small enough to commit and honest enough to review, and a bitstream change shows up as
a changed line in it, which is the case where you want the diff to say so. `--bless` rewrites it
from the code in front of you, so bless on a tree you believe rather than on the one you are
midway through changing.

The bench fixture is not a file anyone has to place by hand either: `bench20.wav` is cut from
`moz.wav` into `.perf_cache/`, which is ignored audio like every other wav, so a fresh clone runs
this with nothing installed and a reboot costs a first run and nothing else.

  python perf_check.py            # check, and print timings
  python perf_check.py --bless    # (re)make the manifest from this code
"""
import contextlib
import hashlib
import io
import json
import os
import runpy
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, ".perf_cache")
MANIFEST = os.path.join(HERE, "perf_manifest.json")

# The audio the runs read, other than the files that are already in the checkout. A derived fixture
# names the file it came from and how much of it, so the provenance is in the list rather than in
# somebody's memory. 20 seconds of moz.wav is the bench material: stereo, 48 kHz, loud enough for
# the output ceiling to be part of the answer, and long enough for the rate solve to mean something.
DERIVED = {"bench20.wav": ("moz.wav", 20.0)}

# Both mantissa grammars, both shape-word widths, a stereo and a mono file, and the two rate
# regimes where the fields change shape. The list is the coverage; dropping one is a hole.
RUNS = (
    ("bench20.wav", ("--bitrate", "24", "--pvq", "--fill", "--ms"), "b24pvq"),
    ("bench20.wav", ("--bitrate", "24", "--fill"), "b24sc"),
    ("bench20.wav", ("--bitrate", "8", "--pvq", "--fill", "--ms"), "b08pvq"),
    ("bench20.wav", ("--bitrate", "64", "--pvq", "--fill", "--ms"), "b64pvq"),
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


def _sha(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(nbytes), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _samples(path):
    """Hash over samples, not bytes, so a file that differs only in how its wav header is written
    is still caught as the same sound, and one that differs in a single sample is not."""
    w = wave.open(path, "rb")
    try:
        raw = w.readframes(w.getnframes())
        return hashlib.sha256(raw).hexdigest()[:16], w.getnframes()
    finally:
        w.close()


def fixture(name):
    """Where a run reads from, cutting the derived files first. Deterministic: the same source and
    the same length always produce the same bytes, so the hashes below mean the same thing on
    every machine that runs this."""
    if name not in DERIVED:
        return name
    out = os.path.join(CACHE, name)
    if os.path.exists(out):
        return out
    src, secs = DERIVED[name]
    src = src if os.path.isabs(src) else os.path.join(HERE, src)
    if not os.path.exists(src):
        raise SystemExit(f"{name} is cut from {src}, which is not in this checkout. The runs that "
                         f"need it cannot be checked until that is fixed.")
    os.makedirs(CACHE, exist_ok=True)
    r = wave.open(src, "rb")
    try:
        n = min(int(secs * r.getframerate()), r.getnframes())
        raw = r.readframes(n)
        params = (r.getnchannels(), r.getsampwidth(), r.getframerate())
    finally:
        r.close()
    w = wave.open(out, "wb")
    w.setnchannels(params[0])
    w.setsampwidth(params[1])
    w.setframerate(params[2])
    w.writeframes(raw)
    w.close()
    return out


def main():
    bless = "--bless" in sys.argv
    man = {}
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))
    elif not bless:
        print("   no perf_manifest.json yet: run perf_check.py --bless on a tree you believe is "
              "right, and every later run will be checked against it")
        sys.exit(1)
    bad, missing = [], []
    print(f"\n  {'config':9s} {'encode':>7s} {'decode':>7s} {'payload':>10s}   check")
    te = td = 0.0
    for src, args, tag in RUNS:
        src = fixture(src)
        cur_p, cur_w = os.path.join(CACHE, f"new_{tag}.rsc3"), os.path.join(CACHE, f"new_{tag}.wav")
        et = encode(src, args, cur_p)
        size = os.path.getsize(cur_p)
        dt = decode(cur_p, cur_w)
        te, td = te + et, td + dt
        got = {"payload": _sha(cur_p), "bytes": size}
        got["samples"], got["frames"] = _samples(cur_w)
        if bless:
            man[tag] = dict(got, src=os.path.relpath(src, HERE), args=list(args))
            note = "blessed"
        else:
            ref = man.get(tag)
            if ref is None:
                missing.append(tag)
                note = "*** no reference: --bless ***"
            else:
                differ = [k for k in ("payload", "bytes", "samples", "frames")
                          if ref.get(k) != got[k]]
                if differ:
                    bad.append((tag, differ))
                note = ("payload and samples identical" if not differ
                        else "*** " + " and ".join(differ) + " DIFFER ***")
        print(f"  {tag:9s} {et:7.2f} {dt:7.2f} {size:10,}   {note}")
    if bless:
        json.dump(man, open(MANIFEST, "w"), indent=1, sort_keys=True)
        print(f"\n  references written to {os.path.relpath(MANIFEST, HERE)}. "
              f"run again with no arguments to check.")
        return
    print(f"\n  totals   encode {te:.2f} s   decode {td:.2f} s")
    if bad:
        print("  MISMATCHED: " + ", ".join(f"{t} ({' '.join(d)})" for t, d in bad))
    if missing:
        print("  NO REFERENCE FOR: " + ", ".join(missing))
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
