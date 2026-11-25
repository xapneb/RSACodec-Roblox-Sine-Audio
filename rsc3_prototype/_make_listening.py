"""Build the three listening files, so the decision is made with ears and not with a metric.

safe is the default and the defensible one; mean is what the early experiment did and it is the
version that sounds like air, because its level comes from the run and not from the worst bin in
it. no-fill is the reference. Same target bitrate, so this is a comparison of what the bits buy,
not of how much each one costs.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "rsc2_testing_lab")
OUT = os.path.join(os.path.dirname(HERE), "listening")
PY = sys.executable
os.makedirs(OUT, exist_ok=True)

for src, br in (("cam15.wav", 10.0), ("ij15.wav", 10.0), ("cam15.wav", 128.0)):
    for tag, extra in (("nofill", ["--no-fill"]), ("safe", ["--fill-mode", "safe"]),
                       ("loud", ["--fill-mode", "mean"])):
        stem = os.path.splitext(src)[0]
        enc = os.path.join(OUT, f"{stem}_{br:g}_{tag}.rsc3")
        wav = os.path.join(OUT, f"{stem}_{br:g}_{tag}.wav")
        for cmd in ([PY, "-u", "-B", os.path.join(".", "rsc3_encode.py"),
                     os.path.join(LAB, src), "-o", enc, "--bitrate", str(br), *extra],
                    [PY, "-u", "-B", os.path.join(".", "rsc3_decode.py"), enc, "-o", wav]):
            p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
            if p.returncode:
                print((p.stdout + p.stderr)[-800:])
                sys.exit(1)
        print(f"  {stem}  {br} kbit/s  {tag:>7}   {os.path.getsize(enc):,} B   {wav}")
print(f"\nfiles are in {OUT}")
