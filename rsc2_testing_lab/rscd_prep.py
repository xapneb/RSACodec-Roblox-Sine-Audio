#!/usr/bin/env python3
"""RSC2 JSON → audio.player.html: a Desmos graph that sings.

The player drives one oscillator bank inside Desmos. It is the same shape of problem as the
Roblox decoder, so it is solved the same way:

  * One setExpressions() call per animation frame, updating both lists. Creating and destroying
    tone() expressions mid-playback made Desmos recompile the graph and the frame rate slid, so
    the expressions are all pre-baked and only their arguments change.
  * A bin keeps its slot while it is sounding, instead of being re-sorted into a different slot
    by amplitude every frame. Slot churn is what made partials pop in and out of existence.
  * Gain is smoothed with a time constant of half a hop rather than stepped to the new frame's
    value. A step every hop is the same click the codec side fixed with overlap-add.
  * Timing runs off performance.now() and requestAnimationFrame. Ticking on setTimeout with an
    integer millisecond value ran about 1% fast, which drifts, and dropping frames is what made
    playback lurch.
  * tone() duration is exactly one hop, not one hop and a bit. Overlapping tones sum, and the
    sum of two overlapping copies of the same oscillator is a level that wobbles.

The JSON it reads comes from rscd_encoder.py. See NOTES.md for what this build still cannot do.

  uv run python rscd_prep.py audio.rscd
"""

import argparse
import json
from pathlib import Path

import rsc2_codec as codec

MAX_PARTIALS_DEFAULT = 192


def build_frames_js(data: dict, max_partials: int, channel: int = 0):
    """Sparse frame tables. Each frame is a list of [bin, gain] pairs, gain already in units
    of tone amplitude with the global peak folded in.

    Slots are not decided here. The player keeps a bin in whichever slot it already occupies,
    so the ordering of this list carries no meaning.
    """
    sr        = data["sample_rate"]
    fft       = data["fft_size"]
    hop       = data["hop_size"]
    frames_in = data["channels"][channel]["frames"]

    hop_sec     = hop / sr
    bin_hz      = sr / fft
    global_peak = max((abs(p["amp_norm"] * f["peak"]) for f in frames_in for p in f["partials"]),
                      default=1.0)
    if global_peak < 1e-12:
        global_peak = 1.0

    frames = []
    for f in frames_in:
        by_bin = {}
        for p in f["partials"]:
            gain = p["amp_norm"] * f["peak"] / global_peak
            if p["freq_hz"] < 20.0 or p["freq_hz"] > 20000.0 or gain <= 1e-5:
                continue
            if p["bin"] not in by_bin or gain > by_bin[p["bin"]]:
                by_bin[p["bin"]] = gain
        if not by_bin:
            frames.append([])
            continue
        keep = sorted(by_bin.items(), key=lambda kv: kv[1], reverse=True)[:max_partials]
        frames.append([[int(b), round(float(g), 5)] for b, g in keep])

    loud = sum(1 for f in frames if f)
    print(f"   {len(frames)} frames, {loud} with sound, {hop_sec * 1000:.2f} ms per frame, "
          f"{bin_hz:.2f} Hz per bin")

    js = "const RSC2_FRAMES=" + json.dumps(frames, separators=(",", ":")) + ";\n"
    meta = {"sample_rate": sr, "fft_size": fft, "hop_size": hop, "bin_hz": bin_hz,
            "hop_sec": hop_sec, "n_frames": len(frames), "max_partials": max_partials}
    return js, meta


def build_desmos_state(meta: dict):
    """Pre-bake one tone() per slot plus the two lists they read."""
    max_partials = meta["max_partials"]

    exprs, txt = [], []

    def add(id_str, latex, folder=None, color="#00ff88"):
        e = {"type": "expression", "id": id_str, "latex": latex, "color": color}
        if folder:
            e["folderId"] = folder
        exprs.append(e)
        txt.append(f"  {id_str:10s}  {latex}")

    exprs.append({"type": "folder", "id": "fold_data", "title": "🔒 Audio Data",
                  "collapsed": True})
    add("F", "F=[" + ",".join(["20"] * max_partials) + "]", folder="fold_data", color="#4488ff")
    add("G", "G=[" + ",".join(["0"] * max_partials) + "]", folder="fold_data", color="#ff8844")
    txt.append("")

    exprs.append({"type": "folder", "id": "fold_osc",
                  "title": f"🔊 Oscillators ({max_partials} slots)", "collapsed": True})
    for i in range(1, max_partials + 1):
        add(f"o{i}", f"tone(F[{i}],G[{i}])", folder="fold_osc")

    state = {"version": 11,
             "graph": {"viewport": {"xmin": 0, "xmax": 20000, "ymin": -0.05, "ymax": 1.1},
                       "xAxisLabel": "Frequency (Hz)", "yAxisLabel": "Amplitude",
                       "showGrid": True, "squareAxes": False, "polarMode": False,
                       "xAxisNumbers": True, "yAxisNumbers": True},
             "expressions": {"list": exprs}, "randomSeed": "rsc2"}
    return state, "\n".join(txt)


PLAYER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSC2/DESMOS player</title>
<script src="https://www.desmos.com/api/v1.12/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
<style>
    body{background:#0d0d1a;color:#e0e0ff;font-family:monospace;margin:0;display:flex;flex-direction:column;height:100vh}
    #header{padding:10px 16px;background:#080810;border-bottom:1px solid #1a1a3a;display:flex;align-items:center;gap:14px}
    h1{font-size:13px;color:#00ff88;letter-spacing:3px;text-transform:uppercase;margin:0}
    button{background:#0d0d1a;border:1px solid #00ff88;color:#00ff88;padding:5px 12px;cursor:pointer}
    button:hover{background:#00ff88;color:#0d0d1a}
    button:disabled{opacity:.3;cursor:default;border-color:#333}
    #seek{flex:1;height:22px;background:#1a1a3a;position:relative;cursor:pointer}
    #seek:focus-visible{outline:2px solid #00ff88;outline-offset:2px}
    #seek-bar{height:100%;background:#00ff88;width:0%}
    #info{font-size:10px;color:#8899aa;white-space:nowrap}
    #calculator{flex:1}
</style>
</head>
<body>
<div id="header">
    <h1>⬡ RSC2/DESMOS</h1>
    <button id="btn-play">Play</button>
    <button id="btn-stop" disabled>Stop</button>
    <div id="seek" role="slider" aria-label="Seek" aria-valuemin="0" aria-valuemax="__N__"
         aria-valuenow="0" tabindex="0"><div id="seek-bar"></div></div>
    <div id="info">frame 0 / __N__</div>
</div>
<div id="calculator"></div>
<script>
"use strict";
__FRAMES_JS__
const HOP_SEC = __HOP_SEC__;
const BIN_HZ  = __BIN_HZ__;
const N       = RSC2_FRAMES.length;
const SLOTS   = __SLOTS__;
const TAU     = HOP_SEC * 0.5;

const Calc = Desmos.GraphingCalculator(document.getElementById("calculator"),
                                       {expressionList: true, settingsMenu: true,
                                        keypad: false, zoomButtons: true});
Calc.setState(__STATE__);

// A bin keeps the slot it was given. Re-sorting slots by gain every frame is what made
// partials change identity mid-note, so this table is keyed by bin and never reshuffled.
const slotBin = new Array(SLOTS).fill(-1);
const target  = new Float64Array(SLOTS);
const current = new Float64Array(SLOTS);
const seen    = new Uint8Array(SLOTS);

function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

function assignFrame(fi) {
    const frame = RSC2_FRAMES[fi];
    seen.fill(0);
    for (const entry of frame) {
        const bin = entry[0], gain = entry[1];
        let slot = slotBin.indexOf(bin);
        if (slot < 0) {
            for (let s = 0; s < SLOTS; s++) {
                if (slotBin[s] < 0) { slot = s; break; }
            }
        }
        if (slot < 0) {                       // no free slot: take the quietest one
            slot = 0;
            for (let s = 1; s < SLOTS; s++) { if (current[s] < current[slot]) slot = s; }
        }
        slotBin[slot] = bin;
        seen[slot]    = 1;
        target[slot]  = gain;
    }
    for (let s = 0; s < SLOTS; s++) {
        if (!seen[s]) {
            target[s] = 0.0;
            if (current[s] < 1e-4) slotBin[s] = -1;
        }
    }
}

function pushLists() {
    const f = new Array(SLOTS), g = new Array(SLOTS);
    for (let s = 0; s < SLOTS; s++) {
        f[s] = (slotBin[s] > 0 ? slotBin[s] : 1) * BIN_HZ;
        g[s] = clamp(current[s], 0.0, 1.0);
    }
    Calc.setExpressions([{id: "F", latex: "F=[" + f.join(",") + "]"},
                         {id: "G", latex: "G=[" + g.join(",") + "]"}]);
}

let playing = false, startTs = 0, lastTs = 0, lastFrame = -1;

function step(ts) {
    if (!playing) return;
    const dt = Math.min(0.1, (ts - lastTs) / 1000) || 0;
    lastTs = ts;
    const elapsed = (ts - startTs) / 1000;
    const fi = clamp(Math.floor(elapsed / HOP_SEC), 0, N - 1);
    if (fi !== lastFrame) { assignFrame(fi); lastFrame = fi; }

    const k = 1.0 - Math.exp(-dt / TAU);          // ramp toward the new frame, no steps
    for (let s = 0; s < SLOTS; s++) current[s] += (target[s] - current[s]) * k;
    pushLists();

    document.getElementById("info").textContent = "frame " + (fi + 1) + " / " + N;
    document.getElementById("seek-bar").style.width = ((fi + 1) / N * 100) + "%";
    document.getElementById("seek").setAttribute("aria-valuenow", String(fi + 1));
    if (elapsed >= N * HOP_SEC) { stop(); return; }
    requestAnimationFrame(step);
}

function start() {
    if (playing) return;
    playing = true; lastFrame = -1;
    startTs = performance.now(); lastTs = startTs;
    document.getElementById("btn-play").disabled = true;
    document.getElementById("btn-stop").disabled = false;
    requestAnimationFrame(step);
}

function stop() {
    playing = false;
    for (let s = 0; s < SLOTS; s++) { target[s] = 0.0; current[s] = 0.0; slotBin[s] = -1; }
    pushLists();
    document.getElementById("btn-play").disabled = false;
    document.getElementById("btn-stop").disabled = true;
}

function jumpTo(seconds) {
    startTs = performance.now() - clamp(seconds, 0, N * HOP_SEC);
    lastFrame = -1;
}

function seekFromEvent(e) {
    const r = document.getElementById("seek").getBoundingClientRect();
    const frac = clamp((e.clientX - r.left) / r.width, 0, 1);
    if (playing) { jumpTo(frac * N * HOP_SEC); return; }
    const fi = clamp(Math.floor(frac * N), 0, N - 1);
    assignFrame(fi);
    for (let s = 0; s < SLOTS; s++) current[s] = target[s];
    pushLists();
    document.getElementById("seek-bar").style.width = ((fi + 1) / N * 100) + "%";
}

document.getElementById("btn-play").addEventListener("click", start);
document.getElementById("btn-stop").addEventListener("click", stop);
document.getElementById("seek").addEventListener("click", seekFromEvent);
document.getElementById("seek").addEventListener("keydown", function (e) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const now = playing ? (performance.now() - startTs) / 1000 : 0;
    jumpTo(now + (e.key === "ArrowRight" ? 5 : -5));
});

pushLists();
</script>
</body>
</html>
"""


def build_player_html(js_data: str, state: dict, meta: dict) -> str:
    return (PLAYER_HTML
            .replace("__FRAMES_JS__", js_data)
            .replace("__STATE__", json.dumps(state))
            .replace("__HOP_SEC__", repr(round(meta["hop_sec"], 9)))
            .replace("__BIN_HZ__", repr(round(meta["bin_hz"], 9)))
            .replace("__SLOTS__", str(meta["max_partials"]))
            .replace("__N__", str(meta["n_frames"])))


def main():
    ap = argparse.ArgumentParser(description="RSC2 JSON → player.html + Desmos state")
    ap.add_argument("input", help="JSON from rscd_encoder.py")
    ap.add_argument("--output", help="base path, default is the input's stem")
    ap.add_argument("--max-partials", type=int, default=MAX_PARTIALS_DEFAULT,
                    help="oscillator slots to pre-bake (default 192)")
    ap.add_argument("--channel", type=int, default=0)
    a = ap.parse_args()

    src  = Path(a.input)
    base = Path(a.output) if a.output else src.with_suffix("")
    print(f"📂 {src}")
    data = json.loads(src.read_text())
    print(f"   {data['sample_rate']} Hz | {data['n_channels']} ch | {data['n_frames']} frames | "
          f"{data['duration_sec']:.2f} s")

    frames_js, meta = build_frames_js(data, a.max_partials, a.channel)
    state, txt      = build_desmos_state(meta)
    html            = build_player_html(frames_js, state, meta)

    player = base.with_suffix(".player.html")
    desmos = base.with_suffix(".desmos.json")
    exprs  = base.with_suffix(".exprs.txt")
    player.write_text(html)
    desmos.write_text(json.dumps(state))
    exprs.write_text("RSC2 → Desmos expressions\n" + "=" * 60 + "\n\n" + txt + "\n")

    print(f"\n🌐 {player}  ({player.stat().st_size / 1024:.0f} KB)  ← open this")
    print(f"📊 {desmos}  ({desmos.stat().st_size / 1024:.1f} KB)")
    print(f"📝 {exprs}\n")
    print("   Click Play, then unmute the graph. Click the bar to seek, arrows to nudge.\n")


if __name__ == "__main__":
    main()
