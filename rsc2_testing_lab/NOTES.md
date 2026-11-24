# RSC2 v2: what changed, and how to check it yourself

## The bug

Frame-to-frame reconstruction had no overlap and no crossfade, so every frame boundary was a
level step. At 48 kHz with hop = 2048 that is a click 23 times a second, on both output paths.

Six separate things made it worse than it needed to be:

1. `hop_size == fft_size`, so frames butted end to end and any disagreement between two frames
   landed as a one-sample step (`rsc_encoder.py`, `rsc_decoder.py`).
2. Rectangular window, which is exactly the wrong choice for butt-joined frames: the whole frame
   is weighted equally, so the join is hard.
3. rfft magnitudes are half a tone, and the decoder divided by `window_sum` only, so the output
   sat 6 dB quiet everywhere except where a frame chose DC or Nyquist, which got 6 dB too hot.
   That is a step in the output, twice per frame boundary if you are unlucky
   (`rsc_decoder.py:252` in the old file).
4. The synthesiser advanced phase *before* emitting the first sample of each partial, so every
   tone was one sample late and Nyquist was inverted (`rsc_decoder.py:96-102`).
5. The peak-picking score rewrote `score[b-1]` and `score[b+1]` inside the same loop that read
   them, and read `score[b+1]` past the end of the array. Selection flickered between
   neighbouring bins, so partials appeared and vanished between frames (`rsc_encoder.py:59-62`).
6. The Roblox player wrote `Volume` with no ramp, and wrote `TimePosition` on every frame, which
   teleports the oscillator mid-sample (`rsc_decoder.lua:158-236`).

## What v2 does

- Periodic Hann window, `hop_size = fft_size / 2`. Hann shifted by half its length sums to
  exactly 1.0, so overlap-add needs no correction term, and two frames describing the same
  partial disagree by exactly the window shift and add back into one clean tone.
  Anything whose `hop` does not equal `fft/2` gets the overlap sum measured and divided out, so
  `hop = fft/4` also reconstructs (at 2.0) if you want finer control resolution.
- Amplitude comes out as `2 * magnitude / window_sum`, with 1.0 instead of 2.0 at DC and Nyquist,
  computed by `rsc2_codec.amplitude_scale` and used by every decoder.
- Phase is advanced after the emitted sample, not before.
- Salience scoring is batched, never mutates what it reads, and never selects DC.
- The Roblox bank keeps a bin in its slot, ramps gain with a time constant of half a hop, and
  only writes `TimePosition` where the gain is zero.
- Frame payload format is unchanged, byte for byte, so the shipped `.rbxl` still parses.
  Version is stamped 2 and decoders accept 1 and 2.

## Checking it

    uv run python measure_clicks.py                     # bundled synthetic material
    uv run python measure_clicks.py --impl-dir /tmp/v1  # measure the old build for comparison
    luajit /tmp/drive_lua.lua audio.rsc2 rsc_decoder.lua

`measure_clicks.py` band-limits a synthetic signal so nothing above the cutoff can be real, then
counts spikes there. Numbers from the 6 s test signal:

| build | level | error-to-signal | clicks per second |
|---|---|---|---|
| v1, hop = fft | 0.43 | +3.5 dB | 1104 |
| v2, hop = fft (all other fixes) | 0.94 | +4.1 dB | 1599 |
| v2, hop = fft/2 | 0.89 | 12.8 dB | under the measurement floor |
| v2, hop = fft/4 | 0.88 | 14.7 dB | under the measurement floor |

The third row is the point: with the same gain, window and phase fixes, overlap-add alone is what
removes the clicks. Row two says the fixes on their own are not enough.

A dead-centre-on-bin tone round-trips at level 1.000000 with 82.8 dB error-to-signal, so the
amplitude and phase conventions are exact. Real material reads lower because 24 partials out of
1025 bins is a lossy model, by design.

`ab_test/audio_before_clicks.wav` and `ab_test/audio_after_clicks.wav` are the same 10 s excerpt
through both builds. `.wav` is gitignored, so treat them as scratch.

## Memory

Analysis used to batch every frame of the file at once, and salience scoring keeps about ten
float64 temporaries alive. On the 230 s stereo source that is roughly 800 MB and it will OOM.
Frames now go through in blocks of `ANALYSIS_BLOCK` (256), which keeps the whole encode under
about 250 MB, mostly the decoded source, and makes it faster too.

- `--analysis-block N` on either encoder. Lower it if you are tight.
- PCM is held as float32 end to end; the accumulator is float64, the output is float32.

## Known limits, not bugs

- **Frequencies are bin-quantised.** `freq = bin * sample_rate / fft_size`, so at 48 kHz a tone
  can be up to 11.7 Hz sharp. That is a tuning problem, not a click. Fixing it means storing a
  measured frequency per partial, which is a new frame format.
- **24 partials per frame.** Everything the codec cannot name is simply not reproduced, which is
  the real reason the output is not the input. Raise `--max-partials`; the Roblox bank scales to
  whatever the header says, up to 192.
- **v1 files still click at their joins.** The new decoders read them correctly and at the right
  level, but a v1 file has no overlap to reconstruct from. Re-encode to get rid of it.
- **The Roblox path cannot do overlap-add.** One looped `Sound` per partial, and you cannot
  window a `Sound`. It gets continuous oscillators plus a gain ramp, which is what stops the
  clicks. Phase drifts a little and nobody can hear that.
- **`_NO_TURNING_BACK.mp3` and the mp3 warning in the README** are still true. Opus-encoded input
  makes this output unpredictable; use mp3, vorbis or aac before encoding.

## AudioStreamWriter: the newer, better playback path

Engine 747 (released 2026-11-09) added `AudioStreamWriter`, which lets Luau write Float32 samples
into a ring buffer that the audio engine drains at 48 kHz. `rsc_decoder.lua` now uses it when it is
available and falls back to the Sound bank otherwise. This is strictly better than the bank, because
the decoder owns the waveform: each partial is a real oscillator at the exact bin frequency, phase
accumulates and is never reset, and no Sound parameter gets written anywhere.

From the reference pages (create.roblox.com/docs/reference/engine/classes/AudioStreamWriter):
SampleRate is 48000 and always read-only, SampleFormat is Float32 and always read-only, `Layout`
sets how many buffers `Write()` takes (Stereo wants two), `Space()` reports free frames, `Write()`
returns frames accepted, there is one Output pin and no inputs, and it must be wired to an
`AudioDeviceOutput` or nothing is heard.

**The caveat you need to know before building on this**: the same page states it can only be used
in Roblox Studio and cannot be used in Team Create sessions, and one reply in the release-notes
thread says it is still in pending updates. So treat the stream path as the Studio-quality renderer
and keep the bank engine for what you publish until that changes. The decoder picks it over the
bank by itself, and if `Write` or `Space` ever throws it drops to the bank and says so.

### How the stream engine is checked without Roblox

`/tmp/drive_stream.lua` is an off-line harness: it stubs `AudioStreamWriter`, `buffer`, `Instance`
and the DataModel, feeds the real decoder file, advances a virtual clock, and dumps the samples it
would have written. `stream_engine_check.py` runs the same algorithm in numpy and compares:

```
luajit /tmp/drive_stream.lua audio.rsc2 "$PWD/rsc_decoder.lua"   # 6 s capture
uv run python stream_engine_check.py /tmp/lua_stream.f32 audio.rsc2
SECONDS=6 luajit /tmp/drive_stream.lua /tmp/click.rsc2 "$PWD/rsc_decoder.lua"   # click material
```

What that found, against the band-limited material `measure_clicks.py` uses:

| path | level | residual above 7 kHz | spikes on frame joins |
|---|---|---|---|
| reference (the material itself) | 1.000 | -85.9 dB | - |
| v1, hop = fft | 0.912 | -49.5 dB | 267, 1598 per second |
| python overlap-add | 0.832 | -140.4 dB | 0 |
| **stream engine** | 0.886 | **-118.9 dB** | **0** |

Single-partial and few-partial tones agree between the Lua and numpy engines to between 75 dB and
148 dB error-to-signal, which is the wavetable's linear-interpolation floor and nothing else, so the
oscillator, the phase accumulation and the gain ramp are all implemented as documented. On dense
real material the two engines land 6 to 10 dB apart while agreeing on every bin and every amplitude
to within 1e-5, which puts the difference in the exact sample where a fading partial is retired and
re-seeded. It is a modelling detail with nothing audible in it, but it is why the numpy mirror is
called a model of the engine rather than a proof of it.

Two things to watch in that engine:

- **It hard-clips.** 24 partials summing can pass full scale, and on the 6 s stereo file that
  happens on 0.05% of samples. Fewer partials is the fix, since the model's own ceiling is what
  pushes the sum up.
- **Cost.** Roughly partials x 48000 oscillator steps per second per channel. This file runs about
  24 per frame per channel, so around 2.3M steps a second, and a table lookup instead of `math.sin`
  is what makes that affordable in Luau.

## v3: packed frame layout

v2 spent 6 bytes per partial on three words: bin, amplitude, phase. That is mostly wasted. The
bin index only ever needs `ceil(log2(fft/2))` bits and the amplitude is relative to the frame peak,
so v3 keeps what the format actually needs and drops what it does not:

- A frame header of peak (f32) and n_partials (u16), unchanged from v2.
- Per partial, **five bytes** instead of six: a 24-bit word packing `bin` in the low bits and the
  amplitude above it, then the phase as a u16. Silent slots are the all-zero word, which replaces
  v1/v2's "point a dead slot at bin 1" trick.

The phase keeps all sixteen bits and the same `(-pi, pi]` grid v2 used, and that is deliberate: the
fidelity budget belongs to phase, not size. A shorter phase field was tried and measured
**-44 dB** of added noise on real audio, because overlap-add rebuilds a partial from two frames
whose phases have to line up. Sixteen bits lands at -87 dB, effectively the v2 floor.

Amplitude, being relative, shrinks for free. At the default `fft_size` the 24-bit word leaves 13
or 14 bits for amplitude, around -78 dB — under anything the analysis itself contributes. A
larger `fft_size` shifts bits from amplitude to bin index; `pack_bits` refuses to pack below
12 amplitude bits so it can never quietly degrade.

Measured on the lab `audio.wav`:

| config | v2 | v3 | ratio |
|---|---|---|---|
| `fft=1024 hop=512 max_partials=192` | 49.98 MB | 41.69 MB | 0.83 |
| `fft=2048 hop=1024 max_partials=24` | 3.24 MB | 2.72 MB | 0.84 |

Decode differences are the amplitude grid and nothing else: worst sample delta about 1e-4, the
numpy model and the engines agree to well under a tenth of a percent. v3 decoders read v2 files
and v2 decoders read v3 files, picked from the header version byte, so old `.rsc2` still plays.

## Psychoacoustic partial selection

v2 selection ranked bins purely by spectral salience: peakiness times magnitude. That is blind in
two ways. It spends slots on partials that sit below the absolute threshold of hearing, and it
cannot see that a partial 3 dB below a louder neighbour in the same critical band is not
audible at all. The psychoacoustic layer replaces "which bins look like peaks" with "which bins
are worth an ear", and is the default now; `--no-psy` brings back the old selection byte for byte.

The model has three parts, all on the ERB scale (Glasberg-Moore), because the Bark approximations
go wrong in the low- and high-frequency corners:

- **Absolute threshold of hearing.** A rough ISO-226 curve, referenced so 0 dBFS is `spl_ref_db`
  (default 96 dB SPL, i.e. loud 16-bit material). Anything under it is not a partial, it is silence.
- **Simultaneous masking.** Tone maskers spread linearly in ERB-rate, -10 dB/ERB upward and
  -25 dB/ERB downward, the Glasberg-Moore pattern. The thresholds of a whole block of frames are
  one vectorised pair of prefix scans; per-frame masks are one maximum-each update.
- **Temporal hysteresis.** A partial that is playing survives sinking 6 dB below threshold, so
  notes do not flicker across frame joins. This is a click question more than a quality question:
  the previous failure mode was selection appearing and vanishing at every frame boundary.

The selection is greedy and self-consistent, which is the part that actually took the work. The
first attempt computed the mask threshold from every peak in the full spectrum, which is what a
transform codec does when it can mask hundreds of lines. With a 24-slot budget that is fiction:
the maskers themselves did not fit, so partials vanished behind shadows cast by tones that were
never encoded, and ES fell **9 dB**. Now the encoder fills its slots one at a time, loud peaks
first, and each accepted partial is what raises the threshold the rest must clear. A dropped
partial is therefore provably hidden behind something that plays.

Then a second, quieter bug. Hysteresis kept last frame's bin alive, and this frame's salience
pass happily voiced the bin next to it: one note that had drifted a single bin, played twice at
full amplitude, in 682 frames. Loudness of decoded audio went above unity and every band lost
8 dB. The kept note and the new peak are now a handoff, never a duet.

Measured on `audio.wav`, 25 s, fft=2048 hop=1024, one channel:

| config | salience ES | psycho ES | wasted slots (below-mask partials) |
|---|---|---|---|
| `max_partials=24` | 14.1 dB | 12.1 dB | 15.9% dropped, 0.3% audibly so |
| `max_partials=16` | 9.2 dB | 10.7 dB | - |
| `max_partials=12` | 7.6 dB | 9.3 dB | - |
| `max_partials=8`  | 6.1 dB | 7.5 dB | - |

At the full budget the psycho codec spends ~16% fewer partials for what an equal-loudness model
rates as a ~1 dB-worse residual, and at tight budgets it wins outright, which is exactly where
slot pressure hurts: the slots it frees go to things the model cannot shadow. Raw ES is the wrong
unit here — masked energy is worth ~0 dB of hearing, which is the whole idea. On the synthetic
click material it is neutral (12.7 vs 12.8 dB) and still adds zero frame-boundary spikes.

Analysis runs ~6x the per-frame greedy loop (0.16 s → 0.93 s per 25 s), which stays far under the
FFT and packing time on real files. Parameters live in `PSY_PARAMS`: slopes, the SPL reference,
the hysteresis window, and the ranking (`level`, loud-first, is default).

## Output headroom

Sine atoms cannot land on a flat top, so they ring past it. On hard-limited material that ringing
is what you hear as clicking: `audio.wav` decodes to a peak of **+3.1 dBFS** with 0.05% of samples
over full scale, and `cam.wav` (which arrives already welded to the rail, 2,988 runs of 4+ samples
at exactly -1.0) reaches **+5.6 dBFS** and 1.5%. Either value is enough for the client's
`math.clamp` to flatten the top of a waveform, and a flat top that snaps back is a transient.

The overshoot is linear in level, which is what makes an encoder-side gain the right lever: a
6 dB-quieter input overshoots by exactly the same factor. So the encoder now measures it instead
of assuming a constant, because the ratio is as much the material's as the codec's.

```
uv run python rsc_encoder.py input.wav output.rsc2          # two-pass, automatic
uv run python rsc_encoder.py input.wav output.rsc2 --ceiling 0 # one pass, no gain
```

Pass one encodes and decodes it; if the decoded peak is above `--ceiling` (default 1.0) the
second pass multiplies the **frame peak** by `ceiling/peak`. `decode` returns un-clamped float,
which is why that measurement is exact. The gain is folded into the peak at pack time rather
than into the samples before analysis, for one important reason: the psychoacoustic gates read
absolute dBFS, so scaling the input would quietly drop the quiet partials that the threshold
gate would otherwise keep. Applied after selection, the partial list, its amplitudes and its
phases are untouched, and only the frame peak bytes change. Verified: same byte length, 69,640
of the changed bytes are all peak-f32 bytes with zero inside the header, `|y2 - g*y1|` peaks at
1.2e-07 — the f32 rounding grid of the peak value itself — and the re-encoded file decodes to
1.00000 where the first pass decoded to 1.42899, taking samples over full scale from 0.048% to
zero.

`rscd_encoder.py` folds the same factor into its `"peak"` field and takes `--gain`, since the
JSON path has no decoder to measure against. The value is whatever `rsc_encoder.py` measured on
the same file: both encoders share `analyse_channel`, so the ratio carries over.

## Files

- `rsc2_codec.py`: format, window, scoring, selection. One copy now, was three.
- `rsc_encoder.py` / `rsc_decoder.py`: `.rsc2` to and from WAV.
- `rscd_encoder.py` / `rscd_prep.py`: the `.rscd` JSON and the Desmos player.
- `measure_clicks.py`: the regression test above.
- `stream_engine_check.py`: numpy mirror of the stream engine, for checking the Lua off-line.
- `rsc_decoder.lua`: the Roblox client, stream engine when available and the Sound bank otherwise.

## See also

[`../rsc3_prototype/`](../rsc3_prototype/README.md) is the block-transform successor: same
lab, same audio, better measured results at lower rate, but no bitstream and no decoder.
Two things carry over from that work and apply to this one too. Raw ES is not the quality
ruler - score error against the masking threshold of the *original* and it agrees with
listening, and fit a gain before comparing two codecs, or a coherent level offset gets
charged as error and caps the score at whatever that offset allows.
