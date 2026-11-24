# RSC3 prototype

Block-transform codec: fast MDCT, uniform scalefactor bands, reverse-waterfill bit
allocation against a psychoacoustic threshold model. It is the successor to the RSC2
additive-synthesis line, and it is **not finished**: it encodes, it measures, and it
produces listening files, but it has no bitstream and no decoder.

```
rsc3_codec.py         model core - transform, thresholds, allocation, quantiser
transform_check.py    the transform's own proof: kernel match + full-signal OLA
ab_codec.py           A/B generator, the protocol that produces every listening file
gainfit_es.py         re-scores any file under a fitted gain (see "Rulers")
audible_sweep.py      quality vs budget, scored in audible-error units
hf_audit.py           bits / signal / error / audible-error per frequency band
fill_experiment.py    noise-fill, priced per run of dead bands, at equal bitrate
```

Run anything from this folder: `python ab_codec.py ..\\rsc2_testing_lab\\ij.wav`
(paths resolve relative to the repo, so the scripts work wherever they're checked out).

## How far along it is

Everything below is measured, on this repo's own audio, at 512 bits/frame.

| | |
|---|---|
| Transform | analysis and synthesis exact to ~1e-15 (`transform_check.py`), fold-validated against the ISO kernel, one complex FFT per block, so it is portable to Luau |
| Scalefactor grid | 128 uniform bands of 16 bins, spanning 0-24 kHz with nothing cut off |
| Rate | ~11.7 kbit/s per channel mantissa-only; ~12.8 with live-band side info |
| Quality | 9.1-12.5 dB gain-fitted ES at 512 b/f, versus 3.9-7.3 dB for shipping RSC2 at mp=8 (~17 kbit/s/ch) |

So the honest pitch is **5.2 dB better than RSC2, at roughly 1.4x lower rate, on both
reference tracks**. That +5.2 dB was constant across files only once the ruler was fixed;
before that it appeared to swing between 2.3 and 6.4 dB, which was measurement, not material.

## Rulers

The codec's quality is decided by two numbers and one window:

1. **Gain-fitted ES.** Fit the least-squares gain, then measure what's left. Without the
   fit, a coherent level offset is charged as error and caps ES at whatever that offset
   allows - a pure 6 dB level loss caps ES at 6 dB even for perfect coding. Every codec
   here loses a few dB to its own headroom, so naive ES ran ~3 dB low across the board.
   `gainfit_es.py` prints both.
2. **Audible error.** `sum max(0, E_bin - THR_bin)` where THR is the masking threshold of
   the *original* signal, shared by both arms so neither is graded against its own curve.
   This is the number that agrees with listening; raw ES is for spotting bugs.
3. **The spectrogram, and your ears.** Both have beaten the metrics four separate times,
   which makes them a required stage, not a formality.

Two rules fall out of this and they are non-negotiable in this directory: a result that
looks *flat* against budget is treated as a broken harness until proven otherwise, and any
trick that costs side info is charged **inside** the budget or the comparison doesn't count.

## What is settled

- **The quantiser step is `2^-b`, not `2^(1-b)`.** Worth +0.45 to +0.66 dB, confirmed by
  A/B and by falsification (`audio` 12.95/12.77/10.97 at 512/256/128 b/f, `tone` 14.52,
  `clicks` 19.06, un-coded transform 306.97 as the sanity bound).
- **Bins below their threshold get zero bits.** Quantising a masked bin with the coarse
  step its bits can buy adds more noise than it hides: this is why the quiet neighbours of
  a pure tone used to eat 96% of its residual.
- **Psychoacoustic gating is a win**, on every track tried, by the audible ledger, even
  where raw ES says the blind allocation wins. Level-match before trusting any listen.
- **Upward band folding is dead.** Mirror alias is proportional to the loudest member of
  each pair, and upward folding pairs the loudest bins (bass) with the emptiest (treble),
  so the alias is always bass-loud. No per-frame energy correction fixes a shape error.
- **Run-based noise fill is affordable, and its placement is not yet solved.** Where a band
  is fully zeroed, the decoder fills it with seeded noise whose level is one scalar per
  contiguous run of dead bands, costing ~6 bits per run rather than per band: 464 mantissa +
  48 fill = 512 b/f, and 16-19 kHz *aggregate* density lands within ~1-2 dB of the source
  instead of -95 dB below it. It costs ~1.2 dB of ES, which is expected, not a regression:
  matched-energy noise carries roughly twice the error power of zeroing, while sounding like
  air instead of a hole.

  The unresolved part is *where inside a run* that one scalar puts the level. Measured per
  dead band, a linear mean gives a median error of -1 dB but a tail to -33 dB, with 40% of
  bands in multi-band runs more than 6 dB off - the mean is dominated by the run's loudest
  band, so the floor lands at the top and the quiet members are left behind. 93% of dead
  bands live in multi-band runs, so this is not an edge case. A floor 20 dB low is
  effectively no fill, and a floor 7 dB high is noise above the very threshold that justified
  zeroing. The mean-in-dB (geometric) alternative was measured against it, at the same price,
  and it is not the fix it looked like:

  | floor       | gain-fitted ES | audible | 16-17 kHz | 18-19 kHz |
  |-------------|---------------:|--------:|----------:|----------:|
  | source      |            --- |     --- |     -62.0 |     -62.0 |
  | linear mean |           7.96 |   -53.1 |     -63.4 |     -64.3 |
  | geometric   |           8.27 |   -53.4 |     -65.9 |     -68.1 |
  | no fill     |           9.11 |   -55.9 |     -74.5 |     -76.8 |

  The geometric mean wins both error metrics by about 0.3 dB for a reason that should not
  fool anyone: it is simply *quieter*, so there is less noise to be wrong about. It buys
  that by giving up 1.6-3.8 dB of the density the fill exists to restore, putting 18-19 kHz
  6.1 dB under source where linear sits 2.3 dB under. A metric that improves when you do
  less of the intended job is not evidence of a better design - this is the zero-fill result
  wearing a different hat. Linear stays.

  Which leaves the honest statement about fill, and it is not the one this section had:
  fill is a *cost*, not a win, by both rulers that decide quality. `nofill` scores 9.11 dB
  and -55.9 dB; `run_lin` scores 7.96 and -53.1. It buys 11-12 dB of restored density above
  16 kHz and nothing either metric rewards, so it is a listening decision priced at about
  1.1 dB of ES and 2.8 dB of audible error. It stays behind a flag, and `fill_experiment.py`
  is the place that price gets re-derived.

## Speed

The encoder is the slow part, and the profile was not where I expected. Analysis is ~60x
realtime on its own and never the problem; `allocate` was 88% of encode time, and inside
`allocate` the water-filling bisection was most of it.

Three changes, each measured, each checked for bit-identity against the version it replaced,
because an allocation that changes is a different codec and not a faster one:

- **Water-filling is a selection, not a search.** Bit *b* of band k exists while
  `floor < S_k - 6.02(b-1)`, so that expression is the *price* of the bit, and the frame's
  total is a count of prices above the floor. The lowest floor fitting a budget is therefore
  the (budget+1)-th largest price, which is `argpartition` rather than sixty passes over the
  frame. About 5x faster on the stage, **bit-identical across every coefficient of two files**
  (1,445,888 comparisons). The bisection is kept as `_waterfill_all` and reachable with
  `allocate(..., search=True)` precisely so that claim stays checkable.
- **Analysis and synthesis run in blocks** rather than a per-frame loop: ~1.7x on each stage,
  identical to ~1e-15. The overlap-add is the pairwise sum that `hop = N/2` admits, but note
  that form drops the previous chunk's tail at every `CHUNK` boundary unless it is carried
  forward - it was wrong for exactly that reason before it was caught.
- **The mask's skirt is a cumulative sum, not a matrix** - and this is the one where *how*
  you vectorize mattered more than whether. The kernel is `10^(-c*(z_i - z_j))` and the
  attenuation is linear in ERB-rate, so it factors as `10^(-c*z_i) * 10^(+c*z_j)`: rank one,
  and summing maskers above or below bin *i* is a prefix sum. O(48 x 2048) exponentials per
  frame becomes O(2048). 1.62 s of a 2.39 s encode became 0.024 s, and the encoder is now
  ~0.61 s for 15 s of `cam` (24.7x realtime) from 2.39 s.

  Two things about that are worth writing down, because both were surprising.

  **Batching alone was a loss.** Running the *same* matrix model over a stack of frames
  instead of frame-by-frame made it 20% *slower*: the batched form touches 353x2048
  intermediates that leave L1 and get reallocated per call, while the loop keeps a
  2048-wide working set hot. Vectorizing is not the same as optimizing, and the algorithmic
  form, not the loop order, is what bought the speed.

  **Self-masking must be excluded, not subtracted.** The first version summed every masker
  including the bin itself and subtracted afterwards. That is the same algebra and a
  numerically terrible one: at a loud bin the two numbers agree to ~16 decades, so the
  answer is cancellation residue. It showed up as a 1.2e-2 dB drift off an explicit
  per-masker loop, and once, on one file, one flipped coefficient. Using *exclusive* prefix
  sums - maskers strictly below and strictly above, never the bin itself - drops that to
  5.6e-8 dB and removes the flip. The lesson generalises: the two forms disagree by up to
  113 dB in raw mask dB, and all 23 disagreeing bins of 722,944 are **below ATH**, where
  `max(mask, ATH)` overwrites them, so the codec never sees the difference. A raw-threshold
  diff that big between two implementations of one model is a scale artifact, not a bug, but
  only because it hides under the ATH floor - which is exactly why the gate is the bit map
  and not the threshold.

- **No format, no decoder.** Nothing here is serializable yet. RSC2 v3 (`../rsc2_testing_lab`)
  is the shipping codec until this has a container and a Luau decoder.
- **High-frequency efficiency.** Two thirds of all audible error on `cursedseptember` sits
  in 5-10 kHz, where that band pays 1.75x its share of damage per bit, and 10-14 kHz pays
  5.8x. Whether that's the waterfill bias, the uniform band widths, or the mask's HF slopes
  is a decision `hf_audit.py` was written to make, not yet made.
- **Noise fill is not yet in the core**, and the level it puts inside a run is unresolved
  (see the previous bullet). Run `fill_experiment.py` for the budget side and read the per-band
  level error before trusting any listening file it produces.
- **PVQ is unresolved.** At honestly-paid bits scalar quantisation beats PVQ by 7-8 dB, but
  the loss-vs-budget curve was flat, which in this repo is a harness symptom, not a finding.
- The model is numpy. Encoding is far too slow for real time, and the decoder does not exist.
