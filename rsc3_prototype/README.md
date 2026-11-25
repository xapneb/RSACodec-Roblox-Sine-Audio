# RSC3 prototype

Block-transform codec: fast MDCT, uniform scalefactor bands, reverse-waterfill bit
allocation against a psychoacoustic threshold model. It is the successor to the RSC2
additive-synthesis line. It now has a container, an encoder and a decoder, and every number
below is measured through the files those programs write. What it does not have is a decoder
that runs *on Roblox*: the Luau port is the remaining work, and until it exists RSC2 is what
ships.

```
rsc3_stream.py          container, bit grammar, and the frame the decoder reads
rsc3_encode.py          encoder CLI: --bitrate searches a budget, --budget spends one
rsc3_decode.py          decoder CLI, which needs nothing but the file
rsc3_codec.py           model core - transform, thresholds, allocation, quantiser
test_stream.py          grammar and sound-identity regressions against _ref_stream.py
test_roundtrip.py       encode -> file -> decode: the two CLIs must not disagree by anything
rate_curve.py           measured kbit/s against quality, every field charged
transform_check.py      the transform's own proof: kernel match + full-signal OLA
ab_codec.py             A/B generator, the protocol that produces every listening file
gainfit_es.py           re-scores any file under a fitted gain (see "Rulers")
audible_sweep.py        quality vs budget, scored in audible-error units
hf_audit.py             bits / signal / error / audible-error per frequency band
fill_experiment.py      noise-fill, priced per run of dead bands, at equal bitrate
vector_hits.py          sparse high-frequency layer, priced combinatorially, at equal bitrate
```

Run anything from this folder: `python ab_codec.py ..\\rsc2_testing_lab\\ij.wav`
(paths resolve relative to the repo, so the scripts work wherever they're checked out).

## How far along it is

Every number in this section is a file on disk: `rate_curve.py` encodes a fixture, reads the
bytes back, and scores the decoded audio. The RSC2 column is `_rsc2_curve.py` doing the same
thing through `rsc_encoder.py` and `rsc_decoder.py`, with the `mp` knob as the only dial.
Both columns therefore pay for side info, container overhead and all, which is the only form of
comparison this repo is allowed to publish.

| | |
|---|---|
| Transform | analysis and synthesis exact to ~1e-15 (`transform_check.py`), fold-validated against the ISO kernel, one complex FFT per block, so it is portable to Luau |
| Scalefactor grid | 128 uniform bands of 16 bins, spanning 0-24 kHz with nothing cut off |
| Rate, low end | down to 1.2 kbit/s/ch, and there the container is the rate: at 16 b/f (5.5 kbit/s) 37% of the file is scale, counts and fill, and only 63% is mantissas |
| Rate, working region | 10.4 kbit/s/ch at 40 b/f, 15.0 at 64, 25.1 at 128 |
| Field cost | b/f at budgets 1 / 16 / 40 / 192: scale 6 / 14 / 21 / 59, counts 13 / 52 / 86 / 159, mantissas 21 / 149 / 304 / 1174, fill 12 / 22 / 32 / 60 |
| Quality | 4.9-10.3 dB gain-fitted ES across that range, measured on the decoded file |

Two of those rows price the bit map that ships on by default. The opt-in shape field adds 22-28 b/f
at these rates and is priced in its own section under "What is settled"; it is not in any number
above.

Every table on this page prices the **default bit map**: mask-safe flat fill, no shape field. The
shaped arm above 14 kHz is priced in its own section under "What is settled", so nothing here
includes bits that are only there if `--fill-shape` was asked for.

Measured against shipping RSC2 (gain-fitted ES, both from files, `mp` = RSC2's partial cap):

| track | rate (kbit/s/ch) | RSC2 | RSC3 | gain |
|-------|------------------|-----:|-----:|-----:|
| `cam15` | 9.7 vs 10.4 | 3.21 | 4.91 | **+1.70 dB** |
| `cam15` | 13.5 vs 13.5 | 3.72 | 5.48 | **+1.76 dB** |
| `cam15` | 17.2 vs 15.0 | 4.10 | 5.75 | **+1.65 dB** |
| `ij15` | 9.7 vs 11.3 | 6.18 | 9.12 | **+2.94 dB** |
| `ij15` | 13.5 vs 14.7 | 6.91 | 9.94 | **+3.03 dB** |
| `ij15` | 17.2 vs 16.3 | 7.36 | 10.32 | **+2.96 dB** |

So the honest pitch is **1.7 dB better than RSC2 on `cam15` and 3.0 dB on `ij15`, at a matched
rate**, and RSC3 gets there with fewer bits on three of the six rows. This replaces the "+5.2 dB
at 1.4x lower rate" line, which was not a measurement: it compared an RSC2 *file* rate against
an RSC3 *budget*, and budgets here run roughly 7x light, because the container charges
(1 + b) bits for every nonzero bin of every live band and the budget only counts bit planes.

## Rulers

The codec's quality is decided by two numbers and one window:

1. **Gain-fitted ES.** Fit the least-squares gain, then measure what's left. Without the
   fit, a coherent level offset is charged as error and caps ES at whatever that offset
   allows - a pure 6 dB level loss caps ES at 6 dB even for perfect coding. On the current
   fixtures the two agree to 0.01-0.05 dB, which is worth saying plainly because the old
   claim here was that naive ES ran "about 3 dB low". It no longer does: with the ceiling
   re-encode in place the decoder gives back about the level it was given, so the fit is
   protection against a regression, not a correction that always applies.
   `gainfit_es.py` prints both, and `rate_curve.py` now prints both columns separately -
   it used to print the fitted number in both, which made the plain column decorative.
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
- **Run-based noise fill is in the format, and it is levelled by the mask.** Where a band
  is fully zeroed the decoder knows it (`b == 0`), so the only thing left to transmit is how
  loud the replacement is. One level word and one slope per contiguous run of dead bands costs
  8 bits, and at 10 kbit/s it turns a band that decoded to *literal silence* into a floor under
  the noise: on `cam15` the 14-20 kHz group goes from nothing measurable at all to -73 dBFS,
  and 10-14 kHz from -68 to -64.

  **The noise is a pure function of `(frame, band)`** - mulberry32 over two integers, normalised
  per band by a value both sides compute. No generator state, no sequence, no dependence on the
  order bands get visited, which is what a Luau port needs and what the old `fill_experiment.py`
  never had, and is the only reason that experiment could not ship.

  **The level is the part that was wrong before.** An earlier version filled each run at its own
  mean energy, and ES said it cost 1.2 dB - which told us nothing, because error power always
  prefers silence to matched noise, so that metric is structurally against the fill and cannot
  answer whether the noise is audible. The ruler that can is the original's own masking
  threshold per bin, and against it the mean came out badly: **24% of the bands the fill touched
  were above their own mask, the worst by 45 dB.** That is not a tuning miss, it is geometry -
  a run's mean is set by its loudest member, and a band quiet enough to need zero bits is
  exactly the kind of band that hides nothing.

  The shipped rule is a max-min: the largest level that stays under the mask at *every* bin of
  the run, plus a 2-bit slope so a run can follow its mask's own fall instead of being pinned by
  its highest bin. Levels round down, never to nearest, so quantisation cannot make the decoder's
  floor illegal. Measured through the real files, on the decoder's own samples:

  | arm | ES @10 kbit/s, cam/ij | filled bins over mask, cam15 @10 | worst dB over |
  |-----|--------------------:|---------------------------------:|--------------:|
  | no fill    | 5.04 / 9.17 |             0 of 0      |      0.0 |
  | mask-safe  | 4.72 / 8.61 |          0 of 35,254  |      0.0 |
  | mean level | 3.80 / 7.94 |    11,099 of 39,348  |     44.7 |

  The price of shipping the safe arm is 0.3-0.6 dB of gain-fitted ES and about 5% of rate; the
  two extra fields per run are charged inside the estimator that picks the budget, so a rate
  search can no longer win bits it does not pay for. `--no-fill` reproduces the previous bit map
  byte for byte, and the header carries a flag so a decoder is *told* which format it is reading
  rather than inferring it. The mean-level arm survives as `--fill-mode mean`: it is louder and
  airier, it is what the old experiments sounded like, and it is measurably audible, so it is an
  option and not the format.

  **What this does not prove:** the audit uses a stationary spectral mask. It says the fill is
  under the mask, and it says nothing about what happens across a transient, where a noise floor
  can be exposed by temporal masking the model does not have. If the fill ever sounds wrong,
  that is the failure mode to look for, not the mean.

- **A shaped fill rides on top of the flat word above 14 kHz, as an option.** The flat word has a
  geometric problem, not a tuning one. A dead run on this material is routinely ~120 bands wide -
  one frame of `cam15` has a single run spanning 1.2 to 23.9 kHz - and one level plus one *straight*
  slope has to follow a mask curve that is not straight. `_ceiling_probe.py` measures the result by
  running the shipped rule and comparing every dead band against its own ceiling, frames 20-70:

  | dead region | bands | median dB under its own ceiling | share >10 dB under |
  |-------------|------:|--------------------------------:|-------------------:|
  | `cam15` 9-12 kHz | 160 | -17.0 | 81% |
  | `cam15` 12-14 kHz | 110 | -26.7 | 85% |
  | `cam15` 14-18 kHz | 210 | -21.4 | 60% |
  | `cam15` 18-20 kHz | 110 | -6.4 | 38% |
  | `cam15` 20-24 kHz | 210 | -10.6 | 53% |

  On `ij15` it is worse, and worse in the direction that matters: -31 to -48 dB across 9-20 kHz,
  with 99-100% of the bands more than 10 dB below what the decoder could legally play. One straight
  slope, pinned wherever it can be legal, has to be wrong on both sides of the curve.

  The shape field is that staircase: each block of 16 dead bands in the region gets its own level
  word, so a run can follow a curve instead of a line. One flag bit per touched run, then a 6-bit
  first-block code and Rice-coded differences; the region is *derived* from `sr`, `n2` and `width`
  rather than transmitted, and a code of 0 means the flat word stands, so every word that shipped
  keeps shipping and keeps being charged. Header flag bit 1 (`SHAPE_FLAG`) declares it, because a
  different bit map is a new codec and not an optimization.

  Measured through files, `--fill-shape` at ~10 kbit/s, rates from the file and audio from the
  decoded samples, mask column graded against the original's per-bin threshold by
  `_fill_verify.py`:

  | track / arm | kbit/s | ES | bins over mask | 14-18 kHz | 18-24 kHz |
  |-------------|-------:|----:|---------------:|----------:|----------:|
  | `cam15` flat | 9.78 | 4.72 | 0 / 35,254 | -72.0 | -77.4 |
  | `cam15` shaped | 9.73 | 4.57 | 0 / 35,524 | **-55.0** | **-60.8** |
  | `ij15` flat | 9.79 | 8.61 | 0 / 40,329 | -105.9 | -113.0 |
  | `ij15` shaped | 9.79 | 8.41 | 0 / 41,015 | **-77.5** | **-87.2** |

  So it buys 16-17 dB on `cam15` and 26-28 dB on `ij15` in the same file size, and not one bin in
  35,524 / 41,015 ends up above the mask - which is the burden of proof a noise floor has to carry
  before any of the other numbers mean anything. It costs 21.8 b/f on `cam15` and 27.5 b/f on
  `ij15` (about 0.5 kbit/s per channel), and it is not an occasional extra: `cam15` lit at least
  one block in 291 of its 353 frames, `ij15` in 349, and no frame paid nothing at all
  (`_shape_rate.py`, which walks the same `shape_field` call the writer is handed). The estimator
  charges it, which is what makes the rate honest: the probe said 9.92 against 9.72 measured, the
  same -2% bias the flat field carries, and field accounting versus the file is 3 bits. It is not
  only a low-rate trick: at 128 kbit/s it still lands 0 bins over mask, buys +6.2 dB in 18-24 kHz on
  `cam15` and +17.3 dB on `ij15`, for 0.2-0.7% of rate.

  **What the rulers say is that neither of them decides this.** At 10 kbit/s, gain-fitted ES goes
  *down* by 0.15-0.2 dB, which is exactly what putting noise where there was silence does to a
  metric that structurally prefers silence - the same trap the mean-level fill fell into, and the
  reason it is not evidence. The audible ledger is flat too, and this time through the files:
  -45.2 dB with the flat word and -45.1 dB shaped on `cam15`, -59.0 against -58.7 on `ij15`, with
  the share of error power that clears the mask the same to a tenth of a percent (54.0% / 53.9%).
  The reason is geometric: on this material that ledger is dominated by 0-10 kHz and the shape field
  is not permitted to touch anything below 14 kHz. An ES that cannot move, in a region the audible
  ledger never reports on, is a decision for the ears:
  `listening/cam15_shape_{flat,blk16,bins}.wav` is that A/B, and until someone has listened it this
  is an option with measured prices, not a win. (`_shape_aud.py` produces both columns by running
  the two CLIs, and scores both arms against the original's mask so neither is graded on its own
  curve.)

  Shape and the sparse hits layer buy different currencies and do not rank against each other. The
  hits arm bought the *positions* of sines that really are above the mask, and on `cam15` that was
  cheap for what it is: 14.6 b/f for 25 dB of 14-18 kHz content. It is also per-frame material -
  `ij15` offers 0.1 hits per frame above 14 kHz, because there is nothing there to place - and its
  `m=1` row decodes 3 dB hot of the source because every hit is given the same magnitude. Shape
  needs no content to exist, so it works on both tracks, and its levels are floored under each
  band's own mask by construction, which a single-magnitude hit layer is not. Neither is a
  substitute: a floor is not a position, and positions are not a floor.

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

- **The format and a Python decoder exist; the Luau one does not.** `rsc3_stream.py` is the
  grammar, `rsc3_decode.py` reads a file with no knowledge of the allocation model, and
  `test_roundtrip.py` proves the two CLIs cannot disagree. RSC2 v3 (`../rsc2_testing_lab`) is
  still what ships until this decodes on Roblox.
- **High-frequency efficiency.** Two thirds of all audible error on `cursedseptember` sits
  in 5-10 kHz, where that band pays 1.75x its share of damage per bit, and 10-14 kHz pays
  5.8x. Whether that's the waterfill bias, the uniform band widths, or the mask's HF slopes
  is a decision `hf_audit.py` was written to make, not yet made.
- **The decoder's cost is only half measured.** Every number above is bits and sound, but the
  player pays a *sine* for each coefficient it plays, so a codec that sounds the same while
  leaving more bins dead is cheaper to run. `vector_hits.py` scores that axis now (sines per
  frame, and it only counts lit bins that carry mantissas or hits, not fill noise), but it is
  scored as a side effect of a bandwidth experiment, not as a budget of its own. Bits and
  oscillators are still two different frontiers and only one of them has been swept.
- **PVQ was never implemented, and what replaces the claim is measured, not algebra.** The old
  line about scalar beating PVQ "by 7-8 dB at honestly-paid bits" is gone: no code in this tree
  produced it. What has been measured is a *sparse hit layer over the dead high-frequency region*
  (`vector_hits.py`), priced exactly per frame as `log2 C(L, R) + R (1 + m)` plus a delta-coded
  region scale. The region is masked out of the scalar waterfill so the bits are new, not spent
  twice, and only bins above the mask are bought. On `cam15` at a matched rate:

  | arm | kbit/s | gain-fitted ES | sines/f | 14-18 kHz |
  |---|---:|---:|---:|---:|
  | scalar, 48 b/f | 11.98 | 5.12 | 63.5 | -71.7 |
  | hits 14k, `m=1`, 14.6/f | 11.84 | 4.46 | 57.6 | -46.5 |

  So 25 dB of 14-18 kHz content for 0.66 dB of ES and six oscillators. At equal *quality* the
  same bandwidth costs +31% rate (ES ~4.17 both: scalar 7.23 kbit/s, hits 9.50). One magnitude
  bit beats two on rate and density both, which is the actual abuse: the position code is where
  the money is, magnitude bits are where the waste is. Three things keep it from being a win:
  above 18 kHz it buys nothing in any arm, the m=1 row
  sits 3 dB *hotter* than the source because every hit decodes to the same level, and on `ij15`
  there are 0.1 hits per frame above 14 kHz because that material's HF is under its own mask -
  the layer is per-frame material, not a feature you can turn on.
- **The premise the experiment was built to test is false.** Buying sines with a vector code does
  not leave slack for *more* oscillators at a matched rate: scalar plays more of them for the same
  bitrate, they just decode to noise nobody can hear. The layer buys bandwidth, not headroom, and
  the intuition that low quantisation noise means purchasable headroom was wrong about *where* the
  headroom is. Whether a different lever - the band's own `b` quantum, or a finer scalefactor
  grid, which is the other half of the algebra this file replaced - is where the real headroom
  sits is still unmeasured.
- **The shape field has no gate.** It is written in nearly every frame that has a dead run, and its
  per-frame cost is tightly clustered rather than heavy-tailed: mean 21.8 b/f, p50 26, p90 29, max 31
  on `cam15` (`_shape_rate.py`), so there is no population of cheap frames to skip. That is why it is
  an option and not the format - it buys density with a standing charge, not with an occasional one.
  The thing to try against it is the cheapest option and needs no new decoding logic, because a
  skipped field already decodes to the flat word: one bit per frame saying this frame's region is not
  worth shaping. Whether that bit ever pays for itself is unmeasured, so it is not claimed.
- The model is numpy. Encoding is far too slow for real time, and the decoder does not exist.
