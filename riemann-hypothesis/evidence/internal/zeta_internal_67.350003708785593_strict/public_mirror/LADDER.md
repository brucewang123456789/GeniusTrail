# The bandwidth ladder, and a methodological warning

## Measured ceilings

All values below use the **complete** three-level seed enumeration `{1.04, 1.975, 2.915}^q`
plus four-level draws, perturbations, uniform starts, and Nelder-Mead + Powell polish. They
are float ceilings on `C`, not certified values.

| `q` | `d = 2q` | `r = q+1` | `T` | deterministic ceiling |
|---|---|---|---|---|
| 6 | 12 | 7 | 7/6 | 0.673457890 (certified value of that line: 0.6734574870891890) |
| 7 | 14 | 8 | 8/7 | 0.673522442 was `q=8`; best `q=7` was 0.673489246 |
| 8 | 16 | 9 | 9/8 | 0.673522454 |
| 9 | 18 | 10 | 10/9 | 0.673529201 |
| 10 | 20 | 11 | 11/10 | 0.673552660 |
| 11 | 22 | 12 | 12/11 | 0.673558943 |
| **12** | **24** | **13** | **13/12** | **0.673577795**  ← frozen here |
| 13 | 26 | 14 | 14/13 | 0.673572287 — **turns over** |

The matrix theorem is proved for general `(d, r)`, so climbing the ladder needs no new
mathematics. What it costs is certification dimension: the local certificates are
`q`-dimensional.

## The methodological warning — read before trusting any high-`q` number

At `q = 7` the three-level seed family has `3^7 = 2187` members and can be enumerated
completely. At `q = 16` it has `3^16 = 43,046,721`. Sampling a few thousand of those is
**exponentially thin coverage**, the local minimisation then misses the true minimiser, the
`epsilon_s` come out too high, and the ceiling is inflated.

Measured inflation from thin sampling (6561 sampled seeds) versus complete enumeration:

| `q` | thin-sample ceiling | complete-enumeration ceiling | inflation |
|---|---|---|---|
| 9 | 0.673549945 | 0.673529201 | +2.07e-5 |
| 10 | 0.673572779 | 0.673552660 | +2.01e-5 |
| 12 | `eps(1.0)` 0.008170902 | 0.008145172 (heavier search) | +2.57e-5 on a single slope |

A thin-sample sweep suggested ceilings of `0.673704` at `q = 16` and rising. **Those numbers
are artefacts.** They were discarded. This is the same failure mode that produced the two
refuted candidates in `REFUTATION.md`, in a new disguise: there the seed family was missing a
structure; here it is present but too sparsely sampled.

**Rule.** Freeze only at a `q` for which the three-level family can be enumerated
*completely* within budget. That is why this repository stops at `q = 12` (`3^12 = 531441`)
and reports `q = 13` (`3^13 = 1594323`) as a single confirming measurement rather than a
platform for further climbing.

## Two tiers, and which to use

| tier | value | certificate dimension | realistic? |
|---|---|---|---|
| practical | `67.351669934653%` (`q = 8`, in the internal archive) | 8 | very hard; the prototype does not close 7 dimensions |
| exploratory | **`67.357785014889%`** (`q = 12`, this repository) | 12 | not feasible by any method known to the author |

Both are candidates. Neither is proved. The `q = 12` figure is published as a map of where
this framework's ceiling lies, not as a claim that it can be certified.
