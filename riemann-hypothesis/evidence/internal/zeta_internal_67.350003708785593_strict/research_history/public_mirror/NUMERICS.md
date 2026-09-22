# Numerical methodology and the complete experimental record

Everything here is **FLOAT** unless marked otherwise. It documents how the configuration
was chosen and, just as importantly, which routes were closed and why.

## 1. How `epsilon_s` must be estimated (and how it was got wrong twice)

`epsilon_s` is the global minimum of `P_loc + s Q_loc` over `[0, inf)^q`. The landscape has
many local minima. Two circulated candidates were published on targets that are **false**
because the multistart seed family consisted only of alternating gaps near `1.04` and
`1.975`. The true minimisers of several slopes contain a **large gap near `2.91`**.

Mandatory seed family: all `3^q` combinations of `{1.04, 1.975, 2.915}`, plus four-level
draws, perturbations at several scales, and uniform random starts; polish each candidate
with **both** Nelder-Mead and Powell. Under the corrected family the minima at some slopes
drop by up to `5.1e-5`, which is two orders of magnitude larger than the `3e-7` margins the
refuted candidates used.

**Evaluator noise.** A warm-archive evaluator (reusing minimisers between configurations)
carries about `1.2e-5` of noise — the same order as the gains being optimised. The same
configuration scored `0.673494302` warm and `0.673481662` cold. Therefore:

> Optimise with whatever is fast, but **never freeze or report a number that has not been
> re-measured by a deterministic, archive-free evaluator.**

Final deterministic re-measurement of all stored configurations (fixed 200k candidate set,
no archive, Nelder-Mead + Powell on the top 12 per slope):

| configuration | deterministic ceiling |
|---|---|
| o3 | 0.673479869393 |
| cd7 | 0.673451441570 |
| **o5 (frozen)** | **0.673487176593** |
| o6 | 0.673481661720 (its warm score of 0.673494302 was noise) |

## 2. Design space and what each degree of freedom is worth

Free objects: the window `v` (coefficients `c_j` and frequencies `omega_j`), the pressure
vector `b`, and the pair weights `a_{ij}`. The only constraints are `v > 0` on `[-1/2,1/2]`,
`b >= 0`, `a >= 0`, and the span-capacity identity (SC). Everything else is design freedom.

Routes probed and **closed** at the frozen configuration:

| direction | measured effect | verdict |
|---|---|---|
| drop reflection symmetry of `a_{ij}` (18 -> 27 parameters) | +2e-6 | symmetric optimum already near-best |
| raise `H` toward the `H`-optimal window (blend `t: 0 -> 1`) | ceiling falls monotonically to 0.673206 | machinery falls faster than `H` rises |
| lower `H` (extrapolate `t: 0 -> -2.5`) | ceiling falls monotonically to 0.670229 | current window is a local max on this line |
| certify more slopes | <= 1e-6 | the ceiling already assumes a dense slope set |
| `q = 8` (`d=16, r=9, T=9/8`) | untuned 0.673367, i.e. 1.1e-4 below tuned `q = 7` | not committed; also raises certificate dimension |

Routes that **paid**:

| direction | measured effect |
|---|---|
| random-direction 3-point line search (replacing axis-aligned compass search) | +5.3e-6 per block, success 4/9 |
| freeing the window frequencies `omega_j` | +7.1e-6 per block, success 5/9 |

Freeing `omega_j` is legitimate and costs nothing in certification: the formulas for
`I1, I2, J` and the Arb evaluation hold for arbitrary real `omega`, and the certificate
dimension stays `q`.

**Ceiling on `H`.** Maximising `H(v) = 2 - (I2+J)/I1^2` over all `v >= 0` supported on
`[-1/2, 1/2]` gives `sup H = 0.672499502`. The frozen window has `H = 0.672190930`, i.e.
`H` is deliberately **below** its own optimum, because the machinery term `C - H` gains more
than `H` loses. Both directions along that line were measured and both are worse.

## 3. Value versus certification margin

Every `epsilon_s` is set a uniform `delta` **below** the hardest float minimum found.
Larger `delta` means an easier certificate and a smaller `C`:

| delta | exact C | percent |
|---|---|---|
| 1e-6 | 0.673484947079927 | 67.348494708% |
| 3e-6 | 0.673483706894675 | 67.348370689% |
| **6e-6** | **0.673481848852621** | **67.348184885%** |
| 1e-5 | 0.673479397503875 | 67.347939750% |

`delta = 6e-6` was chosen: it is ~18x the margin class that was refuted, and the cost in `C`
is only `3.1e-6`. The optimisation target of this programme is deliberately
**"the easiest witness that still clears the bar"**, not "the largest decimal".

## 4. The verifier, and why node count is not the bottleneck

`verifier/bb2.cpp` is a rigorous `q`-dimensional interval branch-and-bound for `C(s)`:
outward-rounded double intervals (`nextafter` widening); rigorous `sinc, sinc', sinc''`
with a Taylor branch for `|z| < 0.9`; a precomputed rigorous cell table of
`min W`, `max |W'|`, `min W''` with sparse range structures; and three lower bounds per
node — natural interval extension, the mean-value bound (valid without convexity), and a
tangent-plane bound accepted only when an LDL elimination certifies the Loewner lower
Hessian positive definite. An unresolved terminal cell or a node-cap hit returns
`INCOMPLETE`; nothing is ever accepted without outward rounding.

Measured: about `2e5` nodes/s single-threaded; `4e7` nodes on one slope ended
`INCOMPLETE` with **no counterexample**. The bottleneck is **terminal-box proof precision**,
not search volume. Adding nodes is the wrong response; adding a convex quadratic lower
bound on terminal and near-terminal boxes is the right one.

## 5. The move to `q = 8`

With `q = 7` exhausted (every route in section 2 closed or decelerated to `2e-6` per block,
and a further enlargement of the window basis worth only `2.1e-6` per block), the only lever
of the right magnitude left was the bandwidth itself. An earlier `q = 8` probe with a crude
weight heuristic scored `0.673367` and was correctly judged not worth committing to; with a
structured seed and the same random-direction line search the trajectory was

    0.673474 -> 0.673482 -> 0.673502 -> 0.673513 -> 0.673522   (about +1.2e-5 per block)

overtaking the best `q = 7` value (`0.673489`) in the second block. The deterministic
archive-free evaluator confirmed the frozen value at `0.673522454`, i.e. the fast evaluator
carried no noise penalty here.

**The price.** `q = 8` means `d = 16`, `r = 9`, `T = 9/8`, and local certificates in
**eight** variables rather than seven. The matrix theorem costs nothing (it is proved for
general `(d, r)`), but the certification burden rises. This is a deliberate trade: the
candidate crosses `67.35%` at the cost of a harder open problem, and `STATUS.md` records it
as such.

## 6. Margin/value table at the frozen `q = 8` configuration

| delta | exact C | percent |
|---|---|---|
| 1e-6 | 0.673520033907028 | 67.352003391% |
| 3e-6 | 0.673518700078865 | 67.351870008% |
| **6e-6** | **0.673516699346528** | **67.351669935%** |
| 1e-5 | 0.673514031721902 | 67.351403172% |
