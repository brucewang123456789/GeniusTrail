# Independent review of the q=8 certification line, and what it changes

Scope: the `exact_candidate` package (`q=8`, `m=535`, `eta=4999/5000`,
`C = 67.350048130121891%`) and the strategy notes that accompanied it.
The configuration is bit-identical to `cfgA_rat.json` (same `H = 672167187145431/10^15`),
so all hardened float minima measured for that configuration apply directly.

## 1. What checks out

* **The exact assembly is correct.** Recomputing the margin sweep independently in exact
  rationals reproduces `67.350069392%` at a uniform `3e-5` margin, matching the reported
  `67.35006939157365%` digit for digit. The arithmetic is not fabricated.
* **The three targets are not refuted.** Against the hardened minima of this configuration
  (complete `3^8` three-level enumeration plus four-level draws, perturbations, uniform
  starts, Nelder-Mead and Powell polish):

  | slope | target | hardened float minimum | margin |
  |---|---|---|---|
  | `1/2` | `27/4000 = 0.00675` | `0.0067800531` | `3.005e-05` |
  | `19/20` | `39519/5000000 = 0.0079038` | `0.0079338933` | `3.009e-05` |
  | `1` | `40069/5000000 = 0.0080138` | `0.0080438403` | `3.004e-05` |

* **The verifier-target bug report is right**, and isolating it rather than reusing the
  affected gate data was the correct call.

## 2. One claim that needs correcting

The interpolation lemma is true: `F_s(g) = P(g) + s Q(g)` is affine in `s`, so pointwise
`F_1 = (F_{0.95} + F_{1.05})/2 >= (eps_{0.95} + eps_{1.05})/2`. Certifying `s = 1` separately
is therefore indeed unnecessary.

**But it buys nothing in the assembly.** The envelope is `p(E) = max_s ( n eps_s - s E )`.
The interpolated line is

    L_1(E) = n eps_1 - E = (1/2) L_{0.95}(E) + (1/2) L_{1.05}(E)  <=  max( L_{0.95}, L_{1.05} )

so it is dominated everywhere and never attains the maximum. Adding it to the certified set
leaves `p`, `R` and `C` unchanged. The `s = 1` certificate is redundant, not "saved".
Consequently the rise from `67.35006939%` to `67.35034419%` came from reallocating margins,
not from the interpolation. The lemma is sound; the causal attribution was not.

## 3. The finding that changes the strategy

**The certification margin is capped at `3.084e-05` by the `67.35%` requirement, and no
choice of certified slopes raises it.** Maximum uniform margin subject to `C > 67.35%`, by
exact-rational bisection over `(m, eta)` for each slope set:

| certified slopes | max uniform margin | resulting `C` |
|---|---|---|
| `1/2, 19/20, 1, 21/20` | **3.084e-05** | 67.350000755826% |
| `1/2, 19/20, 1` | 3.069e-05 | 67.350000634425% |
| `1/2, 1, 21/20` | 2.694e-05 | 67.350000062653% |
| `1/2, 19/20, 21/20` | 2.291e-05 | 67.350003708175% |

The `3e-5` witness already sits on this frontier. Hunting for a "proof-optimal rational
witness" with materially wider margins is therefore a closed route at this configuration.

The margin budget obeys `margin ~= (ceiling - 0.6735) / |dC/d(eps)|` with `|dC/d(eps)| ~ 0.6`.
The frozen `q = 8` ceiling is `0.673522454`, which is exactly what yields `3.1e-5`. So the
only way to buy more margin is to **raise the ceiling**. Two attempts were made and both
failed: a further block of the `q = 8` optimiser gained `2.8e-7`, and freeing the window
frequencies `omega_j` — the move that paid `7.1e-6` per block at `q = 7` — gained `1.4e-7`
here. The `q = 8` configuration is at a genuine local optimum.

**Structural fact.** Clearing `67.35%` requires `q >= 8`: the deterministic ceilings are
`0.673457890` at `q = 6` and `0.673489246` at `q = 7`. Eight-dimensional local certificates
are therefore unavoidable for this target; they are not an artefact of a bad configuration.

## 4. Two concrete improvements to the bottleneck

The diagnosis "the bottleneck is the verifier's domain pruning, not a wrong candidate" is
correct. Two usable strengthenings, both computed in `domred.py` and `dp.py`:

**(a) Separable span-1 domain reduction.** Span-1 pairs depend on one gap each, so

    F(g) >= sum_r phi_r(g_r),     phi_r(g) = b_r g + s a_{r-1,r} W(g).

With `m_r = min phi_r`, any feasible point obeys `phi_r(g_r) <= eps - sum_{r' != r} m_{r'}`.
At `s = 1`, `eps = 0.0080138` this replaces the one-body box `[0, 31.35] x ... x [0, 31.35]`
by admissible sets that are **bounded away from zero**:

    g1,g8 in [0.9004, 15.4411]   g2,g7 in [0.9415, 9.1485]
    g3,g6 in [0.9453, 7.0241]    g4,g5 in [0.9536, 6.1767]

Per-coordinate shrink about 2.2x, **box-volume reduction 2.083e-03**, and the admissible sets
are disconnected (2, 2, 3, 4, 4, 3, 2, 2 components), giving 2304 natural root components
instead of one giant box. The elimination of the whole "all gaps small" region is the part
that matters most: that is where interval extension is weakest.

**(b) Chain dynamic-programming bound.** Span-2 pairs couple only adjacent gaps, so spans 1
and 2 together form a chain and their joint minimum is computable exactly by DP:

| slope | eps | span-1 separable LB | span<=2 chain-DP LB |
|---|---|---|---|
| `1` | 0.0080138 | 0.0043201 | **0.0052526** |
| `19/20` | 0.0079038 | 0.0043184 | 0.0052016 |
| `21/20` | 0.0081182 | 0.0043218 | 0.0053035 |
| `1/2` | 0.00675 | 0.0043031 | 0.0047429 |

This does not close a certificate on its own (spans 1-2 carry only 4 of the 16 units of
pair weight), but restricted to a box it is a far stronger pruning oracle than interval
extension, at cost `O(q n^2)` in the per-coordinate grid size. This is the recommended
replacement for "raise the node limit".

## 5. Verdict

`67.350048130121891%` is **alive**: no counterexample, exact assembly verified, margins
confirmed at `3e-5` against a strong adversarial search. It is **not closed**: the three
eight-dimensional local certificates remain open, and the imported analytic interface D0 and
the block-averaging step are separate open items.

The realistic next step is (a) + (b) inside a fail-closed directed-rounding verifier — not
more nodes, not a wider margin (there is none to be had), and not a different witness.
