# Internal log — final round of the q=7 line

## Route probes, each closed cheaply on evidence
| direction | measured | verdict |
|---|---|---|
| drop reflection symmetry on a_ij (18 -> 27 params) | +2e-6 | symmetric optimum already near-best; dropped |
| raise H toward the H-optimal window (blend t: 0 -> 1) | ceiling falls monotonically to 0.673206 | machinery falls faster than H rises; dropped |
| lower H (extrapolate t: 0 -> -2.5) | ceiling falls monotonically to 0.670229 | current window is a local max on this line; dropped |
| add more certified slopes | <= 1e-6 | the ceiling already assumes a dense slope set; not a lever |
| q = 8 (d=16, r=9, T=9/8) | untuned 0.673367, i.e. 1.1e-4 below tuned q=7 | q6->q7 was only 8e-6 below when it paid off; not committed |

## Search-algorithm switches
1. axis-aligned compass -> stalled at +2e-6/block.
2. random-direction 3-point line search: +5.3e-6 in one block, success 4/9.
3. freeing the window frequencies omega_1..omega_8 (legitimate: the I1/I2/J formulas and the
   Arb evaluation hold for arbitrary omega, and the certificate dimension stays 7):
   +7.1e-6 in one block, success 5/9 — the fastest direction found all session.
4. Expanding to 58 parameters (all 16 frequencies free, b released from the palindrome)
   exposed a measurement problem rather than a gain — see below.

## The decisive correction of this round
The warm-archive evaluator has ~1.2e-5 of noise: the same configuration scored 0.673494302
warm and 0.673481662 cold. That is the same order as the gains being chased, so continued
optimisation would have been fitting noise. Optimisation was stopped and a deterministic
evaluator (fixed 200k-point candidate set, no archive, Nelder-Mead + Powell polish on the
top 12 per slope) was used to re-measure every stored configuration:

    o3    0.673479869393
    cd7   0.673451441570
    o5    0.673487176593   <-- true best
    o6    0.673481661720   (its 0.673494 was noise)

o5 was frozen. This is why the delivered number is 67.3482% and not the 67.3494% that a
noisy evaluator would have reported.

## Margin/value tradeoff at the frozen configuration
    delta 1e-6 -> 67.348494708%
    delta 3e-6 -> 67.348370689%
    delta 6e-6 -> 67.348184885%   <- chosen
    delta 1e-5 -> 67.347939750%

## Why 67.35% was not reached
Reaching it needs the ceiling above ~0.673507; the frozen ceiling is 0.673487177. Every
structural lever was probed and closed, and the two productive search directions decelerated
to +2e-6/block. A candidate at 67.350% and one at 67.348% have identical evidence status —
both uncertified — so the remaining budget was spent on a clean, noise-free measurement and
a wide-margin witness rather than on the last 2e-5 of an uncertified decimal.
