# Refutation of two previously circulated q=7 candidates

Both candidates below are **false**. Each is refuted by an explicit numerical
witness that anyone can re-evaluate in seconds with `refute.py`.

## 1. Candidate 67.350352375073%  (six-slope q=7 package)
Claimed local targets, both violated under the package's own weights `cfg7_rat.json`:

| slope | claimed epsilon_s | true value at witness | verdict |
|---|---|---|---|
| 1/2  | 16899/2500000 = 0.0067596 | 0.006703873 | **FALSE** |
| 1    | 80841/10^7   = 0.0080841  | 0.008033578 | **FALSE** |
| 11/10| 83033/10^7   = 0.0083033  | 0.008284539 | **FALSE** |

## 2. Candidate 67.35006335392536%  (q=7 "Reduced-3" package)
Same weights, three slopes. One target is violated:

| slope | claimed epsilon_s | true value at witness | verdict |
|---|---|---|---|
| 1/2   | 67/10000 = 0.0067        | 0.006703873 | holds (+3.9e-6) |
| 21/20 | 40981/5000000 = 0.0081962 | 0.008128788 | **FALSE** |
| 11/10 | 83003/10^7 = 0.0083003    | 0.008284539 | **FALSE** |

Explicit witness for slope 11/10 (gap vector g_1..g_7):

    g = (1.99140578, 1.04683263, 1.98561048, 1.98374147,
         1.04573175, 1.97741439, 1.04170216)
    P_loc + (11/10) Q_loc = 0.008284538877  <  0.0083003

## Root cause
The epsilon targets were derived from multistart float minimisation whose seed
family consisted of alternating gaps near 1.04 and 1.975 only. The true minimisers
of several slopes contain a **large gap near 2.91**, a structural family absent
from that seed set. Once three-level seeds {1.04, 1.975, 2.915} (all 3^7 = 2187
combinations) are included, the minima drop by up to 5e-5 and the targets fail.

**Any epsilon target in this programme must be produced by a seed family that
includes multi-level gap structures, and must be given a safety margin far larger
than the 3e-7 class previously used.**
