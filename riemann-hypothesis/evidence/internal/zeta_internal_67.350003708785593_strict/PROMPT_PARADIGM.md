# Structured continuation prompt — zeta simple-zero proportion programme
# Author of the work: Libo Wang. Paste verbatim, attach the two archives, then state the task.

---
ROLE
You are continuing an ongoing research programme, not starting one. Prior work is attached.
Never redo closed work; never restate a closed result as new.

OBJECT
kappa = liminf S(T)/N(T), the asymptotic proportion of nontrivial zeta zeros that are simple
and on the critical line. This is NOT a percentage of the Riemann Hypothesis being proved.

EVIDENCE LABELS — every statement carries exactly one
  PROVED          a theorem, proved here or cited precisely
  CERTIFIED       rigorous interval arithmetic with outward rounding, fail-closed
  EXACT           exact arithmetic over Q; no float in any acceptance test
  FLOAT           numerical search; NEVER sufficient to call something true
  IMPORTED        stated, not reproved (currently: the analytic interface D0, and the
                  block-averaging step)
A number with any FLOAT dependency is a CANDIDATE. Never assert acceptance by the
mathematical community; that requires independent review and cannot be self-declared.

CURRENT STATE
  value        : C = 722547711262091300265625000000000/1072825050442925667061714615173641
               = 67.350003708785593%
  witness      : q=8 (d=16, r=9, T=9/8), m=531, eta=1, H > 672167187145431/10^15 (Arb)
  certificates : three local inequalities, all CLOSED — 48/48 shards PROVED, 0 unresolved
                 s=1/2   eps=526/78125     margin 4.725e-5
                 s=19/20 eps=9879/1250000  margin 3.069e-5
                 s=1     eps=20033/2500000 margin 3.064e-5
  benchmark    : the published kernel-verified record is 67.2500703679% (Anthropic 2026,
                 Lean 4 / Zeta23, 0 assumptions, does NOT assume RH), tracked at
                 riemannzeta.fun. Our value exceeds it by +0.0999333409 pp. The ledger's
                 N_0* counts distinct critical-line zeros; simple implies distinct, so a
                 bound on simple-and-on-line is a valid bound for it.
  record origin: 67.2500703679% is exactly sup H(v) over v>=0 on [-1/2,1/2], attained near
                 v(t)=cos(sqrt(2) t) — which is why omega_0=sqrt(2) appears in the window.
                 Our window has H=0.672167187, deliberately BELOW that optimum; the matrix/
                 pressure machinery supplies the further 0.00133285. The gain is the
                 machinery, not a better window.
  now proved   : the finite spectral inequality (B) and the pinching / pressure-charge
                 accounting — see AUDIT.md. Scalar core:
                 min_{n>=0}[(p-n)^2+4n] = 2p-1+Psi(p), exact on both branches.
                 Psi is convex and nonnegative, so pinching gives Delta(M) >= sum_B Delta(M_B)
                 by majorisation; a fixed gap is charged by a window in exactly m-q of the m
                 offsets, giving (m-q)B.
  lean contract: one file proof/Solution.lean, three fixed theorems
                 (candidate_strict_improvement, candidate_critical_line_bound,
                 candidate_critical_line_bound_cumulative); permitted axioms are ONLY
                 propext, Quot.sound, Classical.choice; no sorry, no native_decide, no RH.
                 Upstream ChallengeDeps = anthropics/zeta-23-lean @ 3635e748,
                 lean4 v4.33.0-rc2, mathlib 51e6992e.
  hook point   : Zeta23/ThmD/Functional.lean defines cFun lam v; at lam=1,
                 2 - 1/cFun 1 v = H(v) EXACTLY, so the record is sup_v H(v) — window
                 optimisation with no block term. Attach at thmD_lam_abstract in
                 Zeta23/ThmD/Final.lean; its PaperInputs hypotheses are proved
                 unconditionally upstream. Our contribution replaces Theorem D's
                 Delta(M) >= 0 by Delta(M) >= h(E) + ... .
  L10 DEAD END : do NOT try to port the branch-and-bound proof into Lean. The three
                 certificates are 4.28e8 interval nodes; a submission is replayed by the Lean
                 kernel AND an independent kernel in a credential-free sandbox, and
                 native_decide is excluded by the axiom whitelist. A kernel-checked
                 computation of that size is infeasible. Historical L10 hypothesis (superseded by L11): a
                 SHORT CERTIFICATE was initially sought as (1) rigorous polynomial lower bound for W on the already
                 reduced domain (coordinates confined to about [0.9,15.5], finitely many
                 components after the separable span-1 reduction), (2) an SOS /
                 Positivstellensatz certificate for the resulting polynomial inequality,
                 checkable as a bounded rational computation. Existence at these margins and
                 at manageable degree is OPEN and is the correct next research target.
  L11 Node count is margin-driven. 节点数由余量驱动；先确定真正要超越的阈值，再据此反推可放松的 ε。
      Do not defend decimals that the external threshold does not require. Measure the
      epsilon-to-node curve, choose the target threshold first, then maximize safe relaxation
      subject to a strict exact-rational improvement. The 67.275055959117140% tier is the
      frozen example: 793,374 nodes instead of the historical 4.28e8-scale high-value run.
  refuted      : 67.350352375073% and 67.35006335392536% — explicit counterexamples exist

SHORT-CERTIFICATE TIER — PRIMARY REPRODUCIBLE RELEASE
  value        : C_short = 9484806128780811321/14098548107555200000
               = 67.275055959117140%
  witness      : q=8, m=617, eta=119/125
  certificates : four local inequalities, independently replayed in finalization
                 s=1/2   eps=279/50000       nodes=12,270    PROVED
                 s=19/20 eps=33669/5000000   nodes=212,210   PROVED
                 s=1     eps=34219/5000000   nodes=265,637   PROVED
                 s=21/20 eps=17333/2500000   nodes=303,257   PROVED
                 total=793,374 nodes; all stack_left=0, HARD=0
  exact R      : 1256139843/312500000, attained at E=0
  positioning  : above the 2026-09-17 official riemannzeta.fun displayed record
                 67.2500703679%, but below the separate public 67.3008528% research draft.
                 No accepted Lean submission exists; do not call it an official record.


STRICT 67.350003708785593% TIER — FROZEN CERTIFICATE PATH
  target       : 722547711262091300265625000000000 / 1072825050442925667061714615173641
               = 67.350003708785593%
  witness      : q=8, m=531, eta=1
  local eps    : 1/2 -> 526/78125; 19/20 -> 9879/1250000; 1 -> 20033/2500000
  acceptance   : s=1/2 direct strict B&B + 327/327 independent well boxes PROVED + s=19/20 and s=1 global-complement runs
                 stack_left=0, HARD=0, result=PROVED; exact rational assembly PASS
  verifier     : outward-safe float storage; interval Taylor sin/cos; exact symmetry preflight;
                 fail-closed node cap / terminal boxes.
  claim        : strict finite-dimensional certificate released for independent review;
                 NOT self-declared peer-review acceptance and NOT an accepted Lean entry.

THE CHAIN, IN ONE SCREEN
  D0 (IMPORTED):  S >= H(v) N + Delta(M) - o(N),  H(v) = 2 - (I2+J)/I1^2
  Delta(G) = tr Psi(G),  Psi(t) = (t-1)^2 on [0,2], 2t-3 for t>=2
  Lemma (PROVED): Delta(G) = ||X-U||_F^2 + 2 tr U,  X = G-I,  U = (G-2I)_+
  Matrix theorem (PROVED, general d=2q, r=q+1, T=(q+1)/q, E = 2*sum_{1<=j-i<=q}|G_ij|^2):
      Delta(G) >= h(E);  h(E)=E for E<=T;  h(E)=E-(d/(d+1))(sqrt E - sqrt T)^2 for E>=T
      proof: u^2 <= d D (degree) and u^2 <= tau^2/T (colouring), then minimise in u
  Local certificate C(s) (CERTIFIED): P_loc + s Q_loc >= eps_s on [0,inf)^q
  Span capacity: sum_i a_{i,i+s} = 2 for every span  =>  sum Q_loc <= E
      =>  P + sE >= n eps_s,  p(E) = max(0, max_s(n eps_s - sE)),  n = m-q
  Finite inequality: R = inf_E [h(E) + eta p(E)], attained on a finite kink set (EXACT)
  Averaging (IMPORTED): Delta(M) >= (R/m) S - (eta B (m-q)/m) N,  B = sum b_r = 93/23000
  Conclusion: kappa >= C = (m H - eta B (m-q)) / (m - R)

HARD LESSONS — each was paid for with a wrong result or a stalled week
  L1 Float minima are NOT converged under alternating-gap seeds. True minimisers contain
     multi-level gap structures including a gap near 2.91. Any eps target must come from
     seeds covering {~1.04, ~1.98, ~2.91}^q, enumerated COMPLETELY, not sampled.
  L2 Thin sampling of that family inflates ceilings by about 2e-5 at q>=9. Freeze only at a
     q where the family can be enumerated completely within budget.
  L3 A warm-archive evaluator carries ~1.2e-5 of noise. Never freeze or report a number that
     has not been re-measured by a deterministic, archive-free evaluator.
  L4 The binding vertex sits below T, where h(E)=E is exactly tight. Sharpening h buys
     nothing at the witness. Do not spend effort there.
  L5 In the verifier, a table cell minimum carries error h*|W'|. If that exceeds the margin
     being certified, NO node count can close the proof. Use second-order interpolation with
     a rigorous remainder: W(d) >= W(mid) + W'(mid)d - (1/2)|W''|max d^2.
  L6 A tangent-plane bound gated on positive definiteness almost never fires, because the
     Loewner lower Hessian from min W'' is almost never PD. Use the always-valid alpha-BB
     bound F(c) - sum|dF/dg_k(c)| r_k + (1/2) min(lambda_min,0) sum r_k^2, lambda_min by
     Gershgorin.
  L7 Store box endpoints in double. float imposes a ~1e-6 width floor near coordinates of 10
     and fakes unresolved cells.
  L8 Optimise the WITNESS for proof cost, not for decimals. margin ~ (ceiling-target)/0.6.
     Dropping a slope that costs 1.5e-7 of margin removes an entire eight-dimensional proof.
  L9 A crude untuned probe is weak evidence against a structural lever. q=8 scored 0.673367
     with a heuristic weight guess and looked dead; with a structured seed it reached
     0.673522 in four blocks.

  L12 Separate discovery from trust. A fast historical searcher may propose hard regions, but it
      must not be the final authority. Certify the numerically dangerous wells independently
      with exact-rational interval data, then prove the global complement fail-closed. A box may
      be pruned by a well only after the well checker itself passes. This turns a floating-search
      concern into a small explicit trust boundary and avoids replaying a giant historical tree.

DIRECTIONS ALREADY CLOSED — do not re-spend budget without a new reason
  asymmetric pair weights (+2e-6); raising H; lowering H; adding certified slopes (<=1e-6);
  enlarging the window basis (+2.1e-6/block); freeing window frequencies at q=8 (+1.4e-7).
DIRECTIONS THAT PAID
  random-direction line search; freeing window frequencies at q=7; raising q (but it raises
  certificate dimension — pay knowingly); separable span-1 domain reduction (box volume
  x2e-3); per-node constraint propagation; and L5-L7 above.

WORKING RULES
  Short gates first; design every computation to finish inside one execution window.
  When a route stalls, state the measured rate, compare with the target, and switch.
  Report every tool error, truncated file or self-contradiction immediately and re-derive
  from the last verified state. Prefer widening certification margin over defending a
  prettier decimal.

TASK
<state the single next objective here>
---
