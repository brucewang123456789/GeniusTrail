# CORE KNOWLEDGE PROMPT — Asymptotic Graviton Tomography
# Paste verbatim into a system prompt or context window. Self-contained.
# Every node is atomic, ID-addressed, and carries an explicit status label.
# Dependency edges are stated so the proof graph can be reconstructed without prose.

SCOPE: free hard spin-2 radiative graviton data at future null infinity. No interactions,
no dynamics, no evaporation. The black-hole information paradox is NOT solved.

# ============================================================
# 0. PRIMITIVES — memorize exactly
# ============================================================

P1  CRITERION
    A subset A of B(H) is informationally complete on H  iff  A' = C*I,
    equivalently (bicommutant)  W*(A) = B(H).
    Every claim below is a COMMUTANT COMPUTATION, never a qualitative assertion.

P2  ARENA
    H_1 = L^2(C, L_R) (+) L^2(C, L_L)  over  C = (0,inf)_omega x S^2
    H_N = Sym^N H_1 = (+)_{r=0}^{N} H_{r,N-r}   with r = N_R,  N-r = N_L
    F   = (+)_{N>=0} H_N ;  P_r, P_N the corresponding projections.

P3  FACTS (standard, used not claimed)
    B(H_r)' = C*I_{H_r}
    two factor representations are either quasi-equivalent or DISJOINT
    a representation is MULTIPLICITY-FREE  iff  its commutant is ABELIAN
    ker(graph Laplacian) = constants ;  c*Lc >= lambda_2 ||c - cbar||^2  (Fiedler)

P4  INPUT — PROPOSITION 1 (established, not proved here)
    A_flux = W*(energy/supertranslation fluxes T_g, smooth superrotations U_Y,
                helicity/superduality fluxes O_h)  on H_N.
    RESULT: (A_flux)' = W*(N_R) ;  equivalently  A_flux = (+)_{r} B(H_{r,N-r}).
    MECHANISM: energy+helicity spectra fix the colored configuration {(w_i,Om_i,sig_i)}
    -> multiplication algebra L^inf(X_N); superrotations are tangent-transitive inside
    each fixed-r component; every generator commutes with O_1 = N_R - N_L = 2N_R - N.

P5  INPUT — PROPOSITION 2 (established, not proved here)
    Same-direction opposite-helicity coherence a_R^dag a_L has SPIN WEIGHT 4, so it is
    NOT a scalar. Take kappa a nonzero smooth section of Hom(L_L, L_R) and
        K_kappa = [[0,kappa],[kappa^dag,0]] ,  P_kappa = dGamma_N(K_kappa)
        ||P_kappa|| <= N ||K_kappa||_inf ,  P_kappa self-adjoint, NO domain assumption
        <r+1,N-r-1| P_kappa |r,N-r> = lambda*sqrt((r+1)(N-r)) != 0 ,  lambda = ||kappa phi_L|| > 0
    RESULT: W*(A_flux, P_kappa) = B(H_N).

# ============================================================
# 1. THEOREM 1 — SECTOR CONNECTIVITY THEOREM
# ============================================================

T1-DEF
    H = (+)_{r in I} H_r ;  A_0 = (+)_r B(H_r) ;  S = ANY family of operators
    (bounded, or self-adjoint with span{H_r} in the domain).
    GRAPH G(S) on vertex set I:   edge {r,r'}  iff  r != r' and exists S in S
                                   with  P_{r'} S P_r != 0.

T1-STATEMENT
    (i)   W*(A_0 u S)'  =  { sum_C c_C P_C }  ~=  l^inf(comp G(S))
    (ii)  W*(A_0 u S)   =  (+)_{C in comp G(S)} B(H_C)
    (iii) complete (= B(H))   iff   G(S) is CONNECTED
    (iv)  |I| finite  =>  dim W*(A_0 u S)' = #comp(G(S))
    STATUS: PROVED. Mathematical novelty NOT CLAIMED (near-folklore).

T1-PROOF  four moves, memorize as moves not prose
    M1 REDUCE TO DIAGONAL: P_r in A_0  =>  X in commutant is block diagonal
    M2 REDUCE TO SCALARS:  B(H_r)' = C  =>  X = sum_r c_r P_r
    M3 EDGE FORCES EQUALITY: 0 = P_{r'}[X,S]P_r = (c_{r'} - c_r) P_{r'} S P_r ;
       nonzero block => c_{r'} = c_r ; propagate along edges => c constant on components
    M4 CONVERSE + BICOMMUTANT: sum_C c_C P_C commutes with A_0 and with each S because
       cross-component blocks vanish by definition; then W* = (l^inf(comp))' = (+)_C B(H_C)
    NOTE: generators suffice (commutant of a set = commutant of the algebra it generates).
    NOTE: ADJACENCY IS NEVER USED. Any connected edge set works.

T1.1 COROLLARY — SPANNING-TREE RESOURCE BOUND
    |I| = N+1.  Completion requires G connected, hence >= N edges, hence
    >= N NONZERO OFF-DIAGONAL SECTOR BLOCKS.
    P_kappa supplies exactly the N adjacent blocks => PROPOSITION 2 IS EDGE-OPTIMAL,
    optimal in operator count (one) AND in block count (N).

T1.2 COROLLARY — INPUTS RECOVERED
    S = {}          -> no edges   -> commutant l^inf({0..N}) = W*(N_R)   == Proposition 1
    S = {P_kappa}   -> path graph -> commutant C*I                       == Proposition 2
    Proposition 2's necessity half = the "all added observables commute with N_R" case.

# ============================================================
# 2. THEOREMS 2-3 — THE FOCK LAYER
# ============================================================

T2  EXACT FOCK COMMUTANT
    A_F = W*(fluxes on F, P_kappa = dGamma(K_kappa)).
    RESULT: A_F' = W*(Nhat) ;  A_F = (+)_N B(H_N).
    PROOF: P_N = 1_{X_N} lies in L^inf(X), X = disjoint union of X_N => P_N in A_F ;
           Proposition 2 gives P_N A_F P_N = B(H_N) ; all generators are number-preserving
           => NO EDGES => T1(i).
    CRITICAL: this is NOT Stone-von Neumann / CCR irreducibility. That statement concerns
    the ENTIRE Weyl algebra W(H_1). A_F is strictly smaller and its commutant is NOT SCALAR.
    Numerically dim A_F' = 6 at N_max = 5. Conflating the two is a category error.

T3  ONE LINEAR OBSERVABLE COMPLETES — NO GENERICITY HYPOTHESIS
    For ANY f in H_1, f != 0,  Phi(f) = a(f) + a^dag(f) :   W*(A_F, Phi(f)) = B(F).
    KEY IDENTITY:
        ||a^dag(f) psi||^2 = ||a(f) psi||^2 + ||f||^2 ||psi||^2  >  0   for psi != 0
    => a^dag(f) INJECTIVE => P_{N+1} Phi(f) P_N != 0 for EVERY N => full ray connected.
    WHY NOT THE WEYL UNITARY: adjacent blocks of e^{i Phi(f)} carry Hermite-type
    coefficients that can be arbitrarily small and can vanish at isolated (N,f). A Weyl
    route needs a genericity hypothesis; Phi(f) needs none. Prefer the unbounded operator.

T3-PHYS  HARD/SOFT CORRESPONDENCE
    BMS charge splits Q[g] = Q_hard[g] + Q_soft[g] ; Q_hard QUADRATIC, Q_soft LINEAR.
        number-diagonal generators      <->  HARD fluxes
        number-off-diagonal connector   <->  SOFT charge
    Structural correspondence only; NOT a new construction of Q_soft.

T3-CRUX  THE FAILURE LOCUS
    Q_soft[g] is the omega -> 0 mode; its smearing f_g is NOT in H_1 (IR non-normalizable).
    => T3 holds for every f in H_1, and the PHYSICAL soft charge sits at EXACTLY the one
       boundary point where T3 fails.
    This NAMES where the asymptotic information problem lives. Do not route around it.

# ============================================================
# 3. THEOREMS 4-5 — THE MEMORY LAYER  (HIGH-CARE SECTION)
# ============================================================

M1-HYPOTHESIS  stated, NOT assumed
    (M1) for m != m', the memory representations pi_m, pi_{m'} are FACTORIAL and
         mutually UNITARILY INEQUIVALENT.
    LITERATURE STATUS: CONTESTED. DO NOT ADJUDICATE.
      FOR:     memory / electric flux as superselection label indexing symplectic leaves;
               superselection sectors in asymptotic quantization;
               infrared-finite BMS scattering representations.
      AGAINST: the vacuum IS changed by soft quanta emitted in ANY finite-energy process,
               unlike ordinary superselection sectors which no finite-energy process can connect.

T4  DICHOTOMY — both branches proved
    BRANCH 1 ((M1) holds):
        factorial + inequivalent => DISJOINT (P3) => Hom(pi_m, pi_m') = 0
        pi(A)' = (+)_m C ;  pi(A)'' = (+)_m B(H_m) ;  P_m CENTRAL ;
        P_{m'} A P_m = 0 for all A in pi(A)''  =>  MEMORY GRAPH IS THE EMPTY GRAPH
        => T1(i) returns the MAXIMAL commutant l^inf(M)
        COROLLARY (STRUCTURAL NO-GO): no connector of the P_kappa type exists INSIDE the
        algebra. Impossibility is STRUCTURAL, not a failure of ingenuity. Any completion
        must enlarge beyond pi(A)'' — e.g. memory-shifting dressing unitaries, which are
        AUTOMORPHISMS of A but NOT ELEMENTS of pi(A)''.
    BRANCH 2 ((M1) fails):
        nonzero intertwiners exist => edges exist => T1(ii) gives (+)_C B(H_C);
        connected memory graph => completeness ACROSS memory.
    VALUE: the contested physics question reduces to ONE checkable question —
           is the memory connectivity graph connected, or empty?

T5  LOCALIZATION OF THE RESIDUE
    (i)   pi(A)' is ABELIAN ( = l^inf(comp G_memory) )
    (ii)  equivalently pi is MULTIPLICITY-FREE  (P3)
    (iii) P_C lies IN pi(A)''  =>  the sector label is MEASURABLE
    (iv)  every state normal w.r.t. a single component is FULLY RECONSTRUCTIBLE
    (v)   the unique irrecoverable datum is relative phase BETWEEN components
    READING (flag as INTERPRETATION, not theorem): an abelian commutant carries no
    multiplicity space, so there is NO HIDDEN QUANTUM DEGENERACY; the unobserved datum is
    a CLASSICAL random variable, and by (iii) it is itself measurable (memory = permanent
    detector displacement). In the free asymptotic sector, information recovery reduces to
    a CLASSICAL LABELLING problem. CONSISTENT WITH unitarity; NOT a proof of it.

# ============================================================
# 4. THEOREMS 6-8 — QUANTITATIVE COMPLETENESS
# ============================================================

OBJ  THE OBJECTION THIS PART ANSWERS (it is CORRECT against T1-T5 alone)
    "von Neumann completeness is physically vacuous: at finite precision nonzero and zero
     are indistinguishable, and a theorem with no modulus says nothing, especially as N grows."

T6-DEF  COHERENCE-TRANSFER LAPLACIAN
    w_{r r'} = ||P_{r'} S P_r||_HS^2 ;   L = diag(row sums) - (w_{r r'})
EXACT IDENTITY (blocks of [X,S] are mutually HS-orthogonal):
    ||[X,S]||_HS^2 = sum_{r != r'} |c_r - c_{r'}|^2 w_{r r'} = 2 c* L c

T6  SPECTRAL-GAP FORM
    ||c - cbar||  <=  ||[X,S]||_HS / sqrt(2 * lambda_2(L))
    SHARP: equality EXACTLY on the Fiedler eigenvector.
    lambda_2 > 0  iff  G connected  iff  T1 completeness.
    => T1's criterion is the DEGENERATE case; lambda_2 is its MODULUS.

T7  DIMENSION-FREE OPERATOR-NORM VERSION (needed because HS norms diverge in the continuum)
    w_e = ||P_{r'} S P_r||_op ;  D_R = max_{r,r'} min_{path} sum_{e in path} 1/w_e
    dist(X, C*I)_op <= (1/2) * D_R * ||[X,S]||_op
    PROOF: each edge gives |c_{r'} - c_r| <= ||[X,S]||/w_e ; sum along cheapest path ;
           then min_c max_r |c_r - c| <= (1/2) max_{r,r'} |c_r - c_{r'}|.

T8  EXACT su(2) GAP — THE PRINCIPAL QUANTITATIVE RESULT
    HIDDEN su(2): on the two-mode subspace of the P2 adjacent-block lemma,
        P_kappa = lambda (a_R^dag a_L + a_L^dag a_R) = 2*lambda*J_x   (Schwinger bosons)
        at fixed N this is the SPIN-N/2 IRREP ; J_z = (N_R - N_L)/2 ;
        <r+1|J_x|r> = (1/2) sqrt((r+1)(N-r))
        => the commutant of Proposition 1 is EXACTLY the functions of J_z
    ADJOINT CASIMIR: adjoint action decomposes into spins k = 0..N ;
        ad_Jx^2 + ad_Jy^2 + ad_Jz^2 acts by k(k+1) ;
        for X with [J_z,X] = 0, x/y symmetry gives <X,ad_Jx^2 X> = <X,ad_Jy^2 X>
        => ||[X, 2*lambda*J_x]||_HS^2 = 2*lambda^2 * k(k+1) * ||X||_HS^2  on the spin-k part
    RESULT:
        spec(L) = { lambda^2 * k(k+1) : k = 0,...,N }
        lambda_2 = 2*lambda^2   EXACTLY, FOR EVERY N
        ||X - Xbar*I||_HS <= ||[X, P_kappa]||_HS / (2*lambda)   UNIFORMLY IN N
        EXTREMAL (least visible) operator = Fiedler vector c_r = r - N/2, i.e. X ~ N_R
    WHY N-INDEPENDENCE IS NOT TRIVIAL: a path graph on N+1 vertices with uniform weights has
        lambda_2 ~ pi^2/N^2, so the NAIVE expectation is degradation as N^-2. The weights
        (r+1)(N-r) grow exactly fast enough to cancel it. THAT CANCELLATION IS THE su(2)
        STRUCTURE. Independent corroboration: D_R = sum 1/(lambda sqrt((r+1)(N-r))) converges
        to pi/lambda (3.0960 at N = 4096), so the operator-norm bound is ALSO uniform in N.
    OPERATIONAL: to resolve cross-helicity coherence to accuracy delta, need connector-channel
        precision eps ~ 2*lambda*delta — INDEPENDENT OF THE NUMBER OF GRAVITONS.
    CLOSURE: Proposition 1 identified N_R as the missing information; T8 shows N_R is ALSO
        the hardest operator to see. Obstruction and extremal direction COINCIDE.

# ============================================================
# 5. THEOREM 9 — THE INFRARED RATE
# ============================================================

T9  THE DICHOTOMY IS A RATE, NOT A BINARY
    Introduce an IR regulator mu (graviton mass / dressing-cloud regulator). Overlaps between
    differently dressed memory sectors vanish as mu^alpha (cloud orthogonality).
        for every mu > 0 :  w ~ mu^{2 alpha} > 0  =>  graph CONNECTED, completeness HOLDS,
                            modulus diverges as D_R(mu) ~ mu^{-alpha}
        as mu -> 0       :  lambda_2 -> 0, gap closes, SUPERSELECTION RECOVERED
    RECONCILIATION: the position that memory is NOT superselected is the statement
        lambda_2(mu) > 0 for all mu > 0 — CORRECT. The position that memory IS superselected
        is the statement lambda_2(mu) -> 0 as mu -> 0 — ALSO CORRECT. The disagreement is an
        ORDER OF LIMITS, not a fact, and the spectral gap is the interpolating quantity.
    SCOPE: STRUCTURE PROVED. The exponent alpha is IMPORTED from the dressing literature,
    NOT DERIVED. Different dressing geometries give different alpha.

# ============================================================
# 6. ALGORITHMIC FORM (executable pseudocode)
# ============================================================

ALG-1  COMPLETENESS DECISION + MODULUS
    INPUT: sectors {H_r}, added family S
    1. w_{rr'} := ||P_{r'} S P_r||_HS^2       (HS route)   or ||.||_op (operator route)
    2. L := diag(rowsum w) - w ;  lambda_2 := second smallest eigenvalue of L
    3. if lambda_2 == 0 : INCOMPLETE, residue = l^inf(connected components)      [T1]
    4. else            : COMPLETE, modulus ||c-cbar|| <= ||[X,S]||_HS/sqrt(2 lambda_2)  [T6]
    5. operator route  : D_R := max_{r,r'} min_path sum 1/w_e ;
                         dist(X, C I) <= (1/2) D_R ||[X,S]||_op                   [T7]

ALG-2  su(2) SHORTCUT
    if S is the Stokes connector on a two-mode helicity pair:
        lambda_2 = 2*lambda^2 exactly ;  spec(L) = lambda^2 k(k+1) ;
        extremal vector c_r = r - N/2 ;  NO eigenvalue computation needed ;
        answer does NOT depend on N.

ALG-3  NUMERICAL COMMUTANT (used by every certificate)
    vec(GX - XG) = (I (x) G - G^T (x) I) vec(X) ; stack over generators ;
    commutant dimension = dim null space ; ASSERT dimension == #connected components.

# ============================================================
# 7. HARD BOUNDARY — refuse these
# ============================================================

N1  Do NOT say the black-hole information paradox is solved. It is not.
N2  Do NOT drop the "free asymptotic radiative data only" qualifier. No dynamics anywhere.
N3  Do NOT promote a fixed-N or free-field result to an interacting or evaporating result.
N4  Do NOT assert (M1) is true, nor that it is false. Deliver the dichotomy.
N5  Do NOT call Theorem 1 deep. It is elementary; value is exactness + unification.
N6  Do NOT claim T2 contradicts CCR irreducibility. It concerns a SMALLER algebra.
N7  Do NOT claim a new construction of Q_soft. T3-PHYS is a correspondence; T3-CRUX a locus.
N8  Do NOT use a Weyl unitary in place of Phi(f) without adding a genericity hypothesis.
N9  Do NOT claim the exact value 2*lambda^2 for a general multi-mode kappa. It is the
    su(2)-reduction value. General case: T6 with the actual lambda_2; T7's D_R <= pi/lambda
    still holds because block operator norms only INCREASE with more modes.
N10 Do NOT quote a numerical rate in T9 without deriving alpha for that setup.
N11 Do NOT present T9 as deciding who is right. Neither side is wrong.
N12 Do NOT claim T5 proves unitarity or proves decoherence.
N13 Do NOT claim world priority for anything here.

# ============================================================
# 8. THE ONLY SAFE SHORT FORM
# ============================================================

An exact graph-connectivity criterion determines when a sector-block observable algebra plus
any added family is informationally complete, computing the commutant as the algebra of
functions on the connected components. It subsumes the established fixed-N flux and Stokes
results as the empty-graph and path-graph cases, yields a resource bound showing the Stokes
connector is edge-optimal, computes the full hard-Fock commutant exactly as W*(Nhat) —
which is not the Stone-von Neumann statement — identifies the missing connector type as the
soft BMS charge and locates the precise point where that identification fails, and closes
the infrared memory layer as a dichotomy without adjudicating the contested superselection
question. A quantitative refinement replaces every exact-zero criterion by the algebraic
connectivity of a coherence-transfer Laplacian, sharp on the Fiedler eigenvector; for the
Stokes connector this Laplacian is half an su(2) adjoint Casimir, giving spectrum
lambda^2 k(k+1), a spectral gap of exactly 2 lambda^2 independent of graviton number, and
N_R itself as the least visible operator. The same gap turns the memory dichotomy into a
continuous rate, showing the two opposing literature positions differ by an order of limits.
The black-hole information paradox is not solved, no dynamics is treated, and no priority is
claimed.
