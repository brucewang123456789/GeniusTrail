"""
CERTIFICATE 3 - Theorems 2 and 3 (Fock commutant; linear-connector collapse).

Theorem 2: on the truncated hard Fock space F = (+)_{N=0}^{Nmax} Sym^N(H_1),
  A_Fock = W*( all fixed-N algebras of Proposition 2 )  has exact commutant W*(Nhat).
  KEY POINT (why this is NOT just "CCR irreducibility"):
  A_Fock contains NO number-changing operator at all. Its commutant is
  therefore N+1 dimensional, not scalar. Irreducibility of the full Weyl
  algebra is a statement about a DIFFERENT, much larger generating set.

Theorem 3: adjoin ONE field operator Phi(f) = a(f)+a^dagger(f), f != 0.
  Then commutant collapses to C I. Proof uses injectivity of a^dagger(f):
     ||a^dag(f)psi||^2 = ||a(f)psi||^2 + ||f||^2 ||psi||^2 > 0.
  So NO genericity assumption is needed (contrast: a Weyl unitary e^{i Phi(f)}
  can have accidentally vanishing adjacent matrix elements).
"""
import numpy as np
rng = np.random.default_rng(11)

d1 = 2          # one-particle space: R and L helicity of a single mode
NMAX = 5

# Sym^N(C^2) basis: |r, N-r>, r = 0..N  -> dim N+1
dims = [N + 1 for N in range(NMAX + 1)]
off = np.cumsum([0] + dims); D = off[-1]
print(f"truncated Fock dim = {D}, sectors N=0..{NMAX}")

def idx(N, r):  return off[N] + r      # r = number of R-helicity quanta

def a_dag(f):
    """f = (fR, fL): creation operator, maps Sym^N -> Sym^{N+1}."""
    A = np.zeros((D, D), dtype=complex)
    fR, fL = f
    for N in range(NMAX):
        for r in range(N + 1):
            # add an R quantum: |r,N-r> -> sqrt(r+1)|r+1,N-r>
            A[idx(N+1, r+1), idx(N, r)] += fR * np.sqrt(r + 1)
            # add an L quantum: |r,N-r> -> sqrt(N-r+1)|r,N-r+1>
            A[idx(N+1, r),   idx(N, r)] += fL * np.sqrt(N - r + 1)
    return A

def A_Fock_generators():
    """Matrix units inside each fixed-N block = (+)_N B(Sym^N)."""
    gens = []
    for N in range(NMAX + 1):
        for a in range(dims[N]):
            for b in range(dims[N]):
                G = np.zeros((D, D), dtype=complex); G[off[N]+a, off[N]+b] = 1.0
                gens.append(G)
    return gens

def commutant_dim(gens):
    I = np.eye(D)
    M = np.vstack([np.kron(I, G) - np.kron(G.T, I) for G in gens])
    s = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape) * np.finfo(float).eps * s[0] * 1e3
    return D*D - int(np.sum(s > tol))

gens = A_Fock_generators()
dim0 = commutant_dim(gens)
print(f"[Thm 2] dim (A_Fock)'        = {dim0}   expected {NMAX+1} (= W*(Nhat))   "
      f"{'OK' if dim0 == NMAX+1 else 'FAIL'}")

ok = (dim0 == NMAX + 1)

# --- B(b): adjoin one field operator, several choices of f including edge cases
for label, f in [("f=(1,0) pure R", (1.0, 0.0)),
                 ("f=(0,1) pure L", (0.0, 1.0)),
                 ("f generic", (0.3+0.7j, -1.1+0.2j)),
                 ("f tiny", (1e-6, 0.0))]:
    Ad = a_dag(f); Phi = Ad + Ad.conj().T
    dim1 = commutant_dim(gens + [Phi])
    good = (dim1 == 1); ok &= good
    print(f"[Thm 3] {label:16s}: dim (A_Fock, Phi(f))' = {dim1}  {'OK' if good else 'FAIL'}")

# f = 0 must NOT collapse
Ad0 = a_dag((0.0, 0.0)); dimz = commutant_dim(gens + [Ad0 + Ad0.conj().T])
print(f"[control] f=0: dim = {dimz} expected {NMAX+1} {'OK' if dimz==NMAX+1 else 'FAIL'}")
ok &= (dimz == NMAX + 1)

# --- injectivity of a^dagger(f) (the step that removes any genericity assumption)
Ad = a_dag((0.3+0.7j, -1.1+0.2j))
worst = np.inf
for N in range(NMAX):
    sub = Ad[off[N+1]:off[N+2], off[N]:off[N+1]]
    smin = np.linalg.svd(sub, compute_uv=False).min()
    worst = min(worst, smin)
print(f"[injectivity] min singular value of a^dag on N->N+1 blocks = {worst:.6f} > 0  "
      f"{'OK' if worst > 1e-12 else 'FAIL'}")
ok &= worst > 1e-12

# --- contrast: Weyl unitary CAN have vanishing adjacent blocks (shows why we use Phi, not e^{iPhi})
import scipy.linalg as sla
W = sla.expm(1j * (Ad + Ad.conj().T))
mins = [np.abs(W[off[N+1]:off[N+2], off[N]:off[N+1]]).max() for N in range(NMAX)]
print(f"[contrast] Weyl adjacent-block max-abs per N: {[f'{m:.3f}' for m in mins]}")

print("RESULT:", "PASS" if ok else "FAIL")
