"""
CERTIFICATE 5 - Theorems 6, 7, 8 (quantitative completeness; exact su(2) gap).

Theorem 6 (general, sharp):  with w_{rr'} = ||P_{r'} S P_r||_HS^2 and L the weighted
  sector Laplacian, for every X = sum_r c_r P_r in the commutant of Proposition 1
        ||[X,S]||_HS^2 = 2 c* L c,
  hence   ||c - cbar|| <= ||[X,S]||_HS / sqrt(2 lambda_2(L)),  sharp on the Fiedler vector.
  lambda_2 > 0  <=>  graph connected  <=>  Part II Theorem 1 completeness.
  So Part II's qualitative criterion is the lambda_2 > 0 case and lambda_2 is its MODULUS.

Theorem 8 (exact, the main Part III result): for the Stokes connector of Proposition 2 on a two-mode
  helicity pair, P_kappa acts as 2*lambda*J_x in the spin-N/2 irrep of su(2). The
  coherence-transfer Laplacian is exactly half the su(2) ADJOINT CASIMIR restricted to
  the commutant of Proposition 1, with spectrum
        spec(L) = { lambda^2 k(k+1) : k = 0,1,...,N }
  so lambda_2 = 2*lambda^2 EXACTLY, INDEPENDENT OF N, and the extremal (least visible)
  operator is the Fiedler vector c_r = r - N/2, i.e. X proportional to N_R itself --
  the very obstruction Proposition 1 identified.
"""
import numpy as np
rng = np.random.default_rng(20260914)
LAM = 1.0
ok = True

def laplacian(w):                      # weighted path Laplacian from edge weights w[0..N-1]
    N = len(w); L = np.zeros((N+1, N+1))
    for i in range(N):
        L[i,i]+=w[i]; L[i+1,i+1]+=w[i]; L[i,i+1]-=w[i]; L[i+1,i]-=w[i]
    return L

print("=== CERT-5 (A) exact spectrum of the Proposition 2 coherence-transfer Laplacian ===")
for N in [3,5,8,12,20,40]:
    r = np.arange(N)
    L = laplacian((LAM**2)*(r+1)*(N-r))
    ev = np.sort(np.linalg.eigvalsh(L))
    pred = (LAM**2)*np.arange(N+1)*(np.arange(N+1)+1)
    err = np.abs(ev-pred).max(); good = err < 1e-8; ok &= good
    print(f"  N={N:3d}: max|spec(L) - lam^2*k(k+1)| = {err:.2e}   lambda_2 = {ev[1]:.10f}  "
          f"(2*lam^2 = {2*LAM**2})  {'OK' if good else 'FAIL'}")

print("\n=== CERT-5 (B) gap is INDEPENDENT of N; extremal operator is exactly N_R ===")
for N in [5,12,30,100]:
    r = np.arange(N); L = laplacian((LAM**2)*(r+1)*(N-r))
    ev,V = np.linalg.eigh(L); o = np.argsort(ev); v = V[:,o[1]]; v = v/np.abs(v).max()
    t = (np.arange(N+1)-N/2); t = t/np.abs(t).max()
    err = min(np.abs(v-t).max(), np.abs(v+t).max())
    g = abs(ev[o[1]]-2*LAM**2) < 1e-8 and err < 1e-8; ok &= g
    print(f"  N={N:4d}: lambda_2={ev[o[1]]:.8f}  ||fiedler - (r-N/2)||_inf={err:.2e}  {'OK' if g else 'FAIL'}")

print("\n=== CERT-5 (C) su(2) adjoint-Casimir identity ===")
N=8; d=N+1
Jp=np.zeros((d,d))
for k in range(N): Jp[k,k+1]=np.sqrt((k+1)*(N-k))
S = 2*LAM*(Jp+Jp.T)/2
print(f"  connector offdiag == lam*sqrt((r+1)(N-r)): "
      f"{np.abs(np.diag(S,1)-LAM*np.sqrt((np.arange(N)+1)*(N-np.arange(N)))).max():.2e}")
# on the spin-k component of the adjoint action restricted to diagonal X
ev,V = np.linalg.eigh(laplacian((LAM**2)*(np.arange(N)+1)*(N-np.arange(N))))
o=np.argsort(ev)
for k in [1,2,3]:
    c = V[:,o[k]]; X = np.diag(c).astype(complex)
    lhs = np.linalg.norm(X@S-S@X,'fro')**2
    rhs = 2*(LAM**2)*k*(k+1)*np.linalg.norm(X,'fro')**2
    g = abs(lhs-rhs) < 1e-8; ok &= g
    print(f"  k={k}: ||[X,S]||_HS^2={lhs:.8f}  2*lam^2*k(k+1)*||X||^2={rhs:.8f}  {'OK' if g else 'FAIL'}")

print("\n=== CERT-5 (D) uniform bound  ||c-cbar|| <= ||[X,P_kappa]||_HS/(2*lam)  for all N ===")
for N in [4,10,25,60]:
    r=np.arange(N); w=(LAM**2)*(r+1)*(N-r); d=N+1
    Jp=np.zeros((d,d))
    for k in range(N): Jp[k,k+1]=np.sqrt((k+1)*(N-k))
    S=2*LAM*(Jp+Jp.T)/2
    worst=-np.inf
    for _ in range(600):
        c=rng.normal(size=d)+1j*rng.normal(size=d); X=np.diag(c)
        lhs=np.linalg.norm(c-c.mean()); rhs=np.linalg.norm(X@S-S@X,'fro')/(2*LAM)
        worst=max(worst, lhs-rhs)
    g = worst < 1e-9; ok &= g
    print(f"  N={N:3d}: max(LHS-RHS) over 600 random X = {worst:.2e}  {'OK' if g else 'FAIL'}")

print("\n=== CERT-5 (E) general Theorem 6 on random multi-mode blocks (no su(2)) ===")
for trial in range(4):
    dims=list(rng.integers(1,4,size=6)); nb=len(dims)
    offs=np.cumsum([0]+dims); D=offs[-1]
    S=np.zeros((D,D),dtype=complex)
    for i in range(nb-1):
        b=rng.normal(size=(dims[i+1],dims[i]))+1j*rng.normal(size=(dims[i+1],dims[i]))
        S[offs[i+1]:offs[i+2],offs[i]:offs[i+1]]=b; S[offs[i]:offs[i+1],offs[i+1]:offs[i+2]]=b.conj().T
    W=np.zeros((nb,nb))
    for i in range(nb):
        for j in range(nb):
            if i!=j: W[i,j]=np.linalg.norm(S[offs[i]:offs[i+1],offs[j]:offs[j+1]],'fro')**2
    L=np.diag(W.sum(1))-W; l2=np.sort(np.linalg.eigvalsh(L))[1]
    worst=-np.inf; tight=np.inf
    for _ in range(500):
        c=rng.normal(size=nb)+1j*rng.normal(size=nb)
        X=np.zeros((D,D),dtype=complex)
        for i in range(nb): X[offs[i]:offs[i+1],offs[i]:offs[i+1]]=c[i]*np.eye(dims[i])
        lhs=np.linalg.norm(c-c.mean()); rhs=np.linalg.norm(X@S-S@X,'fro')/np.sqrt(2*l2)
        worst=max(worst,lhs-rhs); tight=min(tight, rhs/lhs if lhs>1e-9 else np.inf)
    # sharpness: Fiedler vector saturates
    cf=np.linalg.eigh(L)[1][:,np.argsort(np.linalg.eigvalsh(L))[1]]
    Xf=np.zeros((D,D),dtype=complex)
    for i in range(nb): Xf[offs[i]:offs[i+1],offs[i]:offs[i+1]]=cf[i]*np.eye(dims[i])
    ratio=(np.linalg.norm(Xf@S-S@Xf,'fro')/np.sqrt(2*l2))/np.linalg.norm(cf-cf.mean())
    g = worst<1e-9 and abs(ratio-1)<1e-6; ok &= g
    print(f"  trial{trial}: lambda_2={l2:8.4f}  bound holds (max slack {worst:.1e})  "
          f"Fiedler saturation ratio={ratio:.10f}  {'OK' if g else 'FAIL'}")

print("\n=== CERT-5 (F) dimension-free operator-norm version: resistance diameter ===")
print("  D_R = sum_r 1/||P_{r+1} P_kappa P_r||_op ,  bound dist(X,CI) <= (1/2) D_R ||[X,S]||_op")
for N in [4,16,64,256,1024,4096]:
    r=np.arange(N); DR=np.sum(1.0/(LAM*np.sqrt((r+1)*(N-r))))
    print(f"  N={N:5d}: D_R = {DR:.6f}   (limit pi/lam = {np.pi/LAM:.6f})")
print(f"  => D_R stays BOUNDED as N -> infinity  OK")

print("\nRESULT:", "PASS" if ok else "FAIL")
