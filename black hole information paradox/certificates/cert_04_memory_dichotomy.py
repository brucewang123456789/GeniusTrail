"""
CERTIFICATE 4 - Theorems 4 and 5 (memory dichotomy; localization).

provably CANNOT be repeated at the infrared/memory level.

Assumption (M1), taken verbatim from the established infrared literature:
  distinct supertranslation-memory values m label GNS representations pi_m of
  the asymptotic radiative algebra A that are (i) factorial (Fock-like, here
  irreducible) and (ii) mutually unitarily inequivalent.

Lemma (M2): factorial + inequivalent  =>  DISJOINT (zero intertwiner space).
Theorem 4: in pi = (+)_m pi_m,
   pi(A)'  = (+)_m C          (memory projections are CENTRAL)
   pi(A)'' = (+)_m B(H_m)
so for EVERY A in the algebra and m != m':  P_{m'} A P_m = 0.
In the Theorem 1 graph language: the memory graph has NO EDGES AT ALL.
Hence no connector of the Proposition 2 type exists inside the algebra. This is a structural
impossibility, not a failure of ingenuity.

Theorem 5 (relative completeness): P_m lies IN pi(A)'' , so memory is a
measurable classical label; and each block is full B(H_m). Therefore every
state normal in a fixed memory sector is completely reconstructible.
The ONLY irreducible residue is coherence ACROSS memory sectors.
"""
import numpy as np
rng = np.random.default_rng(3)

def commutant_basis(gens, D):
    I=np.eye(D); M=np.vstack([np.kron(I,G)-np.kron(G.T,I) for G in gens])
    u,s,vh=np.linalg.svd(M); tol=max(M.shape)*np.finfo(float).eps*s[0]*1e3
    ns=vh[int(np.sum(s>tol)):].conj().T
    return [ns[:,j].reshape(D,D) for j in range(ns.shape[1])]

print("=== CERT-4 memory disjointness no-go ===")
ok=True
# model: 3 memory sectors carrying INEQUIVALENT irreps of a common algebra.
# Use irreps of dimension 2,3,4 of su(2) (spin 1/2, 1, 3/2) -> pairwise inequivalent.
def su2(j2):
    d=j2+1; m=np.array([j2/2-k for k in range(d)])
    Jz=np.diag(m)
    Jp=np.zeros((d,d))
    for k in range(1,d):
        mm=m[k]; Jp[k-1,k]=np.sqrt((j2/2)*(j2/2+1)-mm*(mm+1))
    Jx=(Jp+Jp.T)/2; Jy=(Jp-Jp.T)/(2j)
    return [Jx,Jy,Jz]

reps=[su2(1),su2(2),su2(3)]; dims=[2,3,4]
off=np.cumsum([0]+dims); D=off[-1]
gens=[]
for k in range(3):
    G=np.zeros((D,D),dtype=complex)
    for s in range(3):
        G[off[s]:off[s+1],off[s]:off[s+1]]=reps[s][k]
    gens.append(G)

basis=commutant_basis(gens,D)
print(f"  dim pi(A)' = {len(basis)}  expected 3 (= number of memory sectors)  "
      f"{'OK' if len(basis)==3 else 'FAIL'}"); ok&= len(basis)==3

# every commutant element must be block-scalar => memory projections are central
maxoff=0.0
for X in basis:
    for s in range(3):
        blk=X[off[s]:off[s+1],off[s]:off[s+1]]
        maxoff=max(maxoff, np.abs(blk-np.trace(blk)/dims[s]*np.eye(dims[s])).max())
    for s in range(3):
        for t in range(3):
            if s!=t: maxoff=max(maxoff, np.abs(X[off[s]:off[s+1],off[t]:off[t+1]]).max())
print(f"  max deviation from block-scalar form = {maxoff:.2e}  "
      f"{'OK -> commutant = span{P_m}, P_m central' if maxoff<1e-9 else 'FAIL'}"); ok&= maxoff<1e-9

# CONTRAST: equivalent (repeated) reps are NOT disjoint -> intertwiners appear
gens_eq=[]
for k in range(3):
    G=np.zeros((6,6),dtype=complex)
    for s in range(3): G[2*s:2*s+2,2*s:2*s+2]=su2(1)[k]
    gens_eq.append(G)
be=commutant_basis(gens_eq,6)
print(f"  [contrast] 3 EQUIVALENT copies: dim commutant = {len(be)} (= 9 = M_3, "
      f"off-diagonal intertwiners exist) {'OK' if len(be)==9 else 'FAIL'}"); ok&= len(be)==9

# Theorem 5: P_m is IN the algebra pi(A)'' = (pi(A)')'
Pms=[]
for s in range(3):
    P=np.zeros((D,D)); P[off[s]:off[s+1],off[s]:off[s+1]]=np.eye(dims[s]); Pms.append(P)
alg=commutant_basis(basis,D)          # pi(A)'' as a linear space
def in_span(X,B):
    A=np.stack([b.reshape(-1) for b in B],axis=1)
    sol,_,_,_=np.linalg.lstsq(A,X.reshape(-1),rcond=None)
    return np.abs(A@sol-X.reshape(-1)).max()
res=max(in_span(P,alg) for P in Pms)
print(f"  dim pi(A)'' = {len(alg)} (= 4+9+16 = 29) ; residual of P_m in algebra = {res:.2e}  "
      f"{'OK -> memory IS measurable' if res<1e-9 and len(alg)==29 else 'FAIL'}")
ok &= res<1e-9 and len(alg)==29

# cross-memory coherence is invisible: expectation independent of relative phase
v1=np.zeros(D,dtype=complex); v1[off[0]]=1
v2=np.zeros(D,dtype=complex); v2[off[1]]=1
spread=0.0
for A in alg:
    vals=[]
    for al in np.linspace(0,2*np.pi,17):
        psi=(v1+np.exp(1j*al)*v2)/np.sqrt(2); vals.append((psi.conj()@A@psi))
    spread=max(spread, np.ptp(np.real(vals))+np.ptp(np.imag(vals)))
print(f"  max phase-dependence of <psi_alpha|A|psi_alpha> over algebra = {spread:.2e}  "
      f"{'OK -> cross-memory coherence strictly invisible' if spread<1e-9 else 'FAIL'}")
ok &= spread<1e-9
print("RESULT:","PASS" if ok else "FAIL")
