"""
CERTIFICATE 2 - Corollary 1.1 (spanning-tree resource lower bound).

Corollary of Theorem 1: any completing family needs >= N nonzero off-diagonal
helicity blocks. the single Stokes connector of Proposition 2 supplies exactly N adjacent
blocks, hence EDGE-OPTIMAL, not merely 'one operator suffices'.
"""
import numpy as np, itertools
rng = np.random.default_rng(7)

def commutant_dim(gens, D):
    I = np.eye(D)
    M = np.vstack([np.kron(I, G) - np.kron(G.T, I) for G in gens])
    s = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape)*np.finfo(float).eps*s[0]*1e3
    return D*D - int(np.sum(s > tol))

def setup(nv, d=2):
    dims=[d]*nv; off=np.cumsum([0]+dims); D=off[-1]
    gens=[]
    for r in range(nv):
        for a in range(d):
            for b in range(d):
                G=np.zeros((D,D),dtype=complex); G[off[r]+a,off[r]+b]=1.0; gens.append(G)
    return dims, off, D, gens

def build_S(off, dims, edges):
    D=off[-1]; S=np.zeros((D,D),dtype=complex)
    for (a,b) in edges:
        blk=rng.normal(size=(dims[a],dims[b]))+1j*rng.normal(size=(dims[a],dims[b]))
        S[off[a]:off[a+1],off[b]:off[b+1]]+=blk
        S[off[b]:off[b+1],off[a]:off[a+1]]+=blk.conj().T
    return S

print("=== CERT-2 spanning-tree lower bound ===")
ok=True
N=3                                   # vertices r=0..3 -> spanning tree needs 3 edges
dims,off,D,gens = setup(N+1)
pairs=list(itertools.combinations(range(N+1),2))
for k in range(0,N+2):
    achievable=False
    for edges in itertools.combinations(pairs,k):
        if commutant_dim(gens+[build_S(off,dims,list(edges))], D)==1:
            achievable=True; break
    exp = k>=N
    good = (achievable==exp); ok&=good
    print(f"  {k} nonzero blocks -> completion {'possible ' if achievable else 'IMPOSSIBLE'}"
          f" (expected {'possible ' if exp else 'IMPOSSIBLE'}) {'OK' if good else 'FAIL'}")

adj=[(r,r+1) for r in range(N)]
dc=commutant_dim(gens+[build_S(off,dims,adj)], D)
print(f"  adjacent chain of Proposition 2: {len(adj)} = N blocks, dim(commutant)={dc} "
      f"{'OK -> EDGE-OPTIMAL' if dc==1 else 'FAIL'}")
ok &= dc==1
print("RESULT:","PASS" if ok else "FAIL")
