"""
CERTIFICATE 1 - Theorem 1 (Sector Connectivity Theorem), exact classification.

Claim (Theorem 1):
  Let A0 = (+)_{r=0}^{N} B(H_r) be the Proposition 1 block algebra.
  Let S = {S_1,...,S_k} be ANY family of bounded operators.
  Define graph G on vertices {0..N}: edge (r,r') iff exists S in S with P_{r'} S P_r != 0.
  Then
      W*(A0, S)  = (+)_{C in comp(G)} B(H_C)
      W*(A0, S)' = span{ P_C : C in comp(G) },   dim = #components.
  Complete (= B(H)) iff G is connected.

This strictly generalizes Proposition 2 (which only used the adjacent chain 0-1-2-...-N).
Numerical certificate: commutant dimension must equal the component count,
for randomly generated block patterns.
"""
import numpy as np
import itertools

rng = np.random.default_rng(20260914)

def block_offsets(dims):
    off = np.cumsum([0] + list(dims))
    return off

def projectors(dims):
    off = block_offsets(dims); D = off[-1]
    Ps = []
    for r in range(len(dims)):
        P = np.zeros((D, D)); P[off[r]:off[r+1], off[r]:off[r+1]] = np.eye(dims[r])
        Ps.append(P)
    return Ps

def A0_generators(dims):
    """Matrix units inside each diagonal block generate (+) B(H_r)."""
    off = block_offsets(dims); D = off[-1]; gens = []
    for r, d in enumerate(dims):
        for a in range(d):
            for b in range(d):
                G = np.zeros((D, D), dtype=complex)
                G[off[r]+a, off[r]+b] = 1.0
                gens.append(G)
    return gens

def commutant_dim_and_basis(gens, D):
    """Null space of X -> (GX - XG) stacked over generators."""
    rows = []
    I = np.eye(D)
    for G in gens:
        # vec(GX - XG) = (I kron G - G^T kron I) vec(X)
        rows.append(np.kron(I, G) - np.kron(G.T, I))
    M = np.vstack(rows)
    u, s, vh = np.linalg.svd(M)
    tol = max(M.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0) * 1e3
    null = vh[np.sum(s > tol):].conj().T
    return null.shape[1], null

def components(nv, edges):
    parent = list(range(nv))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    return len({find(i) for i in range(nv)})

def build_S(dims, edges):
    off = block_offsets(dims); D = off[-1]
    S = np.zeros((D, D), dtype=complex)
    for (a, b) in edges:
        blk = rng.normal(size=(dims[a], dims[b])) + 1j*rng.normal(size=(dims[a], dims[b]))
        S[off[a]:off[a+1], off[b]:off[b+1]] += blk
        S[off[b]:off[b+1], off[a]:off[a+1]] += blk.conj().T   # keep self-adjoint
    return S

print("=== CERT-1 Sector Connectivity Theorem: exact commutant dimension ===")
ok = True
tests = [
    ([2,3,2],       [(0,1),(1,2)]),          # adjacent chain = Proposition 2 case -> connected
    ([2,3,2],       [(0,2)]),                # non-adjacent edge only   -> 2 comps
    ([2,2,2,2],     [(0,1),(2,3)]),          # two comps
    ([2,2,2,2],     [(0,3),(1,2)]),          # two comps, crossing
    ([2,2,2,2],     [(0,3),(1,2),(0,1)]),    # connected via non-adjacent edges
    ([3,2,2,3,2],   []),                     # no edges -> N+1 comps (memory case)
    ([2,4,2,3],     [(0,1),(1,2),(2,3)]),    # full chain
    ([2,2,2,2,2],   [(0,2),(2,4),(1,3)]),    # two comps {0,2,4},{1,3}
]
for dims, edges in tests:
    D = sum(dims)
    gens = A0_generators(dims) + [build_S(dims, edges)]
    dim_comm, _ = commutant_dim_and_basis(gens, D)
    ncomp = components(len(dims), edges)
    good = (dim_comm == ncomp)
    ok &= good
    print(f"dims={dims} edges={edges}: dim(commutant)={dim_comm}  #components={ncomp}  {'OK' if good else 'FAIL'}")

# multi-operator family: edges supplied by several different S's
print("--- multi-operator families ---")
for dims, edgesets in [([2,2,2,2], [[(0,1)],[(1,2)],[(2,3)]]),
                       ([2,3,2,2], [[(0,1)],[(2,3)]]),
                       ([2,2,2],   [[(0,1)],[(0,1)]])]:
    D = sum(dims)
    Ss = [build_S(dims, e) for e in edgesets]
    gens = A0_generators(dims) + Ss
    dim_comm, _ = commutant_dim_and_basis(gens, D)
    all_edges = [e for es in edgesets for e in es]
    ncomp = components(len(dims), all_edges)
    good = (dim_comm == ncomp); ok &= good
    print(f"dims={dims} edgesets={edgesets}: dim={dim_comm} comps={ncomp} {'OK' if good else 'FAIL'}")

print("RESULT:", "PASS" if ok else "FAIL")
