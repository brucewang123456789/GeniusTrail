import numpy as np
rng=np.random.default_rng(7)
def Psi(t): return np.where(t<=2,(t-1)**2,2*t-3)
bad=0; worst=1e9
for trial in range(4000):
    n=rng.integers(2,9); s=rng.integers(1,7); b=rng.integers(0,4)
    V=rng.normal(0,1,(n,s))
    nrm=np.linalg.norm(V,axis=0); sc=rng.uniform(0.2,1.0,s)
    V=V/np.maximum(nrm,1e-12)*sc                      # column norms <= 1
    P0=V@V.T; M=V.T@V
    # Q0 real symmetric with at most b positive eigenvalues
    Qr=rng.normal(0,1,(n,n)); Qr=(Qr+Qr.T)/2
    w,Uq=np.linalg.eigh(Qr)
    w=-np.abs(w)                                      # all negative
    if b>0:
        k=min(b,n); idx=rng.choice(n,k,replace=False); w[idx]=np.abs(w[idx])*rng.uniform(0,3,k)
    Q0=Uq@np.diag(w)@Uq.T
    A=P0+Q0
    lhs=np.sum(A*A)
    rhs=4*np.trace(A)-3*s-4*b+np.sum(Psi(np.linalg.eigvalsh(M)))
    d=lhs-rhs
    if d<worst: worst=d
    if d<-1e-9: bad+=1
print("trials 4000  violations:",bad,"  min slack:",worst)
