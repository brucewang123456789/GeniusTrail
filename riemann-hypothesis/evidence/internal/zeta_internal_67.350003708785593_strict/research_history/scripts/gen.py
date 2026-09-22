import numpy as np, math
NQ=200001; tq=np.linspace(-0.5,0.5,NQ); wq=np.full(NQ,1.0/(NQ-1)); wq[0]/=2; wq[-1]/=2
from scipy.optimize import minimize
def Hfun(om,c):
    v=np.cos(np.outer(tq,om))@c
    if v.min()<=0: return None,v.min()
    a=wq*v; I1=a.sum(); I2=wq@(v*v)
    C1=np.concatenate([[0],np.cumsum(a)[:-1]]); C2=np.concatenate([[0],np.cumsum(a*tq)[:-1]])
    return 2-(I2+2*np.sum(a*(tq*C1-C2)))/I1**2, v.min()
def mkW(om,c):
    half=om/2
    def Kv(x):
        sh=x.shape; xr=x.ravel()
        return (0.5*((np.sinc((half[None,:]-math.pi*xr[:,None])/math.pi)+
                      np.sinc((half[None,:]+math.pi*xr[:,None])/math.pi))@c)).reshape(sh)
    K0=float(Kv(np.array([0.0]))[0]); return lambda x:(Kv(x)/K0)**2
SG=np.linspace(0.05,3.0,300)
class Cfg:
    def __init__(s,q,bvec,adict):
        s.q=q; s.d=2*q; s.r=q+1; s.T=(q+1)/q
        s.b=np.array(bvec,float); s.B=s.b.sum()
        P=[(i,j) for i in range(q+1) for j in range(i+1,q+1)]
        s.I=np.array([p[0] for p in P]); s.J=np.array([p[1] for p in P])
        s.a=np.array([adict[p] for p in P],float)
        for sp in range(1,q+1):
            tot=sum(adict[(i,i+sp)] for i in range(q+1-sp))
            assert abs(tot-2)<1e-9,(sp,tot)
    def h(s,E):
        T=s.T; d=s.d
        E=np.asarray(E,float)
        r=np.where(E<=T,E,E-(d/(d+1))*(np.sqrt(np.maximum(E,0))-math.sqrt(T))**2)
        return r if r.ndim else float(r)
rngG=np.random.default_rng(20260915)
def epscurve(cfg,Wf,ntop=6,nstart=60000):
    q=cfg.q
    SEED=np.array([[1.975 if (mk>>k)&1 else 1.04 for k in range(q)] for mk in range(2**q)],float)
    CAND=np.vstack([SEED]+[np.abs(SEED+rngG.normal(0,0.15,SEED.shape)) for _ in range(15)]
                   +[rngG.uniform(0,3.2,(nstart,q)),rngG.uniform(0,1.5,(nstart//2,q))])
    def PQ(G):
        G=np.abs(G); Y=np.concatenate([np.zeros((G.shape[0],1)),np.cumsum(G,axis=1)],axis=1)
        return G@cfg.b, Wf(Y[:,cfg.J]-Y[:,cfg.I])@cfg.a
    def F1(g,sl):
        p,qq=PQ(np.abs(g).reshape(1,q)); return float(p[0]+sl*qq[0])
    Pc,Qc=PQ(CAND); pool=[np.zeros(q)]
    for sl in [0.2,0.4,0.6,0.8,0.9,1.0,1.1,1.25,1.5,2.0,2.6]:
        v=Pc+sl*Qc
        for k in np.argsort(v)[:ntop]:
            r=minimize(F1,CAND[k],args=(sl,),method="Nelder-Mead",
                       options={"maxiter":1500,"fatol":1e-16,"xatol":1e-13})
            pool.append(np.abs(r.x))
    P,Q=PQ(np.array(pool))
    return np.min(P[None,:]+SG[:,None]*Q[None,:],axis=1)
def bestC(H,EPS,cfg,mrange):
    q=cfg.q; B=cfg.B; T=cfg.T
    dE=np.diff(EPS)/np.diff(SG); pv=EPS[:-1]-SG[:-1]*dE
    ok=pv>0; dE=dE[ok]; pv=pv[ok]; mx=float(EPS.max()); E0=float(np.max(EPS/SG))
    def Rmin(n,eta):
        Ec=n*dE; pc=n*pv
        hh=cfg.h(Ec)
        r=float(np.min(hh+eta*pc))
        pT=max(0.0,float(np.max(n*EPS-SG*T)))
        return min(r,T+eta*pT,eta*n*mx,cfg.h(n*E0))
    best=(0,0,0)
    for m in mrange:
        n=m-q; lo,hi=0.35,2.8
        for _ in range(45):
            x=lo+(hi-lo)/3; y=hi-(hi-lo)/3
            cx=(m*H-x*B*n)/(m-Rmin(n,x)); cy=(m*H-y*B*n)/(m-Rmin(n,y))
            if cx<cy: lo=x
            else: hi=y
        eta=(lo+hi)/2; R=Rmin(n,eta)
        if m-R>0:
            v=(m*H-eta*B*n)/(m-R)
            if v>best[0]: best=(v,m,eta)
    return best
