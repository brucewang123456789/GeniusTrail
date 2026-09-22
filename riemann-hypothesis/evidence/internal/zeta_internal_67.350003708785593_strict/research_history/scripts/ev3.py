import numpy as np,math,json,itertools
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
_r=np.random.default_rng(7)
lv=[1.04,1.975,2.915]
TRI=np.array(list(itertools.product(lv,repeat=7)),float)
QUAD=np.array([_r.choice([1.047,1.985,2.90,3.85],7) for _ in range(4000)],float)
CAND=np.vstack([TRI,QUAD]+[np.abs(TRI+_r.normal(0,s,TRI.shape)) for s in [0.05,0.15]]
               +[_r.uniform(0,4.5,(60000,7)),_r.uniform(0,2.4,(30000,7))])
SLO=[0.3,0.5,0.7,0.9,1.0,1.05,1.1,1.3,1.7,2.4]
ARCH=[np.zeros(7)]
def epscurve3(cf,Wf,ntop=6):
    def PQ(G):
        G=np.abs(G); o=[[],[]]
        for s0 in range(0,len(G),25000):
            g=G[s0:s0+25000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
            o[0].append(g@cf.b); o[1].append(Wf(Y[:,cf.J]-Y[:,cf.I])@cf.a)
        return np.concatenate(o[0]),np.concatenate(o[1])
    def F1(g,s):
        p,q=PQ(np.abs(g).reshape(1,7)); return float(p[0]+s*q[0])
    base=np.vstack([CAND,np.array(ARCH)])
    Pc,Qc=PQ(base); pool=[np.zeros(7)]
    for s in SLO:
        v=Pc+s*Qc
        for k in np.argsort(v)[:ntop]:
            rr=minimize(F1,base[k],args=(s,),method="Nelder-Mead",options={"maxiter":2500,"fatol":1e-17,"xatol":1e-14})
            pool.append(np.abs(rr.x))
    for g in pool[1:]:
        if len(ARCH)<1200: ARCH.append(g)
    P,Q=PQ(np.array(pool))
    return np.min(P[None,:]+SG[:,None]*Q[None,:],axis=1)
