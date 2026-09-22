import numpy as np,math,itertools
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
_r=np.random.default_rng(20260916)
lv=[1.04,1.975,2.915]
TRI8=np.array(list(itertools.product(lv,repeat=8)),float)
Q48=np.array([_r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9],8) for _ in range(9000)],float)
C8=np.vstack([TRI8,Q48]+[np.abs(TRI8+_r.normal(0,s,TRI8.shape)) for s in [0.05,0.15]]
             +[_r.uniform(0,5.0,(70000,8)),_r.uniform(0,2.6,(35000,8))])
SLOF=[0.3,0.5,0.7,0.9,1.0,1.05,1.1,1.25,1.5,2.0]
def eps8(cf,Wf,ntop=6):
    def PQ(G):
        G=np.abs(G); o=[[],[]]
        for s0 in range(0,len(G),20000):
            g=G[s0:s0+20000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
            o[0].append(g@cf.b); o[1].append(Wf(Y[:,cf.J]-Y[:,cf.I])@cf.a)
        return np.concatenate(o[0]),np.concatenate(o[1])
    def F1(g,s):
        p,q=PQ(np.abs(g).reshape(1,8)); return float(p[0]+s*q[0])
    Pc,Qc=PQ(C8); pool=[np.zeros(8)]
    for s in SLOF:
        v=Pc+s*Qc
        for k in np.argsort(v)[:ntop]:
            rr=minimize(F1,C8[k],args=(s,),method="Nelder-Mead",options={"maxiter":2500})
            pool.append(np.abs(rr.x))
    P,Q=PQ(np.array(pool))
    return np.min(P[None,:]+SG[:,None]*Q[None,:],axis=1)
