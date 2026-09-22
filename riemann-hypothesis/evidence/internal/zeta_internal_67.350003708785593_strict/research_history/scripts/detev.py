import numpy as np,math,json,itertools,os
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
_r=np.random.default_rng(20260916)
lv=[1.04,1.975,2.915]
TRI=np.array(list(itertools.product(lv,repeat=7)),float)
Q4=np.array([_r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9],7) for _ in range(15000)],float)
CAND=np.vstack([TRI,Q4]+[np.abs(TRI+_r.normal(0,s,TRI.shape)) for s in [0.04,0.1,0.22]]
               +[_r.uniform(0,5.0,(120000,7)),_r.uniform(0,2.6,(60000,7))])
SLOD=[0.3,0.5,0.7,0.9,1.0,1.05,1.1,1.2,1.35,1.7,2.4]
def epsdet(cf,Wf,ntop=12):
    def PQ(G):
        G=np.abs(G); o=[[],[]]
        for s0 in range(0,len(G),25000):
            g=G[s0:s0+25000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
            o[0].append(g@cf.b); o[1].append(Wf(Y[:,cf.J]-Y[:,cf.I])@cf.a)
        return np.concatenate(o[0]),np.concatenate(o[1])
    def F1(g,s):
        p,q=PQ(np.abs(g).reshape(1,7)); return float(p[0]+s*q[0])
    Pc,Qc=PQ(CAND); pool=[np.zeros(7)]
    for s in SLOD:
        v=Pc+s*Qc
        for k in np.argsort(v)[:ntop]:
            for meth in ("Nelder-Mead","Powell"):
                rr=minimize(F1,CAND[k],args=(s,),method=meth,options={"maxiter":4000})
                pool.append(np.abs(rr.x))
    P,Q=PQ(np.array(pool))
    return np.min(P[None,:]+SG[:,None]*Q[None,:],axis=1), (P,Q)
def mkcfg(x,nf):
    om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)])
    if nf: om[1:1+nf]+=x[35:35+nf] if nf==8 else x[42:58]
    c=np.concatenate([[1.0],x[1:17]])
    e=lambda v: np.exp(np.clip(v,-25,25))
    if nf==16:
        b=e(x[17:24]); b=b/b.sum()*(93/23000)
        blk=[(24,28),(28,31),(31,34),(34,36),(36,38)]
    else:
        bp=e(x[17:21]); b=np.concatenate([bp,bp[-2::-1]]); b=b/b.sum()*(93/23000)
        blk=[(21,25),(25,28),(28,31),(31,33),(33,35)]
    A={}
    def lay(sp,vals):
        n=8-sp; ws=np.array([vals[min(i,n-1-i)] for i in range(n)],float); ws=ws/ws.sum()*2
        for i in range(n): A[(i,i+sp)]=ws[i]
    for sp,(lo,hi) in enumerate(blk,1): lay(sp,e(x[lo:hi]))
    lay(6,[1.0]); lay(7,[1.0])
    return om,c,b,A
