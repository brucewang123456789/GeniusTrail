import numpy as np,math,json,itertools,time,sys
from fractions import Fraction as F
from scipy.optimize import minimize
exec(open('heavy.py').read().split('if __name__')[0])
Q=12; D=10**12
x=np.array(json.load(open('oq%d.json'%Q))['x'])
om,c,b,A=build(Q,x)
om0=F(round(om[0]*D),D); cR=[F(1)]+[F(round(v*D),D) for v in c[1:]]
bR=[F(round(v*D),D) for v in b]; s=sum(bR); bR=[t*F(93,23000)/s for t in bR]
aR={}
for sp in range(1,Q+1):
    n=Q+1-sp; h=(n+1)//2
    ww=[F(round(A[(i,i+sp)]*D),D) for i in range(h)]
    if n%2==0: ww[-1]=F(1)-sum(ww[:-1]); full=ww+ww[::-1]
    else: ww[-1]=F(2)-2*sum(ww[:-1]); full=ww[:-1]+[ww[-1]]+ww[:-1][::-1]
    assert sum(full)==2 and all(t>=0 for t in full),(sp,full)
    for i in range(n): aR[(i,i+sp)]=full[i]
json.dump({"q":Q,"omega0":[om0.numerator,om0.denominator],"c":[[v.numerator,v.denominator] for v in cR],
 "b":[[v.numerator,v.denominator] for v in bR],
 "a":{f"{i},{j}":[v.numerator,v.denominator] for (i,j),v in sorted(aR.items())}},open('cfgB_rat.json','w'))
om=np.array([float(om0)]+[2*j*math.pi for j in range(1,17)]); c=np.array([float(v) for v in cR])
b=np.array([float(v) for v in bR]); A={k:float(v) for k,v in aR.items()}
H,mv=Hfun(om,c); print("frozen q=%d: H=%.15f minv=%.8f d=%d r=%d T=%s"%(Q,H,mv,2*Q,Q+1,F(Q+1,Q)),flush=True)
cf=Cfg(Q,b,A); Wf=mkW(om,c)
P=[(i,j) for i in range(Q+1) for j in range(i+1,Q+1)]
I=np.array([t[0] for t in P]);J=np.array([t[1] for t in P]);aw=np.array([A[t] for t in P])
def PQf(G):
    G=np.abs(G); o=[[],[]]
    for s0 in range(0,len(G),15000):
        g=G[s0:s0+15000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
        o[0].append(g@b); o[1].append(Wf(Y[:,J]-Y[:,I])@aw)
    return np.concatenate(o[0]),np.concatenate(o[1])
def F1(g,sv):
    p_,q_=PQf(np.abs(g).reshape(1,Q)); return float(p_[0]+sv*q_[0])
r=np.random.default_rng(31)
lv=[1.04,1.975,2.915]
TRI=np.array(list(itertools.product(lv,repeat=Q)),float)
CA=np.vstack([TRI,np.abs(TRI+r.normal(0,0.08,TRI.shape)),
  np.array([r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9,4.85],Q) for _ in range(30000)],float),
  r.uniform(0,5.0,(120000,Q)), r.uniform(0,2.6,(50000,Q))])
Pc,Qc=PQf(CA); pool=[np.zeros(Q)]
SLc=[F(1,2),F(4,5),F(19,20),F(1),F(21,20),F(11,10),F(6,5),F(13,10)]
mins={}; t0=time.time()
for sf in SLc:
    sv=float(sf); v=Pc+sv*Qc; bst=1e9
    for k in np.argsort(v)[:20]:
        for m in ("Nelder-Mead","Powell"):
            rr=minimize(F1,CA[k],args=(sv,),method=m,options={"maxiter":6000})
            if rr.fun<bst: bst=rr.fun
            pool.append(np.abs(rr.x))
    mins[sf]=bst; print("  eps(%s)=%.10f [%.0fs]"%(sf,bst,time.time()-t0),flush=True)
json.dump({str(k):v for k,v in mins.items()},open('minsB.json','w'))
Pp,Qp=PQf(np.array(pool)); E=np.min(Pp[None,:]+SG[:,None]*Qp[None,:],axis=1)
print("final deterministic ceiling:",bestC(H,E,cf,range(350,1400,1)))
