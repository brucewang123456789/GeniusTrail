import numpy as np,math,json,itertools,time
from fractions import Fraction as F
from scipy.optimize import minimize
exec(open('ev8.py').read())
BL=[4,4,3,3,2,2,1]; IDX={}; p=21
for sp,n in enumerate(BL,1): IDX[sp]=(p,p+n); p+=n
x=np.array(json.load(open('o9.json'))['x']); D=10**12
e=lambda v: np.exp(np.clip(v,-25,25))
om0=F(round(x[0]*D),D); cR=[F(1)]+[F(round(v*D),D) for v in x[1:17]]
bp=e(x[17:21]); bf=np.concatenate([bp,bp[::-1]]); bf=bf/bf.sum()*(93/23000)
bR=[F(round(v*D),D) for v in bf]; s=sum(bR); bR=[t*F(93,23000)/s for t in bR]
aR={}
for sp in range(1,9):
    n=9-sp; h=(n+1)//2
    if sp<8:
        lo,hi=IDX[sp]; vals=e(x[lo:hi]); w=np.array([vals[min(i,n-1-i)] for i in range(n)]); w=w/w.sum()*2
    else: w=np.array([2.0])
    ww=[F(round(w[i]*D),D) for i in range(h)]
    if n%2==0: ww[-1]=F(1)-sum(ww[:-1]); full=ww+ww[::-1]
    else: ww[-1]=F(2)-2*sum(ww[:-1]); full=ww[:-1]+[ww[-1]]+ww[:-1][::-1]
    assert sum(full)==2 and all(t>=0 for t in full),(sp,full)
    for i in range(n): aR[(i,i+sp)]=full[i]
json.dump({"q":8,"omega0":[om0.numerator,om0.denominator],"c":[[v.numerator,v.denominator] for v in cR],
 "b":[[v.numerator,v.denominator] for v in bR],
 "a":{f"{i},{j}":[v.numerator,v.denominator] for (i,j),v in sorted(aR.items())}},open('cfgA_rat.json','w'))
om=np.array([float(om0)]+[2*j*math.pi for j in range(1,17)]); c=np.array([float(v) for v in cR])
b=np.array([float(v) for v in bR]); A={k:float(v) for k,v in aR.items()}
cf=Cfg(8,b,A); Wf=mkW(om,c); H,mv=Hfun(om,c)
print("frozen q=8: H=%.15f minv=%.8f  d=%d r=%d T=%s"%(H,mv,cf.d,cf.r,F(9,8)))
# deterministic strong evaluator
_r2=np.random.default_rng(777)
BIG=np.vstack([TRI8,Q48]+[np.abs(TRI8+_r2.normal(0,s,TRI8.shape)) for s in [0.03,0.09,0.2,0.35]]
              +[_r2.uniform(0,5.5,(150000,8)),_r2.uniform(0,2.8,(70000,8))])
P=[(i,j) for i in range(9) for j in range(i+1,9)]
I=np.array([q[0] for q in P]);J=np.array([q[1] for q in P]);aw=np.array([A[q] for q in P])
def PQ(G):
    G=np.abs(G); o=[[],[]]
    for s0 in range(0,len(G),20000):
        g=G[s0:s0+20000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
        o[0].append(g@b); o[1].append(Wf(Y[:,J]-Y[:,I])@aw)
    return np.concatenate(o[0]),np.concatenate(o[1])
def F1(g,s):
    pp,qq=PQ(np.abs(g).reshape(1,8)); return float(pp[0]+s*qq[0])
Pc,Qc=PQ(BIG); pool=[np.zeros(8)]
SLc=[F(1,2),F(4,5),F(19,20),F(1),F(21,20),F(11,10),F(6,5),F(13,10)]
mins={}; t0=time.time()
for sf in SLc:
    sv=float(sf); v=Pc+sv*Qc; bst=1e9
    for k in np.argsort(v)[:26]:
        for meth in ("Nelder-Mead","Powell"):
            rr=minimize(F1,BIG[k],args=(sv,),method=meth,options={"maxiter":7000})
            if rr.fun<bst: bst=rr.fun
            pool.append(np.abs(rr.x))
    mins[sf]=bst; print("  eps(%s)=%.10f [%.0fs]"%(sf,bst,time.time()-t0),flush=True)
json.dump({str(k):v for k,v in mins.items()},open('minsA.json','w'))
Pp,Qp=PQ(np.array(pool))
EPS=np.min(Pp[None,:]+SG[:,None]*Qp[None,:],axis=1)
print("DETERMINISTIC ceiling:",bestC(H,EPS,cf,range(350,1400,1)))
