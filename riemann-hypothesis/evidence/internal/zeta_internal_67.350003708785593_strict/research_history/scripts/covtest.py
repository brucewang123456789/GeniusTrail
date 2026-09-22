import numpy as np,math,json,sys,itertools,time
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
Q=int(sys.argv[1])
st=json.load(open('oq%d.json'%Q)); x=np.array(st['x'])
BL=[(Q+1-sp+1)//2 for sp in range(1,Q+1)]; IDX={}; p=21
for sp,n in enumerate(BL,1): IDX[sp]=(p,p+n); p+=n
NB=(Q+1)//2; BOFF=p
e=lambda v: np.exp(np.clip(v,-25,25))
om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)]); c=np.concatenate([[1.0],x[1:17]])
bh=e(x[BOFF:BOFF+NB])
b=np.concatenate([bh,bh[::-1]])[:Q] if Q%2==0 else np.concatenate([bh,bh[-2::-1]])
b=b/b.sum()*(93/23000)
A={}
for sp in range(1,Q+1):
    n=Q+1-sp; lo,hi=IDX[sp]; vals=e(x[lo:hi])
    w=np.array([vals[min(i,n-1-i)] for i in range(n)]); w=w/w.sum()*2
    for i in range(n): A[(i,i+sp)]=w[i]
cf=Cfg(Q,b,A); Wf=mkW(om,c)
P=[(i,j) for i in range(Q+1) for j in range(i+1,Q+1)]
I=np.array([t[0] for t in P]);J=np.array([t[1] for t in P]);aw=np.array([A[t] for t in P])
def PQf(G):
    G=np.abs(G); o=[[],[]]
    for s0 in range(0,len(G),15000):
        g=G[s0:s0+15000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
        o[0].append(g@b); o[1].append(Wf(Y[:,J]-Y[:,I])@aw)
    return np.concatenate(o[0]),np.concatenate(o[1])
def F1(g,s):
    p_,q_=PQf(np.abs(g).reshape(1,Q)); return float(p_[0]+s*q_[0])
r=np.random.default_rng(1)
def run(nseed,ntop,heavy):
    lv=[1.04,1.975,2.915]
    C=[np.array([r.choice(lv,Q) for _ in range(nseed)],float)]
    if heavy:
        C.append(np.array([r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9,4.85],Q) for _ in range(nseed)],float))
        C.append(r.uniform(0,5.0,(nseed*3,Q))); C.append(r.uniform(0,2.6,(nseed,Q)))
        # local-pattern seeds: random blocks of repeated values
        blk=np.array([np.repeat(r.choice(lv,max(1,Q//3)),3)[:Q] for _ in range(nseed)],float)
        C.append(blk)
    else:
        C.append(r.uniform(0,5.0,(nseed,Q)))
    CA=np.vstack(C); Pc,Qc=PQf(CA); s=1.0
    v=Pc+s*Qc; best=1e9
    for k in np.argsort(v)[:ntop]:
        meths=("Nelder-Mead","Powell") if heavy else ("Nelder-Mead",)
        for m in meths:
            rr=minimize(F1,CA[k],args=(s,),method=m,options={"maxiter":6000 if heavy else 2200})
            best=min(best,rr.fun)
    return best
t=time.time(); a=run(6561,6,False); b2=run(40000,40,True)
print("q=%d  eps(1.0): standard=%.9f   heavy=%.9f   drop=%.2e   [%.0fs]"%(Q,a,b2,a-b2,time.time()-t))
