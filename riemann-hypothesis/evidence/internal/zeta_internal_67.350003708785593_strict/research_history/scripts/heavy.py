import numpy as np,math,json,sys,itertools,time
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
def build(Q,x):
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
    return om,c,b,A
def heavyeps(Q,om,c,b,A,ntop=18,ret=False):
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
    r=np.random.default_rng(99)
    lv=[1.04,1.975,2.915]
    TRI=np.array(list(itertools.product(lv,repeat=Q)),float)      # COMPLETE 3^Q
    C=[TRI,np.array([r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9,4.85],Q) for _ in range(30000)],float),
       np.abs(TRI+r.normal(0,0.08,TRI.shape)), r.uniform(0,5.0,(120000,Q)), r.uniform(0,2.6,(50000,Q))]
    CA=np.vstack(C); Pc,Qc=PQf(CA); pool=[np.zeros(Q)]
    for s in [0.3,0.5,0.7,0.9,1.0,1.05,1.1,1.25,1.5,2.0]:
        v=Pc+s*Qc
        for k in np.argsort(v)[:ntop]:
            for m in ("Nelder-Mead","Powell"):
                rr=minimize(F1,CA[k],args=(s,),method=m,options={"maxiter":6000})
                pool.append(np.abs(rr.x))
    Pp,Qp=PQf(np.array(pool))
    E=np.min(Pp[None,:]+SG[:,None]*Qp[None,:],axis=1)
    return (E,cf,Pp,Qp) if ret else E
if __name__=="__main__":
    for Q in [int(a) for a in sys.argv[1:]]:
        x=np.array(json.load(open('oq%d.json'%Q))['x'])
        om,c,b,A=build(Q,x); H,mv=Hfun(om,c)
        t=time.time(); E=heavyeps(Q,om,c,b,A)
        r=bestC(H,E,Cfg(Q,b,A),range(350,1500,2))
        print("q=%d  HEAVY ceiling=%.12f  (thin-search value was different)  m=%d eta=%.4f  H=%.9f  [%.0fs]"%(Q,r[0],r[1],r[2],H,time.time()-t),flush=True)
