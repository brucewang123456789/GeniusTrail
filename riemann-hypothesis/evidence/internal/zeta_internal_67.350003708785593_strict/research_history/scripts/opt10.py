import numpy as np,math,json,time,os,itertools
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
Q=10
BL=[(Q+1-sp+1)//2 for sp in range(1,Q+1)]; IDX={}; p=21
for sp,n in enumerate(BL,1): IDX[sp]=(p,p+n); p+=n
NB=(Q+1)//2; BOFF=p; N=p+NB
_r=np.random.default_rng(4242)
lv=[1.04,1.975,2.915]
TRI=np.array(list(itertools.product(lv,repeat=Q)),float)          # complete 3^10 = 59049
CAND=np.vstack([TRI, np.abs(TRI+_r.normal(0,0.09,TRI.shape)),
                np.array([_r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9,4.85],Q) for _ in range(20000)],float),
                _r.uniform(0,5.0,(60000,Q)), _r.uniform(0,2.6,(25000,Q))])
SLOF=[0.3,0.5,0.7,0.9,1.0,1.05,1.1,1.25,1.5,2.0]
def unpack(x):
    e=lambda v: np.exp(np.clip(v,-25,25))
    om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)]); c=np.concatenate([[1.0],x[1:17]])
    bh=e(x[BOFF:BOFF+NB]); b=np.concatenate([bh,bh[::-1]])[:Q]; b=b/b.sum()*(93/23000)
    A={}
    for sp in range(1,Q+1):
        n=Q+1-sp; lo,hi=IDX[sp]; vals=e(x[lo:hi])
        w=np.array([vals[min(i,n-1-i)] for i in range(n)]); w=w/w.sum()*2
        for i in range(n): A[(i,i+sp)]=w[i]
    return om,c,b,A
def ev(x,ntop=5,heavy=False):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1
    cf=Cfg(Q,b,A); Wf=mkW(om,c)
    P=[(i,j) for i in range(Q+1) for j in range(i+1,Q+1)]
    I=np.array([t[0] for t in P]);J=np.array([t[1] for t in P]);aw=np.array([A[t] for t in P])
    def PQf(G):
        G=np.abs(G); o=[[],[]]
        for s0 in range(0,len(G),20000):
            g=G[s0:s0+20000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
            o[0].append(g@b); o[1].append(Wf(Y[:,J]-Y[:,I])@aw)
        return np.concatenate(o[0]),np.concatenate(o[1])
    def F1(g,s):
        p_,q_=PQf(np.abs(g).reshape(1,Q)); return float(p_[0]+s*q_[0])
    Pc,Qc=PQf(CAND); pool=[np.zeros(Q)]
    for s in SLOF:
        v=Pc+s*Qc
        for k in np.argsort(v)[:ntop]:
            ms=("Nelder-Mead","Powell") if heavy else ("Nelder-Mead",)
            for m in ms:
                rr=minimize(F1,CAND[k],args=(s,),method=m,options={"maxiter":5000 if heavy else 2200})
                pool.append(np.abs(rr.x))
    Pp,Qp=PQf(np.array(pool))
    E=np.min(Pp[None,:]+SG[:,None]*Qp[None,:],axis=1)
    return bestC(H,E,cf,range(350,1400,4))[0]
f='o10.json'
if os.path.exists(f):
    st=json.load(open(f)); xb=np.array(st['x']); fb=st['f']
else:
    xb=np.array(json.load(open('oq10.json'))['x']); fb=ev(xb)
    print("q=10 start %.12f"%fb,flush=True)
sc=np.concatenate([[0.02],np.full(16,3e-3),np.full(N-17,0.35)])
rng=np.random.default_rng(int(time.time())%9001); t0=time.time(); sg=0.45; tot=0;succ=0
while time.time()-t0<225:
    d=rng.normal(0,1,N)
    if rng.random()<0.6: d[:17]*=0.15
    d/=np.linalg.norm(d); step=sg*sc*d
    cand=[xb+step,xb-step,xb+2.5*step]; vals=[ev(z) for z in cand]; tot+=1
    k=int(np.argmax(vals))
    if vals[k]>fb: fb=vals[k];xb=cand[k];succ+=1;sg*=1.35;print("  + %.12f [%d/%.0fs]"%(fb,tot,time.time()-t0),flush=True)
    else: sg*=0.82
    if sg<8e-5: sg=0.35
json.dump({'x':list(xb),'f':fb,'q':Q},open(f,'w'))
print("BEST q=10 %.12f iters=%d succ=%d"%(fb,tot,succ))
