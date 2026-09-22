import numpy as np,math,json,time,os,sys,itertools
from scipy.optimize import minimize
exec(open('gen.py').read().split('rngG=')[0])
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
Q=int(sys.argv[1]); TAG='oq%d'%Q
_r=np.random.default_rng(20260916+Q)
lv=[1.04,1.975,2.915]
if Q<=8: TRI=np.array(list(itertools.product(lv,repeat=Q)),float)
else: TRI=np.array([_r.choice(lv,Q) for _ in range(6561)],float)
QD=np.array([_r.choice([1.03,1.05,1.96,1.99,2.88,2.95,3.9],Q) for _ in range(9000)],float)
CAND=np.vstack([TRI,QD]+[np.abs(TRI+_r.normal(0,s,TRI.shape)) for s in [0.05,0.15]]
               +[_r.uniform(0,5.0,(70000,Q)),_r.uniform(0,2.6,(35000,Q))])
SLOF=[0.3,0.5,0.7,0.9,1.0,1.05,1.1,1.25,1.5,2.0]
def epsq(cf,Wf,ntop=6):
    def PQ(G):
        G=np.abs(G); o=[[],[]]
        for s0 in range(0,len(G),20000):
            g=G[s0:s0+20000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
            o[0].append(g@cf.b); o[1].append(Wf(Y[:,cf.J]-Y[:,cf.I])@cf.a)
        return np.concatenate(o[0]),np.concatenate(o[1])
    def F1(g,s):
        p,q=PQ(np.abs(g).reshape(1,Q)); return float(p[0]+s*q[0])
    Pc,Qc=PQ(CAND); pool=[np.zeros(Q)]
    for s in SLOF:
        v=Pc+s*Qc
        for k in np.argsort(v)[:ntop]:
            rr=minimize(F1,CAND[k],args=(s,),method="Nelder-Mead",options={"maxiter":2200})
            pool.append(np.abs(rr.x))
    P,Qq=PQ(np.array(pool))
    return np.min(P[None,:]+SG[:,None]*Qq[None,:],axis=1)
BL=[(Q+1-sp+1)//2 for sp in range(1,Q+1)]      # halves per span
IDX={}; p=21
for sp,n in enumerate(BL,1): IDX[sp]=(p,p+n); p+=n
NB=(Q+1)//2
BOFF=p; N=p+NB
def unpack(x):
    om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)]); c=np.concatenate([[1.0],x[1:17]])
    e=lambda v: np.exp(np.clip(v,-25,25))
    bh=e(x[BOFF:BOFF+NB])
    b=np.concatenate([bh,bh[::-1]])[:Q] if Q%2==0 else np.concatenate([bh,bh[-2::-1]])
    b=b/b.sum()*(93/23000)
    A={}
    for sp in range(1,Q+1):
        n=Q+1-sp; lo,hi=IDX[sp]; vals=e(x[lo:hi])
        w=np.array([vals[min(i,n-1-i)] for i in range(n)]); w=w/w.sum()*2
        for i in range(n): A[(i,i+sp)]=w[i]
    return om,c,b,A
def ev(x):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1
    cf=Cfg(Q,b,A); return bestC(H,epsq(cf,mkW(om,c)),cf,range(350,1500,5))[0]
f=TAG+'.json'
if os.path.exists(f):
    st=json.load(open(f)); xb=np.array(st['x']); fb=st['f']
else:
    prev='oq%d.json'%(Q-1)
    if os.path.exists(prev):
        st=json.load(open(prev)); y=np.array(st['x']); PQb=st['q']
    else:
        y=np.array(json.load(open('o9.json'))['x']); PQb=8
    xb=np.zeros(N); xb[:17]=y[:17]
    BL8=[(PQb+1-sp+1)//2 for sp in range(1,PQb+1)]; I8={}; p8=21
    for sp,n in enumerate(BL8,1): I8[sp]=(p8,p8+n); p8+=n
    NB8=(PQb+1)//2; BOFF8=p8
    for sp in range(1,Q+1):
        lo,hi=IDX[sp]; n=hi-lo
        if sp in I8:
            src8=y[I8[sp][0]:I8[sp][1]]
            xb[lo:hi]=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(src8)),src8)
        else: xb[lo:hi]=np.log(np.linspace(1.0,1.2,n))
    xb[BOFF:BOFF+NB]=np.interp(np.linspace(0,1,NB),np.linspace(0,1,NB8),y[BOFF8:BOFF8+NB8])
    fb=ev(xb); print("q=%d start %.12f"%(Q,fb),flush=True)
    if len(sys.argv)>2 and sys.argv[2]=='probe':
        om,c,b,A=unpack(xb); cf=Cfg(Q,b,A); Wf=mkW(om,c)
        e6=epsq(cf,Wf,ntop=6); e20=epsq(cf,Wf,ntop=20)
        i=int(np.argmin(np.abs(SG-1.0)))
        print("   conv-diag eps(1.0): ntop6=%.9f ntop20=%.9f  drop=%.2e"%(e6[i],e20[i],e6[i]-e20[i]))
        json.dump({'x':list(xb),'f':fb,'q':Q},open(f,'w')); sys.exit(0)
sc=np.concatenate([[0.02],np.full(16,3e-3),np.full(N-17,0.35)])
rng=np.random.default_rng(int(time.time())%9001); t0=time.time(); sg=0.5; tot=0;succ=0
while time.time()-t0<235:
    d=rng.normal(0,1,N)
    if rng.random()<0.6: d[:17]*=0.15
    d/=np.linalg.norm(d); step=sg*sc*d
    cand=[xb+step,xb-step,xb+2.5*step,xb-2.5*step]; vals=[ev(z) for z in cand]; tot+=1
    k=int(np.argmax(vals))
    if vals[k]>fb: fb=vals[k];xb=cand[k];succ+=1;sg*=1.35;print("  + %.12f [%d/%.0fs]"%(fb,tot,time.time()-t0),flush=True)
    else: sg*=0.82
    if sg<8e-5: sg=0.4
json.dump({'x':list(xb),'f':fb,'q':Q},open(f,'w'))
print("BEST q=%d  %.12f iters=%d succ=%d"%(Q,fb,tot,succ))
