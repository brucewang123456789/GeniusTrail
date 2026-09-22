import numpy as np,math,json,time
from scipy.optimize import minimize
r=json.load(open('cfg7_rat.json'))
om=np.array([r['omega0'][0]/r['omega0'][1]]+[2*j*math.pi for j in range(1,17)])
c=np.array([p/q for p,q in r['c']]); b=np.array([p/q for p,q in r['b']])
A={tuple(int(t) for t in k.split(',')):p/q for k,(p,q) in r['a'].items()}
P=[(i,j) for i in range(8) for j in range(i+1,8)]
I=np.array([p[0] for p in P]); J=np.array([p[1] for p in P]); aw=np.array([A[p] for p in P])
half=om/2
def W(x):
    sh=x.shape; xr=x.ravel()
    K=0.5*((np.sinc((half[None,:]-math.pi*xr[:,None])/math.pi)+np.sinc((half[None,:]+math.pi*xr[:,None])/math.pi))@c)
    return (K.reshape(sh)/K0)**2
K0=0.5*float((np.sinc((half-0.0)/math.pi)+np.sinc((half+0.0)/math.pi))@c)
def PQ(G):
    G=np.abs(G); o=[[],[]]
    for s0 in range(0,len(G),20000):
        g=G[s0:s0+20000]; Y=np.concatenate([np.zeros((g.shape[0],1)),np.cumsum(g,axis=1)],axis=1)
        o[0].append(g@b); o[1].append(W(Y[:,J]-Y[:,I])@aw)
    return np.concatenate(o[0]),np.concatenate(o[1])
def F1(g,s):
    p,q=PQ(np.abs(g).reshape(1,7)); return float(p[0]+s*q[0])
tg={0.5:0.0067,1.05:0.0081962,1.1:0.0083003}
rng=np.random.default_rng(2026)
SEED=np.array([[1.975 if (mk>>k)&1 else 1.04 for k in range(7)] for mk in range(128)],float)
big=np.vstack([SEED]+[np.abs(SEED+rng.normal(0,sg,SEED.shape)) for sg in [0.05,0.1,0.2,0.35]*15]
              +[rng.uniform(0,3.6,(150000,7)),rng.uniform(0,1.8,(80000,7)),rng.uniform(0,0.9,(40000,7))])
t0=time.time()
for s,T in tg.items():
    Pc,Qc=PQ(big); v=Pc+s*Qc; best=1e9; barg=None
    for k in np.argsort(v)[:60]:
        for meth in ("Nelder-Mead","Powell"):
            rr=minimize(F1,big[k],args=(s,),method=meth,options={"maxiter":8000,"xtol":1e-14,"ftol":1e-16} if meth=="Powell" else {"maxiter":8000,"fatol":1e-17,"xatol":1e-14})
            if rr.fun<best: best=rr.fun; barg=np.abs(rr.x)
    print("s=%-5s target=%.8f  best-found=%.12f  margin=%+.3e  %s  [%.0fs]"%(
        s,T,best,best-T,"OK" if best>T else "*** COUNTEREXAMPLE ***",time.time()-t0),flush=True)
    if best<=T: print("   witness g =",np.round(barg,8))
