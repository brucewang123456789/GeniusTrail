import numpy as np,math,json,time
exec(open('adv.py').read().split('tg={')[0])
rng=np.random.default_rng(31337)
pats=[]
for lo,hi in [(1.04,1.975),(1.047,1.985),(1.03,1.96),(1.05,1.99),(1.06,2.00),(0.95,1.95)]:
    pats.append(np.array([[hi if (mk>>k)&1 else lo for k in range(7)] for mk in range(128)],float))
base=np.vstack(pats)
big=np.vstack([base]+[np.abs(base+rng.normal(0,sg,base.shape)) for sg in [0.02,0.05,0.1,0.2,0.35,0.6]*8]
              +[rng.uniform(0,4.0,(200000,7)),rng.uniform(0,2.2,(100000,7)),rng.uniform(0,1.2,(60000,7))])
Pc,Qc=PQ(big); t0=time.time()
for s in [1.0,1.05]:
    v=Pc+s*Qc; best=1e9;barg=None
    for k in np.argsort(v)[:120]:
        for meth in ("Nelder-Mead","Powell"):
            rr=minimize(F1,big[k],args=(s,),method=meth,options={"maxiter":9000})
            if rr.fun<best: best=rr.fun;barg=np.abs(rr.x)
        if time.time()-t0>230: break
    print("s=%.3f  hardened-min = %.12f   (previous %.12f)"%(s,best,{1.0:0.008084457,1.05:0.008199540}[s]),flush=True)
    print("   arg",np.round(barg,6))
