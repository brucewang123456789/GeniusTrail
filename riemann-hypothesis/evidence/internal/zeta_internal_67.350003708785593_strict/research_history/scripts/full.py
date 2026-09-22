import numpy as np,math,json,time,itertools
exec(open('adv.py').read().split('tg={')[0])
rng=np.random.default_rng(7)
lv=[1.04,1.975,2.915]
tri=np.array(list(itertools.product(lv,repeat=7)),float)          # 2187
lv2=[1.047,1.985,2.90,3.85]
quad=np.array([rng.choice(lv2,7) for _ in range(6000)],float)
big=np.vstack([tri,quad]+[np.abs(tri+rng.normal(0,sg,tri.shape)) for sg in [0.03,0.08,0.18]]
              +[rng.uniform(0,4.5,(150000,7)),rng.uniform(0,2.4,(80000,7))])
Pc,Qc=PQ(big); pool=[np.zeros(7)]; t0=time.time()
SLO=[0.2,0.35,0.5,0.7,0.85,0.95,1.0,1.05,1.1,1.2,1.4,1.7,2.2,2.8]
for s in SLO:
    v=Pc+s*Qc
    for k in np.argsort(v)[:20]:
        for meth in ("Nelder-Mead","Powell"):
            rr=minimize(F1,big[k],args=(s,),method=meth,options={"maxiter":6000})
            pool.append(np.abs(rr.x))
    if time.time()-t0>215: print("cut",s);break
Pp,Qp=PQ(np.array(pool)); np.save('P3.npy',Pp);np.save('Q3.npy',Qp)
SG=np.linspace(0.05,3.0,300); EPS=np.min(Pp[None,:]+SG[:,None]*Qp[None,:],axis=1); np.save('EPS3.npy',EPS)
for s in [0.5,0.9,1.0,1.05,1.1,1.25]: print("  eps(%.2f)=%.9f"%(s,float(np.min(Pp+s*Qp))))
exec(open('gen.py').read().split('rngG=')[0])
r=json.load(open('cfg7_rat.json')); bb=np.array([p/q for p,q in r['b']])
A={tuple(int(t) for t in k.split(',')):p/q for k,(p,q) in r['a'].items()}
cf=Cfg(7,bb,A); src=open('gen.py').read(); exec(src[src.index("def bestC"):])
print("CEILING with converged curve:", bestC(672112936142361/1e15,EPS,cf,range(300,1200,1)))
