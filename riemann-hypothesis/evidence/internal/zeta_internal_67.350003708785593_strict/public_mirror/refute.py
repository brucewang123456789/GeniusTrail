"""Re-evaluate the refutation witnesses. Runs in seconds."""
import numpy as np, math, json
r=json.load(open('config/cfg7_rat.json'))
om=np.array([r['omega0'][0]/r['omega0'][1]]+[2*j*math.pi for j in range(1,17)])
c=np.array([p/q for p,q in r['c']]); b=np.array([p/q for p,q in r['b']])
A={tuple(int(t) for t in k.split(',')):p/q for k,(p,q) in r['a'].items()}
P=[(i,j) for i in range(8) for j in range(i+1,8)]
I=np.array([p[0] for p in P]);J=np.array([p[1] for p in P]);aw=np.array([A[p] for p in P])
half=om/2
K0=0.5*float((np.sinc(half/math.pi)+np.sinc(half/math.pi))@c)
def W(x):
    xr=np.atleast_1d(x)
    K=0.5*((np.sinc((half[None,:]-math.pi*xr[:,None])/math.pi)+np.sinc((half[None,:]+math.pi*xr[:,None])/math.pi))@c)
    return (K/K0)**2
def F(g,s):
    g=np.abs(np.asarray(g,float)); y=np.concatenate([[0],np.cumsum(g)])
    return float(g@b + s*(W(y[J]-y[I])@aw))
W_=[(11/10,[1.99140578,1.04683263,1.98561048,1.98374147,1.04573175,1.97741439,1.04170216],0.0083003)]
for s,g,t in W_:
    v=F(g,s); print("slope %s: value %.12f  target %.12f  ->  %s"%(s,v,t,"REFUTED" if v<t else "holds"))
