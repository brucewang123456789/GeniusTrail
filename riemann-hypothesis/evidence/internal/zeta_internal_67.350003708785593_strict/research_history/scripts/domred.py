import numpy as np,math,json
exec(open('gen.py').read().split('rngG=')[0])
r=json.load(open('cfgA_rat.json')); Q=8
om=np.array([r['omega0'][0]/r['omega0'][1]]+[2*j*math.pi for j in range(1,17)])
c=np.array([p/q for p,q in r['c']]); b=np.array([p/q for p,q in r['b']])
A={tuple(int(t) for t in k.split(',')):p/q for k,(p,q) in r['a'].items()}
Wf=mkW(om,c)
xs=np.linspace(0,60,600001)
Wv=Wf(xs)
for s,eps in [(1.0,0.0080138),(19/20,0.0079038),(0.5,0.00675),(21/20,0.0081182)]:
    phi=np.array([b[k]*xs+s*A[(k,k+1)]*Wv for k in range(Q)])   # gap index k+1 <-> pair (k,k+1)
    mr=phi.min(axis=1); tot=mr.sum()
    print("s=%-6.4g eps=%.7f  sum of separable minima = %.7f  slack=%.3e"%(s,eps,tot,eps-tot))
    if tot>eps: print("   *** separable bound alone already proves it ***"); continue
    print("   per-gap admissible range (naive eps/b_r  ->  separable):")
    tight=[]
    for k in range(Q):
        cap=eps-(tot-mr[k])
        ok=np.where(phi[k]<=cap)[0]
        lo,hi=(xs[ok[0]],xs[ok[-1]]) if len(ok) else (0,0)
        # admissible set may be disconnected; count components
        gaps=np.where(np.diff(ok)>1)[0]
        tight.append((eps/b[k],lo,hi,len(gaps)+1))
    for k,(nv,lo,hi,nc) in enumerate(tight):
        print("     g%d: [0,%.2f] -> [%.4f,%.4f]  shrink x%.1f  components=%d"%(k+1,nv,lo,hi,nv/max(hi-lo,1e-9),nc))
    vol0=np.prod([t[0] for t in tight]); vol1=np.prod([max(t[2]-t[1],1e-12) for t in tight])
    print("   box-volume reduction factor = %.3e"%(vol1/vol0))
    break
