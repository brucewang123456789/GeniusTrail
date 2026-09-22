import numpy as np,math,json
exec(open('gen.py').read().split('rngG=')[0])
r=json.load(open('cfgA_rat.json')); Q=8
om=np.array([r['omega0'][0]/r['omega0'][1]]+[2*j*math.pi for j in range(1,17)])
c=np.array([p/q for p,q in r['c']]); b=np.array([p/q for p,q in r['b']])
A={tuple(int(t) for t in k.split(',')):p/q for k,(p,q) in r['a'].items()}
Wf=mkW(om,c)
g=np.linspace(0.0,16.0,1601); Wg=Wf(g)
S2=Wf(g[:,None]+g[None,:])                      # W(g_i+g_j)
def dpLB(s,maxspan):
    # node cost for gap k (1-indexed k=1..8) : b_{k-1} g + s*a_{k-1,k} W(g)
    node=[b[k]*g+s*A[(k,k+1)]*Wg for k in range(Q)]
    # edge cost between gap k and k+1 : s*a_{k-1,k+1} W(g_k+g_{k+1})
    edge=[s*A[(k,k+2)]*S2 for k in range(Q-1)]
    f=node[0].copy()
    for k in range(Q-1):
        f=(f[:,None]+edge[k]).min(axis=0)+node[k+1]
    return f.min()
for s,eps in [(1.0,0.0080138),(19/20,0.0079038),(21/20,0.0081182),(0.5,0.00675)]:
    lb1=sum((b[k]*g+s*A[(k,k+1)]*Wg).min() for k in range(Q))
    lb2=dpLB(s,2)
    print("s=%-7.5g eps=%.7f | span1 LB=%.7f | span<=2 chain-DP LB=%.7f | gap to eps=%.3e | closes=%s"%(
        s,eps,lb1,lb2,eps-lb2,lb2>=eps))
