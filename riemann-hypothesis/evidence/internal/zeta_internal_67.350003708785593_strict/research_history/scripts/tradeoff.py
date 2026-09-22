import numpy as np,math,json
from fractions import Fraction as F
exec(open('gen.py').read().split('rngG=')[0])
r=json.load(open('cfg7_rat.json'))
b=np.array([p/q for p,q in r['b']]); A={tuple(int(t) for t in k.split(',')):p/q for k,(p,q) in r['a'].items()}
cf=Cfg(7,b,A); Hf=F(672112936142361,10**15); T=F(8,7); B=F(93,23000)
P2=np.load('P2.npy');Q2=np.load('Q2.npy')
def emin(s): return float(np.min(P2+s*Q2))
def sqrt_lo(x,K=10**30):
    rr=F(math.isqrt(x.numerator*K*K//x.denominator),K)
    while rr*rr>x: rr-=F(1,K)
    return rr
def hlo(E): return E if E<=T else E/15+F(28,15)*sqrt_lo(E*T)-F(14,15)*T
def Cexact(SL,EPSr,m,eta):
    n=m-7; lines=[(s,n*EPSr[s]) for s in SL]
    cand={F(0),T}
    for i,(si,bi) in enumerate(lines):
        cand.add(bi/si)
        for sj,bj in lines[i+1:]:
            E=(bi-bj)/(si-sj)
            if E>0: cand.add(E)
    R=min(hlo(E)+eta*max([F(0)]+[bi-si*E for si,bi in lines]) for E in cand)
    if m-R<=0: return None,None
    return (m*Hf-eta*B*n)/(m-R),R
SL=[F(1,2),F(9,10),F(1),F(21,20),F(11,10),F(6,5)]
G=10**7
print("delta      best C            percent        m    eta")
for delta in [3e-7,1e-6,3e-6,6e-6,1e-5,1.5e-5,2e-5]:
    EPSr={s:F(math.floor((emin(float(s))-delta)*G),G) for s in SL}
    best=(F(0),0,None,None)
    for m in range(450,700):
        for k in range(900,1000):
            eta=F(k,1000); C,R=Cexact(SL,EPSr,m,eta)
            if C and C>best[0]: best=(C,m,eta,R)
    C,m,eta,R=best
    print("%.1e   %.15f  %.9f%%  %d  %s  %s"%(delta,float(C),100*float(C),m,eta,"OK>67.35" if float(C)>0.6735 else "BELOW"))
