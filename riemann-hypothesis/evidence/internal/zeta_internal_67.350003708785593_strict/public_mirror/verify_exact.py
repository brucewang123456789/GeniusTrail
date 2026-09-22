"""Exact-rational verification of the three-certificate q=8 witness."""
import json, math, sys
from fractions import Fraction as F
rat=json.load(open('config/cfgA_rat.json')); ex=json.load(open('config/locked3.json'))
Hf=F(*ex['H']); m=ex['m']; eta=F(*ex['eta']); Rc=F(*ex['R']); Cc=F(*ex['C'])
eps={F(k):F(*v) for k,v in ex['eps'].items()}
q=ex['q']; d=2*q; T=F(q+1,q); B=F(93,23000); ok=True
def ck(n,c):
    global ok; ok&=bool(c); print(("  PASS  " if c else "  FAIL  ")+n)
a={tuple(int(t) for t in k.split(',')):F(*v) for k,v in rat['a'].items()}
for sp in range(1,q+1): ck("span %d capacity == 2"%sp, sum(a[(i,i+sp)] for i in range(q+1-sp))==2)
ck("all a_ij >= 0", all(v>=0 for v in a.values()))
b=[F(*v) for v in rat['b']]
ck("B == 93/23000", sum(b)==B); ck("all b_r >= 0", all(v>=0 for v in b))
def sq(x,K=10**30):
    r=F(math.isqrt(x.numerator*K*K//x.denominator),K)
    while r*r>x: r-=F(1,K)
    return r
def h(E): return E if E<=T else E/(d+1)+F(2*d,d+1)*sq(E*T)-F(d,d+1)*T
ck("h(T) == T", h(T)==T)
n=m-q; lines=[(s,n*v) for s,v in eps.items()]; cand={F(0),T}
for i,(si,bi) in enumerate(lines):
    cand.add(bi/si)
    for sj,bj in lines[i+1:]:
        E=(bi-bj)/(si-sj)
        if E>0: cand.add(E)
worst=min(h(E)+eta*max([F(0)]+[bi-si*E for si,bi in lines]) for E in cand)
ck("R <= min over the complete kink set", Rc<=worst); ck("m - R > 0", m-Rc>0)
C=(m*Hf-eta*B*(m-q))/(m-Rc); ck("C reproduces the stored value", C==Cc)
ck("C > 67.35%", C>F(6735,10000))
print("\n  C = %d/%d"%(C.numerator,C.denominator))
print("  C = %.24f"%float(C)); print("  %% = %.15f"%(100*float(C)))
print("\nRESULT:", "ALL EXACT CHECKS PASS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
