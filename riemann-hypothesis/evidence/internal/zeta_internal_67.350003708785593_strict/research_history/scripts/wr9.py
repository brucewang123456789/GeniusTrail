import json,math
from fractions import Fraction as F
mins={F(k):v for k,v in json.load(open('mins9.json')).items()}
Hf=F(10502983278473,15625000000000); T=F(8,7); B=F(93,23000); G=10**7; d=6e-6
def sq(x,K=10**30):
    r=F(math.isqrt(x.numerator*K*K//x.denominator),K)
    while r*r>x: r-=F(1,K)
    return r
def h(E): return E if E<=T else E/15+F(28,15)*sq(E*T)-F(14,15)*T
SL=sorted(mins); EP={s:F(math.floor((mins[s]-d)*G),G) for s in SL}
def Cx(m,eta):
    n=m-7; L=[(s,n*EP[s]) for s in SL]; cd={F(0),T}
    for i,(si,bi) in enumerate(L):
        cd.add(bi/si)
        for sj,bj in L[i+1:]:
            E=(bi-bj)/(si-sj)
            if E>0: cd.add(E)
    R=min(h(E)+eta*max([F(0)]+[bi-si*E for si,bi in L]) for E in cd)
    return ((m*Hf-eta*B*n)/(m-R),R) if m-R>0 else (None,None)
best=(F(0),0,None,None)
for m in range(560,680):
    for k in range(880,1000):
        eta=F(k,1000); c,R=Cx(m,eta)
        if c and c>best[0]: best=(c,m,eta,R)
C,m,eta,R=best
json.dump({"m":m,"q":7,"eta":[eta.numerator,eta.denominator],"H":[Hf.numerator,Hf.denominator],
 "B":[93,23000],"R":[R.numerator,R.denominator],"C":[C.numerator,C.denominator],
 "eps":{str(s):[EP[s].numerator,EP[s].denominator] for s in SL},"delta":d,
 "percent":"%.14f"%(100*float(C)),"status":"RESEARCH_DRAFT_PENDING_LOCAL_CERTIFICATES"},open('exact9.json','w'))
print("m=%d eta=%s"%(m,eta)); print("C = %d/%d"%(C.numerator,C.denominator))
print("C = %.24f"%float(C)); print("%% = %.12f"%(100*float(C)))
for s in SL: print("  eps(%s)=%s  margin %.2e"%(s,EP[s],mins[s]-float(EP[s])))
