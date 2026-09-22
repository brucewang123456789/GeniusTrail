import json,math
from fractions import Fraction as F
mins={F(k):v for k,v in json.load(open('minsA.json')).items()}
Hf=F(672167187145431,10**15); q=8; d=16; T=F(9,8); B=F(93,23000)
def sq(x,K=10**30):
    r=F(math.isqrt(x.numerator*K*K//x.denominator),K)
    while r*r>x: r-=F(1,K)
    return r
def h(E): return E if E<=T else E/(d+1)+F(2*d,d+1)*sq(E*T)-F(d,d+1)*T
def Cx(EP,m,eta):
    n=m-q; L=[(s,n*EP[s]) for s in EP]; cd={F(0),T}
    for i,(si,bi) in enumerate(L):
        cd.add(bi/si)
        for sj,bj in L[i+1:]:
            E=(bi-bj)/(si-sj)
            if E>0: cd.add(E)
    R=min(h(E)+eta*max([F(0)]+[bi-si*E for si,bi in L]) for E in cd)
    return ((m*Hf-eta*B*n)/(m-R),R) if m-R>0 else (None,None)
def best(EP,mr=range(440,760),kr=range(880,1120)):
    bb=(F(0),0,None,None)
    for m in mr:
        for k in kr:
            eta=F(k,1000); c,R=Cx(EP,m,eta)
            if c and c>bb[0]: bb=(c,m,eta,R)
    return bb
SLall=sorted(mins); G=10**7
print("uniform delta sweep (all 8 slopes):")
for dl in [1e-5,2e-5,3e-5,4e-5,5e-5,6e-5]:
    EP={s:F(math.floor((mins[s]-dl)*G),G) for s in SLall}
    c,m,eta,R=best(EP)
    print("  %.0e  C=%.15f  %.9f%%  %s  m=%d eta=%s"%(dl,float(c),100*float(c),"OK" if c>F(6735,10000) else "below",m,eta))
