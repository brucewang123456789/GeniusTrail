import json,math
from fractions import Fraction as F
from flint import arb, ctx
ctx.prec=300
cfg=json.load(open('cfgA_rat.json')); mins={F(k):v for k,v in json.load(open('minsA.json')).items()}
Aq=lambda pq: arb(pq[0])/arb(pq[1])
om0=Aq(cfg['omega0']); C=[Aq(v) for v in cfg['c']]; PIa=arb.pi()
OM=[om0]+[2*j*PIa for j in range(1,17)]
sinc=lambda z: z.sinc()
Cab=lambda a,b:(sinc((a-b)/2)+sinc((a+b)/2))/2
Aab=lambda a,b:((a/2).sin()/a+2*(a/2).cos()/(a*a))*sinc(b/2)-2*Cab(a,b)/(a*a)
I1=sum((C[j]*sinc(OM[j]/2) for j in range(17)),arb(0)); I2=arb(0);Jv=arb(0)
for i in range(17):
    for j in range(17):
        I2+=C[i]*C[j]*Cab(OM[i],OM[j]); Jv+=C[i]*C[j]*Aab(OM[i],OM[j])
H=2-(I2+Jv)/(I1*I1)
Hf=F(math.floor(float(H.lower())*10**15),10**15)
assert arb(Hf.numerator)/arb(Hf.denominator)<H
print("Arb  H >",Hf,"=",float(Hf))
N=20000; hb=arb(1)/(2*N); mn=1e9
for k in range(-N,N):
    t=arb(2*k+1)/(4*N); t=arb(t.mid(),hb.mid())
    mn=min(mn,float(sum((C[j]*(OM[j]*t).cos() for j in range(17)),arb(0)).lower()))
print("rigorous min v >",mn," positive:",mn>0)
q=8; d=16; T=F(9,8); B=F(93,23000); G=10**7
def sq(x,K=10**30):
    r=F(math.isqrt(x.numerator*K*K//x.denominator),K)
    while r*r>x: r-=F(1,K)
    return r
def h(E): return E if E<=T else E/(d+1)+F(2*d,d+1)*sq(E*T)-F(d,d+1)*T
SL=sorted(mins)
def Cx(EP,m,eta):
    n=m-q; L=[(s,n*EP[s]) for s in SL]; cd={F(0),T}
    for i,(si,bi) in enumerate(L):
        cd.add(bi/si)
        for sj,bj in L[i+1:]:
            E=(bi-bj)/(si-sj)
            if E>0: cd.add(E)
    R=min(h(E)+eta*max([F(0)]+[bi-si*E for si,bi in L]) for E in cd)
    return ((m*Hf-eta*B*n)/(m-R),R) if m-R>0 else (None,None)
print("\ndelta      C                  percent")
res={}
for dl in [1e-6,3e-6,6e-6,1e-5]:
    EP={s:F(math.floor((mins[s]-dl)*G),G) for s in SL}
    best=(F(0),0,None,None)
    for m in range(450,700):
        for k in range(920,1040):
            eta=F(k,1000); c,R=Cx(EP,m,eta)
            if c and c>best[0]: best=(c,m,eta,R)
    res[dl]=best+(EP,)
    print("%.0e    %.15f  %.9f%%  m=%d eta=%s"%(dl,float(best[0]),100*float(best[0]),best[1],best[2]))
dl=6e-6; Cc,m,eta,R,EP=res[dl]
json.dump({"q":8,"m":m,"eta":[eta.numerator,eta.denominator],"H":[Hf.numerator,Hf.denominator],
 "B":[93,23000],"R":[R.numerator,R.denominator],"C":[Cc.numerator,Cc.denominator],
 "eps":{str(s):[EP[s].numerator,EP[s].denominator] for s in SL},"delta":dl,
 "percent":"%.14f"%(100*float(Cc)),"status":"RESEARCH_DRAFT_PENDING_LOCAL_CERTIFICATES"},open('exactA.json','w'))
print("\nFINAL C = %d/%d"%(Cc.numerator,Cc.denominator)); print("      = %.24f"%float(Cc))
print("      = %.12f%%"%(100*float(Cc)))
