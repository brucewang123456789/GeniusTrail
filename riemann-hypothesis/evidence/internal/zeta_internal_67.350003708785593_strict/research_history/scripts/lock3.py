import json,math
from fractions import Fraction as F
exec(open('marg.py').read().split('SLall=')[0])
mins={F(k):v for k,v in json.load(open('minsA.json')).items()}
G=10**7; TGT=F(6735,10000); EH=F(67328,10**7)
S=[F('1/2'),F('19/20'),F('1')]
def bC(EP):
    bb=(F(0),0,None)
    for m in range(500,620):
        for k in range(940,1070):
            eta=F(k,1000); c,R=Cx(EP,m,eta)
            if c and c>bb[0]: bb=(c,m,eta)
    return bb
lo,hi=0.0,6e-5
for _ in range(24):
    mid=(lo+hi)/2
    EP={s:F(math.floor((mins[s]-mid)*G),G) for s in S if s!=F('1/2')}; EP[F('1/2')]=EH
    c,m,eta=bC(EP)
    if c and c>TGT: lo=mid
    else: hi=mid
EP={s:F(math.floor((mins[s]-lo)*G),G) for s in S if s!=F('1/2')}; EP[F('1/2')]=EH
c,m,eta=bC(EP); R=Cx(EP,m,eta)[1]
print("THREE-CERTIFICATE LOCKED WITNESS")
print("  C = %d/%d"%(c.numerator,c.denominator))
print("  = %.15f%%   >67.35%%: %s   m=%d eta=%s"%(100*float(c),c>TGT,m,eta))
for s in sorted(EP,key=lambda z:float(z)):
    print("   eps(%-5s)=%-20s margin %.3e  %s"%(s,"%d/%d"%(EP[s].numerator,EP[s].denominator),
          mins[s]-float(EP[s]),"PROVED" if s==F('1/2') else "open"))
json.dump({"q":8,"m":m,"eta":[eta.numerator,eta.denominator],"H":[672167187145431,10**15],
 "R":[R.numerator,R.denominator],"C":[c.numerator,c.denominator],
 "eps":{str(s):[EP[s].numerator,EP[s].denominator] for s in EP},
 "proved":["1/2"],"open":[str(s) for s in S if s!=F('1/2')],
 "percent":"%.15f"%(100*float(c))},open('locked3.json','w'))
