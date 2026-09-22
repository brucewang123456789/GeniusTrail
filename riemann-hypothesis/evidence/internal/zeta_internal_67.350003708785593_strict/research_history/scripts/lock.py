import json,math
from fractions import Fraction as F
exec(open('marg.py').read().split('SLall=')[0])
mins={F(k):v for k,v in json.load(open('minsA.json')).items()}
G=10**7; TGT=F(6735,10000)
S=[F('1/2'),F('19/20'),F('1'),F('21/20')]
EPS_half=F(67328,10**7)          # PROVED by bb8h run
def bC(EP):
    bb=(F(0),0,None)
    for m in range(515,600):
        for k in range(950,1060):
            eta=F(k,1000); c,R=Cx(EP,m,eta)
            if c and c>bb[0]: bb=(c,m,eta)
    return bb
lo,hi=0.0,6e-5
for _ in range(24):
    mid=(lo+hi)/2
    EP={s:F(math.floor((mins[s]-mid)*G),G) for s in S if s!=F('1/2')}
    EP[F('1/2')]=EPS_half
    c,m,eta=bC(EP)
    if c and c>TGT: lo=mid
    else: hi=mid
EP={s:F(math.floor((mins[s]-lo)*G),G) for s in S if s!=F('1/2')}
EP[F('1/2')]=EPS_half
c,m,eta=bC(EP); R=Cx(EP,m,eta)[1]
print("LOCKED WITNESS (eps_{1/2} at the value already proved)")
print("  C = %d/%d"%(c.numerator,c.denominator))
print("  C = %.18f  = %.12f%%   (> 67.35%%: %s)"%(float(c),100*float(c),c>TGT))
print("  m=%d  eta=%s   uniform margin on the other three = %.4e"%(m,eta,lo))
for s in sorted(EP,key=lambda z:float(z)):
    st="PROVED (43,685,299 nodes)" if s==F('1/2') else "open"
    print("   eps(%-6s)= %-22s float-min %.10f  margin %.3e   %s"%(s,"%d/%d"%(EP[s].numerator,EP[s].denominator),mins[s],mins[s]-float(EP[s]),st))
json.dump({"q":8,"m":m,"eta":[eta.numerator,eta.denominator],"H":[672167187145431,10**15],
 "R":[R.numerator,R.denominator],"C":[c.numerator,c.denominator],
 "eps":{str(s):[EP[s].numerator,EP[s].denominator] for s in EP},
 "proved":["1/2"],"open":["19/20","1","21/20"],
 "percent":"%.15f"%(100*float(c)),
 "status":"ONE_OF_FOUR_LOCAL_CERTIFICATES_CLOSED"},open('locked.json','w'))
