import json,math
from fractions import Fraction as F
exec(open('marg.py').read().split('SLall=')[0])
mins={F(k):v for k,v in json.load(open('minsA.json')).items()}
G=10**7; TGT=F(6735,10000)
S=[F('1/2'),F('19/20'),F('1'),F('21/20')]
def bC(EP):
    bb=(F(0),0,None)
    for m in range(515,600,1):
        for k in range(950,1060,1):
            eta=F(k,1000); c,R=Cx(EP,m,eta)
            if c and c>bb[0]: bb=(c,m,eta)
    return bb
base={s:F(math.floor((mins[s]-3.0839e-5)*G),G) for s in S}
print("base C = %.12f%%"%(100*float(bC(base)[0])))
alloc={}
for s in S:
    lo,hi=3.0839e-5,2e-2
    for _ in range(20):
        mid=(lo+hi)/2
        EP=dict(base); EP[s]=F(math.floor((mins[s]-mid)*G),G)
        c,_,_=bC(EP)
        if c and c>TGT: lo=mid
        else: hi=mid
    alloc[s]=lo
    print("  eps(%-6s) can drop to %.7f   margin %.3e  (x%.1f of uniform)"%(s,mins[s]-lo,lo,lo/3.0839e-5))
