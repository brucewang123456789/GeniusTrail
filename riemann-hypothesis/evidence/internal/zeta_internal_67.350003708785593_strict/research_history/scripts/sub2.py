import json,math
from fractions import Fraction as F
exec(open('marg.py').read().split('SLall=')[0])
mins={F(k):v for k,v in json.load(open('minsA.json')).items()}
G=10**7; TGT=F(6735,10000)
def bC(EP):
    bb=(F(0),0,None)
    for m in range(515,600,2):
        for k in range(950,1070,2):
            eta=F(k,1000); c,R=Cx(EP,m,eta)
            if c and c>bb[0]: bb=(c,m,eta)
    return bb
def maxd(S):
    lo,hi=0.0,1.5e-4
    for _ in range(16):
        mid=(lo+hi)/2
        EP={s:F(math.floor((mins[s]-mid)*G),G) for s in S}
        c,m,eta=bC(EP)
        if c and c>TGT: lo=mid
        else: hi=mid
    EP={s:F(math.floor((mins[s]-lo)*G),G) for s in S}
    return (lo,)+bC(EP)+(EP,)
fs=lambda *a:[F(x) for x in a]
sets=[fs('19/20','1'),fs('19/20','21/20'),fs('1','21/20'),fs('4/5','1'),
      fs('1/2','19/20','1'),fs('1/2','19/20','21/20'),fs('1/2','1','21/20'),
      fs('1/2','19/20','1','21/20'),fs('4/5','19/20','1','21/20')]
out=[]
for S in sets:
    dl,c,m,eta,EP=maxd(S)
    if c>TGT: out.append((dl,S,c,m,eta,EP))
out.sort(key=lambda z:-z[0])
print("max uniform margin  |  certs  |  C  |  witness")
for dl,S,c,m,eta,EP in out:
    print("  %.3e   %d  %-30s  %.12f%%  m=%d eta=%s"%(dl,len(S),str([str(x) for x in S]),100*float(c),m,eta))
dl,S,c,m,eta,EP=out[0]
print("\nBEST proof-optimal witness:")
print("  C = %d/%d = %.15f%%"%(c.numerator,c.denominator,100*float(c)))
print("  m=%d  eta=%s  margin=%.3e on every certified slope"%(m,eta,dl))
for s in S: print("   eps(%s) = %d/%d   float-min %.10f   margin %.3e"%(s,EP[s].numerator,EP[s].denominator,mins[s],mins[s]-float(EP[s])))
json.dump({"q":8,"m":m,"eta":[eta.numerator,eta.denominator],"H":[672167187145431,10**15],
 "eps":{str(s):[EP[s].numerator,EP[s].denominator] for s in S},"C":[c.numerator,c.denominator],
 "margin":dl,"percent":"%.15f"%(100*float(c))},open('proofopt.json','w'))
