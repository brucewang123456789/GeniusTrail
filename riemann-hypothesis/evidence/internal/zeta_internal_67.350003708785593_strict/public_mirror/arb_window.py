import json, math
from flint import arb, ctx
from fractions import Fraction as F
ctx.prec = 300
cfg=json.load(open('config/cfgA_rat.json'))
def A_(pq): return arb(pq[0])/arb(pq[1])
om0=A_(cfg['omega0']); C=[A_(v) for v in cfg['c']]
PI=arb.pi()
OM=[om0]+[2*j*PI for j in range(1,17)]
def sinc(z): return z.sinc()
def Cab(a,b): return (sinc((a-b)/2)+sinc((a+b)/2))/2
def Aab(a,b):
    return ((a/2).sin()/a + 2*(a/2).cos()/(a*a))*sinc(b/2) - 2*Cab(a,b)/(a*a)
I1=sum((C[j]*sinc(OM[j]/2) for j in range(17)), arb(0))
I2=arb(0); J=arb(0)
for i in range(17):
    for j in range(17):
        I2 += C[i]*C[j]*Cab(OM[i],OM[j])
        J  += C[i]*C[j]*Aab(OM[i],OM[j])
H=2-(I2+J)/(I1*I1)
print("I1 =",I1); print("I2 =",I2); print("J  =",J)
print("H  =",H)
lo=H - arb(H.rad())   # conservative
print("H lower bound (float) = %.18g"%float(H.lower()))
# rational floor at 1e-15
Hf=F(math.floor(float(H.lower())*10**15),10**15)
print("H_floor rational =",Hf, "=", float(Hf), " valid:", arb(Hf.numerator)/arb(Hf.denominator) < H)

# --- rigorous positivity of v on [-1/2,1/2] ---
N=20000
h=arb(1)/(2*N)
mn=None
for k in range(-N,N):
    t=arb(2*k+1)/(4*N)
    t=arb(t); t=arb(t.mid(), h.mid())    # ball of radius h
    v=sum((C[j]*(OM[j]*t).cos() for j in range(17)), arb(0))
    l=float(v.lower())
    if mn is None or l<mn: mn=l
print("rigorous min v over [-1/2,1/2] >= %.10f"%mn, " (positive:", mn>0, ")")
