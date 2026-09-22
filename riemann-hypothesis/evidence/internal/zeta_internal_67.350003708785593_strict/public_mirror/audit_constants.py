#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction as F
import json, math, re, sys
ROOT=Path(__file__).resolve().parent
cfg=(ROOT/'verifier'/'cfg8.h').read_text()
rat=json.load(open(ROOT/'cfg8_rat.json')) if (ROOT/'cfg8_rat.json').exists() else None
# cfg8_rat.h is authoritative generated rational header; cfgA_rat.json is copied by release builder.

def arr(name):
    m=re.search(rf'static const double {name}\[[^\]]+\]=\{{([^}}]+)\}};',cfg)
    if not m: raise SystemExit(f'missing {name}')
    return [float(x.strip()) for x in m.group(1).split(',')]
BR=arr('BR'); AW=arr('AW'); CJ=arr('CJ')
# Parse exact rationals from generated header.
h=(ROOT/'verifier'/'cfg8_rat.h').read_text()
def intarr(name):
    m=re.search(rf'static const long long {name}\[[^\]]+\]=\{{([^}}]+)\}};',h)
    if not m: raise SystemExit(f'missing {name}')
    return [int(x.strip()) for x in m.group(1).split(',')]
BN,BD=intarr('BNUM'),intarr('BDEN'); AN,AD=intarr('ANUM'),intarr('ADEN'); CN,CD=intarr('CNUM'),intarr('CDEN')
W0N=int(re.search(r'W0NUM\s*=\s*([0-9]+)',h).group(1)); W0D=int(re.search(r'W0DEN\s*=\s*([0-9]+)',h).group(1))

def ff(x): return F.from_float(x)
def dn(x): return math.nextafter(x,-math.inf)
def up(x): return math.nextafter(x, math.inf)
ok=True
def ck(label,cond):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ')+label)
for i,(d,n,q) in enumerate(zip(BR,BN,BD)):
    ex=F(n,q); ck(f'BR[{i}] lower-rounded <= exact b', ff(dn(d))<=ex)
for i,(d,n,q) in enumerate(zip(AW,AN,AD)):
    ex=F(n,q); ck(f'AW[{i}] lower-rounded <= exact a', ff(dn(d))<=ex)
for i,(d,n,q) in enumerate(zip(CJ,CN,CD)):
    ex=F(n,q); ck(f'CJ[{i}] one-ulp interval contains exact c', ff(dn(d))<=ex<=ff(up(d)))
w0=float(W0N)/float(W0D); half=0.5*w0; exhalf=F(W0N,2*W0D)
ck('omega0/2 one-ulp interval contains exact rational',ff(dn(half))<=exhalf<=ff(up(half)))
# pi enclosure exact decimal comparison at ample precision.
from decimal import Decimal, getcontext
getcontext().prec=80
PI=Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286')
P0=float('3.141592653589793115997963468544185161590576171875')
P1=float('3.141592653589793560087173318606801331043243408203125')
ck('P0 < pi < P1', Decimal.from_float(P0)<PI<Decimal.from_float(P1))
# Frozen slope/epsilon direction checks.
certs=[(F(1,2),F(526,78125)),(F(19,20),F(9879,1250000)),(F(1,1),F(20033,2500000))]
for s,e in certs:
    sf=float(s); ef=float(e)
    ck(f'slope {s}: dn(parsed) <= exact slope',ff(dn(sf))<=s)
    ck(f'eps {e}: up(parsed) >= exact epsilon',ff(up(ef))>=e)
print('RESULT:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
