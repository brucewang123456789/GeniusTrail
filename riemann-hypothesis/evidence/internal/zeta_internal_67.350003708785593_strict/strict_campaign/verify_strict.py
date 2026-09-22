#!/usr/bin/env python3
"""Fail-closed verifier for the frozen 67.350003708785593% strict campaign.
Checks: exact rational assembly, strict terminal states, well-certificate batch, and constants audit.
"""
from pathlib import Path
from fractions import Fraction as F
import json, math, re, sys
ROOT=Path(__file__).resolve().parent
ok=True

def ck(name, cond):
    global ok
    cond=bool(cond); ok &= cond
    print(('PASS ' if cond else 'FAIL ')+name)

# Exact witness.
q=8; m=531; eta=F(1,1); H=F(672167187145431,10**15); B=F(93,23000)
eps={F(1,2):F(526,78125),F(19,20):F(9879,1250000),F(1,1):F(20033,2500000)}
Rlock=F(1113172768314043426732876281699,265625000000000000000000000000)
Clock=F(722547711262091300265625000000000,1072825050442925667061714615173641)
d=2*q; T=F(q+1,q)
def sq_floor(x,K=10**30):
    r=F(math.isqrt(x.numerator*K*K//x.denominator),K)
    while r*r>x:r-=F(1,K)
    return r
def h(E):
    return E if E<=T else E/F(d+1)+F(2*d,d+1)*sq_floor(E*T)-F(d,d+1)*T
n=m-q; lines=[(s,n*e) for s,e in eps.items()]; cand={F(0),T}
for i,(si,bi) in enumerate(lines):
    cand.add(bi/si)
    for sj,bj in lines[i+1:]:
        x=(bi-bj)/(si-sj)
        if x>0:cand.add(x)
worst=min(h(E)+eta*max([F(0)]+[bi-si*E for si,bi in lines]) for E in cand)
ck('R lock is a valid rational lower bound of the complete kink minimum',Rlock<=worst)
C=(m*H-eta*B*(m-q))/(m-Rlock)
ck('exact C equals frozen fraction',C==Clock)
ck('exact C > 0.6735',C>F(6735,10000))

# Independent well checker.
wlog=ROOT/'wells_v2.log'
if wlog.exists():
    t=wlog.read_text(errors='replace')
    ck('327/327 strict well boxes proved',bool(re.search(r'WELLS_V2_TOTAL=327 PROVED=327 FAILED=0 .* RESULT=PROVED',t)))
else: ck('wells_v2.log exists',False)

# Constant-direction audit.
alog=ROOT/'audit_constants.log'
if alog.exists():
    t=alog.read_text(errors='replace')
    ck('constant/rational enclosure audit passed','RESULT: PASS' in t and 'FAIL ' not in t)
else: ck('audit_constants.log exists',False)

# Three local certificate logs. Prefer final bb26 names, but allow s0.5 strict no-well fallback.
expected=[('s0.5',F(1,2),F(526,78125)),('s0.95',F(19,20),F(9879,1250000)),('s1',F(1,1),F(20033,2500000))]
node_total=0
for name,s,e in expected:
    choices=[ROOT/f'{name}_final.log',ROOT/f'{name}_bb26.log',ROOT/f'{name}.log']
    p=next((x for x in choices if x.exists()),None)
    if p is None:
        ck(f'{name}: certificate log exists',False); continue
    t=p.read_text(errors='replace')
    ms=re.findall(r'slope=([0-9.]+) eps=([0-9.eE+-]+) nodes=(\d+) stack_left=(\d+) HARD=(\d+).*result=(PROVED|INCOMPLETE)',t)
    if not ms:
        ck(f'{name}: terminal certificate line found',False); continue
    sl,ee,nodes,stack,hard,res=ms[-1]
    ck(f'{name}: fail-closed terminal state',res=='PROVED' and stack=='0' and hard=='0')
    # Directional numeric parse only; exact targets are audited separately.
    ck(f'{name}: slope label matches',abs(float(sl)-float(s))<1e-12)
    ck(f'{name}: epsilon label matches',abs(float(ee)-float(e))<1e-12)
    node_total+=int(nodes)
print('TOTAL_STRICT_BB_NODES=',node_total)
print('C=',f'{C.numerator}/{C.denominator}')
print('PERCENT=',format(float(C*100),'.15f'))
print('RESULT:', 'ALL STRICT 67.35 CHECKS PASS' if ok else 'FAILURE')
sys.exit(0 if ok else 1)
