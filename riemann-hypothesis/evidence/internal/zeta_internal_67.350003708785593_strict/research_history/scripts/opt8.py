import numpy as np,math,json,time,os
exec(open('ev4.py').read())
c0=np.array([1000000000,12378982,-12602495,4164033,5741405,-1724025,
6219280,-8047828,6321519,-5241981,-892658,560544,-431207,357969,-310433,100000,-100000],float)/1e9
NEW=6   # extra free-(omega,c) window terms
# x: 0 om0 | 1..16 c | 17..20 log b | 21..34 a | 35..40 new omegas | 41..46 new c
def unpack(x):
    om=np.concatenate([[x[0]],[2*j*math.pi for j in range(1,17)],x[35:35+NEW]])
    c=np.concatenate([[1.0],x[1:17],x[35+NEW:35+2*NEW]])
    e=lambda v: np.exp(np.clip(v,-25,25))
    bp=e(x[17:21]); b=np.concatenate([bp,bp[-2::-1]]); b=b/b.sum()*(93/23000)
    A={}
    def lay(sp,vals):
        n=8-sp; ws=np.array([vals[min(i,n-1-i)] for i in range(n)],float); ws=ws/ws.sum()*2
        for i in range(n): A[(i,i+sp)]=ws[i]
    lay(1,e(x[21:25]));lay(2,e(x[25:28]));lay(3,e(x[28:31]));lay(4,e(x[31:33]));lay(5,e(x[33:35]));lay(6,[1.0]);lay(7,[1.0])
    return om,c,b,A
def ev(x):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1
    cf=Cfg(7,b,A); return bestC(H,epsfast(cf,mkW(om,c)),cf,range(320,1150,4))[0]
N=35+2*NEW
if os.path.exists('o8.json'):
    st=json.load(open('o8.json')); xb=np.array(st['x']); fb=st['f']
else:
    y=np.array(json.load(open('o5.json'))['x'])
    xb=np.zeros(N); xb[:35]=y
    xb[35:35+NEW]=np.array([34.0,36.0,38.0,40.0,42.0,44.0])   # new frequencies beyond 32pi
    xb[35+NEW:]=1e-5
    fb=ev(xb); print("start %.12f  (o5 det ceiling 0.673487177)"%fb,flush=True)
sc=np.concatenate([[0.02],np.maximum(np.abs(c0[1:]),8e-4),np.full(18,0.3),np.full(NEW,0.6),np.full(NEW,3e-4)])
rng=np.random.default_rng(int(time.time())%9001); t0=time.time(); sg=0.35; tot=0;succ=0
while time.time()-t0<240:
    d=rng.normal(0,1,N)
    if rng.random()<0.5: d[:35]*=0.15          # focus the new basis block
    d/=np.linalg.norm(d); step=sg*sc*d
    cand=[xb+step,xb-step,xb+2.5*step,xb-2.5*step]; vals=[ev(z) for z in cand]; tot+=1
    k=int(np.argmax(vals))
    if vals[k]>fb: fb=vals[k];xb=cand[k];succ+=1;sg*=1.35;print("  + %.12f [%d/%.0fs]"%(fb,tot,time.time()-t0),flush=True)
    else: sg*=0.82
    if sg<8e-5: sg=0.3
json.dump({'x':list(xb),'f':fb},open('o8.json','w'))
print("BEST %.12f iters=%d succ=%d"%(fb,tot,succ))
