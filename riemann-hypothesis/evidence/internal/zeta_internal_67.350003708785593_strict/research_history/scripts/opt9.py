import numpy as np,math,json,time,os
exec(open('ev8.py').read())
c0=np.array([1000000000,12378982,-12602495,4164033,5741405,-1724025,
6219280,-8047828,6321519,-5241981,-892658,560544,-431207,357969,-310433,100000,-100000],float)/1e9
# x: 0 om0 | 1..16 c | 17..20 log b(4 -> palindromic 8? b has 8 entries) | 21.... a-shape spans1..7
# b: 8 entries palindromic -> 4 free ; a spans 1..8 on 9 points: halves 4,4,3,3,2,2,1,(1 forced)
BL=[4,4,3,3,2,2,1]
IDX={}; p=21
for sp,n in enumerate(BL,1): IDX[sp]=(p,p+n); p+=n
N=p
def unpack(x):
    om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)]); c=np.concatenate([[1.0],x[1:17]])
    e=lambda v: np.exp(np.clip(v,-25,25))
    bp=e(x[17:21]); b=np.concatenate([bp,bp[::-1]]); b=b/b.sum()*(93/23000)
    A={}
    def lay(sp,vals):
        n=9-sp; ws=np.array([vals[min(i,n-1-i)] for i in range(n)],float); ws=ws/ws.sum()*2
        for i in range(n): A[(i,i+sp)]=ws[i]
    for sp in range(1,8):
        lo,hi=IDX[sp]; lay(sp,e(x[lo:hi]))
    lay(8,[1.0])
    return om,c,b,A
def ev(x):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1
    cf=Cfg(8,b,A); return bestC(H,eps8(cf,mkW(om,c)),cf,range(350,1300,5))[0]
if os.path.exists('o9.json'):
    st=json.load(open('o9.json')); xb=np.array(st['x']); fb=st['f']
else:
    y=np.array(json.load(open('o5.json'))['x'])
    xb=np.zeros(N); xb[:17]=y[:17]
    xb[17:21]=np.log([1.0,1.55,1.85,2.0])
    seed={1:[1,1.35,1.65,1.8],2:[0.6,1.3,1.8,1.9],3:[0.7,1.4,1.7],4:[0.8,1.4,1.6],5:[1.0,1.3],6:[1.0,1.2],7:[1.0]}
    for sp in range(1,8):
        lo,hi=IDX[sp]; xb[lo:hi]=np.log(seed[sp])
    fb=ev(xb); print("q8 start %.12f  (q7 best 0.673489246)"%fb,flush=True)
sc=np.concatenate([[0.02],np.maximum(np.abs(c0[1:]),8e-4),np.full(N-17,0.35)])
rng=np.random.default_rng(int(time.time())%9001); t0=time.time(); sg=0.5; tot=0;succ=0
while time.time()-t0<235:
    d=rng.normal(0,1,N)
    if rng.random()<0.6: d[:17]*=0.15
    d/=np.linalg.norm(d); step=sg*sc*d
    cand=[xb+step,xb-step,xb+2.5*step,xb-2.5*step]; vals=[ev(z) for z in cand]; tot+=1
    k=int(np.argmax(vals))
    if vals[k]>fb: fb=vals[k];xb=cand[k];succ+=1;sg*=1.35;print("  + %.12f [%d/%.0fs]"%(fb,tot,time.time()-t0),flush=True)
    else: sg*=0.82
    if sg<8e-5: sg=0.4
json.dump({'x':list(xb),'f':fb},open('o9.json','w'))
print("BEST-q8 %.12f iters=%d succ=%d"%(fb,tot,succ))
