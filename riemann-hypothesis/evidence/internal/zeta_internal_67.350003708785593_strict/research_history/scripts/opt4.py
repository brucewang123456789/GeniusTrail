import numpy as np,math,json,time,os
exec(open('ev3.py').read())
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
c0=np.array([1000000000,12378982,-12602495,4164033,5741405,-1724025,
6219280,-8047828,6321519,-5241981,-892658,560544,-431207,357969,-310433,100000,-100000],float)/1e9
# x = [om0, c1..c16] + b(7) + spans: 7,6,5,4,3,2 free-log values  -> 17+7+27 = 51 slots (span7 forced)
OFF=[17]; 
for n in [7,6,5,4,3,2,1]: pass
SPAN=[7,6,5,4,3,2,1]          # entries per span 1..7
IDX={}; p=17+7
for sp,n in enumerate(SPAN,1):
    IDX[sp]=(p,p+n); p+=n
NX=p
def unpack(x):
    om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)]); c=np.concatenate([[1.0],x[1:17]])
    e=lambda v: np.exp(np.clip(v,-25,25))
    b=e(x[17:24]); b=b/b.sum()*(93/23000)
    A={}
    for sp,n in enumerate(SPAN,1):
        lo,hi=IDX[sp]; w=e(x[lo:hi]); w=w/w.sum()*2
        for i in range(n): A[(i,i+sp)]=w[i]
    return om,c,b,A
def ev(x):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1,None
    cf=Cfg(7,b,A); r=bestC(H,epscurve3(cf,mkW(om,c)),cf,range(300,1150,4))
    return r[0],(H,r[1],r[2])
if os.path.exists('o4.json'):
    st=json.load(open('o4.json')); xb=np.array(st['x']); step=np.array(st['step']); fb=st['f']
else:
    y=np.array(json.load(open('o3.json'))['x'])
    e=lambda v: np.exp(np.clip(v,-25,25))
    bp=e(y[17:21]); b=np.concatenate([bp,bp[-2::-1]])
    x0=np.zeros(NX); x0[:17]=y[:17]; x0[17:24]=np.log(b)
    def lay(sp,vals):
        n=8-sp; return np.array([vals[min(i,n-1-i)] for i in range(n)],float)
    src2={1:e(y[21:25]),2:e(y[25:28]),3:e(y[28:31]),4:e(y[31:33]),5:e(y[33:35]),6:np.array([1.0]),7:np.array([1.0])}
    for sp,n in enumerate(SPAN,1):
        lo,hi=IDX[sp]; x0[lo:hi]=np.log(lay(sp,src2[sp]))
    fb,i0=ev(x0); xb=x0; step=np.full(NX,0.35)
    print("asym start %.12f (sym best was 0.673482502)"%fb,flush=True)
sc=np.concatenate([[0.02],np.maximum(np.abs(c0[1:]),8e-4),np.full(NX-17,0.3)])
t0=time.time(); rng=np.random.default_rng(int(time.time())%997); order=list(range(NX))
while time.time()-t0<235:
    rng.shuffle(order)
    for d in order:
        if time.time()-t0>235: break
        got=False
        for sg in (1,-1):
            x=xb.copy(); x[d]+=sg*step[d]*sc[d]
            f,i=ev(x)
            if f>fb: fb=f;xb=x;step[d]*=1.6;got=True;print("  + %.12f d=%d [%.0fs]"%(fb,d,time.time()-t0),flush=True);break
        if not got: step[d]*=0.55
json.dump({'x':list(xb),'step':list(step),'f':fb},open('o4.json','w'))
print("BEST-asym %.12f"%fb)
