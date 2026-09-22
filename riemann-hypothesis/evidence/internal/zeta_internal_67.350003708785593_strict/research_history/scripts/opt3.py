import numpy as np,math,json,time,os
exec(open('ev3.py').read())
src=open('gen.py').read(); exec(src[src.index("def bestC"):])
c0=np.array([1000000000,12378982,-12602495,4164033,5741405,-1724025,
6219280,-8047828,6321519,-5241981,-892658,560544,-431207,357969,-310433,100000,-100000],float)/1e9
def unpack(x):
    om=np.array([x[0]]+[2*j*math.pi for j in range(1,17)]); c=np.concatenate([[1.0],x[1:17]])
    e=lambda v: np.exp(np.clip(v,-25,25))
    bp=e(x[17:21]); b=np.concatenate([bp,bp[-2::-1]]); b=b/b.sum()*(93/23000)
    A={}
    def lay(sp,vals):
        n=8-sp; ws=np.array([vals[min(i,n-1-i)] for i in range(n)],float); ws=ws/ws.sum()*2
        for i in range(n): A[(i,i+sp)]=ws[i]
    lay(1,e(x[21:25])); lay(2,e(x[25:28])); lay(3,e(x[28:31])); lay(4,e(x[31:33]))
    lay(5,e(x[33:35])); lay(6,[1.0]); lay(7,[1.0])
    return om,c,b,A
def ev(x):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1,None
    cf=Cfg(7,b,A); r=bestC(H,epscurve3(cf,mkW(om,c)),cf,range(300,1150,4))
    return r[0],(H,r[1],r[2])
x0=np.array(json.load(open('cd7.json'))['x'])
sc=np.concatenate([[0.02],np.maximum(np.abs(c0[1:]),8e-4),np.full(18,0.3)])
if os.path.exists('o3.json'):
    st=json.load(open('o3.json')); xb=np.array(st['x']); step=np.array(st['step']); fb=st['f']
else:
    fb,_=ev(x0); xb=x0; step=np.full(35,0.4); print("start (corrected evaluator) %.12f"%fb,flush=True)
t0=time.time(); rng=np.random.default_rng(11); order=list(range(35))
while time.time()-t0<225:
    rng.shuffle(order)
    for d in order:
        if time.time()-t0>225: break
        got=False
        for sg in (1,-1):
            x=xb.copy(); x[d]+=sg*step[d]*sc[d]
            f,i=ev(x)
            if f>fb: fb=f;xb=x;step[d]*=1.6;got=True;print("  + %.12f d=%d [%.0fs]"%(fb,d,time.time()-t0),flush=True);break
        if not got: step[d]*=0.55
json.dump({'x':list(xb),'step':list(step),'f':fb},open('o3.json','w'))
print("BEST(corrected) %.12f   frozen-certified 0.673457487089"%fb)
