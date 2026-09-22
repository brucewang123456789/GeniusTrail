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
    lay(1,e(x[21:25]));lay(2,e(x[25:28]));lay(3,e(x[28:31]));lay(4,e(x[31:33]));lay(5,e(x[33:35]));lay(6,[1.0]);lay(7,[1.0])
    return om,c,b,A
def ev(x):
    om,c,b,A=unpack(x); H,mv=Hfun(om,c)
    if H is None or mv<0.05: return -1
    cf=Cfg(7,b,A); return bestC(H,epscurve3(cf,mkW(om,c)),cf,range(300,1150,4))[0]
sc=np.concatenate([[0.02],np.maximum(np.abs(c0[1:]),8e-4),np.full(18,0.3)])
st=json.load(open('o5.json')) if os.path.exists('o5.json') else json.load(open('o3.json'))
xb=np.array(st['x']); fb=st.get('f',None)
if fb is None or 'o5.json' not in os.listdir('.'): fb=ev(xb)
print("start %.12f"%fb,flush=True)
rng=np.random.default_rng(int(time.time())%9001); t0=time.time(); sg=0.25; succ=0; tot=0
while time.time()-t0<235:
    d=rng.normal(0,1,35); d/=np.linalg.norm(d); step=sg*sc*d
    cand=[xb+step, xb-step, xb+2*step, xb-2*step]
    vals=[ev(x) for x in cand]; tot+=1
    k=int(np.argmax(vals))
    if vals[k]>fb:
        fb=vals[k]; xb=cand[k]; succ+=1; sg*=1.3
        print("  + %.12f  [%d/%.0fs]"%(fb,tot,time.time()-t0),flush=True)
    else: sg*=0.85
    if sg<1e-4: sg=0.2
json.dump({'x':list(xb),'f':fb},open('o5.json','w'))
print("BEST %.12f  (sym-compass was 0.673482502)  iters=%d succ=%d"%(fb,tot,succ))
