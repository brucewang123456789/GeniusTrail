import numpy as np,json,os
exec(open('detev.py').read())
cands=[('o3',35,0),('cd7',35,0),('o5',35,0),('o6',43,8),('o7',58,16)]
best=(0,None,None)
for name,n,nf in cands:
    f=name+'.json'
    if not os.path.exists(f): continue
    x=np.array(json.load(open(f))['x'])
    om,c,b,A=mkcfg(x,nf); H,mv=Hfun(om,c)
    if H is None or mv<0.05: print(name,"invalid"); continue
    cf=Cfg(7,b,A); E,_=epsdet(cf,mkW(om,c))
    r=bestC(H,E,cf,range(300,1200,1))
    print("%-5s  H=%.9f minv=%.4f  DET ceiling=%.12f  m=%d eta=%.4f"%(name,H,mv,r[0],r[1],r[2]),flush=True)
    if r[0]>best[0]: best=(r[0],name,(x,nf))
print("\nWINNER:",best[1],"%.12f"%best[0])
json.dump({'name':best[1],'x':list(best[2][0]),'nf':best[2][1],'f':best[0]},open('winner.json','w'))
