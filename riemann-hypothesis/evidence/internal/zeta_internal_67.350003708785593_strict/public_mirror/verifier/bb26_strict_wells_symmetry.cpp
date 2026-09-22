#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "cfg8.h"
#include "wells_generated.h"
static inline double dn(double x){return std::nextafter(x,-INFINITY);}
static inline double up(double x){return std::nextafter(x, INFINITY);}
static inline float fdn(double x){ float f=(float)x; if((double)f>x) f=std::nextafterf(f,-INFINITY); return std::nextafterf(f,-INFINITY); }
static inline float fup(double x){ float f=(float)x; if((double)f<x) f=std::nextafterf(f, INFINITY); return std::nextafterf(f, INFINITY); }
static inline double alo(double a,double b){return dn(a+b);}
static inline double ahi(double a,double b){return up(a+b);}
static inline double slo(double a,double b){return dn(a-b);}
static inline double shi(double a,double b){return up(a-b);}
static inline double mlo(double a,double b){return dn(a*b);}
static inline double mhi(double a,double b){return up(a*b);}
struct I{double lo,hi;};
static inline I mk(double a,double b){return I{dn(dn(a)),up(up(b))};}
static inline I add(I a,I b){return mk(a.lo+b.lo,a.hi+b.hi);}
static inline I sub(I a,I b){return mk(a.lo-b.hi,a.hi-b.lo);}
static inline I mul(I a,I b){double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
  return mk(*std::min_element(p,p+4),*std::max_element(p,p+4));}
static inline I scl(double k,I a){return k>=0?mk(k*a.lo,k*a.hi):mk(k*a.hi,k*a.lo);}
static I dvi(I a,I b){ if(b.lo<=0&&b.hi>=0) return I{-1e300,1e300};
  double p[4]={a.lo/b.lo,a.lo/b.hi,a.hi/b.lo,a.hi/b.hi};
  return mk(*std::min_element(p,p+4),*std::max_element(p,p+4));}
// Rigorous sin/cos enclosure using only basic IEEE-754 operations, nextafter,
// a one-ulp enclosure of pi, quadrant reduction, and Taylor polynomials on
// |r| <= pi/4.  No libm sin/cos accuracy assumption enters the certificate.
static I coeff_inv_fact(int n,int sign){
  unsigned long long f=1; for(int k=2;k<=n;k++) f*= (unsigned long long)k;
  I den=mk((double)f,(double)f); return dvi(mk((double)sign,(double)sign),den);
}
static void small_sincos(I r,I&S,I&C){
  I r2=mul(r,r);
  I ps=coeff_inv_fact(17,+1); // (-1)^8 / 17!
  I pc=coeff_inv_fact(16,+1); // (+1) / 16!
  for(int k=7;k>=0;k--){
    int ss=(k&1)?-1:+1;
    int cs=(k&1)?-1:+1;
    ps=add(mul(ps,r2),coeff_inv_fact(2*k+1,ss));
    pc=add(mul(pc,r2),coeff_inv_fact(2*k,cs));
  }
  S=mul(r,ps); C=pc;
  // Taylor remainders for |r|<=0.8: sin < 1.2e-19, cos < 2.9e-18.
  S=add(S,mk(-1e-18,1e-18)); C=add(C,mk(-1e-17,1e-17));
}
static void isincos(I x,I&S,I&C){
  const double P0=3.141592653589793115997963468544185161590576171875; // nearest binary64, below pi
  const double P1=3.141592653589793560087173318606801331043243408203125; // next binary64, above pi
  I PI=mk(P0,P1), P2=scl(0.5,PI);
  double mid=0.5*(x.lo+x.hi);
  long long n=llround(mid/(0.5*P0));
  I r=sub(x,scl((double)n,P2));
  double ma=std::max(std::fabs(r.lo),std::fabs(r.hi));
  if(ma>0.80 || r.hi-r.lo>0.01){ S=mk(-1,1); C=mk(-1,1); return; }
  I sr,cr; small_sincos(r,sr,cr);
  int q=(int)(n%4); if(q<0)q+=4;
  if(q==0){S=sr;C=cr;}
  else if(q==1){S=cr;C=scl(-1.0,sr);}
  else if(q==2){S=scl(-1.0,sr);C=scl(-1.0,cr);}
  else {S=scl(-1.0,cr);C=sr;}
  S.lo=std::max(S.lo,-1.0); S.hi=std::min(S.hi,1.0);
  C.lo=std::max(C.lo,-1.0); C.hi=std::min(C.hi,1.0);
}
static I isin(I x){I s,c;isincos(x,s,c);return s;}
static I icos(I x){I s,c;isincos(x,s,c);return c;}
static void sincD(I z,I&s0,I&s1,I&s2){
  double m=std::max(std::fabs(z.lo),std::fabs(z.hi));
  if(m<0.9){ I z2=mul(z,z);
    I a0=mk(0,0),a1=mk(0,0),a2=mk(0,0);
    // Horner on coefficients of z^{2k}
    double f[13]; f[0]=1; for(int k=1;k<13;k++){double v=1;for(int q=2;q<=2*k+1;q++)v*=q; f[k]=((k%2)?-1.0:1.0)/v;}
    a0=mk(f[12],f[12]); for(int k=11;k>=0;k--) a0=add(mul(a0,z2),mk(f[k],f[k]));
    // s1 = sum f[k]*2k*z^{2k-1} = z * sum f[k]*2k*z^{2k-2}
    double g[13]; for(int k=1;k<13;k++) g[k-1]=f[k]*(2*k);
    a1=mk(g[11],g[11]); for(int k=10;k>=0;k--) a1=add(mul(a1,z2),mk(g[k],g[k]));
    a1=mul(a1,z);
    double h2[13]; for(int k=1;k<13;k++) h2[k-1]=f[k]*(2*k)*(2*k-1);
    a2=mk(h2[11],h2[11]); for(int k=10;k>=0;k--) a2=add(mul(a2,z2),mk(h2[k],h2[k]));
    double tl=2e-15; s0=add(a0,mk(-tl,tl)); s1=add(a1,mk(-tl,tl)); s2=add(a2,mk(-tl,tl)); return;}
  I S,C; isincos(z,S,C); I z2=mul(z,z),z3=mul(z2,z);
  s0=dvi(S,z); s1=sub(dvi(C,z),dvi(S,z2));
  s2=add(sub(scl(-1.0,dvi(S,z)),scl(2.0,dvi(C,z2))),scl(2.0,dvi(S,z3)));
}
static void sincD_sc(I z,I S,I C,I&s0,I&s1,I&s2){
  double m=std::max(std::fabs(z.lo),std::fabs(z.hi));
  if(m<0.9){ sincD(z,s0,s1,s2); return; }
  I z2=mul(z,z),z3=mul(z2,z);
  s0=dvi(S,z); s1=sub(dvi(C,z),dvi(S,z2));
  s2=add(sub(scl(-1.0,dvi(S,z)),scl(2.0,dvi(C,z2))),scl(2.0,dvi(S,z3)));
}
static double K0v; static I PII,W0H,SCALEI;
static void Wall(I x,I&W,I&W1,I&W2){
  I K=mk(0,0),K1=mk(0,0),K2=mk(0,0);
  I pix=mul(PII,x), Sx,Cx; isincos(pix,Sx,Cx);
  for(int j=0;j<NC;j++){
    I a0,a1,a2,b0,b1,b2;
    if(j==0){
      I za=sub(W0H,pix), zb=add(W0H,pix);
      sincD(za,a0,a1,a2); sincD(zb,b0,b1,b2);
    } else {
      I za=mul(PII,sub(mk((double)j,(double)j),x));
      I zb=mul(PII,add(mk((double)j,(double)j),x));
      double sg=(j&1)?-1.0:1.0; // (-1)^j
      I Sa=scl(-sg,Sx), Ca=scl(sg,Cx), Sb=scl(sg,Sx), Cb=scl(sg,Cx);
      sincD_sc(za,Sa,Ca,a0,a1,a2); sincD_sc(zb,Sb,Cb,b0,b1,b2);
    }
    I ci=mk(CJ[j],CJ[j]);
    K=add(K,scl(0.5,mul(ci,add(a0,b0))));
    I cpi=scl(0.5,mul(ci,PII));
    K1=add(K1,mul(cpi,sub(b1,a1)));
    I pipi=mul(PII,PII), cp2=scl(0.5,mul(ci,pipi));
    K2=add(K2,mul(cp2,add(a2,b2)));
  }
  W=mul(SCALEI,mul(K,K)); if(W.lo<0) W.lo=0;
  W1=scl(2.0,mul(SCALEI,mul(K,K1)));
  W2=scl(2.0,mul(SCALEI,add(mul(K1,K1),mul(K,K2))));
}
static int NCELL, LOG; static double HC, XMAX;
static std::vector<float> Wmn,W1mx,W2mn;
static std::vector<double> FWm,FW1,FW2a; static std::vector<float> F1mlo,F1mhi,FCmin,FC1max,FC2min; static double HCF; static int NCF;           // per cell
static std::vector<std::vector<float>> SWmn,SW1mx,SW2mn;
static void build(){
  NCELL=(int)std::ceil(XMAX/HC);
  Wmn.resize(NCELL);W1mx.resize(NCELL);W2mn.resize(NCELL);
  {char fn[256]; snprintf(fn,256,"ct_rig14_%g_%d.bin",HC,NCELL); FILE*f=fopen(fn,"rb");
   if(f){ size_t a=fread(Wmn.data(),sizeof(float),NCELL,f), b=fread(W1mx.data(),sizeof(float),NCELL,f), c=fread(W2mn.data(),sizeof(float),NCELL,f); fclose(f);
          if(a==(size_t)NCELL&&b==(size_t)NCELL&&c==(size_t)NCELL){ printf("coarse table loaded from cache\n"); fflush(stdout); goto CTDONE; } }}
  printf("coarse table %d cells building...\n",NCELL); fflush(stdout);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<NCELL;i++){
    I W,W1,W2; Wall(mk(i*HC,(i+1)*HC),W,W1,W2);
    Wmn[i]=fdn(W.lo); W1mx[i]=fup(std::max(std::fabs(W1.lo),std::fabs(W1.hi)));
    W2mn[i]=fdn(W2.lo);
  }
  {char fn[256]; snprintf(fn,256,"ct_rig14_%g_%d.bin",HC,NCELL); FILE*f=fopen(fn,"wb"); if(f){fwrite(Wmn.data(),sizeof(float),NCELL,f);fwrite(W1mx.data(),sizeof(float),NCELL,f);fwrite(W2mn.data(),sizeof(float),NCELL,f);fclose(f);}}
  CTDONE:;
  // fine tables, point evaluation only
  HCF=HC/8.0; double FR=std::min(XMAX,56.0); NCF=(int)std::ceil(FR/HCF)+2;
  FWm.resize(NCF);FW1.resize(NCF);FW2a.resize(NCF);F1mlo.resize(NCF);F1mhi.resize(NCF);FCmin.resize(NCF);FC1max.resize(NCF);FC2min.resize(NCF);
  {char fn[256]; snprintf(fn,256,"ft_rig18_%g_%d.bin",HCF,NCF); FILE*f=fopen(fn,"rb");
   if(f){ size_t a=fread(FWm.data(),8,NCF,f),b=fread(FW1.data(),8,NCF,f),c=fread(FW2a.data(),8,NCF,f);
          size_t h=fread(F1mlo.data(),4,NCF,f),i=fread(F1mhi.data(),4,NCF,f);
          size_t d=fread(FCmin.data(),4,NCF,f),e=fread(FC1max.data(),4,NCF,f),g=fread(FC2min.data(),4,NCF,f); fclose(f);
          if(a==(size_t)NCF&&b==(size_t)NCF&&c==(size_t)NCF&&h==(size_t)NCF&&i==(size_t)NCF&&d==(size_t)NCF&&e==(size_t)NCF&&g==(size_t)NCF){
            printf("fine table loaded from cache\n"); fflush(stdout); goto FTDONE; } }}
  printf("fine table %d cells (h=%g) building...\n",NCF,HCF); fflush(stdout);
  #pragma omp parallel for schedule(static)
  for(int i=0;i<NCF;i++){
    double md=(i+0.5)*HCF; I Wm,W1m,W2m; Wall(mk(md,md),Wm,W1m,W2m);
    FWm[i]=dn(Wm.lo); FW1[i]=0.5*(W1m.lo+W1m.hi); F1mlo[i]=fdn(W1m.lo); F1mhi[i]=fup(W1m.hi);
    I Wc,W1c,W2c; Wall(mk(i*HCF,(i+1)*HCF),Wc,W1c,W2c);
    FW2a[i]=up(std::max(std::fabs(W2c.lo),std::fabs(W2c.hi)));
    FCmin[i]=fdn(Wc.lo); FC1max[i]=fup(std::max(std::fabs(W1c.lo),std::fabs(W1c.hi))); FC2min[i]=fdn(W2c.lo);}
  {char fn[256]; snprintf(fn,256,"ft_rig18_%g_%d.bin",HCF,NCF); FILE*f=fopen(fn,"wb");
   if(f){fwrite(FWm.data(),8,NCF,f);fwrite(FW1.data(),8,NCF,f);fwrite(FW2a.data(),8,NCF,f);
         fwrite(F1mlo.data(),4,NCF,f);fwrite(F1mhi.data(),4,NCF,f);
         fwrite(FCmin.data(),4,NCF,f);fwrite(FC1max.data(),4,NCF,f);fwrite(FC2min.data(),4,NCF,f);fclose(f);} }
  FTDONE:;
  LOG=1; while((1<<LOG)<=NCELL) LOG++;
  SWmn.assign(LOG,{});SW1mx.assign(LOG,{});SW2mn.assign(LOG,{});
  SWmn[0]=Wmn;SW1mx[0]=W1mx;SW2mn[0]=W2mn;
  for(int k=1;k<LOG;k++){int n=NCELL-(1<<k)+1; if(n<=0){SWmn[k].clear();continue;}
    SWmn[k].resize(n);SW1mx[k].resize(n);SW2mn[k].resize(n);
    for(int i=0;i<n;i++){int j=i+(1<<(k-1));
      SWmn[k][i]=std::min(SWmn[k-1][i],SWmn[k-1][j]);
      SW1mx[k][i]=std::max(SW1mx[k-1][i],SW1mx[k-1][j]);
      SW2mn[k][i]=std::min(SW2mn[k-1][i],SW2mn[k-1][j]);}}
}
static inline void rng(int l,int r,double&wm,double&w1,double&w2){
  if(l<0)l=0; if(r>=NCELL)r=NCELL-1; if(r<l){wm=0;w1=1e30;w2=-1e30;return;}
  int k=0; while((1<<(k+1))<=(r-l+1)) k++;
  int j=r-(1<<k)+1;
  wm=std::min(SWmn[k][l],SWmn[k][j]); w1=std::max(SW1mx[k][l],SW1mx[k][j]); w2=std::min(SW2mn[k][l],SW2mn[k][j]);
}
static inline void rngd(double lo,double hi,double&wm,double&w1,double&w2){
  if(lo<0) lo=0; if(hi<lo){wm=0;w1=1e30;w2=-1e30;return;}
  int a=(int)std::floor(lo/HCF), b=(int)std::floor(hi/HCF);
  if(a>=0 && b<NCF && b-a<=16){
    wm=1e300; w1=0; w2=1e300;
    for(int i=a;i<=b;i++){ wm=std::min(wm,(double)FCmin[i]); w1=std::max(w1,(double)FC1max[i]); w2=std::min(w2,(double)FC2min[i]); }
    return;
  }
  rng((int)(lo/HC),(int)(hi/HC),wm,w1,w2);
}
static inline I sqri(I x){
  double a=x.lo*x.lo,b=x.hi*x.hi; double hi=up(std::max(a,b));
  double lo=(x.lo<=0&&x.hi>=0)?0.0:dn(std::min(a,b)); return I{lo,hi};
}
static inline double Wpt_lo(double d){
  int i=(int)std::floor(d/HCF); if(i<0)i=0;
  if(i>=NCF){ double wm,w1,w2; rng((int)(d/HC),(int)(d/HC),wm,w1,w2); return wm; }
  double md=(i+0.5)*HCF;
  I de=sub(mk(d,d),mk(md,md)); I d2=sqri(de);
  I der=I{(double)F1mlo[i],(double)F1mhi[i]}; I lin=mul(der,de);
  double qlo;
  double l2=(double)FC2min[i];
  if(l2>=0) qlo=mlo(0.5*l2,d2.lo); else qlo=mlo(0.5*l2,d2.hi);
  double v=alo(FWm[i],lin.lo); v=alo(v,qlo); return dn(v);
}
static inline I W1ptI(double d){
  int i=(int)std::floor(d/HCF); if(i<0)i=0; if(i>=NCF)i=NCF-1;
  double md=(i+0.5)*HCF; I de=sub(mk(d,d),mk(md,md));
  double ade=up(std::max(std::fabs(de.lo),std::fabs(de.hi)));
  double er=mhi(FW2a[i],ade);
  return mk(slo((double)F1mlo[i],er),ahi((double)F1mhi[i],er));
}
static double SLOPE,EPS; static double BRL[NG],AWL[NP],QCL[NP]; static int NSH=1,SH=0;
static int NSG; static double GMX;
static std::vector<std::vector<float>> PHI;            // phi_k on global grid
static std::vector<std::vector<std::vector<float>>> SPH; // sparse range-min of PHI
static int LOGP;
static inline double phimin(int k,int i,int j){
  i--; j++; if(i<0)i=0; if(j>=NSG)j=NSG-1; if(j<i) return 1e30;
  int t=0; while((1<<(t+1))<=(j-i+1)) t++;
  return std::min(SPH[k][t][i],SPH[k][t][j-(1<<t)+1]);
}
static inline int gidx(double g){ int i=(int)(g/GMX*(NSG-1)); return i<0?0:(i>=NSG?NSG-1:i); } static long long nodes=0,cap; static bool failed=false;

struct Box{double lo[NG],hi[NG];};
static double WMIN=1e-3; /*arg7*/ static std::vector<Box> HARD; static double worstlb=1e30;
// Optional shortcut through independently certified convex well boxes.  The wells
// are NOT trusted merely because they are listed: the release ships and reruns
// strict_wells_batch_v2.cpp, which proves every enabled well with exact-rational
// configuration intervals before this optimization is accepted.
static std::vector<const WellRec*> ACTIVE_WELLS;
static std::unordered_map<long long,std::vector<const WellRec*>> WELL_BUCKETS;
static inline long long wkey(int a,int b){return ((long long)a<<32)^(unsigned int)b;}
static inline bool well_inside(const Box&B){
  if(ACTIVE_WELLS.empty()) return false;
  double mw=0; for(int k=0;k<NG;k++) mw=std::max(mw,B.hi[k]-B.lo[k]);
  if(mw>0.081) return false;
  double c0=0.5*(B.lo[0]+B.hi[0]), c1=0.5*(B.lo[1]+B.hi[1]);
  int a=(int)std::floor(c0*10.0), b=(int)std::floor(c1*10.0);
  for(int da=-1;da<=1;da++) for(int db=-1;db<=1;db++){
    auto it=WELL_BUCKETS.find(wkey(a+da,b+db)); if(it==WELL_BUCKETS.end()) continue;
    for(const WellRec*w:it->second){
      bool ok=true;
      for(int k=0;k<NG;k++){
        double wl=dn(w->c[k]-w->r), wh=up(w->c[k]+w->r);
        if(B.lo[k]<wl || B.hi[k]>wh){ok=false;break;}
      }
      if(ok) return true;
    }
  }
  return false;
}
static inline void pair_range(const Box&B,int p,double&l,double&h){
  l=0;h=0; for(int k=PI_[p];k<PJ_[p];k++){l=alo(l,B.lo[k]);h=ahi(h,B.hi[k]);}
}
static bool contract(Box&B){
  for(int it=0;it<3;it++){
    double lng=0;
    for(int p=0;p<NP;p++){ if(PJ_[p]-PI_[p]<2) continue;
      double l,h; pair_range(B,p,l,h);
      double wm,w1,w2; rngd(l,h,wm,w1,w2); lng=alo(lng,mlo(QCL[p],wm)); }
    double mk[NG],tot=lng;
    for(int k=0;k<NG;k++){ mk[k]=phimin(k,gidx(B.lo[k]),gidx(B.hi[k])); tot=alo(tot,mk[k]); }
    if(tot>=EPS) return false;
    bool ch=false;
    for(int k=0;k<NG;k++){
      double other=lng; for(int j=0;j<NG;j++) if(j!=k) other=alo(other,mk[j]); double cap=up(EPS-other);
      int i=gidx(B.lo[k]), j=gidx(B.hi[k]);
      int a=i,bb=j;
      while(a<bb && phimin(k,a,a)>cap) a++;
      while(bb>a && phimin(k,bb,bb)>cap) bb--;
      if(phimin(k,a,a)>cap) return false;
      double step=up(GMX/(NSG-1));
      double nl=dn(GMX*(double)a/(NSG-1)), nh=up(GMX*(double)bb/(NSG-1));
      nl=std::max((double)B.lo[k],slo(nl,step)); nh=std::min((double)B.hi[k],ahi(nh,step));
      if(nh<nl) return false;
      if(nl>B.lo[k]+1e-12||nh<B.hi[k]-1e-12) ch=true;
      B.lo[k]=nl; B.hi[k]=nh;
    }
    if(!ch) break;
  }
  return true;
}
static double lowerBound2(const Box&B){
  double sum=0; for(int k=0;k<NG;k++) sum=alo(sum,mlo(BRL[k],B.lo[k]));
  for(int p=0;p<NP;p++){ double l,h; pair_range(B,p,l,h);
    double wm,w1,w2; rngd(l,h,wm,w1,w2); sum=alo(sum,mlo(QCL[p],wm)); }
  return sum;
}
static inline I center_pair_interval(const double cen[NG],int p){
  I z=mk(0,0); for(int k=PI_[p];k<PJ_[p];k++) z=add(z,mk(cen[k],cen[k])); return z;
}
static inline double Wcenter_lo(I d){
  if(d.hi<0) return 0;
  double m=0.5*(d.lo+d.hi), r=up(std::max(m-d.lo,d.hi-m));
  if(m<0) m=0;
  if(m>=HCF*NCF){ double wm,w1,w2; rngd(d.lo,d.hi,wm,w1,w2); return wm; }
  double v=Wpt_lo(m), wm,w1,w2; rngd(d.lo,d.hi,wm,w1,w2);
  return slo(v,mhi(w1,r));
}
static inline I W1centerI(I d){
  double m=0.5*(d.lo+d.hi), r=up(std::max(m-d.lo,d.hi-m));
  if(m>=0 && m<HCF*NCF){
    I base=W1ptI(m); int a=(int)std::floor(std::max(0.0,d.lo)/HCF), b=(int)std::floor(std::max(0.0,d.hi)/HCF);
    if(a<0)a=0;if(b>=NCF)b=NCF-1; double m2=0;
    for(int i=a;i<=b && i<NCF;i++) m2=std::max(m2,FW2a[i]);
    double e=mhi(m2,r); return add(base,mk(-e,e));
  }
  I W,W1,W2; Wall(d,W,W1,W2); return W1;
}

// Direct interval strong-convexity certificate. This path deliberately bypasses
// all cached float tables: Wall() is evaluated on the actual box/centre intervals.
// If Hessian >= mu I on B, then
// F(c+y) >= F(c)+g(c).y + mu/2 ||y||^2.
// We minimize the RHS coordinate-wise using an enclosure of g(c).
static bool prove_strong_convex_direct(const Box &B){
  double maxw=0; for(int k=0;k<NG;k++) maxw=std::max(maxw,(double)B.hi[k]-(double)B.lo[k]);
  if(maxw>0.08) return false;
  double cen[NG], rad[NG];
  I Fc=mk(0,0), gi[NG];
  for(int k=0;k<NG;k++){
    cen[k]=0.5*((double)B.lo[k]+(double)B.hi[k]);
    rad[k]=up(std::max(cen[k]-B.lo[k],B.hi[k]-cen[k]));
    Fc=add(Fc,mul(mk(BRL[k],BRL[k]),mk(cen[k],cen[k])));
    gi[k]=mk(BRL[k],BRL[k]);
  }
  I HL[NG][NG];
  for(int i=0;i<NG;i++) for(int j=0;j<NG;j++) HL[i][j]=mk(0,0);
  for(int p=0;p<NP;p++){
    double l,h; pair_range(B,p,l,h);
    I Wr,W1r,W2r; Wall(mk(l,h),Wr,W1r,W2r);
    I coeff=scl(QCL[p],W2r);
    for(int i=PI_[p];i<PJ_[p];i++) for(int j=PI_[p];j<PJ_[p];j++) HL[i][j]=add(HL[i][j],coeff);
    I dc=center_pair_interval(cen,p), Wc,W1c,W2c; Wall(dc,Wc,W1c,W2c);
    Fc=add(Fc,scl(QCL[p],Wc));
    I gt=scl(QCL[p],W1c);
    for(int k=PI_[p];k<PJ_[p];k++) gi[k]=add(gi[k],gt);
  }
  // Gershgorin lower eigenvalue bound for a symmetric interval Hessian.
  double mu=1e300;
  for(int i=0;i<NG;i++){
    double off=0;
    for(int j=0;j<NG;j++) if(j!=i){
      double a=up(std::max(std::fabs(HL[i][j].lo),std::fabs(HL[i][j].hi)));
      off=ahi(off,a);
    }
    double row=slo(HL[i][i].lo,off);
    mu=std::min(mu,row);
  }
  if(!(mu>0) || !std::isfinite(mu)) return false;
  double lb=Fc.lo;
  for(int k=0;k<NG;k++){
    double ga=up(std::max(std::fabs(gi[k].lo),std::fabs(gi[k].hi)));
    double mur=mlo(mu,rad[k]);
    double term;
    if(ga<=mur){
      // -ga^2/(2 mu), rounded downward conservatively.
      term=dn(-up(ga*ga)/(dn(2.0*mu)));
    } else {
      term=alo(-mhi(ga,rad[k]), dn(0.5*mlo(mu,mlo(rad[k],rad[k]))));
    }
    lb=alo(lb,term);
  }
  lb=slo(lb,1e-13);
  return lb>=EPS;
}
static bool prove(Box B){
  double Plo=0; for(int k=0;k<NG;k++) Plo=alo(Plo,mlo(BRL[k],B.lo[k]));
  if(Plo>=EPS) return true;
  double dlo[NP],dhi[NP],sum=Plo;
  for(int p=0;p<NP;p++){double l,h; pair_range(B,p,l,h);
    dlo[p]=l;dhi[p]=h; double wm,w1,w2; rngd(l,h,wm,w1,w2);
    sum=alo(sum,mlo(QCL[p],wm));}
  if(sum>=EPS) return true;
  // tangent-plane bound under certified convexity
  double cen[NG],rad[NG],Fc=0,gc[NG];
  for(int k=0;k<NG;k++){cen[k]=0.5*((double)B.lo[k]+B.hi[k]);rad[k]=up(std::max(cen[k]-B.lo[k],B.hi[k]-cen[k]));
    Fc=alo(Fc,mlo(BRL[k],cen[k])); gc[k]=BRL[k];}
  double ML[NG][NG]={{0}}, MU[NG][NG]={{0}};
  for(int p=0;p<NP;p++){
    I dcI=center_pair_interval(cen,p);
    Fc=alo(Fc,mlo(QCL[p],Wcenter_lo(dcI)));
    double bm,b1,b2; rngd(dlo[p],dhi[p],bm,b1,b2);
    double s2=dn(QCL[p]*b2);
    for(int i=PI_[p];i<PJ_[p];i++) for(int j=PI_[p];j<PJ_[p];j++){ ML[i][j]=alo(ML[i][j],s2); MU[i][j]=ahi(MU[i][j],s2); }
  }
  // |grad_k| upper bound using |W'| max over the centre cell
  double gmax[NG]; for(int k=0;k<NG;k++) gmax[k]=BRL[k];
  for(int p=0;p<NP;p++){ double l,h; pair_range(B,p,l,h);
    double bm,b1,b2; rngd(l,h,bm,b1,b2);
    for(int k=PI_[p];k<PJ_[p];k++) gmax[k]=ahi(gmax[k],mhi(QCL[p],b1));}
  // mean-value bound over the whole box (valid without convexity)
  { double gb[NG]; for(int k=0;k<NG;k++) gb[k]=BRL[k];
    for(int p=0;p<NP;p++){ double bm,b1,b2; rngd(dlo[p],dhi[p],bm,b1,b2);
      for(int k=PI_[p];k<PJ_[p];k++) gb[k]=ahi(gb[k],mhi(QCL[p],b1)); }
    double mv=Fc; for(int k=0;k<NG;k++) mv=slo(mv,mhi(gb[k],rad[k])); mv=slo(mv,1e-13);
    if(mv>=EPS) return true; }
  // Certified centre-gradient intervals for the lower witness F_lo.
  I gci[NG]; for(int k=0;k<NG;k++) gci[k]=mk(BRL[k],BRL[k]);
  for(int p=0;p<NP;p++){ I dcI=center_pair_interval(cen,p);
    I di=W1centerI(dcI), ti=scl(QCL[p],di);
    for(int k=PI_[p];k<PJ_[p];k++) gci[k]=add(gci[k],ti);
  }
  // Gershgorin lower bound from interval-enclosed entries of M.
  double lmin=1e300;
  for(int i=0;i<NG;i++){ double off=0;
    for(int j=0;j<NG;j++) if(j!=i){ double a=up(std::max(std::fabs(ML[i][j]),std::fabs(MU[i][j]))); off=ahi(off,a); }
    double row=slo(ML[i][i],off); lmin=std::min(lmin,row);
  }
  double q2=0; for(int k=0;k<NG;k++) q2=ahi(q2,mhi(rad[k],rad[k]));
  double tb=Fc;
  for(int k=0;k<NG;k++){ double ga=up(std::max(std::fabs(gci[k].lo),std::fabs(gci[k].hi))); tb=slo(tb,mhi(ga,rad[k])); }
  if(lmin<0) tb=alo(tb,dn(0.5*lmin*q2));
  tb=slo(tb,1e-13);
  if(tb>=EPS) return true;
  return false;
}
struct RootResult{ long long nodes; int hard; int fail; double worst; };
static RootResult solve_root(Box root,long long rootcap){
  RootResult rr{0,0,0,1e300}; std::vector<Box> q; q.push_back(root);
  while(!q.empty()){
    if(rr.nodes>rootcap){rr.fail=1;break;}
    Box B=q.back(); q.pop_back(); rr.nodes++;
    if(well_inside(B)) continue;
    if(!contract(B)) continue;
    if(well_inside(B)) continue;
    if(prove(B)) continue;
    int d=0; double w=-1;
    for(int k=0;k<NG;k++){double ww=B.hi[k]-B.lo[k]; if(ww>w){w=ww;d=k;}}
    if(w<WMIN){ rr.hard++; double lb=lowerBound2(B); if(lb<rr.worst) rr.worst=lb; rr.fail=1; continue; }
    Box L=B,R=B; double mid=0.5*(B.lo[d]+B.hi[d]); L.hi[d]=mid;R.lo[d]=mid;
    q.push_back(L);q.push_back(R);
  }
  if(!q.empty()) rr.fail=1;
  return rr;
}
int main(int argc,char**argv){
  for(int k=0;k<NG;k++) BRL[k]=dn(BR[k]);
  for(int p=0;p<NP;p++) AWL[p]=dn(AW[p]);
  SLOPE=dn(atof(argv[1])); EPS=up(atof(argv[2])); for(int p=0;p<NP;p++) QCL[p]=dn(SLOPE*AWL[p]); cap=atoll(argv[3]);
  // Enable only the three frozen high-value certificates whose well boxes are
  // independently verified by strict_wells_batch_v2. Any other arguments get no shortcut.
  int want_sn=0,want_sd=1; double sa=atof(argv[1]),ea=atof(argv[2]);
  if(std::fabs(sa-0.5)<1e-15 && std::fabs(ea-(526.0/78125.0))<1e-15){want_sn=1;want_sd=2;}
  else if(std::fabs(sa-0.95)<1e-15 && std::fabs(ea-(9879.0/1250000.0))<1e-15){want_sn=19;want_sd=20;}
  else if(std::fabs(sa-1.0)<1e-15 && std::fabs(ea-(20033.0/2500000.0))<1e-15){want_sn=1;want_sd=1;}
  if(want_sn) for(int wi=0;wi<NWELLS;wi++) if(WELLS[wi].sn==want_sn&&WELLS[wi].sd==want_sd) ACTIVE_WELLS.push_back(&WELLS[wi]);
  for(const WellRec*w:ACTIVE_WELLS){int a=(int)std::floor(w->c[0]*10.0),b=(int)std::floor(w->c[1]*10.0);WELL_BUCKETS[wkey(a,b)].push_back(w);}
  printf("active independently-certified wells=%zu\n",ACTIVE_WELLS.size());
  HC=argc>4?atof(argv[4]):1.0/4096;
  if(argc>6){NSH=atoi(argv[5]);SH=atoi(argv[6]);}
  if(argc>7) WMIN=atof(argv[7]);
  {
   const double P0=3.141592653589793115997963468544185161590576171875;
   const double P1=3.141592653589793560087173318606801331043243408203125;
   PII=mk(P0,P1);
   double w0=1416973254337.0/1000000000000.0; W0H=mk(0.5*w0,0.5*w0);
   I k0,k01,k02; sincD(W0H,k0,k01,k02);
   SCALEI=dvi(mk(1,1),mul(k0,k0)); K0v=0.5*(k0.lo+k0.hi);
  }
  double gmaxk=0; for(int k=0;k<NG;k++) gmaxk=std::max(gmaxk,up(EPS/BRL[k]));
  XMAX=0; for(int k=0;k<NG;k++) XMAX=ahi(XMAX,up(EPS/BRL[k])); XMAX=ahi(XMAX,1.0);
  build();
  printf("table cells=%d h=%g Xmax=%g K0=%.12g\n",NCELL,HC,XMAX,K0v);
  // ---- span-1 separable domain reduction ----
  const int NS=240001; double gmaxall=0; for(int k=0;k<NG;k++) gmaxall=std::max(gmaxall,up(EPS/BRL[k]));
  std::vector<double> gs(NS), phi(NS); double mr[NG];
  double dg=up(gmaxall/(NS-1)); for(int i=0;i<NS;i++) gs[i]=gmaxall*i/(NS-1);
  std::vector<std::vector<double>> PH(NG,std::vector<double>(NS));
  for(int k=0;k<NG;k++){
    int p1=-1; for(int p=0;p<NP;p++) if(PI_[p]==k&&PJ_[p]==k+1) p1=p;
    double qc=(p1>=0)?QCL[p1]:0.0; mr[k]=1e30;
    for(int i=0;i<NS;i++){ double gl=std::max(0.0,slo(gs[i],0.5*dg)), gh=std::min(gmaxall,ahi(gs[i],0.5*dg));
      double wm,w1,w2; rngd(gl,gh,wm,w1,w2);
      PH[k][i]=alo(mlo(BRL[k],gl),mlo(qc,wm)); mr[k]=std::min(mr[k],PH[k][i]); }
  }
  NSG=NS; GMX=gmaxall; PHI.assign(NG,std::vector<float>(NS));
  for(int k=0;k<NG;k++){ mr[k]=1e300; for(int i=0;i<NS;i++){ PHI[k][i]=fdn(PH[k][i]); mr[k]=std::min(mr[k],(double)PHI[k][i]); }}
  LOGP=1; while((1<<LOGP)<=NS) LOGP++;
  SPH.assign(NG,{});
  for(int k=0;k<NG;k++){ SPH[k].assign(LOGP,{}); SPH[k][0]=PHI[k];
    for(int t=1;t<LOGP;t++){ int n2=NS-(1<<t)+1; if(n2<=0){SPH[k][t].clear();continue;}
      SPH[k][t].resize(n2);
      for(int i=0;i<n2;i++) SPH[k][t][i]=std::min(SPH[k][t-1][i],SPH[k][t-1][i+(1<<(t-1))]); } }
  double tot=0; for(int k=0;k<NG;k++) tot=alo(tot,mr[k]);
  printf("separable LB = %.9g  vs eps %.9g\n",tot,EPS);
  if(tot>=EPS){ printf("PROVED by separable bound alone\n"); return 0; }
  // admissible components per coordinate
  std::vector<std::vector<std::pair<double,double>>> comp(NG);
  for(int k=0;k<NG;k++){
    double other=0; for(int j=0;j<NG;j++) if(j!=k) other=alo(other,mr[j]);
    double cap=up(EPS-other); bool in=false; double a0=0;
    for(int i=0;i<NS;i++){
      bool ok=(double)PHI[k][i]<=cap;
      if(ok&&!in){in=true;a0=dn(gs[std::max(i-1,0)]);}
      if(!ok&&in){in=false;comp[k].push_back({a0,up(gs[std::min(i,NS-1)])});}
    }
    if(in) comp[k].push_back({a0,up(gs[NS-1])});
    printf("  g%d: %zu components, span [%.4f,%.4f]\n",k+1,comp[k].size(),comp[k][0].first,comp[k].back().second);
  }
  std::vector<Box> st;
  {
    // Exact reversal symmetry preflight. If any generated component or coefficient
    // fails bitwise symmetry, abort rather than use symmetry reduction.
    bool sym=true;
    for(int k=0;k<NG;k++){
      int rk=NG-1-k;
      if(comp[k].size()!=comp[rk].size()){sym=false;break;}
      for(size_t j=0;j<comp[k].size();j++){
        if(comp[k][j].first!=comp[rk][j].first || comp[k][j].second!=comp[rk][j].second){sym=false;break;}
      }
      if(!sym) break;
    }
    for(int k=0;k<NG;k++) if(BR[k]!=BR[NG-1-k]) sym=false;
    for(int p=0;p<NP;p++){
      bool found=false; int ri=NG-PJ_[p], rj=NG-PI_[p];
      for(int u=0;u<NP;u++) if(PI_[u]==ri && PJ_[u]==rj && AW[u]==AW[p]){found=true;break;}
      if(!found){sym=false;break;}
    }
    if(!sym){fprintf(stderr,"SYMMETRY_PREFLIGHT=FAIL\n"); return 3;}
    printf("SYMMETRY_PREFLIGHT=PASS\n");

    long long skipped=0,mirror_skipped=0; long long nroot=1; for(int k=0;k<NG;k++) nroot*=comp[k].size();
    printf("root components = %lld\n",nroot);
    for(long long t=0;t<nroot;t++){
      long long z=t; Box r; int ix[NG];
      for(int k=0;k<NG;k++){ int m2=comp[k].size(); int j=z%m2; z/=m2; ix[k]=j;
        r.lo[k]=comp[k][j].first; r.hi[k]=comp[k][j].second; }
      bool greater=false;
      for(int k=0;k<NG;k++){
        if(ix[k]<ix[NG-1-k]) break;
        if(ix[k]>ix[NG-1-k]){greater=true;break;}
      }
      if(greater){mirror_skipped++; continue;}
      double pl=0; for(int k=0;k<NG;k++) pl=alo(pl,mlo(BRL[k],r.lo[k]));
      if(pl<EPS){ if(NSH<=1 || (int)(st.size()+skipped)%NSH==SH) st.push_back(r); else skipped++; }
    }
    printf("symmetry mirror roots skipped = %lld\n",mirror_skipped);
    printf("roots this shard = %zu (shard %d/%d) tables done\n",st.size(),SH,NSH); fflush(stdout);
  }
  long long totalNodes=0; long long totalHard=0; int anyFail=0; double worst=1e300;
  #pragma omp parallel for schedule(dynamic,1) reduction(+:totalNodes,totalHard) reduction(|:anyFail)
  for(long long ri=0; ri<(long long)st.size(); ri++){
    RootResult rr=solve_root(st[(size_t)ri],cap);
    totalNodes+=rr.nodes; totalHard+=rr.hard; anyFail|=rr.fail;
    if(rr.worst<1e299){
      #pragma omp critical(worst_update)
      { if(rr.worst<worst) worst=rr.worst; }
    }
  }
  failed=(anyFail!=0)||(totalNodes>cap);
  printf("slope=%g eps=%.10g nodes=%lld stack_left=0 HARD=%lld worst_lb=%.10g (eps=%.10g gap=%.3e) result=%s\n",
         SLOPE,EPS,totalNodes,totalHard,worst,EPS,EPS-worst,failed?"INCOMPLETE":"PROVED");
  return failed?1:0;
}
