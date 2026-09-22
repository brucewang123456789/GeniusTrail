#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include "cfg8_rat.h"

namespace strict8 {
static inline double dn(double x){ return std::nextafter(x,-INFINITY); }
static inline double up(double x){ return std::nextafter(x, INFINITY); }
struct I{ double lo,hi; };
static inline I mk(double a,double b){ return I{dn(a),up(b)}; }
static inline I rat(long long n,long long d){ double x=(double)n/(double)d; return mk(x,x); }
static inline I add(I a,I b){ return mk(a.lo+b.lo,a.hi+b.hi); }
static inline I sub(I a,I b){ return mk(a.lo-b.hi,a.hi-b.lo); }
static inline I neg(I a){ return mk(-a.hi,-a.lo); }
static inline I mul(I a,I b){
  double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
  return mk(*std::min_element(p,p+4),*std::max_element(p,p+4));
}
static inline I divi(I a,I b){
  if(b.lo<=0 && b.hi>=0) return I{-INFINITY,INFINITY};
  double p[4]={a.lo/b.lo,a.lo/b.hi,a.hi/b.lo,a.hi/b.hi};
  return mk(*std::min_element(p,p+4),*std::max_element(p,p+4));
}
static inline I scl(double k,I a){ return k>=0?mk(k*a.lo,k*a.hi):mk(k*a.hi,k*a.lo); }
static inline I sqr(I a){
  if(a.lo<=0 && a.hi>=0) return mk(0,std::max(a.lo*a.lo,a.hi*a.hi));
  double p=a.lo*a.lo,q=a.hi*a.hi; return mk(std::min(p,q),std::max(p,q));
}
static inline double absup(I a){ return up(std::max(std::fabs(a.lo),std::fabs(a.hi))); }

// Fixed enclosure of pi: adjacent binary64 values bracketing the real pi.
static inline I PI(){
  const double P0=3.141592653589793115997963468544185161590576171875;
  const double P1=3.141592653589793560087173318606801331043243408203125;
  return mk(P0,P1);
}

static inline I invfact(int n,int sign){
  unsigned long long f=1; for(int k=2;k<=n;k++) f*= (unsigned long long)k;
  return divi(mk((double)sign,(double)sign),mk((double)f,(double)f));
}

// Certified sin/cos on |r| <= 0.8 by Taylor-Horner with explicit remainder cushions.
static inline void small_sincos(I r,I&S,I&C){
  I r2=mul(r,r);
  I ps=invfact(17,+1), pc=invfact(16,+1);
  for(int k=7;k>=0;k--){
    int sg=(k&1)?-1:+1;
    ps=add(mul(ps,r2),invfact(2*k+1,sg));
    pc=add(mul(pc,r2),invfact(2*k,sg));
  }
  S=mul(r,ps); C=pc;
  S=add(S,mk(-1e-18,1e-18));
  C=add(C,mk(-1e-17,1e-17));
}

// Certified sin/cos interval. No libm sin/cos enters the enclosure.
static inline void sincos_i(I x,I&S,I&C){
  I pi=PI(), p2=scl(0.5,pi);
  const double P0=3.141592653589793115997963468544185161590576171875;
  double mid=0.5*(x.lo+x.hi);
  long long n=llround(mid/(0.5*P0));
  I r=sub(x,scl((double)n,p2));
  double ma=std::max(std::fabs(r.lo),std::fabs(r.hi));
  if(ma>0.80 || r.hi-r.lo>0.01){ S=mk(-1,1); C=mk(-1,1); return; }
  I sr,cr; small_sincos(r,sr,cr);
  int q=(int)(n%4); if(q<0) q+=4;
  if(q==0){S=sr;C=cr;}
  else if(q==1){S=cr;C=neg(sr);}
  else if(q==2){S=neg(sr);C=neg(cr);}
  else {S=neg(cr);C=sr;}
  S.lo=std::max(S.lo,-1.0);S.hi=std::min(S.hi,1.0);
  C.lo=std::max(C.lo,-1.0);C.hi=std::min(C.hi,1.0);
}

// sinc and first 3 derivatives wrt z. Series near 0, quotient identities elsewhere.
static inline void sincD3(I z,I&s0,I&s1,I&s2,I&s3){
  double m=std::max(std::fabs(z.lo),std::fabs(z.hi));
  if(m<0.9){
    I z2=mul(z,z);
    // f_k=(-1)^k/(2k+1)!, k=0..12. Build recursively so no
    // machine-integer factorial overflow can enter the certificate.
    I f[13]; f[0]=mk(1,1);
    for(int k=1;k<13;k++){
      I den=mk((double)((2*k)*(2*k+1)),(double)((2*k)*(2*k+1)));
      f[k]=neg(divi(f[k-1],den));
    }
    s0=f[12]; for(int k=11;k>=0;k--) s0=add(mul(s0,z2),f[k]);
    I p1=scl(24.0,f[12]); // k=12 -> 2k
    for(int k=11;k>=1;k--) p1=add(mul(p1,z2),scl((double)(2*k),f[k]));
    s1=mul(z,p1);
    I p2=scl(24.0*23.0,f[12]);
    for(int k=11;k>=1;k--) p2=add(mul(p2,z2),scl((double)((2*k)*(2*k-1)),f[k]));
    s2=p2;
    I p3=scl(24.0*23.0*22.0,f[12]);
    for(int k=11;k>=2;k--) p3=add(mul(p3,z2),scl((double)((2*k)*(2*k-1)*(2*k-2)),f[k]));
    s3=mul(z,p3);
    // Vastly conservative for |z|<0.9; true omitted tails are << 1e-14.
    I e=mk(-1e-14,1e-14); s0=add(s0,e);s1=add(s1,e);s2=add(s2,e);s3=add(s3,e);
    return;
  }
  I S,C; sincos_i(z,S,C);
  I z2=mul(z,z), z3=mul(z2,z), z4=mul(z2,z2);
  s0=divi(S,z);
  s1=sub(divi(C,z),divi(S,z2));
  s2=add(sub(neg(divi(S,z)),scl(2.0,divi(C,z2))),scl(2.0,divi(S,z3)));
  s3=add(add(neg(divi(C,z)),scl(3.0,divi(S,z2))),
          add(scl(6.0,divi(C,z3)),scl(-6.0,divi(S,z4))));
}

struct KernelCtx{
  I pi, halfw0, c[NC], b[NG], a[NP], scale;
  KernelCtx(){
    pi=PI(); halfw0=scl(0.5,rat(W0NUM,W0DEN));
    for(int j=0;j<NC;j++) c[j]=rat(CNUM[j],CDEN[j]);
    for(int k=0;k<NG;k++) b[k]=rat(BNUM[k],BDEN[k]);
    for(int p=0;p<NP;p++) a[p]=rat(ANUM[p],ADEN[p]);
    // K(0) = sum c_j sinc(a_j), since the symmetric pair is duplicated.
    I K=mk(0,0);
    for(int j=0;j<NC;j++){
      I aj = (j==0)?halfw0:scl((double)j,pi);
      I s0,s1,s2,s3; sincD3(aj,s0,s1,s2,s3);
      K=add(K,mul(c[j],s0));
    }
    scale=divi(mk(1,1),sqr(K));
  }

  void wall3(I x,I&W,I&W1,I&W2,I&W3) const{
    I K=mk(0,0),K1=mk(0,0),K2=mk(0,0),K3=mk(0,0);
    I pix=mul(pi,x), pi2=mul(pi,pi), pi3=mul(pi2,pi);
    for(int j=0;j<NC;j++){
      I aj=(j==0)?halfw0:scl((double)j,pi);
      I za=sub(aj,pix), zb=add(aj,pix);
      I a0,a1,a2,a3,b0,b1,b2,b3;
      sincD3(za,a0,a1,a2,a3); sincD3(zb,b0,b1,b2,b3);
      I halfc=scl(0.5,c[j]);
      K=add(K,mul(halfc,add(a0,b0)));
      K1=add(K1,mul(mul(halfc,pi),sub(b1,a1)));
      K2=add(K2,mul(mul(halfc,pi2),add(a2,b2)));
      K3=add(K3,mul(mul(halfc,pi3),sub(b3,a3)));
    }
    W=mul(scale,sqr(K)); if(W.lo<0) W.lo=0;
    W1=scl(2.0,mul(scale,mul(K,K1)));
    W2=scl(2.0,mul(scale,add(sqr(K1),mul(K,K2))));
    W3=scl(2.0,mul(scale,add(scl(3.0,mul(K1,K2)),mul(K,K3))));
  }
};

static inline I qmul(I slope,I a){ return mul(slope,a); }
}
