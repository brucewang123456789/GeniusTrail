#include <cstdio>
#include <cmath>
#include <algorithm>
#include "strict_core.hpp"
using namespace strict8;

static I sinc_i(I z){I s0,s1,s2,s3;sincD3(z,s0,s1,s2,s3);return s0;}
static I sin_i(I z){I s,c;sincos_i(z,s,c);return s;}
static I cos_i(I z){I s,c;sincos_i(z,s,c);return c;}
static I Cab(I a,I b){return scl(0.5,add(sinc_i(scl(0.5,sub(a,b))),sinc_i(scl(0.5,add(a,b)))));}
static I Aab(I a,I b){
  I ah=scl(0.5,a), a2=mul(a,a);
  I term=add(divi(sin_i(ah),a), divi(scl(2.0,cos_i(ah)),a2));
  return sub(mul(term,sinc_i(scl(0.5,b))), divi(scl(2.0,Cab(a,b)),a2));
}
int main(){
  KernelCtx K;
  I OM[NC]; OM[0]=scl(2.0,K.halfw0); for(int j=1;j<NC;j++) OM[j]=scl(2.0*j,K.pi);
  I I1=mk(0,0),I2=mk(0,0),J=mk(0,0);
  for(int j=0;j<NC;j++) I1=add(I1,mul(K.c[j],sinc_i(scl(0.5,OM[j]))));
  for(int i=0;i<NC;i++)for(int j=0;j<NC;j++){
    I cc=mul(K.c[i],K.c[j]);
    I2=add(I2,mul(cc,Cab(OM[i],OM[j])));
    J=add(J,mul(cc,Aab(OM[i],OM[j])));
  }
  I H=sub(mk(2,2),divi(add(I2,J),mul(I1,I1)));
  I Hf=rat(672167187145431LL,1000000000000000LL);
  bool hpass=H.lo>Hf.hi;
  printf("I1=[%.17g, %.17g]\nI2=[%.17g, %.17g]\nJ=[%.17g, %.17g]\nH=[%.17g, %.17g]\n",I1.lo,I1.hi,I2.lo,I2.hi,J.lo,J.hi,H.lo,H.hi);
  printf("H_TARGET=[%.17g, %.17g] H_LOWER_PASS=%s\n",Hf.lo,Hf.hi,hpass?"PASS":"FAIL");

  // Rigorous positivity: cover [-1/2,1/2] by N closed cells.  Use exact dyadic endpoints.
  const int N=20000; double minlo=INFINITY; int minidx=-1;
  for(int k=0;k<N;k++){
    double lo=-0.5 + (double)k/N;
    double hi=-0.5 + (double)(k+1)/N;
    I t=mk(dn(lo),up(hi)), v=mk(0,0);
    for(int j=0;j<NC;j++) v=add(v,mul(K.c[j],cos_i(mul(OM[j],t))));
    if(v.lo<minlo){minlo=v.lo;minidx=k;}
  }
  bool ppass=minlo>0;
  printf("WINDOW_POSITIVITY_N=%d MIN_LO=%.17g CELL=%d RESULT=%s\n",N,minlo,minidx,ppass?"PASS":"FAIL");
  printf("STRICT_WINDOW_RESULT=%s\n",(hpass&&ppass)?"PASS":"FAIL");
  return (hpass&&ppass)?0:1;
}
