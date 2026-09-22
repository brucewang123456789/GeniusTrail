#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include "strict_core.hpp"
using namespace strict8;

struct W2Table{
  const KernelCtx &K; double h,xmax; int n; std::vector<double> lo;
  W2Table(const KernelCtx&kk,double X,double H):K(kk),h(H),xmax(X){
    n=(int)std::ceil(xmax/h); lo.resize(n);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++){
      double a=dn(i*h), b=up(std::min(xmax,(i+1)*h));
      double m=0.5*(a+b), r=up(std::max(m-a,b-m));
      I w,w1,w2,w3,cw,cw1,cw2,cw3;
      K.wall3(mk(m,m),w,w1,w2,w3);
      K.wall3(mk(a,b),cw,cw1,cw2,cw3);
      double e=up(absup(cw3)*r);
      lo[i]=dn(w2.lo-e);
    }
  }
  double minlo(double a,double b)const{
    a=std::max(0.0,a); b=std::min(xmax,b); if(b<a)return -INFINITY;
    int ia=std::max(0,(int)std::floor(a/h));
    int ib=std::min(n-1,(int)std::floor(b/h));
    double z=INFINITY; for(int i=ia;i<=ib;i++)z=std::min(z,lo[i]); return dn(z);
  }
};

static bool ldl_mu(I A[NG][NG],double mu){
  I L[NG][NG],D[NG];
  for(int i=0;i<NG;i++)for(int j=0;j<NG;j++)L[i][j]=mk(0,0);
  for(int i=0;i<NG;i++){
    I d=sub(A[i][i],mk(mu,mu));
    for(int k=0;k<i;k++)d=sub(d,mul(mul(L[i][k],L[i][k]),D[k]));
    if(!std::isfinite(d.lo)||d.lo<=0)return false;
    D[i]=d; L[i][i]=mk(1,1);
    for(int j=i+1;j<NG;j++){
      I num=A[j][i];
      for(int k=0;k<i;k++)num=sub(num,mul(mul(L[j][k],L[i][k]),D[k]));
      L[j][i]=divi(num,D[i]);
    }
  }
  return true;
}

#include "wells_generated.h"
static bool prove_one(const KernelCtx&K,const W2Table&tab,const WellRec&wr,long long en,long long ed,double &gap,double &mu_out){
  I slope=rat(wr.sn,wr.sd), eps=rat(en,ed), q[NP];
  for(int p=0;p<NP;p++)q[p]=mul(slope,K.a[p]);

  // Define exactly the same outward well box used by the B&B shortcut.
  I C[NG]; double blo[NG],bhi[NG],rad[NG];
  for(int k=0;k<NG;k++){
    C[k]=mk(wr.c[k],wr.c[k]);
    blo[k]=dn(wr.c[k]-wr.r); bhi[k]=up(wr.c[k]+wr.r);
    rad[k]=up(std::max(C[k].hi-blo[k],bhi[k]-C[k].lo));
  }

  I M[NG][NG]; for(int i=0;i<NG;i++)for(int j=0;j<NG;j++)M[i][j]=mk(0,0);
  I Fc=mk(0,0),grad[NG];
  for(int k=0;k<NG;k++){ Fc=add(Fc,mul(K.b[k],C[k])); grad[k]=K.b[k]; }

  for(int p=0;p<NP;p++){
    I dc=mk(0,0); double dlo=0,dhi=0;
    for(int k=PI_[p];k<PJ_[p];k++){
      dc=add(dc,C[k]); dlo=dn(dlo+blo[k]); dhi=up(dhi+bhi[k]);
    }
    double w2lo=tab.minlo(dlo,dhi);
    // q is positive; if w2lo<0 the upper endpoint of q gives the safer lower product.
    double clo = w2lo>=0 ? dn(q[p].lo*w2lo) : dn(q[p].hi*w2lo);
    I ci=mk(clo,clo);
    for(int i=PI_[p];i<PJ_[p];i++)for(int j=PI_[p];j<PJ_[p];j++)M[i][j]=add(M[i][j],ci);
    I w,w1,w2,w3; K.wall3(dc,w,w1,w2,w3);
    Fc=add(Fc,mul(q[p],w)); I gi=mul(q[p],w1);
    for(int k=PI_[p];k<PJ_[p];k++)grad[k]=add(grad[k],gi);
  }

  double mu=-1;
  for(double cand:{0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.005,0.002,0.001})
    if(ldl_mu(M,cand)){mu=cand;break;}

  double lb=Fc.lo;
  if(mu>0){
    for(int k=0;k<NG;k++){
      double ga=absup(grad[k]), r=rad[k], mur=dn(mu*r),term;
      if(ga<=mur){ double den=dn(2.0*mu); term=dn(-up(ga*ga)/den); }
      else term=dn(-up(ga*r)+dn(0.5*dn(mu*dn(r*r))));
      lb=dn(lb+term);
    }
  }else lb=-INFINITY;
  gap=dn(lb-eps.hi); mu_out=mu;
  return mu>0 && lb>=eps.hi;
}

int main(){
  KernelCtx K; W2Table tab(K,20.0,1.0/4096.0);
  int ok=0,fail=0; double mingap=INFINITY; const char*minid="";
  for(int i=0;i<NWELLS;i++){
    const WellRec&w=WELLS[i]; long long en=0,ed=1;
    if(w.sn==1&&w.sd==2){en=526;ed=78125;}
    else if(w.sn==19&&w.sd==20){en=9879;ed=1250000;}
    else if(w.sn==1&&w.sd==1){en=20033;ed=2500000;}
    else {printf("UNKNOWN %s\n",w.id);fail++;continue;}
    double gap,mu; bool pass=prove_one(K,tab,w,en,ed,gap,mu);
    if(gap<mingap){mingap=gap;minid=w.id;}
    if(pass)ok++; else {fail++;printf("FAIL %s s=%d/%d r=%.17g gap=%+.17g mu=%.6g\n",w.id,w.sn,w.sd,w.r,gap,mu);}
  }
  printf("WELLS_V2_TOTAL=%d PROVED=%d FAILED=%d MIN_GAP=%+.17g MIN_ID=%s RESULT=%s\n",NWELLS,ok,fail,mingap,minid,fail?"FAIL":"PROVED");
  return fail?1:0;
}
