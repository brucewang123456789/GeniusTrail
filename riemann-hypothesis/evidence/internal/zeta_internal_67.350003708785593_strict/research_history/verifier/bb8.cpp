#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "cfg8.h"
static inline double dn(double x){return std::nextafter(x,-INFINITY);}
static inline double up(double x){return std::nextafter(x, INFINITY);}
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
static I isin(I x){ if(x.hi-x.lo>=6.283185307179587) return I{-1,1};
  double lo=std::min(std::sin(x.lo),std::sin(x.hi)),hi=std::max(std::sin(x.lo),std::sin(x.hi));
  double t=M_PI_2+2*M_PI*(std::floor((x.lo-M_PI_2)/(2*M_PI))+1); if(t<=x.hi) hi=1;
  double u=-M_PI_2+2*M_PI*(std::floor((x.lo+M_PI_2)/(2*M_PI))+1); if(u<=x.hi) lo=-1;
  return mk(lo-1e-16,hi+1e-16);}
static I icos(I x){return isin(add(x,mk(M_PI_2,M_PI_2)));}
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
    double tl=1e-16; s0=add(a0,mk(-tl,tl)); s1=add(a1,mk(-tl,tl)); s2=add(a2,mk(-tl,tl)); return;}
  I S=isin(z),C=icos(z),z2=mul(z,z),z3=mul(z2,z);
  s0=dvi(S,z); s1=sub(dvi(C,z),dvi(S,z2));
  s2=add(sub(scl(-1.0,dvi(S,z)),scl(2.0,dvi(C,z2))),scl(2.0,dvi(S,z3)));
}
static double K0v;
static void Wall(I x,I&W,I&W1,I&W2){
  I K=mk(0,0),K1=mk(0,0),K2=mk(0,0);
  for(int j=0;j<NC;j++){
    I za=sub(mk(OM[j]/2,OM[j]/2),scl(M_PI,x)), zb=add(mk(OM[j]/2,OM[j]/2),scl(M_PI,x));
    I a0,a1,a2,b0,b1,b2; sincD(za,a0,a1,a2); sincD(zb,b0,b1,b2);
    K =add(K ,scl(0.5*CJ[j],add(a0,b0)));
    K1=add(K1,scl(0.5*CJ[j]*M_PI,sub(b1,a1)));
    K2=add(K2,scl(0.5*CJ[j]*M_PI*M_PI,add(a2,b2)));}
  double s=1.0/(K0v*K0v);
  W=scl(s,mul(K,K)); W1=scl(2*s,mul(K,K1)); W2=scl(2*s,add(mul(K1,K1),mul(K,K2)));
}
static int NCELL, LOG; static double HC, XMAX;
static std::vector<float> Wmn,W1mx,W2mn;           // per cell
static std::vector<std::vector<float>> SWmn,SW1mx,SW2mn;
static void build(){
  NCELL=(int)std::ceil(XMAX/HC);
  Wmn.resize(NCELL);W1mx.resize(NCELL);W2mn.resize(NCELL);
  for(int i=0;i<NCELL;i++){
    I W,W1,W2; Wall(mk(i*HC,(i+1)*HC),W,W1,W2);
    Wmn[i]=(float)dn(W.lo); W1mx[i]=(float)up(std::max(std::fabs(W1.lo),std::fabs(W1.hi)));
    W2mn[i]=(float)dn(W2.lo);}
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
static double SLOPE,EPS; static long long nodes=0,cap; static bool failed=false;
struct Box{float lo[NG],hi[NG];};
static bool prove(Box B){
  double Plo=0; for(int k=0;k<NG;k++) Plo+=BR[k]*B.lo[k];
  if(Plo>=EPS) return true;
  double dlo[NP],dhi[NP],sum=Plo;
  for(int p=0;p<NP;p++){double l=0,h=0;
    for(int k=PI_[p];k<PJ_[p];k++){l+=B.lo[k];h+=B.hi[k];}
    dlo[p]=l;dhi[p]=h; double wm,w1,w2; rng((int)(l/HC),(int)(h/HC),wm,w1,w2);
    sum+=SLOPE*AW[p]*wm;}
  if(sum>=EPS) return true;
  // tangent-plane bound under certified convexity
  double cen[NG],rad[NG],Fc=0,gc[NG];
  for(int k=0;k<NG;k++){cen[k]=0.5*((double)B.lo[k]+B.hi[k]);rad[k]=0.5*((double)B.hi[k]-B.lo[k]);
    Fc+=BR[k]*cen[k]; gc[k]=BR[k];}
  double M[NG][NG]={{0}};
  for(int p=0;p<NP;p++){
    double dc=0; for(int k=PI_[p];k<PJ_[p];k++) dc+=cen[k];
    double wm,w1,w2; int ci=(int)(dc/HC); rng(ci,ci,wm,w1,w2);
    Fc+=SLOPE*AW[p]*wm;
    for(int k=PI_[p];k<PJ_[p];k++) gc[k]+=SLOPE*AW[p]*w1*0;   // handled via |grad| bound below
    double bm,b1,b2; rng((int)(dlo[p]/HC),(int)(dhi[p]/HC),bm,b1,b2);
    double s2=SLOPE*AW[p]*b2;
    for(int i=PI_[p];i<PJ_[p];i++) for(int j=PI_[p];j<PJ_[p];j++) M[i][j]+=s2;
  }
  // |grad_k| upper bound using |W'| max over the centre cell
  double gmax[NG]; for(int k=0;k<NG;k++) gmax[k]=BR[k];
  for(int p=0;p<NP;p++){double dc=0; for(int k=PI_[p];k<PJ_[p];k++) dc+=cen[k];
    double wm,w1,w2; int ci=(int)(dc/HC); rng(ci,ci,wm,w1,w2);
    for(int k=PI_[p];k<PJ_[p];k++) gmax[k]+=SLOPE*AW[p]*w1;}
  // mean-value bound over the whole box (valid without convexity)
  { double gb[NG]; for(int k=0;k<NG;k++) gb[k]=BR[k];
    for(int p=0;p<NP;p++){ double bm,b1,b2; rng((int)(dlo[p]/HC),(int)(dhi[p]/HC),bm,b1,b2);
      for(int k=PI_[p];k<PJ_[p];k++) gb[k]+=SLOPE*AW[p]*b1; }
    double mv=Fc; for(int k=0;k<NG;k++) mv-=gb[k]*rad[k]; mv-=1e-13;
    if(mv>=EPS) return true; }
  double A2[NG][NG]; for(int i=0;i<NG;i++)for(int j=0;j<NG;j++)A2[i][j]=M[i][j];
  bool psd=true;
  for(int i=0;i<NG&&psd;i++){ if(A2[i][i]<=1e-13){psd=false;break;}
    for(int j=i+1;j<NG;j++){double f=A2[j][i]/A2[i][i];
      for(int k=i;k<NG;k++) A2[j][k]-=f*A2[i][k];}}
  if(psd){ double tb=Fc; for(int k=0;k<NG;k++) tb-=gmax[k]*rad[k]; tb-=1e-13;
    if(tb>=EPS) return true; }
  return false;
}
int main(int argc,char**argv){
  SLOPE=atof(argv[1]); EPS=atof(argv[2]); cap=atoll(argv[3]);
  HC=argc>4?atof(argv[4]):1.0/4096;
  {I W,W1,W2; I K=mk(0,0);
   // K0
   double acc=0; for(int j=0;j<NC;j++) acc+=CJ[j]*( (OM[j]/2==0)?1.0:std::sin(OM[j]/2)/(OM[j]/2) );
   K0v=acc;}
  double gmaxk=0; for(int k=0;k<NG;k++) gmaxk=std::max(gmaxk,EPS/BR[k]);
  XMAX=0; for(int k=0;k<NG;k++) XMAX+=EPS/BR[k]; XMAX+=1;
  build();
  printf("table cells=%d h=%g Xmax=%g K0=%.12g\n",NCELL,HC,XMAX,K0v);
  // ---- span-1 separable domain reduction ----
  const int NS=240001; double gmaxall=0; for(int k=0;k<NG;k++) gmaxall=std::max(gmaxall,EPS/BR[k]);
  std::vector<double> gs(NS), phi(NS); double mr[NG];
  for(int i=0;i<NS;i++) gs[i]=gmaxall*i/(NS-1);
  std::vector<std::vector<double>> PH(NG,std::vector<double>(NS));
  for(int k=0;k<NG;k++){
    int p1=-1; for(int p=0;p<NP;p++) if(PI_[p]==k&&PJ_[p]==k+1) p1=p;
    double aa=(p1>=0)?AW[p1]:0.0; mr[k]=1e30;
    for(int i=0;i<NS;i++){ double wm,w1,w2; int ci=(int)(gs[i]/HC); rng(ci,ci,wm,w1,w2);
      PH[k][i]=BR[k]*gs[i]+SLOPE*aa*wm; mr[k]=std::min(mr[k],PH[k][i]); }
  }
  double tot=0; for(int k=0;k<NG;k++) tot+=mr[k];
  printf("separable LB = %.9g  vs eps %.9g\n",tot,EPS);
  if(tot>=EPS){ printf("PROVED by separable bound alone\n"); return 0; }
  // admissible components per coordinate
  std::vector<std::vector<std::pair<double,double>>> comp(NG);
  for(int k=0;k<NG;k++){
    double cap=EPS-(tot-mr[k]); bool in=false; double a0=0;
    for(int i=0;i<NS;i++){
      bool ok=PH[k][i]<=cap;
      if(ok&&!in){in=true;a0=gs[std::max(i-1,0)];}
      if(!ok&&in){in=false;comp[k].push_back({a0,gs[std::min(i,NS-1)]});}
    }
    if(in) comp[k].push_back({a0,gs[NS-1]});
    printf("  g%d: %zu components, span [%.4f,%.4f]\n",k+1,comp[k].size(),comp[k][0].first,comp[k].back().second);
  }
  std::vector<Box> st;
  {
    std::vector<int> idx(NG,0); long long nroot=1; for(int k=0;k<NG;k++) nroot*=comp[k].size();
    printf("root components = %lld\n",nroot);
    for(long long t=0;t<nroot;t++){
      long long z=t; Box r;
      for(int k=0;k<NG;k++){ int m2=comp[k].size(); int j=z%m2; z/=m2;
        r.lo[k]=(float)comp[k][j].first; r.hi[k]=(float)comp[k][j].second; }
      double pl=0; for(int k=0;k<NG;k++) pl+=BR[k]*r.lo[k];
      if(pl<EPS) st.push_back(r);
    }
    printf("roots after one-body screen = %zu\n",st.size());
  }
  while(!st.empty()){
    if(nodes>cap){failed=true;break;}
    Box B=st.back(); st.pop_back(); nodes++;
    if(prove(B)) continue;
    int d=0; double w=-1;
    for(int k=0;k<NG;k++){double ww=B.hi[k]-B.lo[k]; if(ww>w){w=ww;d=k;}}
    if(w<1e-7){failed=true;printf("TERMINAL width=%g\n",w);break;}
    Box L=B,R=B; float mid=0.5f*(B.lo[d]+B.hi[d]); L.hi[d]=mid;R.lo[d]=mid;
    st.push_back(L);st.push_back(R);
  }
  printf("slope=%g eps=%.10g nodes=%lld result=%s\n",SLOPE,EPS,nodes,failed?"INCOMPLETE":"PROVED");
  return failed?1:0;
}
