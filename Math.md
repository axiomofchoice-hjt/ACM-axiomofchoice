# 数学

- [1. 数论](#1-数论)
  - [1.1. 整数环相关 and 扩展欧几里得](#11-整数环相关-and-扩展欧几里得)
  - [1.2. 防爆模乘](#12-防爆模乘)
  - [1.3. 最大公约数](#13-最大公约数)
  - [1.4. CRT + extra](#14-crt--extra)
  - [1.5. 离散对数 using BSGS + extra](#15-离散对数-using-bsgs--extra)
  - [1.6. 阶与原根](#16-阶与原根)
  - [1.7. N 次剩余](#17-n-次剩余)
  - [1.8. 数论函数](#18-数论函数)
    - [1.8.1. 单个欧拉函数](#181-单个欧拉函数)
    - [1.8.2. 离线乘法逆元](#182-离线乘法逆元)
    - [1.8.3. 线性筛](#183-线性筛)
    - [1.8.4. 杜教筛](#184-杜教筛)
    - [1.8.5. min\_25 筛](#185-min_25-筛)
  - [1.9. 素数约数相关](#19-素数约数相关)
    - [1.9.1. 唯一分解 / 质因数分解](#191-唯一分解--质因数分解)
    - [1.9.2. 素数判定 using Miller-Rabin](#192-素数判定-using-miller-rabin)
    - [1.9.3. 大数分解 using Pollard-rho](#193-大数分解-using-pollard-rho)
    - [1.9.4. 求约数 / 因数](#194-求约数--因数)
- [2. 组合数学](#2-组合数学)
  - [2.1. 组合数取模 using Lucas+extra](#21-组合数取模-using-lucasextra)
  - [2.2. 类欧几里得算法](#22-类欧几里得算法)
- [3. 博弈论](#3-博弈论)
  - [3.1. SG 定理](#31-sg-定理)
- [4. 置换群](#4-置换群)
- [5. 多项式](#5-多项式)
  - [5.1. 拉格朗日插值](#51-拉格朗日插值)
  - [5.2. 快速傅里叶变换 / FFT + 任意模数](#52-快速傅里叶变换--fft--任意模数)
  - [5.3. 多项式全家桶 数组版](#53-多项式全家桶-数组版)
  - [5.4. 快速沃尔什变换 / FWT](#54-快速沃尔什变换--fwt)
  - [5.5. 多项式复合](#55-多项式复合)
  - [5.6. 多项式多点求值](#56-多项式多点求值)
  - [5.7. 多项式快速插值](#57-多项式快速插值)
  - [5.8. k 进制异或卷积 / k 进制 FWT](#58-k-进制异或卷积--k-进制-fwt)
  - [5.9. 任意长度 FFT using Bluestein 算法](#59-任意长度-fft-using-bluestein-算法)
  - [5.10. 递推式插值 using BM 算法](#510-递推式插值-using-bm-算法)
  - [5.11. 分治 FFT](#511-分治-fft)
  - [5.12. 第二类斯特林数·行](#512-第二类斯特林数行)
- [6. 矩阵](#6-矩阵)
  - [6.1. 矩阵乘法 and 矩阵快速幂](#61-矩阵乘法-and-矩阵快速幂)
  - [6.2. 高斯消元](#62-高斯消元)
  - [6.3. 异或方程组](#63-异或方程组)
  - [6.4. 线性基](#64-线性基)
  - [6.5. 线性规划 using 单纯形法](#65-线性规划-using-单纯形法)
- [7. 数学杂项](#7-数学杂项)
  - [7.1. struct of 区间](#71-struct-of-区间)
  - [7.2. struct of 高精度](#72-struct-of-高精度)
  - [7.3. struct of 分数](#73-struct-of-分数)
  - [7.4. 表达式求值](#74-表达式求值)
  - [7.5. 数值积分 using 自适应辛普森法](#75-数值积分-using-自适应辛普森法)

## 1. 数论

### 1.1. 整数环相关 and 扩展欧几里得

```cpp
ll mul(ll a,ll b,ll m=mod){return a*b%m;} // 模乘
ll qpow(ll a,ll b,ll m=mod){ // 快速幂
    ll ans=1;
    for(;b;a=mul(a,a,m),b>>=1)
        if(b&1)ans=mul(ans,a,m);
    return ans;
}
void exgcd(ll a,ll b,ll &d,ll &x,ll &y){ // ax + by = gcd(a, b) = d
    if(!b)d=a,x=1,y=0;
    else exgcd(b,a%b,d,y,x),y-=x*(a/b);
}
ll gcdinv(ll v,ll m=mod){ // 扩欧版逆元
    ll d,x,y;
    exgcd(v,m,d,x,y);
    return (x%m+m)%m;
}
ll getinv(ll v,ll m=mod){ // 快速幂版逆元，m 必须是质数!!
    return qpow(v,m-2,m);
}
ll qpows(ll a,ll b,ll m=mod){
    if(b>=0)return qpow(a,b,m);
    else return getinv(qpow(a,-b,m),m);
}
```

### 1.2. 防爆模乘

```cpp
// int128版本
ll mul(ll a,ll b,ll m=mod){return (__int128)a*b%m;}
// long double版本（欲防爆，先自爆）
ll mul(ll a,ll b,ll m){
    ll c=a*b-(ll)((long double)a*b/m+0.5)*m;
    return c<0?c+m:c;
}
// 每位运算一次版本，注意这是真·龟速乘，O(logn)
ll mul(ll a,ll b,ll m=mod){
    ll ans=0;
    while(b){
        if(b&1)ans=(ans+a)%m;
        a=(a+a)%m;
        b>>=1;
    }
    return ans;
}
// 把 b 分成两部分版本，要保证 m 小于 1<<42（约等于 4e12），a, b < m
ll mul(ll a,ll b,ll m=mod){
    a%=m,b%=m;
    ll l=a*(b>>21)%m*(1ll<<21)%m;
    ll r=a*(b&(1ll<<21)-1)%m;
    return (l+r)%m;
}
```

### 1.3. 最大公约数

```cpp
__gcd(a, b) // 内置 gcd，推荐
ll gcd(ll a, ll b) { return b == 0 ? a : gcd(b, a % b); } // 不推荐，比内置 gcd 慢
ll gcd(ll a, ll b) { // 卡常 gcd！！
    #define tz __builtin_ctzll
    if (!a || !b) return a | b;
    int t = tz(a | b);
    a >>= tz(a);
    while (b) {
        b >>= tz(b);
        if (a > b) swap(a, b);
        b -= a;
    }
    return a << t;
    #undef tz
}
```

- 实数 gcd

```cpp
lf fgcd(lf a,lf b){return abs(b)<1e-5?a:fgcd(b,fmod(a,b));}
```

### 1.4. CRT + extra

```cpp
// CRT，m[i]两两互质
ll crt(ll a[],ll m[],int n){ // ans%m[i]==a[i]
    repeat(i,0,n)a[i]%=m[i];
    ll M=1,ans=0;
    repeat(i,0,n)
        M*=m[i];
    repeat(i,0,n){
        ll k=M/m[i],t=gcdinv(k%m[i],m[i]); // 扩欧!!
        ans=(ans+a[i]*k*t)%M; // 两个乘号可能都要mul
    }
    return (ans+M)%M;
}
// exCRT，m[i]不需要两两互质，基于扩欧exgcd和龟速乘mul
ll excrt(ll a[],ll m[],int n){ // ans%m[i]==a[i]
    repeat(i,0,n)a[i]%=m[i]; // 根据情况做适当修改
    ll M=m[0],ans=a[0],g,x,y; // M是m[0..i]的最小公倍数
    repeat(i,1,n){
        ll c=((a[i]-ans)%m[i]+m[i])%m[i];
        exgcd(M,m[i],g,x,y); // Ax=c(mod B)
        if(c%g)return -1;
        ans+=mul(x,c/g,m[i]/g)*M; // 龟速乘
        M*=m[i]/g;
        ans=(ans%M+M)%M;
    }
    return (ans+M)%M;
}
```

### 1.5. 离散对数 using BSGS + extra

- 求 $a^x \equiv b \pmod m$ ，$O(\sqrt m)$

```cpp
// BSGS，a 和 mod 互质
ll bsgs(ll a,ll b,ll mod){ // a^ans%mod==b
    a%=mod,b%=mod;
    static unordered_map<ll,ll> m; m.clear();
    ll t=(ll)sqrt(mod)+1,p=1;
    repeat(i,0,t){
        m[mul(b,p,mod)]=i; // p==a^i
        p=mul(p,a,mod);
    }
    a=p; p=1;
    repeat(i,0,t+1){
        if(m.count(p)){ // p==a^i
            ll ans=t*i-m[p];
            if(ans>0)return ans;
        }
        p=mul(p,a,mod);
    }
    return -1;
}
// exBSGS，a 和 mod不需要互质，基于 BSGS
ll exbsgs(ll a,ll b,ll mod){ // a^ans%mod==b
    a%=mod,b%=mod;
    if(b==1)return 0;
    ll ans=0,c=1,g;
    while((g=__gcd(a,mod))!=1){
        if(b%g!=0)return -1;
        b/=g,mod/=g;
        c=mul(c,a/g,mod);
        ans++;
        if(b==c)return ans;
    }
    ll t=bsgs(a,mul(b,getinv(c,mod),mod),mod); // 必须扩欧逆元!!
    if(t==-1)return -1;
    return t+ans;
}
```

### 1.6. 阶与原根

- 一些原根

```cpp
if (m == 167772161) return 3;
if (m == 469762049) return 3;
if (m == 754974721) return 11;
if (m == 998244353) return 3;
```

- 判断是否有原根：若 m 有原根，则 m 一定是下列形式：$2,4,p^a,2p^a$（p 是奇素数，a 是正整数）。
- 求所有原根：若 g 为 m 的一个原根，则 $g^s\space(1\le s\le\varphi(m),\gcd(s,\varphi(m))=1)$ 给出了 m 的所有原根。
  - 因此若 m 有原根，则 m 有 $\varphi(\varphi(m))$ 个原根。
- 求一个原根，$O(n\log\log n)$（实际上大概率达不到）。

```cpp
ll getG(ll n){ // 求 n 最小的原根
    static vector<ll> a; a.clear();
    ll k=n-1;
    repeat(i,2,sqrt(k+1)+1)
    if(k%i==0){
        a.push_back(i); // a 存放 (n-1) 的质因数
        while(k%i==0)k/=i;
    }
    if(k!=1)a.push_back(k);
    repeat(i,2,n){ // 枚举答案
        bool f=1;
        for(auto j:a)
        if(qpow(i,(n-1)/j,n)==1){
            f=0;
            break;
        }
        if(f)return i;
    }
    return -1;
}
```

### 1.7. N 次剩余

- 求 $x^a \equiv b \pmod m$ ，基于 BSGS、原根

```cpp
// 只求一个
ll residue(ll a,ll b,ll mod){ // ans^a%mod==b
    ll g=getG(mod),c=bsgs(qpow(g,a,mod),b,mod);
    if(c==-1)return -1;
    return qpow(g,c,mod);
}
// 求所有N次剩余
vector<ll> ans;
void allresidue(ll a,ll b,ll mod){ // ans^a%mod==b
    ll g=getG(mod),c=bsgs(qpow(g,a,mod),b,mod);
    ans.clear();
    if(b==0){ans.push_back(0);return;}
    if(c==-1)return;
    ll now=qpow(g,c,mod);
    ll step=(mod-1)/__gcd(a,mod-1);
    ll ps=qpow(g,step,mod);
    for(ll i=c%step;i<mod-1;i+=step,now=mul(now,ps,mod))
        ans.push_back(now);
    sort(ans.begin(),ans.end());
}
```

### 1.8. 数论函数

#### 1.8.1. 单个欧拉函数

- $\varphi(n)=$ 小于 `n` 且与 `n` 互质的正整数个数
- 令 `n` 的唯一分解式 $n=\prod({p_k}^{a_k})$，则有

$$\varphi(n)=n\cdot \prod(1-\dfrac 1 {p_k})$$

- $O(\sqrt n)$

```cpp
int getphi(int n){
    int ans=n;
    repeat(i,2,sqrt(n)+2)
    if(n%i==0){
        while(n%i==0)n/=i;
        ans=ans/i*(i-1);
    }
    if(n>1)ans=ans/n*(n-1);
    return ans;
}
```

#### 1.8.2. 离线乘法逆元

- 求 $1..(n-1)$ 的逆元，$O(n)$。

```cpp
void get_inv(int n,int m=mod){
    inv[1]=1;
    repeat(i,2,n)inv[i]=m-m/i*inv[m%i]%m;
}
```

- 求 $a_{1..n}$ 的逆元，离线，$O(n)$。

```cpp
void get_inv(int a[],int n){ // 求a[1..n]的逆元，存在inv[1..n]中
    static int pre[N];
    pre[0]=1;
    repeat(i,1,n+1)
        pre[i]=(ll)pre[i-1]*a[i]%mod;
    int inv_pre=qpow(pre[n],mod-2,mod);
    repeat_back(i,1,n+1){
        inv[i]=(ll)pre[i-1]*inv_pre%mod;
        inv_pre=(ll)inv_pre*a[i]%mod;
    }
}
```

#### 1.8.3. 线性筛

- 定理：求出 $f(p)$（p 为质数）的复杂度不超过 $O(\log p)$ 的积性函数可以被线性筛。

筛素数和最小质因数

- `prime[i]` 表示第 $i+1$ 个质数，`vis[i] == 0` 表示 i 是素数，`lpf[i]` 为 i 的最小质因数。
- $O(n)$。

```cpp
struct Sieve {
    static const int N = 1000010;
    bool vis[N]; int lpf[N]; vector<int> prime;
    Sieve() {
        vis[1] = 1;
        repeat (i, 2, N) {
            if (!vis[i]) prime.push_back(i), lpf[i] = i;
            for (auto j : prime) {
                if (i * j >= N) break;
                vis[i * j] = 1; lpf[i * j] = j;
                if (i % j == 0) break;
            }
        }
    }
} sieve;
```

筛欧拉函数

- 线性版，$O(n)$

```cpp
bool vis[N]; int phi[N]; vector<int> a;
void get_phi(){
    vis[1]=1; phi[1]=1;
    repeat(i,2,N){
        if(!vis[i])a.push_back(i),phi[i]=i-1;
        for(auto j:a){
            if(i*j>=N)break;
            vis[i*j]=1;
            if(i%j==0){phi[i*j]=phi[i]*j; break;}
            phi[i*j]=phi[i]*(j-1);
        }
    }
}
```

- 不是线性但节省力气和空间版，$O(n\log\log n)$

```cpp
void get_phi(){
    phi[1]=1; // 其他的值初始化为0
    repeat(i,2,N)if(!phi[i])
    for(int j=i;j<N;j+=i){
        if(!phi[j])phi[j]=j;
        phi[j]=phi[j]/i*(i-1);
    }
}
```

筛莫比乌斯函数

- $O(n)$

```cpp
bool vis[N]; int mu[N]; vector<int> a;
void get_mu(){
    vis[1]=1; mu[1]=1;
    repeat(i,2,N){
        if(!vis[i])a.push_back(i),mu[i]=-1;
        for(auto j:a){
            if(i*j>=N)break;
            vis[i*j]=1;
            if(i%j==0){mu[i*j]=0; break;}
            mu[i*j]=-mu[i];
        }
    }
}
```

筛约数个数

```cpp
bool vis[N]; int d[N]; vector<int> a;
void get_d(){
    vector<int> c(N); vis[1]=1; d[1]=1,c[1]=0;
    repeat(i,2,N){
        if(!vis[i])a.push_back(i),d[i]=2,c[i]=1;
        for(auto j:a){
            if(i*j>=N)break;
            vis[i*j]=1;
            if(i%j==0){
                d[i*j]=d[i]/(c[i]+1)*(c[i]+2);
                c[i*j]=c[i]+1;
                break;
            }
            d[i*j]=d[i]*2,c[i*j]=1;
        }
    }
}
```

筛约数之和

```cpp
bool vis[N]; int d[N]; vector<int> a;
void get_d(){
    vector<int> c(N),sum(N);
    vis[1]=1; d[1]=1,c[1]=0,sum[1]=0;
    repeat(i,2,N){
        if(!vis[i])a.push_back(i),d[i]=i+1,c[i]=i,sum[i]=1+i;
        for(auto j:a){
            if(i*j>=N)break; vis[i*j]=1;
            if(i%j==0){
                c[i*j]=c[i]*j;
                sum[i*j]=sum[i]+c[i*j];
                d[i*j]=d[i]/sum[i]*sum[i*j];
                break;
            }
            d[i*j]=d[i]*d[j],c[i*j]=j,sum[i*j]=1+j;
        }
    }
}
```

筛 gcd

```cpp
int gcd[N][N];
void get_gcd(int n,int m){
    repeat(i,1,n+1)
    repeat(j,1,m+1)
    if(!gcd[i][j])
    repeat(k,1,min(n/i,m/j)+1)
        gcd[k*i][k*j]=k;
}
```

#### 1.8.4. 杜教筛

- 杜教筛只能筛部分积性函数。

$$g(1)S(n)=\sum_{i=1}^n(f*g)(i)-\sum_{i=2}^n g(i)S(\lfloor\dfrac n i \rfloor),S(n)=\sum_{i=1}^nf(i)$$

如果能找到合适的 $g(n)$，能快速计算

$$\displaystyle\sum_{i=1}^n(f\ast g)(i)$$

就能快速计算 $S(n)$。

例如

$$\begin{array}{lll}f(n)=\mu(n)&g(n)=1&(f\ast g)(n)=[n=1]\newline f(n)=\varphi(n)&g(n)=1&(f\ast g)(n)=n\newline f(n)=n\cdot\varphi(n)&g(n)=n&(f\ast g)(n)=n^2\newline f(n)=d(n)&g(n)=\mu(n)&(f\ast g)(n)=1\newline f(n)=\sigma(n)&g(n)=\mu(n)&(f\ast g)(n)=n\end{array}$$

- （有些不用杜教筛可以更快，比如 $\displaystyle\sum_{i=1}^{n}\sigma(n)=\sum_{i=1}^{n}i\cdot\lfloor\dfrac n i\rfloor$）
- $O(n^{\tfrac 2 3})$，注意有递归的操作就要记忆化

```cpp
struct DU{
    static const int N=2000010;
    int sum[N];
    DU(){
        vector<int> a,mu(N,1),vis(N,0);
        repeat(i,2,N){
            if(!vis[i])a.push_back(i),mu[i]=-1;
            for(auto j:a){
                if(i*j>=N)break;
                vis[i*j]=1;
                if(i%j==0){mu[i*j]=0; break;}
                mu[i*j]=-mu[i];
            }
        }
        repeat(i,1,N)sum[i]=sum[i-1]+mu[i];
    }
    ll sum_mu(ll n){
        if(n<N)return sum[n];
        static map<ll,ll> rec; if(rec.count(n))return rec[n];
        ll ans=1;
        for(ll l=2,r;l<=n;l=r+1){
            r=n/(n/l);
            ans-=sum_mu(n/l)*(r-l+1);
        }
        return rec[n]=ans;
    }
    ll sum_phi(ll n){
        ll ans=0;
        for(ll l=1,r;l<=n;l=r+1){
            r=n/(n/l);
            ans+=(sum_mu(r)-sum_mu(l-1))*(n/l)*(n/l);
        }
        return ((ans-1)>>1)+1;
    }
    ll sum_d(ll n){
        static map<ll,ll> rec; if(rec.count(n))return rec[n];
        ll ans=n;
        for(ll l=2,r;l<=n;l=r+1){
            r=n/(n/l);
            ans-=(sum_mu(r)-sum_mu(l-1))*sum_d(n/l);
        }
        return rec[n]=ans;
    }
    ll sum_sigma(ll n){
        ll ans=0;
        for(ll l=1,r;l<=n;l=r+1){
            r=n/(n/l);
            ans+=(l+r)*(r-l+1)/2*(n/l);
        }
        return ans;
    }
}du;
```

#### 1.8.5. min_25 筛

- 学不会。
- 求 $[1,n]$ 内的素数个数。

```cpp
#include<cstdio>
#include<math.h>
#define ll long long
const int N = 316300;
ll n, g[N<<1], a[N<<1];
int id, cnt, sn, prime[N];
inline int Id(ll x){return x<=sn?x:id-n/x+1;}
int main() {
    scanf("%lld", &n), sn=sqrt(n);
    for(ll i=1; i<=n; i=a[id]+1) a[++id]=n/(n/i), g[id]=a[id]-1;
    for(int i=2; i<=sn; ++i) if(g[i]!=g[i-1]){
        // 这里 i 必然是质数，因为 g[] 是前缀质数个数
        // 当 <i 的质数的倍数都被筛去，让 g[] 发生改变的位置只能是下一个质数
        // 别忘了 i<=sn 时，ID(i) 就是 i。
        prime[++cnt]=i;
        ll sq=(ll)i*i;
        for(int j=id; a[j]>=sq; --j) g[j]-=g[Id(a[j]/i)]-(cnt-1);
    }
    return printf("%lld\n", g[id]), 0;
}
```

- 求 $[1,n]$ 内的素数之和。

```cpp
namespace Min25 {
    int prime[N],id1[N],id2[N],flag[N],cnt,m;
    ll g[N],sum[N],a[N],T,n;
    int ID(ll x){return x<=T?id1[x]:id2[n/x];}
    ll getsum(ll x){return x*(x+1)/2-1;}
    ll f(ll x){return x;}
    void work(){
        T=sqrt(n+0.5);cnt=0;fill(flag,flag+T+1,0);m=0;
        for(int i=2;i<=T;i++){
            if(!flag[i]) prime[++cnt]=i,sum[cnt]=sum[cnt-1]+i;
            for(int j=1;j<=cnt && i*prime[j]<=T;j++){
                flag[i*prime[j]]=1;
                if(i%prime[j]==0) break;
            }
        }
        for(ll l=1;l<=n;l=n/(n/l)+1){
            a[++m]=n/l;
            if(a[m]<=T)id1[a[m]]=m;else id2[n/a[m]]=m;
            g[m]=getsum(a[m]);
        }
        for(int i=1;i<=cnt;i++)
            for(int j=1;j <= m && 1ll*prime[i]*prime[i]<=a[j];j++)
                g[j]=g[j]-1ll*prime[i]*(g[ID(a[j]/prime[i])]-sum[i-1]);
    }
    ll solve(ll x){
        if(x<=1) return x;
        return n=x,work(),g[ID(n)];
    }
}
```

### 1.9. 素数约数相关

#### 1.9.1. 唯一分解 / 质因数分解

- 用数组表示数字唯一分解式的素数的指数，如 $50=[0,0,1,0,0,2,0,\ldots]$。
- 可以用来计算阶乘和乘除操作。

```cpp
void fac(int a[],ll n){
    repeat(i,2,(int)sqrt(n)+2)
    while(n%i==0)a[i]++,n/=i;
    if(n>1)a[n]++;
}
```

- set 维护版

```cpp
struct fac{
    #define facN 1010
    ll a[facN]; set<ll> s; // 乘法就是multiset
    fac(){mst(a,0); s.clear();}
    void lcm(ll n){ // self=lcm(self,n)
        repeat(i,2,facN)
        if(n%i==0){
            ll cnt=0;
            while(n%i==0)cnt++,n/=i;
            a[i]=max(a[i],cnt); // 改成a[i]+=cnt就变成了乘法
        }
        if(n>1)s.insert(n);
    }
    ll value(){ // return self%mod
        ll ans=1;
        repeat(i,2,facN)
            if(a[i])ans=ans*qpow(i,a[i],mod)%mod;
        for(auto i:s)ans=ans*i%mod;
        return ans;
    }
}f;
```

#### 1.9.2. 素数判定 using Miller-Rabin

- $O(\log^3 n)$

```cpp
bool MR(ll x, ll b) {
    ll k = x - 1;
    while (k) {
        ll cur = qpow(b, k, x);
        if (cur != 1 && cur != x - 1) return 0;
        if (k % 2 == 1 || cur == x - 1) return 1;
        k >>= 1;
    }
    return 1;
}
bool isprime(ll x) {
    if (x < 2) return false;
    for (int i : {2, 3, 5, 7, 13, 29, 37, 89}) {
        if (x == i) return true;
        if (!MR(x, i)) return false;
    }
    return true;
}
```

#### 1.9.3. 大数分解 using Pollard-rho

- $O(n^{\tfrac 1 4})$，基于 Miller-Rabin 素性测试

```cpp
ll PR(ll x) {
    ll s = 0, t = 0, c = rnd() % (x - 1) + 1;
    int stp = 0, goal = 1; ll val = 1;
    for (goal = 1;; goal <<= 1, s = t, val = 1) {
        for (stp = 1; stp <= goal; ++stp) {
            t = ((__int128)t * t + c) % x;
            val = (__int128)val * abs(t - s) % x;
            if (stp % 127 == 0) {
                ll d = __gcd(val, x);
                if (d > 1) return d;
            }
        }
        ll d = __gcd(val, x);
        if (d > 1) return d;
    }
}
vector<ll> ans; // result
void rho(ll n) {
    if (n == 1) return;
    if (isprime(n)) {
        ans.push_back(n);
        return;
    }
    ll t;
    do { t = PR(n); } while (t >= n);
    rho(t);
    rho(n / t);
}
```

#### 1.9.4. 求约数 / 因数

- $O(\sqrt n)$

```cpp
void get_divisor(int n) {
    d.clear();
    for (int i = 1; i < n; i = n / (n / (i + 1)))
        if (n % i == 0) d.push_back(i);
    d.push_back(n);
}
```

- 小常数版（要求 $n\le 10^7$），基于线性筛

```cpp
vector<pii> pd; vector<ll> v; // pd: <k, p>; v: divisors
void dfs(int x,int y,ll s){
    if(x==(int)pd.size()){v.push_back(s); return;}
    dfs(x+1,0,s);
    if(y<pd[x].se)dfs(x,y+1,s*pd[x].fi);
}
void get_divisor(ll n){
    pd.clear(); v.clear();
    while(n!=1){
        if(!pd.empty() && pd.back().fi==lpf[n])pd.back().se++;
        else pd.push_back({lpf[n],1});
        n/=lpf[n]; // needs initialized
    }
    dfs(0,0,1);
}
```

## 2. 组合数学

### 2.1. 组合数取模 using Lucas+extra

- Lucas定理用来求模意义下的组合数。
- 真·Lucas，p 是质数。

```cpp
ll lucas(ll a,ll b,ll p){ // a>=b
    if(b==0)return 1;
    return mul(C(a%p,b%p,p),lucas(a/p,b/p,p),p);
}
```

- 特例：如果p=2，可能lucas失效。（？）

```cpp
ll C(ll a,ll b){ // a>=b，p=2的情况
    return (a&b)==b;
}
```

- 快速阶乘和 exLucas
- `qfac.A(x),qfac.B(x)` 满足 $A\equiv \dfrac{x!}{p^B}\pmod {p^k}$。
- `qfac.C(a,b)` $\equiv C_a^b \pmod {p^k}$。
- $\text{exlucas}(a,b,m)\equiv C_a^b \pmod m$，函数内嵌中国剩余定理。

```cpp
struct Qfac{
    ll s[2000010];
    ll p,m;
    ll A(ll x){ // 快速阶乘的A值
        if(x==0)return 1;
        ll c=A(x/p);
        return s[x%m]*qpow(s[m],x/m,m)%m*c%m;
    }
    ll B(ll x){ // 快速阶乘的B值
        int ans=0;
        for(ll i=x;i;i/=p)ans+=i/p;
        return ans;
    }
    ll C(ll a,ll b){ // 组合数，a>=b
        ll k=B(a)-B(b)-B(a-b);
        return A(a)*gcdinv(A(b),m)%m
            *gcdinv(A(a-b),m)%m
            *qpow(p,k,m)%m;
    }
    void init(ll _p,ll _m){ // 一定要满足m=p^k
        p=_p,m=_m;
        s[0]=1;
        repeat(i,1,m+1)
            if(i%p)s[i]=s[i-1]*i%m;
            else s[i]=s[i-1];
    }
}qfac;
ll exlucas(ll a,ll b,ll mod){
    ll ans=0,m=mod;
    for(ll i=2;i<=m;i++) // 不能repeat
    if(m%i==0){
        ll p=i,k=1;
        while(m%i==0)m/=i,k*=i;
        qfac.init(p,k);
        ans=(ans+qfac.C(a,b)*(mod/k)%mod*gcdinv(mod/k,k)%mod)%mod;
    }
    return (ans+mod)%mod;
}
```

### 2.2. 类欧几里得算法

- 求 $\displaystyle \sum _{x = 0}^n x^{k_1}\lfloor \dfrac {ax+b} {c}\rfloor^{k_2}$

```cpp
namespace eu {
#define int ll
const int N = 12, M = 100; // N = k + 2
int f[M][N][N], C[N][N], A[N][N];
int work(int d, int k1, int k2, int a, int b, int c, ll n) {
    if (f[d][k1][k2] != -1)
        return f[d][k1][k2];
    int &res = f[d][k1][k2];
    res = 0;
    if (!a) {
        int x = 1;
        repeat (k, 0, k1 + 2) {
            (res += A[k1][k] * x) %= mod;
            (x *= n % mod) %= mod;
        }
        return (res *= qpow(b / c, k2)) %= mod;
    }
    if (a >= c || b >= c) {
        int e = a % c, f = b % c, ap[N], bp[N];
        repeat (i, 0, k2 + 1) {
            ap[i] = qpow(a / c, i);
            bp[i] = qpow(b / c, i);
        }
        repeat (i, 0, k2 + 1)
        repeat (j, 0, k2 - i + 1) {
            int k = k2 - i - j;
            int t = ap[i] * bp[j] % mod * work(d + 1, k1 + i, k, e, f, c, n) % mod;
            (res += C[k2][i] * C[k2 - i][j] % mod * t) %= mod;
        }
        return res;
    }
    int x = 1; ll v = (a * n + b) / c;
    repeat (k, 0, k1 + 2) {
        (res += A[k1][k] * x) %= mod;
        (x *= n % mod) %= mod;
    }
    (res *= qpow(v % mod, k2)) %= mod;
    repeat (i, 0, k2) {
        int sum = 0;
        repeat (j, 0, k1 + 2)
            (sum += A[k1][j] * work(d + 1, i, j, c, c - b - 1, a, v - 1)) %= mod;
        (res -= sum * C[k2][i]) %= mod;
    }
    return res;
}
void init() {
    repeat (i, 0, N) {
        C[i][0] = 1;
        repeat (j, 1, i + 1) C[i][j] = D(C[i - 1][j - 1] + C[i - 1][j]);
    }
    A[0][1] = A[0][0] = 1;
    repeat (i, 1, N - 1) {
        repeat (j, 0, i + 2) A[i][j] = C[i + 1][j];
        repeat (j, 0, i)
        repeat (k, 0, j + 2)
            (A[i][k] -= C[i + 1][j] * A[j][k]) %= mod;
        int inv = qpow(i + 1, mod - 2);
        repeat (j, 0, i + 2) (A[i][j] *= inv) %= mod;
    }
}
int solve(int k1, int k2, int a, int b, int c, int n) {
    memset(f, -1, sizeof f);
    return (work(0, k1, k2, a, b, c, n) + mod) % mod; 
}
}
```

## 3. 博弈论

### 3.1. SG 定理

- 有向无环图中，两个玩家轮流推多颗棋子，不能走的判负。
- 假设 x 的后继状态为 $y_1,y_2,...,y_k$。
- 则 $SG[x]=\text{mex}\{SG[y_i]\}$，$\text{mex} S$ 表示不属于集合 S 的最小自然数。
- 当且仅当所有起点SG值的异或和为 0 时先手必败。
- （如果只有一个起点，SG的值可以只考虑01）。
- 例题：拿 n 堆石子，每次只能拿一堆中的斐波那契数颗石子。

```cpp
void getSG(int n){
    mst(SG,0);
    repeat(i,1,n+1){
        mst(S,0);
        for(int j=0;f[j]<=i && j<=N;j++)
            S[SG[i-f[j]]]=1;
        for(int j=0;;j++)
        if(!S[j]){
            SG[i]=j;
            break;
        }
    }
}
```

## 4. 置换群

- 定理：排列逆序对数奇偶性和 $n-$ 环数奇偶性相同。
- 求 $A^x$，编号从 0 开始，$O(n)$。

```cpp
void qpow(int a[],int n,int x){
    static int rec[N],c[N];
    static bool vis[N];
    fill(vis,vis+n,0);
    repeat(i,0,n)if(!vis[i]){
        int cnt=0; rec[cnt++]=i;
        for(int p=a[i];p!=i;p=a[p])
            rec[cnt++]=p,vis[p]=1;
        repeat(J,0,cnt)
            c[rec[J]]=a[rec[(J+x-1)%cnt]];
        repeat(J,0,cnt)
            a[rec[J]]=c[rec[J]];
    }
}
```

- $A^k=B$ 求任一 A，编号从 0 开始，$O(n)$。（暂无判断有解操作）

```cpp
repeat(i,0,n){
    a[read()-1]=i;
    vis[i]=0;
}
repeat(i,0,n)if(!vis[i]){
    int cnt=0; rec[cnt++]=i;
    for(int p=a[i];p!=i;p=a[p])
        rec[cnt++]=p,vis[p]=1;
    repeat(J,0,cnt)
        c[1ll*J*k%cnt]=rec[J];
    repeat(J,0,cnt)
        ans[c[(J+1)%cnt]]=c[J];
}
```

## 5. 多项式

技能树：拉格朗日反演，快速插值和多点求值。

### 5.1. 拉格朗日插值

- 函数曲线通过n个点 $(x_i,y_i)$，求 $f(k)$。
- 拉格朗日插值：$\displaystyle f(x)=\sum_{i=1}^n\left[y_i\prod_{j!=i}\dfrac{x-x_j}{x_i-x_j}\right]$。
- $O(n^2)$。

```cpp
ll solve(int n,int x0){
    ll ans=0; x0%=mod;
    repeat(i,0,n)x[i]%=mod,y[i]%=mod;
    repeat(i,0,n){
        int s1=y[i],s2=1;
        repeat(j,0,n)
        if(i!=j){
            s1=s1*(x0-x[j])%mod;
            s2=s2*(x[i]-x[j])%mod;
        }
        ans=(ans+s1*qpow(s2,mod-2)%mod+mod)%mod;
    }
    return ans;
}
```

```cpp
ll solve(int n,int x0){ // (i,y[i]),i=1..n的优化
    ll ans=0,up=1; x0%=mod;
    if(x0>=1 && x0<=n)return y[x0];
    repeat(i,1,n+1)
        up=up*(x0-i)%mod;
    repeat(i,1,n+1){
        ans+=y[i]*up%mod*qpow((x0-i)*((n+i)%2?-1:1)*C.fac[i-1]%mod*C.fac[n-i]%mod,mod-2)%mod;
    }
    return ans%mod;
}
```

### 5.2. 快速傅里叶变换 / FFT + 任意模数

- 离散傅里叶变换(DFT)即求 $(\omega_n^k,f(\omega_n^k))$
- 离散傅里叶反变换(IDFT)即求 $(\omega_n^{-k},g(\omega_n^{-k}))$
- 求两个多项式的卷积，$O(n\log n)$

```cpp
struct FFT{
    static const int N=1<<20;
    struct cp{
        long double a,b;
        cp(){}
        cp(const long double &a,const long double &b):a(a),b(b){}
        cp operator+(const cp &t)const{return cp(a+t.a,b+t.b);}
        cp operator-(const cp &t)const{return cp(a-t.a,b-t.b);}
        cp operator*(const cp &t)const{return cp(a*t.a-b*t.b,a*t.b+b*t.a);}
        cp conj()const{return cp(a,-b);}
    };
    cp wn(int n,int f){
        static const long double pi=acos(-1.0);
        return cp(cos(pi/n),f*sin(pi/n));
    }
    int g[N];
    void dft(cp a[],int n,int f){
        repeat(i,0,n)if(i>g[i])swap(a[i],a[g[i]]);
        for(int i=1;i<n;i<<=1){
            cp w=wn(i,f);
            for(int j=0;j<n;j+=i<<1){
                cp e(1,0);
                for(int k=0;k<i;e=e*w,k++){
                    cp x=a[j+k],y=a[j+k+i]*e;
                    a[j+k]=x+y,a[j+k+i]=x-y;
                }
            }
        }
        if(f==-1){
            cp Inv(1.0/n,0);
            repeat(i,0,n)a[i]=a[i]*Inv;
        }
    }
    #ifdef CONV
    cp a[N],b[N];
    vector<ll> conv(const vector<ll> &u,const vector<ll> &v){ // 一般fft
        const int n=(int)u.size()-1,m=(int)v.size()-1;
        const int k=__lg(n+m+1)+1,s=1<<k;
        g[0]=0; repeat(i,1,s)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
        repeat(i,0,s){
            a[i]=cp(i<=n?u[i]:0,0);
            b[i]=cp(i<=m?v[i]:0,0);
        }
        dft(a,s,1); dft(b,s,1);
        repeat(i,0,s)a[i]=a[i]*b[i];
        dft(a,s,-1);
        vector<ll> ans;
        repeat(i,0,n+m+1)ans<<llround(a[i].a);
        return ans;
    }
    #endif
    #ifdef CONV_MOD
    cp a[N],b[N],Aa[N],Ab[N],Ba[N],Bb[N];
    vector<ll> conv_mod(const vector<ll> &u,const vector<ll> &v,ll mod){ // 任意模数fft
        const int n=(int)u.size()-1,m=(int)v.size()-1,M=sqrt(mod)+1;
        const int k=__lg(n+m+1)+1,s=1<<k;
        g[0]=0; repeat(i,1,s)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
        repeat(i,0,s){
            a[i]=i<=n?cp(u[i]%mod%M,u[i]%mod/M):cp();
            b[i]=i<=m?cp(v[i]%mod%M,v[i]%mod/M):cp();
        }
        dft(a,s,1); dft(b,s,1);
        repeat(i,0,s){
            int j=(s-i)%s;
            cp t1=(a[i]+a[j].conj())*cp(0.5,0);
            cp t2=(a[i]-a[j].conj())*cp(0,-0.5);
            cp t3=(b[i]+b[j].conj())*cp(0.5,0);
            cp t4=(b[i]-b[j].conj())*cp(0,-0.5);
            Aa[i]=t1*t3,Ab[i]=t1*t4,Ba[i]=t2*t3,Bb[i]=t2*t4;
        }
        repeat(i,0,s){
            a[i]=Aa[i]+Ab[i]*cp(0,1);
            b[i]=Ba[i]+Bb[i]*cp(0,1);
        }
        dft(a,s,-1); dft(b,s,-1);
        vector<ll> ans;
        repeat(i,0,n+m+1){
            ll t1=llround(a[i].a)%mod;
            ll t2=llround(a[i].b)%mod;
            ll t3=llround(b[i].a)%mod;
            ll t4=llround(b[i].b)%mod;
            ans+=(t1+(t2+t3)*M%mod+t4*M*M)%mod;
        }
        return ans;
    }
    #endif
}fft;
```

### 5.3. 多项式全家桶 数组版

- 多项式开根 `g[0]=1`，若 `g[0]` 不是 1 则另需二次剩余板子。
- 若 $\ln f(x)$ 存在，则

$$[x^0]f(x)=1$$

- 若 $\exp f(x)$ 存在，则

$$[x^0]f(x)=0$$

```cpp
void eachfac(ll a[],int n){ // ans[i]=a[i]*i!
    ll p=1;
    repeat(i,1,n)p=p*i%mod,a[i]=a[i]*p%mod;
}
void eachfacinv(ll a[],int n){ // ans[i]=a[i]/i!
    ll p=1;
    repeat(i,1,n)p=p*i%mod,a[i]=a[i]*qpow(p,mod-2)%mod;
}
void shift(ll a[],int n,int p){ // ans=a*x^p
    if(p>0){
        repeat_back(i,p,n)a[i]=a[i-p];
        repeat(i,0,p)a[i]=0;
    }
    if(p<0){
        repeat(i,0,n+p)a[i]=a[i-p];
        repeat(i,n+p,n)a[i]=0;
    }
}
```

```cpp
const int mod=998244353;
inline ll D(ll x){return x>=mod?x-mod:x;}
#define ex(a) fill(a+n,a+n*2,0)
int polyinit(ll a[],int n1){ // a[0..n1-1], n>=n1, n=2^k
    int n=1; while(n<n1)n<<=1;
    fill(a+n1,a+n,0);
    return n;
}
void read(ll a[],int n){
    repeat(i,0,n)a[i]=read();
}
void print(const ll a[],int n){
    repeat(i,0,n)print(a[i],i==n-1);
}
void der(const ll a[],int n,ll b[]){ // n=2^k, b = d a(x) / dx
    repeat(i,1,n){b[i-1]=i*a[i]%mod;} b[n-1]=0;
}
void cal(const ll a[],int n,ll b[]){ // n=2^k, b = integral a(x) dx
    repeat_back(i,1,n){b[i]=qpow(i,mod-2,mod)*a[i-1]%mod;} b[0]=0;
}
void ntt(ll a[],ll n,ll op){ // n=2^k
    for(int i=1,j=n>>1;i<n-1;++i){
        if(i<j)swap(a[i],a[j]);
        int k=n>>1;
        while(k<=j)j-=k,k>>=1;
        j+=k;
    }
    for(int len=2;len<=n;len<<=1){
        ll rt=qpow(3,(mod-1)/len,mod);
        for(int i=0;i<n;i+=len){
            ll w=1;
            repeat(j,i,i+len/2){
                ll u=a[j],t=1ll*a[j+len/2]*w%mod;
                a[j]=D(u+t),a[j+len/2]=D(u-t+mod);
                w=1ll*w*rt%mod;
            }
        }
    }
    if(op==-1){
        reverse(a+1,a+n);
        ll in=qpow(n,mod-2,mod);
        repeat(i,0,n)a[i]=1ll*a[i]*in%mod;
    }
}
void conv(ll a[],ll b[],int n,ll c[],const function<ll(ll,ll)> &f=[](ll a,ll b){return a*b%mod;}){ // n=2^k, c=a*b
    ntt(a,n,1); if(b!=a)ntt(b,n,1);
    repeat(i,0,n)c[i]=f(a[i],b[i]);
    ntt(c,n,-1);
}
void inv(const ll a[],int n,ll b[]){ // n=2^k, a!=b, a*b=1
    static ll t[N];
    if(n==1){b[0]=qpow(a[0],mod-2); b[1]=0; return;}
    inv(a,n/2,b); copy(a,a+n,t);
    ex(b); ex(t);
    conv(b,t,n*2,b,[](ll a,ll b){
        return a*(2-a*b%mod+mod)%mod;
    });
    ex(b);
}
const int inv2=qpow(2,mod-2);
void sqrt(const ll a[],int n,ll b[]){ // n=2^k, a!=b, b*b=a
    static ll f[N],g[N];
    if(n==1){b[0]=1; /*sqrtmod(a[0])*/ return;}
    sqrt(a,n/2,b);
    inv(b,n,f);
    copy(a,a+n,g); ex(g);
    conv(g,f,n*2,g);
    repeat(i,0,n)b[i]=inv2*(g[i]+b[i])%mod;
}
void ln(const ll a[],int n,ll b[]){ // n=2^k, a!=b
    static ll t[N];
    der(a,n,t); inv(a,n,b); ex(t);
    conv(t,b,n*2,t);
    cal(t,n,b); ex(b);
}
void exp(const ll a[],int n,ll b[]){ // n=2^k, a!=b
    static ll t[N];
    if(n==1){b[0]=1; return;}
    exp(a,n/2,b); ln(b,n,t);
    repeat(i,0,n){t[i]=D(a[i]-t[i]+mod);} t[0]++;
    ex(b);
    conv(b,t,n*2,b);
}
void divmod(const ll a[],const ll b[],int n,int m,ll d[],ll r[]){ // n=2^k, m<n, |a|=n, |b|=m, |d|=n-m+1, |r|=m-1, a=d*b+r
    #define er(a,m) fill(a+m,a+n*2,0)
    static ll f[N],g[N];
    reverse_copy(b,b+m,f); er(f,m);
    inv(f,n,g);
    reverse_copy(a,a+n,f);
    conv(f,g,n*2,d);
    reverse(d,d+n-m+1); er(d,n-m+1);
    copy(d,d+n*2,f);
    copy(b,b+n,g); er(g,n);
    conv(f,g,n*2,r);
    repeat(i,0,n*2)r[i]=D(a[i]-r[i]+mod);
}
const ll im=911660635; // im = sqrtmod(-1), imaginary number
namespace tri{
    ll f[N],g[N];
    void getexp(const ll a[],int n){ // n=2^k, f=exp(ia), g=exp(-ia)
        repeat(i,0,n)g[i]=a[i]*im%mod;
        exp(g,n,f);
        inv(f,n,g);
    }
    void sin(const ll a[],int n,ll b[]){ // n=2^k
        getexp(a,n);
        repeat(i,0,n)b[i]=D(f[i]-g[i]+mod)*inv2%mod*(mod-im)%mod;
    }
    void cos(const ll a[],int n,ll b[]){ // n=2^k
        getexp(a,n);
        repeat(i,0,n)b[i]=D(f[i]+g[i])*inv2%mod;
    }
    void tan(const ll a[],int n,ll b[]){ // n=2^k
        getexp(a,n);
        repeat(i,0,n)
            tie(f[i],g[i])=make_pair(
                D(f[i]-g[i]+mod)*inv2%mod*(mod-im)%mod,
                D(f[i]+g[i])*inv2%mod
            );
        inv(g,n,b); ex(f);
        conv(f,b,n*2,b);
    }
    void asin(const ll a[],int n,ll b[]){ // n=2^k
        der(a,n,f); ex(f);
        copy(a,a+n,g); ex(g);
        conv(g,g,n*2,g,[](ll a,ll b){
            return D(1-a*b%mod+mod);
        });
        sqrt(g,n,b); inv(b,n,g);
        conv(f,g,n*2,g); cal(g,n,b);
    }
    void acos(const ll a[],int n,ll b[]){ // n=2^k
        asin(a,n,b);
        repeat(i,0,n)b[i]=D(mod-b[i]);
    }
    void atan(const ll a[],int n,ll b[]){ // n=2^k
        der(a,n,f); ex(f);
        copy(a,a+n,g); ex(g);
        conv(g,g,n*2,g,[](ll a,ll b){
            return D(1+a*b%mod);
        });
        inv(g,n,b); conv(f,b,n*2,g);
        cal(g,n,b);
    }
}
ll getmod(const char s[],int mod){ // ans=s%mod
    ll ans=0;
    repeat(i,0,strlen(s))ans=(ans*10+s[i]-'0')%mod;
    return ans;
}
void qpow_trivial(const ll a[],ll p,int n,ll b[]){ // n=2^k, b=a^p
    static ll t[N];
    ln(a,n,t);
    repeat(i,0,n)(t[i]*=p)%=mod;
    exp(t,n,b);
}
void qpow(const ll a[],const char s[],int n,ll b[]){ // n=2^k, b=a^s
    static ll f[N],g[N];
    ll m=getmod(s,mod),m1=getmod(s,mod-1);
    ll d=0; while(d<n && a[d]==0)d++;
    if(d*m>=n || (d && strlen(s)>=8)){
        fill(b,b+n,0); return;
    }
    int in=qpow(a[d],mod-2,mod),owe=qpow(a[d],m1,mod);
    repeat(i,0,n-d){f[i]=a[i+d]*in%mod;} fill(f+n-d,f+n,0);
    qpow_trivial(f,m,n,g); d*=m;
    repeat(i,0,d)b[i]=0;
    repeat_back(i,d,n)b[i]=g[i-d]*owe%mod;
}
```

### 5.4. 快速沃尔什变换 / FWT

- 计算 $\displaystyle c_i=\sum_{i=f(j,k)}a_jb_k$，$O(n\log n)$。

```cpp
ll &ad(ll &x) { if (x < mod) x += mod; return x = D(x); }
void fwt(ll a[],int n,int flag,char c){ // flag = -1 / 1
    if(c=='|'){
        for(int w=1;w<n;w<<=1)
        for(int i=0;i<n;i+=w*2)
        repeat(j,0,w)
            ad(a[i+j+w]+=a[i+j]*flag);
    }
    else if(c=='&'){
        for(int w=1;w<n;w<<=1)
        for(int i=0;i<n;i+=w*2)
        repeat(j,0,w)
            ad(a[i+j]+=a[i+j+w]*flag);
    }
    else if(c=='^'){
        if(flag==-1)flag=qpow(2,mod-2,mod);
        for(int w=1;w<n;w<<=1)
        for(int i=0;i<n;i+=w*2)
        repeat(j,0,w){
            ad(a[i+j]+=a[i+j+w]);
            ad(ad(a[i+j+w]=a[i+j]-a[i+j+w]*2));
            (a[i+j]*=flag)%=mod;
            (a[i+j+w]*=flag)%=mod;
        }
    }
}
void bitmul(ll a[],ll b[],int n,char c){ // n=2^k
    fwt(a,n,1,c); fwt(b,n,1,c);
    repeat(i,0,n)a[i]=(a[i]*b[i])%mod;
    fwt(a,n,-1,c);
}
```

### 5.5. 多项式复合

- 求 $H(x)\equiv F(G(x)) (\bmod x^{n+1})$，$n=20000$，有点卡常

```cpp
ll RT[17][N];
struct INIT{INIT(){ // 预处理原根
    for(int i=0;i<17;++i){
        ll *G=RT[i]; G[0]=1;
        const int gi=G[1]=qpow(3,(mod-1)/(1<<i+1));
        for(int j=2;j<1<<i;++j)G[j]=G[j-1]*gi%mod;
    }
}}Init;
void ntt(ll a[],ll n,ll op){
    for(int i=1,j=n>>1;i<n-1;++i){
        if(i<j)swap(a[i],a[j]);
        int k=n>>1;
        while(k<=j)j-=k,k>>=1;
        j+=k;
    }
    for(int len=2;len<=n;len<<=1){
        const ll *G=RT[__builtin_ctz(len)-1];
        for(int i=0;i<n;i+=len){
            ll w=1;
            repeat(j,i,i+len/2){
                ll u=a[j],t=1ll*a[j+len/2]*G[j-i]%mod;
                a[j]=D(u+t),a[j+len/2]=D(u-t+mod);
            }
        }
    }
    if(op==-1){
        reverse(a+1,a+n);
        ll in=qpow(n,mod-2,mod);
        repeat(i,0,n)a[i]=1ll*a[i]*in%mod;
    }
}
void conv(ll a[],ll b[],int n,ll c[],const function<ll(ll,ll)> &f=[](ll a,ll b){return a*b%mod;}){ // n=2^k, c=a*b
    ntt(a,n,1); if(b!=a)ntt(b,n,1);
    repeat(i,0,n)c[i]=f(a[i],b[i]);
    ntt(c,n,-1);
}
const int L=142; // sqrt(n1)
ll f[N],g[L+1][N],ng[L+1][N],G[L+1][N],nG[L+1][N];
void prework(ll g[][N],ll ng[][N],int n){
    #define cpy(a,b) copy(a,a+n,b)
    n*=2; g[0][0]=1;
    static ll e[N]; cpy(g[1],e); ntt(e,n,1);
    repeat(i,1,L+1){
        cpy(g[i-1],ng[i-1]); ntt(ng[i-1],n,1);
        repeat(j,0,n)g[i][j]=e[j]*ng[i-1][j]%mod;
        ntt(g[i],n,-1); fill(g[i]+n/2,g[i]+n,0);
    }
}
void Solve(){
    int n1=read()+1,m1=read()+1;
    read(f,n1); int n=polyinit(f,max(n1,m1));
    read(g[1],m1); prework(g,ng,n);
    cpy(g[L],G[1]); prework(G,nG,n);
    static ll ans[N];
    repeat(i,0,L){
        static ll s[N]; fill(s,s+n*2,0);
        repeat(j,0,L){
            int x=i*L+j;
            if(x<n1){
                repeat(k,0,n1)
                    (s[k]+=f[x]*g[j][k])%=mod;
            }
        }
        ntt(s,n*2,1);
        repeat(j,0,n*2)s[j]=s[j]*nG[i][j]%mod;
        ntt(s,n*2,-1);
        repeat(k,0,n1)ad(ans[k]+=s[k]);
    }
    print(ans,n1);
}
```

### 5.6. 多项式多点求值

- 已知多项式 f 和序列 a，求 $f(a_1),f(a_2),\ldots,f(a_m)$
- 线性算法指输入 n 维向量 x，经过 $m\times n$ 矩阵 A 变换后输出 m 维向量 $y=Ax$ 的算法
- 转置原理指出，如果存在 $x'=A^Ty'$ 的算法，那么就有存在相同复杂度的 $y=Ax$ 的算法。将 $A^T$ 分解为三种指令 `x[i]+=x[j],x[i]*=c,swap(x[i],x[i])`，那么 A 即倒着执行这些指令，并且将第一种指令变为 `x[j]+=x[i]`。（$A=E_1E_2\ldots E_k\rightarrow A^T=E_k^TE_{k-1}^T\ldots E_1^T$）
- $O(n\log^2n)$，常数极大（1.8s 跑 64000）

```cpp
ll D(ll x){return x>=mod?x-mod:x;}
ll &ad(ll &x){return x=D(x);}
typedef vector<ll> vi;
#define rs(a) [&]{if((int)a.size()<n)a.resize(n,0);}()
#define cut(a) fill(a.begin()+n/2,a.begin()+n,0)
int polyn(int n1){ // return 2^k >= n1
    return 1<<(__lg(n1-1)+1);
}
void ntt(vi &a,ll n,ll op){ // n=2^k
    rs(a);
    for(int i=1,j=n>>1;i<n-1;++i){
        if(i<j)swap(a[i],a[j]);
        int k=n>>1;
        while(k<=j)j-=k,k>>=1;
        j+=k;
    }
    for(int len=2;len<=n;len<<=1){
        ll rt=qpow(3,(mod-1)/len,mod);
        for(int i=0;i<n;i+=len){
            ll w=1;
            repeat(j,i,i+len/2){
                ll u=a[j],t=1ll*a[j+len/2]*w%mod;
                a[j]=D(u+t),a[j+len/2]=D(u-t+mod);
                w=1ll*w*rt%mod;
            }
        }
    }
    if(op==-1){
        reverse(a.begin()+1,a.begin()+n);
        ll in=qpow(n,mod-2,mod);
        repeat(i,0,n)a[i]=1ll*a[i]*in%mod;
    }
}
vi conv(vi a,vi b,int n,const function<ll(ll,ll)> &f=[](ll a,ll b){return a*b%mod;}){ // n=2^k, ans=a*b
    n*=2; rs(a),rs(b); cut(a),cut(b);
    ntt(a,n,1); ntt(b,n,1);
    repeat(i,0,n)a[i]=f(a[i],b[i]);
    ntt(a,n,-1); cut(a);
    return a;
}
vi inv(const vi &a,int n){ // n=2^k, ans=1/a
    if(n==1)return vi(1,qpow(a[0],mod-2,mod));
    return conv(inv(a,n/2),a,n,[](ll a,ll b){
        return a*(2-a*b%mod+mod)%mod;
    });
}
vi convauto(vi a,vi b,const function<ll(ll,ll)> &f=[](ll a,ll b){return a*b%mod;}){
    int n1=a.size()+b.size()-1,n=polyn(n1);
    rs(a),rs(b);
    ntt(a,n,1); ntt(b,n,1);
    repeat(i,0,n)a[i]=f(a[i],b[i]);
    ntt(a,n,-1);
    a.resize(n1);
    return a;
}
vi convtr(vi a,vi b,const function<ll(ll,ll)> &f=[](ll a,ll b){return a*b%mod;}){
    int n1=a.size()+b.size()-1,n=polyn(n1);
    rs(a),rs(b);
    reverse(a.begin(),a.end());
    ntt(a,n,1); ntt(b,n,1);
    repeat(i,0,n)a[i]=f(a[i],b[i]);
    ntt(a,n,-1);
    reverse(a.begin(),a.end());
    return a;
}
vi prod[N]; ll ans[N],a[N];
#define lc x*2
#define rc x*2+1
void getprod(int x,int l,int r){
    if(l==r){
        prod[x]={1,D(mod-a[l])};
        return;
    }
    int mid=(l+r)/2;
    getprod(lc,l,mid);
    getprod(rc,mid+1,r);
    prod[x]=convauto(prod[lc],prod[rc]);
}
void dfs(int x,int l,int r,vi G){
    G.resize(r-l+1);
    if(l==r){
        ans[l]=G[0];
        return;
    }
    int mid=(l+r)/2;
    dfs(lc,l,mid,convtr(G,prod[rc]));
    dfs(rc,mid+1,r,convtr(G,prod[lc]));
}
void Solve(){
    int n=read()+1,m=read();
    vi f; repeat(i,0,n)f<<read(); // poly
    repeat(i,0,m)
        a[i]=read(); // query
    getprod(1,0,m-1);
    vi v=inv(prod[1],polyn(m+1));
    dfs(1,0,m-1,convtr(f,v));
    repeat(i,0,m)
        print(ans[i],1);
}
```

### 5.7. 多项式快速插值

- $O(n\log^2 n)$，常数极大（2.85s 跑 100000）

```cpp
const int N=(1<<18)+5;
int D(int x){return x>=mod?x-mod:x;}
int r[19][N],w[2][N],lg[N],inv[19];
void Pre(){
    repeat(d,1,18+1){
        repeat(i,1,(1<<d))r[d][i]=(r[d][i>>1]>>1)|((i&1)<<(d-1));
        lg[1<<d]=d,inv[d]=qpow(1<<d,mod-2);
    }
    for(int t=(mod-1)>>1,i=1,x,y; i<262144; i<<=1,t>>=1){
        x=qpow(3,t),y=qpow(332748118,t),w[0][i]=w[1][i]=1;
        repeat(k,1,i){
            w[1][k+i]=mul(w[1][k+i-1],x);
            w[0][k+i]=mul(w[0][k+i-1],y);
        }
    }
}
int lim,e,n,m;
void init(int len){
    lim=1,e=0;
    while(lim<len)lim<<=1,++e;
}
void NTT(int *a,int ty){
    repeat(i,0,lim)if(i<r[e][i])swap(a[i],a[r[e][i]]);
    for(int mid=1; mid<lim; mid<<=1)
    for(int j=0,t; j<lim; j+=(mid<<1))
    repeat(k,0,mid){
        t=mul(w[ty][mid+k],a[j+k+mid]);
        a[j+k+mid]=D(a[j+k]-t+mod),a[j+k]=D(a[j+k]+t);
    }
    if(!ty)repeat(i,0,lim)a[i]=mul(a[i],inv[e]);
}
void Inv(int *a,int *b,int len){
    if(len==1)return b[0]=qpow(a[0],mod-2),void();
    Inv(a,b,len>>1),lim=(len<<1),e=lg[lim];
    static int c[N],d[N];
    repeat(i,0,len)c[i]=a[i],d[i]=b[i];
    repeat(i,len,lim)c[i]=d[i]=0;
    NTT(c,1),NTT(d,1);
    repeat(i,0,lim)c[i]=mul(c[i],mul(d[i],d[i]));
    NTT(c,0);
    repeat(i,0,len)b[i]=D(D(b[i]*2)-c[i]+mod);
    repeat(i,len,lim)b[i]=0;
}
struct node {
    node *lc,*rc;
    vector<int> vec;
    int deg;
    void Mod(const int *a,int *r,int n){
        static int b[N],c[N],d[N];
        int len=1;
        while(len<=n-deg)len<<=1;
        repeat(i,0,n+1)b[i]=a[n-i];
        repeat(i,0,deg+1)c[i]=vec[deg-i];
        repeat(i,n-deg+1,len)c[i]=0;
        Inv(c,d,len);
        lim=(len<<1),e=lg[lim];
        repeat(i,n-deg+1,lim)b[i]=d[i]=0;
        NTT(b,1),NTT(d,1);
        repeat(i,0,lim)b[i]=mul(b[i],d[i]);
        NTT(b,0);
        reverse(b,b+n-deg+1);
        init(n+1);
        repeat(i,n-deg+1,lim)b[i]=0;
        repeat(i,0,deg+1)c[i]=vec[i];
        repeat(i,deg+1,lim)c[i]=0;
        NTT(b,1),NTT(c,1);
        repeat(i,0,lim)b[i]=mul(b[i],c[i]);
        NTT(b,0);
        repeat(i,0,deg)r[i]=D(a[i]-b[i]+mod);
    }
    void Mul(){
        static int a[N],b[N];
        deg=lc->deg+rc->deg,vec.resize(deg+1),init(deg+1);
        repeat(i,0,lc->deg+1)a[i]=lc->vec[i];
        repeat(i,lc->deg+1,lim)a[i]=0;
        repeat(i,0,rc->deg+1)b[i]=rc->vec[i];
        repeat(i,rc->deg+1,lim)b[i]=0;
        NTT(a,1),NTT(b,1);
        repeat(i,0,lim)a[i]=mul(a[i],b[i]);
        NTT(a,0);
        repeat(i,0,deg+1)vec[i]=a[i];
    }
} pool[N],*rt;
struct bnode {
    bnode *lc,*rc;
    vector<int> vec;
    int l,r;
    void Mul(node* p){
        static int a[N],b[N],c[N],d[N];
        int mid=(l+r)>>1;
        init(r-l+1+1);
        repeat(i,0,mid-l+1+1)a[i]=lc->vec[i],c[i]=p->lc->vec[i];
        repeat(i,mid-l+2,lim)a[i]=c[i]=0;
        repeat(i,0,r-mid+1)b[i]=rc->vec[i],d[i]=p->rc->vec[i];
        repeat(i,r-mid+1,lim)b[i]=d[i]=0;
        NTT(a,1),NTT(b,1),NTT(c,1),NTT(d,1);
        repeat(i,0,lim)a[i]=D(mul(a[i],d[i])+mul(b[i],c[i]));
        NTT(a,0);
        vec.resize(r-l+2);
        repeat(i,0,r-l+1+1)vec[i]=a[i];
    }
} o[N],*brt;
int a[N],tot,cnt;
node *newnode(){
    return &pool[tot++];
}
bnode *newbnode(){
    return &o[cnt++];
}
void solve(node *&p,int l,int r){
    p=newnode();
    if(l==r)return p->deg=1,p->vec.resize(2),p->vec[0]=mod-a[l],p->vec[1]=1,void();
    int mid=(l+r)>>1;
    solve(p->lc,l,mid),solve(p->rc,mid+1,r);
    p->Mul();
}
int b[25],f[N];
void calc(node *p,int l,int r,const int *d){
    if(r-l<=512){
        repeat(i,l,r+1){
            int x=a[i],c1,c2,c3,c4,now=d[r-l];
            b[0]=1;
            repeat(j,1,16+1)b[j]=mul(b[j-1],x);
            for(int j=r-l-1; j-15>=0; j-=16){
                c1=(1ll*now*b[16]+1ll*d[j]*b[15]+1ll*d[j-1]*b[14]+1ll*d[j-2]*b[13])%mod,
                c2=(1ll*d[j-3]*b[12]+1ll*d[j-4]*b[11]+1ll*d[j-5]*b[10]+1ll*d[j-6]*b[9])%mod,
                c3=(1ll*d[j-7]*b[8]+1ll*d[j-8]*b[7]+1ll*d[j-9]*b[6]+1ll*d[j-10]*b[5])%mod,
                c4=(1ll*d[j-11]*b[4]+1ll*d[j-12]*b[3]+1ll*d[j-13]*b[2]+1ll*d[j-14]*b[1])%mod,
                now=(0ll+c1+c2+c3+c4+d[j-15])%mod;
            }
            repeat_back(j,0,(r-l)%16)now=(1ll*now*x+d[j])%mod;
            f[i]=now;
        }
        return;
    }
    int mid=(l+r)>>1,b[p->deg+1];
    p->lc->Mod(d,b,p->deg-1),calc(p->lc,l,mid,b);
    p->rc->Mod(d,b,p->deg-1),calc(p->rc,mid+1,r,b);
}
int x[N],y[N],A[N];
void loli(bnode *&u,node *p,int l,int r){
    u=newbnode(),u->l=l,u->r=r;
    if(l==r)return u->vec.resize(2),u->vec[0]=mul(y[l],qpow(f[l],mod-2)),u->vec[1]=0,void();
    int mid=(l+r)>>1;
    loli(u->lc,p->lc,l,mid),loli(u->rc,p->rc,mid+1,r);
    u->Mul(p);
}
int main(){
    n=read(),Pre();
    repeat(i,1,n+1)x[i]=a[i]=read(),y[i]=read();
    solve(rt,1,n);
    repeat(i,1,n+1)A[i-1]=mul(rt->vec[i],i);
    A[n]=0;
    calc(rt,1,n,A);
    loli(brt,rt,1,n);
    repeat(i,0,n)print(brt->vec[i]);
    return 0;
}
```

### 5.8. k 进制异或卷积 / k 进制 FWT

- 令 $a \oplus_k b$ 为 k 进制异或（k 进制无进位加法）
- 求 $\displaystyle c_m=\sum_{i\oplus_k j=m}a_ib_j$

```cpp
const int k=7;
const ll rt=qpow(3,(mod-1)/k);
void kfwt(ll a[],int n,int op){ // n = k^(int), op = -1 / 1
    static ll t[k],w[k];
    w[0]=1; repeat(i,1,k)w[i]=w[i-1]*rt%mod;
    for(int len=1;len<n;len*=k)
    for(int i=0;i<n;i+=len*k)
    repeat(j,i,i+len){
        repeat(x,0,k){
            t[x]=0;
            repeat(y,0,k)
                (t[x]+=a[j+y*len]*w[y*(k+op*x)%k])%=mod;
        }
        repeat(x,0,k)a[j+x*len]=t[x];
    }
    if(op==-1){
        ll inv=qpow(n,mod-2);
        repeat(i,0,n)a[i]=a[i]*inv%mod;
    }
}
```

### 5.9. 任意长度 FFT using Bluestein 算法

- 编号从 0 开始，$O(n\log n)$。

```cpp
typedef complex<lf> cp;
void DFT(cp a[],int n,int op){ // op=-1 时不是真正的 IDFT
    static int r[N]{};
    repeat(i,1,n){
        r[i]=(r[i>>1]>>1)|(i&1?n>>1:0);
        if(i<r[i])swap(a[i],a[r[i]]);
    }
    for(int len=2;len<=n;len<<=1){
        cp w=cp(cos(pi*2/len),op*sin(pi*2/len));
        for(int i=0;i<n;i+=len){
            cp d(1,0);
            repeat(j,i,i+len/2){
                cp x=a[j],y=a[j+len/2]*d;
                a[j]=x+y; a[j+len/2]=x-y;
                d=d*w;
            }
        }
    }
}
void bluestein(cp a[],int n,int op){ // n is arbitrary, op in {-1,1}
    static cp t[N],u[N];
    int k=1;
    while(k<4*n)k<<=1;
    repeat(i,0,k)t[i]=u[i]=0;
    repeat(i,0,n)
        t[i]=a[i]*cp(cos(pi*i*i/n),op*sin(pi*i*i/n));
    repeat(i,0,n*2)
        u[i]=cp(cos(pi*(i-n)*(i-n)/n),-op*sin(pi*(i-n)*(i-n)/n));
    DFT(t,k,1); DFT(u,k,1);
    repeat(i,0,k)t[i]=t[i]*u[i];
    DFT(t,k,-1);
    repeat(i,0,n)
        a[i]=t[i+n]*cp(cos(pi*i*i/n),op*sin(pi*i*i/n))/lf(k);
}
```

### 5.10. 递推式插值 using BM 算法

- 已知数列前几项，求递推式系数 $C_0a_i+C_1a_{i+1}+...+C_ka_{i+k}=0,C_k=-1$
- 用来找规律

```cpp
using vtr = vector<ll>;
vtr mul(const vtr &a, const vtr &b, const vtr &m, int k) {
    vtr r(2 * k - 1);
    repeat(i, 0, k) repeat(j, 0, k) (r[i + j] += a[i] * b[j]) %= mod;
    repeat_back(i, 0, k - 1) {
        repeat(j, 0, k) (r[i + j] += r[i + k] * m[j]) %= mod;
        r.pop_back();
    }
    return r;
}
vtr qpow(const vtr &m, ll n) {
    int k = (int)m.size() - 1;
    vtr r(k), x(k);
    r[0] = x[1] = 1;
    for (; n; n >>= 1, x = mul(x, x, m, k))
        if (n & 1) r = mul(x, r, m, k);
    return r;
}
ll go(const vtr &a, const vtr &x, ll n) {
    int k = (int)a.size() - 1;
    if (n <= k) return x[n - 1];
    if (a.size() == 2) return x[0] * qpow(a[0], n - 1, mod) % mod;
    vtr r = qpow(a, n - 1);
    ll ans = 0;
    repeat(i, 0, k) (ans += r[i] * x[i]) %= mod;
    return (ans + mod) % mod;
}
vtr BM(const vtr &x) {
    vtr C = {-1}, B = {-1};
    ll L = 0, m = 1, b = 1;
    repeat(n, 0, x.size()) {
        ll d = 0;
        repeat(i, 0, L + 1) (d += C[i] * x[n - i]) %= mod;
        if (d == 0) { m++; continue; }
        vtr T = C;
        ll c = mod - d * qpow(b, mod - 2, mod) % mod;
        repeat(i, C.size(), B.size() + m) C.push_back(0);
        repeat(i, 0, B.size()) (C[i + m] += c * B[i]) %= mod;
        if (2 * L > n) { m++; continue; }
        L = n + 1 - L; B.swap(T); b = d; m = 1;
    }
    reverse(C.begin(), C.end());
    return C;
}
```

### 5.11. 分治 FFT

- 比如 $\displaystyle f[i]=\sum_{j=1}^if[i-j]g[i]$，把要求的多项式分成两边，先算 $f[0..n-1]$ 对自己的贡献（此时 $f[0..n-1]$ 已确定），然后算 $f[0..n-1]$ 对 $f[n..2n-1]$ 的贡献，再算 $f[n..2n-1]$ 对自己的贡献，$O(n\log^2n)$

```cpp
int g[N],f[N],GG[N],FF[N];
void work(int l,int r){
    if(r-l==1)return;
    int m=(l+r)/2;
    work(l,m);
    copy(g,g+r-l,GG);
    copy(f+l,f+m,FF+l); fill(FF+m,FF+r,0);
    conv(GG,FF+l,r-l,FF+l);
    repeat(i,m,r)ad(f[i]+=FF[i]);
    work(m,r);
}
// int n=polyinit(g,n1); fill(f,f+n,0); f[0]=1; work(0,n);
```

- 卡特兰数 $\displaystyle S_n=\sum_{k=0}^{n-1}S_kS_{n-1-k}$

```cpp
int f[N],GG[N],FF[N];
void work(int l,int r){
    if(r-l==1)return;
    int m=(l+r)/2;
    work(l,m);
    copy(f,f+r-l,GG+1); GG[0]=0;
    copy(f+l,f+m,FF+l); fill(FF+m,FF+r,0);
    conv(GG,FF+l,r-l,FF+l);
    int t=1+(l!=0);
    repeat(i,m,r)(f[i]+=FF[i]*t)%=mod;
    work(m,r);
}
// fill(f,f+n,0); f[0]=1; work(0,n);
```

- 超级卡特兰数 $\displaystyle S_n=S_{n-1}+\sum_{k=0}^{n-1}S_kS_{n-1-k}$

```cpp
ll f[N],GG[N],FF[N],f0[N];
void work(int l,int r){
    if(r-l==1)return;
    int m=(l+r)/2;
    work(l,m);
    copy(f,f+r-l,GG+1); GG[0]=0;
    copy(f+l,f+m,FF+l); fill(FF+m,FF+r,0);
    conv(GG,FF+l,r-l,FF+l);
    int t=1+(l!=0);
    repeat(i,m,r)(f0[i]+=f0[i-1]+FF[i]*t)%=mod;
    ad(f0[r]+=f0[r-1]);
    repeat(i,m,r)ad(f[i]+=f0[i]),f0[i]=0;
    work(m,r);
}
// fill(f,f+n,0); fill(f0,f0+n,0); f[0]=f0[0]=1; work(0,n);
```

### 5.12. 第二类斯特林数·行

- $\displaystyle S(n,r)=[x^r](\sum_{i=0}^n\dfrac{(-1)^i}{i!}x^i)(\sum_{i=0}^{n}\dfrac{i^n}{i!}x^i)$

```cpp
repeat(i,0,n1+1){
    a[i]=C.inv[i]; if(i%2==1)a[i]=-a[i];
    b[i]=qpow(i,n1)*C.inv[i]%mod;
}
int n=polyinit(a,n1+1); polyinit(b,n1+1);
conv(a,b,n,a);
```

## 6. 矩阵

### 6.1. 矩阵乘法 and 矩阵快速幂

- 已并行优化，矩乘 $O(n^3)$，矩快 $O(n^3\log m)$

```cpp
struct mat {
    static const int N = 110;
    ll a[N][N];
    explicit mat(ll e = 0) {
        repeat (i, 0, n)
        repeat (j, 0, n)
            a[i][j] = e * (i == j);
    }
    mat operator*(const mat &b) const {
        mat ans(0);
        repeat (i, 0, n)
        repeat (k, 0, n) {
            ll t = a[i][k];
            repeat (j, 0, n)
                (ans.a[i][j] += t * b.a[k][j]) %= mod;
        }
        return ans;
    }
    vector<ll> operator*(const vector<ll> &b) const {
        vector<ll> ans(n);
        repeat (i, 0, n)
        repeat (j, 0, n)
            (ans[i] += a[i][j] * b[j]) %= mod;
        return ans;
    }
    void input() {
        repeat (i, 0, n) repeat (j, 0, n) a[i][j] = read();
    }
    void print() {
        cout << "mat size = " << n << endl;
        repeat (i, 0, n)
        repeat (j, 0, n)
            printf("%lld%c", a[i][j], " \n"[j == n - 1]);
    }
    friend mat qpow(mat a, ll b) {
        mat ans(1);
        while (b) {
            if (b & 1) ans = ans * a;
            a = a * a; b >>= 1;
        }
        return ans;
    }
    ll *operator[](int x) { return a[x]; }
    const ll *operator[](int x) const { return a[x]; }
};
```

### 6.2. 高斯消元

- 求行列式、[解线性方程组](https://www.luogu.com.cn/problem/P3389)、[求逆矩阵](https://www.luogu.com.cn/problem/P4783)
- 编号从 1 开始，$O(n^3)$
- 质数模数版

```cpp
struct mat{
    static const int N=410;
    vector<ll> a[N];
    mat(){for(auto &i:a)i.assign(N*2,0);} // if get_inv is needed, N*2
    ll det;
    void r_div(int x,int m,ll k){ // a[x][]/=k
        ll r=qpow(k,mod-2);
        repeat(i,0,m)
            a[x][i]=a[x][i]*r%mod;
        det=det*k%mod;
    }
    void r_plus(int x,int y,int m,ll k){ // a[x][]+=a[y][]*k
        repeat(i,0,m)
            a[x][i]=(a[x][i]+a[y][i]*k)%mod;
    }
    bool gauss(int n,int m){ // n<=m, return whether succuss
        det=1;
        repeat(i,0,n){
            int t=-1;
            repeat(j,i,n)if(a[j][i]){t=j; break;}
            if(t==-1){det=0; return 0;}
            if(t!=i){a[i].swap(a[t]); det=-det;}
            r_div(i,m,a[i][i]);
            repeat(j,0,n)
            if(j!=i && a[j][i])
                r_plus(j,i,m,mod-a[j][i]);
        }
        return 1;
    }
    ll get_det(int n){gauss(n,n); return (det+mod)%mod;} // return det
    bool get_inv(int n){ // self=inv(self), return whether success
        repeat(i,0,n)
        repeat(j,0,n)
            a[i][j+n]=(i==j);
        bool t=gauss(n,n*2);
        repeat(i,0,n)
        repeat(j,0,n)
            a[i][j]=a[i][j+n];
        return t;
    }
    vector<ll> &operator[](int x){return a[x];}
    const vector<ll> &operator[](int x)const{return a[x];}
}a;
```

- 浮点版

```cpp
struct mat{
    static const int N=110;
    vector<lf> a[N];
    mat(){for(auto &i:a)i.assign(N*2,0);} // if get_inv is needed, N*2
    lf det;
    void r_div(int x,int m,lf k){ // a[x][]/=k
        lf r=1/k;
        repeat(i,0,m)a[x][i]*=r;
        det*=k;
    }
    void r_plus(int x,int y,int m,lf k){ // a[x][]+=a[y][]*k
        repeat(i,0,m)a[x][i]+=a[y][i]*k;
    }
    bool gauss(int n,int m){ // n<=m, return whether succuss
        det=1;
        repeat(i,0,n){
            int t=-1;
            repeat(j,i,n)if(abs(a[j][i])>eps){t=j; break;}
            if(t==-1){det=0; return 0;}
            if(t!=i){a[i].swap(a[t]); det=-det;}
            r_div(i,m,a[i][i]);
            repeat(j,0,n)
            if(j!=i && abs(a[j][i])>eps)
                r_plus(j,i,m,-a[j][i]);
        }
        return 1;
    }
    lf get_det(int n){gauss(n,n); return det;} // return det
    bool get_inv(int n){ // self=inv(self), return whether success
        repeat(i,0,n)
        repeat(j,0,n)
            a[i][j+n]=(i==j);
        bool t=gauss(n,n*2);
        repeat(i,0,n)
        repeat(j,0,n)
            a[i][j]=a[i][j+n];
        return t;
    }
    vector<lf> &operator[](int x){return a[x];}
    const vector<lf> &operator[](int x)const{return a[x];}
}a;
```

- [任意模数行列式](http://acm.hdu.edu.cn/showproblem.php?pid=2827)
- $O(n^3\log C)$

```cpp
int n;
struct mat{
    static const int N=110;
    vector< vector<ll> > a;
    mat():a(N,vector<ll>(N)){}
    ll det(int n){
        ll ans=1;
        repeat(i,0,n){
            repeat(j,i+1,n)
            while(a[j][i]){
                ll t=a[i][i]/a[j][i];
                repeat(k,i,n)a[i][k]=(a[i][k]-a[j][k]*t)%mod;
                swap(a[i],a[j]);
                ans=-ans;
            }
            ans=ans*a[i][i]%mod;
            if(!ans)return 0;
        }
        return (ans+mod)%mod;
    }
}a;
```

### 6.3. 异或方程组

- 编号从 0 开始，高斯消元部分 $O(n^3)$（luogu P2962）

```cpp
bitset<N> a[N]; bool l[N];
int n,ans;
int gauss(int n){ // -1 : no solution, 0 : multi, 1 : single
    repeat(i,0,n){
        int t=-1;
        repeat(j,i,n)if(a[j][i]){t=j; break;}
        if(t==-1)continue;
        if(t!=i)swap(a[i],a[t]);
        repeat(j,0,n)
        if(i!=j && a[j][i])
            a[j]^=a[i];
    }
    repeat(i,0,n)if(!a[i][i] && a[i][n])return -1;
    repeat(i,0,n)if(!a[i][i])return 0;
    return 1;
}
void dfs(int x=n-1,int num=0){
    if(num>ans)return;
    if(x==-1){ans=num; return;}
    if(a[x][x]){
        bool v=a[x][n];
        repeat(i,x+1,n)
        if(a[x][i])
            v^=l[i];
        dfs(x-1,num+v);
    }
    else{
        dfs(x-1,num);
        l[x]=1;
        dfs(x-1,num+1);
        l[x]=0;
    }
}
int solve(){ // 返回满足方程组的sum(xi)最小值
    ans=inf; gauss(n); dfs(n-1,0);
    return ans;
}
```

### 6.4. 线性基

- 线性基是一系列线性无关的基向量组成的集合

异或线性基

- 结论：$basis.exist(a^b)$ 等价于 a, b 在 $basis$ 里消去关键位后相等（要求是最简线性基，即第一个板子）
- 插入、查询 $O(\log M)$

```cpp
struct basis{
    static const int n=63;
    #define B(x,i) ((x>>i)&1)
    ll a[n],sz;
    bool failpush; // 是否线性相关
    void init(){mst(a,0); sz=failpush=0;}
    void push(ll x){ // 插入元素
        repeat(i,0,n)if(B(x,i))x^=a[i];
        if(x!=0){
            int p=__lg(x); sz++;
            repeat(i,p+1,n)if(B(a[i],p))a[i]^=x;
            a[p]=x;
        }
        else failpush=1;
    }
    ll top(){ // 最大值
        ll ans=0;
        repeat(i,0,n)ans^=a[i];
        return ans;
    }
    bool exist(ll x){ // 是否存在
        repeat_back(i,0,n)
        if((x>>i)&1){
            if(a[i]==0)return 0;
            else x^=a[i];
        }
        return 1;
    }
    ll kth(ll k){ // 第k小，不存在返回-1
        if(failpush)k--; // 如果认为0是可能的答案就加这句话
        if(k>=(1ll<<sz))return -1;
        ll ans=0;
        repeat(i,0,n)
        if(a[i]!=0){
            if(k&1)ans^=a[i];
            k>>=1;
        }
        return ans;
    }
}b;
basis operator+(basis a,const basis &b){ // 将b并入a
    repeat(i,0,a.n)
    if(b.a[i])a.push(b.a[i]);
    a.failpush|=b.failpush;
    return a;
}
```

- 这个版本中求 kth 需要 rebuild $O(\log^2 n)$

```cpp
struct basis{
    // ...
    void push(ll x){ // 插入元素
        repeat_back(i,0,n)
        if((x>>i)&1){
            if(a[i]==0){a[i]=x; sz++; return;}
            else x^=a[i];
        }
        failpush=1;
    }
    ll top(){ // 最大值
        ll ans=0;
        repeat_back(i,0,n)
            ans=max(ans,ans^a[i]);
        return ans;
    }
    void rebuild(){ // 求第k小的前置操作
        repeat_back(i,0,n)
        repeat_back(j,0,i)
        if((a[i]>>j)&1)
            a[i]^=a[j];
    }
}b;
```

实数线性基

- 编号从 0 开始，插入、查询 $O(n^2)$

```cpp
struct basis{
    lf a[N][N]; bool f[N]; int n; // f[i]表示向量a[i]是否被占
    void init(int _n){
        n=_n;
        fill(f,f+n,0);
    }
    bool push(lf x[]){ // 返回0表示可以被线性表示，不需要插入
        repeat(i,0,n)
        if(abs(x[i])>1e-5){ // 这个值要大一些
            if(f[i]){
                lf t=x[i]/a[i][i];
                repeat(j,0,n)x[j]-=t*a[i][j];
            }
            else{
                f[i]=1;
                repeat(j,0,n)a[i][j]=x[j];
                return 1;
            }
        }
        return 0;
    }
}b;
```

线性基求交

- 未测试，$O(\log^2 W)$。

```cpp
typedef array<int,64> Base;
Base merge(Base a,Base b){
    Base tmp(a),ans{};
    int cur,d;
    repeat(i,0,64)if(b[i]){
        cur=0,d=b[i];
        repeat_back(j,0,i+1)if(d>>j&1){
            if(tmp[j]){
                d^=tmp[j],cur^=a[j];
                if(d)continue;
                ans[i]=cur;
            }
            else tmp[j]=d,a[j]=cur;
            break;
        }
    }
    return ans;
}
```

### 6.5. 线性规划 using 单纯形法

- 声明：还没学会

$$\left[\begin{array}{ccccccc} a & a & a & a & a & a & b \newline a & a & a & a & a & a & b \newline a & a & a & a & a & a & b \newline c & c & c & c & c & c & v \end{array}\right]$$

- 每行表示一个约束，$\sum ax\le b$，并且所有 $x\ge 0$，求 $\sum cx$ 的最大值
- 对偶问题：每列表示一个约束，$\sum ax\ge c$，并且所有 $x\ge 0$，求 $\sum bx$ 的最小值
- 先找 `c[y]>0` 的 y，再找 `b[x]>0` 且 `b[x]/a[x][y]` 最小的  x（找不到 y 则 v，找不到 x 则 INF），用行变换将 `a[x][y]` 置 1，将其他 `a[i][y]` 和 `c[y]` 置 0
- 编号从 1 开始，$O(n^3)$，缺init

```cpp
const int M=1010; const lf eps=1e-6;
int n,m;
lf a[N][M],b[N],c[M],v; // a[1..n][1..m],b[1..n],c[1..m]
void pivot(int x,int y){
    b[x]/=a[x][y];
    repeat(j,1,m+1)if(j!=y)
        a[x][j]/=a[x][y];
    a[x][y]=1/a[x][y];
    repeat(i,1,n+1)
    if(i!=x && abs(a[i][y])>eps){
        b[i]-=a[i][y]*b[x];
        repeat(j,1,m+1)if(j!=y)
            a[i][j]-=a[i][y]*a[x][j];
        a[i][y]=-a[i][y]*a[x][y];
    }
    v+=c[y]*b[x];
    repeat(j,1,m+1)if(j!=y)
        c[j]-=c[y]*a[x][j];
    c[y]=-c[y]*a[x][y];
}
lf simplex(){ // 返回INF表示无限制，否则返回答案
    while(1){
        int x,y;
        for(y=1;y<=m;y++)if(c[y]>eps)break;
        if(y==m+1)return v;
        lf mn=INF;
        repeat(i,1,n+1)
        if(a[i][y]>eps && mn>b[i]/a[i][y])
            mn=b[i]/a[i][y],x=i;
        if(mn==INF)return INF; // unbounded
        pivot(x,y);
    }
}
void init(){v=0;}
```

## 7. 数学杂项

struct of 自动取模

- 未测，不好用，别用了。

```cpp
struct mint{
    ll v;
    mint(ll _v){v=_v%mod;}
    mint operator+(const mint &b)const{return v+b.v;}
    mint operator-(const mint &b)const{return v-b.v;}
    mint operator*(const mint &b)const{return v*b.v;}
    explicit operator ll(){return (v+mod)%mod;}
};
```

### 7.1. struct of 区间

- 未测，不好用，别用了。
- 用 `pii` 即 `pair<int, int>` 表示闭区间。

```cpp
bool isinter(pii a, pii b) { // 判断是否相交，相交于一个值也可以
    int l = max(a.fi, b.fi), r = min(a.se, b.se);
    return l <= r;
}
pii inter(pii a, pii b) { // 求交集
    int l = max(a.fi, b.fi), r = min(a.se, b.se);
    if (l > r) return {0, -1};
    return {l, r};
}
pii unioned(pii a, pii b) { // 求并集，相交 (isinter) 结果才有意义
    return {min(a.fi, b.fi), max(a.se, b.se)};
}
void simplify(vector<pii> &a) { // 集合 a 的最简区间表示
    if (a.empty()) return;
    sort(a.begin(), a.end());
    int pre = 0;
    for (int i = 1; i < (int)a.size(); i++) {
        if (isinter(a[pre], a[i]))
            a[pre] = unioned(a[pre], a[i]);
        else
            a[++pre] = a[i];
    }
    a.erase(a.begin() + pre + 1, a.end());
}
vector<pii> inter(const vector<pii> &a, const vector<pii> &b) { // 求集合 a 和集合 b 的交集（a, b 可能需要 simplify 一下）
    int i = 0, j = 0;
    vector<pii> ans;
    while (i < (int)a.size() && j < (int)b.size()) {
        if (isinter(a[i], b[j])) {
            ans.push_back(inter(a[i], b[j]));
        }
        if (a[i].se < b[j].se) i++;
        else j++;
    }
    return ans;
}
```

### 7.2. struct of 高精度

- 乘除 $O(n^2)$，且除法常数巨大。
- [二进制高精度参考写法](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/BigInt.cpp)（无除法）。
- 一般建议 Java。

```cpp
struct big{
    vector<ll> a;
    static const ll k=1000000000,w=9;
    int size()const{return a.size();} // memory size
    explicit big(const ll &x=0){ // from ll
        *this=big(to_string(x));
    }
    explicit big(const string &s){ // from string
        static ll p10[9]={1};
        repeat(i,1,w)p10[i]=p10[i-1]*10;
        int len=s.size();
        int f=(s[0]=='-')?-1:1;
        a.resize(len/w+1);
        repeat(i,0,len-(f==-1))
            a[i/w]+=f*(s[len-1-i]-48)*p10[i%w];
        adjust();
    }
    int sgn(){return a.back()>=0?1:-1;} // sign (please used after adjust())
    void shrink(){ // pop zeros (will not release memory)
        while(size()>1 && a.back()==0)a.pop_back();
    }
    void adjust(){ // weak adjust, a[i] in (-k, k)
        repeat(i,0,3)a.push_back(0);
        repeat(i,0,size()-1){
            a[i+1]+=a[i]/k;
            a[i]%=k;
        }
        shrink();
    }
    void final_adjust(){ // strong adjust, a[i] have same sign
        adjust();
        int f=sgn();
        repeat(i,0,size()-1){
            ll t=(a[i]+k*f)%k;
            a[i+1]+=(a[i]-t)/k;
            a[i]=t;
        }
        shrink();
    }
    explicit operator string(){ // to string
        static char s[N]; char *p=s;
        final_adjust();
        if(sgn()==-1)*p++='-';
        repeat_back(i,0,size()){
            sprintf(p,i==size()-1?"%lld":"%09lld",abs(a[i]));
            p+=strlen(p);
        }
        return s;
    }
    const ll &operator[](int n)const{ // visit
        return a[n];
    }
    ll &operator[](int n){ // flexible visit
        repeat(i,0,n-size()+1)a.push_back(0);
        return a[n];
    }
    friend big operator+(big a,const big &b){ // <big + big>
        repeat(i,0,b.size())a[i]+=b[i];
        a.adjust();
        return a;
    }
    friend big operator-(big a,const big &b){ // <big - big>
        repeat(i,0,b.size())a[i]-=b[i];
        a.adjust();
        return a;
    }
    friend big operator*(const big &a,const big &b){ // <big * big>
        big ans;
        repeat(i,0,a.size()){
            repeat(j,0,b.size())
                ans[i+j]+=a[i]*b[j];
            ans.adjust();
        }
        return ans;
    }
    friend big operator*(big a,int b){ // <big * int>
        repeat(i,0,a.size())a[i]*=b;
        a.adjust();
        return a;
    }
    friend int to_abs(big &a){ // used in divide and mod
        a.final_adjust();
        int t=a.sgn(); a=a*t;
        return t;
    }
    friend big operator/(big a,int b){ // <big / int>
        int f=to_abs(a) * ((b>0)-(b<0));
        b=abs(b);
        ll rem=0;
        repeat_back(i,0,a.size()){
            ll s=rem*a.k+a[i];
            a[i]=s/b;
            rem=s%b;
        }
        return a*f;
    }
    friend big div(big &a,const big &b){ // used in <big / big>
        if(a<b)return big();
        big ans=div(a,b*2)*2;
        if(!(a<b)){
            a=a-b;
            ans=ans+big(1);
        }
        return ans;
    }
    friend big operator/(big a,big b){ // <big / big>
        int f=to_abs(a)*to_abs(b);
        return div(a,b)*f;
    }
    friend ll operator%(const big &a,ll mod){ // <big % ll>
        ll ans=0,p=1; mod=abs(mod);
        repeat(i,0,a.size()){
            (ans+=p*a[i])%=mod;
            (p*=a.k)%=mod;
        }
        return (ans+mod)%mod;
    }
    friend big operator%(big a,big b){ // <big % big>
        to_abs(b);
        int f=to_abs(a);
        return (a-a/b*b)*f;
    }
    friend bool operator<(big a,big b){ // <big less than big>
        a.final_adjust();
        b.final_adjust();
        repeat_back(i,0,max(a.size(),b.size()))
            if(a[i]!=b[i])return a[i]<b[i];
        return 0;
    }
    friend bool operator==(big a,big b){ // <big == big>
        a.final_adjust();
        b.final_adjust();
        repeat_back(i,0,max(a.size(),b.size()))
            if(a[i]!=b[i])return 0;
        return 1;
    }
};
```

### 7.3. struct of 分数

- （可以直接哈希）（避免0/0，会当成0/1处理）

```cpp
struct frac {
    ll u, d;
    explicit frac(ll u = 0, ll d = 1) : u(u), d(d) { init(); }
    void init() {
        if (d < 0) {
            u = -u, d = -d;
        }
        if (u == 0) { // 0
            d = 1;
        } else if (d == 0) { // 无穷大
            u = 1;
        } else {
            ll x = abs(__gcd(u, d));
            u /= x;
            d /= x;
        }
    }
    frac operator-() const { return frac(-u, d); }
    friend frac operator+(const frac &a, const frac &b) {
        return frac(a.u * b.d + a.d * b.u, a.d * b.d);
    }
    friend frac operator-(const frac &a, const frac &b) { return a + -b; }
    friend frac operator*(const frac &a, const frac &b) {
        return frac(a.u * b.u, a.d * b.d);
    }
    friend frac operator/(const frac &a, const frac &b) {
        return frac(a.u * b.d, a.d * b.u);
    }
    friend ostream &operator<<(ostream &cout, const frac &f) {
        return cout << f.u << '/' << f.d;
    }
    bool operator<(const frac &b) const { return u * b.d < d * b.u; }
    bool operator==(const frac &b) const { return u == b.u && d == b.d; }
};
```

### 7.4. 表达式求值

```cpp
inline int lvl(const string &c){ // 运算优先级，小括号要排最后
    if(c=="*")return 2;
    if(c=="(" || c==")")return 0;
    return 1;
}
string convert(const string &in) { // 中缀转后缀
    stringstream ss;
    stack<string> op;
    string ans,s;
    repeat(i,0,in.size()-1){
        ss<<in[i];
        if(!isdigit(in[i]) || !isdigit(in[i+1])) // 插入空格
            ss<<" ";
    }
    ss<<in.back();
    while(ss>>s){
        if(isdigit(s[0]))ans+=s+" ";
        else if(s=="(")op.push(s);
        else if(s==")"){
            while(!op.empty() && op.top()!="(")
                ans+=op.top()+" ",op.pop();
            op.pop();
        }
        else{
            while(!op.empty() && lvl(op.top())>=lvl(s))
                ans+=op.top()+" ",op.pop();
            op.push(s);
        }
    }
    while(!op.empty())ans+=op.top()+" ",op.pop();
    return ans;
}
ll calc(const string &in){ // 后缀求值
    stack<ll> num;
    stringstream ss;
    ss<<in;
    string s;
    while(ss>>s){
        char c=s[0];
        if(isdigit(c))
            num.push((stoll(s))%mod);
        else{
            ll b=num.top(); num.pop();
            ll a=num.top(); num.pop();
            if(c=='+')num.push((a+b)%mod);
            if(c=='-')num.push((a-b)%mod);
            if(c=='*')num.push((a*b)%mod);
            // if(c=='^')num.push(qpow(a,b));
        }
    }
    return num.top();
}
```

### 7.5. 数值积分 using 自适应辛普森法

- 求 $\displaystyle \int_{l}^{r}f(x)\mathrm{d}x$ 的近似值

```cpp
lf raw(lf l,lf r){ // 辛普森公式
    return (f(l)+f(r)+4*f((l+r)/2))*(r-l)/6;
}
lf asr(lf l,lf r,lf eps,lf ans){
    lf m=(l+r)/2;
    lf x=raw(l,m),y=raw(m,r);
    if(abs(x+y-ans)<=15*eps)
        return x+y-(x+y-ans)/15;
    return asr(l,m,eps/2,x)+asr(m,r,eps/2,y);
}
// 调用方法：asr(l,r,eps,raw(l,r))
```
