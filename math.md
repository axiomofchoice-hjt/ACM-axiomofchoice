<!-- TOC -->

- [数学](#数学)
	- [数论](#数论)
		- [基本操作](#基本操作)
			- [模乘 模幂 模逆 扩欧](#模乘-模幂-模逆-扩欧)
			- [阶乘 组合数](#阶乘-组合数)
			- [防爆模乘](#防爆模乘)
			- [最大公约数](#最大公约数)
		- [高级模操作](#高级模操作)
			- [同余方程组 | CRT+extra](#同余方程组--crtextra)
			- [离散对数 | BSGS+extra](#离散对数--bsgsextra)
			- [阶与原根](#阶与原根)
			- [N次剩余](#n次剩余)
		- [数论函数的生成](#数论函数的生成)
			- [单个欧拉函数](#单个欧拉函数)
			- [线性递推乘法逆元](#线性递推乘法逆元)
			- [线性筛](#线性筛)
				- [筛素数](#筛素数)
				- [筛欧拉函数](#筛欧拉函数)
				- [筛莫比乌斯函数](#筛莫比乌斯函数)
				- [筛其他函数](#筛其他函数)
			- [min_25筛](#min_25筛)
		- [素数约数相关](#素数约数相关)
			- [唯一分解 质因数分解](#唯一分解-质因数分解)
			- [素数判定 | 朴素 or Miller-Rabin](#素数判定--朴素-or-miller-rabin)
			- [大数分解 | Pollard-rho](#大数分解--pollard-rho)
			- [求约数](#求约数)
			- [反素数生成](#反素数生成)
		- [数论杂项](#数论杂项)
			- [数论分块](#数论分块)
			- [高斯整数](#高斯整数)
			- [二次剩余](#二次剩余)
			- [莫比乌斯反演](#莫比乌斯反演)
			- [杜教筛](#杜教筛)
			- [斐波那契数列](#斐波那契数列)
			- [佩尔方程×Pell](#佩尔方程pell)
	- [组合数学](#组合数学)
		- [组合数取模 | Lucas+extra](#组合数取模--lucasextra)
		- [康托展开+逆 编码与解码](#康托展开逆-编码与解码)
		- [置换群计数](#置换群计数)
		- [组合数学的一些结论](#组合数学的一些结论)
	- [博弈论](#博弈论)
		- [SG函数 SG定理](#sg函数-sg定理)
		- [Nim游戏](#nim游戏)
		- [删边游戏×Green Hachenbush](#删边游戏green-hachenbush)
		- [翻硬币游戏](#翻硬币游戏)
		- [高维组合游戏 | Nim积](#高维组合游戏--nim积)
		- [不平等博弈 | 超现实数](#不平等博弈--超现实数)
		- [其他博弈结论](#其他博弈结论)
	- [代数结构](#代数结构)
		- [置换群](#置换群)
		- [多项式](#多项式)
			- [拉格朗日插值](#拉格朗日插值)
			- [多项式基本操作](#多项式基本操作)
			- [快速傅里叶变换×FTT+任意模数](#快速傅里叶变换ftt任意模数)
			- [快速数论变换×NTT](#快速数论变换ntt)
			- [快速沃尔什变换×FWT](#快速沃尔什变换fwt)
			- [多项式运算](#多项式运算)
			- [多项式的一些结论及生成函数](#多项式的一些结论及生成函数)
				- [普通生成函数×OGF](#普通生成函数ogf)
				- [指数生成函数×EGF](#指数生成函数egf)
		- [矩阵](#矩阵)
			- [矩阵乘法 矩阵快速幂](#矩阵乘法-矩阵快速幂)
			- [矩阵高级操作](#矩阵高级操作)
			- [异或方程组](#异或方程组)
			- [线性基](#线性基)
			- [线性规划 | 单纯形法](#线性规划--单纯形法)
			- [矩阵的一些结论](#矩阵的一些结论)
	- [数学杂项](#数学杂项)
		- [主定理](#主定理)
		- [质数表](#质数表)
		- [struct of 自动取模](#struct-of-自动取模)
		- [struct of 高精度](#struct-of-高精度)
		- [表达式求值](#表达式求值)
		- [一些数学结论](#一些数学结论)
			- [约瑟夫问题](#约瑟夫问题)
			- [格雷码×gray 汉诺塔](#格雷码gray-汉诺塔)
			- [Stern-Brocot树 Farey序列](#stern-brocot树-farey序列)
			- [浮点与近似计算](#浮点与近似计算)
			- [others of 数学杂项](#others-of-数学杂项)

<!-- /TOC -->

# 数学

## 数论

### 基本操作

#### 模乘 模幂 模逆 扩欧

```c++
ll mul(ll a,ll b,ll m=mod){return a*b%m;} //模乘
ll qpow(ll a,ll b,ll m=mod){ //快速幂
	ll ans=1;
	for(;b;a=mul(a,a,m),b>>=1)
		if(b&1)ans=mul(ans,a,m);
	return ans;
}
void exgcd(ll a,ll b,ll &d,ll &x,ll &y){ //ax+by=gcd(a,b), d=gcd
	if(!b)d=a,x=1,y=0;
	else exgcd(b,a%b,d,y,x),y-=x*(a/b);
}
ll gcdinv(ll v,ll m=mod){ //扩欧版逆元
	ll d,x,y;
	exgcd(v,m,d,x,y);
	return (x%m+m)%m;
}
ll getinv(ll v,ll m=mod){ //快速幂版逆元，m必须是质数!!
	return qpow(v,m-2,m);
}
ll qpows(ll a,ll b,ll m=mod){
	if(b>=0)return qpow(a,b,m);
	else return getinv(qpow(a,-b,m),m);
}
```

#### 阶乘 组合数

- $O(n)$ 初始化，$O(1)$ 查询

```c++
struct CC{
	static const int N=100010;
	ll fac[N],inv[N];
	CC(){
		fac[0]=1;
		repeat(i,1,N)fac[i]=fac[i-1]*i%mod;
		inv[N-1]=qpow(fac[N-1],mod-2,mod);
		repeat_back(i,1,N)inv[i-1]=inv[i]*i%mod;
	}
	ll operator()(ll a,ll b){ //a>=b
		if(a<b || b<0)return 0;
		return fac[a]*inv[a-b]%mod*inv[b]%mod;
	}
	ll A(ll a,ll b){ //a>=b
		if(a<b || b<0)return 0;
		return fac[a]*inv[a-b]%mod;
	}
}C;
```

#### 防爆模乘

```c++
//int128版本
ll mul(ll a,ll b,ll m=mod){return (__int128)a*b%m;}
//long double版本（欲防爆，先自爆）
ll mul(ll a,ll b,ll m){
	ll c=a*b-(ll)((long double)a*b/m+0.5)*m;
	return c<0?c+m:c;
}
//每位运算一次版本，注意这是真·龟速乘，O(logn)
ll mul(ll a,ll b,ll m=mod){
	ll ans=0;
	while(b){
		if(b&1)ans=(ans+a)%m;
		a=(a+a)%m;
		b>>=1;
	}
	return ans;
}
//把b分成两部分版本，要保证m小于1<<42（约等于4e12），a,b<m
ll mul(ll a,ll b,ll m=mod){
	a%=m,b%=m;
	ll l=a*(b>>21)%m*(1ll<<21)%m;
	ll r=a*(b&(1ll<<21)-1)%m;
	return (l+r)%m;
}
```

#### 最大公约数

```c++
__gcd(a,b) //内置gcd，推荐
ll gcd(ll a,ll b){return b==0?a:gcd(b,a%b);} //不推荐233，比内置gcd慢
ll gcd(ll a,ll b){ //卡常gcd来了！！
	#define tz __builtin_ctzll
	if(!a || !b)return a|b;
	int t=tz(a|b);
	a>>=tz(a);
	while(b){
		b>>=tz(b);
		if(a>b)swap(a,b);
		b-=a;
	}
	return a<<t;
	#undef tz
}
```

- 实数gcd

```c++
lf fgcd(lf a,lf b){return abs(b)<1e-5?a:fgcd(b,fmod(a,b));}
```

### 高级模操作

#### 同余方程组 | CRT+extra

```c++
//CRT，m[i]两两互质
ll crt(ll a[],ll m[],int n){ //ans%m[i]==a[i]
	repeat(i,0,n)a[i]%=m[i];
	ll M=1,ans=0;
	repeat(i,0,n)
		M*=m[i];
	repeat(i,0,n){
		ll k=M/m[i],t=gcdinv(k%m[i],m[i]); //扩欧!!
		ans=(ans+a[i]*k*t)%M; //两个乘号可能都要mul
	}
	return (ans+M)%M;
}
//exCRT，m[i]不需要两两互质，基于扩欧exgcd和龟速乘mul
ll excrt(ll a[],ll m[],int n){ //ans%m[i]==a[i]
	repeat(i,0,n)a[i]%=m[i]; //根据情况做适当修改
	ll M=m[0],ans=a[0],g,x,y; //M是m[0..i]的最小公倍数
	repeat(i,1,n){
		ll c=((a[i]-ans)%m[i]+m[i])%m[i];
		exgcd(M,m[i],g,x,y); //Ax=c(mod B)
		if(c%g)return -1;
		ans+=mul(x,c/g,m[i]/g)*M; //龟速乘
		M*=m[i]/g;
		ans=(ans%M+M)%M;
	}
	return (ans+M)%M;
}
```

#### 离散对数 | BSGS+extra

- 求 $a^x \equiv b \pmod m$ ，$O(\sqrt m)$

```c++
//BSGS，a和mod互质
ll bsgs(ll a,ll b,ll mod){ //a^ans%mod==b
	a%=mod,b%=mod;
	static unordered_map<ll,ll> m; m.clear();
	ll t=(ll)sqrt(mod)+1,p=1;
	repeat(i,0,t){
		m[mul(b,p,mod)]=i; //p==a^i
		p=mul(p,a,mod);
	}
	a=p; p=1;
	repeat(i,0,t+1){
		if(m.count(p)){ //p==a^i
			ll ans=t*i-m[p];
			if(ans>0)return ans;
		}
		p=mul(p,a,mod);
	}
	return -1;
}
//exBSGS，a和mod不需要互质，基于BSGS
ll exbsgs(ll a,ll b,ll mod){ //a^ans%mod==b
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
	ll t=bsgs(a,mul(b,getinv(c,mod),mod),mod); //必须扩欧逆元!!
	if(t==-1)return -1;
	return t+ans;
}
```

#### 阶与原根

- 判断是否有原根：若 $m$ 有原根，则 $m$ 一定是下列形式：$2,4,p^a,2p^a$（ $p$ 是奇素数， $a$ 是正整数）
- 求所有原根：若 $g$ 为 $m$ 的一个原根，则 $g^s\space(1\le s\le\varphi(m),\gcd(s,\varphi(m))=1)$ 给出了 $m$ 的所有原根。因此若 $m$ 有原根，则 $m$ 有 $\varphi(\varphi(m))$ 个原根
- 求一个原根，$O(n\log\log n)$ ~~实际远远不到~~

```c++
ll getG(ll n){ //求n最小的原根
	static vector<ll> a; a.clear();
	ll k=n-1;
	repeat(i,2,sqrt(k+1)+1)
	if(k%i==0){
		a.push_back(i); //a存放(n-1)的质因数
		while(k%i==0)k/=i;
	}
	if(k!=1)a.push_back(k);
	repeat(i,2,n){ //枚举答案
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

#### N次剩余

- 求 $x^a \equiv b \pmod m$ ，基于BSGS、原根

```c++
//只求一个
ll residue(ll a,ll b,ll mod){ //ans^a%mod==b
	ll g=getG(mod),c=bsgs(qpow(g,a,mod),b,mod);
	if(c==-1)return -1;
	return qpow(g,c,mod);
}
//求所有N次剩余
vector<ll> ans;
void allresidue(ll a,ll b,ll mod){ //ans^a%mod==b
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

### 数论函数的生成

#### 单个欧拉函数

- $\varphi(n)=$ 小于 `n` 且与 `n` 互质的正整数个数
- 令 `n` 的唯一分解式 $n=Π({p_k}^{a_k})$，则 $\varphi(n)=n\cdot Π(1-\dfrac 1 {p_k})$
- $O(\sqrt n)$

```c++
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

#### 线性递推乘法逆元

求1..(n-1)的逆元，$O(n)$

```c++
void get_inv(int n,int m=mod){
	inv[1]=1;
	repeat(i,2,n)inv[i]=m-m/i*inv[m%i]%m;
}
```

求a[1..n]的逆元，离线，$O(n)$

```c++
void get_inv(int a[],int n){ //求a[1..n]的逆元，存在inv[1..n]中
	static int pre[N];
	pre[0]=1;
	repeat(i,1,n+1)
		pre[i]=(ll)pre[i-1]*a[i]%mod;
	int inv_pre=qpow(pre[n],mod-2);
	repeat_back(i,1,n+1){
		inv[i]=(ll)pre[i-1]*inv_pre%mod;
		inv_pre=(ll)inv_pre*a[i]%mod;
	}
}
```

#### 线性筛

- 定理：求出 $f(p)$（$p$ 为质数）的复杂度不超过 $O(\log p)$ 的积性函数可以被线性筛

##### 筛素数

- `a[i]` 表示第 $i+1$ 个质数，`vis[i]==0` 表示 $i$ 是素数，`rec[i]` 为 $i$ 的最小质因数
- $O(n)$

```c++
bool vis[N]; int rec[N]; vector<int> a;
void get_prime(){
	vis[1]=1;
	repeat(i,2,N){
		if(!vis[i])a.push_back(i),rec[i]=i;
		for(auto j:a){
			if(i*j>=N)break;
			vis[i*j]=1; rec[i*j]=j;
			if(i%j==0)break;
		}
	}
}
```

##### 筛欧拉函数

- 线性版，$O(n)$

```c++
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

```c++
void get_phi(){
	phi[1]=1; //其他的值初始化为0
	repeat(i,2,N)if(!phi[i])
	for(int j=i;j<N;j+=i){
		if(!phi[j])phi[j]=j;
		phi[j]=phi[j]/i*(i-1);
	}
}
```

##### 筛莫比乌斯函数

- $O(n)$

```c++
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

##### 筛其他函数

- 筛约数个数

```c++
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

- 筛gcd

```c++
int gcd[N][N];
void get_gcd(int n,int m){
	repeat(i,1,n+1)
	repeat(j,1,m+1)
	if(!gcd[i][j])
	repeat(k,1,min(n/i,m/j)+1)
		gcd[k*i][k*j]=k;
}
```

#### min_25筛

- 求 $[1,n]$ 内的素数个数

```c++
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

- 求 $[1,n]$ 内的素数之和

```c++
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

### 素数约数相关

#### 唯一分解 质因数分解

- 用数组表示数字唯一分解式的素数的指数，如 $50=\{1,0,2,0,…\}$
- 可以用来计算阶乘和乘除操作

```c++
void fac(int a[],ll n){
	repeat(i,2,(int)sqrt(n)+2)
	while(n%i==0)a[i]++,n/=i;
	if(n>1)a[n]++;
}
```

- set维护版

```c++
struct fac{
	#define facN 1010
	ll a[facN]; set<ll> s; //乘法就是multiset
	fac(){mst(a,0); s.clear();}
	void lcm(ll n){ //self=lcm(self,n)
		repeat(i,2,facN)
		if(n%i==0){
			ll cnt=0;
			while(n%i==0)cnt++,n/=i;
			a[i]=max(a[i],cnt); //改成a[i]+=cnt就变成了乘法
		}
		if(n>1)s.insert(n);
	}
	ll value(){ //return self%mod
		ll ans=1;
		repeat(i,2,facN)
			if(a[i])ans=ans*qpow(i,a[i],mod)%mod;
		for(auto i:s)ans=ans*i%mod;
		return ans;
	}
}f;
```

#### 素数判定 | 朴素 or Miller-Rabin

- 朴素算法，$O(\sqrt n)$

```c++
bool isprime(int n){
	if(n<=3)return n>=2;
	if(n%2==0 || n%3==0)return 0;
	repeat(i,1,int(sqrt(n)+1.5)/6+1)
		if(n%(i*6-1)==0 || n%(i*6+1)==0)return 0;
	return 1;
}
```

- Miller-Rabin素性测试，$O(\cdot\log^3 n)$

```c++
bool mr(ll x,ll b){
    ll k=x-1;
    while(k){
        ll cur=qpow(b,k,x);
        if(cur!=1 && cur!=x-1)return 0;
        if(k%2==1 || cur==x-1)return 1;
        k>>=1;
    }
    return 1;
}
bool isprime(ll x){
    if(x<2 || x==46856248255981ll)
	    return 0;
    if(x<4 || x==61) // raw: if(x==2 || x==3 || x==7 || x==61 || x==24251)
        return 1;
    return mr(x,2) && mr(x,61);
}
```

#### 大数分解 | Pollard-rho

- $O(n^{\tfrac 1 4})$，基于MR素性测试

```c++
ll pollard_rho(ll x){
    ll s=0,t=0,c=rnd()%(x-1)+1;
    int stp=0,goal=1; ll val=1;
    for(goal=1;;goal<<=1,s=t,val=1){
        for(stp=1;stp<=goal;++stp){
            t=((__int128)t*t+c)%x;
            val=(__int128)val*abs(t-s)%x;
            if(stp%127==0){
                ll d=__gcd(val,x);
                if(d>1)return d;
            }
        }
        ll d=__gcd(val,x);
        if(d>1)return d;
    }
}
vector<ll> ans; //result
void rho(ll n){
	if(isprime(n)){
		ans.push_back(n);
		return;
	}
	ll t;
	do{t=pollard_rho(n);}while(t>=n);
	rho(t);
	rho(n/t);
}
```

#### 求约数

```c++
int get_divisor(int n){
	int ans=0;
	for(int i=1;i<n;i=n/(n/(i+1)))
	if(n%i==0)
		ans++; //v.push_back(i);
	return ans+1; //v.push_back(n);
}
```

- 卡常(n<=1e7)，基于线性筛
- 下面那个可以处理出所有约数的质因数个数

```c++
vector<pii> pd; vector<ll> v;
void dfs(int x,int y,ll s){
	if(x==(int)pd.size()){v.push_back(s); return;}
	dfs(x+1,0,s);
	if(y<pd[x].se)dfs(x,y+1,s*pd[x].fi);
}
void get_divisor(ll n){
	pd.clear(); v.clear();
	while(n!=1){
		if(!pd.empty() && pd.back().fi==rec[n])pd.back().se++;
		else pd.push_back({rec[n],1});
		n/=rec[n]; //needs initialized
	}
	dfs(0,0,1);
}
/*
vector<pii> pd; vector<pii> v;
void dfs(int x,int y,ll s,int cnt){
	if(x==(int)pd.size()){v.push_back({s,cnt}); return;}
	dfs(x+1,0,s,cnt);
	if(y<pd[x].se)dfs(x,y+1,s*pd[x].fi,cnt+1);
}
void get_divisor(ll n){
	pd.clear(); v.clear();
	for(ll i=2;i*i<n;i++){
		ll cnt=0;
		while(n%i==0)n/=i,cnt++;
		if(cnt)pd.push_back({i,cnt});
	}
	if(n>1)pd.push_back({n,1});
	dfs(0,0,1,0);
}
*/
```

#### 反素数生成

- 求因数最多的数（因数个数一样则取最小）
- 性质：$M = {p_1}^{k_1}{p_2}^{k_2}...$ 其中，$p_i$ 是从 $2$ 开始的连续质数，$k_i-k_{i+1}∈\{0,1\}$
- 先打出质数表再 $dfs$，枚举 $k_n$，$O(\exp)$

```c++
int pri[16]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
ll n; //范围
pair<ll,ll> ans; //ans是结果，ans.fi是最大反素数，ans.se是反素数约数个数
void dfs(ll num=1,ll cnt=1,int *p=pri,int pre=inf){ //注意ans要初始化
	if(make_pair(cnt,-num)>make_pair(ans.se,-ans.fi))
		ans={num,cnt};
	num*=*p;
	for(int i=1;i<=pre && num<=n;i++,num*=*p)
		dfs(num,p+1,i,cnt*(i+1));
}
```

- $n$ 以内约数个数最大值是 $O(n^{\tfrac {1.066}{\ln\ln n}})$

| 范围 | 1e4 | 1e5 | 1e6 | 1e9 | 1e16 |
| :------------: | :--: | :---: | :----: | :-------: | :--------------: |
| 最大反素数 | 7560 | 83160 | 720720 | 735134400 | 8086598962041600 |
| 反素数约数个数 | 64 | 128 | 240 | 1344 | 41472 |

### 数论杂项

#### 数论分块

- $n=(k-n\%k)(n/k)+(n\%k)(n/k+1)$
- 将 $\lfloor \dfrac{n}{x}\rfloor=C$ 的 $[x_{\min},x_{\max}]$ 作为一块，其中区间内的任一整数 $x_0$ 满足 $x_{\max}=n/(n/x_0)$

```c++
for(int l=l0,r;l<=r0;l=r+1){
	r=min(r0,n/(n/l));
	//c=n/l;
	//len=r-l+1;
}
```

- 将 $\lceil \dfrac{n}{x}\rceil=C$ 的 $[x_{\min},x_{\max}]$ 作为一块：

```c++
for(int l=l0,r;l<=r0;l=r+1){
	r=min(r0,n/(n/l)); if(n%r==0)r=max(r-1,l);
	//c=(n+l-1)/l;
	//len=l-r+1;
}
```

#### 高斯整数

- 高斯整数：$\{a+bi\ |\ a,b∈\Z\}$
- 高斯素数：无法分解为两个高斯整数 $\not∈\{\pm1,\pm i\}$ 之积的高斯整数
- $a+bi$ 是高斯素数当前仅当
	- $a,b$ 一个为 $0$，另一个绝对值为 $4k+3$ 型素数
	- $a^2+b^2$ 为 $4k+1$ 型素数或 $2$
- 带余除法

```c++
vec operator/(vec a,vec b){
	double x=b.x*b.x+b.y*b.y;
	return {llround((a.x*b.x+a.y*b.y)/x),llround((a.y*b.x-a.x*b.y)/x)};
}
vec operator%(vec a,vec b){return a-a/b*b;}
vec gcd(vec a,vec b){
	while(b.x || b.y)a=a%b,swap(a,b);
	return a;
}
```

#### 二次剩余

- 对于奇素数模数 $p$，存在 $\frac {p-1} 2$ 个二次剩余 $\{1^2,2^2,...,(\frac {p-1} 2)^2\}$，和相同数量的二次非剩余
- 对于奇素数模数 $p$，如果 $n^{\frac{p-1}2}\equiv1\pmod{p}$ ，则 $n$ 是一个二次剩余；如果 $n^{\frac{p-1}2}\equiv-1\pmod{p}$，则 $n$ 是一个二次非剩余
- 对于奇素数模数 $p$，二次剩余的乘积是二次剩余，二次剩余与非二次剩余乘积为非二次剩余，非二次剩余乘积是二次剩余
- 费马-欧拉素数定理：$(4n+1)$ 型素数只能用一种方法表示为一个范数（两个完全平方数之和），$(4n+3)$ 型素数不能表示为一个范数
- 二次互反律：记 $p^{\frac{q-1}2}$ 的符号为 $(\dfrac p q)$ ，则对奇素数 $p,q$ 有 $(\dfrac p q)\cdot(\dfrac q p)=(-1)^{\tfrac{p-1}2\cdot\tfrac{q-1}2}$
- 求二次剩余，要求 mod 是质数，sqrtmod() 返回其中一个 sqrt，另一个为 mod 减这个返回值（如果 mod=2 就没有第二个）；返回 $-1$ 表示无解

```c++
ll w;
struct vec{ //x+y*sqrt(w)
	ll x,y;
	vec operator*(vec b){
		return {(x*b.x+y*b.y%mod*w)%mod,(x*b.y+y*b.x)%mod};
	}
};
vec qpow(vec a,ll b){
	vec ans={1,0};
	for(;b;a=a*a,b>>=1)
		if(b&1)ans=ans*a;
	return ans;
}
ll Leg(ll a){return qpow(a,(mod-1)>>1,mod)!=mod-1;}
ll sqrtmod(ll b){
	if(mod==2)return 1;
	if(!Leg(b))return -1;
	ll a;
	do{a=rnd()%mod; w=((a*a-b)%mod+mod)%mod;}while(Leg(w));
	return qpow({a,1},(mod+1)>>1).x;
}
```

#### 莫比乌斯反演

- 引理1：$\lfloor \dfrac{a}{bc}\rfloor=\lfloor \dfrac{\lfloor \dfrac{a}{b}\rfloor}{c}\rfloor$；引理2：$n$ 的因数个数 $≤\lfloor 2\sqrt n \rfloor$
- 狄利克雷卷积：$(f*g)(n)=\sum\limits_{d|n}f(d)g(\dfrac n d)$，有交换律、结合律、对加法的分配律
- 积性函数：$\{f(n)|\gcd(n,m)=1\Rightarrow f(nm)=f(n)f(m)\}$
- 单位函数：$\varepsilon(n)=[n=1]$ 为狄利克雷卷积的单位元
- 恒等函数：$id(n)=n$
- 约数个数：$d(n)=1*1$
- 约数之和：$\sigma(n)=1*id$
- 莫比乌斯函数性质：$\mu(n)=\begin{cases} 1&n=1\\0&n含有平方因子\\(-1)^k&k为n的质因数个数\end{cases}$
- 结论：$(\forall f)(f*\varepsilon=f),\mu*1=\varepsilon,\varphi*1=id,d*\mu=id$
- 莫比乌斯反演：若$f=g*1$，则$g=f*\mu$；或者，若$f(n)=\sum\limits_{d|n}g(d)$，则$g(n)=\sum\limits_{d|n}\mu(d)f(\dfrac n d)$

***

- 例题：求模意义下的 $\sum\limits_{i=1}^n \sum\limits_{j=1}^m \dfrac{i\cdot j}{\gcd(i,j)}$
- $ans=\sum\limits_{i=1}^n\sum\limits_{j=1}^m\sum\limits_{d|i,d|j,\gcd(\frac i d,\frac j d)=1}\dfrac{i\cdot j}d$
- 非常经典的化法：
- $ans=\sum\limits_{d=1}^n d\cdot\sum\limits_{i=1}^{\lfloor\frac nd\rfloor}\sum\limits_{j=1}^{\lfloor\frac md\rfloor}[\gcd(i,j)=1]i\cdot j$
- 设 $sum(n,m)=\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m}[\gcd(i,j)=1]i\cdot j$
- $sum(n,m)=\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m}\sum\limits_{c|i,c|j}{\mu(c)}\cdot i\cdot j$
- 设 $i'=\dfrac i c,j'=\dfrac j c$
- $sum(n,m)=\sum\limits_{c=1}^n\mu(c)\cdot c^2\cdot\sum\limits_{i'=1}^{\lfloor\frac nc\rfloor}\sum\limits_{j'=1}^{\lfloor\frac mc\rfloor} i'\cdot j'$
- 易得 $\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m} i\cdot j=\dfrac 1 4 n(n+1) m(m+1)$

#### 杜教筛

- $g(1)S(n)=\sum\limits_{i=1}^n(f*g)(i)-\sum\limits_{i=2}^n g(i)S(\lfloor\dfrac n i \rfloor),S(n)=\sum\limits_{i=1}^nf(i)$
- 如果能找到合适的 $g(n)$，能快速计算 $\sum\limits_{i=1}^n(f*g)(i)$，就能快速计算 $S(n)$
- $f(n)=\mu(n),g(n)=1,(f*g)(n)=[n=1]$
- $f(n)=\varphi(n),g(n)=1,(f*g)(n)=n$
- $f(n)=n\cdot\varphi(n),g(n)=n,(f*g)(n)=n^2$
- $f(n)=d(n),g(n)=\mu(n),(f*g)(n)=1$
- $f(n)=\sigma(n),g(n)=\mu(n),(f*g)(n)=n$（但是用公式 $\sum\limits_{i=1}^{n}\sigma(n)=\sum\limits_{i=1}^{n}i\cdot\lfloor\dfrac n i\rfloor$ 更好）
- $O(n^{\tfrac 2 3})$，注意有递归的操作就要记忆化

```c++
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

#### 斐波那契数列

- 定义：$F_0=0,F_1=1,F_n=F_{n-1}+F_{n-2}$
- $F_n=\dfrac 1 {\sqrt{5}} [(\dfrac{1+\sqrt 5}2)^n-(\dfrac{1-\sqrt 5}2)^n)]$ （公式中若 $5$ 是二次剩余则可以化简，比如 $\sqrt 5\equiv 383008016\pmod {1000000009}$）
- $F_{a+b-1}=F_{a-1}F_{b-1}+F_aF_b$ （重要公式）
- $F_{n-1}F_{n+1}-F_n^2=(-1)^n$ （卡西尼性质）
- $F_{n}^2+F_{n+1}^2=F_{2n+1}$
- $F_{n+1}^2-F_{n-1}^2=F_{2n}$ （由上一条写两遍相减得到）
- $F_1+F_3+F_5+...+F_{2n-1}=F_{2n}$ （奇数项求和）
- $F_2+F_4+F_6+...+F_{2n}=F_{2n+1}-1$ （偶数项求和）
- $F_1^2+F_2^2+F_3^2+...+F_n^2=F_nF_{n+1}$
- $F_1+2F_2+3F_3+...+nF_n=nF_{n+2}-F_{n+3}+2$
- $-F_1+F_2-F_3+...+(-1)^nF_n=(-1)^n(F_{n+1}-F_n)+1$
- $F_{2n-2m-2}(F_{2n}+F_{2n+2})=F_{2m+2}+F_{4n-2m}$
- $F_a \mid F_b \Leftrightarrow a \mid b$
- $\gcd(F_a,F_b)=F_{\gcd(a,b)}$
- 当 $p$ 为 $5k\pm 1$ 型素数时，$\begin{cases} F_{p-1}\equiv 0\pmod p \\ F_p\equiv 1\pmod p \\ F_{p+1}\equiv 1\pmod p \end{cases}$
- 当 $p$ 为 $5k\pm 2$ 型素数时，$\begin{cases} F_{p-1}\equiv 1\pmod p \\ F_p\equiv -1\pmod p \\ F_{p+1}\equiv 0\pmod p \end{cases}$
- $F_{n+2}$ 为集合 `{1,2,3,...,n-2}` 中不包含相邻正整数的子集个数（包括空集）
- `F(n)%m` 的周期 $\le 6m$（$m=2\times 5^k$ 取等号）
- 齐肯多夫定理：任何正整数都可以表示成若干个不连续的斐波那契数（$F_2$ 开始）可以用贪心实现
- $a_0=1,a_n=a_{n-1}+a_{n-3}+a_{n-5}+...(n\ge 1)$，则 $a_n=F_n(n\ge 1)$

快速倍增法求$F_n$，返回二元组$(F_n,F_{n+1})$ ，$O(\log n)$

```c++
pii fib(ll n){ //fib(n).fi即结果
	if(n==0)return {0,1};
	pii p=fib(n>>1);
	ll a=p.fi,b=p.se;
	ll c=a*(2*b-a)%mod;
	ll d=(a*a+b*b)%mod;
	if(n&1)return {d,(c+d)%mod};
	else return {c,d};
}
```

#### 佩尔方程×Pell

- $x^2-dy^2=1$，$d$ 是正整数
- 若 $d$ 是完全平方数，只有平凡解 $(\pm 1,0)$，其余情况总有非平凡解
- 若最小正整数解 $(x_1,y_1)$，则递推公式
- $\begin{cases}x_n=x_1x_{n-1}+dy_1y_{n-1}\\y_n=y_1x_{n-1}+x_1y_{n-1}\end{cases}$
- $\left[\begin{array}{c}x_n\\y_n\end{array}\right]=\left[\begin{array}{cc}x_1 & dy_1\\y_1 & x_1\end{array}\right]\left[\begin{array}{c}x_{n-1}\\y_{n-1}\end{array}\right]$
- 最小解（可能溢出）

```c++
bool PQA(ll D, ll &x, ll &y){
	ll d=llround(sqrt(D));
	if(d*d==D)return 0;
	ll u=0,v=1,a=int(sqrt(D)),a0=a,lastx=1,lasty=0;
	x=a,y=1;
	do{
		u=a*v-u; v=(D-u*u)/v;
		a=(a0+u)/v;
		ll thisx=x,thisy=y;
		x=a*x+lastx; y=a*y+lasty;
		lastx=thisx; lasty=thisy;
	}while(v!=1 &&a<=a0);
	x=lastx; y=lasty;
	if(x*x-D*y*y==-1){
		x=lastx*lastx+D*lasty*lasty;
		y=2*lastx*lasty;
	}
	return 1;
}
```

## 组合数学

### 组合数取模 | Lucas+extra

- Lucas定理用来求模意义下的组合数
- 真·Lucas，$p$ 是质数（~~后面的exLucas都不纯正~~）

```c++
ll lucas(ll a,ll b,ll p){ //a>=b
	if(b==0)return 1;
	return mul(C(a%p,b%p,p),lucas(a/p,b/p,p),p);
}
```

- 特例：如果p=2，可能lucas失效（？）

```c++
ll C(ll a,ll b){ //a>=b，p=2的情况
	return (a&b)==b;
}
```

- 快速阶乘和exLucas
- $qfac.A(x),qfac.B(x)$ 满足 $A\equiv \dfrac{x!}{p^B}\pmod {p^k}$
- $qfac.C(a,b)\equiv C_a^b \pmod {p^k}$
- $exlucas(a,b,m)\equiv C_a^b \pmod m$，函数内嵌中国剩余定理

```c++
struct Qfac{
	ll s[2000010];
	ll p,m;
	ll A(ll x){ //快速阶乘的A值
		if(x==0)return 1;
		ll c=A(x/p);
		return s[x%m]*qpow(s[m],x/m,m)%m*c%m;
	}
	ll B(ll x){ //快速阶乘的B值
		int ans=0;
		for(ll i=x;i;i/=p)ans+=i/p;
		return ans;
	}
	ll C(ll a,ll b){ //组合数，a>=b
		ll k=B(a)-B(b)-B(a-b);
		return A(a)*gcdinv(A(b),m)%m
			*gcdinv(A(a-b),m)%m
			*qpow(p,k,m)%m;
	}
	void init(ll _p,ll _m){ //一定要满足m=p^k
		p=_p,m=_m;
		s[0]=1;
		repeat(i,1,m+1)
			if(i%p)s[i]=s[i-1]*i%m;
			else s[i]=s[i-1];
	}
}qfac;
ll exlucas(ll a,ll b,ll mod){
	ll ans=0,m=mod;
	for(ll i=2;i<=m;i++) //不能repeat
	if(m%i==0){
		ll p=i,k=1;
		while(m%i==0)m/=i,k*=i;
		qfac.init(p,k);
		ans=(ans+qfac.C(a,b)*(mod/k)%mod*gcdinv(mod/k,k)%mod)%mod;
	}
	return (ans+mod)%mod;
}
```

### 康托展开+逆 编码与解码

<H4>康托展开+逆</H4>

- 康托展开即排列到整数的映射
- 排列里的元素都是从1到n

```c++
//普通版，O(n^2)
int cantor(int a[],int n){
	int f=1,ans=1; //假设答案最小值是1
	repeat_back(i,0,n){
		int cnt=0;
		repeat(j,i+1,n)cnt+=a[j]<a[i];
		ans=(ans+f*cnt%mod)%mod; //ans+=f*cnt;
		f=f*(n-i)%mod; //f*=(n-i);
	}
	return ans;
}
//树状数组优化版，基于树状数组，O(nlogn)
int cantor(int a[],int n){
	static BIT t; t.init(); //树状数组
	ll f=1,ans=1; //假设答案最小值是1
	repeat_back(i,0,n){
		ans=(ans+f*t.sum(a[i])%mod)%mod; //ans+=f*t.sum(a[i]);
		t.add(a[i],1);
		f=f*(n-i)%mod; //f*=(n-i);
	}
	return ans;
}
//逆展开普通版，O(n^2)
int *decantor(int x,int n){
	static int f[13]={1};
	repeat(i,1,13)f[i]=f[i-1]*i;
	static int ans[N];
	set<int> s;
	x--;
	repeat(i,1,n+1)s.insert(i);
	repeat(i,0,n){
		int q=x/f[n-i-1];
		x%=f[n-i-1];
		auto it=s.begin();
		repeat(i,0,q)it++; //第q+1小的数
		ans[i]=*it;
		s.erase(it);
	}
	return ans;
}
```

<H4>编码与解码问题</H4>

<1>

- 给定一个字符串，求出它的编号
- 例，输入acab，输出5（aabc,aacb,abac,abca,acab,...）
- 用递归，令d(S)是小于S的排列数，f(S)是S的全排列数
- 小于acab的第一个字母只能是a，所以d(acab)=d(cab)
- 第二个字母是a,b,c，所以d(acab)=f(bc)+f(ac)+d(ab)
- d(ab)=0
- 因此d(acab)=4，加1之后就是答案

<2>

- 给定编号求字符串，对每一位进行尝试即可

### 置换群计数

Polya定理

- 例：立方体 $n=6$ 个面，每个面染上 $m=3$ 种颜色中的一种
- 两个染色方案相同意味着两个立方体经过旋转可以重合
- 其染色方案数为：$\dfrac{\sum m^{k_i}}{|k|}$（$k_i$ 为某一置换可以拆分的循环置换数，$|k|$ 为所有置换数）

```
不旋转，{U|D|L|R|F|B}，k=6，共1个
对面中心连线为轴的90度旋转，{U|D|L R F B}，k=3，共6个
对面中心连线为轴的180度旋转，{U|D|L R|F B}，k=4，共3个
对棱中点连线为轴的180度旋转，{U L|D R|F B}，k=3，共6个
对顶点连线为轴的120度旋转，{U L F|D R B}，k=2，共8个
```

- 因此 $\dfrac{3^6+3^3 \cdot 6+3^4 \cdot 3+3^3 \cdot 6+3^2 \cdot 8}{1+6+3+6+8}=57$
- 例题（poj1286），n个点连成环，染3种颜色，允许旋转和翻转

```c++
ans=0,cnt=0;
//只考虑旋转，不考虑翻转
repeat(i,1,n+1)
	ans+=qpow(3,__gcd(i,n));
cnt+=n;
//考虑翻转
if(n%2==0)ans+=(qpow(3,n/2+1)+qpow(3,n/2))*(n/2);
else ans+=qpow(3,(n+1)/2)*n;
cnt+=n;
cout<<ans/cnt<<endl;
```

### 组合数学的一些结论

***

- 组合数
- `C(n,k)=(n-k+1)*C(n,k-1)/k​`

```c++
const int N=20;
repeat(i,0,N){
	C[i][0]=C[i][i]=1;
	repeat(j,1,i)C[i][j]=C[i-1][j]+C[i-1][j-1];
}
```

- 二项式反演
	- $\displaystyle f_n=\sum_{i=0}^n{n\choose i}g_i\Leftrightarrow g_n=\sum_{i=0}^n(-1)^{n-i}{n\choose i}f_i$
	- $\displaystyle f_k=\sum_{i=k}^n{i\choose k}g_i\Leftrightarrow g_k=\sum_{i=k}^n(-1)^{i-k}{i\choose k}f_i$
- $\displaystyle \sum_{i=1}^{n}i{n\choose i}=n 2^{n-1}$
- $\displaystyle \sum_{i=1}^{n}i^2{n\choose i}=n(n+1) 2^{n-2}$
- $\displaystyle \sum_{i=1}^{n}\dfrac{1}{i}{n\choose i}=\sum_{i=1}^{n}\dfrac{1}{i}$
- $\displaystyle \sum_{i=0}^{n}{n\choose i}^2={2n\choose n}$

***

- 卡塔兰数×卡特兰数×Catalan，$H_n=\dfrac{\binom{2n}n}{n+1}$，$H_n=\dfrac{H_{n-1}(4n-2)}{n+1}$

***

- 贝尔数×Bell，划分n个元素的集合的方案数

```c++
B[0]=B[1]=1;
repeat(i,2,N){
	B[i]=0;
	repeat(j,0,i)
		B[i]=(B[i]+C(i-1,j)*B[j]%mod)%mod;
}
```

***

- 错排数，$D_n=n![\dfrac 1{0!}-\dfrac 1{1!}+\dfrac 1{2!}-...+\dfrac{(-1)^n}{n!}]$

```c++
D[0]=1;
repeat(i,0,N-1){
	D[i+1]=D[i]+(i&1?C.inv[i+1]:mod-C.inv[i+1]);
	D[i]=1ll*D[i]*fac[i]%mod;
}
```

***

- 第一类斯特林数×Stirling
- 多项式 $x(x-1)(x-2) \cdots (x-n+1)$ 展开后 $x^r$ 的系数绝对值记作 $s(n,r)$ （系数符号 $(-1)^{n+r}$）
- 也可以表示 $n$ 个元素分成 $r$ 个环的方案数
- 递推式 $s(n,r) = (n-1)s(n-1,r)+s(n-1,r-1)$
- $\displaystyle n!=\sum_{i=0}^n s(n,i)$
- $\displaystyle A_x^n=\sum_{i=0}^n s(n,i)(-1)^{n-i}x^i$
- $\displaystyle A_{x+n-1}^n=\sum_{i=0}^n s(n,i)x^i$

***

- 第二类斯特林数×Stirling
- $n$ 个不同的球放入 $r$ 个相同的盒子且无空盒的方案数，记作 $S(n,r)$ 或 $S_n^r$
- 递推式 $S(n,r) = r S(n-1,r) + S(n-1,r-1)$
- 通项公式 $\displaystyle S(n,r)=\frac{1}{r!}\sum_{i=0}^r(-1)^i{r\choose i}(r-i)^n$
- $\displaystyle m^n=\sum_{i=0}^mS(n,i)A_m^i$
- $\displaystyle \sum_{i=1}^n i^k=\sum_{i=0}^kS(k,i)i!{n+1\choose i+1}$
- 斯特林反演
- $\displaystyle f(n)=\sum_{i=1}^n S(n,i)g(i)\Leftrightarrow g(n)=\sum_{i=0}^n(-1)^{n-i}s(n,i)f(i)$

***

- $a$ 个相同的球放入 $b$ 个不同的盒子，方案数为 $C_{a+b-1}^{b-1}$（隔板法）

***

- 一个长为 $n+m$ 的数组，$n$ 个 $1$，$m$ 个 $-1$，限制前缀和最大为 $k$，则方案数为 $C_{n+m}^{m+k}-C_{n+m}^{m+k+1}$

***

- $2n$ 个带标号的点两两匹配，方案数为 $(2n-1)!!=\dfrac{(2n)!}{2^nn!}$
- $1,2,...,n$ 中无序地选择 $r$ 个互不相同且互不相邻的数字，则这 $r$ 个数字之积对所有方案求和的结果为 $C_{n+1}^{2r}(2r-1)!!=\dfrac{C_{n+1}^{2r}(2r)!}{2^rr!}$（问题可以转换为，$(n+1)$ 个点无序匹配 $r$ 对点的方案数）

```c++
int M(int a,int b){
	static const int inv2=qpow(2,mod-2);
	return C(a+1,2*b)*C.fac[2*b]%mod*qpow(inv2,b)%mod*C.inv[b]%mod;
}
```

***

## 博弈论

### SG函数 SG定理

- 有向无环图中，两个玩家轮流推多颗棋子，不能走的判负
- 假设 $x$ 的后继状态为 $y_1,y_2,...,y_k$
- 则 $SG[x]=mex\{SG[y_i]\}$，$mex(S)$ 表示不属于集合 $S$ 的最小自然数
- 当且仅当所有起点SG值的异或和为 $0$ 时先手必败
- （如果只有一个起点，SG的值可以只考虑01）
- 例题：拿 $n$ 堆石子，每次只能拿一堆中的斐波那契数颗石子

```c++
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

### Nim游戏

***

Nim

- $n$ 堆石子 $a_1,a_2,...,a_n$，每次选择 $1$ 堆石子拿任意非空的石子，拿不了的人失败
- $SG_i=a_i,NimSum=\oplus\{SG_i\}$，先手必败当且仅当 $NimSum=0$
- 注：先手必胜策略是找到满足 `(a[i]>>(63-__builtin_clzll(NimSum)))&1` 的 $a[i]$，并取走 $a[i]-a[i]\oplus NimSum$ 个石子
- Bash Game：一堆石子 $n$，最多取 $k$ 个，$SG=n\%(k+1)$

***

Moore's Nimk

- $n$ 堆石子，每次最多选取 $k$ 堆石子，选中的每一堆都取走任意非空的石子
- 先手必胜当且仅当
	- 存在 $t$ 使得 `sum{(a[i]>>t)&1}%(k+1)!=0`

***

扩展威佐夫博弈×Extra Wythoff's Game

- 两堆石子，分别为 $a,b$，每次取一堆的任意非空的石子或者取两堆数量之差的绝对值小于等于 $k$ 的石子
- 解：假设 $a\le b$，当且仅当存在自然数 $n$ 使得 $a=\lfloor n\dfrac{\sqrt{(k+1)^2+4}-(k-1)}2\rfloor,b=a+n(k+1)$，先手必败
- Betty定理与Betty数列：$\alpha,\beta$ 为正无理数且 $\dfrac 1 {\alpha}+\dfrac 1 {\beta}=1$，数列 $\{\lfloor \alpha n\rfloor\},\{\lfloor \beta n\rfloor\},n=1,2,...$ 无交集且覆盖正整数集合

***

斐波那契博弈×Fibonacci Nim

- 一堆石子 $n,n\ge 2$，先手第一次只能取 $[1,n-1]$，之后每次取的石子数不多于对手刚取的石子数的 $2$ 倍且非空
- 先手必败当且仅当 $n$ 是Fibonacci数

***

阶梯Nim×Staircase Nim

- $n$ 堆石子，每次选择一堆取任意非空的石子放到前一堆，第 $1$ 堆的石子可以放到第 $0$ 堆
- 先手必败当且仅当奇数堆的石子数异或和为 $0$

***

Lasker's Nim

- $n$ 堆石子，每次可以选择一堆取任意非空石子，或者选择某堆至少为 $2$，分成两堆非空石子
- $SG(0)=0,SG(4k+1)=4k+1,SG(4k+2)=4k+2,SG(4k+3)=4k+4,SG(4k+4)=4k+3$

***

k倍动态减法博弈

- 一堆石子 $n,n\ge 2$，先手第一次只能取 $[1,n-1]$，之后每次取的石子数不多于对手刚取的石子数的 $k$ 倍且非空

```c++
int calc(ll n,int k){ //n<=1e8,k<=1e5
	static ll a[N],b[N],ans; //N=750010
	int t=1;
	a[1]=b[1]=1;
	for(int j=0;;){
		t++,a[t]=b[t-1]+1;
		if(a[t]>=n)break;
		while(a[j+1]*k<a[t])j++;
		b[t]=a[t]+b[j];
	}
	while(a[t]>n)t--;
	if(a[t]==n)return -1;
	while(n){
		while(a[t]>n)t--;
		n-=a[t]; ans=a[t];
	}
	return ans;
}
```

***

Anti-SG | SJ定理

- $n$ 个游戏，移动不了的人获胜
- 先手必胜当且仅当
	- $(\forall i)SG_i\le 1$ 且 $NimSum=0$
	- $(\exist i)SG_i>1$ 且 $NimSum\not=0$

***

Every-SG

- $n$ 个游戏，每次都要移动所有可移动的游戏
- 对于先手来说，必胜态的游戏要越长越好，必败态的游戏要越短越好
- u是终止态，step(u)=0
- u->v,SG(u)=0,SG(v)>0，step(u)=max(step(v))+1
- u->v,SG(v)=0，step(u)=min(step(v))+1
- 先手必胜当且仅当所有游戏的step的最大值为奇数

### 删边游戏×Green Hachenbush

- 树上删边游戏
	- 一棵有根树，每次可以删除一条边并移除不和根连接的部分
	- 叶子的 $SG$ 为 $0$，非叶子的 $SG$ 为(所有儿子的 $SG$ 值 $+1$)的异或和
- 无向图删边游戏
	- 奇环可以缩为一个点加一条边，偶环可以缩为一点，变为树上删边游戏

### 翻硬币游戏

- $n$ 枚硬币排成一排，玩家的操作有一定约束，并且翻动的硬币中，最右边的必须是从正面翻到反面，不能操作的玩家失败
- 定理：局面的 $SG$ 值等于所有正面朝上的硬币单一存在时的 $SG$ 值的异或和（把这个硬币以外的所有硬币翻到反面后的局面的 $SG$ 值）
- 编号从 $1$ 开始
	- 每次翻一枚或两枚硬币 $SG(n)=n$
	- 每次翻转连续的 $k$ 个硬币 $SG(n)=[n\%k=0]$
	- Ruler Game，每次翻转一个区间的硬币，$SG(n)=lowbit(n)$
	- Mock Turtles Game，每次翻转不多于 $3$ 枚硬币 $SG(n)=2n-1-popcount(n-1)\%2$

### 高维组合游戏 | Nim积

- Nim和与Nim积的关系类似加法与乘法
- Tartan定理：对于一个高维的游戏（多个维度的笛卡尔积），玩家的操作也是笛卡尔积的形式，那么对每一维度单独计算SG值，最终的SG值为它们的Nim积
- 比如，在 $n\times m$ 硬币中翻转 $4$ 个硬币，$4$ 个硬币构成一个矩形，这个矩形是每一维度（翻转两个硬币）的笛卡尔积
- $O(\log^2 n)$

```c++
struct Nim{
	ll rec[256][256];
	ll f(ll x,ll y,int len=32) {
		if(x==0 || y==0) return 0;
		if(x==1 || y==1) return x*y;
		if(len<=4 && rec[x][y]) return rec[x][y];
		ll xa=x>>len,xb=x^(xa<<len),ya=y>>len,yb=y^(ya<<len);
		ll a=f(xb,yb,len>>1),b=f(xa^xb,ya^yb,len>>1),c=f(xa,ya,len>>1),d=f(c,1ll<<(len-1),len>>1);
		ll ans=((b^a)<<len)^a^d;
		if(len<=4)rec[x][y]=ans;
		return ans;
	}
}nim;
//int x=read(),y=read(),z=read();
//ans^=nim.f(SG(x),nim.f(SG(y),SG(z)));
```

### 不平等博弈 | 超现实数

***

- 超现实数(Surreal Number)
- 超现实数由左右集合构成，是最大的兼容四则运算的全序集合，包含实数集和“无穷大”
- 博弈局面的值可以看作左玩家比右玩家多进行的次数，独立的局面可以相加
- 如果值 $>0$ 则左玩家必胜，$<0$ 则右玩家必胜，$=0$ 则后手必胜
- 一个博弈局面，$L$ 为左玩家操作一次后的博弈局面的最大值，$R$ 为右玩家操作一次后的博弈局面的最小值，那么该博弈局面的值 $G=\dfrac A {2^B},L<G<R$，并且 $B$ 尽可能小（$B=0$ 则 $|A|$ 尽可能小）
- 如果存在 $L=R$ 需要引入Irregular surreal number就不讨论了（比如两个玩家能进行同一操作即Nim）

***

- Blue-Red Hackenbush string
- 若干个 BW 串，player-W 只能拿 W，player-B 只能拿 B，每次拿走一个字符后其后缀也会消失，最先不能操作者输
- 对于每个串计算超现实数(Surreal Number)并求和，若 $> 0$ 则 W 必胜；若 $= 0$ 则后手必胜；若 $< 0$ 则 B 必胜

```c++
ll calc(char s[]){
	int n=strlen(s);
	ll ans=0,k=1LL<<50; int i;
	for(i=0;i<n && s[i]==s[0];i++)
		ans+=(s[i]=='W'?k:-k);
	for(;i<n;i++)
		k>>=1,ans+=(s[i]=='W'?k:-k);
	return ans;
}
```

```c++
int ans[N];
void calc(char s[]){
	int n=strlen(s);
	int p=0; while(s[p]==s[0])p++;
	ans[0]+=(s[0]=='W'?p:-p);
	repeat(i,p,n)ans[i-p+1]+=(s[i]=='W'?1:-1);
}
void adjust(){
	repeat_back(i,1,N)
		ans[i-1]+=ans[i]/2,ans[i]%=2;
}
```

***

- Blue-Red Hackenbush tree
- 若干棵树，点权为 W 或 B，player-W 只能删 W，player-B 只能删 B，每次删点后与根不相连部分也移除
- 对于 W 点，先求所有儿子的值之和 $x$。如果 $x \ge 0$，那么直接加一即可。否则 $x$ 变为 $x$ 的小数部分加一，乘以 $2^{-\lfloor|x|\rfloor}$

***

- Alice's Game
- $x\times y$ 方格，如果 $x>1$ Alice可以水平切，如果 $y>1$ Bob可以垂直切，超现实数计算如下

```c++
ll calc(int x,int y){ //get surreal number
	while(x>1 && y>1)x>>=1,y>>=1;
	return x-y;
}
```

***

### 其他博弈结论

***

欧几里得的游戏

- 两个数 $a,b$，每次对一个数删去另一个数的整数倍，出现 $0$ 则失败
- $a\ge 2b$ 则先手必胜，否则递归处理

***

无向点地理问题×Undirected vertex geography problem

- 二分图上移动棋子，不能经过重复点
- 先手必败当且仅当存在一个不包含起点的最大匹配

***

- 1到n，每次拿一个数或差值为1的两个数
	- 先手必胜，第一步拿最中间的1/2个数，之后对称操作
- $n\times m$ 棋盘上两个棋子，每次双方可以操控自己的棋子移动到同一行/列的位置，不能经过对方棋子所在行/列
	- 后手必胜当且仅当两个棋子的横坐标之差等于纵坐标之差
- 2个数字，每次把一个数字减少，最小1，但是不能出现重复数字
	- $SG(a,b)=((a-1)\oplus(b-1))-1$
- 3个数字，每次把一个数字减少，最小1，但是不能出现重复数字
	- 后手必胜当且仅当 $a\oplus b\oplus c=0$

***

## 代数结构

### 置换群

- 求 $A^x$，编号从 $0$ 开始，$O(n)$

```c++
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

- $A^k=B$ 求任一 $A$，编号从 $0$ 开始，$O(n)$（暂无判断有解操作）

```c++
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

### 多项式

技能树：拉格朗日反演，多项式开根、快速幂、除法、三角反三角，分治FFT，快速插值和多点求值

#### 拉格朗日插值

- 函数曲线通过n个点 $(x_i,y_i)$，求 $f(k)$
- 拉格朗日插值：$f(x)=\sum\limits_{i=1}^n[y_i\Pi_{j!=i}\dfrac{x-x_j}{x_i-x_j}]$
- $O(n^2)$

```c++
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

```c++
ll solve(int n,int x0){ //(i,y[i]),i=1..n的优化
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

#### 多项式基本操作

```c++
inline ll D(ll x){return x>=mod?x-mod:x<0?x+mod:x;}
inline ll &ad(ll &x){return x=D(x);}
int polyinit(ll a[],int n1){
	int n=1; while(n<n1)n<<=1;
	fill(a+n1,a+n,0);
	return n;
}
void polyder(ll a[],int n,ll b[]){ //b=da/dx
	repeat(i,1,n)b[i-1]=i*a[i]%mod; b[n-1]=0;
}
void polycal(ll a[],int n,ll b[]){ //b=∫adx
	repeat_back(i,1,n)b[i]=qpow(i,mod-2)*a[i-1]%mod; b[0]=0;
}
void polymul_special(ll a[],ll b[],int n,ll c[]){ //c[i]=a[j]*b[k],i=j*k
	fill(c,c+n,0);
	repeat(i,0,n)
	repeat(j,0,n){
		if(i*j>=n)break;
		(c[i*j]+=a[i]*b[j])%=mod;
	}
}
```

#### 快速傅里叶变换×FTT+任意模数

- 离散傅里叶变换(DFT)即求 $(\omega_n^k,f(\omega_n^k))$，多项式 $\displaystyle d_k=\sum_{i=0}^{n-1}a_i(\omega_n^k)^i$
- 离散傅里叶反变换(IDFT)即求多项式 $(\omega_n^{-k},g(\omega_n^{-k}))$，多项式 $\displaystyle c_k=\sum_{i=0}^{n-1}d_i(\omega_n^{-k})^i$，最后 $a_i=\dfrac {c_i}{n}$
- 求两个多项式的卷积，$O(n\log n)$

```c++
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
	vector<ll> conv(const vector<ll> &u,const vector<ll> &v){ //一般fft
		const int n=(int)u.size()-1,m=(int)v.size()-1;
		const int k=32-__builtin_clz(n+m+1),s=1<<k;
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
	vector<ll> conv_mod(const vector<ll> &u,const vector<ll> &v,ll mod){ //任意模数fft
		const int n=(int)u.size()-1,m=(int)v.size()-1,M=sqrt(mod)+1;
		const int k=32-__builtin_clz(n+m+1),s=1<<k;
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

#### 快速数论变换×NTT

```c++
//const ll mod=998244353;
void ntt(ll a[],ll n,ll op){
	for(int i=1,j=n>>1;i<n-1;++i){
		if(i<j)swap(a[i],a[j]);
		int k=n>>1;
		while(k<=j)j-=k,k>>=1;
		j+=k;
	}
	for(int len=2;len<=n;len<<=1){
		ll rt=qpow(3,(mod-1)/len);
		for(int i=0;i<n;i+=len){
			ll w=1;
			repeat(j,i,i+len/2){
				ll u=a[j],t=1ll*a[j+len/2]*w%mod;
				a[j]=D(u+t),a[j+len/2]=D(u-t);
				w=1ll*w*rt%mod;
			}
		}
	}
	if(op==-1){
		reverse(a+1,a+n);
		ll in=qpow(n,mod-2);
		repeat(i,0,n)a[i]=1ll*a[i]*in%mod;
	}
}
void conv(ll a[],ll b[],int n,ll c[],const function<ll(ll,ll)> &f=[](ll a,ll b){return a*b%mod;}){ //n=2^k
	//fill(a+n,a+n*2,0); fill(b+n,b+n*2,0); n*=2;
	ntt(a,n,1); ntt(b,n,1);
	repeat(i,0,n)c[i]=f(a[i],b[i]);
	ntt(c,n,-1);
}
```

#### 快速沃尔什变换×FWT

- 计算 $\displaystyle c_i=\sum_{i=f(j,k)}a_jb_k$，$O(n\log n)$

```c++
void fwt(ll a[],int n,int flag,char c){
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
		if(flag==-1)flag=qpow(2,mod-2);
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
void polymul_bit(ll a[],ll b[],int n,char c){ //n=2^k
	fwt(a,n,1,c); fwt(b,n,1,c);
	repeat(i,0,n)a[i]=(a[i]*b[i])%mod;
	fwt(a,n,-1,c);
}
```

#### 多项式运算

- $O(n\log n)$，两倍空间
- 多项式开方 $g[0]\not=0$ 则另需二次剩余板子（$$
- 若 $\ln f(x)$ 存在，则 $[x^0]f(x)=1$；若 $\exp f(x)$ 存在，则 $[x^0]f(x)=0$

```c++
void polyinv(ll h[],int n,ll f[]){ //n=2^k, h!=f, the answer satisfies h*f=1 (mod x^n)
	static ll tmp[N];
	if(n==1){f[0]=qpow(h[0],mod-2); return;}
	polyinv(h,n/2,f);
	fill(f+n/2,f+n,0);
	copy(h,h+n,tmp);
	conv(f,tmp,n*2,f,[](ll a,ll b){
		return a*(2-a*b%mod+mod)%mod;
	});
}
const int inv2=qpow(2,mod-2);
void polysqrt(ll g[],int n,ll f[]){ //n=2^k, g!=f, the answer satisfies f*f=g (mod x^n)
	static ll g0[N],finv[N];
	if(n==1){f[0]=1; return;} //f[0]=sqrtmod(g[0]);
	polysqrt(g,n/2,f);
	polyinv(f,n,finv);
	fill(finv+n,finv+n*2,0);
	copy(g,g+n,g0);
	conv(g0,finv,n*2,g0);
	repeat(i,0,n)f[i]=inv2*(g0[i]+f[i])%mod;
}
void polyln(ll a[],int n,ll b[]){ //n=2^k, g!=f
	static ll t[N];
	polyder(a,n,t); polyinv(a,n,b);
	conv(t,b,2*n,t);
	polycal(t,n,b); fill(b+n,b+n*2,0);
}
void polyexp(ll a[],int n,ll b[]){ //n=2^k, g!=f
	static ll lnb[N];
	if(n==1){b[0]=1; return;}
	polyexp(a,n>>1,b);
	polyln(b,n,lnb);
	repeat(i,0,n)lnb[i]=D(a[i]-lnb[i]); lnb[0]++;
	conv(b,lnb,n*2,b);
}
```

#### 多项式的一些结论及生成函数

- 遇到 $\displaystyle \sum_{i=0}^n[i\%k=0]f(i)$ 可以转换为 $\displaystyle \sum_{i=0}^n\dfrac 1 k\sum_{j=0}^{k-1}(\omega_k^i)^jf(i)$
- 广义二项式定理 $\displaystyle (1+x)^{\alpha}=\sum_{i=0}^{\infty}{n\choose \alpha}x^i$

##### 普通生成函数×OGF

- 普通生成函数：$A(x)=a_0+a_1x+a_2x^2+...=\langle a_0,a_1,a_2...\rangle$
- $1+x^k+x^{2k}+...=\dfrac{1}{1-x^k}$
- 取对数后 $\displaystyle=-\ln(1-x^k)=\sum_{i=1}^{\infty}\dfrac{1}{i}x^{ki}$ 即 $\displaystyle\sum_{i=1}^{\infty}\dfrac{1}{i}x^i\otimes x^k$（polymul_special）
- $x+\dfrac{x^2}{2}+\dfrac{x^3}{3}+...=-\ln(1-x)$
- $1+x+x^2+...+x^{m-1}=\dfrac{1-x^m}{1-x}$
- $1+2x+3x^2+...=\dfrac{1}{(1-x)^2}$（借用导数，$nx^{n-1}=(x^n)'$）
- $C_m^0+C_m^1x+C_m^2x^2+...+C_m^mx^m=(1+x)^m$（二项式定理）
- $C_m^0+C_{m+1}^1x^1+C_{m+2}^2x^2+...=\dfrac{1}{(1-x)^{m+1}}$（归纳法证明）
- $\displaystyle\sum_{n=0}^{\infty}F_nx^n=\dfrac{(F_1-F_0)x+F_0}{1-x-x^2}$（$F$ 为斐波那契数列，列方程 $G(x)=xG(x)+x^2G(x)+(F_1-F_0)x+F_0$）
- $\displaystyle\sum_{n=0}^{\infty} H_nx^n=\dfrac{1-\sqrt{n-4x}}{2x}$（$H$ 为卡特兰数）
- 前缀和 $\displaystyle \sum_{n=0}^{\infty}s_nx^n=\dfrac{1}{1-x}f(x)$
- 五边形数定理：$\displaystyle \prod_{i=1}^{\infty}(1-x^i)=\sum_{k=0}^{\infty}(-1)^kx^{\frac 1 2k(3k\pm 1)}$

##### 指数生成函数×EGF

- 指数生成函数：$A(x)=a_0+a_1x+a_2\dfrac{x^2}{2!}+a_3\dfrac{x^3}{3!}+...=\langle a_0,a_1,a_2,a_3,...\rangle$
- 普通生成函数转换为指数生成函数：系数乘以 $n!$
- $1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+...=\exp x$
- 长度为 $n$ 的循环置换数为 $P(x)=-\ln(1-x)$，长度为 $n$ 的置换数为 $\exp P(x)=\dfrac{1}{1-x}$（注意是**指数**生成函数）
- 推广：
	- $n$ 个点的生成树个数是 $\displaystyle P(x)=\sum_{n=1}^{\infty}n^{n-2}\dfrac{x^n}{n!}$，$n$ 个点的生成森林个数是 $\exp P(x)$
	- $n$ 个点的无向连通图个数是 $P(x)$，$n$ 个点的无向图个数是 $\displaystyle\exp P(x)=\sum_{n=0}^{\infty}2^{\frac 1 2 n(n-1)}\dfrac{x^n}{n!}$
	- 长度为 $n(n\ge 2)$ 的循环置换数是 $P(x)=-\ln(1-x)-x$，长度为 $n$ 的错排数是 $\exp P(x)$

### 矩阵

#### 矩阵乘法 矩阵快速幂

- 已并行优化，矩乘 $O(n^3)$，矩快 $O(n^3\log b)$

```c++
struct mat{
	static const int N=110;
	ll a[N][N];
	explicit mat(ll e=0){
		repeat(i,0,n)
		repeat(j,0,n)
			a[i][j]=e*(i==j);
	}
	mat operator*(const mat &b)const{
		mat ans(0);
		repeat(i,0,n)
		repeat(k,0,n){
			ll t=a[i][k];
			repeat(j,0,n)
				(ans.a[i][j]+=t*b.a[k][j])%=mod;
		}
		return ans;
	}
	ll *operator[](int x){return a[x];}
	const ll *operator[](int x)const{return a[x];}
};
mat qpow(mat a,ll b){
	mat ans(1); //mat ans; repeat(i,0,n)ans[i][i]=1;
	while(b){
		if(b&1)ans=ans*a;
		a=a*a; b>>=1;
	}
	return ans;
}
```

#### 矩阵高级操作

- 行列式、逆矩阵（luogu P3389 && luogu P4783）
- $O(n^3)$

```c++
int n,m;
#define T ll
struct mat{
	static const int N=110;
	vector< vector<T> > a;
	mat():a(N,vector<T>(N*2)){} //如果要求逆这里乘2
	T det;
	void r_div(int x,T k){ //第x行除以实数k
		T r=qpow(k,mod-2);
		repeat(i,0,m) //从x开始也没太大关系（对求det来说）
			a[x][i]=a[x][i]*r%mod;
		det=det*k%mod;
	}
	void r_plus(int x,int y,T k){ //第x行加上第y行的k倍
		repeat(i,0,m)
			a[x][i]=(a[x][i]+a[y][i]*k)%mod;
	}
	/*
	void r_div(int x,T k){ //lf版
		T r=1/k;
		repeat(i,0,m)a[x][i]*=r;
		det*=k;
	}
	void r_plus(int x,int y,T k){ //lf版
		repeat(i,0,m)a[x][i]+=a[y][i]*k;
	}
	*/
	bool gauss(){ //返回是否满秩，注意必须n<=m
		det=1;
		repeat(i,0,n){
			int t=-1;
			repeat(j,i,n)
			if(abs(a[j][i])>eps){t=j; break;}
			if(t==-1){det=0; return 0;}
			if(t!=i){a[i].swap(a[t]); det=-det;}
			r_div(i,a[i][i]);
			repeat(j,0,n) //如果只要det可以从i+1开始
			if(j!=i && abs(a[j][i])>eps)
				r_plus(j,i,-a[j][i]);
		}
		return 1;
	}
	T get_det(){gauss(); return det;} //返回行列式
	bool get_inv(){ //把自己变成逆矩阵，返回是否成功
		if(n!=m)return 0;
		repeat(i,0,n)
		repeat(j,0,n)
			a[i][j+n]=i==j; //生成增广矩阵
		m*=2; bool t=gauss(); m/=2;
		repeat(i,0,n)
		repeat(j,0,n)
			a[i][j]=a[i][j+n];
		return t;
	}
	//vector<T> &operator[](int x){return a[x];}
	//const vector<T> &operator[](int x)const{return a[x];}
}a;
```

- 任意模数行列式（HDOJ 2827）
- $O(n^3\log C)$

```c++
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

#### 异或方程组

- 编号从 $0$ 开始，高斯消元部分 $O(n^3)$（luogu P2962）

```c++
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
int solve(){ //返回满足方程组的sum(xi)最小值
	ans=inf; gauss(n); dfs(n-1,0);
	return ans;
}
```

#### 线性基

- 线性基是一系列线性无关的基向量组成的集合

<H4>异或线性基</H4>

- 结论：$basis.exist(a^b)$ 等价于 $a,b$ 在 $basis$ 里消去关键位后相等（要求是最简线性基，即第一个板子）
- 插入、查询 $O(\log M)$

```c++
struct basis{
	static const int n=63;
	#define B(x,i) ((x>>i)&1)
	ll a[n],sz;
	bool failpush; //是否线性相关
	void init(){mst(a,0); sz=failpush=0;}
	void push(ll x){ //插入元素
		repeat(i,0,n)if(B(x,i))x^=a[i];
		if(x!=0){
			int p=63-__builtin_clzll(x); sz++;
			repeat(i,p+1,n)if(B(a[i],p))a[i]^=x;
			a[p]=x;
		}
		else failpush=1;
	}
	ll top(){ //最大值
		ll ans=0;
		repeat(i,0,n)ans^=a[i];
		return ans;
	}
	bool exist(ll x){ //是否存在
		repeat_back(i,0,n)
		if((x>>i)&1){
			if(a[i]==0)return 0;
			else x^=a[i];
		}
		return 1;
	}
	ll kth(ll k){ //第k小，不存在返回-1
		if(failpush)k--; //如果认为0是可能的答案就加这句话
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
basis operator+(basis a,const basis &b){ //将b并入a
	repeat(i,0,a.n)
	if(b.a[i])a.push(b.a[i]);
	a.failpush|=b.failpush;
	return a;
}
```

- 这个版本中求kth需要rebuild $O(\log^2 n)$

```c++
struct basis{
	//...
	void push(ll x){ //插入元素
		repeat_back(i,0,n)
		if((x>>i)&1){
			if(a[i]==0){a[i]=x; sz++; return;}
			else x^=a[i];
		}
		failpush=1;
	}
	ll top(){ //最大值
		ll ans=0;
		repeat_back(i,0,n)
			ans=max(ans,ans^a[i]);
		return ans;
	}
	void rebuild(){ //求第k小的前置操作
		repeat_back(i,0,n)
		repeat_back(j,0,i)
		if((a[i]>>j)&1)
			a[i]^=a[j];
	}
}b;
```

<H4>实数线性基</H4>

- 编号从 $0$ 开始，插入、查询 $O(n^2)$

```c++
struct basis{
	lf a[N][N]; bool f[N]; int n; //f[i]表示向量a[i]是否被占
	void init(int _n){
		n=_n;
		fill(f,f+n,0);
	}
	bool push(lf x[]){ //返回0表示可以被线性表示，不需要插入
		repeat(i,0,n)
		if(abs(x[i])>1e-5){ //这个值要大一些
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

#### 线性规划 | 单纯形法

- 声明：还没学会
- $\left[\begin{array}{ccccccc} a & a & a & a & a & a & b \\ a & a & a & a & a & a & b \\ a & a & a & a & a & a & b \\ c & c & c & c & c & c & v \end{array}\right]$
- 每行表示一个约束，$\sum ax\le b$，并且所有 $x\ge 0$，求 $\sum cx$ 的最大值
- 对偶问题：每列表示一个约束，$\sum ax\ge c$，并且所有 $x\ge 0$，求 $\sum bx$ 的最小值
- 先找 $c[y]>0$ 的 $y$，再找 $b[x]>0$ 且 $\dfrac {b[x]}{a[x][y]}$ 最小的x（找不到 $y$ 则 $v$，找不到 $x$ 则 INF），用行变换将 $a[x][y]$ 置 $1$，将其他 $a[i][y]$ 和 $c[y]$ 置 $0$
- 编号从 $1$ 开始，$O(n^3)$，缺init

```c++
const int M=1010; const lf eps=1e-6;
int n,m;
lf a[N][M],b[N],c[M],v; //a[1..n][1..m],b[1..n],c[1..m]
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
lf simplex(){ //返回INF表示无限制，否则返回答案
	while(1){
		int x,y;
		for(y=1;y<=m;y++)if(c[y]>eps)break;
		if(y==m+1)return v;
		lf mn=INF;
		repeat(i,1,n+1)
		if(a[i][y]>eps && mn>b[i]/a[i][y])
			mn=b[i]/a[i][y],x=i;
		if(mn==INF)return INF; //unbounded
		pivot(x,y);
	}
}
void init(){v=0;}
```

#### 矩阵的一些结论

***

- $n\times n$ 方阵 $A$ 有：$\left[\begin{array}{c}A&E\\O&E\end{array}\right]^{k+1}=\left[\begin{array}{c}A^k&E+A+A^2+...+A^k\\O&E\end{array}\right]$

***

- 线性递推转矩快

$$
f_{n+3}=af_{n+2}+bf_{n+1}+cf_{n}
$$

$$
\Leftrightarrow\left[\begin{array}{c}a&b&c\\1&0&0\\0&1&0\end{array}\right]^n \left[\begin{array}{c}f_2\\f_1\\f_0\end{array}\right]=\left[\begin{array}{c}f_{n+2}\\f_{n+1}\\f_{n}\end{array}\right]
$$

***

- 追赶法解周期性方程（未测）
- $\left[\begin{array}{ccccc}a_0 & b_0 & c_0 \\ & a_1 & b_1 & c_1 \\ & & ... & ... & ... \\ c_{n-2} & & & a_{n-2}& b_{n-2} \\ b_{n-1} & c_{n-1}\end{array}\right]X=\left[\begin{array}{c}x_0\\x_1\\x_2\\...\\x_{n-1}\end{array}\right]$

```c++
lf a[N],b[N],c[N],x[N]; //结果存x
void run(){
	c[0]/=b[0]; a[0]/=b[0]; x[0]/=b[0];
	for(int i=1;i<N-1;i++){
		lf temp=b[i]-a[i]*c[i-1];
		c[i]/=temp;
		x[i]=(x[i]-a[i]*x[i-1])/temp;
		a[i]=-a[i]*a[i-1]/temp;
	}
	a[N-2]=-a[N-2]-c[N-2];
	for(int i=N-3;i>=0;i--){
		a[i]=-a[i]-c[i]*a[i+1];
		x[i]-=c[i]*x[i+1];
	}
	x[N-1]-=(c[N-1]*x[0]+a[N-1]*x[N-2]);
	x[N-1]/=(c[N-1]*a[0]+a[N-1]*a[N-2]+b[N-1]);
	for(int i=N-2;i>=0;i--)
		x[i]+=a[i]*x[N-1];
}
```

***

## 数学杂项

### 主定理

- 对于 $T(n)=aT(\dfrac nb)+n^k$ （要估算 $n^k$ 的 $k$ 值）
- 若 $\log_ba>k$，则 $T(n)=O(n^{\log_ba})$
- 若 $\log_ba=k$，则 $T(n)=O(n^k\log n)$
- 若 $\log_ba<k$（有省略），则 $T(n)=O(n^k)$

### 质数表

42737, 46411, 50101, 52627, 54577, 191677, 194869, 210407, 221831, 241337, 578603, 625409, 713569, 788813, 862481, 2174729, 2326673, 2688877, 2779417, 3133583, 4489747, 6697841, 6791471, 6878533, 7883129, 9124553, 10415371, 11134633, 12214801, 15589333, 17148757, 17997457, 20278487, 27256133, 28678757, 38206199, 41337119, 47422547, 48543479, 52834961, 76993291, 85852231, 95217823, 108755593, 132972461, 171863609, 173629837, 176939899, 207808351, 227218703, 306112619, 311809637, 322711981, 330806107, 345593317, 345887293, 362838523, 373523729, 394207349, 409580177, 437359931, 483577261, 490845269, 512059357, 534387017, 698987533, 764016151, 906097321, 914067307, 954169327

1572869, 3145739, 6291469, 12582917, 25165843, 50331653 （适合哈希的素数）

19260817   原根15，是某个很好用的质数
1000000007 原根5
998244353  原根3

- NTT素数表， $g$ 是模 $(r \cdot 2^k+1)$ 的原根

```c++
            r*2^k+1   r  k  g
                  3   1  1  2
                  5   1  2  2
                 17   1  4  3
                 97   3  5  5
                193   3  6  5
                257   1  8  3
               7681  15  9 17
              12289   3 12 11
              40961   5 13  3
              65537   1 16  3
             786433   3 18 10
            5767169  11 19  3
            7340033   7 20  3
           23068673  11 21  3
          104857601  25 22  3
          167772161   5 25  3
          469762049   7 26  3
          998244353 119 23  3
         1004535809 479 21  3
         2013265921  15 27 31
         2281701377  17 27  3
         3221225473   3 30  5
        75161927681  35 31  3
        77309411329   9 33  7
       206158430209   3 36 22
      2061584302081  15 37  7
      2748779069441   5 39  3
      6597069766657   3 41  5
     39582418599937   9 42  5
     79164837199873   9 43  5
    263882790666241  15 44  7
   1231453023109121  35 45  3
   1337006139375617  19 46  3
   3799912185593857  27 47  5
   4222124650659841  15 48 19
   7881299347898369   7 50  6
  31525197391593473   7 52  3
 180143985094819841   5 55  6
1945555039024054273  27 56  5
4179340454199820289  29 57  3
```

### struct of 自动取模

- 不好用，别用了

```c++
struct mint{
	ll v;
	mint(ll _v){v=_v%mod;}
	mint operator+(const mint &b)const{return v+b.v;}
	mint operator-(const mint &b)const{return v-b.v;}
	mint operator*(const mint &b)const{return v*b.v;}
	explicit operator ll(){return (v+mod)%mod;}
};
```

### struct of 高精度

- 加、减、乘、单精度取模、小于号和等于号（其他不等号用rel_ops命名空间）
- 如果涉及除法，~~那就完蛋~~，用java吧；如果不想打这么多行也用java吧

```c++
struct big{
	vector<ll> a;
	static const ll k=1000000000,w=9;
	int size()const{return a.size();}
	explicit big(const ll &x=0){ //接收ll
		*this=big(to_string(x));
	}
	explicit big(const string &s){ //接收string
		static ll p10[9]={1};
		repeat(i,1,w)p10[i]=p10[i-1]*10;
		int len=s.size();
		int f=(s[0]=='-')?-1:1;
		a.resize(len/w+1);
		repeat(i,0,len-(f==-1))
			a[i/w]+=f*(s[len-1-i]-48)*p10[i%w];
		adjust();
	}
	int sgn(){return a.back()>=0?1:-1;} //这个只能在强/弱调整后使用
	void shrink(){ //收缩（内存不收缩）
		while(size()>1 && a.back()==0)a.pop_back();
	}
	void adjust(){ //弱调整
		repeat(i,0,3)a.push_back(0);
		repeat(i,0,size()-1){
			a[i+1]+=a[i]/k;
			a[i]%=k;
		}
		shrink();
	}
	void final_adjust(){ //强调整
		adjust();
		int f=sgn();
		repeat(i,0,size()-1){
			ll t=(a[i]+k*f)%k;
			a[i+1]+=(a[i]-t)/k;
			a[i]=t;
		}
		shrink();
	}
	explicit operator string(){ //转换成string
		static char s[N]; char *p=s;
		final_adjust();
		if(sgn()==-1)*p++='-';
		repeat_back(i,0,size())
			sprintf(p,i==size()-1?"%lld":"%09lld",abs(a[i])),p+=strlen(p);
		return s;
	}
	const ll &operator[](int n)const{ //访问
		return a[n];
	}
	ll &operator[](int n){ //弹性访问
		repeat(i,0,n-size()+1)a.push_back(0);
		return a[n];
	}
};
big operator+(big a,const big &b){
	repeat(i,0,b.size())a[i]+=b[i];
	a.adjust();
	return a;
}
big operator-(big a,const big &b){
	repeat(i,0,b.size())a[i]-=b[i];
	a.adjust();
	return a;
}
big operator*(const big &a,const big &b){
	big ans;
	repeat(i,0,a.size()){
		repeat(j,0,b.size())
			ans[i+j]+=a[i]*b[j];
		ans.adjust();
	}
	return ans;
}
void operator*=(big &a,ll b){ //有时被卡常
	big ans;
	repeat(i,0,a.size())a[i]*=b;
	a.adjust();
}
ll operator%(const big &a,ll mod){
	ll ans=0,p=1;
	repeat(i,0,a.size()){
		ans=(ans+p*a[i])%mod;
		p=(p*a.k)%mod;
	}
	return (ans+mod)%mod;
}
bool operator<(big a,big b){
	a.final_adjust();
	b.final_adjust();
	repeat_back(i,0,max(a.size(),b.size()))
		if(a[i]!=b[i])return a[i]<b[i];
	return 0;
}
bool operator==(big a,big b){
	a.final_adjust();
	b.final_adjust();
	repeat_back(i,0,max(a.size(),b.size()))
		if(a[i]!=b[i])return 0;
	return 1;
}
```

### 表达式求值

```c++
inline int lvl(const string &c){ //运算优先级，小括号要排最后
	if(c=="*")return 2;
	if(c=="(" || c==")")return 0;
	return 1;
}
string convert(const string &in) { //中缀转后缀
	stringstream ss;
	stack<string> op;
	string ans,s;
	repeat(i,0,in.size()-1){
		ss<<in[i];
		if(!isdigit(in[i]) || !isdigit(in[i+1])) //插入空格
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
ll calc(const string &in){ //后缀求值
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
			//if(c=='^')num.push(qpow(a,b));
		}
	}
	return num.top();
}
```

### 一些数学结论

#### 约瑟夫问题

- n个人编号0..(n-1)，每次数到k出局，求最后剩下的人的编号
- 线性算法，$O(n)$

```c++
int jos(int n,int k){
	int res=0;
	repeat(i,1,n+1)res=(res+k)%i;
	return res; //res+1，如果编号从1开始
}
```

- 对数算法，适用于k较小情况，$O(k\log n)$

```c++
int jos(int n,int k){
	if(n==1 || k==1)return n-1;
	if(k>n)return (jos(n-1,k)+k)%n; //线性算法
	int res=jos(n-n/k,k)-n%k;
	if(res<0)res+=n; //mod n
	else res+=res/(k-1); //还原位置
	return res; //res+1，如果编号从1开始
}
```

#### 格雷码×gray 汉诺塔

<H5>格雷码</H5>

- 一些性质：
- 相邻格雷码只变化一次
- `grey(n-1)` 到 `grey(n)` 修改了二进制的第 `(__builtin_ctzll(n)+1)` 位
- `grey(0)..grey(2^k-1)` 是k维超立方体顶点的哈密顿回路，其中格雷码每一位代表一个维度的坐标
- 格雷码变换，正 $O(1)$，逆 $O(logn)$

```c++
ll grey(ll n){ //第n个格雷码
	return n^(n>>1);
}
ll degrey(ll n){ //逆格雷码变换，法一
	repeat(i,0,63) //or 31
		n=n^(n>>1);
	return n;
}
ll degrey(ll n){ //逆格雷码变换，法二
	int ans=0;
	while(n){
		ans^=n;
		n>>=1;
	}
	return ans;
}
```

<H5>汉诺塔</H5>

- 假设盘数为n，总共需要移动 `(1<<n)-1` 次
- 第k次移动第 `i=__builtin_ctzll(n)+1` 小的盘子
- 该盘是第 `(k>>i)+1` 次移动
- （可以算出其他盘的状态：总共移动了 `((k+(1<<(i-1)))>>i)` 次）
- 该盘的移动顺序是：
	`A->C->B->A（当i和n奇偶性相同）`
	`A->B->C->A（当i和n奇偶性不同）`

```c++
cin>>n; //层数
repeat(k,1,(1<<n)){
	int i=__builtin_ctzll(k)+1;
	int p1=(k>>i)%3; //移动前状态
	int p2=(p1+1)%3; //移动后状态
	if(i%2==n%2){
		p1=(3-p1)%3;
		p2=(3-p2)%3;
	}
	cout<<"move "<<i<<": "<<"ABC"[p1]<<" -> "<<"ABC"[p2]<<endl;
}
```

- 4个柱子的汉诺塔情况：令 $k=\lfloor n+1-\sqrt{2n+1}+0.5\rfloor$，让前k小的盘子用4个柱子的方法移到2号柱，其他盘子用3个柱子的方法移到4号柱，最后再移一次前k小，最短步数 $f(n)=2f(k)+2^{n-k}-1$

#### Stern-Brocot树 Farey序列

- 分数序列：在 $[\dfrac 0 1,\dfrac 1 0]$ 中不断在 $\dfrac a b$ 和 $\dfrac c d$ 之间插入 $\dfrac {a+c}{b+d}$
- 性质：所有数都是既约分数、可遍历所有既约分数、保持单调递增
- Stern-Brocot树：二叉树，其第 $k$ 行是分数序列第 $k$ 次操作新加的数
- Farey序列：$F_n$ 是所有分子分母 $\le n$ 的既约分数按照分数序列顺序排列后的序列
- $F_n$ 的长度 $=1+\sum\limits_{i=1}^n\varphi(i)$

#### 浮点与近似计算

<H5>数值积分 | 自适应辛普森法</H5>

- 求 $\int_{l}^{r}f(x)\mathrm{d}x$ 的近似值

```c++
lf raw(lf l,lf r){ //辛普森公式
	return (f(l)+f(r)+4*f((l+r)/2))*(r-l)/6;
}
lf asr(lf l,lf r,lf eps,lf ans){
	lf m=(l+r)/2;
	lf x=raw(l,m),y=raw(m,r);
	if(abs(x+y-ans)<=15*eps)
		return x+y-(x+y-ans)/15;
	return asr(l,m,eps/2,x)+asr(m,r,eps/2,y);
}
//调用方法：asr(l,r,eps,raw(l,r))
```

<H5>牛顿迭代法</H5>

- 求 $f(x)$ 的零点：$x_{n+1}=x_n-\dfrac{f(x)}{f'(x)}$
- 检验 $x_{n+1}=g(x_n)$ 多次迭代可以收敛于 $x_0$ 的方法：看 $|g'(x_0)|\le1$ 是否成立

```c++
lf newton(lf n){ //sqrt
	lf x=1;
	while(1){
		lf y=(x+n/x)/2;
		if(abs(x-y)<eps)return x;
		x=y;
	}
}
```

- java高精度的整数平方根

```java
public static BigInteger isqrtNewton(BigInteger n){
	BigInteger a=BigInteger.ONE.shiftLeft(n.bitLength()/2);
	boolean d=false;
	while(true){
		BigInteger b=n.divide(a).add(a).shiftRight(1);
		if(a.compareTo(b)==0 || a.compareTo(b)<0 && d)
			break;
		d=a.compareTo(b)>0;
		a=b;
	}
	return a;
}
```

<H5>others of 浮点与近似计算</H5>

- $\lim\limits_{n\rightarrow\infty}\dfrac{错排(n)}{n!}=\dfrac 1 e,e\approx 2.718281828459045235360287471352$
- $\lim\limits_{n\rightarrow\infty}(\sum\frac 1 n-\ln n)=\gamma\approx 0.577215664901532860606$

#### others of 数学杂项

***

- 埃及分数Engel展开
- 待展开的数为 $x$，令 $u_1=x, u_{i+1}=u_i\times\lceil\dfrac 1 {u_i}\rceil-1$（到0为止）
- 令 $a_i=\lceil\dfrac 1 {u_i}\rceil$
- 则 $x=\dfrac 1{a_1}+\dfrac 1{a_1a_2}+\dfrac 1{a_1a_2a_3}+...$

***

- 三个水杯容量为 $a,b,c$（正整数），$a=b+c$，初始 $a$ 装满水，则得到容积为 $\dfrac a 2$ 的水需要倒 $\dfrac a{\gcd(b,c)}-1$ 次水（无解条件为 $\dfrac a{\gcd(b,c)}\%2=1$）

***

- 兰顿蚂蚁（白色异或右转，黑色异或左转），约一万步后出现周期为104步的无限重复（高速公路）

***

- 任意勾股数能由复数 $(a+bi)^2\space(a,b∈\Z)$ 得到

***

- 任意正整数 $a$ 都存在正整数 $b,c$ 使得 $a<b<c$ 且 $a^2,b^2,c^2$ 成等差数列：构造 $b=5a,c=7a$

***

- 拉格朗日四平方和定理：每个正整数都能表示为4个整数平方和
- 对于偶素数 $2$ 有 $2=1^2+1^2+0^2+0^2$
- 对于奇素数 $p$ 有 $p=a^2+b^2+1^2+0^2$ （容斥可证）
- 对于所有合数 $n$ 有 $n=z_1^2+z_2^2+z_3^2+z_4^2=(x_1^2+x_2^2+x_3^2+x_4^2)\cdot(y_1^2+y_2^2+y_3^2+y_4^2)$
- 其中 $\begin{cases} z_1=x_1y_1+x_2y_2+x_3y_3+x_4y_4 \\ z_2=x_1y_2-x_2y_1-x_3y_4+x_4y_3 \\ z_3=x_1y_3-x_3y_1+x_2y_4-x_4y_2 \\ z_4=x_1y_4-x_4y_1-x_2y_3+x_3y_2\end{cases}$

***

- 雅可比四平方和定理：设 $a^2+b^2+c^2+d^2=n$ 的整数解个数为 $S(n)$，有 $S(2^k m)=\begin{cases}8d(m) & if\ k=0\\24d(m) & if\ k>0\end{cases}(m\ is\ odd)$，$d(n)$ 为 n 的约数和

***

- 基姆拉尔森公式
- 已知年月日，返回星期几

```c++
int week(int y,int m,int d){
	if(m<=2)m+=12,y--;
	return (d+2*m+3*(m+1)/5+y+y/4-y/100+y/400)%7+1;
}
```

***

标准阳历与儒略日转换

```c++
int DateToInt(int y, int m, int d){
	return
	1461 * (y + 4800 + (m - 14) / 12) / 4 +
	367 * (m - 2 - (m - 14) / 12 * 12) / 12 -
	3 * ((y + 4900 + (m - 14) / 12) / 100) / 4 +
	d - 32075;
}
void IntToDate(int jd, int &y, int &m, int &d){
	int x, n, i, j;
	x = jd + 68569;
	n = 4 * x / 146097;
	x -= (146097 * n + 3) / 4;
	i = (4000 * (x + 1)) / 1461001;
	x -= 1461 * i / 4 - 31;
	j = 80 * x / 2447;
	d = x - 2447 * j / 80;
	x = j / 11;
	m = j + 2 - 12 * x;
	y = 100 * (n - 49) + i + x;
}
```
***

求 sum(n/i)

```c++
int f(int n){
	int ans=0;
	int t=sqrt(n);
	repeat(i,1,t+1)ans+=n/i;
	return ans*2-t*t;
}
```

***

n 维超立方体有 $2^{n-i}×C(n, i)$ 个 i 维元素

***
