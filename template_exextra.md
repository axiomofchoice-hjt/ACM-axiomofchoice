- [unclassified](#unclassified)
	- [Notice](#notice)
	- [费马-欧拉素数定理补充](#费马-欧拉素数定理补充)
	- [卡常操作](#卡常操作)
	- [多项式除法+取模](#多项式除法取模)
	- [带通配符的字符串匹配 using FFT](#带通配符的字符串匹配-using-fft)
	- [图哈希](#图哈希)
	- [递推式插值 using BM算法](#递推式插值-using-bm算法)
	- [美术馆定理 using 多边形三角剖分](#美术馆定理-using-多边形三角剖分)
	- [排序网络](#排序网络)
	- [汉明码](#汉明码)
	- [矩形里的小球](#矩形里的小球)
	- [平面图欧拉定理](#平面图欧拉定理)
	- [平面图5-染色](#平面图5-染色)
	- [N车存在问题](#n车存在问题)
	- [到给定点距离之和最小的直线](#到给定点距离之和最小的直线)
	- [Raney 引理](#raney-引理)
	- [整数分解为2的幂的方案数](#整数分解为2的幂的方案数)

# unclassified

## Notice

- 求 $\displaystyle B_i = \sum_{k=i}^n C_k^iA_k$，即 $\displaystyle B_i=\dfrac{1}{i!}\sum_{k=i}^n\dfrac{1}{(k-i)!}\cdot k!A_k$，反转后卷积

## 费马-欧拉素数定理补充

- 对于 $2$ 有 $2=1^2+1^2$
- 对于模 $4$ 余 $1$ 的素数有费马-欧拉素数定理
- 对于完全平方数有 $x^2=x^2+0^2$
- 对于合数有 $(a^2+b^2)(c^2+d^2)=(ac+bd)^2+(ad-bc)^2$
- 对于无法用上述方式，即合数但是指数为奇数的素因子模 $4$ 余 $3$，不能分解为两整数平方和

## 卡常操作

```c++
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
int mul(int a,int b,int m=mod){ //模乘
	int ret;
	__asm__ __volatile__ ("\tmull %%ebx\n\tdivl %%ecx\n"
		:"=d"(ret):"a"(a),"b"(b),"c"(m));
	return ret;
}
unsigned rev(unsigned x){ //反转二进制x
	#define ma(i) (~0u / (1<<i | 1))
	#define rr(i) x = (x & ma(i))<<i | (x>>i & ma(i))
	rr(1); rr(2); rr(4); rr(8); rr(16);
	return x;
}
int abs(int x){
	int y=x>>31;
	return (x+y)^y;
}
int max(int x,int y){
	int m=(x-y)>>31;
	return (y&m) | (x&~m);
}
```

## 多项式除法+取模

- 计算 $d=f/g,r=f\%g$，即 $f=g\times d+r$
- $O(n\log n)$，极丑代码警告（为了与以前的屎山代码兼容）

```c++
void polydivmod(ll f[],ll g[],int n,int m,ll d[],ll r[]){
	static ll A[N],B[N];
	fill(f+n,f+n*2,0);
	fill(g+m,g+n*2,0);
	reverse_copy(g,g+m,A);
	fill(A+m,A+n*2,0);
	polyinv(A,n,B);
	fill(B+n,B+n*2,0); 
	reverse_copy(f,f+n,A);
	fill(A+n,A+n*2,0);
	conv(A,B,n*2,d);
	reverse(d,d+n-m+1);
	fill(d+n-m+1,d+n*2,0);
	copy(d,d+n*2,A);
	copy(g,g+n*2,B);
	conv(A,B,n*2,r);
	repeat(i,0,n*2)r[i]=D(f[i]-r[i]);
}
```

## 带通配符的字符串匹配 using FFT

- 模式串 $A(x)$ 长为 $m$，文本串 $B(x)$ 长为 $n$，通配符数值为 $0$
- 反转 $A(i)=A'(m-i-1)$
- 令 $C(x,y)=[A(x)-B(y)]^2A(x)B(y)$
$$
\begin{array}{ccl}
P(x)
&=&\displaystyle\sum_{i=0}^{m-1}C(i,x+i)\\
&=&\displaystyle\sum_{i=0}^{m-1}[A(i)-B(x+i)]^2A(i)B(x+i)\\
&=&\displaystyle\sum_{i=0}^{m-1}[A^3(i)B(x+i)-2A^2(i)B^2(x+i)+A(i)B^3(x+i)]\\
&=&\displaystyle\sum_{i=0}^{m-1}[A'^3(m-i-1)B(x+i)-2A'^2(m-i-1)B^2(x+i)+A'(m-i-1)B^3(x+i)]
\end{array}
$$
- 先计算 $A,A^2,A^3,B,B^2,B^3$ 然后 FFT/NTT

```c++
int func(char c){return c=='*'?0:c-'a'+1;}
char sa[N],sb[N];
ll A[N],B[N],A2[N],A3[N],B2[N],B3[N];
void Solve(){
	int m1=read(),n1=read();
	scanf("%s%s",sa,sb);
	repeat(i,0,m1)A[i]=func(sa[m1-i-1]);
	repeat(i,0,n1)B[i]=func(sb[i]);
	int n=polyinit(B,n1);
	fill(A+m1,A+n,0);
	repeat(i,0,n){
		A2[i]=A[i]*A[i]; A3[i]=A2[i]*A[i];
		B2[i]=B[i]*B[i]; B3[i]=B2[i]*B[i];
	} //(A,A2,A3,B,B2,B3)[n,n*2-1] uninitialized
	ntt(A,n*2,1); ntt(A2,n*2,1); ntt(A3,n*2,1);
	ntt(B,n*2,1); ntt(B2,n*2,1); ntt(B3,n*2,1);
	repeat(i,0,n*2)
		A[i]=D((A[i]*B3[i]%mod-2*A2[i]*B2[i]%mod+A3[i]*B[i])%mod);
	ntt(A,n*2,-1);
	vector<int> ans;
	repeat(i,m1-1,n1)if(A[i]==0)ans<<i-m1+2;
	printf("%d\n",(int)ans.size());
	for(auto i:ans)printf("%d ",i);
}
```

## 图哈希

- 枚举起点 $s$，令所有点的权值 $f_0(i)=1$，迭代：
- $\displaystyle f_{j+1}(u)=\left[A\cdot f_j(u)+B\cdot\sum_{u\rightarrow w}f_j(w)+C\cdot\sum_{w\rightarrow u}f_j(w)+D\cdot[u=s]\right]\bmod P$
- 取 $f_k(s)$。对所有 $s$ 取 $n$ 个值组成集合
- 如果是无向图就去掉 $C$ 项；如果会超时，就去掉 $D$ 项

## 递推式插值 using BM算法

- 已知数列前几项，求递推式系数 $C_0a_i+C_1a_{i+1}+...+C_ka_{i+k}=0,C_k=-1$
- 用来找规律

```c++
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

## 美术馆定理 using 多边形三角剖分

- 多边形三角剖分：设 $A,B,C$ 为连续的三点，若所有其他顶点不在 $\triangle ABC$ 内，则将原多边形用线段 $AC$ 划分；否则必然存在 $\triangle ABC$ 内且离线段 $AC$ 最远的点 $D$，则将原多边形用线段 $BD$ 划分
- 美术馆定理：对任意 $n$ 边形美术馆，一定可以放置 $\lfloor \dfrac n 3\rfloor$ 个守卫（守卫具有 $360\degree$ 视角）来看守整个美术馆
- 对美术馆进行三角剖分，并对所有顶点 3-染色，保证任意两条边有不同颜色（任意三角形的顶点有三种颜色）。对三种颜色的点集取最小的点集即可

## 排序网络

- $O(\log^2 n)$

```
--X------X----X------X--------X----X----
--X------|-X--X------|-X------|-X--X----
----X----|-X----X----|-|-X----X-|----X--
----X----X------X----|-|-|-X----X----X--
--X------X----X------|-|-|-X--X----X----
--X------|-X--X------|-|-X----|-X--X----
----X----|-X----X----|-X------X-|----X--
----X----X------X----X----------X----X--
```

## 汉明码

- 长度为 $n=2^k$，$\log n+1$ 位冗余（$S_0,S_{2^i}$），$\displaystyle \oplus_{i=0}^{n-1}S_i=0, (\forall b)\oplus_{i\cap 2^b\not=0}S_i=0$，可以检测最多两位噪声，定位最多一位噪声

## 矩形里的小球

- $(n+1)\times (m+1)$ 矩形中，小球从左下顶点往上 $a(a=0..n)$ 格的位置向右上发射，在矩形边界处反弹，回到起点后停止。问有几个格子经过了奇数次（起点只记一次）
- 情况1：撞到角上完全反弹，$\gcd(n,m)\mid a$（包含 $a=0$），答案为 $2$（角和起点）
- 情况2：没有撞到角上，$\gcd(n,m)\nmid a$，答案为 $\dfrac{2(nm-n(m/g-1)-m(n/g-1))}{g}$（经过的格子数 $\tfrac {2nm}{g}$ 减去经过两次的格子数）（这种情况最多经过两次）

参考：03年IOI论文姜尚仆

## 平面图欧拉定理

- $|V|-|E|+|F|=2$
- 给定连通简单平面图，若 $|V|≥3$，则 $|E|≤3|V|-6$
- 可知平面图的边数为 $O(V)$
- 补充：给定连通简单平面图，若 $|V|≥3$，则 $\exist v,deg(v)\le 5$

## 平面图5-染色

- 由 $\exist deg(v)\le 5$，找到度最小的点 $u$，递归地对剩下的图进行5-染色
- 然后考虑 $u$ 的颜色，如果 $u$ 的邻居中 5 种颜色没有都出现，就直接染色，否则考虑顺时针的 5 个邻居 $v_1,v_2,v_3,v_4,v_5$，考虑两个子图，与 $v_1$ 或 $v_3$ 颜色相同的点集的导出子图 $G_1$，与 $v_2$ 或 $v_4$ 颜色相同的点集的导出子图 $G_2$，则 $v_1,v_3$ 在 $G_1$ 中不连通、$v_2,v_4$ 在 $G_2$ 中不连通两个命题必然有一个成立（否则出现两个相互嵌套的环，不构成平面图），假设 $v_1,v_3$ 在 $G_1$ 中不连通，那么将 $G_1$ 中 $v_1$ 的连通分量的所有顶点颜色取反（$color[v_1]$ 与 $color[v_2]$ 互换），这样 $u$ 的邻居变为 $4$ 种颜色，将 $u$ 染色为原来的 $color[v_1]$
- 如果顺时针的关系很难得到，就尝试两次（$[v_1,v_2,v_3,v_4],[v_1,v_3,v_2,v_4]$）

## N车存在问题

- 给定 $n\times m$ 有障碍地图，车不能越过障碍物，要使车不互相攻击，最多放置多少车
- 将每个极大的 $1\times k$ 的空地作为点集 $A$，每个极大的 $k times 1$ 的空地作为点集 $B$，若两个点对应的空地区域有交集则连边，跑二分图最大匹配

## 到给定点距离之和最小的直线

- shrink一下让点集没有三点共线
- 最优解一定是两点连线且把剩下点基本平分，因此可以绕直线上的某个点旋转直到碰到另外一点，然后绕新点旋转，重复上述操作，$O(n^2)$

参考：04年IOI论文金恺

## Raney 引理

- 设整数序列 $A = [a_1,a_2,...,a_n]$，前缀和 $S_k=a_1+...+a_k$，所有数字之和 $S_n=1$
- 则在 $A$ 的 $n$ 个循环表示中，有且仅有一个序列满足其前缀和 $S_i$ 均大于零
- 证明：画成折线图后最低且最后的那一点作为起点

## 整数分解为2的幂的方案数

- 即 A018819 $[1, 1, 2, 2, 4, 4, 6, 6, 10, 10, 14, 14, ...]$
- 递推式 $a_{2m+1} = a_{2m}, a_{2m} = a_{2m-1} + a_{m}$
- 用矩阵乘法可以加速，达到 $O(\log^4 n)$

```c++
void Solve(){
	int q=read();
	n=1;
	mat t(1); vector<ll> a={1};
	while(q){
		if(q&1)a=t*a;
		t=t*t;
		n++;
		copy_n(t[n-2],n,t[n-1]);
		t[n-1][n-1]=1;
		a.push_back(a.back());
		q>>=1;
	}
	cout<<a.back()<<endl;
}
```