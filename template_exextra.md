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
	- [最小 k 度限制生成树 次小生成树](#最小-k-度限制生成树-次小生成树)
	- [坐船问题](#坐船问题)
	- [Minkowski 和](#minkowski-和)
	- [整点向量线性基](#整点向量线性基)
	- [多边形构造](#多边形构造)
	- [欧拉图更新 using 套圈算法](#欧拉图更新-using-套圈算法)
	- [Voronoi 图](#voronoi-图)
	- [矩形孔明棋](#矩形孔明棋)
	- [削平山顶](#削平山顶)
	- [瓶颈费用最大流](#瓶颈费用最大流)
	- [点集同构](#点集同构)
	- [最优序列 using 费用流](#最优序列-using-费用流)
	- [环形匹配](#环形匹配)
	- [旋转矩阵](#旋转矩阵)
	- [反归并排序](#反归并排序)
	- [矩形匹配 using 根号算法](#矩形匹配-using-根号算法)
	- [高维曼哈顿距离](#高维曼哈顿距离)
	- [三角形四心一点](#三角形四心一点)
	- [二维欧几里得算法](#二维欧几里得算法)
	- [分母最小的分数](#分母最小的分数)
	- [Shannon 开关游戏](#shannon-开关游戏)
	- [计算几何板子补充](#计算几何板子补充)
	- [圆的离散化](#圆的离散化)
	- [杨表 / Young tableaux](#杨表--young-tableaux)
	- [奇怪的矩阵公式](#奇怪的矩阵公式)
	- [最大权不下降子序列](#最大权不下降子序列)
	- [强连通分量SCC using Kosaraju](#强连通分量scc-using-kosaraju)
	- [2-sat using SCC](#2-sat-using-scc)
	- [网格路径计数](#网格路径计数)
	- [线性代数复习](#线性代数复习)
	- [组合数学结论](#组合数学结论)
	- [多项式全家桶 vector 版](#多项式全家桶-vector-版)
	- [多项式快速幂](#多项式快速幂)
	- [多项式复合](#多项式复合)
	- [多项式多点求值](#多项式多点求值)
	- [括号序列专题](#括号序列专题)
	- [减一的杨辉矩阵](#减一的杨辉矩阵)
	- [区间历史最值](#区间历史最值)
	- [K-D tree 模板更新](#k-d-tree-模板更新)

# unclassified

## Notice

- 求 $\displaystyle B_i = \sum_{k=i}^n C_k^iA_k$，即 $\displaystyle B_i=\dfrac{1}{i!}\sum_{k=i}^n\dfrac{1}{(k-i)!}\cdot k!A_k$，反转后卷积
- `__builtin_expect(!!(expr),1),__builtin_expect(!!(expr),0)` 放在 if() 中，如果 expr 大概率为真/假则可以优化常数
- 多物网络流：$k$ 个源汇点，$S_i$ 需要流 $f_i$ 单位流量至 $T_i$。多物网络流只能用线性规划解决
- 树上倍增lca板子里，dfs应该在dis赋值的后面！！！
- 范德蒙德卷积公式：$\displaystyle{\sum_{k}\binom{r}{k}\binom{s}{n-k}=\binom{r+s}{n}}$

## 费马-欧拉素数定理补充

- 对于 $2$ 有 $2=1^2+1^2$
- 对于模 $4$ 余 $1$ 的素数有费马-欧拉素数定理
- 对于完全平方数有 $x^2=x^2+0^2$
- 对于合数有 $(a^2+b^2)(c^2+d^2)=(ac+bd)^2+(ad-bc)^2$
- 对于无法用上述方式，即存在模 $4$ 余 $3$ 的、指数为奇数的素因子，不能分解为两整数平方和
- （本质上是一个整数分解为高斯素数的过程）
- 令 $\chi[1^+]=1,0,-1,0,1,0,-1\ldots$，是一个完全积性函数。正整数 $n$ 分解为两个整数平方和的方案数为 $n$ 所有约数 $\chi$ 值之和，$f(n)=\sum_{d\mid n}\chi(d)$

## 卡常操作

```c++
int mul(int a,int b,int m=mod){ // 汇编模乘
	int ret;
	__asm__ __volatile__ ("\tmull %%ebx\n\tdivl %%ecx\n"
		:"=d"(ret):"a"(a),"b"(b),"c"(m));
	return ret;
}
int bitrev(int x){ // 反转二进制x
	int z;
	#define op(i) \
		z = ~0u / (1 << i | 1); \
		x = (x & z) << i | (x >> i & z)
	op(1); op(2); op(4); op(8); op(16);
	#undef op
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
int kthbit(int x,int k){ // 从低到高第k个1的位置
	int ans=0,z;
	#define op(b) \
		z=__builtin_popcount(x & ((1 << b) - 1)); \
		if(z<k)k-=z,ans|=b,x>>=b;
	op(16); op(8); op(4); op(2); op(1);
	#undef op
	return ans;
}
```

查表法反转二进制位

```c++
u32 rev_tbl[65537];
void bitrevinit() {
	rev_tbl[0] = 0;
	for (int i=1; i<65536; ++i)
	rev_tbl[i] = (rev_tbl[i>>1]>>1) | ((i&1)<<15);
}
u32 bitrev(u32 x) {
	return (rev_tbl[x & 65535] << 16) | rev_tbl[x >> 16];
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
	} // (A,A2,A3,B,B2,B3)[n,n*2-1] uninitialized
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

```text
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

参考：CTSC 03 姜尚仆

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

参考：CTSC 04 金恺

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

## 最小 k 度限制生成树 次小生成树

最小 $k$ 度生成树

- 即某个点 $v_0$ 度不大于 $k$ 的最小生成树
- 去掉 $v_0$ 跑一遍最小生成森林，然后从小到大访问 $v_0$ 的边 $(v_0,v)$ 考虑是否能加入边集
- 如果 $v$ 与 $v_0$ 不连通就直接加，否则判断路径 $v_0 - v$ 上的最大边是否大于 $(v_0,v)$，大于就将它替换为 $(v_0,v)$
- 令 $Best(v)$ 为路径 $v_0-v$ 的最大边，每次树的形态改变后更新 $Best$
- 可以证明最优性，$O(E\log E+kV)$（不知道可不可以数据结构维护）

次小生成树

- 即所有生成树中第二小的生成树
- 跑一遍Kruskal，然后对剩下的边依次询问这条边两个端点的路径最长边，更新答案。树上倍增优化，$O(E\log E)$

## 坐船问题

- $n$ 个学生，同姓的或者同名的两个人可以一起坐船，问最少需要多少船
- 构建一个森林，左儿子和父亲同姓，右儿子和父亲同名。记录一个数组表示该姓/名的最低结点，将学生按姓名排序后依次加入即可
- 贪心匹配。找到最低结点，如果是独生子就和父亲匹配，否则如果父亲是祖父的左儿子就让父亲和其右儿子配对，相反则左儿子
- 由于每棵树都一定只留下 $0$ 或 $1$ 个结点因此贪心成立，$O(n\log n)$

## Minkowski 和

- 两个凸包 $A,B$，定义它们的 Minkowski 和为 $\{a+b\mid a \in A,b\in B\}$
- 将 $A,B$ 的有向边放在一起极角排序，顺序连接，得到答案的形状。再取几个点确定位置

## 整点向量线性基

- $n$ 个向量的集合 $\{(x_i,y_i)\}$ 可以构造线性等价的两个向量的集合 $\{(a_1,b_1),(a_2,b_2)\},(b_2=0)$，即 $\displaystyle\{\sum_{i=1}^n t_i(x_i,y_i)\mid t\in \Z^n\}=\{t_1(a_1,b_1)+t_2(a_2,b_2)\mid t\in \Z^2\}$
- `linear::push(x,y)`: 添加向量
- `linear::query(x,y)`: 询问向量能否被线性表示

```c++
struct linear{
	ll a1,b1,a2; // (a1,b1),(a2,0)
	linear(){a1=b1=a2=0;}
	void push(ll x,ll y){
		ll A,B,d;
		exgcd(y,b1,d,A,B);
		a2=__gcd(a2,abs(y*a1-x*b1)/d);
		b1=d;
		a1=A*x+B*a1;
		if(a2)a1=(a1%a2+a2)%a2;
	}
	bool query(ll x,ll y){
		if(b1!=0 && y%b1==0){
			ll r=x-y/b1*a1;
			return (a2!=0 && r%a2==0) || a2==r;
		}
		else if(b1==0)
			return (a1!=0 && x%a1==0) || a1==x;
		else return false;
	}
};
```

## 多边形构造

- 给 $n$ 个数，构造凸 $n$ 边形使得边具有给定长度
- 若两倍最长边小于等于周长，那么一定可以构造圆内接 $n$ 边形，对圆的半径二分即可
- 判断考虑圆心是否在多边形内的情况，即对半径为最长边的二分之一判断

```c++
const lf eps=1e-10;
int a[N],n; lf th[N];
bool work(lf r,bool state){ // state=1 means outside
	th[0]=asin(a[0]/(2*r))*2;
	if(state)th[0]=pi*2-th[0];
	repeat(i,1,n)
		th[i]=th[i-1]+asin(a[i]/(2*r))*2;
	return (th[n-1]<pi*2)^state;
}
void Solve(){
	n=read(); int sum=0;
	repeat(i,0,n){a[i]=read(); sum+=a[i];}
	swap(a[0],*max_element(a,a+n)); 
	if(a[0]*2>=sum){
		puts("NO SOLUTION");
		return;
	}
	lf l=a[0]/2.0,r=1e7;
	bool state=work(l,0);
	while((r-l)/r>eps){
		lf mid=(l+r)/2;
		if(work(mid,state))r=mid;
		else l=mid;
	}
	repeat(i,0,n){
		printf("%.12f %.12f\n",cos(th[i])*r,sin(th[i])*r);
	}
}
```

## 欧拉图更新 using 套圈算法

- 默认是连通图！
- 无向图
	- 若存在则路径为 DFS 退出序（最后的序列还要再反过来）（如果for从小到大，可以得到最小字典序）
	- （不记录点的 $vis$，只记录边的 $vis$）
- 有向图
	- 欧拉回路存在当且仅当所有点入度等于出度
	- 欧拉路径存在当且仅当除了起点终点外所有点入度等于出度
	- 跑反图退出序
- 混合图
	- 欧拉回路存在当且仅当 `indeg+outdeg+undirdeg` 是偶数，且 `max(indeg,outdeg)*2<=indeg+outdeg+undirdeg`
	- 欧拉路径还没研究过
	- 无向边任意定向算法。每次找 `outdeg>indeg` 的点向 `outdeg<indeg` 的任意点连一条任意路径，要求只经过无向边，将路径上的边转换为有向边，欧拉路径存在条件仍满足。当所有点都有 `outdeg==indeg`，直接反图跑退出序

## Voronoi 图

- Delaunay 三角剖分后，每个线段中垂线构成Voronoi 图
- 它们互为对偶图

## 矩形孔明棋

- 无限大棋盘上有 $n\times m$ 棋子，移动方法同孔明棋，求最小剩下的棋子数

```c++
scanf("%d%d",&n,&m);
if(m==1 || n==1)cout<<(n+m)/2<<endl;
else if(n%3==0 || m%3==0)cout<<2<<endl;
else cout<<1<<endl;
```

## 削平山顶

- $H_i$ 表示位置 $i$ 的高度，且 $H_0=a_{H+1}=0$。若一段连续的平地比两边都高，则称该平地为山顶。每次操作可以让某个位置下降 $1$ 高度。问 $k$ 次操作后山顶数最小值
- 对每个高度 $j$，令 $H_i\ge j$ 的连续一块为一个结点。若下面的块支撑上面的块，则连一条边。可知这是一棵树，叶子即山顶。然后树形DP

## 瓶颈费用最大流

- 使流量为 $k$ 的最大边权最小
- 修改一下 MCMF 费用流算法即可

## 点集同构

- 给两个点集，问能否通过平移、旋转、翻转、缩放操作后重合
- 求出质心，以质心到最远点的距离缩放，然后极角排序（第二关键字为距离），将二元组(极角差分,距离)列出来为 $P,Q$，求出 $PP$ 中是否有 $Q$ 的出现即可
- 翻转其中一个点集后再做一遍。特判质心位置处有点的情况

## 最优序列 using 费用流

- $n$ 个正整数的序列，从中选取和最大的子序列，满足所有长度为 $m$ 的区间里选取的数字不超过 $k$ 个
- 建图跑最大费用流（费用已取反）

```c++
repeat(i,0,n){ // add(x,y,w,cost)
	add(S,i,inf,0);
	if(i+1<n)add(i,i+1,inf,0);
	add(i,i+n,1,-a[i]);
	if(i+m<n)add(i+n,i+m,inf,0);
	add(i+n,T,inf,0);
}
```

## 环形匹配

- refer to CTSC 07 陈雪
- 一个环上有 $n$ 个 A 类点和 $n$ 个 B 类点，给出点的坐标，求 A 类点与 B 类点的完美匹配使得所有边的长度之和最小，其中边的长度定义为环上的距离
- 假设这是一条直线，先对所有点的坐标排序。令 $d[i]$ 表示第 $i$ 个点及以前 A 类点数减 B 类点数。那么每个相邻点的间隔的贡献为 $|d[i]|$ 乘以间隔长度。接下来开始滚动，考虑将最后一个点放到第一个点的位置，那么 $d[i]$ 整体加 1 / 减 1。因此可以维护 $h[i]$ 表示初始 $d[j]=i$ 的 $j$ 的个数，即可维护当前 $d[i]$ 大于、等于、小于 $0$ 的个数

## 旋转矩阵

- 绕原点逆时针旋转 $\theta$ 弧度：

$$
\left[\begin{array}{cc}
	\cos\theta & -\sin\theta & 0 \\
	\sin\theta & \cos\theta & 0 \\
	0 & 0 & 1
\end{array}\right]
\left[\begin{array}{c} x \\ y \\ 1 \end{array}\right]
$$

- 绕 $(x_0,y_0)$ 逆时针旋转 $\theta$ 弧度：

$$
\left[\begin{array}{cc}
	\cos\theta & -\sin\theta & -x_0\cos\theta+y_0\sin\theta+x_0 \\
	\sin\theta & \cos\theta & -x_0\sin\theta-y_0\cos\theta+y_0 \\
	0 & 0 & 1
\end{array}\right]
\left[\begin{array}{c} x \\ y \\ 1 \end{array}\right]
$$

- 平移 $(D_x,D_y)$：

$$
\left[\begin{array}{cc}
	1 & 0 & D_x \\
	0 & 1 & D_y \\
	0 & 0 & 1
\end{array}\right]
\left[\begin{array}{c} x \\ y \\ 1 \end{array}\right]
$$

## 反归并排序

- ural 1568 Train Car Sorting
- 给 n 个数的排列（互不相同），一次操作可以选择一个子序列，将子序列按照原来顺序移动至最前，其余元素按照原来顺序放在最后。问最小操作次数和一种可行的操作方案

```c++
// input: n, a[1..n]
int getcnt(){ // 得到有序分割的大小
	repeat(i,1,n+1)pos[a[i]]=i;
	int cnt=1;
	repeat(i,1,n+1){
		if(i>1 && pos[i-1]>pos[i])cnt++;
		f[pos[i]]=cnt;
	}
	return cnt;
}
void work(){ // 进行一次操作
	getcnt();
	int t=0;
	repeat(i,1,n+1)if(f[i]%2==1)b[++t]=a[i];
	repeat(i,1,n+1)if(f[i]%2==0)b[++t]=a[i];
	copy(b+1,b+n+1,a+1);
}
```

## 矩形匹配 using 根号算法

- refer to CTSC 08 day2 张煜承
- 平面上 $n$ 个点，以任意 $4$ 个顶点组成四条边平行于坐标轴的矩形，求这样的矩形数
- 若第 $i$ 行的点数 $>k$，直接处理这一行的贡献后删除该行。处理方式为，先将第 $i$ 行的列号处理为一个集合，统计第 $j$ 行里出现在集合的点数 $=h(j)$，$\displaystyle\sum_j {h(j)\choose 2}$ 即为贡献。操作最多 $\dfrac n k$ 行，因此复杂度 $O(\dfrac{n^2}{k})$
- 剩下的点中每行点数 $\le k$，考虑对每行的点集两两匹配，暴力统计 $(x_1,y)(x_2,y)$ 中 $y$ 的个数 $=h(x_1,x_2)$，$\displaystyle\sum_{x_1,x_2}{h(x_1,x_2)\choose 2}$ 即为贡献。操作最少 $\dfrac n k$ 行，因此复杂度 $O(nk)$

## 高维曼哈顿距离

- $|\Delta x|+|\Delta y|+|\Delta z|=\max_{f_x,f_y,f_z=\pm 1}(f_x\Delta x+f_y\Delta y+f_z\Delta z)$

## 三角形四心一点

- 三角形重心到三个顶点平方和最小，到三边距离之积最大（三角形内）
- 三角形四心一点（未测试）

```c++
vec circumcenter(vec a,vec b,vec c){ // 外心
	line u,v;
	u.p1=(a+b)*0.5;
	u.p2=u.p1+(a-b).left();
	v.p1=(a+c)*0.5;
	v.p2=v.p1+(a-c).left();
	return u.PI(v);
}
vec incenter(vec a,vec b,vec c){ // 内心
	auto fun=[](vec a,vec b,vec c){
		lf th=((b-a).theta()+(c-a).theta())/2;
		return line(a,a+vec(cos(th),sin(th)));
	};
	line u=fun(a,b,c),v=fun(b,a,c);
	return u.PI(v);
}
vec orthocenter(vec a,vec b,vec c){ // 垂心
	line u(a,a+(b-c).left()),v(b,b+(a-c).left());
	return u.PI(v);
}
vec centroid(vec a,vec b,vec c){return (a+b+c)*(1.0/3);} // 重心
vec fermatpoint(vec a,vec b,vec c){ // 费马点
	if(cross(a-b,a-c)<0)swap(b,c);
	vec cc=b+(a-b).rotate(pi/3);
	return line(b,b+(c-cc).rotate(pi/3)).PI(line(c,cc));
}
```

## 二维欧几里得算法

- 已知二维向量 $a,b$，求 $|ax+by|$ 的最小值 $(x,y\in \Z)$
- 若 $a\cdot b<0$，则将 $b$ 反向
- 若 $\cos\langle a,b\rangle<\dfrac 1 2$，则答案为 $\min(|a|,|b|)$
- 若 $\cos\langle a,b\rangle\ge\dfrac 1 2$，由于 $ans(a,b)=ans(a,b+a)$，假设 $|a|<|b|$，过 $b$ 作 $a$ 的垂线交于，若 $ka$ 和 $(k+1)a$ 在垂线两侧 $(k\ge 0)$，则 $\langle a,b-ka\rangle$ 和 $\langle -a,b-(k+1)a\rangle$ 中选取一个夹角更大的替换 $a,b$，如此反复

## 分母最小的分数

- 求 $\dfrac a b<\dfrac p q<\dfrac c d$ 的分母最小的 $\dfrac p q$
- 若存在整数则直接解决。否则有 $0\le \dfrac{a \bmod b}{b}<\dfrac{p'}{q'}<\dfrac{c\bmod d}{d}<1$，$\dfrac{p'}{q'}+a/b=\dfrac p q$，再取倒数 $\dfrac{d}{c\bmod d}<\dfrac{q'}{p'}<\dfrac{b}{a\bmod b}$，如此反复

## Shannon 开关游戏

- refer to CTSC 07 刘雨辰
- 给无向图，玩家P可以在没有标记的边上标+号，玩家N可以在删除一条没有标记的边，轮流操作直到不能操作。若最终的图连通则玩家P获胜
- 玩家P后手必胜当且仅当存在两棵边独立的生成树
- 若玩家P获胜条件改为顶点 $u,v$ 连通，则玩家P后手必胜当且仅当原图的一个包含 $u,v$ 的导出子图存在两棵边独立的生成树

## 计算几何板子补充

```c++
vec projection(vec v,vec a,vec b){ // v 在 line(a,b) 上的投影
	vec d=b-a;
	return a+d*(dot(v-a,d)/d.sqr());
}
struct cir{
	vec v; lf r;
	void PI(vec a,vec b,vec &A,vec &B){ // PI with line(a,b)
		vec H=projection(v,a,b);
		vec D=(a-b).trunc(sqrt(r*r-(v-H).sqr()));
		A=H+D; B=H-D;
	}
	void PI(cir b,vec &A,vec &B){ // PI with circle
		vec d=b.v-v;
		lf dis=abs(b.r*b.r-r*r-d.sqr())/(2*d.len());
		vec H=v+d.trunc(dis);
		vec D=d.left().trunc(sqrt(r*r-dis*dis));
		A=H+D; B=H-D;
	}
};
```

## 圆的离散化

- refer to CTSC 07 高逸涵
- 若干圆，任意两圆不相切，求未被圆覆盖的闭合图形个数
- 将圆的上下顶点和两两圆的交点的y作为事件，取相邻事件中点 $e[i]$，分析其状态，对相邻的 $e[i]$ 用并查集判连通

```c++
// poj 1688 但是 wa 了
DSU d;
vector<lf> ovo,e;
vector<pair<lf,lf> > rec[N];
vector<int> lab[N]; int labcnt,ans;
void segunion(vector<pair<lf,lf> > &a){ // 区间合并至最简
	if(a.empty())return;
	sort(a.begin(),a.end()); int pre=0;
	repeat(i,0,a.size()){
		if(a[i].fi>a[pre].se-eps)a[++pre]=a[i];
		else a[pre].se=max(a[pre].se,a[i].se);
	}
	a.erase(a.begin()+pre+1,a.end());
}
void segcomplement(vector<pair<lf,lf> > &a){ // 区间取反
	a.push_back(pair<lf,lf>(0,inf));
	repeat_back(i,0,a.size()-1){
		a[i+1].fi=a[i].se;
		a[i].se=a[i].fi;
	}
	a[0].fi=-inf;
}
void Solve(){
	int n=read(); ovo.clear(); e.clear(); labcnt=ans=0;
	repeat(i,0,n){
		a[i].v.x=read(),a[i].v.y=read(),a[i].r=read();
		ovo.push_back(a[i].v.y-a[i].r);
		ovo.push_back(a[i].v.y+a[i].r);
	}
	repeat(i,0,n)
	repeat(j,i+1,n)
	if((a[i].v-a[j].v).len()<a[i].r+a[j].r){
		vec A,B; a[i].PI(a[j],A,B);
		ovo.push_back(A.y);
		ovo.push_back(B.y);
	}
	sort(ovo.begin(),ovo.end());
	e.push_back(-inf);
	repeat(i,0,ovo.size()-1)
		e.push_back((ovo[i]+ovo[i+1])/2);
	e.push_back(inf);
	repeat(j,0,e.size()){
		rec[j].clear();
		repeat(i,0,n)
		if(abs(a[i].v.y-e[j])<a[i].r-eps){
			lf d=sqrt(a[i].r*a[i].r-(a[i].v.y-e[j])*(a[i].v.y-e[j]));
			rec[j].push_back(pair<lf,lf>(a[i].v.x-d,a[i].v.x+d));
		}
		segunion(rec[j]); 
		segcomplement(rec[j]);
		lab[j].assign(rec[j].size(),0);
		repeat(i,0,lab[j].size())lab[j][i]=labcnt++;
	}
	d.init(labcnt);
	repeat(i,0,e.size()-1){
		unsigned p1=0,p2=0;
		while(p1<rec[i].size() && p2<rec[i+1].size()){
			if(rec[i][p1].se>rec[i+1][p2].fi-eps && rec[i][p1].fi<rec[i+1][p2].se+eps){
				int x=lab[i][p1],y=lab[i+1][p2];
				if(d[x]!=d[y]){
					ans++;
					d[x]=d[y];
				}
			}
			(rec[i][p1].se<rec[i+1][p2].se?p1:p2)++;
		}
	}
	printf("%d\n",labcnt-ans-1);
}
```

## 杨表 / Young tableaux

- 杨图：令 $\lambda = (\lambda_1,\lambda_2,\ldots,\lambda_m)$ 满足 $\lambda_1\ge\lambda_2\ge\ldots\lambda_m\ge 1,n=\sum \lambda_i$。一个形状为 $\lambda$ 的杨图是一个表格，第 $i$ 行有 $\lambda_i$ 个方格，其坐标分别为 $(i,1)(i,2)\ldots(i,\lambda_i)$。
- 半标准杨表：将杨图填上数字，满足每行数字单调不减，每列数字单调递增。
- 标准杨表：将 $1,2,\ldots,n$ 填入杨图，满足每行、每列数字单调递增。下图为 $n=9,\lambda=(4,2,2,1)$ 的杨图和标准杨表。
$$
\left[\begin{array}{c}
* & * & * & * \\
* & *         \\
* & *         \\
*
\end{array}\right]
\left[\begin{array}{c}
1 & 4 & 7 & 8 \\
2 & 5         \\
3 & 9         \\
6
\end{array}\right]
$$
- 斜杨图：令 $\lambda = (\lambda_1,\lambda_2,\ldots,\lambda_m),\mu=(\mu_1,\mu_2,\ldots,\mu_{m'})$，则形状为 $\lambda/\mu$ 的斜杨图为杨图 $\lambda$ 中扣去杨图 $\mu$ 后剩下的部分。

***

- 插入操作：从第一行开始，在当前行中找最小的比 $x$ 大的数字 $y$ (upperbound)，交换 $x,y$，转到下一行继续操作；若所有数字比 $x$ 小则把 $x$ 放在该行末尾并退出
- 排列与两个标准杨表一一对应：将排列按顺序插入到杨表A中，并在杨表B中对应位置记录下标
- 对合排列和标准杨表一一对应（对合排列意味着自己乘自己是单位元）
- 将排列插入到杨表中，若比较运算反过来（小于变大于等于），得到的杨图（杨表的形状）和原来的杨图是转置关系
- Dilworth定理：把一个数列划分成最少的最长不升子序列的数目就等于这个数列的最长上升子序列的长度。可知 $k$ 个不相交的不下降子序列的长度之和最大值等于最长的 ( 最长下降子序列长度不超过 $k$ ) 的子序列长度
- 序列生成的杨图前 k 行方格数即 $k$ 个不相交的不下降子序列的长度之和最大值。但是不能用杨图求出这 k 个 LIS
- 第一行为最长上升序列长度，第一列为最长下降序列长度，可得指定 LIS 和 LDS 长度的排列数为 $\displaystyle\sum_{\lambda_1=\alpha,m=\beta} f_\lambda^2$，可由钩子公式计算 $f_\lambda$

***

- $n$ 个元素的标准杨表个数
	- A000085：$[1,1,2,4,10,26,76,232,764,2620,9496,\ldots]$
	- $f(n)=f(n−1)+(n−1)f(n−2), f(0)=f(1)=1$
- 钩子公式：勾长 $h_{\lambda}(x)$ 定义为正右方方格数 + 正下方方格数 + 1。给一个杨图 $\lambda$，其标准杨表个数为 
$$
f_{\lambda}=\dfrac{n!}{\prod h_{\lambda}(x)}=n!\dfrac{\prod_{1\le i<j\le m}(\lambda_i-i-\lambda_j+j)}{\prod_{i=1}^{m}(\lambda_i+m-i)!}
$$

```c++
int n;
int calc(vector<int> &a){ // #define int ll
	int m=a.size(),ans=1;
	repeat(i,0,m)
	repeat(j,i+1,m)
		(ans*=a[i]-i-a[j]+j)%=mod;
	repeat(i,0,m)
		(ans*=C.inv[a[i]+m-i-1])%=mod;
	(ans*=C.fac[n])%=mod;
	(ans+=mod)%=mod;
	return ans;
}
```

- 钩子公式也可以用 FFT 加速至 $O(n\log n)$
- $f_\lambda=n!\dfrac{\prod_{1\le i<j\le m}(r_i-r_j)}{\prod r_i!},r_i=a_i+m-i$

```c++
const int nn=1000010; // 比 n 大就行
ll A[N],B[N],r[N];
ll solve(ll a[],int m){
	repeat(i,0,nn*2)A[i]=B[i]=0;
	repeat(i,1,m+1){
		r[i]=a[i]+m-i;
		A[r[i]]=1;
		B[nn-r[i]]=1;
	}
	int polyn=ntt::polyinit(A,nn*2);
	ll ans=1;
	ntt::conv(A,B,polyn,A);
	repeat(i,1,nn)if(A[i+nn])
		(ans*=qpow(i,A[i+nn]))%=mod;
	// 这里还要乘以 n! 除以 prod r[i]!
	return ans;
}
```

- $2\times n$ 标准杨表个数为卡特兰数 $C_n$
- 初始 $[0]*m$，每次选一个数加 1，最终变成 $[a_1,a_2,\ldots,a_m]$ 的方案数为 $\dfrac{(\sum a_i)!}{\sum a_i!}$
- 上题中，如果要保持序列不下降，则可以看作杨图 $\lambda=a$ 中填入到这个位置的时间戳，得到一个标准杨表，方案数为 $f_\lambda$，[例题](https://loj.ac/p/6051)

***

- [长度为 n 的排列中 LIS 长度的期望](https://www.luogu.com.cn/problem/P4484)
- A003316：$[1, 3, 12, 58, 335, 2261, 17465, 152020, 1473057,\ldots]$（长度为 n 的排列的 LIS 长度之和）

```c++
ll ans=0; vector<int> a;
void dfs(int n,int pre){
	if(n==0){
		int x=calc(a);
		(ans+=x*x%mod*a[0])%=mod;
		return;
	}
	repeat(i,1,min(n,pre)+1){
		a.push_back(i);
		dfs(n-i,i);
		a.pop_back();
	}
}
void Solve(){
	n=read();
	dfs(n,n);
	cout<<ans*C.inv[n]%mod<<endl;
}
```

***

- [双杨表维护 kLIS](https://www.luogu.com.cn/problem/P3774)：支持末尾插入一个数，询问 $k$ 个不相交的不下降子序列的长度之和最大值。两个杨表可以在 $O(\sqrt n \log n)$（应该跑不满）内维护整个杨表的插入

```c++
template<typename less,int N=233> // N>sqrt(::N)
struct young{
	vector<int> a[N];
	void init(){
		for(auto &v:a)v.clear();
	}
	void insert(int x){
		for(auto &v:a){
			auto it=upper_bound(v.begin(),v.end(),x,less());
			if(it==v.end()){
				v.push_back(x);
				return;
			}
			swap(x,*it);
		}
	}
	int topk(int k){
		int ans=0;
		repeat(i,0,min(k,N)){
			ans+=a[i].size();
		}
		return ans;
	}
	int leftNtok(int k){
		int ans=0;
		for(auto &v:a){
			if(min((int)v.size(),k)<=N)break;
			ans+=min((int)v.size(),k)-N;
		}
		return ans;
	}
};
struct doubleyoung{
	young<less<int>> a;
	young<greater_equal<int>> b;
	void insert(int x){a.insert(x); b.insert(x);}
	int query(int k){return a.topk(k)+b.leftNtok(k);}
}Y;
```

***

- 杨图随机游走：初始随机出现在杨图任一位置（每个位置概率 $\tfrac 1 n$），然后往右或往下走（每个位置概率 $\tfrac 1 {h_\lambda(x)}$），则走到边角 $(r,s)$ 概率为
$$
\dfrac 1 n\prod_{i=1}^{r-1}\dfrac{h_\lambda(i,s)}{h_\lambda(i,s)-1}\prod_{j=1}^{s-1}\dfrac{h_\lambda(r,j)}{h_\lambda(r,j)-1}
$$

- 杨图带权随机游走：每行权重 $x_i$，每列权重 $y_j$，初始随机出现在杨图某一位置（概率权重 $x_iy_j$），向下走到某位置的概率权重为目标行的权重，向右为列的权重，则走到边角 $(r,s)$ 概率为
$$
\dfrac{x_ry_s}{\sum x_iy_j}\prod_{i=1}^{r-1}\left(1+\dfrac{x_i}{\sum x_{i+1..r}+\sum y_{s+1..\lambda_i}}\right)\prod_{j=1}^{s-1}\left(1+\dfrac{y_j}{\sum x_{r+1..\lambda^T_j}+\sum y_{j+1..s}}\right)
$$

- 斜半标准杨表计数：
$$
f'_{\lambda/\mu}=\det\left[\dbinom{\lambda_j-j-\mu_i+i+z-1}{\lambda_j-j-\mu_i+i}\right]_{i,j=1}^m
$$
- 斜标准杨表计数：
$$
f_{\lambda/\mu}=(\sum_{i=1}^{m}(\lambda_i-\mu_i))!\det\left[\dfrac{1}{(\lambda_j-j-\mu_i+i)!}\right]_{i,j=1}^m
$$
- 列数不超过 $2k$ 的，元素都在 $[1, n]$ 内的且每行大小为偶数的半标准杨表和长度均为 $2n + 2$ 的 k-Dyck Path 形成双射关系，且计数公式如下：
$$
b_{n,k}=\prod_{1\le i\le j\le n}\dfrac{2k+i+j}{i+j}
$$
- 半标准杨表计数：
$$
f'_\lambda=\prod_{i,j\in\lambda}\dfrac{n+j-i}{h_\lambda(i,j)}=\prod_{1\le i<j\le m}\dfrac{\lambda_i-i-\lambda_j+j}{j-i}
$$

参考：IOI 19 袁方舟

## 奇怪的矩阵公式

$$
\begin{array}{l}
\quad\det\left[(X_i+A_{n-1})\ldots(X_i+A_{j+1})(X_i+B_j)\ldots(X_i+B_1)\right]_{i,j=0}^{n-1}\\
=\prod_{0\le i<j\le n-1}(X_i-X_j)\prod_{1\le i\le j\le n-1}(B_i-A_j)
\end{array}
$$

$$
\det\left[C_{\alpha_i+j}\right]_{i,j=0}^{n-1}=\prod_{0\le i<j\le n-1}(\alpha_j-\alpha_i)\prod_{i=0}^{n-1}\dfrac{(i+n)!(2\alpha_i)!}{(2i)!\alpha_i!(\alpha_i+n)!}
$$

（$C_n$ 为卡特兰数）

## 最大权不下降子序列

- 即把 $\langle x,v\rangle$ 拆成 $v$ 个 $x$ 后求一遍 LIS

```c++
void insert(map<ll,ll> &mp,vector<pii> &push,vector<pii> &pop){ // pii = <x,v>
	for(auto i:push){
		int key=i.fi,val=i.se;
		auto r=mp.lower_bound(key);
		if(r->fi==key)
			r->se+=val,++r;
		else
			mp.emplace_hint(r,key,val);
		auto s=r; int sum=0;
		while(sum+s->se<=val){
			pop.push_back(*s);
			sum+=s->se;
			++s;
		}
		if(s->fi!=INF)pop.push_back({s->fi,val-sum});
		s->se-=val-sum;
		mp.erase(r,s);
	}
	push.clear();
}
// init: mp.clear(),mp[INF]=INF;
// query: INF-mp[i].rbegin()->se
```

## 强连通分量SCC using Kosaraju

- 编号从 $1$ 开始，$O(V+E)$

```c++
int co[N],sz[N]; // (output) co: vertex color, sz: number of vertices of color i
bool vis[N]; vector<int> q; // private
vector<int> a[N],b[N]; // (input) a: graph, b:invgraph
int cnt; // (output) cnt: color number
void dfs1(int x){
	vis[x]=1;
	for(auto p:a[x])if(!vis[p])dfs1(p);
	q.push_back(x);
}
void dfs2(int x,int c){
	vis[x]=0; co[x]=c; sz[c]++;
	for(auto p:b[x])if(vis[p])dfs2(p,c);
}
void getscc(int n){
	fill(vis,vis+n+1,0);
	fill(sz,sz+n+1,0);
	cnt=0; q.clear();
	repeat(i,1,n+1)if(!vis[i])dfs1(i);
	reverse(q.begin(),q.end());
	for(auto i:q)if(vis[i])dfs2(i,++cnt);
}
```

## 2-sat using SCC

- 编号从 $1$ 开始，$O(V+E)$，注意 $N$ 需要手动两倍

```c++
bool ans[N]; // shows one solution if possible
bool twosat(int n){ // return whether possible
	getscc(n*2);
	repeat(i,1,n+1)
		if(co[i]==co[i+n])return 0;
	repeat(i,1,n+1)
		ans[i]=(co[i]<co[i+n]);
	return 1;
}
```

## 网格路径计数

- 从 $(0,0)$ 走到 $(a,b)$，每次只能从 $(x,y)$ 走到 $(x+1,y-1)$ 或 $(x+1,y+1)$，方案数记为 $f(a,b)=\dbinom{a}{\tfrac{a+b}{2}}$
- 若路径和直线 $y=k,k\notin [0,b]$ 不能有交点，则方案数为 $f(a,b)−f(a,2k−b)$
- 若路径和两条直线 $y=k_1,y=k_2,k_1<0\le b<k_2$ 不能有交点，方案数记为 $g(a,b,k_1,k_2)$，必须碰到 $y=k_1$ 不能碰到 $y=k_2$ 的方案数记为 $h(a,b,k_1,k_2)$，可递归求解（递归过程中两条直线距离会越来越大），$O(n)$

```c++
ll f(ll a,ll b){ // (0,0) -> (a,b)
	if((a+b)%2==1)return 0;
	return C(a,(a+b)/2);
}
ll h(ll,ll,ll,ll);
ll g(ll a,ll b,ll k1,ll k2){ // (0,0) -> (a,b), can't meet y=k1 or y=k2
	if(a<abs(b))return 0;
	return (f(a,b)-f(a,2*k2-b)-h(a,b,k1,k2)+mod+mod)%mod;
}
ll h(ll a,ll b,ll k1,ll k2){ // (0,0) -> (a,b), must meet y=k1, can't meet y=k2
	if(a<abs(b) || a<abs(2*k1-b))return 0;
	return (g(a,2*k1-b,2*k1-k2,k2)+h(a,b,2*k1-k2,k2))%mod;
}
```

- 从 $(0,0)$ 走到 $(a,0)$，只能右上/右下，必须有恰好一次传送（向下 $b$ 单位），不能走到 $x$ 轴下方，方案数为 $\dbinom{a+1}{\frac{a-b}{2}+k+1}$

refer to [博客](https://www.luogu.com.cn/blog/wolfind/yi-suo-ge-lu-jing-ji-shuo-wen-ti)

## 线性代数复习

- 伴随矩阵 $A^*=|A|A^{-1}$，$A^*_{j,i}=A$ 去掉 $i$ 行 $j$ 列后的矩阵的行列式乘以 $(-1)^{i+j}$，注意转置的问题

## 组合数学结论

- 卡特兰数 $C_n$ 
- 有 $2\nmid C_n\rightarrow n=2^k-1$
- Hankel 矩阵：$n\times n$ 矩阵 $A_{i,j}=C_{i+j-2}$，有 $\det A=1$，$B_{i,j}=C_{i+j-1}$，也有 $\det B=1$。反过来可以用 $A,B$ 定义卡特兰数

***

- Bell 数 $B_n$
- $\sum_{n=0}^{\infty}B_n\dfrac{x^n}{n!}=e^{e^x-1}$

***

- 超级卡特兰数 $S_n$，
- 若 $n\times n$ 矩阵 $A_{i,j}=S_{i+j-1}$，则 $\det A=2^{\tfrac{n(n+1)}{2}}$
- $S_n$ 为在 $n\times n$ 的矩形中选定 $n$ 个点和 $n$ 条水平/竖直的线段，满足每条线段恰好经过一个点且每个点恰好只被一条线段经过且线段直接不出现十字交叉，这 $n$ 条线段把矩形划分成 $n+1$ 个小矩形的方案数

***

- A001006 Motzkin 数 $M_n$
- 1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572, 853467
- 表示在圆上 $n$ 个点连接任意个不相交弦的方案数
- 也是 $(0,0)$ 走到 $(n,0)$ ，只能右/右上/右下走，不能走到 $x$ 轴下方的方案数
- $M_0=1,M_n=\tfrac{2n+1}{n+2}M_{n-1}+\tfrac{3n-3}{n+2}M_{n-2}$

***

- Eulerian 数 $\left\langle n\atop m\right\rangle$
- 表示 $1\ldots n$ 的排列，有 $m$ 个数比它前一个数大的方案数
- $\left\langle 1\atop 0\right\rangle=1,\left\langle n\atop m\right\rangle=(n-m)\left\langle n-1\atop m-1\right\rangle+(m+1)\left\langle n-1\atop m\right\rangle$
- $\sum_{m=0}^{n-1}\left\langle n\atop m\right\rangle=n!$

***

- Narayana 数 $N(n,k)$
- $N(n,k)=\dfrac 1 n\dbinom{n
}{k}\dbinom{n}{k-1}$
- $C_n=\sum\limits_{i=1}^nN(n,i)$
- 表示 $n$ 对匹配括号组成的字符串中有 $k$ 个 `()` 子串的方案数
- 表示 $(0,0)$ 走到 $(2n,0)$，只能右上/左下，有 $k$ 个波峰的方案数

***

- Delannoy 数 $D(m,n)$
- 表示 $(0,0)$ 走到 $(m,n)$，只能右/上/右上的方案数
- 递推公式即简单 dp
- $D(m,n)=\sum\limits_{k=0}^{\min(m,n)}{m+n-k\choose m}{m\choose k}$
- $D(m,n)=\sum\limits_{k=0}^{\min(m,n)}{m\choose k}{n\choose k}2^k$

***

- A001003 Hipparchus 数 / 小 Schroeder 数 $S(n)$
- 1, 1, 3, 11, 45, 197, 903, 4279, 20793, 103049, 518859, 2646723, 13648869, 71039373
- 表示 $(0,0)$ 走到 $(n,n)$，只能右/上/右上，不能沿 $y=x$ 走，且只能在 $y\le x$ 区域走的方案数
- $S(0)=S(1)=1,S(n)=\frac{(6n-3)S(n-1)-(n-2)S(n-2)}{n+1}$ `s[i]=((6*i-3)*s[i-1]-(i-2)*s[i-2])/(i+1)`
- 表示 $n+2$ 边形的多边形剖分数

***

- A000670 Fubini 数 $a(n)$
- 1, 1, 3, 13, 75, 541, 4683, 47293, 545835, 7087261, 102247563, 1622632573, 28091567595
- 表示 $n$ 个元素组成偏序集的个数
- $a(0)=1,a(n)=\sum_{k=1}^{n}\dbinom n k a(n-k)$

***

- A000111 Euler 数 $E(n)$
- 1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521, 353792, 2702765, 22368256, 199360981
- 其指数型生成函数为 $\dfrac{1}{\cos x}+\tan x$，前者提供偶数项 (A000364)，后者提供奇数项
- 表示满足 $x_1>x_2<x_3>x_4<\ldots x_n$ 的排列的方案数

```c++
vi calc(int n){
	n=polyn(n);
	return eachfac(conv(
		inv(cos(vi({0,1}),n),n), // 1/cos(x)
		sin(vi({0,1}),n), // sin(x)
		n,fxy((x*y+x)%mod)
	),n); // 1/cos(x)+tan(x)
}
```

## 多项式全家桶 vector 版

- vector 版常数很大，多项式开根中 $n=10^5$ 有 3 倍常数

```c++
ll D(ll x){return x>=mod?x-mod:x<0?x+mod:x;}
ll &ad(ll &x){return x=D(x);}
typedef vector<ll> vi;
#define rs(a) [&]{if((int)a.size()<n)a.resize(n,0);}()
#define cut(a) fill(a.begin()+n/2,a.begin()+n,0)
#define fxy(z) [&](ll x,ll y){return z;} // not appeared
int polyn(int n1){ // return 2^k >= n1
	return 1<<(31-__builtin_clz(n1-1)+1);
}
vi der(vi a,int n){ // b=da/dx
	rs(a);
	repeat(i,1,n)a[i-1]=i*a[i]%mod; a[n-1]=0;
	return a;
}
vi cal(vi a,int n){ // b=\int adx
	rs(a);
	repeat_back(i,1,n)a[i]=qpow(i,mod-2,mod)*a[i-1]%mod; a[0]=0;
	return a;
}
vi eachfac(vi a,int n){ // ans[i]=a[i]*i!
	ll p=1;
	repeat(i,1,n)p=p*i%mod,a[i]=a[i]*p%mod;
	return a;
}
vi readpoly(int n){
	vi a; repeat(i,0,n)a<<read();
	return a;
}
void print(vi a,int n){
	rs(a);
	repeat(i,0,n)printf("%lld%c",a[i]," \n"[i==n-1]);
}

// ********************************

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
				a[j]=D(u+t),a[j+len/2]=D(u-t);
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

// ********************************

vi inv(const vi &a,int n){ // n=2^k, ans=1/a
	if(n==1)return vi(1,qpow(a[0],mod-2,mod));
	return conv(inv(a,n/2),a,n,[](ll a,ll b){
		return a*(2-a*b%mod+mod)%mod;
	});
}
const int inv2=qpow(2,mod-2,mod);
vi sqrt(const vi &a,int n){ // n=2^k
	if(n==1)return vi(1,1); // vi(1,sqrtmod(a[0]));
	vi f=sqrt(a,n/2); rs(f);
	vi gg=conv(a,inv(f,n),n);
	repeat(i,0,n)f[i]=inv2*(gg[i]+f[i])%mod;
	return f;
}
vi ln(const vi &a,int n){ // n=2^k
	return cal(conv(der(a,n),inv(a,n),n),n);
}
vi exp(const vi &a,int n){ // n=2^k
	if(n==1)return vi(1,1);
	vi b=exp(a,n/2);
	vi lnb=ln(b,n);
	repeat(i,0,n)lnb[i]=D(a[i]-lnb[i]); lnb[0]++;
	return conv(b,lnb,n);
}
pair<vi,vi> divmod(vi a,const vi &b,int n){ // n=2^k, |fi|=n-m+1, |se|=m-1, fi*b+se=a
	int m=b.size(); rs(a);
	auto rev=[](vi a){
		reverse(a.begin(),a.end());
		return a;
	};
	vi d=conv(rev(a),inv(rev(b),n),n);
	d.resize(n-m+1); d=rev(d);
	vi r=conv(d,b,n); r.resize(m-1);
	repeat(i,0,m-1)r[i]=D(a[i]-r[i]);
	return {d,r};
}
const ll im=911660635; // im = sqrtmod(-1)
vi sin(vi a,int n){ // n=2^k
	rs(a);
	repeat(i,0,n)(a[i]*=im)%=mod;
	a=exp(a,n);
	auto b=inv(a,n);
	repeat(i,0,n)a[i]=D(a[i]-b[i])*inv2%mod*(mod-im)%mod;
	return a;
}
vi cos(vi a,int n){ // n=2^k
	rs(a);
	repeat(i,0,n)(a[i]*=im)%=mod;
	a=exp(a,n);
	auto b=inv(a,n);
	repeat(i,0,n)a[i]=(a[i]+b[i])*inv2%mod;
	return a;
}
vi tan(vi a,int n){ // n=2^k
	return conv(sin(a,n),inv(cos(a,n),n),n);
}
vi asin(vi a,int n){ // n=2^k
	vi d=der(a,n);
	a=conv(a,a,n,[](ll a,ll b){
		return D(1-a*b%mod);
	});
	return cal(conv(d,inv(sqrt(a,n),n),n),n);
}
vi acos(vi a,int n){ // n=2^k
	a=asin(a,n);
	repeat(i,0,n)a[i]=D(-a[i]);
	return a;
}
vi atan(vi a,int n){ // n=2^k
	vi d=der(a,n);
	a=conv(a,a,n,[](ll a,ll b){
		return D(1+a*b%mod);
	});
	return cal(conv(d,inv(a,n),n),n);
}
```

## 多项式快速幂

```c++
ll getmod(char s[],int mod){
	ll x=0;
	repeat(i,0,strlen(s))x=(x*10+s[i]-48)%mod;
	return x;
}
vi qpow_trivial(vi a,ll m,int n){ // n=2^k, a[0]=1
	a=ln(a,n);
	repeat(i,0,n)(a[i]*=m)%=mod;
	return exp(a,n);
}
vi qpow(vi a,char s[],int n){ // n=2^k
	rs(a); ll m=getmod(s,mod),m1=getmod(s,mod-1);
	ll l; for(l=0;l<n && a[l]==0;l++);
	if(l*m>=n)return vi();
	if(l && strlen(s)>=7)return vi(); 
	int in=qpow(a[l],mod-2,mod);
	int owe=qpow(a[l],m1,mod); 
	repeat(i,0,n-l)a[i]=a[i+l]*in%mod;
	a=qpow_trivial(a,m,n);
	l*=m;
	repeat_back(i,l,n)a[i]=a[i-l]*owe%mod;
	repeat(i,0,l)a[i]=0;
	return a;
}
```

## 多项式复合

- [link](https://www.luogu.com.cn/problem/P5050) 卡常失败

```c++
const int L=142; // sqrt(n1)
vi f,g[L+1],ng[L+1],G[L+1],nG[L+1];
void prework(vi g[],vi ng[],int n){
	n*=2; g[0]=vi(1,1); rs(g[0]),rs(g[1]);
	vi e=g[1]; ntt(e,n,1);
	repeat(i,1,L+1){
		rs(g[i]); ng[i-1]=g[i-1]; ntt(ng[i-1],n,1);
		repeat(j,0,n)g[i][j]=e[j]*ng[i-1][j]%mod;
		ntt(g[i],n,-1); cut(g[i]);
	}
}
void Solve(){
	int n1=read()+1,m1=read()+1,n=polyn(max(n1,m1));
	f=readpoly(n1);
	g[1]=readpoly(m1); prework(g,ng,n);
	G[1]=g[L]; prework(G,nG,n);
	vi ans(n,0);
	repeat(i,0,L){
		static vi s; s.assign(n*2,0);
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

## 多项式多点求值

- 已知多项式 $f$ 和序列 $a$，求 $f(a_1),f(a_2),\ldots,f(a_m)$
- 线性算法指输入 $n$ 维向量 $x$，经过 $m\times n$ 矩阵 $A$ 变换后输出 $m$ 维向量 $y=Ax$ 的算法
- 转置原理指出，如果存在 $x'=A^Ty'$ 的算法，那么就有存在相同复杂度的 $y=Ax$ 的算法。将 $A^T$ 分解为三种指令 `x[i]+=x[j],x[i]*=c,swap(x[i],x[i])`，那么 $A$ 即倒着执行这些指令，并且将第一种指令变为 `x[j]+=x[i]`。（$A=E_1E_2\ldots E_k\rightarrow A^T=E_k^TE_{k-1}^T\ldots E_1^T$）
- $O(n\log^2n)$，常数极大

```c++
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
		prod[x]={1,D(-a[l])};
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
	vi f=readpoly(n); // poly
	repeat(i,0,m)
		a[i]=read(); // query
	getprod(1,0,m-1);
	vi v=inv(prod[1],polyn(m+1));
	dfs(1,0,m-1,convtr(f,v));
	repeat(i,0,m)
		print(ans[i],1);
}
```

## 括号序列专题

- refer to [OI-Wiki](https://oi-wiki.org/topic/bracket/)
- 括号序列后继，假设 `"("` 字典序小于 `")"`

```c++
bool next_brastr(string &s) {
	int n=s.size(),dep=0;
	repeat_back(i,0,n){
		dep+=(s[i]==')')*2-1;
		if(s[i]=='(' && dep>0){
			dep--;
			int L=(n-i-1-dep)/2;
			int R=n-i-1-L;
			s.resize(i);
			s+=')'+string(L,'(')+string(R,')');
			return 1;
		}
	}
	return 0;
}
```

- 括号序列康托展开+逆
- A053121：设 $f(i,j)$ 表示长度为 $i$ 且存在 $j$ 个未匹配的右括号且不存在未匹配的左括号的括号序列的个数。
- $f(0,0)=1,f(i,j) = f(i-1,j-1)+f(i-1,j+1)$
- $f(n, m) = \dfrac{m+1}{n+1}\dbinom{n+1}{\frac{n-m}{2}}(\text{if } n-m \text{ is even}),0(\text{if } n-m \text{ is odd})$

```c++
ll f(int n,int m){
	if((n-m)%2==1)return 0;
	return C(n+1,(n-m)/2)*(m+1)%mod*qpow(n+1,mod-2)%mod;
}
ll order(string s){
	int n=s.size(),dep=0; ll ans=0;
	repeat(i,0,n){
		if(s[i]==')')ans+=f(n-i-1,dep+1);
		dep+=(s[i]=='(')*2-1;
	}
	return ans%mod;
}
string cantor(ll order,int n){ // 要去掉函数 f 的取模
	int dep=0; string s;
	repeat(i,0,n){
		s+='(';
		if(order>=f(n-i-1,dep+1)){
			s.back()=')';
			order-=f(n-i-1,dep+1);
		}
		dep+=(s[i]=='(')*2-1;
	}
	return s;
}
```

## 减一的杨辉矩阵

- [A014430](http://oeis.org/A014430)（但是下标的含义不太一样）
$$
\left[\begin{array}{c}
1 & 2 & 3 & 4 & 5 \\
2 & 5 & 9 & 14 & 20 \\
3 & 9 & 19 & 34 & 55 \\
4 & 14 & 34 & 69 &125 \\
5 & 20 & 55 & 125 & 251 \\
\end{array}\right]
$$
- 定义：$T(n,m)=\dbinom{n+m+2}{n+1}-1$
- 递推式：$T(n,k)=T(n-1,k)+T(n,k-1)+1, T(0,0)=1$
- 它是杨辉矩阵前缀和：$\displaystyle T(n,m)=\sum_{i=0}^n\sum_{j=0}^m\dbinom{i+j}{i}$

## 区间历史最值

- [模板题](https://www.luogu.com.cn/problem/P6242)，支持区间加、区间取min、区间和、区间max、区间历史max（$\max_{i\in[1,t]j\in [l,r]}h_i[j]$），$O(\log^2n)$

```c++
struct Segbeats { // init: build(1, 1, n, a)
	struct Node {
		int l, r;
		int mx, mxh, se, cnt;
		ll sum;
		int a1, a1h, a2, a2h;
	} a[N * 4];
#define lc (u << 1)
#define rc (u << 1 | 1)
	void up(int u) { // private
		a[u].sum = a[lc].sum + a[rc].sum;
		a[u].mxh = max(a[lc].mxh, a[rc].mxh);
		if (a[lc].mx == a[rc].mx) {
			a[u].mx = a[lc].mx;
			a[u].se = max(a[lc].se, a[rc].se);
			a[u].cnt = a[lc].cnt + a[rc].cnt;
		} else if (a[lc].mx > a[rc].mx) {
			a[u].mx = a[lc].mx;
			a[u].se = max(a[lc].se, a[rc].mx);
			a[u].cnt = a[lc].cnt;
		} else {
			a[u].mx = a[rc].mx;
			a[u].se = max(a[lc].mx, a[rc].se);
			a[u].cnt = a[rc].cnt;
		}
	}
	void update(int u, int k1, int k1h, int k2, int k2h) { // private
		a[u].sum += 1ll * k1 * a[u].cnt + 1ll * k2 * (a[u].r - a[u].l + 1 - a[u].cnt);
		a[u].mxh = max(a[u].mxh, a[u].mx + k1h);
		a[u].a1h = max(a[u].a1h, a[u].a1 + k1h);
		a[u].mx += k1, a[u].a1 += k1;
		a[u].a2h = max(a[u].a2h, a[u].a2 + k2h);
		if (a[u].se != -INF) a[u].se += k2;
		a[u].a2 += k2;
	}
	void down(int u) { // private
		int tmp = max(a[lc].mx, a[rc].mx);
		if (a[lc].mx == tmp)
			update(lc, a[u].a1, a[u].a1h, a[u].a2, a[u].a2h);
		else
			update(lc, a[u].a2, a[u].a2h, a[u].a2, a[u].a2h);
		if (a[rc].mx == tmp)
			update(rc, a[u].a1, a[u].a1h, a[u].a2, a[u].a2h);
		else
			update(rc, a[u].a2, a[u].a2h, a[u].a2, a[u].a2h);
		a[u].a1 = a[u].a1h = a[u].a2 = a[u].a2h = 0;
	}
	void build(int u, int l, int r, int in[]) {
		a[u].l = l, a[u].r = r;
		a[u].a1 = a[u].a1h = a[u].a2 = a[u].a2h = 0;
		if (l == r) {
			a[u].sum = a[u].mxh = a[u].mx = in[l];
			a[u].se = -INF, a[u].cnt = 1;
			return;
		}
		int mid = l + r >> 1;
		build(lc, l, mid, in);
		build(rc, mid + 1, r, in);
		up(u);
	}
	void add(int u, int x, int y, int k) {
		if (a[u].l > y || a[u].r < x) return;
		if (x <= a[u].l && a[u].r <= y) {
			update(u, k, k, k, k);
			return;
		}
		down(u);
		add(lc, x, y, k), add(rc, x, y, k);
		up(u);
	}
	void tomin(int u, int x, int y, int k) {
		if (a[u].l > y || a[u].r < x || k >= a[u].mx) return;
		if (x <= a[u].l && a[u].r <= y && k > a[u].se) {
			update(u, k - a[u].mx, k - a[u].mx, 0, 0);
			return;
		}
		down(u);
		tomin(lc, x, y, k), tomin(rc, x, y, k);
		up(u);
	}
	ll qsum(int u, int x, int y) {
		if (a[u].l > y || a[u].r < x) return 0;
		if (x <= a[u].l && a[u].r <= y) return a[u].sum;
		down(u);
		return qsum(lc, x, y) + qsum(rc, x, y);
	}
	int qmax(int u, int x, int y) {
		if (a[u].l > y || a[u].r < x) return -INF;
		if (x <= a[u].l && a[u].r <= y) return a[u].mx;
		down(u);
		return max(qmax(lc, x, y), qmax(rc, x, y));
	}
	int qhmax(int u, int x, int y) {
		if (a[u].l > y || a[u].r < x) return -INF;
		if (x <= a[u].l && a[u].r <= y) return a[u].mxh;
		down(u);
		return max(qhmax(lc, x, y), qhmax(rc, x, y));
	}
#undef lc
#undef rc
} sgt;
```

## K-D tree 模板更新

- K-D tree 可以维护多维空间的点集，用替罪羊树的方法保证复杂度。
- 建树、询问近邻参考第二段代码。
- [模板题](https://www.luogu.com.cn/problem/P4148)，支持在线在 $(x,y)$ 处插入值、查询二维区间和。
- 插入的复杂度为 $O(\log n)$。
- 二维区间查询最坏复杂度为 $O(n^{1-\tfrac 1 k})=O(\sqrt n)$。
- 询问近邻等很多骚操作的最坏复杂度为 $O(n)$，最好用别的算法替代。

```c++
struct kdt{
	#define U(x,y) ((x)+(y))
	#define U0 0
	struct range{
		int l,r;
		range operator|(range b)const{
			return {min(l,b.l),max(r,b.r)};
		}
		bool out(range b){
			return l>b.r || b.l>r;
		}
		bool in(range b){
			return b.l<=l && r<=b.r;
		}
	};
	struct node{
		int x,y,a; // (x,y): coordinate, a: value
		int s,d,sz; // s: sum of value in subtree, d: cut direction, sz: size of subtree
		range xx,yy; // xx/yy: range of coordinate x/y
		node *l,*r; // left/right child
		void up(){
			sz=l->sz+r->sz+1;
			s=U(a,U(l->s,r->s));
			xx=range{x,x} | l->xx | r->xx;
			yy=range{y,y} | l->yy | r->yy;
		}
		node *&ch(int px,int py){ // in which child
			if(d==0)return px<x ? l : r;
			else return py<y ? l : r;
		}
		node(){
			sz=0; a=s=U0;
			xx=yy={inf,-inf};
			l=r=Null;
		}
		node(int x_,int y_,int a_){
			x=x_,y=y_,a=a_;
			l=r=Null;
			up();
		}
	}*rt;
	static node Null[N],*pl;
	vector<node *> cache;
	void init(){ // while using kdtrees, notice Null is static
		rt=pl=Null;
	}
	node *build(node **l,node **r,int d){ // private
		if(l>=r)return Null;
		node **mid=l+(r-l)/2;
		if(d==0)
			nth_element(l,mid,r,[&](node *a,node *b){
				return a->x < b->x;
			});
		else
			nth_element(l,mid,r,[&](node *a,node *b){
				return a->y < b->y;
			});
		node *u=*mid;
		u->d=d;
		u->l=build(l,mid,d^1);
		u->r=build(mid+1,r,d^1);
		u->up();
		return u;
	}
	void pia(node *u){ // private
		if(u==Null)return;
		pia(u->l);
		cache.push_back(u);
		pia(u->r);
	}
	void insert(node *&u,int x,int y,int v){ // private
		if(u==Null){
			*++pl=node(x,y,v); u=pl; u->d=0;
			return;
		}
		insert(u->ch(x,y),x,y,v);
		u->up();
		if(0.725*u->sz <= max(u->l->sz,u->r->sz)){
			cache.clear();
			pia(u);
			u=build(cache.data(),cache.data()+cache.size(),u->d);
		}
	}
	void insert(int x,int y,int v){
		insert(rt,x,y,v);
	}
	range qx,qy;
	int query(node *u){ // private
		if(u==Null)return U0;
		if(u->xx.out(qx) || u->yy.out(qy))return U0;
		if(u->xx.in(qx) && u->yy.in(qy))return u->s;
		return U((range{u->x,u->x}.in(qx) && range{u->y,u->y}.in(qy) ? u->a : U0),
			U(query(u->l),query(u->r)));
	}
	int query(int x1,int y1,int x2,int y2){
		qx={x1,x2};
		qy={y1,y2};
		return query(rt);
	}
}tr;
kdt::node kdt::Null[N],*kdt::pl;
```

- [模板题](https://www.luogu.com.cn/problem/P1429)，支持询问近邻。

```c++
struct kdt{
	struct range{
		lf l,r;
		range operator|(range b)const{
			return {min(l,b.l),max(r,b.r)};
		}
		lf dist(lf x){
			return x<l ? l-x : x>r ? x-r : 0;
		}
	};
	struct node{
		lf x,y; // (x,y): coordinate
		int d,sz; // d: cut direction, sz: size of subtree
		range xx,yy; // xx/yy: range of coordinate x/y
		node *l,*r; // left/right child
		lf dist(lf x,lf y){
			return hypot(xx.dist(x),yy.dist(y));
		}
		void up(){
			sz=l->sz+r->sz+1;
			xx=range{x,x} | l->xx | r->xx;
			yy=range{y,y} | l->yy | r->yy;
		}
		node(){
			sz=0;
			xx=yy={inf,-inf};
			l=r=Null;
		}
		node(lf x_,lf y_){
			x=x_,y=y_;
			l=r=Null;
			up();
		}
	}*rt;
	static node Null[N],*pl;
	vector<node *> cache;
	void init(){ // while using kdtrees, notice Null is static
		rt=pl=Null;
	}
	node *build(node **l,node **r,int d){ // private
		if(l>=r)return Null;
		node **mid=l+(r-l)/2;
		if(d==0)
			nth_element(l,mid,r,[&](node *a,node *b){
				return a->x < b->x;
			});
		else
			nth_element(l,mid,r,[&](node *a,node *b){
				return a->y < b->y;
			});
		node *u=*mid;
		u->d=d;
		u->l=build(l,mid,d^1);
		u->r=build(mid+1,r,d^1);
		u->up();
		return u;
	}
	void build(pair<lf,lf> a[],int n){
		init(); cache.clear();
		repeat(i,0,n){
			*++pl=node(a[i].fi,a[i].se);
			cache.push_back(pl);
		}
		rt=build(cache.data(),cache.data()+cache.size(),0);
	}
	lf ans;
	void mindis(node *u,lf x,lf y,node *s){ // private
		if(u==Null)return;
		if(u!=s)ans=min(ans,hypot(x-u->x,y-u->y));
		lf d1=u->l->dist(x,y);
		lf d2=u->r->dist(x,y);
		if(d1<d2){ // optimize
			if(ans>d1)mindis(u->l,x,y,s);
			if(ans>d2)mindis(u->r,x,y,s);
		}
		else{
			if(ans>d2)mindis(u->r,x,y,s);
			if(ans>d1)mindis(u->l,x,y,s);
		}
	}
	lf mindis(lf x,lf y,node *s){ // min distance from (x,y), while delete node s
		ans=inf;
		mindis(rt,x,y,s);
		return ans;
	}
}tr;
kdt::node kdt::Null[N],*kdt::pl;
```

- 删除操作。

```c++
struct kdt{
	struct range{
		int l,r;
		range operator|(range b){
			return {min(l,b.l),max(r,b.r)};
		}
		int dist(int x){
			return x>0 ? r*x : l*x;
		}
	};
	struct node{
		int x,y; // (x,y): coordinate
		int d,sz,exsz,exist; // d: cut direction, sz: size of subtree
		range xx,yy; // xx/yy: range of coordinate x/y
		node *l,*r; // left/right child
		int dist(int x,int y){
			return xx.dist(x)+yy.dist(y);
		}
		node *&ch(int px,int py){ // in which child
			if(d==0)return make_pair(px,py)<make_pair(x,y) ? l : r;
			else return make_pair(py,px)<make_pair(y,x) ? l : r;
		}
		void up(){
			exsz=l->exsz+r->exsz+!!exist;
			sz=l->sz+r->sz+1;
			xx=range{x,x} | l->xx | r->xx;
			yy=range{y,y} | l->yy | r->yy;
		}
		node(){
			exist=exsz=sz=0;
			xx=yy={inf,-inf};
			l=r=Null;
		}
		node(int x_,int y_){
			exist=1;
			x=x_,y=y_;
			l=r=Null;
			up();
		}
	}*rt;
	static node Null[N],*pl;
	vector<node *> cache;
	void init(){ // while using kdtrees, notice Null is static
		rt=pl=Null;
	}
	node *build(node **l,node **r,int d){ // private
		if(l>=r)return Null;
		node **mid=l+(r-l)/2;
		if(d==0)
			nth_element(l,mid,r,[&](node *a,node *b){
				return make_pair(a->x,a->y) < make_pair(b->x,b->y);
			});
		else
			nth_element(l,mid,r,[&](node *a,node *b){
				return make_pair(a->y,a->x) < make_pair(b->y,b->x);
			});
		node *u=*mid;
		u->d=d;
		u->l=build(l,mid,d^1);
		u->r=build(mid+1,r,d^1);
		u->up();
		return u;
	}
	void pia(node *u){ // private
		if(u==Null)return;
		pia(u->l);
		if(u->exist)cache.push_back(u);
		pia(u->r);
	}
	void insert(node *&u,int x,int y){ // private
		if(u==Null){
			*++pl=node(x,y); u=pl; u->d=0;
			return;
		}
		if(u->x==x && u->y==y){
			u->exist++;
			return;
		}
		insert(u->ch(x,y),x,y);
		u->up();
		if(0.9*u->sz <= max(u->l->sz,u->r->sz) || 0.5*u->sz >= u->exsz){
			cache.clear();
			pia(u);
			u=build(cache.data(),cache.data()+cache.size(),u->d);
		}
	}
	void insert(int x,int y){
		insert(rt,x,y);
	}
	void erase(node *&u,int x,int y){
		if(u->x==x && u->y==y){
			u->exist--;
		}
		else{
			erase(u->ch(x,y),x,y);
		}
		u->up();
		if(0.9*u->sz <= max(u->l->sz,u->r->sz) || 0.5*u->sz >= u->exsz){
			cache.clear();
			pia(u);
			u=build(cache.data(),cache.data()+cache.size(),u->d);
		}
	}
	void erase(int x,int y){
		erase(rt,x,y);
	}
	int ans;
	void mindis(node *u,int x,int y){ // private
		if(u==Null)return;
		if(u->exist)ans=max(ans,x*u->x+y*u->y);
		int d1=u->l->dist(x,y);
		int d2=u->r->dist(x,y);
		if(d1>d2){ // optimize
			if(ans<d1 && u->l->sz)mindis(u->l,x,y);
			if(ans<d2 && u->r->sz)mindis(u->r,x,y);
		}
		else{
			if(ans<d2 && u->r->sz)mindis(u->r,x,y);
			if(ans<d1 && u->l->sz)mindis(u->l,x,y);
		}
	}
	int mindis(int x,int y){ // min distance from (x,y), while delete node s
		ans=-INF;
		mindis(rt,x,y);
		return ans;
	}
	void dfs(node *u){
		if(u==Null)return;
		printf("(%lld,%lld)[",u->x,u->y);
		dfs(u->l); printf(",");
		dfs(u->r); printf("]");
	}
	void print(){
		dfs(rt);
		puts("");
	}
}tr;
kdt::node kdt::Null[N],*kdt::pl;
```