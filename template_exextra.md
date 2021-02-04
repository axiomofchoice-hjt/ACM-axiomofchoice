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
	- [Voronoi图](#voronoi图)
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

# unclassified

## Notice

- 求 $\displaystyle B_i = \sum_{k=i}^n C_k^iA_k$，即 $\displaystyle B_i=\dfrac{1}{i!}\sum_{k=i}^n\dfrac{1}{(k-i)!}\cdot k!A_k$，反转后卷积
- `__builtin_expect(!!(exp),1),__builtin_expect(!!(exp),0)` 放在if()中，如果exp大概率为真/假则可以优化常数
- 多物网络流：$k$ 个源汇点，$S_i$ 需要流 $f_i$ 单位流量至 $T_i$。多物网络流只能用线性规划解决

## 费马-欧拉素数定理补充

- 对于 $2$ 有 $2=1^2+1^2$
- 对于模 $4$ 余 $1$ 的素数有费马-欧拉素数定理
- 对于完全平方数有 $x^2=x^2+0^2$
- 对于合数有 $(a^2+b^2)(c^2+d^2)=(ac+bd)^2+(ad-bc)^2$
- 对于无法用上述方式，即合数但是指数为奇数的素因子模 $4$ 余 $3$，不能分解为两整数平方和

## 卡常操作

```c++
int mul(int a,int b,int m=mod){ // 模乘
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
	- 若存在则路径为 $dfs$ 退出序（最后的序列还要再反过来）（如果for从小到大，可以得到最小字典序）
	- （不记录点的 $vis$，只记录边的 $vis$）
- 有向图
	- 欧拉回路存在当且仅当所有点入度等于出度
	- 欧拉路径存在当且仅当除了起点终点外所有点入度等于出度
	- 跑反图退出序
- 混合图
	- 欧拉回路存在当且仅当 `indeg+outdeg+undirdeg` 是偶数，且 `max(indeg,outdeg)*2<=indeg+outdeg+undirdeg`
	- 欧拉路径还没研究过
	- 无向边任意定向算法。每次找 `outdeg>indeg` 的点向 `outdeg<indeg` 的任意点连一条任意路径，要求只经过无向边，将路径上的边转换为有向边，欧拉路径存在条件仍满足。当所有点都有 `outdeg==indeg`，直接反图跑退出序

## Voronoi图

- Delaunay三角剖分后，每个线段中垂线构成Voronoi图
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
