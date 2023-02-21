# ExConclusion

- [ExConclusion](#exconclusion)
  - [计算几何](#计算几何)
    - [到给定点距离之和最小的直线](#到给定点距离之和最小的直线)
    - [点集同构](#点集同构)
    - [多边形构造](#多边形构造)
  - [数论](#数论)
    - [分母最小的分数](#分母最小的分数)
  - [组合数学](#组合数学)
    - [小众组合数学函数](#小众组合数学函数)
  - [博弈论](#博弈论)
    - [高维组合游戏 using Nim 积](#高维组合游戏-using-nim-积)
    - [不平等博弈 using 超现实数](#不平等博弈-using-超现实数)
    - [其他博弈结论](#其他博弈结论)
  - [图论](#图论)
    - [广义串并联图](#广义串并联图)
    - [最小 k 度限制生成树 次小生成树](#最小-k-度限制生成树-次小生成树)
    - [美术馆定理 using 多边形三角剖分](#美术馆定理-using-多边形三角剖分)
    - [边匹配](#边匹配)
    - [平面图 5-染色](#平面图-5-染色)
    - [网络流](#网络流)
    - [完全图欧拉回路](#完全图欧拉回路)
  - [其他](#其他)
    - [矩形里的小球](#矩形里的小球)
    - [整数分解为 2 的幂的方案数](#整数分解为-2-的幂的方案数)
    - [矩形孔明棋](#矩形孔明棋)
    - [矩形匹配 using 根号算法](#矩形匹配-using-根号算法)
    - [排序网络](#排序网络)

## 计算几何

### 到给定点距离之和最小的直线

- shrink 一下让点集没有三点共线
- 最优解一定是两点连线且把剩下点基本平分，因此可以绕直线上的某个点旋转直到碰到另外一点，然后绕新点旋转，重复上述操作，$O(n^2)$

参考：CTSC 04 金恺

### 点集同构

- 给两个点集，问能否通过平移、旋转、翻转、缩放操作后重合
- 求出质心，以质心到最远点的距离缩放，然后极角排序（第二关键字为距离），将二元组(极角差分,距离)列出来为 P, Q，求出 $PP$ 中是否有 Q 的出现即可
- 翻转其中一个点集后再做一遍。特判质心位置处有点的情况

### 多边形构造

- 给 n 个数，构造凸 n 边形使得边具有给定长度
- 若两倍最长边小于等于周长，那么一定可以构造圆内接 n 边形，对圆的半径二分即可
- 判断考虑圆心是否在多边形内的情况，即对半径为最长边的二分之一判断

```cpp
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

## 数论

### 分母最小的分数

- 求 $\dfrac a b<\dfrac p q<\dfrac c d$ 的分母最小的 $\dfrac p q$
- 若存在整数则直接解决。否则有 $0\le \dfrac{a \bmod b}{b}<\dfrac{p'}{q'}<\dfrac{c\bmod d}{d}<1$，$\dfrac{p'}{q'}+a/b=\dfrac p q$，再取倒数 $\dfrac{d}{c\bmod d}<\dfrac{q'}{p'}<\dfrac{b}{a\bmod b}$，如此反复

## 组合数学

### 小众组合数学函数

- 超级卡特兰数 $S_n$
- (0, 0) 走到 (n, n) 方案数，只能往右、上、右上走，且满足 $y\le x$
- $S_{0..10}=1, 2, 6, 22, 90, 394, 1806, 8558, 41586, 206098, 1037718$
- $\displaystyle S_n=S_{n-1}+\sum_{k=0}^{n-1}S_kS_{n-1-k}$
- $F_0=S_0,2F_i=S_i,F_n=\dfrac{(6n-3)F_{n-1}-(n-2)F_{n-2}}{n+1}$
- 通项公式 $\displaystyle S_n=\dfrac{1}{n}\sum_{k=1}^n2^kC_n^kC_n^{k-1},n\ge1$
- 若 $n\times n$ 矩阵 $A_{i,j}=S_{i+j-1}$，则 $\det A=2^{\tfrac{n(n+1)}{2}}$
- $S_n$ 为在 $n\times n$ 的矩形中选定 n 个点和 n 条水平/竖直的线段，满足每条线段恰好经过一个点且每个点恰好只被一条线段经过且线段直接不出现十字交叉，这 n 条线段把矩形划分成 $n+1$ 个小矩形的方案数

***

- A001006 Motzkin 数 $M_n$
- 1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572, 853467
- 表示在圆上 n 个点连接任意个不相交弦的方案数
- 也是 (0, 0) 走到 (n, 0) ，只能右/右上/右下走，不能走到 x 轴下方的方案数
- $M_0=1,M_n=\tfrac{2n+1}{n+2}M_{n-1}+\tfrac{3n-3}{n+2}M_{n-2}$

***

- Eulerian 数 $\left\langle n\atop m\right\rangle$
- 表示 $1\ldots n$ 的排列，有 m 个数比它前一个数大的方案数
- $\left\langle 1\atop 0\right\rangle=1,\left\langle n\atop m\right\rangle=(n-m)\left\langle n-1\atop m-1\right\rangle+(m+1)\left\langle n-1\atop m\right\rangle$
- $\sum_{m=0}^{n-1}\left\langle n\atop m\right\rangle=n!$

***

- Narayana 数 $N(n,k)$
- $N(n,k)=\dfrac 1 n\dbinom{n
}{k}\dbinom{n}{k-1}$
- $C_n=\sum\limits_{i=1}^nN(n,i)$（卡特兰数）
- 表示 n 对匹配括号组成的字符串中有 k 个 `()` 子串的方案数
- 表示 (0, 0) 走到 $(2n,0)$，只能右上/左下，有 k 个波峰的方案数

***

- Delannoy 数 $D(m,n)$
- 表示 (0, 0) 走到 (m, n)，只能右/上/右上的方案数
- 递推公式即简单 dp
- $D(m,n)=\sum\limits_{k=0}^{\min(m,n)}{m+n-k\choose m}{m\choose k}$
- $D(m,n)=\sum\limits_{k=0}^{\min(m,n)}{m\choose k}{n\choose k}2^k$

***

- A001003 Hipparchus 数 / 小 Schroeder 数 $S(n)$
- 1, 1, 3, 11, 45, 197, 903, 4279, 20793, 103049, 518859, 2646723, 13648869, 71039373
- 表示 (0, 0) 走到 (n, n)，只能右/上/右上，不能沿 $y=x$ 走，且只能在 $y\le x$ 区域走的方案数
- $S(0)=S(1)=1,S(n)=\frac{(6n-3)S(n-1)-(n-2)S(n-2)}{n+1}$ `s[i]=((6*i-3)*s[i-1]-(i-2)*s[i-2])/(i+1)`
- 表示 $n+2$ 边形的多边形剖分数

***

- A000670 Fubini 数 $a(n)$
- 1, 1, 3, 13, 75, 541, 4683, 47293, 545835, 7087261, 102247563, 1622632573, 28091567595
- 表示 n 个元素组成偏序集的个数
- $a(0)=1,a(n)=\sum_{k=1}^{n}\dbinom n k a(n-k)$

***

- A000111 Euler 数 $E(n)$
- 1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521, 353792, 2702765, 22368256, 199360981
- 其指数型生成函数为 $\dfrac{1}{\cos x}+\tan x$，前者提供偶数项 (A000364)，后者提供奇数项
- 表示满足 $x_1>x_2<x_3>x_4<\ldots x_n$ 的排列的方案数

```cpp
vi calc(int n){
    n=polyn(n);
    return eachfac(conv(
        inv(cos(vi({0,1}),n),n), // 1/cos(x)
        sin(vi({0,1}),n), // sin(x)
        n,fxy((x*y+x)%mod)
    ),n); // 1/cos(x)+tan(x)
}
```

***

- [A014430](http://oeis.org/A014430) 减一的杨辉矩阵（但是下标的含义不太一样）

$$\left[\begin{array}{c}1 & 2 & 3 & 4 & 5 \newline 2 & 5 & 9 & 14 & 20 \newline 3 & 9 & 19 & 34 & 55 \newline 4 & 14 & 34 & 69 &125 \newline 5 & 20 & 55 & 125 & 251 \newline \end{array}\right]$$

- 定义：$T(n,m)=\dbinom{n+m+2}{n+1}-1$
- 递推式：$T(n,k)=T(n-1,k)+T(n,k-1)+1, T(0,0)=1$
- 它是杨辉矩阵前缀和：$\displaystyle T(n,m)=\sum_{i=0}^n\sum_{j=0}^m\dbinom{i+j}{i}$

***

- 拉氏数 Lah

$$\begin{aligned}L(n,k)&=(-1)^n\dfrac{n!}{k!}\dbinom{n-1}{k-1}\newline L'(n,k)&=\dfrac{n!}{k!}\dbinom{n-1}{k-1}\end{aligned}$$

- 拉氏反演：（存疑）

$$\begin{aligned}a_n&=\sum_{k=0}^{n}L(n,k)b_k\newline b_n&=\sum_{k=0}^{n}L(n,k)a_k\end{aligned}$$

***

- 高斯系数

$$\dbinom{n}{k}_q=\dfrac{(q^n-1)(q^{n-1}-1)\ldots(q^{n-(m-1)}-1)}{(q^m-1)(q^{m-1}-1)\ldots(q-1)}$$

- 高斯系数反演：（存疑）

$$\begin{aligned}a_n&=\sum_{k=0}^{n}\dbinom{n}{k}_qb_k\newline b_n&=\sum_{k=0}^{n}(-1)^{n-k}q^{\tbinom{n-k}{2}}\dbinom{n}{k}_qa_k\end{aligned}$$

***

[A002137](http://oeis.org/A002137) $n\times n$ 对称矩阵的个数，对角为 0，行和为 2，$a_n = (n-1)(a_{n-1}+a_{n-2}) - \dfrac{1}{2}(n-1)(n-2)a_{n-3}$

## 博弈论

### 高维组合游戏 using Nim 积

- Nim 和与Nim 积的关系类似加法与乘法
- Tartan 定理：对于一个高维的游戏（多个维度的笛卡尔积），玩家的操作也是笛卡尔积的形式，那么对每一维度单独计算 SG 值，最终的 SG 值为它们的 Nim 积
- 比如，在 $n\times m$ 硬币中翻转 4 个硬币，4 个硬币构成一个矩形，这个矩形是每一维度（翻转两个硬币）的笛卡尔积
- $O(\log^2 n)$

```cpp
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
// int x=read(),y=read(),z=read();
// ans^=nim.f(SG(x),nim.f(SG(y),SG(z)));
```

### 不平等博弈 using 超现实数

- 超现实数(Surreal Number)
- 超现实数由左右集合构成，是最大的兼容四则运算的全序集合，包含实数集和“无穷大”
- 博弈局面的值可以看作左玩家比右玩家多进行的次数，独立的局面可以相加
- 如果值 $>0$ 则左玩家必胜，$<0$ 则右玩家必胜，$=0$ 则后手必胜
- 一个博弈局面，L 为左玩家操作一次后的博弈局面的最大值，R 为右玩家操作一次后的博弈局面的最小值，那么该博弈局面的值 $G=\dfrac A {2^B},L<G<R$，并且 B 尽可能小（$B=0$ 则 $|A|$ 尽可能小）
- 如果存在 $L=R$ 需要引入Irregular surreal number就不讨论了（比如两个玩家能进行同一操作即Nim）

***

- Blue-Red Hackenbush string
- 若干个 BW 串，player-W 只能拿 W，player-B 只能拿 B，每次拿走一个字符后其后缀也会消失，最先不能操作者输
- 对于每个串计算超现实数(Surreal Number)并求和，若 $> 0$ 则 W 必胜；若 $= 0$ 则后手必胜；若 $< 0$ 则 B 必胜

```cpp
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

```cpp
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
- 对于 W 点，先求所有儿子的值之和 x。如果 $x \ge 0$，那么直接加一即可。否则 x 变为 x 的小数部分加一，乘以 $2^{-\lfloor|x|\rfloor}$

***

- Alice's Game
- $x\times y$ 方格，如果 $x>1$ Alice可以水平切，如果 $y>1$ Bob可以垂直切，超现实数计算如下

```cpp
ll calc(int x,int y){ // get surreal number
    while(x>1 && y>1)x>>=1,y>>=1;
    return x-y;
}
```

### 其他博弈结论

欧几里得的游戏

- 两个数 a, b，每次对一个数删去另一个数的整数倍，出现 0 则失败
- $a\ge 2b$ 则先手必胜，否则递归处理

***

无向点地理问题 / Undirected vertex geography problem

- 二分图上移动棋子，不能经过重复点
- 先手必败当且仅当存在一个不包含起点的最大匹配

***

博弈论 Shannon 开关游戏

- refer to CTSC 07 刘雨辰
- 给无向图，玩家P可以在没有标记的边上标+号，玩家N可以在删除一条没有标记的边，轮流操作直到不能操作。若最终的图连通则玩家P获胜
- 玩家P后手必胜当且仅当存在两棵边独立的生成树
- 若玩家P获胜条件改为顶点 u, v 连通，则玩家P后手必胜当且仅当原图的一个包含 u, v 的导出子图存在两棵边独立的生成树

***

- 1 到 n，每次拿一个数或差值为 1 的两个数
  - 先手必胜，第一步拿最中间的 1 或 2 个数，之后对称操作
- $n\times m$ 棋盘上两个棋子，每次双方可以操控自己的棋子移动到同一行/列的位置，不能经过对方棋子所在行/列
  - 后手必胜当且仅当两个棋子的横坐标之差等于纵坐标之差
- 2 个数字，每次把一个数字减少，最小 1，但是不能出现重复数字
  - $\text{SG}(a,b)=((a-1)\oplus(b-1))-1$
- 3 个数字，每次把一个数字减少，最小 1，但是不能出现重复数字
  - 后手必胜当且仅当 $a\oplus b\oplus c=0$
- Octal Game 表明一些 Nim 游戏的 SG 值存在周期性（允许一些例外）

***

奇偶博弈（名字乱取的）

- n 堆石子，玩家可选若干堆石子，每堆取一个，不能不取。
- 先手必胜当且仅当存在奇数个石子的堆。

## 图论

### 广义串并联图

- 无 $K_4$ 子图的无向连通图称为广义串并联图。
- 简单广义串并联图有 $E\le 2V$。
- 广义串并联图可经过若干收缩操作变成一个顶点的图。
  - 删除度为 1 的点。
  - 删除重边并用权为两边较小值的边替换。
  - 删除度为 2 的点并用权为两边之和的边替换。（优先执行操作 2）
- 收缩操作可建立串并联树。
  - 初始每个点 v 和边 e 都对应叶子 $T_v,T_e$。
  - 对于删度为 1 的点 x，对应边为 $e=\langle x,y\rangle$，新建点 $T_z$ 作为 $T_e,T_x,T_y$ 的父亲并将 y 对应节点设为 $T_z$。
  - 删除重边 a, b，新建点 $T_c$ 作为 $T_a,T_b$ 的父亲并对应原图新加的边 c。
  - 对于删度为 2 的点 x，对应边为 a, b，新建点 $T_c$ 作为 $T_x,T_a,T_b$ 的父亲并对应原图新加的边 c。
- 非叶子节点 u
- 咕了

### 最小 k 度限制生成树 次小生成树

最小 k 度生成树

- 即某个点 $v_0$ 度不大于 k 的最小生成树
- 去掉 $v_0$ 跑一遍最小生成森林，然后从小到大访问 $v_0$ 的边 $(v_0,v)$ 考虑是否能加入边集
- 如果 v 与 $v_0$ 不连通就直接加，否则判断路径 $v_0 - v$ 上的最大边是否大于 $(v_0,v)$，大于就将它替换为 $(v_0,v)$
- 令 $Best(v)$ 为路径 $v_0-v$ 的最大边，每次树的形态改变后更新 $Best$
- 可以证明最优性，$O(E\log E+kV)$（不知道可不可以数据结构维护）

次小生成树

- 即所有生成树中第二小的生成树
- 跑一遍Kruskal，然后对剩下的边依次询问这条边两个端点的路径最长边，更新答案。树上倍增优化，$O(E\log E)$

### 美术馆定理 using 多边形三角剖分

- 多边形三角剖分：设 $A,B,C$ 为连续的三点，若所有其他顶点不在 $\triangle ABC$ 内，则将原多边形用线段 $AC$ 划分；否则必然存在 $\triangle ABC$ 内且离线段 $AC$ 最远的点 D，则将原多边形用线段 $BD$ 划分
- 美术馆定理：对任意 n 边形美术馆，一定可以放置 $\lfloor \dfrac n 3\rfloor$ 个守卫（守卫具有 $360\degree$ 视角）来看守整个美术馆
- 对美术馆进行三角剖分，并对所有顶点 3-染色，保证任意两条边有不同颜色（任意三角形的顶点有三种颜色）。对三种颜色的点集取最小的点集即可

### 边匹配

- 求无向图最大的相邻边二元组集合，两两二元组无公共边。
- $O(n)$

```cpp
vector<pii> a[N]; // a[][].second: edge id
bool vis[N],instk[N];
vector<pii> ans; // result, set of <edge id, edge id>
void push(int &x,int &y){
    if(x!=-1 && y!=-1)
        ans.push_back({x,y}),x=y=-1;
}
int dfs(int x){
    vis[x]=1; instk[x]=1;
    int r=-1;
    for(auto i:a[x]){
        int p=i.fi,e=i.se;
        if(instk[p])continue;
        if(!vis[p]){
            int t=dfs(p);
            push(e,t);
        }
        push(e,r);
        if(r==-1)r=e;
    }
    instk[x]=0;
    return r;
}
void solve(){
    repeat(i,0,n)if(!vis[i])dfs(i);
}
```

### 平面图 5-染色

- 平面图欧拉定理：
  - $|V|-|E|+|F|=2$
  - 给定连通简单平面图，若 $|V|≥3$，则 $|E|≤3|V|-6$
  - 可知平面图的边数为 $O(V)$
  - 补充：给定连通简单平面图，若 $|V|≥3$，则 $\exist v,deg(v)\le 5$
- 由 $\exist deg(v)\le 5$，找到度最小的点 u，递归地对剩下的图进行5-染色
- 然后考虑 u 的颜色，如果 u 的邻居中 5 种颜色没有都出现，就直接染色，否则考虑顺时针的 5 个邻居 $v_1,v_2,v_3,v_4,v_5$，考虑两个子图，与 $v_1$ 或 $v_3$ 颜色相同的点集的导出子图 $G_1$，与 $v_2$ 或 $v_4$ 颜色相同的点集的导出子图 $G_2$，则 $v_1,v_3$ 在 $G_1$ 中不连通、$v_2,v_4$ 在 $G_2$ 中不连通两个命题必然有一个成立（否则出现两个相互嵌套的环，不构成平面图），假设 $v_1,v_3$ 在 $G_1$ 中不连通，那么将 $G_1$ 中 $v_1$ 的连通分量的所有顶点颜色取反（$color[v_1]$ 与 $color[v_2]$ 互换），这样 u 的邻居变为 4 种颜色，将 u 染色为原来的 $color[v_1]$
- 如果顺时针的关系很难得到，就尝试两次（$[v_1,v_2,v_3,v_4],[v_1,v_3,v_2,v_4]$）

### 网络流

- 多物网络流：k 个源汇点，$S_i$ 需要流 $f_i$ 单位流量至 $T_i$。多物网络流只能用线性规划解决。

### 完全图欧拉回路

- 可以得到任意连续 n − 2 个点都两两不重复的完全图欧拉回路（n 是奇数）。

```cpp
vector<int> euler(int n) {
    vector<int> ans = {n - 1};
    repeat (i, 0, n / 2) {
        int sgn = 1, ct = i;
        repeat (d, 1, n) {
            ans.push_back(ct);
            ct = (ct + sgn * d + n - 1) % (n - 1);
            sgn *= -1;
        }
        ans.push_back(n - 1);
    }
    return ans;
}
```

## 其他

### 矩形里的小球

- $(n+1)\times (m+1)$ 矩形中，小球从左下顶点往上 $a(a=0..n)$ 格的位置向右上发射，在矩形边界处反弹，回到起点后停止。问有几个格子经过了奇数次（起点只记一次）
- 情况1：撞到角上完全反弹，$\gcd(n,m)\mid a$（包含 $a=0$），答案为 2（角和起点）
- 情况2：没有撞到角上，$\gcd(n,m)\nmid a$，答案为 $\dfrac{2(nm-n(m/g-1)-m(n/g-1))}{g}$（经过的格子数 $\tfrac {2nm}{g}$ 减去经过两次的格子数）（这种情况最多经过两次）

参考：CTSC 03 姜尚仆

### 整数分解为 2 的幂的方案数

- 即 A018819 $[1, 1, 2, 2, 4, 4, 6, 6, 10, 10, 14, 14, ...]$
- 递推式 $a_{2m+1} = a_{2m}, a_{2m} = a_{2m-1} + a_{m}$
- 用矩阵乘法可以加速，$O(\log^4 n)$

```cpp
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

### 矩形孔明棋

- 无限大棋盘上有 $n\times m$ 棋子，移动方法同孔明棋，求最小剩下的棋子数

```cpp
scanf("%d%d",&n,&m);
if(m==1 || n==1)cout<<(n+m)/2<<endl;
else if(n%3==0 || m%3==0)cout<<2<<endl;
else cout<<1<<endl;
```

### 矩形匹配 using 根号算法

- refer to CTSC 08 day2 张煜承
- 平面上 n 个点，以任意 4 个顶点组成四条边平行于坐标轴的矩形，求这样的矩形数
- 若第 i 行的点数 $>k$，直接处理这一行的贡献后删除该行。处理方式为，先将第 i 行的列号处理为一个集合，统计第 j 行里出现在集合的点数 $=h(j)$，$\displaystyle\sum_j {h(j)\choose 2}$ 即为贡献。操作最多 $\dfrac n k$ 行，因此复杂度 $O(\dfrac{n^2}{k})$
- 剩下的点中每行点数 $\le k$，考虑对每行的点集两两匹配，暴力统计 $(x_1,y)(x_2,y)$ 中 y 的个数 $=h(x_1,x_2)$，$\displaystyle\sum_{x_1,x_2}{h(x_1,x_2)\choose 2}$ 即为贡献。操作最少 $\dfrac n k$ 行，因此复杂度 $O(nk)$

### 排序网络

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
