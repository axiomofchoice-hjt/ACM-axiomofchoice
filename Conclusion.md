# Conclusion

- [计算几何](#计算几何)
  - [空间变换矩阵](#空间变换矩阵)
  - [三角形四心一点](#三角形四心一点)
  - [二维欧几里得算法](#二维欧几里得算法)
  - [正幂反演](#正幂反演)
  - [曼哈顿距离 切比雪夫距离](#曼哈顿距离-切比雪夫距离)
  - [Pick 定理](#pick-定理)
  - [圆的面积并](#圆的面积并)
  - [圆的扫描线](#圆的扫描线)
- [球面几何](#球面几何)
  - [几何公式](#几何公式)
  - [计算几何 结论](#计算几何-结论)
- [数据结构](#数据结构)
  - [矩形加，离线矩形和](#矩形加离线矩形和)
  - [数据结构 归档](#数据结构-归档)
- [数论](#数论)
  - [反素数](#反素数)
  - [高斯整数](#高斯整数)
  - [二次剩余](#二次剩余)
  - [数论分块](#数论分块)
  - [莫比乌斯反演](#莫比乌斯反演)
  - [斐波那契数列](#斐波那契数列)
  - [佩尔方程 / Pell](#佩尔方程--pell)
  - [费马-欧拉素数定理补充](#费马-欧拉素数定理补充)
  - [素数表 or 质数表](#素数表-or-质数表)
  - [数论 归档](#数论-归档)
- [组合数学](#组合数学)
  - [组合数学函数](#组合数学函数)
    - [组合数](#组合数)
    - [卡特兰数 Catalan](#卡特兰数-catalan)
    - [贝尔数 Bell](#贝尔数-bell)
    - [错排数](#错排数)
    - [斯特林数 Stirling](#斯特林数-stirling)
    - [伯努利数](#伯努利数)
  - [球盒模型](#球盒模型)
  - [康托展开 + 逆 and 编码与解码](#康托展开--逆-and-编码与解码)
  - [置换群计数](#置换群计数)
  - [杨表 / Young tableaux](#杨表--young-tableaux)
  - [网格路径计数](#网格路径计数)
  - [组合数学 归档](#组合数学-归档)
- [代数系统](#代数系统)
  - [循环矩阵理论](#循环矩阵理论)
  - [带状矩阵高斯消元](#带状矩阵高斯消元)
  - [矩阵 归档](#矩阵-归档)
  - [带通配符的字符串匹配 using FFT](#带通配符的字符串匹配-using-fft)
  - [多项式 归档](#多项式-归档)
- [博弈论](#博弈论)
- [数学的其他操作](#数学的其他操作)
  - [主定理 / Master Theorem](#主定理--master-theorem)
  - [约瑟夫问题](#约瑟夫问题)
  - [格雷码 / Gray Code](#格雷码--gray-code)
  - [汉诺塔](#汉诺塔)
  - [Stern-Brocot 树 and Farey 序列](#stern-brocot-树-and-farey-序列)
  - [浮点与近似计算](#浮点与近似计算)
  - [日期换算](#日期换算)
  - [数学 结论](#数学-结论)
- [图论](#图论)
  - [图论的一些概念](#图论的一些概念)
  - [欧拉图 using 套圈算法](#欧拉图-using-套圈算法)
  - [DFS 树 and BFS 树](#dfs-树-and-bfs-树)
  - [最小环](#最小环)
  - [差分约束](#差分约束)
  - [同余最短路](#同余最短路)
  - [最小树形图 using 朱刘算法](#最小树形图-using-朱刘算法)
  - [绝对中心 and 最小直径生成树 / MDST](#绝对中心-and-最小直径生成树--mdst)
  - [弦图 and 区间图](#弦图-and-区间图)
  - [树、图的哈希](#树图的哈希)
  - [二分图 归档](#二分图-归档)
  - [网络流 归档](#网络流-归档)
  - [矩阵树定理](#矩阵树定理)
  - [Prufer 序列](#prufer-序列)
  - [LGV 引理](#lgv-引理)
  - [图论 with 组合数学 归档](#图论-with-组合数学-归档)
  - [图论 结论](#图论-结论)
- [其他](#其他)
  - [异或字典树](#异或字典树)
  - [分散层叠 / Fractional Cascading](#分散层叠--fractional-cascading)
  - [Raney 引理](#raney-引理)
  - [括号序列专题](#括号序列专题)
  - [其他 结论](#其他-结论)

## 计算几何

### 空间变换矩阵

- 绕原点逆时针旋转 $\theta$ 弧度：

$$\left[\begin{array}{cc}    \cos\theta & -\sin\theta & 0 \newline     \sin\theta & \cos\theta & 0 \newline     0 & 0 & 1\end{array}\right]\left[\begin{array}{c} x \newline  y \newline  1 \end{array}\right]$$

- 绕 $(x_0,y_0)$ 逆时针旋转 $\theta$ 弧度：

$$\left[\begin{array}{cc}    \cos\theta & -\sin\theta & -x_0\cos\theta+y_0\sin\theta+x_0 \newline     \sin\theta & \cos\theta & -x_0\sin\theta-y_0\cos\theta+y_0 \newline     0 & 0 & 1\end{array}\right]\left[\begin{array}{c} x \newline  y \newline  1 \end{array}\right]$$

- 平移 $(D_x,D_y)$：

$$\left[\begin{array}{cc}    1 & 0 & D_x \newline     0 & 1 & D_y \newline     0 & 0 & 1\end{array}\right]\left[\begin{array}{c} x \newline  y \newline  1 \end{array}\right]$$

### 三角形四心一点

- 三角形重心到三个顶点平方和最小，到三边距离之积最大（三角形内）
- 三角形四心一点（未测试）

```cpp
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

### 二维欧几里得算法

- 已知二维向量 a, b，求 $|ax+by|$ 的最小值 $(x,y\in \mathbb{Z})$
- 若 $a\cdot b<0$，则将 b 反向
- 若 $\cos\langle a,b\rangle<\dfrac 1 2$，则答案为 $\min(|a|,|b|)$
- 若 $\cos\langle a,b\rangle\ge\dfrac 1 2$，由于 $ans(a,b)=ans(a,b+a)$，假设 $|a|<|b|$，过 b 作 a 的垂线交于，若 $ka$ 和 $(k+1)a$ 在垂线两侧 $(k\ge 0)$，则 $\langle a,b-ka\rangle$ 和 $\langle -a,b-(k+1)a\rangle$ 中选取一个夹角更大的替换 a, b，如此反复

### 正幂反演

- 给定反演中心 O 和反演半径 R。若直线上的点 $OPQ$ 满足 $|OP|\cdot|OQ|=R^2$，则 P 和 Q 互为反演点（令 $R=1$ 也可）
- 不经过反演中心的圆的反演图形是圆（计算时取圆上靠近/远离中心的两个点）
- 经过反演中心的圆的反演图形是直线（计算时取远离中心的点，做垂线）

### 曼哈顿距离 切比雪夫距离

- 曼：`mdist=|x1-x2|+|y1-y2|`
- 切：`cdist=max(|x1-x2|,|y1-y2|)`
- 转换：
  - `mdist((x,y),*)=cdist((x+y,x-y),**)`
  - `cdist((x,y),*)=mdist(((x+y)/2,(x-y)/2),**)`
- 高维：$|\Delta x|+|\Delta y|+|\Delta z|=\max_{f_x,f_y,f_z=\pm 1}(f_x\Delta x+f_y\Delta y+f_z\Delta z)$

### Pick 定理

- 可以用Pick定理求多边形内部整点个数，其中一条线段上的点数为 $\gcd(|x_1-x_2|,|y_1-y_2|)+1$
- 正方形点阵：`面积 = 内部点数 + 边上点数 / 2 - 1`
- 三角形点阵：`面积 = 2 * 内部点数 + 边上点数 - 2`

### 圆的面积并

- 格林公式
- empty

### 圆的扫描线

- empty

## 球面几何

```cpp
vec to_vec(lf lng,lf lat){ // lng经度，lat纬度，-90<lat<90
    lng*=pi/180,lat*=pi/180;
    lf z=sin(lat),m=cos(lat);
    lf x=cos(lng)*m,y=sin(lng)*m;
    return vec(x,y,z);
};
lf to_lng(vec v){return atan2(v.y,v.x)*180/pi;}
lf to_lat(vec v){return asin(v.z)*180/pi;}
lf angle(vec a,vec b){return acos(dot(a,b));}
```

### 几何公式

- 三角形面积 $S=\sqrt{P(P-a)(P-b)(P-c)}$，P 为半周长。

```cpp
lf getArea(lf a, lf b, lf c) {
    lf p = (a + b + c) / 2;
    return sqrt(p * (p - a) * (p - b) * (p - c));
}
```

- 斯特瓦尔特定理：$BC$ 上一点 P，有 $AP=\sqrt{AB^2\cdot \dfrac{CP}{BC}+AC^2\cdot \dfrac{BP}{BC}-BP\cdot CP}$
- 三角形内切圆半径 $r=\dfrac {2S} C$，外接圆半径 $R=\dfrac{a}{2\sin A}=\dfrac{abc}{4S}$
- 四边形有 $a^2+b^2+c^2+d^2=D_1^2+D_2^2+4M^2$，$D_1,D_2$ 为对角线，M 为对角线中点连线
- 圆内接四边形有 $ac+bd=D_1D_2$，$S=\sqrt{(P-a)(P-b)(P-c)(P-d)}$，P 为半周长
- 棱台体积 $V=\dfrac 13(S_1+S_2+\sqrt{S_1S_2})h$，$S_1,S_2$ 为上下底面积
- 正棱台侧面积 $\dfrac 1 2(C_1+C_2)L$，$C_1,C_2$ 为上下底周长，L 为斜高（上下底对应的平行边的距离）
- 球全面积 $S=4\pi r^2$，体积 $V=\dfrac 43\pi r^3$，
- 球台(球在平行平面之间的部分)有 $h=|\sqrt{r^2-r_1^2}\pm\sqrt{r^2-r_2^2}|$，侧面积 $S=2\pi r h$，体积 $V=\dfrac{1}{6}\pi h[3(r_1^2+r_2^2)+h^2]$，$r_1,r_2$ 为上下底面半径
- 正三角形面积 $S=\dfrac{\sqrt 3}{4}a^2$，正四面体面积 $S=\dfrac{\sqrt 2}{12}a^3$
- 四面体体积公式

```cpp
lf sqr(lf x){return x*x;}
lf V(lf a,lf b,lf c,lf d,lf e,lf f){ // a,b,c共顶点
    lf A=b*b+c*c-d*d;
    lf B=a*a+c*c-e*e;
    lf C=a*a+b*b-f*f;
    return sqrt(4*sqr(a*b*c)-sqr(a*A)-sqr(b*B)-sqr(c*C)+A*B*C)/12;
}
```

### 计算几何 结论

- Minkowski 和：两个凸包 A, B，定义它们的 Minkowski 和为 $\{a+b\mid a \in A,b\in B\}$。
  - 将 A, B 的有向边放在一起极角排序，顺序连接，得到答案的形状。再取几个点确定位置。

## 数据结构

### 矩形加，离线矩形和

扫描线，对于矩形加 `(x1, y1) - (x2, y2) + v`：

```cpp
e.push_back({x1, -v, y1, y2, 0});
e.push_back({x2 + 1, v, y1, y2, 0});
```

对于矩形和 `ans[id]: (x1, y1) - (x2, y2)`：

```cpp
e.push_back({x1 - 1, inf - 1, y1, y2, id});
e.push_back({x2, inf, y1, y2, id});
```

扫描线时：

```cpp
int l = i[2], r = i[3];
if (i[1] == inf - 1) {
    ans[i[4]] -= (i[0] * ti.query(l, r) + t.query(l, r))%mod;
} else if (i[1] == inf) {
    ans[i[4]] += (i[0] * ti.query(l, r) + t.query(l, r))%mod;
} else {
    ti.add(l, r, i[1]);
    t.add(l, r, -(i[0] - 1) * i[1] % mod);
}
```

### 数据结构 归档

- 区间 mex（最小没出现的自然数）：回滚莫队
- 区间绝对众数（出现次数的两倍大于区间长度）：线段树上摩尔投票，求出的结果需要验证。

```cpp
struct Node {
    int m, cnt;
    Node operator+(const Node &b) const {
        if (m == b.m) return {m, cnt + b.cnt};
        if (cnt > o.cnt) return {m, cnt - b.cnt};
        return {b.m, b.cnt - cnt};
    }
};
```

- 对于重复出现视为出现一次的题（如区间不同数字个数），可以令 `pre[i]` 表示最大的 j 满足 `a[j]=a[i],j<i`

双头优先队列可以用 multiset。

区间众数：离线用莫队，在线用分块。

支持插入、查询中位数可以用双堆。

```cpp
struct Median {
    priority_queue<ll> h1; // 大根堆
    priority_queue<ll, vector<ll>, greater<ll>> h2; // 小根堆
    void push(ll x) {
        #define maintain(h1, h2, b) { h1.push(x); if (h1.size() > h2.size() + b) h2.push(h1.top()), h1.pop(); }
        if (h1.empty() || h1.top() > x) maintain(h1, h2, 1)
        else maintain(h2, h1, 0);
    }
    ll middle() { return h1.top(); } // size() 为奇数时可以这么写，偶数看题目定义
    ll size() { return h1.size() + h2.size(); }
};
```

双关键字堆可以用两个 multiset 模拟。

```cpp
struct Heap{
    multiset<pii> a[2];
    void init(){a[0].clear(); a[1].clear();}
    pii rev(pii x){return {x.second,x.first};}
    void push(pii x){
        a[0].insert(x);
        a[1].insert(rev(x));
    }
    pii top(int p){
        pii t=*--a[p].end();
        return p?rev(t):t;
    }
    void pop(int p){
        auto t=--a[p].end();
        a[p^1].erase(a[p^1].lower_bound(rev(*t)));
        a[p].erase(t);
    }
};
```

高维前缀和

- 以二维为例，t 是维数。
- 法一 $O(n^t2^t)$。
- 法二 $O(n^tt)$。

```cpp
// <1>
for(int i=1;i<=n;i++)
for(int j=1;j<=m;j++)
    b[i][j]=b[i-1][j]+b[i][j-1]-b[i-1][j-1]+a[i][j];
// <2>
for(int i=1;i<=n;i++)
for(int j=1;j<=m;j++)
    a[i][j]+=a[i][j-1];
for(int i=1;i<=n;i++)
for(int j=1;j<=m;j++)
    a[i][j]+=a[i-1][j];
```

一个 01 串，支持把某位置的 1 改成 0，查询某位置之后第一个 1 的位置，可以用并查集。（删除 `d[x]=d[x+1]`，查询 `d[x]`）

手写 deque 很可能比 STL deque 慢。（吸氧时）

## 数论

### 反素数

- 求因数最多的数（因数个数一样则取最小）
- 性质：$M = {p_1}^{k_1}{p_2}^{k_2}...$ 其中，$p_i$ 是从 2 开始的连续质数，$k_i-k_{i+1}∈\{0,1\}$
- 先打出质数表再 DFS，枚举 $k_n$，$O(\exp)$

```cpp
int pri[16]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
ll n; // 范围
pair<ll,ll> ans; // ans是结果，ans.fi是最大反素数，ans.se是反素数约数个数
void dfs(ll num=1,ll cnt=1,int *p=pri,int pre=inf){ // 注意ans要初始化
    if(make_pair(cnt,-num)>make_pair(ans.se,-ans.fi))
        ans={num,cnt};
    num*=*p;
    for(int i=1;i<=pre && num<=n;i++,num*=*p)
        dfs(num,p+1,i,cnt*(i+1));
}
```

- n 以内约数个数最大值是 $O(n^{\tfrac {1.066}{\ln\ln n}})$。（用 $O(\sqrt n)$ 来估计太不准确了）

|      范围      |  1e4  |  1e5  |  1e6   |    1e9    |       1e16       |
| :------------: | :---: | :---: | :----: | :-------: | :--------------: |
|   最大反素数   | 7560  | 83160 | 720720 | 735134400 | 8086598962041600 |
| 反素数约数个数 |  64   |  128  |  240   |   1344    |      41472       |

- 前 n 个质数乘积为 $O(\exp(n\log n))$。

### 高斯整数

- 高斯整数：$\{a+bi\ |\ a,b∈\mathbb{Z}\}$。
- 高斯素数：无法分解为两个高斯整数 $\not∈\{\pm1,\pm i\}$ 之积的高斯整数。
- $a+bi$ 是高斯素数当前仅当：
  - a, b 一个为 0，另一个绝对值为 $4k+3$ 型素数。
  - $a^2+b^2$ 为 $4k+1$ 型素数或 2。
- 带余除法：

```cpp
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

### 二次剩余

- 对于奇素数模数 p，存在 $\frac {p-1} 2$ 个二次剩余 $\{1^2,2^2,...,(\frac {p-1} 2)^2\}$，和相同数量的二次非剩余。
- 对于奇素数模数 p，如果 $n^{\frac{p-1}2}\equiv1\pmod{p}$ ，则 n 是一个二次剩余；如果 $n^{\frac{p-1}2}\equiv-1\pmod{p}$，则 n 是一个二次非剩余。
- 对于奇素数模数 p，二次剩余的乘积是二次剩余，二次剩余与非二次剩余乘积为非二次剩余，非二次剩余乘积是二次剩余。
- 费马-欧拉素数定理：$(4n+1)$ 型素数只能用一种方法表示为一个范数（两个完全平方数之和），$(4n+3)$ 型素数不能表示为一个范数。
- 二次互反律：记 $p^{\frac{q-1}2}$ 的符号为 $(\dfrac p q)$ ，则对奇素数 p, q 有 $(\dfrac p q)\cdot(\dfrac q p)=(-1)^{\tfrac{p-1}2\cdot\tfrac{q-1}2}$。
- 求二次剩余，要求 mod 是质数，sqrtmod() 返回其中一个 sqrt，另一个为 mod 减这个返回值（如果 mod=2 就没有第二个）；返回 $-1$ 表示无解。

```cpp
ll ww;
struct vec{ // x+y*sqrt(w)
    ll x,y;
    vec operator*(vec b){
        return {(x*b.x+y*b.y%mod*ww)%mod,(x*b.y+y*b.x)%mod};
    }
};
vec qpow(vec a,ll b){
    vec ans={1,0};
    for(;b;a=a*a,b>>=1)
        if(b&1)ans=ans*a;
    return ans;
}
ll leg(ll a){return qpow(a,(mod-1)>>1,mod)!=mod-1;}
ll sqrtmod(ll b){
    // if(b==0)return 0;
    if(mod==2)return 1;
    if(!leg(b))return -1;
    ll a;
    do{a=rnd()%mod; ww=(a*a-b+mod)%mod;}while(leg(ww));
    ll ans=qpow(vec{a,1},(mod+1)>>1).x;
    return min(ans,mod-ans);
}
```

### 数论分块

- $n=(k-n\bmod k)(n/k)+(n\bmod k)(n/k+1)$
- 将 $\lfloor \dfrac{n}{x}\rfloor=C$ 的 $[x_{\min},x_{\max}]$ 作为一块，其中区间内的任一整数 $x_0$ 满足 $x_{\max}=n/(n/x_0)$。

```cpp
for(int l=l0,r;l<=r0;l=r+1){
    r=min(r0,n/(n/l));
    // c=n/l;
    // len=r-l+1;
}
```

- 将 $\lceil \dfrac{n}{x}\rceil=C$ 的 $[x_{\min},x_{\max}]$ 作为一块：

```cpp
for(int l=l0,r;l<=r0;l=r+1){
    r=min(r0,n/(n/l)); if(n%r==0)r=max(r-1,l);
    // c=(n+l-1)/l;
    // len=l-r+1;
}
```

### 莫比乌斯反演

- 引理1：$\lfloor \dfrac{a}{bc}\rfloor=\lfloor \dfrac{\lfloor \dfrac{a}{b}\rfloor}{c}\rfloor$；引理2：n 的因数个数 $≤\lfloor 2\sqrt n \rfloor$。
- 狄利克雷卷积：$\displaystyle(f*g)(n)=\sum_{d|n}f(d)g(\dfrac n d)$，有交换律、结合律、对加法的分配律。
- 积性函数：满足 $\gcd(n,m)=1\Rightarrow f(nm)=f(n)f(m)$ 的函数。
- 单位函数：$\varepsilon(n)=[n=1]$ 为狄利克雷卷积的单位元。
- 恒等函数：$id(n)=n$。
- 约数个数：$d(n)=1*1$。
- 约数之和：$\sigma(n)=1*id$。
- 莫比乌斯函数性质：$\mu(n)=\begin{cases} 1&n=1\newline 0&n含有平方因子\newline (-1)^k&k为n的质因数个数\end{cases}$。
- 结论：$(\forall f)(f*\varepsilon=f),\mu*1=\varepsilon,\varphi*1=id,d*\mu=id$。
- 莫比乌斯反演：若 $f=g*1$，则 $g=f*\mu$；或者，若 $\displaystyle f(n)=\sum_{d|n}g(d)$，则 $\displaystyle g(n)=\sum_{d\mid n}\mu(d)f(\dfrac n d)$。

***

- 例题：求模意义下的 $\displaystyle\sum_{i=1}^n \sum_{j=1}^m \text{lcm}(i,j)$。

$$\begin{aligned}&\sum_{i=1}^n \sum_{j=1}^m \dfrac{i\cdot j}{\gcd(i,j)}\newline =&\sum_{i=1}^n\sum_{j=1}^m\sum_{d|i,d|j,\gcd(\frac i d,\frac j d)=1}\dfrac{i\cdot j}{d}\newline =&\sum_{d=1}^n d\cdot\sum_{i=1}^{\lfloor\frac nd\rfloor}\sum_{j=1}^{\lfloor\frac md\rfloor}[\gcd(i,j)=1]i\cdot j\newline &(\text{let }n'=\lfloor\dfrac{n}{d}\rfloor,m'=\lfloor\dfrac{m}{d}\rfloor)\newline =&\sum_{d=1}^n d\cdot\sum_{i=1}^{n'}\sum_{j=1}^{m'}[\gcd(i,j)=1]i\cdot j\newline =&\sum_{d=1}^n d\cdot\sum_{i=1}^{n'}\sum_{j=1}^{m'}\sum_{c|i,c|j}{\mu(c)}\cdot i\cdot j\newline &(\text{let }i'=\dfrac i c,j'=\dfrac j c)\newline =&\sum_{d=1}^n d\cdot\sum_{c=1}^{\max(n',m')}\mu(c)\cdot c^2\cdot\sum_{i'=1}^{\lfloor\frac {n'}c\rfloor}\sum_{j'=1}^{\lfloor\frac {m'}c\rfloor} i'\cdot j'\newline =&\sum_{d=1}^n d\cdot\sum_{c=1}^{\max(n',m')}\mu(c)\cdot c^2\cdot\dfrac 1 4 \lfloor\frac {n'}c\rfloor(\lfloor\frac {n'}c\rfloor+1) \lfloor\frac {m'}c\rfloor(\lfloor\frac {m'}c\rfloor+1)\end{aligned}$$

### 斐波那契数列

- 递推式：$F_0=0,F_1=1,F_n=F_{n-1}+F_{n-2}$
- 通项公式：$F_n=\dfrac 1 {\sqrt{5}} [(\dfrac{1+\sqrt 5}2)^n-(\dfrac{1-\sqrt 5}2)^n]$ （公式中若 5 是二次剩余则可以化简，比如 $\sqrt 5\equiv 383008016\pmod {1000000009}$）
- $F_{a+b-1}=F_{a-1}F_{b-1}+F_aF_b$
- 卡西尼性质：$F_{n-1}F_{n+1}-F_n^2=(-1)^n$
- $F_{n}^2+F_{n+1}^2=F_{2n+1}$
- $F_{n+1}^2-F_{n-1}^2=F_{2n}$（由上一条写两遍相减得到）
- $a_0=1,a_n=a_{n-1}+a_{n-3}+a_{n-5}+...(n\ge 1)$，则 $a_n=F_n(n\ge 1)$
- 齐肯多夫定理：任何正整数都可以表示成若干个不连续的斐波那契数之和（$F_2$ 开始）可以用贪心实现。

求和公式

- 奇数项求和：$F_1+F_3+F_5+...+F_{2n-1}=F_{2n}$
- 偶数项求和：$F_2+F_4+F_6+...+F_{2n}=F_{2n+1}-1$
- 平方和：$F_1^2+F_2^2+F_3^2+...+F_n^2=F_nF_{n+1}$
- $F_1+2F_2+3F_3+...+nF_n=nF_{n+2}-F_{n+3}+2$
- $-F_1+F_2-F_3+...+(-1)^nF_n=(-1)^n(F_{n+1}-F_n)+1$
- $F_{2n-2m-2}(F_{2n}+F_{2n+2})=F_{2m+2}+F_{4n-2m}$

数论性质

- $F_a \mid F_b \Leftrightarrow a \mid b$
- $\gcd(F_a,F_b)=F_{\gcd(a,b)}$
- 当 p 为 $5k\pm 1$ 型素数时，$\begin{cases} F_{p-1}\equiv 0\pmod p \newline  F_p\equiv 1\pmod p \newline  F_{p+1}\equiv 1\pmod p \end{cases}$
- 当 p 为 $5k\pm 2$ 型素数时，$\begin{cases} F_{p-1}\equiv 1\pmod p \newline  F_p\equiv -1\pmod p \newline  F_{p+1}\equiv 0\pmod p \end{cases}$
- $F_{n+2}$ 为集合 `{1,2,3,...,n-2}` 中不包含相邻正整数的子集个数（包括空集）。
- `F(n)%m` 的周期 $\le 6m$（$m=2\times 5^k$ 取等号）
- 既是斐波那契数又是平方数的有且仅有 $1,144$。
- $\gcd({a^x-1},{a^y-1})=a^{\gcd(x,y)}-1$（虽然和斐波那契没关系）

快速倍增法求$F_n$，返回二元组$(F_n,F_{n+1})$ ，$O(\log n)$

```cpp
pii fib(ll n){ // fib(n).fi即结果
    if(n==0)return {0,1};
    pii p=fib(n>>1);
    ll a=p.fi,b=p.se;
    ll c=a*(2*b-a)%mod;
    ll d=(a*a+b*b)%mod;
    if(n&1)return {d,(c+d)%mod};
    else return {c,d};
}
```

### 佩尔方程 / Pell

- $x^2-dy^2=1$，d 是正整数。
- 若 d 是完全平方数，只有平凡解 $(\pm 1,0)$，其余情况总有非平凡解。
- 若最小正整数解 $(x_1,y_1)$，则递推公式：

$$\begin{cases}x_n=x_1x_{n-1}+dy_1y_{n-1}\newline y_n=y_1x_{n-1}+x_1y_{n-1}\end{cases}\newline \left[\begin{array}{c}x_n\newline y_n\end{array}\right]=\left[\begin{array}{cc}x_1 & dy_1\newline y_1 & x_1\end{array}\right]\left[\begin{array}{c}x_{n-1}\newline y_{n-1}\end{array}\right]$$

- 最小解：（可能溢出）

```cpp
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

### 费马-欧拉素数定理补充

- 对于 2 有 $2=1^2+1^2$
- 对于模 4 余 1 的素数有费马-欧拉素数定理
- 对于完全平方数有 $x^2=x^2+0^2$
- 对于合数有 $(a^2+b^2)(c^2+d^2)=(ac+bd)^2+(ad-bc)^2$
- 对于无法用上述方式，即存在模 4 余 3 的、指数为奇数的素因子，不能分解为两整数平方和
- （本质上是一个整数分解为高斯素数的过程）
- 令 $\chi[1^+]=1,0,-1,0,1,0,-1\ldots$，是一个完全积性函数。正整数 n 分解为两个整数平方和的方案数为 n 所有约数 $\chi$ 值之和，$f(n)=\sum_{d\mid n}\chi(d)$

### 素数表 or 质数表

42737, 46411, 50101, 52627, 54577, 191677, 194869, 210407, 221831, 241337, 578603, 625409, 713569, 788813, 862481, 2174729, 2326673, 2688877, 2779417, 3133583, 4489747, 6697841, 6791471, 6878533, 7883129, 9124553, 10415371, 11134633, 12214801, 15589333, 17148757, 17997457, 20278487, 27256133, 28678757, 38206199, 41337119, 47422547, 48543479, 52834961, 76993291, 85852231, 95217823, 108755593, 132972461, 171863609, 173629837, 176939899, 207808351, 227218703, 306112619, 311809637, 322711981, 330806107, 345593317, 345887293, 362838523, 373523729, 394207349, 409580177, 437359931, 483577261, 490845269, 512059357, 534387017, 698987533, 764016151, 906097321, 914067307, 954169327

- 1572869, 3145739, 6291469, 12582917, 25165843, 50331653 （适合哈希的素数）
- 19260817   原根15，是某个很好用的质数
- 1000000007 原根5
- 998244353  原根3
- NTT素数表， g 是模 $(r \cdot 2^k+1)$ 的原根

```cpp
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

### 数论 归档

雅可比四平方和定理：设 $a^2+b^2+c^2+d^2=n$ 的整数解个数为 $S(n)$，有

$$S(2^k m)=\begin{cases}8\text{d}(m) & \text{if }k=0\newline 24\text{d}(m) & \text{if }k>0\end{cases}(m\text{ is odd})$$

（$\text{d}(n)$ 为 n 的约数和）

***

欧拉反演

$$\sum_{i=1}^n\gcd(i,n)=\sum_{d\mid n}\tfrac{n}{d}\varphi(d)$$

***

$$\text{lcm}_{i=1}^n i\approx(e^n)$$

***

- $f(i)=2^i-\sum_{d\mid i,d\not=i}f(d)$，则有 $f(i)=\sum_{d\mid i}\mu(d)2^{i/d}$，其前缀和可杜教筛。

***

拓展欧拉定理

$$a^b \equiv \begin{cases}a^{b \bmod \phi(p)} & \gcd(a,p)=1 \newline a^{b} & \gcd(a,p)\neq 1,b<\phi(p) \newline a^{(b \bmod \phi(p))+\phi(p)} & \gcd(a,p)\neq 1,b\geq \phi(p)\end{cases}\pmod p$$

## 组合数学

### 组合数学函数

#### 组合数

- 递推式 `C(n,k)=(n-k+1)*C(n,k-1)/k​`
- 二维数组预处理：

```cpp
repeat (i, 0, n) {
    C[i][0] = 1; if (i < m) C[i][i] = 1;
    repeat (j, 1, min(i, m))
        C[i][j] = (C[i - 1][j] + C[i - 1][j - 1]) % mod;
}
```

- 组合数前缀和 多组询问
- 令 $\displaystyle S(n,m)=\sum_{i=0}^{m}C(n,i)$
- 则 $S(n,m+1)=S(n,m)+C(n,m+1),S(n+1,m)=2S(n,m)-C(n,m)$
- 可以莫队

```cpp
struct MO{
    int n,m,ans;
    MO(){n=m=0; ans=1;}
    int query(int n1,int m1){
        while(m<m1){
            (ans+=C(n,m+1))%=mod;
            m++;
        }
        while(m>m1){
            m--;
            (ans-=C(n,m+1))%=mod;
        }
        while(n<n1){
            (ans=ans*2-C(n,m))%=mod;
            n++;
        }
        while(n>n1){
            n--;
            static const int inv2=qpow(2,mod-2);
            (ans=(ans+C(n,m))*inv2)%=mod;
        }
        return ans;
    }
};
```

- 二项式反演
  - $\displaystyle f_n=\sum_{i=0}^n{n\choose i}g_i\Leftrightarrow g_n=\sum_{i=0}^n(-1)^{n-i}{n\choose i}f_i$
  - $\displaystyle f_k=\sum_{i=k}^n{i\choose k}g_i\Leftrightarrow g_k=\sum_{i=k}^n(-1)^{i-k}{i\choose k}f_i$
- $\displaystyle \sum_{i=1}^{n}i{n\choose i}=n 2^{n-1}$
- $\displaystyle \sum_{i=1}^{n}i^2{n\choose i}=n(n+1) 2^{n-2}$
- $\displaystyle \sum_{i=1}^{n}\dfrac{1}{i}{n\choose i}=\sum_{i=1}^{n}\dfrac{1}{i}$
- $\displaystyle \sum_{i=0}^{n}{n\choose i}^2={2n\choose n}$

#### 卡特兰数 Catalan

- $C_n=\dfrac{\binom{2n}n}{n+1}$，$C_n=\dfrac{C_{n-1}(4n-2)}{n+1}$
- 有 $2\nmid C_n\rightarrow n=2^k-1$
- Hankel 矩阵：$n\times n$ 矩阵 $A_{i,j}=C_{i+j-2}$，有 $\det A=1$，$B_{i,j}=C_{i+j-1}$，也有 $\det B=1$。反过来可以用 A, B 定义卡特兰数

#### 贝尔数 Bell

- 划分n个元素的集合的方案数
- 有 $\displaystyle\sum_{n=0}^{\infty}B_n\dfrac{x^n}{n!}=e^{e^x-1}$

```cpp
B[0]=B[1]=1;
repeat(i,2,N){
    B[i]=0;
    repeat(j,0,i)
        B[i]=(B[i]+C(i-1,j)*B[j]%mod)%mod;
}
```

#### 错排数

- $D_n=n![\dfrac 1{0!}-\dfrac 1{1!}+\dfrac 1{2!}-...+\dfrac{(-1)^n}{n!}]$

```cpp
D[0]=1;
repeat(i,0,N-1){
    D[i+1]=D[i]+(i&1?C.inv[i+1]:mod-C.inv[i+1]);
    D[i]=1ll*D[i]*fac[i]%mod;
}
```

#### 斯特林数 Stirling

第一类

- 多项式 $x(x-1)(x-2) \cdots (x-n+1)$ 展开后 $x^r$ 的系数绝对值记作 $s(n,r)$ （系数符号 $(-1)^{n+r}$）
- 也可以表示 n 个元素分成 r 个环的方案数
- 递推式 $s(n,r) = (n-1)s(n-1,r)+s(n-1,r-1)$
- $\displaystyle n!=\sum_{i=0}^n s(n,i)$
- $\displaystyle A_x^n=\sum_{i=0}^n s(n,i)(-1)^{n-i}x^i$
- $\displaystyle A_{x+n-1}^n=\sum_{i=0}^n s(n,i)x^i$

第二类

- n 个不同的球放入 r 个相同的盒子且无空盒的方案数，记作 $S(n,r)$ 或 $S_n^r$
- 递推式 $S(n,r) = r S(n-1,r) + S(n-1,r-1)$

```cpp
s2[0][0]=1;
repeat(i,1,N){
    repeat(j,1,i+1)
        s2[i][j]=(j*s2[i-1][j]+s2[i-1][j-1])%mod;
}
```

- 通项公式 $\displaystyle S(n,r)=\frac{1}{r!}\sum_{i=0}^r(-1)^i{r\choose i}(r-i)^n$
- $\displaystyle m^n=\sum_{i=0}^mS(n,i)A_m^i$
- $\displaystyle \sum_{i=1}^n i^k=\sum_{i=0}^kS(k,i)i!{n+1\choose i+1}$

斯特林反演

- $\displaystyle f(n)=\sum_{i=1}^n S(n,i)g(i)\Leftrightarrow g(n)=\sum_{i=0}^n(-1)^{n-i}s(n,i)f(i)$

#### 伯努利数

- [A027642](http://oeis.org/A027642) 伯努利数 $B_i$（oeis 仅列了分母）
- $[1,-\dfrac 1 2,\dfrac 1 6,0,\dfrac 1 {30}]$
- 定义：$\displaystyle\sum_{i=0}^{n}B_i\dbinom{n+1}{i}=0,B_0=1$
- EGF $=\dfrac{x}{e^x-1}$
- 自然数幂和：$\displaystyle\sum_{i=0}^{n-1}i^k=\dfrac{1}{k+1}\sum_{i=0}^k\dbinom{k+1}{i}B_in^{k+1-i}$
- 伯努利反演：

$$\begin{aligned}a_n & =\sum_{k=0}^{n}\dbinom{n}{k}(n-k+1)^{-1}b_k\newline b_n & =\sum_{k=0}^{n}\dbinom{n}{k}B_{n-k}a_k\end{aligned}$$

- EGF 生成伯努利数：

```cpp
fill(a,a+n+1,1);
eachfacinv(a,n+1);
repeat(i,0,n)a[i]=a[i+1];
inv(a,n,b);
eachfac(b,n);
```

暴力板子

```cpp
struct Bern {
    int B[N];
    Bern() {
        B[0] = 1;
        repeat (i, 1, N) {
            B[i] = 0;
            repeat (j, 0, i) B[i] = (B[i] - 1ll * C(i + 1, j) * B[j]) % mod;
            B[i] = 1ll * (B[i] + mod) * qpow(C(i + 1, i), mod - 2, mod) % mod;
        }
    }
    int calc(int k, int n) { // 1 ** k + 2 ** k + ... + n ** k
        int ans = 0;
        repeat (i, 0, k + 1)
            ans = (ans + 1ll * C(k + 1, i) * B[i] % mod * qpow(n + 1, k + 1 - i, mod)) % mod;
        return 1ll * ans * qpow(k + 1, mod - 2, mod) % mod;
    }
} B;
```

### 球盒模型

假设有 n 个球、m 个盒子。

|                  |  球同盒同   |       球同盒异       |         球异盒同         |   球异盒异   |
| :--------------: | :---------: | :------------------: | :----------------------: | :----------: |
|       可空       |  高斯系数   |  $C_{n+m-1}^{m-1}$   | $\displaystyle\sum_{i=1}^{m}S_2(n,i)$ |    $m^n$     |
|       非空       |  高斯系数   |   $C_{n-1}^{m-1}$    |        $S_2(n,m)$        | $m!S_2(n,m)$ |
| 可空容量有上限 k | 高斯系数(1) |       容斥(2)        |            ?             |      ?       |
|     有下限 k     |  高斯系数   | $C_{n+m-kn-1}^{m-1}$ |            ?             |    DP(3)     |

(1) 高斯系数

$$[x^n] \prod_{i=1}^{m} \dfrac{1 - x^{i + k}}{1 - x^i}$$

(2) 容斥，枚举超出限制的盒子数

$$\sum_{i=0}^m(-1)^i\dbinom{m}{i}\dbinom{n-i(k+1)+m-1}{m-1}$$

(3) DP，`dp[i][j]` 表示前 i 个盒子放了 j 个球的方案数，$O(m^2k^2)$。

### 康托展开 + 逆 and 编码与解码

康托展开 + 逆

- 康托展开即排列到整数的映射
- 排列里的元素都是从 1 到 n

```cpp
// 普通版，O(n^2)
int cantor(int a[],int n){
    int f=1,ans=1; // 假设答案最小值是1
    repeat_back(i,0,n){
        int cnt=0;
        repeat(j,i+1,n)cnt+=a[j]<a[i];
        ans=(ans+f*cnt%mod)%mod; // ans+=f*cnt;
        f=f*(n-i)%mod; // f*=(n-i);
    }
    return ans;
}
// 树状数组优化版，基于树状数组，O(nlogn)
int cantor(int a[],int n){
    static BIT t; t.init(); // 树状数组
    ll f=1,ans=1; // 假设答案最小值是1
    repeat_back(i,0,n){
        ans=(ans+f*t.sum(a[i])%mod)%mod; // ans+=f*t.sum(a[i]);
        t.add(a[i],1);
        f=f*(n-i)%mod; // f*=(n-i);
    }
    return ans;
}
// 逆展开普通版，O(n^2)
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
        repeat(i,0,q)it++; // 第q+1小的数
        ans[i]=*it;
        s.erase(it);
    }
    return ans;
}
```

编码与解码问题

编码

- 给定一个字符串，求出它的编号
- 例，输入 acab，输出 5（aabc, aacb, abac, abca, acab, ...）
- 用递归，令 d(S) 是小于 S 的排列数，f(S) 是 S 的全排列数
- 小于 acab 的第一个字母只能是 a，所以 d(acab) = d(cab)
- 第二个字母是 a, b, c，所以 d(acab) = f(bc) + f(ac) + d(ab)
- d(ab) = 0
- 因此 d(acab) = 4，加 1 之后就是答案

解码

- 给定编号求字符串，对每一位进行尝试即可

### 置换群计数

Polya定理

- 例：立方体 $n=6$ 个面，每个面染上 $m=3$ 种颜色中的一种
- 两个染色方案相同意味着两个立方体经过旋转可以重合
- 其染色方案数为：$\dfrac{\sum m^{k_i}}{|k|}$（$k_i$ 为某一置换可以拆分的循环置换数，$|k|$ 为所有置换数）

```text
不旋转，{U|D|L|R|F|B}，k=6，共1个
对面中心连线为轴的90度旋转，{U|D|L R F B}，k=3，共6个
对面中心连线为轴的180度旋转，{U|D|L R|F B}，k=4，共3个
对棱中点连线为轴的180度旋转，{U L|D R|F B}，k=3，共6个
对顶点连线为轴的120度旋转，{U L F|D R B}，k=2，共8个
```

- 因此 $\dfrac{3^6+3^3 \cdot 6+3^4 \cdot 3+3^3 \cdot 6+3^2 \cdot 8}{1+6+3+6+8}=57$
- 例题（poj1286），n个点连成环，染3种颜色，允许旋转和翻转

```cpp
ll ans=0,cnt=0;
// 只考虑旋转，不考虑翻转
repeat(i,1,n+1)
    ans+=qpow(m,__gcd(i,n));
cnt+=n;
// 考虑翻转
if(n%2==0)ans+=(qpow(m,n/2+1)+qpow(m,n/2))*(n/2)%mod;
else ans+=qpow(m,(n+1)/2)*n%mod;
cnt+=n;
cout<<ans%mod*qpow(cnt,mod-2)%mod<<endl;
```

### 杨表 / Young tableaux

- 杨图：令 $\lambda = (\lambda_1,\lambda_2,\ldots,\lambda_m)$ 满足 $\lambda_1\ge\lambda_2\ge\ldots\lambda_m\ge 1,n=\sum \lambda_i$。一个形状为 $\lambda$ 的杨图是一个表格，第 i 行有 $\lambda_i$ 个方格，其坐标分别为 $(i,1)(i,2)\ldots(i,\lambda_i)$。
- 半标准杨表：将杨图填上数字，满足每行数字单调不减，每列数字单调递增。
- 标准杨表：将 $1,2,\ldots,n$ 填入杨图，满足每行、每列数字单调递增。下图为 $n=9,\lambda=(4,2,2,1)$ 的杨图和标准杨表。

$$\left[\begin{array}{c}* & * & * & * \newline * & * \newline * & * \newline *\end{array}\right]\left[\begin{array}{c}1 & 4 & 7 & 8 \newline 2 & 5 \newline 3 & 9 \newline 6\end{array}\right]$$

- 斜杨图：令 $\lambda = (\lambda_1,\lambda_2,\ldots,\lambda_m),\mu=(\mu_1,\mu_2,\ldots,\mu_{m'})$，则形状为 $\lambda/\mu$ 的斜杨图为杨图 $\lambda$ 中扣去杨图 $\mu$ 后剩下的部分。

***

- 插入操作：从第一行开始，在当前行中找最小的比 x 大的数字 y (upperbound)，交换 x, y，转到下一行继续操作；若所有数字比 x 小则把 x 放在该行末尾并退出
- 排列与两个标准杨表一一对应：将排列按顺序插入到杨表A中，并在杨表B中对应位置记录下标
- 对合排列和标准杨表一一对应（对合排列意味着自己乘自己是单位元）
- 将排列插入到杨表中，若比较运算反过来（小于变大于等于），得到的杨图（杨表的形状）和原来的杨图是转置关系
- Dilworth 定理：把一个数列划分成最少的最长不升子序列的数目就等于这个数列的最长上升子序列的长度。可知 k 个不相交的不下降子序列的长度之和最大值等于最长的 ( 最长下降子序列长度不超过 k ) 的子序列长度
- 序列生成的杨图前 k 行方格数即 k 个不相交的不下降子序列的长度之和最大值。但是不能用杨图求出这 k 个 LIS
- 第一行为最长上升序列长度，第一列为最长下降序列长度，可得指定 LIS 和 LDS 长度的排列数为 $\displaystyle\sum_{\lambda_1=\alpha,m=\beta} f_\lambda^2$，可由钩子公式计算 $f_\lambda$

***

- n 个元素的标准杨表个数
  - A000085：$[1,1,2,4,10,26,76,232,764,2620,9496,\ldots]$
  - $f(n)=f(n-1)+(n-1)f(n-2), f(0)=f(1)=1$
- 钩子公式：勾长 $h_{\lambda}(x)$ 定义为正右方方格数 + 正下方方格数 + 1。给一个杨图 $\lambda$，其标准杨表个数为：

$$f_{\lambda}=\dfrac{n!}{\prod h_{\lambda}(x)}=n!\dfrac{\prod_{1\le i<j\le m}(\lambda_i-i-\lambda_j+j)}{\prod_{i=1}^{m}(\lambda_i+m-i)!}$$

```cpp
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

```cpp
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

```cpp
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

- [双杨表维护 kLIS](https://www.luogu.com.cn/problem/P3774)：支持末尾插入一个数，询问 k 个不相交的不下降子序列的长度之和最大值。两个杨表可以在 $O(\sqrt n \log n)$（应该跑不满）内维护整个杨表的插入

```cpp
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

- 杨图随机游走：初始随机出现在杨图任一位置（每个位置概率 $\tfrac 1 n$），然后往右或往下走（每个位置概率 $\tfrac 1 {h_\lambda(x)}$），则走到边角 (r, s) 概率为

$$\dfrac 1 n\prod_{i=1}^{r-1}\dfrac{h_\lambda(i,s)}{h_\lambda(i,s)-1}\prod_{j=1}^{s-1}\dfrac{h_\lambda(r,j)}{h_\lambda(r,j)-1}$$

- 杨图带权随机游走：每行权重 $x_i$，每列权重 $y_j$，初始随机出现在杨图某一位置（概率权重 $x_iy_j$），向下走到某位置的概率权重为目标行的权重，向右为列的权重，则走到边角 (r, s) 概率为

$$\dfrac{x_ry_s}{\sum x_iy_j}\prod_{i=1}^{r-1}\left(1+\dfrac{x_i}{\sum x_{i+1..r}+\sum y_{s+1..\lambda_i}}\right)\prod_{j=1}^{s-1}\left(1+\dfrac{y_j}{\sum x_{r+1..\lambda^T_j}+\sum y_{j+1..s}}\right)$$

- 斜半标准杨表计数：

$$f'_{\lambda/\mu}=\det\left[\dbinom{\lambda_j-j-\mu_i+i+z-1}{\lambda_j-j-\mu_i+i}\right]_{i,j=1}^m$$

- 斜标准杨表计数：

$$f_{\lambda/\mu}=(\sum_{i=1}^{m}(\lambda_i-\mu_i))!\det\left[\dfrac{1}{(\lambda_j-j-\mu_i+i)!}\right]_{i,j=1}^m$$

- 列数不超过 $2k$ 的，元素都在 $[1, n]$ 内的且每行大小为偶数的半标准杨表和长度均为 $2n + 2$ 的 k-Dyck Path 形成双射关系，且计数公式如下：

$$b_{n,k}=\prod_{1\le i\le j\le n}\dfrac{2k+i+j}{i+j}$$

- 表内数为 $[1, m]$ 的半标准杨表计数：（定义 $i > n$ 时 $\lambda_i = 0$）

$$f'_\lambda=\prod_{i,j\in\lambda}\dfrac{n+j-i}{h_\lambda(i,j)}=\prod_{1\le i<j\le m}\dfrac{\lambda_i-i-\lambda_j+j}{j-i}$$

参考：IOI 19 袁方舟

### 网格路径计数

- 从 (0, 0) 走到 (a, b)，每次只能从 (x, y) 走到 $(x+1,y-1)$ 或 $(x+1,y+1)$，方案数记为 $f(a,b)=\dbinom{a}{\tfrac{a+b}{2}}$
- 若路径和直线 $y=k,k\notin [0,b]$ 不能有交点，则方案数为 $f(a,b)-f(a,2k-b)$
- 若路径和两条直线 $y=k_1,y=k_2,k_1<0\le b<k_2$ 不能有交点，方案数记为 $g(a,b,k_1,k_2)$，必须碰到 $y=k_1$ 不能碰到 $y=k_2$ 的方案数记为 $h(a,b,k_1,k_2)$，可递归求解（递归过程中两条直线距离会越来越大），$O(n)$

```cpp
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

- 从 (0, 0) 走到 (a, 0)，只能右上/右下，必须有恰好一次传送（向下 b 单位），不能走到 x 轴下方，方案数为 $\dbinom{a+1}{\frac{a-b}{2}+k+1}$

refer to [博客](https://www.luogu.com.cn/blog/wolfind/yi-suo-ge-lu-jing-ji-shuo-wen-ti)

### 组合数学 归档

- a 个相同的球放入 b 个不同的盒子，方案数为 $C_{a+b-1}^{b-1}$（隔板法）

***

- 一个长为 $n+m$ 的数组，n 个 1，m 个 $-1$，限制前缀和最大为 k，则方案数为 $C_{n+m}^{m+k}-C_{n+m}^{m+k+1}$

***

- $2n$ 个带标号的点两两匹配，方案数为 $(2n-1)!!=\dfrac{(2n)!}{2^n n!}$

***

- $1,2,...,n$ 中无序地选择 r 个互不相同且互不相邻的数字，则这 r 个数字之积对所有方案求和的结果为 $C_{n+1}^{2r}(2r-1)!!=\dfrac{C_{n+1}^{2r}(2r)!}{2^rr!}$（问题可以转换为，$(n+1)$ 个点无序匹配 r 对点的方案数）

```cpp
int M(int a,int b){
    static const int inv2=qpow(2,mod-2);
    return C(a+1,2*b)*C.fac[2*b]%mod*qpow(inv2,b)%mod*C.inv[b]%mod;
}
```

***

- 范德蒙德卷积公式：$\displaystyle{\sum_{k}\binom{r}{k}\binom{s}{n-k}=\binom{r+s}{n}}$。

***

- 拉格朗日恒等式

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}(a_ib_j-a_jb_i)^2=(\sum_{i=1}^{n}a_i)^2(\sum_{i=1}^{n}b_i)^2-(\sum_{i=1}^{n}a_ib_i)^2$$

## 代数系统

### 循环矩阵理论

- n 阶循环矩阵形式如下：

$$A=\left[\begin{array}{c}a_0&a_1&a_2&\cdots&a_{n-1}\newline a_{n-1}&a_0&a_1&\cdots&a_{n-2}\newline a_{n-2}&a_{n-1}&a_0&\cdots&a_{n-3}\newline \vdots&\vdots&\vdots&&\vdots\newline a_1&a_2&a_3&\cdots&a_0\end{array}\right]$$

- 记为 $A=\langle a_0, a_1, a_2, \ldots, a_{n-1} \rangle$。

- 基础循环矩阵为 $J=\langle 0, 1, 0, \ldots, 0\rangle$，一般循环矩阵可表示为多项式 $A=a_0 I + a_1 J + a_2 J^2 + \ldots + a_{n-1} J^{n-1}$。
- 循环矩阵乘积还是循环矩阵，$\displaystyle AB=\langle \sum_{i+j\equiv k \pmod n}a_ib_j\rangle_{k=0}^{n-1}$，可以卷积。
- 循环矩阵行列式，令 $f(x)=a_0+a_1x+a_2x^2+...+a_{n-1}x^{n-1}$，$\omega_n$ 为 n 次单位根，则 $\displaystyle\det A=\prod_{i=0}^{n-1}f(\omega_n^i)$，可以用任意长度 FFT 计算。
- [A052182](http://oeis.org/A052182) $[1, -3, 18, -160, 1875, -27216, 470596, -9437184, 215233605]$
  - （定义）$a_n=\det \langle 1,2,\ldots n\rangle$
  - $a_n=(-1)^{n-1} \dfrac {(n + 1)n^{n-1}} 2$

### 带状矩阵高斯消元

$$\left[\begin{array}{c}* & * &   &   &   \newline * & * & * &   &   \newline   & * & * & * &   \newline   &   & * & * & * \newline   &   &   & * & *\end{array}\right]$$

- [求解带状线性方程](https://codeforces.com/contest/963/problem/E) 等高斯消元问题。
- `a` 是系数矩阵，`b` 是常数向量也是结果。
- `d` 是第一列系数个数 - 1，`r` 是第一行系数个数 - 1。
- 编号从 0 开始，$O(nd^2)$。

```cpp
struct vtr:vector<ll>{
    int l,r;
    void init(int _l,int _r){l=_l; r=_r; assign(r-l+1,0);}
    ll &operator[](int x){return at(x-l);}
};
struct mat{
    static const int N=200010;
    vtr a[N]; ll b[N]; int d,r,n; ll det;
    void init(int n,int _d,int _r){
        d=_d; r=_r;
        repeat(i,0,n)a[i].init(i-d,i+r+d),b[i]=0;
    }
    void r_div(int x,ll k){ // a[x][]/=k
        ll r=qpow(k,mod-2);
        for(auto &i:a[x])
            i=i*r%mod;
        b[x]=b[x]*r%mod;
        det=det*k%mod;
    }
    void r_plus(int x,int y,ll k){ // a[x][]+=a[y][]*k
        repeat(i,max(a[x].l,a[y].l),min(a[x].r,a[y].r)+1)
            (a[x][i]+=a[y][i]*k)%=mod;
        (b[x]+=b[y]*k)%=mod;
    }
    void r_swap(int x,int y){ // swap(a[x][],a[y][])
        repeat(i,max(a[x].l,a[y].l),min(a[x].r,a[y].r)+1)
            swap(a[x][i],a[y][i]);
        swap(b[x],b[y]);
        det=-det;
    }
    bool gauss(int n){ // return whether succuss
        this->n=n; det=1;
        repeat(i,0,n){
            int t=-1;
            repeat(j,i,min(i+d+1,n))
                if(a[j][i]){t=j; break;}
            if(t==-1){det=0; return 0;}
            if(t!=i)r_swap(i,t);
            r_div(i,a[i][i]);
            repeat(j,i+1,min(i+d+1,n))
                if(a[j][i])r_plus(j,i,-a[j][i]);
        }
        repeat_back(i,0,n){
            repeat(j,max(0,i-d-r),i)
                if(a[j][i])r_plus(j,i,-a[j][i]);
        }
        return 1;
    }
    // ll get_det(int n){gauss(n); return det;} // return det
    vtr &operator[](int x){return a[x];}
    const vtr &operator[](int x)const{return a[x];}
}a;
```

### 矩阵 归档

- $n\times n$ 方阵 A 有：$\left[\begin{array}{c}A&E\newline O&E\end{array}\right]^{k+1}=\left[\begin{array}{c}A^k&E+A+A^2+...+A^k\newline O&E\end{array}\right]$
- 类似原理可计算 $\displaystyle\sum_{i=0}^n iq^i$，完美解决模数不是质数的情况：

```cpp
int sum_qi(int _n, int q) { // sum q^i, for i = 0 to n
    n = 2; mat A; // 2 * 2 mat
    A[0][0] = q % mod; A[0][1] = A[1][1] = 1;
    A = qpow(A, _n + 1);
    return A[0][1];
}
int sum_iqi(int _n, int q) { // sum i*q^i, for i = 0 to n
    n = 4; mat A; // 4 * 4 mat
    A[0][0] = A[1][0] = A[1][1] = q % mod;
    A[0][2] = A[1][3] = 1 = A[2][2] = A[3][3] = 1;
    A = qpow(A, _n + 1);
    return A[1][2];
}
```

***

- 线性递推转矩快

$$f_{n+3}=af_{n+2}+bf_{n+1}+cf_{n}$$

$$\Leftrightarrow\left[\begin{array}{c}a&b&c\newline 1&0&0\newline 0&1&0\end{array}\right]^n \left[\begin{array}{c}f_2\newline f_1\newline f_0\end{array}\right]=\left[\begin{array}{c}f_{n+2}\newline f_{n+1}\newline f_{n}\end{array}\right]$$

***

- 多次询问同一矩阵的幂与向量的乘积，可以先计算该矩阵的 $2^i$ 次方 $O(n^3\log M+qn^2\log M)$（询问时计算 $\log M$ 次向量和矩阵的乘法）

***

- 矩阵公式

$$\begin{array}{l}
\quad\det\left[(X_i+A_{n-1})\ldots(X_i+A_{j+1})(X_i+B_j)\ldots(X_i+B_1)\right]_{i,j=0}^{n-1}\newline
=\prod_{0\le i<j\le n-1}(X_i-X_j)\prod_{1\le i\le j\le n-1}(B_i-A_j)
\end{array}$$

$$\det\left[C_{\alpha_i+j}\right]_{i,j=0}^{n-1}=\prod_{0\le i<j\le n-1}(\alpha_j-\alpha_i)\prod_{i=0}^{n-1}\dfrac{(i+n)!(2\alpha_i)!}{(2i)!\alpha_i!(\alpha_i+n)!}$$

（$C_n$ 为卡特兰数）

***

- 范德蒙德行列式：

$$\det\left[\begin{matrix}
1 & 1 & \cdots & 1 \newline
x_1 & x_2 & \cdots & x_n \newline
\vdots & \vdots & \ddots & \vdots \newline
x_1^{n-1} & x_2^{n-1} & \cdots & x_n^{n-1}
\end{matrix}\right]=\prod_{1\le i<j\le n} (x_i-x_j)$$

变形：

$$\det\left[\prod _{k=0}^{j-1}(x_i+k)\right]_{i,j=1}^n=\prod_{i=1}^n x_i \prod_{1\le i<j\le n}(x_i-x_j)$$

***

- 伍德伯里矩阵恒等式：

$$(A+UCV)^{−1}=A^{−1}−A^{−1}U(C^{−1}+VA^{−1}U)^{−1}VA^{−1}$$

***

- 伴随矩阵 $A^{\ast}=|A|A^{-1}$，$A^{\ast}_{j,i}=A$ 去掉 i 行 j 列后的矩阵的行列式乘以 $(-1)^{i+j}$，注意转置的问题
- 矩阵行列式引理 Matrix Determinant Lemma：$n\times n$ 可逆矩阵 A 和 n 维列向量 u, v 有 $\det(A+uv^T)=\det(A)(1+v^TA^{-1}u)$

### 带通配符的字符串匹配 using FFT

- 模式串 $A(x)$ 长为 m，文本串 $B(x)$ 长为 n，通配符数值为 0
- 反转 $A(i)=A'(m-i-1)$
- 令 $C(x,y)=[A(x)-B(y)]^2A(x)B(y)$

$$\begin{array}{ccl}P(x)&=&\displaystyle\sum_{i=0}^{m-1}C(i,x+i)\newline &=&\displaystyle\sum_{i=0}^{m-1}[A(i)-B(x+i)]^2A(i)B(x+i)\newline &=&\displaystyle\sum_{i=0}^{m-1}[A^3(i)B(x+i)-2A^2(i)B^2(x+i)+A(i)B^3(x+i)]\newline &=&\displaystyle\sum_{i=0}^{m-1}[A'^3(m-i-1)B(x+i)-2A'^2(m-i-1)B^2(x+i)+A'(m-i-1)B^3(x+i)]\end{array}$$

- 先计算 $A,A^2,A^3,B,B^2,B^3$ 然后 FFT/NTT

```cpp
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
        A[i]=D((A[i]*B3[i]%mod-2*A2[i]*B2[i]%mod+A3[i]*B[i])%mod+mod);
    ntt(A,n*2,-1);
    vector<int> ans;
    repeat(i,m1-1,n1)if(A[i]==0)ans<<i-m1+2;
    printf("%d\n",(int)ans.size());
    for(auto i:ans)printf("%d ",i);
}
```

### 多项式 归档

- 求 $\displaystyle B_i = \sum_{k=i}^n C_k^iA_k$，即 $\displaystyle B_i=\dfrac{1}{i!}\sum_{k=i}^n\dfrac{1}{(k-i)!}\cdot k!A_k$，反转后卷积。
- NTT中，$\omega_n=$ `qpow(G,(mod-1)/n))`。
- 遇到 $\displaystyle \sum_{i=0}^n[i\bmod k=0]f(i)$ 可以转换为 $\displaystyle \sum_{i=0}^n\dfrac 1 k\sum_{j=0}^{k-1}(\omega_k^i)^jf(i)$。（单位根卷积）
- 广义二项式定理 $\displaystyle (1+x)^{\alpha}=\sum_{i=0}^{\infty}{n\choose \alpha}x^i$。

普通生成函数 / OGF

- 普通生成函数：$A(x)=a_0+a_1x+a_2x^2+...=\langle a_0,a_1,a_2,...\rangle$
- $1+x^k+x^{2k}+...=\dfrac{1}{1-x^k}$
- 取对数后 $\displaystyle=-\ln(1-x^k)=\sum_{i=1}^{\infty}\dfrac{1}{i}x^{ki}$ 即 $\displaystyle\sum_{i=1}^{\infty}\dfrac{1}{i}x^i\otimes x^k$（polymul_special）
- $x+\dfrac{x^2}{2}+\dfrac{x^3}{3}+...=-\ln(1-x)$
- $1+x+x^2+...+x^{m-1}=\dfrac{1-x^m}{1-x}$
- $1+2x+3x^2+...=\dfrac{1}{(1-x)^2}$（借用导数，$nx^{n-1}=(x^n)'$）
- $C_m^0+C_m^1x+C_m^2x^2+...+C_m^mx^m=(1+x)^m$（二项式定理）
- $C_m^0+C_{m+1}^1x^1+C_{m+2}^2x^2+...=\dfrac{1}{(1-x)^{m+1}}$（归纳法证明）
- $\displaystyle\sum_{n=0}^{\infty}F_nx^n=\dfrac{(F_1-F_0)x+F_0}{1-x-x^2}$（F 为斐波那契数列，列方程 $G(x)=xG(x)+x^2G(x)+(F_1-F_0)x+F_0$）
- $\displaystyle\sum_{n=0}^{\infty} H_nx^n=\dfrac{1-\sqrt{n-4x}}{2x}$（H 为卡特兰数）
- 前缀和 $\displaystyle \sum_{n=0}^{\infty}s_nx^n=\dfrac{1}{1-x}f(x)$
- 五边形数定理：$\displaystyle \prod_{i=1}^{\infty}(1-x^i)=\sum_{k=0}^{\infty}(-1)^kx^{\frac 1 2k(3k\pm 1)}$

指数生成函数 / EGF

- 指数生成函数：$A(x)=a_0+a_1x+a_2\dfrac{x^2}{2!}+a_3\dfrac{x^3}{3!}+...=\langle a_0,a_1,a_2,a_3,...\rangle$
- 普通生成函数转换为指数生成函数：系数乘以 $n!$
- $1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+...=\exp x$
- 长度为 n 的循环置换数为 $P(x)=-\ln(1-x)$，长度为 n 的置换数为 $\exp P(x)=\dfrac{1}{1-x}$（注意是**指数**生成函数）
- 推广：
  - n 个点的生成树个数是 $\displaystyle P(x)=\sum_{n=1}^{\infty}n^{n-2}\dfrac{x^n}{n!}$，n 个点的生成森林个数是 $\exp P(x)$
  - n 个点的无向连通图个数是 $P(x)$，n 个点的无向图个数是 $\displaystyle\exp P(x)=\sum_{n=0}^{\infty}2^{\frac 1 2 n(n-1)}\dfrac{x^n}{n!}$
  - 长度为 $n(n\ge 2)$ 的循环置换数是 $P(x)=-\ln(1-x)-x$，长度为 n 的错排数是 $\exp P(x)$

## 博弈论

Nim

- n 堆石子 $a_1,a_2,...,a_n$，每次选择 1 堆石子拿任意非空的石子，拿不了的人失败
- $SG_i=a_i,NimSum=\oplus\{SG_i\}$，先手必败当且仅当 $NimSum=0$
- 注：先手必胜策略是找到满足 `(a[i]>>__lg(NimSum))&1` 的 $a[i]$，并取走 $a[i]-a[i]\oplus NimSum$ 个石子
- Bash Game：一堆石子 n，最多取 k 个，$SG=n\bmod (k+1)$

***

Moore's Nimk

- n 堆石子，每次最多选取 k 堆石子，选中的每一堆都取走任意非空的石子
- 先手必胜当且仅当
  - 存在 t 使得 `sum{(a[i]>>t)&1}%(k+1)!=0`

***

扩展威佐夫博弈 / Extra Wythoff's Game

- 两堆石子，分别为 a, b，每次取一堆的任意非空的石子或者取两堆数量之差的绝对值小于等于 k 的石子
- 解：假设 $a\le b$，当且仅当存在自然数 n 使得 $a=\lfloor n\dfrac{\sqrt{(k+1)^2+4}-(k-1)}2\rfloor,b=a+n(k+1)$，先手必败
- Betty定理与Betty数列：$\alpha,\beta$ 为正无理数且 $\dfrac 1 {\alpha}+\dfrac 1 {\beta}=1$，数列 $\{\lfloor \alpha n\rfloor\},\{\lfloor \beta n\rfloor\},n=1,2,...$ 无交集且覆盖正整数集合

***

斐波那契博弈 / Fibonacci Nim

- 一堆石子 $n,n\ge 2$，先手第一次只能取 $[1,n-1]$，之后每次取的石子数不多于对手刚取的石子数的 2 倍且非空
- 先手必败当且仅当 n 是Fibonacci数

***

阶梯Nim / Staircase Nim

- n 堆石子，每次选择一堆取任意非空的石子放到前一堆，第 1 堆的石子可以放到第 0 堆
- 先手必败当且仅当奇数堆的石子数异或和为 0

***

Lasker's Nim

- n 堆石子，每次可以选择一堆取任意非空石子，或者选择某堆至少为 2，分成两堆非空石子
- $SG(0)=0,SG(4k+1)=4k+1,SG(4k+2)=4k+2,SG(4k+3)=4k+4,SG(4k+4)=4k+3$

***

k 倍动态减法博弈

- 一堆石子 $n,n\ge 2$，先手第一次只能取 $[1,n-1]$，之后每次取的石子数不多于对手刚取的石子数的 k 倍且非空

```cpp
int calc(ll n,int k){ // n<=1e8,k<=1e5
    static ll a[N],b[N],ans; // N=750010
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

Anti-SG using SJ定理

- n 个游戏，移动不了的人获胜
- 先手必胜当且仅当
  - $(\forall i)SG_i\le 1$ 且 $NimSum=0$
  - $(\exists i)SG_i>1$ 且 $NimSum\not=0$

***

Every-SG

- n 个游戏，每次都要移动所有可移动的游戏
- 对于先手来说，必胜态的游戏要越长越好，必败态的游戏要越短越好
- u是终止态，step(u)=0
- u->v,SG(u)=0,SG(v)>0，step(u)=max(step(v))+1
- u->v,SG(v)=0，step(u)=min(step(v))+1
- 先手必胜当且仅当所有游戏的step的最大值为奇数

***

删边游戏 / Green Hachenbush

- 树上删边游戏
  - 一棵有根树，每次可以删除一条边并移除不和根连接的部分
  - 叶子的 SG 为 0，非叶子的 SG 为 (所有儿子的 SG 值 + 1) 的异或和
- 无向图删边游戏
  - 奇环可以缩为一个点加一条边，偶环可以缩为一点，变为树上删边游戏

***

翻硬币游戏

- n 枚硬币排成一排，玩家的操作有一定约束，并且翻动的硬币中，最右边的必须是从正面翻到反面，不能操作的玩家失败
- 定理：局面的 SG 值等于所有正面朝上的硬币单一存在时的 SG 值的异或和（把这个硬币以外的所有硬币翻到反面后的局面的 SG 值）
- 编号从 1 开始
  - 每次翻一枚或两枚硬币 $\text{SG}(n)=n$
  - 每次翻转连续的 k 个硬币 $\text{SG}(n)=[n\bmod k=0]$
  - Ruler Game，每次翻转一个区间的硬币，$\text{SG}(n)=\text{lowbit}(n)$
  - Mock Turtles Game，每次翻转不多于 3 枚硬币 $\text{SG}(n)=2n-1-\text{popcount}(n-1)\bmod 2$

## 数学的其他操作

### 主定理 / Master Theorem

- 对于 $T(n)=aT(\dfrac nb)+n^k$ （要估算 $n^k$ 的 k 值）
- 若 $\log_ba>k$，则 $T(n)=O(n^{\log_ba})$
- 若 $\log_ba=k$，则 $T(n)=O(n^k\log n)$
- 若 $\log_ba<k$（有省略），则 $T(n)=O(n^k)$

### 约瑟夫问题

- n个人编号0..(n-1)，每次数到k出局，求最后剩下的人的编号
- 线性算法，$O(n)$

```cpp
int jos(int n,int k){
    int res=0;
    repeat(i,1,n+1)res=(res+k)%i;
    return res; // res+1，如果编号从1开始
}
```

- 对数算法，适用于k较小情况，$O(k\log n)$

```cpp
int jos(int n,int k){
    if(n==1 || k==1)return n-1;
    if(k>n)return (jos(n-1,k)+k)%n; // 线性算法
    int res=jos(n-n/k,k)-n%k;
    if(res<0)res+=n; // mod n
    else res+=res/(k-1); // 还原位置
    return res; // res+1，如果编号从1开始
}
```

### 格雷码 / Gray Code

- 一些性质：
- 相邻格雷码只变化一次
- `grey(n-1)` 到 `grey(n)` 修改了二进制的第 `(__builtin_ctzll(n)+1)` 位
- `grey(0)..grey(2^k-1)` 是k维超立方体顶点的哈密顿回路，其中格雷码每一位代表一个维度的坐标
- 格雷码变换，正 $O(1)$，逆 $O(\log n)$

```cpp
ll grey(ll n){ // 第n个格雷码
    return n^(n>>1);
}
ll degrey(ll n){ // 逆格雷码变换
    repeat(i,0,63) // or 31
        n=n^(n>>1);
    // n^=n>>1; n^=n>>2; n^=n>>4; n^=n>>8; n^=n>>16; n^=n>>32; // O(loglogn) 操作
    return n;
}
```

### 汉诺塔

- 假设盘数为n，总共需要移动 `(1<<n)-1` 次
- 第k次移动第 `i=__builtin_ctzll(n)+1` 小的盘子
- 该盘是第 `(k>>i)+1` 次移动
- （可以算出其他盘的状态：总共移动了 `((k+(1<<(i-1)))>>i)` 次）
- 该盘的移动顺序是：
    `A->C->B->A（当i和n奇偶性相同）`
    `A->B->C->A（当i和n奇偶性不同）`

```cpp
cin>>n; // 层数
repeat(k,1,(1<<n)){
    int i=__builtin_ctzll(k)+1;
    int p1=(k>>i)%3; // 移动前状态
    int p2=(p1+1)%3; // 移动后状态
    if(i%2==n%2){
        p1=(3-p1)%3;
        p2=(3-p2)%3;
    }
    cout<<"move "<<i<<": "<<"ABC"[p1]<<" -> "<<"ABC"[p2]<<endl;
}
```

- 4个柱子的汉诺塔情况：令 $k=\lfloor n+1-\sqrt{2n+1}+0.5\rfloor$，让前k小的盘子用4个柱子的方法移到2号柱，其他盘子用3个柱子的方法移到4号柱，最后再移一次前k小，最短步数 $f(n)=2f(k)+2^{n-k}-1$

### Stern-Brocot 树 and Farey 序列

- 分数序列：在 $[\dfrac 0 1,\dfrac 1 0]$ 中不断在 $\dfrac a b$ 和 $\dfrac c d$ 之间插入 $\dfrac {a+c}{b+d}$
- 性质：所有数都是既约分数、可遍历所有既约分数、保持单调递增
- Stern-Brocot 树：二叉树，其第 k 行是分数序列第 k 次操作新加的数
- Farey 序列：$F_n$ 是所有分子分母 $\le n$ 的既约分数按照分数序列顺序排列后的序列
- $F_n$ 的长度 $\displaystyle=1+\sum_{i=1}^n\varphi(i)$

### 浮点与近似计算

牛顿迭代法

- 求 $f(x)$ 的零点：$x_{n+1}=x_n-\dfrac{f(x)}{f'(x)}$
- 检验 $x_{n+1}=g(x_n)$ 多次迭代可以收敛于 $x_0$ 的方法：看 $|g'(x_0)|\le1$ 是否成立

```cpp
lf newton(lf n){ // sqrt
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

others of 浮点与近似计算

$$\lim_{n\rightarrow\infty}\dfrac{错排(n)}{n!}=\dfrac 1 e,e\approx 2.718281828459045235360287471352$$

$$\lim_{n\rightarrow\infty}(\sum\frac 1 n-\ln n)=\gamma\approx 0.577215664901532860606$$

### 日期换算

- 基姆拉尔森公式（已知年月日，求星期数）

```cpp
int week(int y,int m,int d){
    if(m<=2)m+=12,y--;
    return (d+2*m+3*(m+1)/5+y+y/4-y/100+y/400)%7+1;
}
```

- 标准阳历与儒略日转换

```cpp
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

### 数学 结论

- 如果加法变成“乘法”，那么“加法”就是 $\min$ 或 $\max$，它们构成了热带半环。（当然 $x\oplus y=-\ln(e^{-x}+e^{-y})$ 也可以）[](https://zhuanlan.zhihu.com/p/113990049)类似还有位与和位异或（每一位是模 2 的乘法和加法）。

***

- 若排列可以分解为两个 LIS，那么将排列按分割点 $\max a_{1..i}<\min a_{i+1..n}$ 划分后，每个区间分解为两个 LIS 的方法都是唯一的。

***

- 保序回归：给定实数序列 $a_i$，求不下降实数序列 $b_i$，使得 $\displaystyle\sum_{i=1}^{n}(a_i-b_i)^2$ 最小化。
  - 用单调栈保存答案，找第一个 $a_i>a_{i-1}$，然后找最大的 k，区间 $[k,i]$ 里的答案即 $a_i$ 的平均值。
  - 如果代价函数改成绝对值，也可以用单调栈，双堆维护中位数。

***

- 埃及分数 Engel 展开
- 待展开的数为 x，令 $u_1=x, u_{i+1}=u_i\times\lceil\dfrac 1 {u_i}\rceil-1$（到 0 为止）
- 令 $a_i=\lceil\dfrac 1 {u_i}\rceil$
- 则 $x=\dfrac 1{a_1}+\dfrac 1{a_1a_2}+\dfrac 1{a_1a_2a_3}+...$

***

- 三个水杯容量为 $a,b,c$（正整数），$a=b+c$，初始 a 装满水，则得到容积为 $\dfrac a 2$ 的水需要倒 $\dfrac a{\gcd(b,c)}-1$ 次水（无解条件为 $\dfrac a{\gcd(b,c)}\bmod 2=1$）

***

- 兰顿蚂蚁（白色异或右转，黑色异或左转），约一万步后出现周期为104步的无限重复（高速公路）

***

- 任意勾股数能由复数 $(a+bi)^2\space(a,b∈\mathbb{Z})$ 得到

***

- 任意正整数 a 都存在正整数 b, c 使得 $a<b<c$ 且 $a^2,b^2,c^2$ 成等差数列：构造 $b=5a,c=7a$

***

- 拉格朗日四平方和定理：每个正整数都能表示为4个整数平方和
- 对于偶素数 2 有 $2=1^2+1^2+0^2+0^2$
- 对于奇素数 p 有 $p=a^2+b^2+1^2+0^2$ （容斥可证）
- 对于所有合数 n 有 $n=z_1^2+z_2^2+z_3^2+z_4^2=(x_1^2+x_2^2+x_3^2+x_4^2)\cdot(y_1^2+y_2^2+y_3^2+y_4^2)$
- 其中 $\begin{cases} z_1=x_1y_1+x_2y_2+x_3y_3+x_4y_4 \newline  z_2=x_1y_2-x_2y_1-x_3y_4+x_4y_3 \newline  z_3=x_1y_3-x_3y_1+x_2y_4-x_4y_2 \newline  z_4=x_1y_4-x_4y_1-x_2y_3+x_3y_2\end{cases}$

***

求 $\displaystyle \sum_{i=1}^n \lfloor \dfrac n i \rfloor$

```cpp
int f(int n){
    int ans=0;
    int t=sqrt(n);
    repeat(i,1,t+1)ans+=n/i;
    return ans*2-t*t;
}
```

***

求 $\displaystyle \sum _ {i = 0} ^ {n}\lfloor\dfrac{i}{c}\rfloor$：`(n / c - 1) * (n / c) / 2 * c + (n % c + 1) * (n / c)`

***

n 维超立方体有 $\displaystyle 2^{n-i} {n \choose i}$ 个 i 维元素

## 图论

### 图论的一些概念

- 基环图：树加一条边。
- 简单图：不含重边和自环（默认）。
- 完全图：顶点两两相连的无向图。
- 竞赛图：顶点两两相连的有向图。
- 点 u 到 v 可达：有向图中，存在 u 到 v 的路径。
- 点 u 和 v 联通：无向图中，存在 u 到 v 的路径。
- 生成子图：点集和原图相同。
- 导出子图 / 诱导子图：选取一个点集，尽可能多加边。
- 正则图：所有点的度均相同的无向图。

***

- 强正则图：$\forall (u,v)\in E,|\omega(u)\cap \omega(v)|=\text{const}$，且 $\forall (u,v)\not\in E,|\omega(u)\cap \omega(v)|=\text{const}$ 的正则图（$\omega(u)$ 为 u 的邻域）。
- 强正则图的点数 v，度 k，相邻的点的共度 $\lambda$，不相邻的点的共度 $\mu$ 有 $k(k-1-\lambda)=\mu(v-1-k)$。
- 强正则图的例子：所有完全图、所有 nk 顶点满 n 分图。

***

- 点割集：极小的，把图分成多个联通块的点集
- 割点：自身就是点割集的点
- 边割基：极小的，把图分成多个联通块的边集
- 桥：自身就是边割集的边
- 点联通度：最小点割集的大小
- 边联通度：最小边割集的大小
- Whitney 定理：点联通度≤边联通度≤最小度

***

- 最大团：最大完全子图
- 最大独立集：最多的两两不连接的顶点
- 最小染色数：相邻的点不同色的最少色数
- 最小团覆盖数：覆盖整个图的最少团数
- 最大独立集即补图最大团
- 最小染色数等于补图最小团覆盖数

***

- 哈密顿通路：通过所有顶点有且仅有一次的路径，若存在则为半哈密顿图/哈密顿图。
- 哈密顿回路：通过所有顶点有且仅有一次的回路，若存在则为哈密顿图。
- 完全图 $K_{2k+1}$ 的边集可以划分为 k 个哈密顿回路。
- 完全图 $K_{2k}$ 的边集去掉 k 条互不相邻的边后可以划分为 $k-1$ 个哈密顿回路。

***

- 连通块数 = 点数 - 边数

### 欧拉图 using 套圈算法

- 假设是连通图。
- 无向图：
  - 若存在则路径为 DFS 退出序。（最后的序列还要再反过来）（如果 for 从小到大，可以得到最小字典序）
  - （不记录点的 `vis`，只记录边的 `vis`）
- 有向图：
  - 欧拉回路存在当且仅当连通且所有点入度等于出度。
  - 欧拉路径存在当且仅当连通且除了起点终点外所有点入度等于出度。
  - 跑反图退出序。
- 混合图：
  - 欧拉回路存在当且仅当 `indeg + outdeg + undirdeg` 是偶数，且 `max(indeg, outdeg) * 2 <= indeg + outdeg + undirdeg`。
  - 欧拉路径还没研究过。
  - 无向边任意定向算法。每次找 `outdeg > indeg` 的点向 `outdeg < indeg` 的任意点连一条任意路径，要求只经过无向边，将路径上的边转换为有向边，欧拉路径存在条件仍满足。当所有点都有 `outdeg == indeg`，直接反图跑退出序。
- 无 / 有向图代码：（前向星访问反向边 + 当前弧优化）

```cpp
int n,deg[N];
struct edge{int to,nxt,id;};
vector<edge> a; int head[N];
vector<int> ans;
int undirect=1; // undirected graph
void ae(int x,int y,int id){ // add edge
    a.push_back({y,head[x],id});
    head[x]=a.size()-1;
}
int cur[N];
void dfs(int x){
    for(int &i=cur[x];i!=-1;){
        edge t=a[i];
        if(undirect)a[i^1].to=-1;
        i=a[i].nxt;
        if(t.to!=-1){
            dfs(t.to);
            ans.push_back(t.id);
        }
    }
}
void Solve() {
    int t=read(); undirect=(t==1); // t 表示是否无向
    n=read(); int m=read();
    fill(head,head+n+1,-1);
    int s;
    repeat(i,1,m+1){
        int x=read(),y=read(); s=x;
        if(undirect)ae(x,y,-i); ae(y,x,i);
        deg[x]++,deg[y]--;
    }
    // repeat(i,1,n+1)if(deg[i]%2!=0)s=i;
    // repeat(i,1,n+1)if(deg[i]==1)s=i;
    if(m){
        copy(head,head+n+1,cur);
        dfs(s);
    }
    if((int)ans.size()!=m || (undirect?deg[s]%2!=0:deg[s]!=0))puts("NO"); // || 后面的表达式表示是一个欧拉回路
    else{
        puts("YES");
        for(auto i:ans)print(i);
    }
}
```

### DFS 树 and BFS 树

- 无向图 DFS 树：树边、返祖边。
- 有向图 DFS 树：树边、返祖边、横叉边、前向边。
- 无向图 BFS 树：树边、返祖边、横叉边。
- 空缺

### 最小环

- 有向图最小环Dijkstra，$O(VE\log E)$：对每个点 v 进行Dijkstra，到达 v 的边更新答案，适用稀图
- 有向图最小环Floyd，$O(V^3)$：Floyd完之后，任意两点计算 $dis_{u,v}+dis_{v,u}$，适用稠图
- 无边权无向图最小环：以每个顶点为根生成BFS树（不是DFS），横叉边更新答案，$O(VE)$
- 有边权无向图最小环：上面的BFS改成Dijkstra，$O(VE \log E)$

```cpp
// 无边权无向图最小环
int dis[N],fa[N],n,ans;
vector<int> a[N];
queue<int> q;
void bfs(int s){ // 求经过s的最小环（不一定是简单环）
    fill(dis,dis+n,-1); dis[s]=0;
    q.push(s); fa[s]=-1;
    while(!q.empty()){
        int x=q.front(); q.pop();
        for(auto p:a[x])
        if(p!=fa[x]){
            if(dis[p]==-1){
                dis[p]=dis[x]+1;
                fa[p]=x;
                q.push(p);
            }
            else ans=min(ans,dis[x]+dis[p]+1);
        }
    }
}
int mincycle(){
    ans=inf;
    repeat(i,0,n)bfs(i); // 只要遍历最小环可能经过的点即可
    return ans;
}
```

### 差分约束

- $a_i-a_j\le c$，建边 $(j,i,c)$

### 同余最短路

- k 种木棍，每种木棍个数不限，长度分别为 $l_i$，求这些木棍可以拼凑出多少小于等于 h 的整数（包括 0）
- 以任一木棍 $n=l_0$ 为剩余系，连边 $(i,(i+l_j)\bmod n,l_j),j>1$，跑最短路后 $dis[i]$ 表示 $dis[i]+tn,t∈\mathbb{N}$ 都可以被拼凑出来
- 编号从 0 开始，$O(nk\log(nk))$

```cpp
ll solve(ll h,int l[],int k){
    n=l[0];
    repeat(i,0,n)
    repeat(j,1,k)
        a[i]<<pii((i+l[j])%n,l[j]);
    dij(0);
    ll ans=0;
    repeat(i,0,n)
    if(dis[i]<=h)
        ans+=(h-dis[i])/n+1;
    return ans;
}
```

### 最小树形图 using 朱刘算法

- 其实有更高级的Tarjan算法 $O(E+V\log V)$，~~但是学不会~~
- 编号从1开始，求的是叶向树形图，$O(VE)$

```cpp
int n;
struct edge{int x,y,w;};
vector<edge> eset; // 会在solve中被修改
ll solve(int rt){ // 返回最小的边权和，返回-1表示没有树形图
    static int fa[N],id[N],top[N],minw[N];
    ll ans=0;
    while(1){
        int cnt=0;
        repeat(i,1,n+1)
            id[i]=top[i]=0,minw[i]=inf;
        for(auto &i:eset) // 记录权最小的父亲
        if(i.x!=i.y && i.w<minw[i.y]){
            fa[i.y]=i.x;
            minw[i.y]=i.w;
        }
        minw[rt]=0;
        repeat(i,1,n+1){ // 标记所有环
            if(minw[i]==inf)return -1;
            ans+=minw[i];
            for(int x=i;x!=rt && !id[x];x=fa[x])
            if(top[x]==i){
                id[x]=++cnt;
                for(int y=fa[x];y!=x;y=fa[y])
                    id[y]=cnt;
                break;
            }
            else top[x]=i;
        }
        if(cnt==0)return ans; // 无环退出
        repeat(i,1,n+1)
        if(!id[i])
            id[i]=++cnt;
        for(auto &i:eset){ // 缩点
            i.w-=minw[i.y];
            i.x=id[i.x],i.y=id[i.y];
        }
        n=cnt;
        rt=id[rt];
    }
}
```

### 绝对中心 and 最小直径生成树 / MDST

- 绝对中心：到所有点距离最大值最小的点，可以在边上
- 最小直径生成树：直径最小的生成树，可构造绝对中心为根的最短路径树
- 返回绝对中心所在边，生成树直径为 `d[x][rk[x][n-1]]+d[y][rk[y][n-1]]-d[x][y]`
- 编号从 0 开始，$O(n^3)$，$n=1000$ 勉强能过

```cpp
int rk[N][N],d[N][N];
pii solve(int g[][N],int n){
    lf ds1=0,ds2=0;
    repeat(i,0,n)repeat(j,0,n)d[i][j]=g[i][j];
    repeat(k,0,n)repeat(i,0,n)repeat(j,0,n)
        d[i][j]=min(d[i][j],d[i][k]+d[k][j]);
    repeat(i,0,n){
        iota(rk[i],rk[i]+n,0);
        sort(rk[i],rk[i]+n,[&](int a,int b){
            return d[i][a]<d[i][b];
        });
    }
    int ans=inf,s1=-1,s2=-1;
    repeat(x,0,n){
        if(d[x][rk[x][n-1]]*2<ans){
            ans=d[x][rk[x][n-1]]*2;
            s1=s2=x; ds1=ds2=0;
        }
        repeat(y,0,n){
            if(g[x][y]==inf)continue;
            int k=n-1;
            repeat_back(i,0,n-1)
            if(d[y][rk[x][i]]>d[y][rk[x][k]]){
                int now=d[x][rk[x][i]]+d[y][rk[x][k]]+g[x][y];
                if(now<ans){
                    ans=now; s1=x,s2=y;
                    ds1=0.5*now-d[x][rk[x][i]];
                    ds2=g[x][y]-ds1;
                }
                k=i;
            }
        }
    }
    return {s1,s2};
}
// init: repeat(i,0,n)repeat(j,0,n)g[i][j]=inf*(i!=j);
```

### 弦图 and 区间图

- 弦是连接环上不相邻点的边；弦图是所有长度大于3的环都有弦的无向图（类似三角剖分）
- 单纯点：所有与v相连的点构成一个团，则v是一个单纯点
- 完美消除序列：即点集的一个排列 $[v_1,v_2,...,v_n]$ 满足任意 $v_i$ 在 $[v_{i+1},...,v_n]$ 的导出子图中是一个单纯点
- 定理：无向图是弦图 $\Leftrightarrow$ 无向图存在完美消除序列
- 定理：最大团顶点数 $\le$ 最小染色数（弦图取等号）
- 定理：最大独立集顶点数 $\le$ 最小团覆盖（弦图取等号）

***

- 最大势算法MCS求完美消除序列：每次求出与 $[v_{i+1},...,v_n]$ 相邻点数最大的点作为 $v_i$
- `e[][]`点编号从 1 开始！`rec` 下标从 1 开始！桶优化，$O(V+E)$

```cpp
vector<int> e[N];
int n,rec[N]; // rec[1..n]是结果
int h[N],nxt[N],pre[N],vis[N],lab[N];
void del(int x){
    int w=lab[x];
    if(h[w]==x)h[w]=nxt[x];
    pre[nxt[x]]=pre[x];
    nxt[pre[x]]=nxt[x];
}
void mcs(){
    fill(h,h+n+1,0);
    fill(vis,vis+n+1,0);
    fill(lab,lab+n+1,0);
    iota(nxt,nxt+n+1,1);
    iota(pre,pre+n+1,-1);
    nxt[n]=0;
    h[0]=1;
    int w=0;
    repeat_back(i,1,n+1){
        int x=h[w];
        rec[i]=x;
        del(x);
        vis[x]=1;
        for(auto p:e[x])
        if(!vis[p]){
            del(p);
            lab[p]++;
            nxt[p]=h[lab[p]];
            pre[h[lab[p]]]=p;
            h[lab[p]]=p;
            pre[p]=0;
        }
        w++;
        while(h[w]==0)w--;
    }
}
```

***

- 判断弦图（判断是否为完美消除序列）：对所有 $v_i$，$[v_{i+1},...,v_n]$ 中与 $v_i$ 相连的最靠前一个点 $v_j$ 是否与与 $v_i$ 连接的其他点相连
- 编号规则同上，大佬：$O(V+E)$，我：$O((V+E)\log V)$

```cpp
bool judge(){ // 返回是否是完美消除序列（先要跑一遍MCS）
    static int s[N],rnk[N];
    repeat(i,1,n+1){
        rnk[rec[i]]=i;
        sort(e[i].begin(),e[i].end()); // 方便二分查找，内存足够直接unmap
    }
    repeat(i,1,n+1){
        int top=0,x=rec[i];
        for(auto p:e[x])
        if(rnk[x]<rnk[p]){
            s[++top]=p;
            if(rnk[s[top]]<rnk[s[1]])
                swap(s[1],s[top]);
        }
        repeat(j,2,top+1)
        if(!binary_search(e[s[1]].begin(),e[s[1]].end(),s[j]))
            return 0;
    }
    return 1;
}
```

***

- 其他弦图算法

```cpp
int color(){ // 返回最大团点数/最小染色数
    return *max_element(lab+1,lab+n+1)+1;
    /* // 以下求最大团
    static int rnk[N];
    repeat(i,1,n+1)rnk[rec[i]]=i;
    int x=max_element(lab+1,lab+n+1)-lab;
    rec2.push_back(x);
    for(auto p:e[x])
    if(rnk[x]<rnk[p])
        rec2.push_back(x);
    */
}
int maxindset(){ // 返回最大独立集点数/最小团覆盖数
    int ans=0;
    fill(vis,vis+n+1,0);
    repeat(i,1,n+1){
        int x=rec[i];
        if(!vis[x]){
            ans++; // rec2.push_back(x); // 记录最大独立集
            for(auto p:e[x])
                vis[p]=1;
        }
    }
    return ans;
}
int cliquecnt(){ // 返回极大团数
    static int s[N],fst[N],rnk[N],cnt[N];
    int ans=0;
    repeat(i,1,n+1)rnk[rec[i]]=i;
    repeat(i,1,n+1){
        int top=0,x=rec[i];
        for(auto p:e[x])
        if(rnk[x]<rnk[p]){
            s[++top]=p;
            if(rnk[s[top]]<rnk[s[1]])
                swap(s[1],s[top]);
        }
        fst[x]=s[1]; cnt[x]=top;
    }
    fill(vis,vis+n+1,0);
    repeat(i,1,n+1){
        int x=rec[i];
        if(!vis[x])ans++;
        if(cnt[x]>0 && cnt[x]>=cnt[fst[x]]+1)
            vis[fst[x]]=1;
    }
    return ans;
}
```

***

- 区间图：给出的每个区间都看成点，有公共部分的两个区间之间连一条边
- 区间图是弦图（反过来不一定），可以应用弦图的所有算法
- 区间图的判定：所有弦图可以写成一个极大团树（所有极大团看成一个顶点，极大团之间有公共顶点就连一条边），区间图的极大团树是一个链

### 树、图的哈希

树哈希

- $\displaystyle Hash[u]=sz[u]\sum_{v_i} Hash[v_i]B^{i-1}$（$v_i$ 根据哈希值排序）
- $\displaystyle Hash[u]=\oplus(C\cdot Hash[v_i]+sz[v_i])$
- $\displaystyle Hash[u]=1+\sum_{v_i}Hash[v_i]\cdot prime[sz[v_i]]$
- 无根树哈希可以找重心为根（重心最多只有两个）。
- 一种自创哈希方式。

```cpp
vector<int> a[N];
pii H[N];
void dfs(int x,int fa){ // the answer is H[rt]
    H[x]=pii(1,1);
    for(auto p:a[x])if(p!=fa)dfs(p,x);
//  sort(a[x].begin(),a[x].end(),[](int x,int y){
//      return pii(H[x].fi^H[x].se,H[x].fi)
//          <  pii(H[y].fi^H[y].se,H[y].fi);
//  });
    repeat(i,0,a[x].size()){
        H[x].fi^=H[a[x][i]].fi+H[a[x][i]].se;
        H[x].se+=H[a[x][i]].fi^H[a[x][i]].se;
    }
}
```

图哈希

- 枚举起点 s，令所有点的权值 $f_0(i)=1$，迭代：
- $\displaystyle f_{j+1}(u)=\left[A\cdot f_j(u)+B\cdot\sum_{u\rightarrow w}f_j(w)+C\cdot\sum_{w\rightarrow u}f_j(w)+D\cdot[u=s]\right]\bmod P$
- 取 $f_k(s)$。对所有 s 取 n 个值组成集合
- 如果是无向图就去掉 C 项；如果会超时，就去掉 D 项

### 二分图 归档

- 最小点覆盖（最小的点集，使所有边都能被覆盖） = 最大匹配
- 最小边覆盖（最小的边集，使所有点都能与某边关联） = 顶点数 - 最大匹配
- 最大独立集 = 顶点数 - 最大匹配
- 最小带权点覆盖 = 点权之和 - 最大带权独立集（左式用最小割求）

***

- DAG 最小不相交路径覆盖 = （开点前）顶点数 - 最大匹配，右顶点未被匹配的都看作起点
- 有向图最小可相交路径覆盖 = 其传递闭包的 SCC 缩点后的 DAG 最小不相交路径覆盖
- 最长反链（DAG 最大两两不可达点集）= DAG 最小可相交路径覆盖

***

- 霍尔定理：最大匹配 = 左顶点数 $\Leftrightarrow$ 所有左顶点子集 S 都有 $|S|\le|\omega(S)|$ ，$\omega(S)$ 是 S 的领域
- 运用：若在最大匹配中有 t 个左顶点失配，因此最大匹配 = 左顶点数 - t
- 对任意左顶点子集 S 都有 $|S|\le|\omega(S)|+t$，$t\ge|S|-|\omega(S)|$ ，求右式最大值即可求最大匹配

***

- 给定 $n\times m$ 有障碍地图，车不能越过障碍物，要使车不互相攻击，最多放置多少车
  - 将每个极大的 $1\times k$ 的空地作为点集 A，每个极大的 $k \times 1$ 的空地作为点集 B，若两个点对应的空地区域有交集则连边，跑二分图最大匹配

***

- 给定一个无向图和 $d_i$（$1\le d_i\le 2$），求是否能删去一些边后满足点 i 的度刚好是 $d_i$

```cpp
::n=n*2+m*2; // ::n是带花树板子里的n
repeat(i,1,n+1)cnt+=deg[i]=read();
repeat(i,1,m+1){
    int x=read(),y=read();
    if(deg[x]==2 && deg[y]==2){ // (x,e)(x',e)(y,e')(y',e')(e,e')
        add(x,n*2+i),add(x+n,n*2+i),add(y,n*2+m+i),add(y+n,n*2+m+i),add(n*2+i,n*2+m+i);
        cnt+=2;
    }
    else{ // (x,y), 度为2再添边
        add(x,y); if(deg[x]==2)add(x+n,y); if(deg[y]==2)add(x,y+n);
    }
}
puts(solve()*2==cnt?"Yes":"No");
```

### 网络流 归档

- $c(u,v)$ 为 u 到 v 的容量，$f(u,v)$ 为 u 到 v 的流量，$f(u,v)<c(u,v)$
- $c[X,Y]$ 为 X 到 Y 的容量和，不包括 Y 到 X 的容量；$f(X,Y)$ 为 X 到 Y 的流量和，要减去 Y 到 X 的流量

***

- 费用流（最小费用最大流）：保证最大流后的最小费用

***

最大流最小割定理

- 割：割 $[S,T]$ 是点集的一个分割且 S 包含源点，T 包含汇点，称 $f(S,T)$ 为割的净流，$c[S,T]$ 为割的容量
- 最大流最小割定理：最大流即最小割容量
- 求最小割：在最大流残量网络中，令源点可达的点集为 S，其余的为 T 即可（但是满流边不一定都在 S, T 之间）

***

最大权闭合子图

- 闭合子图：子图内所有点的儿子都在子图内。点权之和最大的闭合子图为最大权闭合子图。
- 求最大权闭合子图：点权为正则 s 向该点连边，边权为点权，为负则向 t 连边，边权为点权绝对值，原图所有边的权设为 inf，跑最小割。如果连 s 的边被割则不选这个点，若连 t 的边被割则选这个点。

***

- 最优序列：n 个正整数的序列，从中选取和最大的子序列，满足所有长度为 m 的区间里选取的数字不超过 k 个。
  - 建图跑最大费用流（费用已取反）。

```cpp
repeat(i,0,n){ // add(x,y,w,cost)
    add(S,i,inf,0);
    if(i+1<n)add(i,i+1,inf,0);
    add(i,i+n,1,-a[i]);
    if(i+m<n)add(i+n,i+m,inf,0);
    add(i+n,T,inf,0);
}
```

***

- 给一个图，选择一些边组成基环树，可以对边 $e:(x,y)$ 连 `x->e1, y->e1, e1->e2`，容量为 1。（相当于看作内向基环树）

### 矩阵树定理

无向图矩阵树定理

- 生成树计数

```cpp
void matrix::addedge(int x,int y){
    a[x][y]--,a[y][x]--;
    a[x][x]++,a[y][y]++;
}
lf matrix::treecount(){
    // for(auto i:eset)addedge(i.fi,i.se); // 加边
    n--,m=n; // a[n-1][n-1]的余子式（选任一结点均可）
    return get_det();
}
```

有向图矩阵树定理

- 根向树形图计数，每条边指向父亲
- （叶向树形图，即每条边指向儿子，只要修改一个地方）
- 如果要求所有根的树形图之和，就求逆的主对角线之和乘以行列式（$A^*=|A|A^{-1}$）

```cpp
void matrix::addedge(int x,int y){
    a[x][y]--;
    a[x][x]++; // 叶向树形图改成a[y][y]++;
}
ll matrix::treecount(int s){ // s是根结点
    // for(auto i:eset)addedge(i.fi,i.se); // 加边
    repeat(i,s,n)
    repeat(j,0,n)
        a[i][j]=a[i+1][j];
    repeat(i,0,n)
    repeat(j,s,n)
        a[i][j]=a[i][j+1];
    n--,m=n; // a[s][s]的余子式
    return get_det();
}
```

BSET 定理

- 有向欧拉图的欧拉回路总数等于任意根的根向树形图个数乘以 $\Pi(deg(v)-1)!$（←阶乘）（$deg(v)$ 是 v 的入度或出度，~~反正入度等于出度~~）

### Prufer 序列

- n 个点的无根树与长度 $n-2$ 值域 $[1,n]$ 的序列有双射关系，Prufer序列就是其中一种
- 性质：i 出现次数等于节点 i 的度 - 1
- 无根树转 Prufer：设无根树点数为 n，每次删除度为 1 且编号最小的结点并把它所连接的点的编号加入 Prufer 序列，进行 $n-2$ 次操作
- Prufer 转无根树：计算每个点的度为在序列中出现的次数加 1，每次找度为 1 的编号最小的点与序列中第一个点连接，并将后者的度减 1
- Cayley 定理：完全图 $K_n$ 有 $n^{n-2}$ 棵生成树
- 扩展：k 个联通块，第 i 个联通块有 $s_i$ 个点，则添加 $k-1$ 条边使整个图联通的方案数有 $\displaystyle n^{k-2}\prod_{i=1}^k s_i$ 个

### LGV 引理

- DAG 上固定 2n 个点 $[A_1,\ldots,A_n,B_1,\ldots,B_n]$，若有 n 条路径 $[A_1→B_1,\ldots,A_n→B_n]$ 两两不相交，则方案数为：

$$P=\det\left[\begin{array}{c}e(A_1,B_1)&\cdots &e(A_1,B_n)\newline \vdots&\ddots&\vdots\newline e(A_n,B_1)&\cdots&e(A_n,B_n)\end{array}\right]=\det\left[e(A_i,B_j)\right]_{i,j=1}^n$$

- 其中 $e(u,v)$ 表示 $u→v$ 的路径计数

### 图论 with 组合数学 归档

Enumerative properties of Ferrers graphs

- 二分图，左顶点连编号为 $1,2,...,a_i$ 的右顶点，则该图的生成树个数为 $\dfrac{\prod_{i\in A}\text{deg}(i)}{\max_{i\in A}\text{deg}(i)}\cdot\dfrac{\prod_{i\in B}\text{deg}(i)}{\max_{i\in B}\text{deg}(i)}$ 左顶点度之积（去掉度最大的）乘以右顶点度之积（去掉度最大的）

***

无向图三元环计数

- 无向图定向，$(\text{deg}(i),i)>(\text{deg}(j),j)\Leftrightarrow$ 建立有向边 (i, j)。然后暴力枚举 u，将 u 的所有儿子 $\omega(u)$ 标记为 dcnt，暴力枚举 $v\in\omega(u)$，若 v 的儿子被标记为 dcnt 则 ans++，$O(E\log E)$

***

- 若一棵无根树的贡献为无根树的节点数，相当于设置一个根的方案数。

### 图论 结论

竞赛图判定 using 兰道定理

- 竞赛图对出度序列排序后有 $\displaystyle\sum_{i=1}^k d_i\ge\dfrac{k(k-1)}{2},k=1,2,\ldots,n$ 且 $k=n$ 时取等号。

```cpp
int n=read(); int ans=1;
repeat(i,1,n+1)a[i]=read(); sort(a+1,a+n+1);
repeat(i,1,n+1)a[i]+=a[i-1];
repeat(i,1,n)if(a[i]<1ll*i*(i-1)/2)ans=0;
if(a[n]!=1ll*n*(n-1)/2)ans=0;
puts(ans?"T":"F");
```

- 另外，若允许出现无向边，出度记为 0.5，定理也成立（存疑）

***

Havel-Hakimi 定理

- 给定一个度序列，反向构造出这个图
- 解：贪心，每次让剩余度最大的顶点 k 连接其余顶点中剩余度最大的 $deg_k$ 个顶点
- （我认为二路归并比较快，可是找到的代码都用了 `sort()`）

***

- 不能连续两次走一条边，可以构造新图，新图顶点为原图的边。（点边交换）[](https://www.luogu.com.cn/problem/P2151)
- Delaunay 三角剖分后，每个线段中垂线构成 Voronoi 图，它们互为对偶图。

## 其他

### 异或字典树

```cpp
namespace trie {
const int N = 12000010;
static const int B = 30;
int top;
struct node {
    int to[2];
    int &operator[](int n) { return to[n]; }
    int &at(int n) {
        if (to[n] == 0) a[top++] = node(), to[n] = top;
        return to[n];
    }
    int lazy;
} a[N];
void init() { top = 0; a[top++] = node(); }
void ige(ll s, ll t) { // x ^ s >= t
    int k = 0;
    repeat_back (i, 0, B) {
        int x = (s >> i) & 1, y = (t >> i) & 1;
        if (x == 0 && y == 0)
            a[a[k].at(1)].lazy++;
        else if (x == 1 && y == 0)
            a[a[k].at(0)].lazy++;
        k=a[k].at(x ^ y);
    }
    a[k].lazy++;
}
void ile(ll s, ll t) { // x ^ s <= t
    int k = 0;
    repeat_back (i, 0, B) {
        int x = (s >> i) & 1, y = (t >> i) & 1;
        if (x == 1 && y == 1)
            a[a[k].at(1)].lazy++;
        else if (x == 0 && y == 1)
            a[a[k].at(0)].lazy++;
        k=a[k].at(x ^ y);
    }
    a[k].lazy++;
}
}
```

### 分散层叠 / Fractional Cascading

refer to 集训队 2020 蒋明润

- 多次询问 x 在 n 个有序数组 $a_i$ 中的 lowerbound。
- 编号从 0 开始，初始化 $O(\sum\text{len }a_i)$，询问 $O(n)$。

```cpp
struct Cascade{
    vector<array<int,3>> c[N];
    int ans[N]; // answer to the query
    void init(vector<int> a[],int n){ // a: raw arrays, n: number of arrays
        repeat(i,0,n)c[i].clear();
        repeat(i,0,a[n-1].size())
            c[n-1].push_back({a[n-1][i],i,0});
        repeat_back(i,0,n-1){
            int p1=0,p2=0;
            auto &y=c[i+1];
            repeat(j,0,a[i].size()){
                while(p1<(int)y.size() && y[p1][0]<=a[i][j])
                    c[i].push_back({y[p1][0],j,p1}),p1+=2;
                while(p2<(int)y.size() && y[p2][0]<a[i][j])
                    p2++;
                c[i].push_back({a[i][j],j,p2});
            }
            while(p1<(int)y.size()){
                c[i].push_back({y[p1][0],(int)a[i].size(),p1});
                p1+=2;
            }
        }
    }
    void query(int x,int n){ // x: query, n: number of arrays
        int pos=lower_bound(c[0].begin(),c[0].end(),array<int,3>{x,0,0})-c[0].begin();
        repeat(i,0,n){
            if(pos && c[i][pos-1][0]>=x)pos--;
            if(pos<(int)c[i].size()){
                ans[i]=a[i][c[i][pos][1]]; // if index is needed, c[i][pos][1]
                pos=c[i][pos][2];
            }
            else{
                ans[i]=0; // no lowerbound
                pos=c[i+1].size();
            }
        }
    }
}c;
```

### Raney 引理

- 设整数序列 $A = [a_1,a_2,...,a_n]$，前缀和 $S_k=a_1+...+a_k$，所有数字之和 $S_n=1$
- 则在 A 的 n 个循环表示中，有且仅有一个序列满足其前缀和 $S_i$ 均大于零
- 证明：画成折线图后最低且最后的那一点作为起点

### 括号序列专题

- refer to [OI-Wiki](https://oi-wiki.org/topic/bracket/)
- 括号序列后继，假设 `"("` 字典序小于 `")"`

```cpp
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
- A053121：设 $f(i,j)$ 表示长度为 i 且存在 j 个未匹配的右括号且不存在未匹配的左括号的括号序列的个数。
- $f(0,0)=1,f(i,j) = f(i-1,j-1)+f(i-1,j+1)$
- $f(n, m) = \dfrac{m+1}{n+1}\dbinom{n+1}{\frac{n-m}{2}}(\text{if } n-m \text{ is even}),0(\text{if } n-m \text{ is odd})$

```cpp
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

### 其他 结论

- 将序列分割成任意段，不能直接二分（每段 $O(n\log n)$）的话需要倍增规约二分（每段 $O(L_i\log L_i)$）
- x, y 异或的二进制 1 的个数 `popcount(x^y)`，相当于，把 x, y 看作超立方体的顶点，这两个点的最短路径。超立方体顶点 x 的连边是 `(x,x^(1ll<<i))`（例：popcount(x^y)为边权，求最小生成树。BFS 处理与超立方体上顶点x最近的实点from[x]，对超立方体每个边(x,y)都生成一个边(from[x],from[y])，然后 Kruskal）
- 一个未知的 01 序列，已知一些区间内 1 的个数的奇偶性，判断是否矛盾，可以用种类并查集。
