# 数据结构

- [数据结构](#数据结构)
  - [并查集 / DSU](#并查集--dsu)
    - [<补充> 可持久化并查集](#补充-可持久化并查集)
  - [树状数组 / BIT](#树状数组--bit)
    - [<补充> 二维超级树状数组](#补充-二维超级树状数组)
  - [ST 表 / Sparse Table](#st-表--sparse-table)
    - [<补充> 猫树](#补充-猫树)
  - [单调队列](#单调队列)
  - [线段树 / Segment Tree](#线段树--segment-tree)
    - [<补充> 高拓展性线段树](#补充-高拓展性线段树)
    - [<补充> 权值线段树（动态开点 线段树合并 线段树分裂）](#补充-权值线段树动态开点-线段树合并-线段树分裂)
    - [<补充> zkw 线段树](#补充-zkw-线段树)
    - [<补充> 可持久化数组](#补充-可持久化数组)
    - [<补充> 主席树](#补充-主席树)
    - [<补充> 李超线段树](#补充-李超线段树)
    - [<补充> 吉老师线段树 / Segbeats](#补充-吉老师线段树--segbeats)
  - [堆](#堆)
    - [可删堆](#可删堆)
    - [左偏树](#左偏树)
  - [平衡树](#平衡树)
    - [无旋 Treap](#无旋-treap)
    - [<补充> 可持久化 Treap](#补充-可持久化-treap)
    - [K-D Tree](#k-d-tree)
    - [Splay](#splay)
    - [动态森林 using LCT](#动态森林-using-lct)
  - [莫队](#莫队)
    - [普通莫队](#普通莫队)
    - [带修莫队](#带修莫队)
    - [回滚莫队](#回滚莫队)
  - [冷门数据结构](#冷门数据结构)
    - [珂朵莉树 / 老司机树](#珂朵莉树--老司机树)
    - [划分树](#划分树)
    - [析合树](#析合树)
  - [造轮子](#造轮子)
    - [struct of 二维数组](#struct-of-二维数组)
    - [Hash 表](#hash-表)

## 并查集 / DSU

精简版并查集

- 只有路径压缩。（特定情况下会被卡成 $O(\log n)$）

```cpp
struct DSU {  // join: d[x] = d[y], query: d[x] == d[y]
    int a[N];
    void init(int n) { iota(a, a + n + 1, 0); }
    int fa(int x) { return a[x] == x ? x : a[x] = fa(a[x]); }
    int &operator[](int x) { return a[fa(x)]; }
} d;
```

普通并查集

- 路径压缩 + 启发式合并，$O(α(n))$，可视为 $O(1)$。

```cpp
struct DSU {
    int a[N], sz[N];
    void init(int n) {
        iota(a, a + n + 1, 0);
        fill(sz, sz + n + 1, 1);
    }
    int fa(int x) {
        return a[x] == x ? x : a[x] = fa(a[x]);
    }
    bool query(int x, int y) { // ask if set x == set y
        return fa(x) == fa(y);
    }
    void join(int x, int y) { // join set x and set y
        x = fa(x), y = fa(y);
        if (x == y) return;
        if (sz[x] > sz[y]) swap(x, y);
        a[x] = y; sz[y] += sz[x];
    }
    int operator[](int x) { return fa(x); }
} d;
```

种类并查集

```cpp
struct DSU{
    int a[N],r[N];
    void init(int n){
        repeat(i,0,n+1)a[i]=i,r[i]=0;
    }
    int plus(int a,int b){ // 关系 a + 关系 b，类似向量相加
        if(a==b)return -a;
        return a+b;
    }
    int inv(int a){ // 关系 a 的逆
        return -a;
    }
    int fa(int x){ // 返回根结点
        if(a[x]==x)return x;
        int f=a[x],ff=fa(f);
        r[x]=plus(r[x],r[f]);
        return a[x]=ff;
    }
    bool query(int x,int y){ // 是否存在关系
        return fa(x)==fa(y);
    }
    int R(int x,int y){ // 查找关系
        return plus(r[x],inv(r[y]));
    }
    void join(int x,int y,int r2){ // 按 r2 关系合并
        r2=plus(R(y,x),r2);
        x=fa(x),y=fa(y);
        a[x]=y,r[x]=r2;
    }
}d;
```

可撤销种类并查集

- （维护是否有奇环）

```cpp
namespace DSU{
    int a[N],r[N],sz[N];
    vector<pair<int *,int>> rec;
    void init(int n){repeat(i,0,n+1)a[i]=i,r[i]=0,sz[i]=1;}
    int plus(int a,int b){return a^b;}
    int inv(int a){return a;}
    int fa(int x){return a[x]==x?x:fa(a[x]);}
    int R(int x){return a[x]==x?r[x]:plus(r[x],R(a[x]));} // relation<x,fa(x)>
    int R(int x,int y){return plus(R(x),inv(R(y)));} // relation<x,y>
    void join(int x,int y,int r2){
        r2=plus(R(y,x),r2); x=fa(x),y=fa(y);
        if(sz[x]>sz[y])swap(x,y),r2=inv(r2);
        rec.push_back({&a[x],a[x]});
        rec.push_back({&r[x],r[x]});
        rec.push_back({&sz[y],sz[y]});
        a[x]=y; r[x]=r2; sz[y]+=sz[x];
    }
    void undo(){
        repeat_back(i,0,rec.size())*rec[i].fi=rec[i].se;
        rec.clear();
    }
}using namespace DSU;
```

### <补充> 可持久化并查集

- 启发式合并，不能路径压缩，$O(\log^2 n)$

```cpp
struct seg *pl; int segl,segr;
struct seg{
    ll a; seg *lc,*rc;
}pool[N*20*3];
pair<seg *,seg *> h[N];
void init(int l,int r){
    segl=l,segr=r; pl=pool;
    iota(in+l,in+r+1,l); h[0].fi=pl; pl->init(l,r);
    fill(in+l,in+r+1,1); h[0].se=++pl; pl->init(l,r);
}
int fa(seg *a,int x){
    int t; while((t=a->query(x))!=x)x=t; return x;
}
void join(seg *&a,seg *&sz,int x,int y){ // a=h[i].fi,sz=h[i].se
    x=fa(a,x); y=fa(a,y);
    if(x!=y){
        int sx=sz->query(x),sy=sz->query(y);
        if(sx<sy)swap(x,y),swap(sx,sy);
        a=update(a,y,x),sz=update(sz,x,sx+sy);
    }
}
```

## 树状数组 / BIT

普通树状数组

- 单点修改 \& 区间和，建树 $O(n)$，修改查询 $O(\log n)$。

```cpp
struct BIT {
    static int lb(int x) { return x & -x; } // lowbit
    ll a[N]; int n, dx;
    void init(int l, int r, ll in[] = nullptr) { // 初始化 range = [l, r], A[i] = in[i]
        dx = -l + 1, n = r - l + 1;
        fill(a, a + n + 1, 0);
        if (in) build(in);
    }
    void build(ll in[]) { // 线性建树 A[i] = in[i]
        repeat(i, 1, n + 1) {
            a[i] = in[i - dx];
            for (int k = 1; k < lb(i); k <<= 1) a[i] += a[i - k];
        }
    }
    void add(int x, ll k) { // 单点加 A[x] += k
        x += dx;
        while (x <= n) a[x] += k, x += lb(x);
    }
    ll sum(int x) { // 前缀和 sum A[start .. x]
        ll ans = 0;
        x += dx;
        while (x > 0) ans += a[x], x -= lb(x);
        return ans;
    }
    void diffadd(int l, int r, ll k) { add(l, k), add(r + 1, -k); } // 差分修改（和 sum(int) 配合使用）
    ll sum(int l, int r) { return sum(r) - sum(l - 1); } // 区间和（和 add(int, ll) 配合使用）
} bit;
```

- 大佬的第 k 小（权值树状数组）

```cpp
int findkth(int k){
    int ans=0,cnt=0;
    for(int i=20;i>=0;--i){
        ans+=1<<i;
        if (ans>=n || cnt+t[ans]>=k)ans-=1<<i;
        else cnt+=t[ans];
    }
    return ans+1;
}
```

超级树状数组

- 基于树状数组，区间加 \& 区间和，$O(\log n)$。

```cpp
struct SPBIT {
    BIT a, a1;
    void init(int l, int r) {
        a.init(l, r);
        a1.init(l, r);
    }
    void add(ll x, ll y, ll k) {
        a.add(x, k);
        a.add(y + 1, -k);
        a1.add(x, k * (x - 1));
        a1.add(y + 1, -k * y);
    }
    ll sum(ll x, ll y) {
        return y * a.sum(y) - (x - 1) * a.sum(x - 1) -
               (a1.sum(y) - a1.sum(x - 1));
    }
} spbit;
```

### <补充> 二维超级树状数组

- 修改查询 $O(\log n\cdot\log m)$。

```cpp
int n,m;
#define lb(x) (x&(-x))
struct BIT{
    ll t[N][N]; // 一倍内存吧
    void init(){
        mst(t,0);
    }
    void add(int x,int y,ll k){ // 位置 (x,y) 加上 k
        // x++,y++; // 如果要从 0 开始编号
        for(int i=x;i<=n;i+=lb(i))
        for(int j=y;j<=m;j+=lb(j))
            t[i][j]+=k;
    }
    ll sum(int x,int y){ // 求 (1..x,1..y) 的和
        // x++,y++; // 如果要从 0 开始编号
        ll ans=0;
        for(int i=x;i!=0;i-=lb(i))
        for(int j=y;j!=0;j-=lb(j))
            ans+=t[i][j];
        return ans;
    }
};
struct SPBIT{
    BIT a,ax,ay,axy;
    void add(int x,int y,int k){
        a.add(x,y,k);
        ax.add(x,y,k*x);
        ay.add(x,y,k*y);
        axy.add(x,y,k*x*y);
    }
    ll sum(int x,int y){
        return a.sum(x,y)*(x*y+x+y+1)
            -ax.sum(x,y)*(y+1)
            -ay.sum(x,y)*(x+1)
            +axy.sum(x,y);
    }
    void add(int x0,int y0,int x1,int y1,int k){ // 区间修改
        add(x0,y0,k);
        add(x0,y1+1,-k);
        add(x1+1,y0,-k);
        add(x1+1,y1+1,k);
    }
    ll sum(int x0,int y0,int x1,int y1){ // 区间查询
        return sum(x1,y1)
            -sum(x0-1,y1)
            -sum(x1,y0-1)
            +sum(x0-1,y0-1);
    }
}spbit;
```

## ST 表 / Sparse Table

普通 ST 表

- 编号从 0 开始，初始化 $O(n\log n)$ 查询 $O(1)$。

```cpp
const int logN = __lg(N) + 1;
struct ST {
    ll U(ll x, ll y) { return min(x, y); }
    ll a[N][logN];
    void init(int n) {
        repeat (i, 0, n)
            a[i][0] = s[i];
        repeat (k, 1, logN)
        repeat (i, 0, n - (1 << k) + 1)
            a[i][k] = U(a[i][k - 1], a[i + (1 << (k - 1))][k - 1]);
    }
    ll query(int l, int r) {
        int s = __lg(r - l + 1);
        return U(a[l][s], a[r - (1 << s) + 1][s]);
    }
} st;
```

二维 ST 表

- 编号从 0 开始，初始化 $O(nm\log n\log m)$ 查询 $O(1)$。

```cpp
struct ST{ // 注意 logN = __lg(N) + 2
    #define logN 9
    #define U(x,y) max(x,y)
    int f[N][N][logN][logN];
    void build(){
        repeat(k,0,logN)
        repeat(l,0,logN)
        repeat(i,0,n-(1<<k)+1)
        repeat(j,0,m-(1<<l)+1){
            int &t=f[i][j][k][l];
            if(k==0 && l==0)t=a[i][j];
            else if(k)
                t=U(f[i][j][k-1][l],f[i+(1<<(k-1))][j][k-1][l]);
            else
                t=U(f[i][j][k][l-1],f[i][j+(1<<(l-1))][k][l-1]);
        }
    }
    int query(int x0,int y0,int x1,int y1){
        int k=__lg(x1-x0+1),l=__lg(y1-y0+1);
        return U(U(U(
            f[x0][y0][k][l],
            f[x1-(1<<k)+1][y0][k][l]),
            f[x0][y1-(1<<l)+1][k][l]),
            f[x1-(1<<k)+1][y1-(1<<l)+1][k][l]);
    }
}st;
```

### <补充> 猫树

- 编号从 0 开始，初始化 $O(n\log n)$ 查询 $O(1)$。
- 树上猫树待补。

```cpp
struct cat {
    #define U(a,b) (a + b)
    #define logN 21
    vector<Node> a[logN];
    vector<Node> v;
    void init() {
        repeat (i, 0, logN) a[i].clear();
        v.clear();
    }
    void push(ll in) {
        v.push_back(Node(in));
        int n = v.size() - 1;
        repeat (s, 1, logN) {
            int len = 1 << s; int l = n / len * len;
            if (n % len == len / 2 - 1) {
                repeat (i, 0, len) a[s].push_back(Node(0));
                a[s][l + len / 2 - 1] = v[l + len / 2 - 1];
                repeat_back (i, 0, len / 2 - 1) a[s][l + i] = U(v[l + i], a[s][l + i + 1]);
            }
            if (n % len == len / 2) a[s][n] = v[n];
            if (n % len > len / 2) a[s][n] = U(a[s][n - 1], v[n]);
        }
    }
    Node query(int l,int r){ // l--, r--;
        if (l == r) return v[l];
        int s = __lg(l ^ r) + 1;
        return U(a[s][l], a[s][r]);
    }
}tr;
```

## 单调队列

- 求所有长度为k的区间中的最大值，线性复杂度

```cpp
struct MQ{ // 查询就用 mq.q.front().first
    deque<pii> q; // first: 保存的最大值; second: 时间戳
    void init(){q.clear();}
    void push(int x,int k){
        static int T=0; T++;
        while(!q.empty() && q.back().fi<=x) // max
            q.pop_back();
        q.push_back({x,T});
        while(!q.empty() && q.front().se<=T-k)
            q.pop_front();
    }
    void work(function<int&(int)> a,int n,int k){ // 原地保存，编号从 0 开始
        init();
        repeat(i,0,n){
            push(a(i),k);
            if(i+1>=k)a(i+1-k)=q.front().fi;
        }
    }
    void work(int a[][N],int n,int m,int k1,int k2){ // 原地保存，编号从 0 开始
        repeat(i,0,n){
            init();
            repeat(j,0,m){
                push(a[i][j],k2);
                if(j+1>=k2)a[i][j+1-k2]=q.front().fi;
            }
        }
        m-=k2-1;
        repeat(j,0,m){
            init();
            repeat(i,0,n){
                push(a[i][j],k1);
                if(i+1>=k1)a[i+1-k1][j]=q.front().fi;
            }
        }
    }
}mq;
// 求 n * m 矩阵中所有 k * k 连续子矩阵最大值之和 // 编号从 1 开始
repeat(i,1,n+1)
    mq.work([&](int x)->int&{return a[i][x+1];},m,k);
repeat(j,1,m-k+2)
    mq.work([&](int x)->int&{return a[x+1][j];},n,k);
ll ans=0; repeat(i,1,n-k+2)repeat(j,1,m-k+2)ans+=a[i][j];
// 或者
mq.work((int(*)[N])&(a[1][1]),n,m,k1,k2);
```

## 线段树 / Segment Tree

- 基本上适用于所有（线段树能实现的）区间+区间
- 我删了修改运算的零元，加了偷懒状态 (state)，~~终于能支持赋值操作.jpg~~
- 初始化: init(), 修改查询: tr->sth()
- U(x,y): 查询运算
- a0: 查询运算的零元
- toz(x): 把 x 加载到懒标记
- toa(): 懒标记加载到数据（z别忘了清空）
- a: 数据, z: 懒标记, state: 懒标记使用状态
- 改写线段树时，能 down 就 down。

```cpp
struct seg {
    ll U(ll x, ll y) { return x + y; }
    static const ll a0 = 0;
    void toz(ll x) { z += x, state = 1; }
    void toa() { a += z * (r - l + 1), z = 0, state = 0; }
    ll a, z; bool state;
    int l, r; seg *lc, *rc; 
    void init(int _l, int _r, seg *&pl) {
        l = _l, r = _r; state = 0; z = 0;
        if (l == r) { a = a0; return; }
        int m = (l + r) >> 1;
        lc = ++pl; lc->init(l, m, pl);
        rc = ++pl; rc->init(m + 1, r, pl);
        up();
    }
    void build(ll in[]) {
        if (l == r) { a = in[l]; return; }
        lc->build(in); rc->build(in);
        up();
    } 
    void up() { a = U(lc->a, rc->a); }
    void down() {
        if (!state) return;
        if (l < r) { lc->toz(z); rc->toz(z); }
        toa();
    }
    void update(int x, int y, ll k) {
        if (x > r || y < l) { down(); return; }
        if (x <= l && y >= r) { toz(k); down(); return; }
        down();
        lc->update(x, y, k);
        rc->update(x, y, k);
        up();
    }
    ll query(int x, int y) {
        if (x > r || y < l) return a0;
        down();
        if (x <= l && y >= r) return a;
        return U(lc->query(x, y), rc->query(x, y));
    }
} tr[N * 2], *pl;
void init(int l, int r, ll in[] = nullptr) {
    pl = tr;
    tr->init(l, r, pl);
    if (in) tr->build(in);
}
```

- 保存一下旧版线段树

```cpp
struct seg {
    #define U(x, y) (x + y)
    #define a0 0
    void toz(ll x) { z += x, state = 1; }
    void toa() { a += z * (r - l + 1), z = 0, state = 0; }
    ll a, z; bool state;
    int l, r; seg *lc, *rc;
    void init(int, int);
    void up() { a = U(lc->a, rc->a); }
    void down() {
        if (!state) return;
        if (l < r) { lc->toz(z); rc->toz(z); }
        toa();
    }
    void update(int x, int y, ll k) {
        if (x > r || y < l) { down(); return; }
        if (x <= l && y >= r) { toz(k); down(); return; }
        down();
        lc->update(x, y, k);
        rc->update(x, y, k);
        up();
    }
    ll query(int x, int y) {
        if (x > r || y < l) return a0;
        down();
        if (x <= l && y >= r) return a;
        return U(lc->query(x, y), rc->query(x, y));
    }
} tr[N * 2], *pl;
void seg::init(int _l, int _r) {
    l = _l, r = _r; state = 0; z = 0;
    if (l == r) { a = in[l]; return; }
    int m = (l + r) >> 1;
    lc = ++pl; lc->init(l, m);
    rc = ++pl; rc->init(m + 1, r);
    up();
}
void init(int l, int r) {
    pl = tr;
    tr->init(l, r);
}
```

### <补充> 高拓展性线段树

- 例：luogu P3373 线段树 2，支持区间加、区间乘、区间和查询

```cpp
struct Z{ // lazy tag
    int x,y; explicit Z(int x=1,int y=0):x(x),y(y){}
    void push(Z b,int l,int r){
        x=(x*b.x)%mod;
        y=(y*b.x+b.y)%mod;
    }
};
struct A{ // seg tag
    int x; explicit A(int x=0):x(x){}
    void push(Z b,int l,int r){
        x=(x*b.x+b.y*(r-l+1))%mod;
    }
    A operator+(A b)const{return A((x+b.x)%mod);}
};
struct seg{
    A a; Z z; bool state; int l,r; seg *lc,*rc;
    void init(int,int);
    void up(){a=lc->a+rc->a;}
    void toz(Z x){z.push(x,l,r),state=1;}
    void down(){
        if(!state)return;
        if(l<r){lc->toz(z); rc->toz(z);}
        a.push(z,l,r),z=Z(),state=0;
    }
    void update(int x,int y,Z k){
        if(x>r || y<l){down(); return;}
        if(x<=l && y>=r){toz(k); down(); return;}
        down();
        lc->update(x,y,k);
        rc->update(x,y,k);
        up();
    }
    A query(int x,int y){
        if(x>r || y<l)return A();
        down(); if(x<=l && y>=r)return a;
        return lc->query(x,y)+rc->query(x,y);
    }
}tr[N*2],*pl;
int in[N];
void seg::init(int _l,int _r){
    l=_l,r=_r; state=0;
    if(l==r){a=in[l]; return;}
    int m=(l+r)>>1;
    lc=++pl; lc->init(l,m);
    rc=++pl; rc->init(m+1,r);
    up();
}
void init(int l,int r){
    pl=tr; tr->init(l,r);
}
```

### <补充> 权值线段树（动态开点 线段树合并 线段树分裂）

- 初始 n 个线段树，支持对某个线段树插入权值、合并两个线段树、查询某个线段树第 k 小数
- 编号从 1 开始，$O(n\log n)$

```cpp
DSU d;
struct seg{
    seg *lc,*rc; int sz;
}tr[N<<5],*pl,*rt[N];
#define LL lc,l,m
#define RR rc,m+1,r
int size(seg *s){return s?s->sz:0;}
seg *newnode(){*pl=seg(); return pl++;}
void up(seg *s){s->sz=size(s->lc)+size(s->rc);}
void insert(seg *&s,int l,int r,int v,int num=1){ // insert v, (s=rt[d[x]])
    if(!s)s=newnode(); s->sz+=num;
    if(l==r)return;
    int m=(l+r)/2;
    if(v<=m)insert(s->LL,v,num);
    else insert(s->RR,v,num);
}
seg *merge(seg *a,seg *b,int l,int r){ // private, return the merged tree
    if(!a)return b; if(!b)return a;
    a->sz+=b->sz;
    if(l==r)return a;
    int m=(l+r)/2;
    a->lc=merge(a->lc,b->LL);
    a->rc=merge(a->rc,b->RR);
    return a;
}
void merge(int x,int y,int l,int r){ // merge tree x and y
    if(d[x]==d[y])return;
    rt[d[x]]=merge(rt[d[x]],rt[d[y]],l,r);
    d[y]=d[x];
}
int kth(seg *s,int l,int r,int k){ // kth in s, (k=1,2,...,sz, s=rt[d[x]])
    if(l==r)return l;
    int m=(l+r)/2,lv=size(s->lc);
    if(k<=lv)return kth(s->LL,k);
    else return kth(s->RR,k-lv);
}
int query(seg *s,int l,int r,int x,int y){ // count the numbers between [x,y] (s=rt[d[x]])
    if(!s || x>r || y<l)return 0;
    if(x<=l && y>=r)return s->sz;
    int m=(l+r)/2;
    return query(s->LL,x,y)+query(s->RR,x,y);
}
void split(seg *&s,int l,int r,int x,int y,seg *&t){ // the numbers between [x,y] trans from s to t, (s=rt[d[x]], t=rt[d[y]])
    if(!s || x>r || y<l)return;
    if(x<=l && y>=r){t=merge(s,t,l,r); s=0; return;}
    if(!t)t=newnode();
    int m=(l+r)/2;
    split(s->LL,x,y,t->lc);
    split(s->RR,x,y,t->rc);
    up(s); up(t);
}
void init(int n){ // create n trees
    pl=tr; d.init(n);
    fill(rt,rt+n+1,nullptr);
}
```

### <补充> zkw 线段树

- 单点 + 区间，编号从 0 开始，建树 $O(n)$ 修改查询 $O(\log n)$。（存在区间 + 区间的 zkw，但是看起来不太好用）
- 代码量和常数都和树状数组差不多。

```cpp
struct zkw {
    ll U(ll x, ll y) { return x + y; } // 询问操作符
    const ll a0 = 0; // 询问操作符的零元
    int n; ll a[N * 4];
    void init(int inn, ll in[] = nullptr) { // 下标从 0 到 inn - 1，初始化 A[x] = a0 或者 A[x] = in[x]
        n = 1; while (n < inn) n <<= 1;
        fill(a + n, a + n * 2, a0);
        if (in) repeat (i, 0, inn) a[n + i] = in[i];
        repeat_back (i, 1, n) up(i);
    }
    void up(int x) { // private
        a[x] = U(a[x * 2], a[x * 2 + 1]);
    }
    void assign(int x, ll k) { // 单点赋值 A[x] = k
        a[x += n] = k;
        while (x >>= 1) up(x);
    }
    void add(int x, ll k) { // 单点加 A[x] += k
        a[x += n] += k;
        while (x >>= 1) up(x);
    }
    ll query(int l, int r) { // 区间询问 U(A[l], ..., A[r])
        ll ans = a0; l += n - 1, r += n + 1;
        while (l + 1 < r){
            if (~l & 1) ans = U(ans, a[l + 1]);
            if ( r & 1) ans = U(ans, a[r - 1]);
            l >>= 1, r >>= 1;
        }
        return ans;
    }
    ll operator[](int x) const { return a[x + n]; } // 单点询问 A[x]
} tr;
```

### <补充> 可持久化数组

- 单点修改并创建新版本：`h[top]=update(h[i],x,v);`（每次 $O(\log n)$ 额外内存）
- 单点查询 `h[i]->query(x);`
- 初始化 $O(n)$，修改查询 $O(\log n)$

```cpp
struct seg *pl; int segl,segr;
struct seg{
    ll a; seg *lc,*rc;
    ll query(int x,int l=segl,int r=segr){
        if(l==r)return a;
        int m=(l+r)>>1;
        if(x<=m)return lc->query(x,l,m);
        else return rc->query(x,m+1,r);
    }
    friend seg *update(seg *u,int x,ll v,int l=segl,int r=segr){
        *++pl=*u; u=pl;
        if(l==r)u->a=v;
        else{
            int m=(l+r)>>1;
            if(x<=m)u->lc=update(u->lc,x,v,l,m);
            else u->rc=update(u->rc,x,v,m+1,r);
        }
        return u;
    }
    void init(int l,int r){
        if(l==r){a=in[l]; return;}
        int m=(l+r)>>1;
        lc=++pl; lc->init(l,m);
        rc=++pl; rc->init(m+1,r);
    }
}pool[N*20],*h[N]; // h: versions
void init(int l,int r){
    segl=l,segr=r;
    pl=pool; pl->init(l,r); h[0]=pl;
}
```

### <补充> 主席树

- 初始化 `init(l,r)`，版本复制 `his[i]=his[j]`（先复制再修改）
- 单点修改 `update(his[i],x,k)`，区间询问 `query(his[i],x,y)`
- 权值线段树 $his[i]\setminus his[j]$ 询问 k 小 `kth(his[i],his[j],k)`
- 静态区间k小：构造 $his[i]$ 为区间 $[1,i]$ 的权值线段树，`kth(his[r],his[l-1],k)` 即区间k小
- 修改询问 $O(\log n)$

```cpp
namespace seg{
    struct{
        ll x; int l,r;
    }a[N<<5];
    int his[N],cnt,l0,r0;
    void init(int l,int r){
        l0=l,r0=r;
        cnt=0;
    }
    void update(int &u,int x,ll k,int l=l0,int r=r0){ // tr[u][x]+=k
        a[++cnt]=a[u]; u=cnt;
        if(l==r){a[u].x+=k; return;}
        int m=(l+r)/2;
        if(x<=m)update(a[u].l,x,k,l,m);
        else update(a[u].r,x,k,m+1,r);
        a[u].x=a[a[u].l].x+a[a[u].r].x;
    }
    ll query(int u,int x,int y,int l=l0,int r=r0){ // sum(tr[u][x..y])
        if(!u || x>r || y<l)return 0;
        if(x<=l && y>=r)return a[u].x;
        int m=(l+r)/2;
        return query(a[u].l,x,y,l,m)+query(a[u].r,x,y,m+1,r);
    }
    ll kth(int u,int v,int k,int l=l0,int r=r0){ // kth in (tr[u]-tr[v])[x..y]
        if(l==r)return l;
        int m=(l+r)/2,lv=a[a[u].l].x-a[a[v].l].x;
        if(k<=lv)return kth(a[u].l,a[v].l,k,l,m);
        else return kth(a[u].r,a[v].r,k-lv,m+1,r);
    }
}using namespace seg;
```

### <补充> 李超线段树

- 支持插入线段、查询所有线段与 $x=x_0$ 交点最高的那条线段
- 修改 $O(\log^2n)$，查询 $O(\log n)$

```cpp
int funx; // 这是y()的参数
struct func{
    lf k,b; int id;
    lf y()const{return k*funx+b;} // funx点处的高度
    bool operator<(const func &b)const{
        return make_pair(y(),-id)<make_pair(b.y(),-b.id);
    }
};
struct seg{ // 初始化init()更新update()查询query()，func::y()是高度
    func a;
    int l,r;
    seg *ch[2];
    void init(int,int);
    void push(func d){
        funx=(l+r)/2;
        if(a<d)swap(a,d); // 这个小于要用funx
        if(l==r)return;
        ch[d.k>a.k]->push(d);
    }
    void update(int x,int y,const func &d){ // 更新[x,y]区间
        x=max(x,l); y=min(y,r); if(x>y)return;
        if(x==l && y==r)push(d);
        else{
            ch[0]->update(x,y,d);
            ch[1]->update(x,y,d);
        }
    }
    const func &query(int x){ // 询问
        funx=x;
        if(l==r)return a;
        const func &b=ch[(l+r)/2<x]->query(x);
        return max(a,b); // 这个max要用funx
    }
}tr[N*2],*pl;
void seg::init(int _l,int _r){
    l=_l,r=_r; a={0,-inf,-1}; // 可能随题意改变
    if(l==r)return;
    int m=(l+r)/2;
    (ch[0]=++pl)->init(l,m);
    (ch[1]=++pl)->init(m+1,r);
}
void init(int l,int r){
    pl=tr;
    tr->init(l,r);
}
void add(int x0,int y0,int x1,int y1){ // 线段处理并更新
    if(x0>x1)swap(x0,x1),swap(y0,y1);
    lf k,b;
    if(x0==x1)k=0,b=max(y0,y1);
    else{
        k=lf(y1-y0)/(x1-x0);
        b=y0-k*x0;
    }
    id++;
    tr->update(x0,x1,{k,b,id});
}
```

### <补充> 吉老师线段树 / Segbeats

区间最值操作

- 区间取 min，区间 max，区间和，$O(n\log n)$

```cpp
int in[N],n;
struct seg{
    #define lc (u*2)
    #define rc (u*2+1)
    int mx[N<<2],se[N<<2],cnt[N<<2],tag[N<<2];
    ll sum[N<<2];
    void up(int u){ // private
        int x=lc,y=rc;
        sum[u]=sum[x]+sum[y];
        if(mx[x]==mx[y]){
            mx[u]=mx[x];
            se[u]=max(se[x],se[y]);
            cnt[u]=cnt[x]+cnt[y];
        }
        else{
            if(mx[x]<mx[y])swap(x,y);
            mx[u]=mx[x];
            se[u]=max(se[x],mx[y]);
            cnt[u]=cnt[x];
        }
    }
    void pushtag(int u,int tg){ // private
        if(mx[u]<=tg)return;
        sum[u]+=(1ll*tg-mx[u])*cnt[u];
        mx[u]=tag[u]=tg;
    }
    void down(int u){ // private
        if(tag[u]==-1)return;
        pushtag(lc,tag[u]),pushtag(rc,tag[u]);
        tag[u]=-1;
    }
    void build(int u=1,int l=1,int r=n){
        tag[u]=-1;
        if(l==r){
            sum[u]=mx[u]=in[l],se[u]=-1,cnt[u]=1;
            return;
        }
        int m=(l+r)>>1;
        build(lc,l,m),build(rc,m+1,r);
        up(u);
    }
    void tomin(int x,int y,int v,int u=1,int l=1,int r=n){
        if(x>r || l>y || mx[u]<=v)return;
        if(x<=l && r<=y && se[u]<v)return pushtag(u,v);
        int m=(l+r)>>1; down(u);
        tomin(x,y,v,lc,l,m);
        tomin(x,y,v,rc,m+1,r);
        up(u);
    }
    int qmax(int x,int y,int u=1,int l=1,int r=n){
        if(x<=l && r<=y)return mx[u];
        if(x>r || l>y)return -inf;
        int m=(l+r)>>1; down(u);
        return max(qmax(x,y,lc,l,m),qmax(x,y,rc,m+1,r));
    }
    ll qsum(int x,int y,int u=1,int l=1,int r=n){
        if(x<=l && r<=y)return sum[u];
        if(x>r || l>y)return 0;
        int m=(l+r)>>1; down(u);
        return qsum(x,y,lc,l,m)+qsum(x,y,rc,m+1,r);
    }
}tr;
```

- 区间取 min，区间取 max，区间加，区间 min，区间 max，区间和，$O(n\log^2 n)$

```cpp
int in[N],n;
struct seg{
    #define lc (u*2)
    #define rc (u*2+1)
    struct node{
        int mx,mx2,mn,mn2,cmx,cmn,tmx,tmn,tad;
        ll sum;
    };
    node t[N<<2];
    void up(int u){ // private
        int x=lc,y=rc;
        t[u].sum=t[x].sum+t[y].sum;
        if(t[x].mx==t[y].mx){
            t[u].mx=t[x].mx,t[u].cmx=t[x].cmx+t[y].cmx;
            t[u].mx2=max(t[x].mx2,t[y].mx2);
        }
        else{
            if(t[x].mx<t[y].mx)swap(x,y);
            t[u].mx=t[x].mx,t[u].cmx=t[x].cmx;
            t[u].mx2=max(t[x].mx2,t[y].mx);
        }
        if(t[x].mn==t[y].mn){
            t[u].mn=t[x].mn,t[u].cmn=t[x].cmn+t[y].cmn;
            t[u].mn2=min(t[x].mn2,t[y].mn2);
        }
        else{
            if(t[x].mn>t[y].mn)swap(x,y);
            t[u].mn=t[x].mn,t[u].cmn=t[x].cmn;
            t[u].mn2=min(t[x].mn2,t[y].mn);
        }
    }
    void push_add(int u,int l,int r,int v){ // private
        t[u].sum+=(r-l+1ll)* v;
        t[u].mx+=v,t[u].mn+=v;
        if(t[u].mx2!=-inf)t[u].mx2+=v;
        if(t[u].mn2!=inf)t[u].mn2+=v;
        if(t[u].tmx!=-inf)t[u].tmx+=v;
        if(t[u].tmn!=inf)t[u].tmn+=v;
        t[u].tad+=v;
    }
    void push_min(int u,int tg){ // private
        if(t[u].mx<=tg)return;
        t[u].sum+=(tg*1ll-t[u].mx)*t[u].cmx;
        if(t[u].mn2==t[u].mx)t[u].mn2=tg;
        if(t[u].mn==t[u].mx)t[u].mn=tg;
        if(t[u].tmx>tg)t[u].tmx=tg;
        t[u].mx=tg,t[u].tmn=tg;
    }
    void push_max(int u,int tg){ // private
        if(t[u].mn>tg)return;
        t[u].sum+=(tg*1ll-t[u].mn)*t[u].cmn;
        if(t[u].mx2==t[u].mn)t[u].mx2=tg;
        if(t[u].mx==t[u].mn)t[u].mx=tg;
        if(t[u].tmn<tg)t[u].tmn=tg;
        t[u].mn=tg,t[u].tmx=tg;
    }
    void down(int u,int l,int r){ // private
        const int m=(l+r)>>1;
        if(t[u].tad)
            push_add(lc,l,m,t[u].tad),push_add(rc,m+1,r,t[u].tad);
        if(t[u].tmx!=-inf)push_max(lc,t[u].tmx),push_max(rc,t[u].tmx);
        if(t[u].tmn!=inf)push_min(lc,t[u].tmn),push_min(rc,t[u].tmn);
        t[u].tad=0,t[u].tmx=-inf,t[u].tmn=inf;
    }
    void build(int u=1,int l=1,int r=n){
        t[u].tmn=inf,t[u].tmx=-inf;
        if(l==r){
            t[u].sum=t[u].mx=t[u].mn=in[l];
            t[u].mx2=-inf,t[u].mn2=inf;
            t[u].cmx=t[u].cmn=1;
            return;
        }
        int m=(l+r)>>1;
        build(lc,l,m),build(rc,m+1,r);
        up(u);
    }
    void add(int x,int y,int v,int u=1,int l=1,int r=n){
        if(y<l || r<x)return;
        if(x<=l && r<=y)return push_add(u,l,r,v);
        int m=(l+r)>>1;
        down(u,l,r);
        add(x,y,v,lc,l,m),add(x,y,v,rc,m+1,r);
        up(u);
    }
    void tomin(int x,int y,int v,int u=1,int l=1,int r=n){
        if(y<l || r<x || t[u].mx<=v)return;
        if(x<=l && r<=y && t[u].mx2<v)return push_min(u,v);
        int m=(l+r)>>1;
        down(u,l,r);
        tomin(x,y,v,lc,l,m),tomin(x,y,v,rc,m+1,r);
        up(u);
    }
    void tomax(int x,int y,int v,int u=1,int l=1,int r=n){
        if(y<l || r<x || t[u].mn>=v)return;
        if(x<=l && r<=y && t[u].mn2>v)return push_max(u,v);
        int m=(l+r)>>1;
        down(u,l,r);
        tomax(x,y,v,lc,l,m),tomax(x,y,v,rc,m+1,r);
        up(u);
    }
    ll qsum(int x,int y,int u=1,int l=1,int r=n){
        if(y<l || r<x)return 0;
        if(x<=l && r<=y)return t[u].sum;
        int m=(l+r)>>1;
        down(u,l,r);
        return qsum(x,y,lc,l,m)+qsum(x,y,rc,m+1,r);
    }
    ll qmax(int x,int y,int u=1,int l=1,int r=n){
        if(y<l || r<x)return -inf;
        if(x<=l && r<=y)return t[u].mx;
        int m=(l+r)>>1;
        down(u,l,r);
        return max(qmax(x,y,lc,l,m),qmax(x,y,rc,m+1,r));
    }
    ll qmin(int x,int y,int u=1,int l=1,int r=n){
        if(y<l || r<x)return inf;
        if(x<=l && r<=y)return t[u].mn;
        int m=(l+r)>> 1;
        down(u,l,r);
        return min(qmin(x,y,lc,l,m),qmin(x,y,rc,m+1,r));
    }
}tr;
```

区间历史最值

- [模板题](https://www.luogu.com.cn/problem/P6242)，支持区间加、区间取min、区间和、区间max、区间历史max（$\max_{i\in[1,t]j\in [l,r]}h_i[j]$），$O(\log^2n)$

```cpp
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

## 堆

### 可删堆

- 原理：双堆模拟。

```cpp
struct Heap{
    priority_queue<int> a,b;  // heap=a-b
    void push(int x){a.push(x);}
    void erase(int x){b.push(x);}
    int top(){
        while(!b.empty() && a.top()==b.top())
            a.pop(),b.pop();
        return a.top();
    }
    void pop(){
        while(!b.empty() && a.top()==b.top())
            a.pop(),b.pop();
        a.pop();
    }
    int top2(){ // 次大值
        int t=top(); pop();
        int ans=top(); push(t);
        return ans;
    }
    int size(){return a.size()-b.size();}
    void clear() { a = b = priority_queue<int>(); }
};
```

### 左偏树

- 万年不用，$O(\log n)$。

```cpp
struct leftist{ // 编号从 1 开始，因为空的左右儿子会指向 0
    #define lc LC[x]
    #define rc RC[x]
    vector<int> val,dis,exist,dsu,LC,RC;
    void init(){add(0);dis[0]=-1;}
    void add(int v){
        int t=val.size();
        val.pb(v);
        dis.pb(0);
        exist.pb(1);
        dsu.pb(t);
        LC.pb(0);
        RC.pb(0);
    }
    int top(int x){
        return dsu[x]==x?x:dsu[x]=top(dsu[x]);
    }
    void join(int x,int y){
        if(exist[x] && exist[y] && top(x)!=top(y))
            merge(top(x),top(y));
    }
    int merge(int x,int y){
        if(!x || !y)return x+y;
        if(val[x]<val[y]) // 大根堆
            swap(x,y);
        rc=merge(rc,y);
        if(dis[lc]<dis[rc])
            swap(lc,rc);
        dsu[lc]=dsu[rc]=dsu[x]=x;
        dis[x]=dis[rc]+1;
        return x;
    }
    void pop(int x){
        x=top(x);
        exist[x]=0;
        dsu[lc]=lc;
        dsu[rc]=rc;
        dsu[x]=merge(lc,rc); // 指向x的dsu也能正确指向top
    }
    #undef lc
    #undef rc
}lt;
// 添加元素lt.add(v),位置是lt.val.size()-1
// 是否未被pop：lt.exist(x)
// 合并：lt.join(x,y)
// 堆顶：lt.val[lt.top(x)]
// 弹出：lt.pop(x)
```

```cpp
struct node{
    int v,dis;
    node *l,*r;
}pool[N],*pl,*tr[N];
int dis(node *u){return u?u->dis:0;}
node *newnode(){*pl=node(); return pl++;}
void init(){
    pl=pool;
}
node *merge(node *x,node *y){
    if(!x)return y; if(!y)return x;
    if(x->v < y->v)swap(x,y);
    x->r=merge(x->r,y);
    if(dis(x->l) < dis(x->r))swap(x->l,x->r);
    x->dis=dis(x->r)+1;
    return x;
}
void pop(node *&u){u=merge(u->l,u->r);}
int top(node *u){return u->v;}
```

## 平衡树

### 无旋 Treap

- 普通平衡树按 v 分裂，文艺平衡树按 sz 分裂。
- `insert,erase` 操作在普通平衡树中，`push_back,output(dfs)` 在文艺平衡树中。
- `build` (笛卡尔树线性构建) 在普通平衡树中。
- 普通平衡树

```cpp
struct treap{
    struct node{
        int pri,v,sz;
        node *l,*r;
        node(int _v){pri=rnd(); v=_v; l=r=0; sz=1;}
        node(){}
        friend int size(node *u){return u?u->sz:0;}
        void up(){sz=1+size(l)+size(r);}
        friend pair<node *,node *> split(node *u,int key){ // 按v分裂
            if(u==0)return {0,0};
            if(key<u->v){
                auto o=split(u->l,key);
                u->l=o.se; u->up();
                return {o.fi,u};
            }
            else{
                auto o=split(u->r,key);
                u->r=o.fi; u->up();
                return {u,o.se};
            }
        }
        friend node *merge(node *x,node *y){
            if(x==0)return y;
            if(y==0)return x;
            if(x->pri>y->pri){
                x->r=merge(x->r,y); x->up();
                return x;
            }
            else{
                y->l=merge(x,y->l); y->up();
                return y;
            }
        }
        int find_by_order(int ord){
            if(ord==size(l))return v;
            if(ord<size(l))return l->find_by_order(ord);
            else return r->find_by_order(ord-size(l)-1);
        }
    }pool[N],*pl,*rt;
    void init(){
        pl=pool;
        rt=0;
        // top=0;
    }
    /*
    node *stk[N]; int top;
    void build(int pri,int v){ // v is nondecreasing
        node *i=++pl; *i=node(v); i->pri=pri;
        int cur=top;
        while(cur && stk[cur]->pri<i->pri)cur--;
        if(cur<top)i->l=stk[cur+1],up(i);
        if(cur)stk[cur]->r=i,up(stk[cur]); else rt=i;
        stk[++cur]=i;
        top=cur;
    }
    */
    void insert(int key){
        auto o=split(rt,key);
        *++pl=node(key);
        o.fi=merge(o.fi,pl);
        rt=merge(o.fi,o.se);
    }
    void erase_all(int key){
        auto o=split(rt,key-1),s=split(o.se,key);
        rt=merge(o.fi,s.se);
    }
    void erase_one(int key){
        auto o=split(rt,key-1),s=split(o.se,key);
        rt=merge(o.fi,merge(merge(s.fi->l,s.fi->r),s.se));
    }
    int order(int key){
        auto o=split(rt,key-1);
        int ans=size(o.fi);
        rt=merge(o.fi,o.se);
        return ans;
    }
    int operator[](int x){
        return rt->find_by_order(x);
    }
    int lower_bound(int key){
        auto o=split(rt,key-1);
        int ans=o.se->find_by_order(0);
        rt=merge(o.fi,o.se);
        return ans;
    }
    int nxt(int key){return lower_bound(key+1);}
}tr;
// if(opt==1)tr.insert(x);
// if(opt==2)tr.erase_one(x);
// if(opt==3)cout<<tr.order(x)+1<<endl; // x的排名
// if(opt==4)cout<<tr[x-1]<<endl; // 排名为x
// if(opt==5)cout<<tr[tr.order(x)-1]<<endl; // 前驱
// if(opt==6)cout<<tr.nxt(x)<<endl; // 后继
```

- 文艺平衡树，`tag` 表示翻转子树（区间）

```cpp
struct treap{
    struct node{
        int pri,v,sz,tag;
        node *l,*r;
        node(int _v){pri=(int)rnd(); v=_v; l=r=0; sz=1; tag=0;}
        node(){}
        friend int size(node *u){return u?u->sz:0;}
        void up(){sz=1+size(l)+size(r);}
        void down(){
            if(tag){
                swap(l,r);
                if(l)l->tag^=1;
                if(r)r->tag^=1;
                tag=0;
            }
        }
        friend pair<node *,node *> split(node *u,int key){ // 按sz分裂
            if(u==0)return {0,0};
            u->down();
            if(key<size(u->l)){
                auto o=split(u->l,key);
                u->l=o.se; u->up();
                return {o.fi,u};
            }
            else{
                auto o=split(u->r,key-size(u->l)-1);
                u->r=o.fi; u->up();
                return {u,o.se};
            }
        }
        friend node *merge(node *x,node *y){
            if(x==0 || y==0)return max(x,y);
            if(x->pri>y->pri){
                x->down();
                x->r=merge(x->r,y); x->up();
                return x;
            }
            else{
                y->down();
                y->l=merge(x,y->l); y->up();
                return y;
            }
        }
    }pool[N],*pl,*rt;
    void init(){
        pl=pool;
        rt=0;
    }
    void push_back(int v){
        *++pl=node(v);
        rt=merge(rt,pl);
    }
    void add_tag(int l,int r){ // 编号从0开始
        node *a,*b,*c;
        tie(a,b)=split(rt,l-1);
        tie(b,c)=split(b,r-l);
        if(b)b->tag^=1;
        rt=merge(a,merge(b,c));
    }
    void output(node *u){
        if(u==0)return; u->down();
        output(u->l); cout<<u->v<<' '; output(u->r);
    }
}tr;
```

带懒标记 Treap

```cpp
struct treap{
    struct node{
        int pri,sz; ll v,tag,s;
        node *l,*r;
        node(int _v){pri=(int)rnd(); v=s=_v; l=r=0; sz=1; tag=0;}
        node(){}
        friend int size(node *u){return u?u->sz:0;}
        friend ll sum(node *u){return u?(u->down(),u->s):0;}
        void up(){ // private
            sz=1+size(l)+size(r);
            s=v+sum(l)+sum(r);
        }
        void down(){ // private
            if(tag){
                v+=tag; s+=sz*tag;
                if(l)l->tag+=tag;
                if(r)r->tag+=tag;
                tag=0;
            }
        }
        friend pair<node *,node *> split(node *u,int key){ // private
            if(u==0)return {0,0};
            u->down();
            if(key<size(u->l)){
                auto o=split(u->l,key);
                u->l=o.se; u->up();
                return {o.fi,u};
            }
            else{
                auto o=split(u->r,key-size(u->l)-1);
                u->r=o.fi; u->up();
                return {u,o.se};
            }
        }
        friend node *merge(node *x,node *y){ // private
            if(x==0 || y==0)return max(x,y);
            if(x->pri>y->pri){
                x->down();
                x->r=merge(x->r,y); x->up();
                return x;
            }
            else{
                y->down();
                y->l=merge(x,y->l); y->up();
                return y;
            }
        }
    }pool[N],*pl,*rt;
    void init(){
        pl=pool;
        rt=0;
    }
    void push_back(int v){
        *++pl=node(v);
        rt=merge(rt,pl);
    }
    void add_tag(int l,int r,ll tag){
        node *a,*b,*c;
        tie(a,b)=split(rt,l-1);
        tie(b,c)=split(b,r-l);
        if(b)b->tag+=tag;
        rt=merge(a,merge(b,c));
    }
    ll query(int l,int r){
        node *a,*b,*c;
        tie(a,b)=split(rt,l-1);
        tie(b,c)=split(b,r-l);
        ll ans=sum(b);
        rt=merge(a,merge(b,c));
        return ans;
    }
    void output(node *u){
        if(u==0)return; u->down();
        output(u->l); cout<<u->v<<' '; output(u->r);
    }
}tr;
```

### <补充> 可持久化 Treap

- 其他函数从 `treap` 板子里照搬（但是把 `rt` 作为参数）
- `h[i]=h[j]` 就是克隆
- 普通平衡树

```cpp
struct node *pl;
struct node{
    int pri,v,sz;
    node *l,*r;
    node(int _v){pri=rnd(); v=_v; l=r=0; sz=1;}
    node(){}
    friend int size(node *u){return u?u->sz:0;}
    void up(){sz=1+size(l)+size(r);}
    friend pair<node *,node *> split(node *u,int key){
        if(u==0)return {0,0};
        node *w=++pl; *w=*u;
        if(key<u->v){
            auto o=split(u->l,key);
            w->l=o.se; w->up();
            return {o.fi,w};
        }
        else{
            auto o=split(u->r,key);
            w->r=o.fi; w->up();
            return {w,o.se};
        }
    }
    friend node *merge(node *x,node *y){
        if(x==0)return y;
        if(y==0)return x;
        node *w=++pl;
        if(x->pri>y->pri){
            *w=*x;
            w->r=merge(x->r,y);
        }
        else{
            *w=*y;
            w->l=merge(x,y->l);
        }
        w->up();
        return w;
    }
}pool[N*60],*h[N];
```

### K-D Tree

- K-D tree 可以维护多维空间的点集，用替罪羊树的方法保证复杂度。
- 建树、询问近邻参考第二段代码。
- [模板题](https://www.luogu.com.cn/problem/P4148)，支持在线在 (x, y) 处插入值、查询二维区间和。
- 插入的复杂度为 $O(\log n)$。
- 二维区间查询最坏复杂度为 $O(n^{1-\tfrac 1 k})=O(\sqrt n)$。
- 询问近邻等很多骚操作的最坏复杂度为 $O(n)$，最好用别的算法替代。

```cpp
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

```cpp
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

```cpp
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

### Splay

- 均摊 $O(\log n)$

```cpp
struct Splay{
    struct node{
        node *ch[2],*fa;
        int v,sz;
        void up(){sz=ch[0]->sz+ch[1]->sz+1;}
        node(int _v){v=_v; ch[0]=ch[1]=fa=Null; up();}
        node(){ch[0]=ch[1]=fa=Null; sz=0;}
        bool cmp(int key){return v<key;}
    }*rt;
    bool get(node *u){return u==u->fa->ch[1];}
    static node Null[N],*pl;
    void init(){
        rt=pl=Null;
    }
    void rotate(node *r){ // private
        node *u=r->fa;
        bool d=get(r);
        if(r->ch[d^1]!=Null)r->ch[d^1]->fa=u;
        if(u->fa!=Null)u->fa->ch[get(u)]=r;
        u->ch[d]=r->ch[d^1];
        r->ch[d^1]=u;
        r->fa=u->fa; u->fa=r;
        u->up(); r->up();
    }
    node *splay(node *u){ // private
        for(node *f=u->fa;f!=Null;rotate(u),f=u->fa)
            if(f->fa!=Null)rotate(get(u)==get(f)?f:u);
        return rt=u;
    }
    void insert(int v){
        node *u=rt,*f=Null;
        while(u!=Null){
            f=u; u=u->ch[u->cmp(v)];
        }
        *++pl=node(v); pl->fa=f;
        if(f!=Null)f->ch[f->cmp(v)]=pl;
        splay(pl);
    }
    node *merge(node *u,node *r){ // private
        if(u==Null)return r;
        if(r==Null)return u;
        while(u->ch[1]!=Null)u=u->ch[1];
        splay(u);
        u->ch[1]=r; r->fa=u; u->up();
        return u;
    }
    node *find(node *u,int key){ // private
        if(u->v==key)return u;
        return find(u->ch[u->cmp(key)],key);
    }
    void erase(int key){
        splay(find(rt,key));
        node *u=rt->ch[0],*v=rt->ch[1];
        u->fa=Null; v->fa=Null;
        rt=merge(u,v);
    }
    int order(int key){
        int ans=0; node *u=rt,*f=Null;
        while(u!=Null){
            int d=u->cmp(key);
            if(d)ans+=u->ch[0]->sz+1;
            f=u; u=u->ch[d];
        }
        splay(f);
        return ans;
    }
    int kth(int k){
        node *u=rt;
        while(1){
            int s=u->ch[0]->sz;
            if(k==s){splay(u); return u->v;}
            if(k>s)k-=s+1,u=u->ch[1];
            else u=u->ch[0];
        }
    }
}tr;
Splay::node Splay::Null[N],*Splay::pl;
// tr.insert(x); // 插入 x
// tr.erase(x); // 删除单个 x
// printf("%d\n",tr.order(x)+1); // x 的排名
// printf("%d\n",tr.kth(x-1)); // 排名为 x
// printf("%d\n",tr.kth(tr.order(x)-1)); // x 的前驱
// printf("%d\n",tr.kth(tr.order(x+1))); // x 的后继
```

文艺

```cpp
struct Splay{
    struct node{
        node *ch[2],*fa;
        int v,sz,tag;
        void up(){sz=ch[0]->sz+ch[1]->sz+1;}
        node(int _v){v=_v; ch[0]=ch[1]=fa=Null; tag=0; up();}
        node(){ch[0]=ch[1]=fa=Null; sz=0;}
        bool cmp(int key){return v<key;}
        void down(){
            if(tag){
                swap(ch[0],ch[1]);
                ch[0]->tag^=1;
                ch[1]->tag^=1;
                tag=0;
            }
        }
    }*rt;
    bool get(node *u){return u==u->fa->ch[1];}
    static node Null[N],*pl;
    void init(){
        rt=pl=Null;
    }
    void rotate(node *r){ // private
        node *u=r->fa;
        bool d=get(r);
        if(r->ch[d^1]!=Null)r->ch[d^1]->fa=u;
        if(u->fa!=Null)u->fa->ch[get(u)]=r;
        u->ch[d]=r->ch[d^1];
        r->ch[d^1]=u;
        r->fa=u->fa; u->fa=r;
        u->up(); r->up();
    }
    node *splay(node *u){ // private
        for(node *f=u->fa;f!=Null;rotate(u),f=u->fa)
            if(f->fa!=Null)rotate(get(u)==get(f)?f:u);
        return rt=u;
    }
    template<typename ptr>
    void build(ptr l,ptr r){
        init(); sort(l,r);
        for(auto i=l;i!=r;i++){
            *++pl=node(*i);
            if(rt!=Null)rt->fa=pl;
            pl->ch[0]=rt;
            rt=pl;
            rt->up();
        }
    }
    node *kth(node *u,int k){
        while(1){
            u->down();
            int s=u->ch[0]->sz;
            if(k==s){splay(u); return u;}
            if(k>s)k-=s+1,u=u->ch[1];
            else u=u->ch[0];
        }
    }
    node *merge(node *u,node *r){ // private
        if(u==Null)return r;
        if(r==Null)return u;
        while(u->down(),u->ch[1]!=Null)u=u->ch[1];
        splay(u);
        u->ch[1]=r; r->fa=u; u->up();
        return u;
    }
    pair<node *,node *> split(node *u,int k){ // private
        if(k==-1)return {Null,u};
        u=kth(u,k);
        node *v=u->ch[1];
        if(v!=Null){u->ch[1]=Null; v->fa=Null; u->up();}
        return {u,v};
    }
    void add_tag(int l,int r){
        node *a,*b,*c;
        tie(a,b)=split(rt,l-1);
        tie(b,c)=split(b,r-l);
        b->tag^=1;
        rt=merge(a,merge(b,c));
    }
    void output(node *u){
        if(u==Null)return; u->down();
        output(u->ch[0]); cout<<u->v<<' '; output(u->ch[1]);
    }
    void print(){
        output(rt); cout<<endl;
    }
}tr;
Splay::node Splay::Null[N],*Splay::pl;
```

### 动态森林 using LCT

- $O(\log n)$ 但是常数很大。

```cpp
#define lc ch[0]
#define rc ch[1]
struct lct{
    struct node{
        node *ch[2],*fa;
        int v,s; bool rev;
        node(){lc=rc=fa=Null; v=s=0;}
        node(int _v){lc=rc=fa=Null; v=_v; up();}
        void up(){s=lc->s ^ rc->s ^ v;}
        bool nroot(){
            return fa->lc==this || fa->rc==this;
        }
        void down(){
            if(rev){
                swap(lc,rc);
                lc->rev^=1;
                rc->rev^=1;
                rev=0;
            }
        }
    };
    static node Null[N],*pl;
    void init(){
        pl=Null;
    }
    void rotate(node *u){
        node *f=u->fa,*z=f->fa;
        bool k=(f->rc==u);
        node *w=u->ch[!k];
        if(f->nroot())z->ch[z->rc==f]=u;
        u->ch[!k]=f; f->ch[k]=w;
        if(w!=Null)w->fa=f;
        f->fa=u; u->fa=z;
        f->up();
    }
    void downchain(node *u){
        if(u->nroot())downchain(u->fa);
        u->down();
    }
    void splay(node *u){
        downchain(u);
        while(u->nroot()){
            node *f=u->fa,*z=f->fa;
            if(f->nroot())
                rotate((f->lc==u)!=(z->lc==f)?u:f);
            rotate(u);
        }
        u->up();
    }
    void access(node *u){
        for(node *c=Null;u!=Null;c=u,u=u->fa)
            splay(u),u->rc=c,u->up();
    }
    void makeroot(node *u){
        access(u);
        splay(u);
        u->rev^=1;
    }
    node *findroot(node *u){
        access(u);
        splay(u);
        while(u->lc!=Null)u->down(),u=u->lc;
        splay(u);
        return u;
    }
    void split(node *x,node *y){
        makeroot(x);
        access(y);
        splay(y);
    }
    void link(node *x,node *y){
        makeroot(x);
        if(findroot(y)!=x)x->fa=y;
    }
    void cut(node *x,node *y){
        makeroot(x);
        if(findroot(y)==x && y->fa==x && y->lc==Null){
            y->fa=x->rc=Null;
            x->up();
        }
    }
}tr;
lct::node lct::Null[N],*lct::pl;
lct::node *a[N];
void Solve(){
    int n=read(),m=read();
    tr.init();
    repeat(i,1,n+1){
        *++lct::pl=lct::node(read());
        a[i]=lct::pl;
    }
    while(m--){
        int op=read(),x=read(),y=read();
        if(op==0){
            tr.split(a[x],a[y]); // 将 x 到 y 的路径剖出来
            print(a[y]->s,1);
        }
        if(op==1){
            tr.link(a[x],a[y]); // 连接 x, y
        }
        if(op==2){
            tr.cut(a[x],a[y]); // 删除 (x, y) 边
        }
        if(op==3){
            tr.splay(a[x]);
            a[x]->v=y;
        }
    }
}
```

## 莫队

- 离线（甚至在线）处理区间问题，~~猛得一批~~

### 普通莫队

- 移动指针 l, r 来求所有区间的答案
- 块大小为 $\sqrt n$，$O(n^{\tfrac 3 2})$

```cpp
int unit,n,bkt[N],a[N],final[N]; // bkt是桶
ll ans;
struct node{
    int l,r,id;
    bool operator<(const node &b)const{
        if(l/unit!=b.l/unit)return l<b.l; // 按块排序
        if((l/unit)&1) // 奇偶化排序
            return r<b.r;
        return r>b.r;
    }
};
vector<node> query; // 查询区间
void update(int x,int d){
    int &b=bkt[a[x]];
    ans-=C(b,2); // 操作示例
    b+=d;
    ans+=C(b,2); // 操作示例
}
void solve(){ // final[]即最终答案
    fill(bkt,bkt+n+1,0);
    unit=sqrt(n)+1;
    sort(query.begin(),query.end());
    int l=1,r=0; ans=0; // 如果原数组a编号从1开始
    for(auto i:query){
        while(l<i.l)update(l++,-1);
        while(l>i.l)update(--l,1);
        while(r<i.r)update(++r,1);
        while(r>i.r)update(r--,-1);
        final[i.id]=ans;
    }
}
// repeat(i,0,m)query.push_back({read(),read(),i}); // 输入查询区间
```

### 带修莫队

- 相比与普通莫队，多了一个时间轴。
- 块大小为 $\sqrt[3]{nt}$，$O(\sqrt[3]{n^4t})$。

```cpp
int unit,n,bkt[1000010],a[N],final[N];
ll ans;
struct node{
    int l,r,t,id;
    bool operator<(const node &b)const{
        if(l/unit!=b.l/unit)return l<b.l;
        if(r/unit!=b.r/unit)return r<b.r;
        return r/unit%2?t<b.t:t>b.t;
    }
};
struct node2{int x,pre,nxt;};
vector<node> query;
vector<node2> change;
void update(int x,int d){
    int &b=bkt[a[x]];
    ans-=!!b;
    b+=d;
    ans+=!!b;
}
void solve(){
    unit=pow(1.0*n*change.size(),1.0/3)+1;
    sort(query.begin(),query.end());
    int l=1,r=0,t=change.size(); ans=0;
    for(auto i:query){
        while(l<i.l)update(l++,-1);
        while(l>i.l)update(--l,1);
        while(r<i.r)update(++r,1);
        while(r>i.r)update(r--,-1);
        while(t<i.t){
            int f=(change[t].x>=i.l && change[t].x<=i.r);
            if(f)update(change[t].x,-1);
            a[change[t].x]=change[t].nxt;
            if(f)update(change[t].x,1);
            t++;
        }
        while(t>i.t){
            t--;
            int f=(change[t].x>=i.l && change[t].x<=i.r);
            if(f)update(change[t].x,-1);
            a[change[t].x]=change[t].pre;
            if(f)update(change[t].x,1);
        }
        final[i.id]=ans;
    }
}
void Solve(){
    n=read(); int q=read();
    repeat(i,1,n+1)a[i]=read();
    while(q--){
        static char s[10]; scanf("%s",s);
        if(*s=='Q'){
            int l=read(),r=read();
            query.push_back((node){l,r,(int)change.size(),(int)query.size()});
        }
        else{
            int x=read(),y=read();
            change.push_back((node2){x,a[x],y});
            a[x]=y;
        }
    }
    solve();
    repeat(i,0,query.size())
        printf("%d\n",final[i]);
}
```

### 回滚莫队

- 解决区间只能扩张不能收缩的问题。
- 对于块内区间，直接暴力；对于所有左端点在一个块的区间，从块的右端点出发，先扩张右边界再扩张左边界再回滚左边界。
- $O(n^{\tfrac 3 2})$。

```cpp
int n,unit;
ll a[N],ans[N];
typedef array<int,3> node; // l,r,id 
vector<node> q;
struct bucket{
    ll a[N],vis[N],dcnt; ll ans=0;
    vector<pair<ll *,ll>> rec;
    void init(){dcnt++; ans=0; rec.clear();}
    void push(ll x,int flag){
        if(vis[x]!=dcnt)vis[x]=dcnt,a[x]=0;
        if(flag){
            rec.push_back({&a[x],a[x]});
            rec.push_back({&ans,ans});
        }
        a[x]++;
        ans=max(ans,x*a[x]);
    }
    void rollback(){
        repeat_back(i,0,rec.size())*rec[i].fi=rec[i].se;
        rec.clear();
    }
}bkt;
void Solve(){
    n=read(); int Q=read(); unit=sqrt(n)+1;
    repeat(i,0,n)a[i]=read();
    repeat(i,0,Q){
        int x=read()-1,y=read()-1;
        if(x/unit==y/unit){
            bkt.init();
            repeat(k,x,y+1)bkt.push(a[k],0);
            ans[i]=bkt.ans;
        }
        else q.push_back({x,y,i});
    }
    sort(q.begin(),q.end(),[](node a,node b){
        return pii(a[0]/unit,a[1])<pii(b[0]/unit,b[1]);
    });
    int p=0;
    for(int i=unit;i<n;i+=unit){
        bkt.init(); int r=i-1;
        while(p!=(int)q.size() && q[p][0]<i){
            while(r<q[p][1])bkt.push(a[++r],0);
            repeat(j,q[p][0],i)bkt.push(a[j],1);
            ans[q[p][2]]=bkt.ans;
            bkt.rollback();
            p++;
        }
    }
    repeat(i,0,Q)printf("%lld\n",ans[i]);
}
```

## 冷门数据结构

### 珂朵莉树 / 老司机树

- 珂朵莉数以区间形式存储数据，~~非常暴力~~，适用于有区间赋值操作且数据随机的题。
- 对于随机数据有均摊 $O(n\log\log n)$，不随机数据可能被卡。

```cpp
struct ODT{
    struct node{
        int l,r;
        mutable int v; // 强制可修改
        bool operator<(const node &b)const{return l<b.l;}
    };
    set<node> a;
    void init(){
        a.clear();
        a.insert({-inf,inf,0});
    }
    set<node>::iterator split(int x){ // 分裂区间
        auto it=--a.upper_bound({x,0,0});
        if(it->l==x)return it;
        int l=it->l,r=it->r,v=it->v;
        a.erase(it);
        a.insert({l,x-1,v});
        return a.insert({x,r,v}).first;
    }
    void assign(int l,int r,int v){ // 区间赋值
        auto y=split(r+1),x=split(l);
        a.erase(x,y);
        a.insert({l,r,v});
    }
    int sum(int l,int r){ // 操作示例：区间求和
        auto y=split(r+1),x=split(l);
        int ans=0;
        for(auto i=x;i!=y;i++){
            ans+=(i->r-i->l+1)*i->v;
        }
        return ans;
    }
}odt;
```

### 划分树

- 静态区间第 k 小，可代替主席树
- 编号从 1 开始，初始化 $O(n\log n)$，查询 $O(\log n)$

```cpp
struct divtree{
    int a[N],pos[25][N],tr[25][N],n;
    void build(int l,int r,int dep){ // private
        if(l==r)return;
        int m=(l+r)>>1;
        int same=m-l+1;
        repeat(i,l,r+1)
            same-=(tr[dep][i]<a[m]);
        int ls=l,rs=m+1;
        repeat(i,l,r+1){
            int flag=0;
            if(tr[dep][i]<a[m] || (tr[dep][i]==a[m] && same>0)){
                flag=1;
                tr[dep+1][ls++]=tr[dep][i];
                same-=(tr[dep][i]==a[m]);
            }
            else{
                tr[dep+1][rs++]=tr[dep][i];
            }
            pos[dep][i]=pos[dep][i-1]+flag;
        }
        build(l,m,dep+1);
        build(m+1,r,dep+1);
    }
    int query(int ql,int qr,int k,int L,int R,int dep=0){ // private
        if(ql==qr)
            return tr[dep][ql];
        int m=(L+R)>>1;
        int x=pos[dep][ql-1]-pos[dep][L-1];
        int y=pos[dep][qr]-pos[dep][L-1];
        int rx=ql-L-x,ry=qr-L-y;
        int cnt=y-x;
        if(cnt>=k)
            return query(L+x,L+y-1,k,L,m,dep+1);
        else
            return query(m+rx+1,m+1+ry,k-cnt,m+1,R,dep+1);
    }
    int mink(int l,int r,int k){ // k>=1, k<=r-l+1
        return query(l,r,k,1,n);
    }
    int maxk(int l,int r,int k){ // k>=1, k<=r-l+1
        return query(l,r,r-l+2-k,1,n);
    }
    void init(int _n){
        n=_n;
        repeat(i,1,n+1)tr[0][i]=a[i]=in[i];
        sort(a+1,a+n+1);
        build(1,n,0);
    }
}tr;
```

### 析合树

- 定义连续段为一个区间，区间内所有数排序后为连续正整数
- 析合树：每个节点对应一个连续段，合点由儿子顺序或倒序组成，析点为乱序
- 给定一个排列，询问包含给定区间的最短连续段
- 注意代码里N已经是两倍大小了，编号从 1 开始，$O(n\log n)$

```cpp
int n, m, a[N], st1[N], st2[N], tp1, tp2, rt;
int L[N], R[N], M[N], id[N], cnt, typ[N], st[N], tp;
struct RMQ {
    int mn[N][17], mx[N][17];
    void build() {
        repeat(i, 1, n + 1) mn[i][0] = mx[i][0] = a[i];
        repeat(i, 1, 17) repeat(j, 1, n - (1 << i) + 2) {
            mn[j][i] = min(mn[j][i - 1], mn[j + (1 << (i - 1))][i - 1]);
            mx[j][i] = max(mx[j][i - 1], mx[j + (1 << (i - 1))][i - 1]);
        }
    }
    int qmin(int l, int r) {
        int t = __lg(r - l + 1);
        return min(mn[l][t], mn[r - (1 << t) + 1][t]);
    }
    int qmax(int l, int r) {
        int t = __lg(r - l + 1);
        return max(mx[l][t], mx[r - (1 << t) + 1][t]);
    }
} D;
#define ls (k << 1)
#define rs (k << 1 | 1)
struct SEG {
    int a[N << 1], z[N << 1];
    void up(int k) { a[k] = min(a[ls], a[rs]); }
    void mfy(int k, int v) { a[k] += v, z[k] += v; }
    void down(int k) {
        if (z[k]) mfy(ls, z[k]), mfy(rs, z[k]), z[k] = 0;
    }
    void update(int k, int l, int r, int x, int y, int v) {
        if (x > r || y < l) return;
        if (x<=l && r<=y) {
            mfy(k, v);
            return;
        }
        down(k);
        int mid = (l + r) >> 1;
        update(ls, l, mid, x, y, v);
        update(rs, mid + 1, r, x, y, v);
        up(k);
    }
    int query(int k, int l, int r) {
        if (l == r) return l;
        down(k);
        int mid = (l + r) >> 1;
        return a[ls] == 0 ? query(ls, l, mid) : query(rs, mid + 1, r);
    }
} T;
int dep[N], fa[N][18];
vector<int> e[N];
void add(int u, int v) { e[u].push_back(v); }
void dfs(int u) {
    repeat(i, 1, __lg(dep[u]) + 1) fa[u][i] = fa[fa[u][i - 1]][i - 1];
    for (auto v : e[u]) {
        dep[v] = dep[u] + 1;
        fa[v][0] = u;
        dfs(v);
    }
}
int go(int u, int d) {
    for (int i = 0; i < 18 && d; ++i)
        if (d & (1 << i)) d ^= 1 << i, u = fa[u][i];
    return u;
}
int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    u = go(u, dep[u] - dep[v]);
    if (u == v) return u;
    for (int i = 17; ~i; --i)
        if (fa[u][i] != fa[v][i]) u = fa[u][i], v = fa[v][i];
    return fa[u][0];
}
bool judge(int l, int r) { return D.qmax(l, r) - D.qmin(l, r) == r - l; }
void build() {
    repeat(i, 1, n + 1) {
        while (tp1 && a[i] <= a[st1[tp1]])
            T.update(1, 1, n, st1[tp1 - 1] + 1, st1[tp1], a[st1[tp1]]), tp1--;
        while (tp2 && a[i] >= a[st2[tp2]])
            T.update(1, 1, n, st2[tp2 - 1] + 1, st2[tp2], -a[st2[tp2]]), tp2--;
        T.update(1, 1, n, st1[tp1] + 1, i, -a[i]);
        st1[++tp1] = i;
        T.update(1, 1, n, st2[tp2] + 1, i, a[i]);
        st2[++tp2] = i;
        id[i] = ++cnt;
        L[cnt] = R[cnt] = i;
        int le = T.query(1, 1, n), now = cnt;
        while (tp && L[st[tp]] >= le) {
            if (typ[st[tp]] && judge(M[st[tp]], i)) {
                R[st[tp]] = i, add(st[tp], now), now = st[tp--];
            } else if (judge(L[st[tp]], i)) {
                typ[++cnt] = 1;
                L[cnt] = L[st[tp]], R[cnt] = i, M[cnt] = L[now];
                add(cnt, st[tp--]), add(cnt, now);
                now = cnt;
            } else {
                add(++cnt, now);
                do {
                    add(cnt, st[tp--]);
                } while (tp && !judge(L[st[tp]], i));
                L[cnt] = L[st[tp]], R[cnt] = i, add(cnt, st[tp--]);
                now = cnt;
            }
        }
        st[++tp] = now;
        T.update(1, 1, n, 1, i, -1);
    }
    rt = st[1];
}
void query(int &l, int &r) {
    int x = id[l], y = id[r];
    int z = lca(x, y);
    if (typ[z] & 1)
        l = L[go(x, dep[x] - dep[z] - 1)], r = R[go(y, dep[y] - dep[z] - 1)];
    else
        l = L[z], r = R[z];
}
int main() {
    scanf("%d", &n);
    repeat(i, 1, n + 1) scanf("%d", &a[i]);
    D.build();
    build();
    dfs(rt);
    scanf("%d", &m);
    while (m--) {
        int x, y;
        scanf("%d%d", &x, &y);
        query(x, y);
        printf("%d %d\n", x, y);
    }
    return 0;
}
```

## 造轮子

### struct of 二维数组

- （可以存储类似 $n\times m\le 2\times 10^5$ 的二维数组）

```cpp
struct mat {
    ll a[N]; int n, m;
    void init(int _n, int _m) { n = _n, m = _m; }
    ll *operator[](int x) { return a + x * m; }
    /*
    void print(){
        repeat(i,0,n)
        repeat(j,0,m)
            printf("%3lld%c",a[i*m+j]," \n"[j==m-1]);
    }
    friend mat T(mat &a){
        mat b; b(a.m,a.n);
        repeat(i,0,b.n)
        repeat(j,0,b.m)
            b[i][j]=a[j][i];
        return b;
    }
    */
};
```

### Hash 表

- 该 Hash 表耗时约为 unordered_map 的 40%。
- uN 取值表：

```text
1009, 2003, 3001, 5003,
10007, 20011, 30011, 50021,
100003, 200003, 300007, 500009,
1000003, 2000003, 3000017, 5000011,
10000019, 20000003, 30000023
```

```cpp
template<typename A, typename B>
struct unmap {
    struct node {
        A u; B v; int nxt;
    };
    static const unsigned uN = 20000003;
    vector<node> e;
    int head[uN];
    unmap() { clear(); }
    void clear() { memset(head, -1, sizeof head); e.clear(); }
    bool count(A u) {
        int h = u % uN;
        for (int i = head[h]; ~i; i = e[i].nxt)
            if (e[i].u == u) return 1;
        return 0;
    }
    B &operator[](A u) {
        int h = u % uN;
        for (int i = head[h]; ~i; i = e[i].nxt)
            if (e[i].u == u) return e[i].v;
        e.push_back({u, B(), head[h]}); head[h] = e.size() - 1;
        return e.back().v;
    }
    void foreach(const function<void(B &)> &f) {
        for (int h : head)
        for (int i = h; ~i; i = e[i].nxt)
            f(e[i].v);
    }
};
```
