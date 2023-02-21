# 图论

- [图论](#图论)
  - [图论基础](#图论基础)
    - [邻接表](#邻接表)
    - [前向星](#前向星)
    - [拓扑排序 / Toposort](#拓扑排序--toposort)
  - [线段树优化建图](#线段树优化建图)
  - [最短路径](#最短路径)
    - [单源正权 using Dijkstra](#单源正权-using-dijkstra)
    - [多源 using Floyd](#多源-using-floyd)
    - [一般单源 using SPFA](#一般单源-using-spfa)
    - [多源 using Johnson](#多源-using-johnson)
    - [n2 Dijkstra](#n2-dijkstra)
  - [最小生成树 / MST](#最小生成树--mst)
    - [Kruskal](#kruskal)
    - [Boruvka](#boruvka)
    - [n2 Prim](#n2-prim)
  - [树论](#树论)
    - [树的直径](#树的直径)
    - [树的重心](#树的重心)
    - [最近公共祖先 / LCA](#最近公共祖先--lca)
      - [树上倍增解法](#树上倍增解法)
      - [欧拉序列建 ST 表解法](#欧拉序列建-st-表解法)
      - [树链剖分解法](#树链剖分解法)
      - [Tarjan 解法](#tarjan-解法)
      - [一些关于lca的问题](#一些关于lca的问题)
    - [树链剖分](#树链剖分)
    - [树分治](#树分治)
      - [点分治](#点分治)
    - [虚树](#虚树)
    - [树上启发式合并](#树上启发式合并)
  - [联通性相关](#联通性相关)
    - [强联通分量 SCC + 缩点](#强联通分量-scc--缩点)
    - [边双连通分量 using Tarjan](#边双连通分量-using-tarjan)
    - [割点 / 割顶](#割点--割顶)
  - [2-Sat 问题](#2-sat-问题)
  - [支配树 using Lengauer-Tarjan 算法](#支配树-using-lengauer-tarjan-算法)
  - [图上的NP问题](#图上的np问题)
    - [最大团 and 极大团计数](#最大团-and-极大团计数)
    - [最小染色数](#最小染色数)
  - [仙人掌 using 圆方树](#仙人掌-using-圆方树)
  - [二分图](#二分图)
    - [二分图匹配 / 最大匹配](#二分图匹配--最大匹配)
    - [二分图最大权匹配 using KM](#二分图最大权匹配-using-km)
    - [稳定婚姻 using 延迟认可](#稳定婚姻-using-延迟认可)
    - [一般图最大匹配 using 带花树](#一般图最大匹配-using-带花树)
    - [一般图最大权匹配 using 带权带花树](#一般图最大权匹配-using-带权带花树)
  - [网络流](#网络流)
    - [最大流 using Dinic](#最大流-using-dinic)
    - [最小费用最大流 using MCMF](#最小费用最大流-using-mcmf)
    - [上下界网络流](#上下界网络流)
  - [图论杂项](#图论杂项)
    - [Kruskal 重构树](#kruskal-重构树)
    - [DSU 重构树](#dsu-重构树)

## 图论基础

### 邻接表

- 通用化的尝试

```cpp
struct edge {
    ll y, w;
    ll to() { return y; }
    ll dis() { return w; }
};
vector<edge> a[N];
```

### 前向星

```cpp
struct edge { int to, w, nxt; }; // 指向，权值，下一条边
vector<edge> a;
int head[N];
void addedge(int x, int y, int w) {
    a.push_back({y, w, head[x]});
    head[x] = a.size() - 1;
}
void init(int n) {
    a.clear();
    fill(head, head + n, -1);
}
// for (int i = head[x]; i != -1; i = a[i].nxt) // 遍历 x 出发的边
```

### 拓扑排序 / Toposort

- $O(V+E)$。

```cpp
vector<int> topo;
void toposort(int n){
    static int deg[N]; fill(deg,deg+n,0);
    static queue<int> q;
    repeat(x,0,n)for(auto p:a[x])deg[p]++;
    repeat(i,0,n)if(deg[i]==0)q.push(i);
    while(!q.empty()){
        int x=q.front(); q.pop(); topo.push_back(x);
        for(auto p:a[x])if(--deg[p]==0)q.push(p);
    }
}
```

## 线段树优化建图

- 建两棵线段树，第一棵每个结点连向其左右儿子，第二棵每个结点连向其父亲，两棵树所有叶子对应连无向边。
- `add(x1, y1, x2, y2, w)` 表示 $[x_1,y_1]$ 每个结点向 $[x_2,y_2]$ 每个结点连 w 边。
- `a[i+tr.n]` 表示结点 i。
- 建议 10 倍内存，编号从 0 开始，$O(n\log n)$。

```cpp
typedef vector<pii> node;
node a[N]; int top;
struct seg{
    int n;
    void init(int inn){
        for(n=1;n<inn;n<<=1); top=n*4;
        repeat(i,0,n*4)a[i].clear();
        repeat(i,1,n){
            a[i]<<pii(i*2,0);
            a[i]<<pii(i*2+1,0);
            a[i*2+n*2]<<pii(i+n*2,0);
            a[i*2+1+n*2]<<pii(i+n*2,0);
        }
        repeat(i,0,inn){
            a[i+n]<<pii(i+n*3,0);
            a[i+n*3]<<pii(i+n,0);
        }
    }
    void b_add(int l,int r,int x,int w){
        for(l+=n-1,r+=n+1;l^r^1;l>>=1,r>>=1){
            if(~l & 1)a[(l^1)+n*2]<<pii(x,w);
            if(r & 1)a[(r^1)+n*2]<<pii(x,w);
        }
    }
    void a_add(int l,int r,int x,int w){
        for(l+=n-1,r+=n+1;l^r^1;l>>=1,r>>=1){
            if(~l & 1)a[x]<<pii(l^1,w);
            if(r & 1)a[x]<<pii(r^1,w);
        }
    }
}tr;
void add(int x1,int y1,int x2,int y2,int w){
    int s=top++; a[s].clear();
    tr.b_add(x1,y1,s,w);
    tr.a_add(x2,y2,s,0);
}
int f(int x){return x+tr.n;}
```

## 最短路径

### 单源正权 using Dijkstra

- 仅限正权，$O(E\log E)$

```cpp
struct node {
    int to; ll dis;
    bool operator<(const node &b) const {
        return dis > b.dis;
    }
};
bool vis[N];
vector<node> a[N];
ll dis[N]; // result
void dij(int s, int n){ // s: start
    fill(vis, vis + n + 1, 0);
    fill(dis, dis + n + 1, INF); dis[s] = 0; // last[s] = -1;
    static priority_queue<node> q; q.push({s, 0});
    while (!q.empty()) {
        int x = q.top().to; q.pop();
        if (vis[x]) { continue; } vis[x] = 1;
        for (auto i : a[x]) {
            int p = i.to;
            if (dis[p] > dis[x] + i.dis) {
                dis[p] = dis[x] + i.dis;
                q.push({p, dis[p]});
                // last[p] = x; // last 可以记录最短路（倒着）
            }
        }
    }
}
```

### 多源 using Floyd

- $O(V^3)$

```cpp
repeat(k,0,n)
repeat(i,0,n)
repeat(j,0,n)
    f[i][j]=min(f[i][j],f[i][k]+f[k][j]);
```

- 补充：`bitset` 优化（只考虑是否可达），$O(V^3)$

```cpp
// bitset<N> g<N>;
repeat(i,0,n)
repeat(j,0,n)
if(g[j][i])
    g[j]|=g[i];
```

### 一般单源 using SPFA

- SPFA搜索中，有一个点入队 $n+1$ 次即存在负环
- 编号从 0 开始，$O(VE)$

```cpp
int cnt[N]; bool vis[N]; ll h[N]; // h意思和dis差不多，但是Johnson里需要区分
int n;
struct node{int to; ll dis;};
vector<node> a[N];
bool spfa(int s){ // 返回是否有负环（s为起点）
    repeat(i,0,n+1)
        cnt[i]=vis[i]=0,h[i]=inf;
    h[s]=0; // last[s]=-1;
    static deque<int> q; q.assign(1,s);
    while(!q.empty()){
        int x=q.front(); q.pop_front();
        vis[x]=0;
        for(auto i:a[x]){
            int p=i.to;
            if(h[p]>h[x]+i.dis){
                h[p]=h[x]+i.dis;
                // last[p]=x; // last可以记录最短路（倒着）
                if(vis[p])continue;
                vis[p]=1;
                q.push_back(p); // 可以SLF优化
                if(++cnt[p]>n)return 1;
            }
        }
    }
    return 0;
}
bool negcycle(){ // 返回是否有负环
    a[n].clear();
    repeat(i,0,n)
        a[n].push_back({i,0}); // 加超级源点
    return spfa(n);
}
```

### 多源 using Johnson

- SPFA + Dijkstra 实现多源最短路，编号从 0 开始，$O(VE\log E)$

```cpp
ll dis[N][N];
bool jn(){ // 返回是否成功
    if(negcycle())return 0;
    repeat(x,0,n)
    for(auto &i:a[x])
        i.dis+=h[x]-h[i.to];
    repeat(x,0,n)dij(x,dis[x]);
    repeat(x,0,n)
    repeat(p,0,n)
    if(dis[x][p]!=inf)
        dis[x][p]+=h[p]-h[x];
    return 1;
}
```

### n2 Dijkstra

- $O(n^2)$，未测

```cpp
bool vis[N]; int g[N][N], dis[N];
void dij(int s, int n) {
    fill(vis, vis + n, 0);
    fill(dis, dis + n + 1, inf); dis[s] = 0;
    repeat (i, 0, n) {
        int x = n;
        repeat (j, 0, n) if (!vis[j] && dis[j] < dis[x]) x = j;
        if (x == n) return; vis[x] = 1;
        repeat (p, 0, n) if (!vis[p]) dis[p] = min(dis[p], dis[x] + g[x][p]);
    }
}
```

## 最小生成树 / MST

### Kruskal

- 对边长排序，然后添边，并查集判联通，$O(E\log E)$，排序是瓶颈

```cpp
DSU d;
struct edge { int u, v, dis; };
vector<edge> e;
ll kru(int n) {
    ll ans = 0, cnt = 0; d.init(n);
    sort(e.begin(), e.end(), [](edge a, edge b) {
        return a.dis < b.dis;
    });
    for (auto i : e) {
        int x = d[i.u], y = d[i.v];
        if (x == y) continue;
        d[x] = d[y];
        ans += i.dis;
        cnt++;
        if (cnt == n - 1) break;
    }
    if (cnt != n - 1) return -1;
    else return ans;
}
```

### Boruvka

- 类似Prim算法，但是可以多路增广（~~名词迷惑行为~~），$O(E\log V)$

```cpp
DSU d;
struct edge{int u,v,dis;}e[200010];
ll bor(){
    ll ans=0;
    d.init(n);
    e[m].dis=inf;
    vector<int> b; // 记录每个联通块的增广路（名词迷惑行为）
    bool f=1;
    while(f){
        b.assign(n,m);
        repeat(i,0,m){
            int x=d[e[i].u],y=d[e[i].v];
            if(x==y)continue;
            if(e[i].dis<e[b[x]].dis)
                b[x]=i;
            if(e[i].dis<e[b[y]].dis)
                b[y]=i;
        }
        f=0;
        for(auto i:b)
        if(i!=m){
            int x=d[e[i].u],y=d[e[i].v];
            if(x==y)continue;
            ans+=e[i].dis;
            d.join(x,y);
            f=1;
        }
    }
    return ans;
}
```

### n2 Prim

- $O(n^2)$

```cpp
bool vis[N]; int g[N][N], dis[N];
int prim(int n) {
    fill(vis, vis + n, 0); fill(dis, dis + n + 1, inf);
    dis[0] = 0; int ans = 0;
    repeat (i, 0, n) {
        int p = n;
        repeat (j, 0, n) if (!vis[j] && dis[j] < dis[p]) p = j;
        if (p == n) return -1; vis[p] = 1; ans += dis[p];
        repeat (j, 0, n) if (!vis[j]) dis[j] = min(dis[j], g[p][j]);
    }
    return ans;
}
```

## 树论

### 树的直径

- 直径：即最长路径
- 求直径：以任意一点出发所能达到的最远结点为一个端点，以这个端点出发所能达到的最远结点为另一个端点（也可以树上dp）

### 树的重心

- 重心：以重心为根，其最大儿子子树最小
- 性质
  - 以重心为根，所有子树大小不超过整棵树的一半
  - 重心最多有两个
  - 重心到所有结点距离之和最小
  - 两棵树通过一条边相连，则新树的重心在是原来两棵树重心的路径上
  - 一棵树添加或删除一个叶子，重心最多移动一条边的距离
  - 重心不一定在直径上

```cpp
void dfs(int x,int fa=-1){
    static int sz[N],maxx[N];
    sz[x]=1; maxx[x]=0;
    for(auto p:a[x])if(p!=fa){
        dfs(p,x);
        maxx[x]=max(maxx[x],sz[p]);
        sz[x]+=sz[p];
    }
    maxx[x]=max(maxx[x],n-sz[x]);
    if(maxx[x]<maxx[rt])rt=x;
}
```

### 最近公共祖先 / LCA

#### 树上倍增解法

- sum 保存链上倍增信息，可用于查询链最小值等操作。
- 编号从哪开始都可以，初始化 $O(n\log n)$，查询 $O(\log n)$。

```cpp
namespace lca {
    ll U(ll x, ll y) { return x + y; }
    vector<edge> *e = b;
    int dep[N], fa[N][22];
    ll sum[N][22];
    void dfs(int x) { // private
        repeat (i, 1, 22) {
            fa[x][i] = fa[fa[x][i - 1]][i - 1];
            sum[x][i] = U(sum[x][i - 1], sum[fa[x][i - 1]][i - 1]);
        }
        for (auto i : e[x]) {
            int p = i.to();
            if (p == fa[x][0]) continue;
            fa[p][0] = x, dep[p] = dep[x] + 1;
            sum[p][0] = i.dis();
            dfs(p);
        }
    }
    void init(int s) {
        fa[s][0] = s;
        dep[s] = 1;
        dfs(s);
    }
    int lca(int x, int y) {
        if (dep[x] < dep[y]) swap(x, y);
        while (dep[x] > dep[y]) x = fa[x][__lg(dep[x] - dep[y])];
        if (x == y) return x;
        repeat_back (i, 0, 22)
        if (fa[x][i] != fa[y][i])
            x = fa[x][i], y = fa[y][i];
        return fa[x][0];
    }
    int la(int x, int len) { // father ^ len (x)
        repeat (i, 0, 22)
        if (len >> i & 1)
            x = fa[x][i];
        return x;
    }
    ll qsumla(int x, int len) { // query from x to father ^ len (x)
        ll ans = 0;
        repeat (i, 0, 22)
        if (len >> i & 1) {
            ans = U(ans, sum[x][i]);
            x = fa[x][i];
        }
        return ans;
    }
    ll qsum(int x, int y) { // query from x to y
        int l = lca(x, y);
        return U(qsumla(x, dep[x] - dep[l]), qsumla(y, dep[y] - dep[l]));
    }
}
```

#### 欧拉序列建 ST 表解法

- 编号从 0 开始，初始化 $O(n\log n)$，查询 $O(1)$

```cpp
int n, m;
vector<int> eu;
vector<edge> e[N];
int pos[N], dep[N], len[N];
#define mininarr(a, x, y) (a[x] < a[y] ? x : y)
struct RMQ {
#define logN 21
    int f[N * 2][logN];
    void build() {
        int n = eu.size();
        repeat(i, 0, n) f[i][0] = eu[i];
        repeat(k, 1, logN)
        repeat(i, 0, n - (1 << k) + 1)
            f[i][k] = mininarr(dep, f[i][k - 1], f[i + (1 << (k - 1))][k - 1]);
    }
    int query(int l, int r) {
        if (l > r) swap(l, r);  // !!
        int s = __lg(r - l + 1);
        return mininarr(dep, f[l][s], f[r - (1 << s) + 1][s]);
    }
} rmq;
void dfs(int x, int fa = -1) {
    eu.push_back(x);
    pos[x] = eu.size() - 1;
    for (auto i : e[x]) {
        int p = i.to();
        if (p == fa) continue;
        dep[p] = dep[x] + 1;
        len[p] = len[x] + i.dis();
        dfs(p, x);
        eu.push_back(x);
    }
}
void init(int s) {
    eu.clear();
    dep[s] = len[s] = 0;
    dfs(s);
    rmq.build();
}
int lca(int x, int y) { return rmq.query(pos[x], pos[y]); }
```

#### 树链剖分解法

- 编号从哪开始都可以，初始化 $O(n)$，查询 $O(\log n)$

```cpp
vector<edge> a[N];
namespace lca {
    vector<edge> *e = a;
    int dep[N], dis[N], son[N], sz[N], top[N], fa[N]; // son: heaviest son, top: top of the chain
    void dfs1(int x) { // get (dep, sz, son, fa), private
        sz[x] = 1;
        son[x] = -1;
        for (auto i : e[x]) {
            int p = i.to();
            if (p == fa[x]) continue;
            fa[p] = x; dep[p] = dep[x] + 1;
            dis[p] = dis[x] + i.dis();
            dfs1(p);
            sz[x] += sz[p];
            if (son[x] == -1 || sz[son[x]] < sz[p])
                son[x] = p;
        }
    }
    void dfs2(int x, int tv) { // get top, private
        top[x] = tv;
        if (son[x] == -1) return;
        dfs2(son[x], tv);
        for (auto i : e[x]) {
            int p = i.to();
            if (p == fa[x] || p == son[x]) continue;
            dfs2(p, p);
        }
    }
    void init(int s) { // s is the root
        fa[s] = -1; dep[s] = dis[s] = 0;
        dfs1(s);
        dfs2(s, s);
    }
    int lca(int x, int y) {
        while (top[x] != top[y])
            if (dep[top[x]] >= dep[top[y]]) x = fa[top[x]];
            else y = fa[top[y]];
        return dep[x] < dep[y] ? x : y;
    }
}
```

#### Tarjan 解法

- 离线算法，基于并查集
- qry 和 ans 编号从 0 开始，$O(n+m)$，大常数（不看好）

```cpp
vector<int> e[N]; vector<pii> qry,q[N]; // qry输入
DSU d; bool vis[N]; int ans[N]; // ans输出
void dfs(int x){
    vis[x]=1;
    for(auto i:q[x])if(vis[i.fi])ans[i.se]=d[i.fi];
    for(auto p:e[x])if(!vis[p])dfs(p),d[p]=x;
}
void solve(int n,int s){
    repeat(i,0,qry.size()){
        q[qry[i].fi].push_back({qry[i].se,i});
        q[qry[i].se].push_back({qry[i].fi,i});
    }
    d.init(n); dfs(s);
}
```

#### 一些关于lca的问题

```cpp
int length(int x,int y){ // 路径长度
    return dep[x]+dep[y]-2*dep[lca(x,y)];
}
```

```cpp
int intersection(int x,int y,int xx,int yy){ // 树上两条路径公共点个数
    int t[4]={lca(x,xx),lca(x,yy),lca(y,xx),lca(y,yy)};
    sort(t,t+4,[](int x,int y){return dep[x]<dep[y];});
    int r=lca(x,y),rr=lca(xx,yy);
    if(dep[t[0]]<min(dep[r],dep[rr]) || dep[t[2]]<max(dep[r],dep[rr]))
        return 0;
    int tt=lca(t[2],t[3]);
    return 1+dep[t[2]]+dep[t[3]]-dep[tt]*2;
}
```

### 树链剖分

- 编号从 0 开始，处理链 $O(\log^2 n)$，处理子树 $O(\log n)$

```cpp
vector<int> e[N];
int dep[N],son[N],sz[N],top[N],fa[N];
int id[N],arcid[N],idcnt; // id[x]:结点x在树剖序中的位置，arcid相反
void dfs1(int x){
    sz[x]=1; son[x]=-1; dep[x]=dep[fa[x]]+1;
    for(auto p:e[x]){
        if(p==fa[x])continue;
        fa[p]=x; dfs1(p);
        sz[x]+=sz[p];
        if(son[x]==-1 || sz[son[x]]<sz[p])
            son[x]=p;
    }
}
void dfs2(int x,int tv){
    arcid[idcnt]=x; id[x]=idcnt++; top[x]=tv;
    if(son[x]==-1)return;
    dfs2(son[x],tv);
    for(auto p:e[x]){
        if(p==fa[x] || p==son[x])continue;
        dfs2(p,p);
    }
}
int lab[N]; // 初始点权
seg tr[N*2],*pl; // if(l==r){a=lab[arcid[l]];return;}
void init(int s){
    idcnt=0; fa[s]=s;
    dfs1(s); dfs2(s,s);
    seginit(0,idcnt-1); // 线段树的初始化
}
void upchain(int x,int y,int d){
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]])swap(x,y);
        tr->update(id[top[x]],id[x],d);
        x=fa[top[x]];
    }
    if(dep[x]>dep[y])swap(x,y);
    tr->update(id[x],id[y],d);
}
ll qchain(int x,int y){
    ll ans=0;
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]])swap(x,y);
        ans+=tr->query(id[top[x]],id[x]);
        x=fa[top[x]];
    }
    if(dep[x]>dep[y])swap(x,y);
    ans+=tr->query(id[x],id[y]);
    return ans;
}
void uptree(int x,int d){
    tr->update(id[x],id[x]+sz[x]-1,d);
}
ll qtree(int x){
    return tr->query(id[x],id[x]+sz[x]-1);
}
```

### 树分治

#### 点分治

- 每次找树的重心（最大子树最小的点），去掉它后对所有子树进行相同操作
- 一般 $O(n\log n)$
- 例：luogu P3806，带边权的树，询问长度为 $q_i$ 的路径是否存在

```cpp
const int debug = 1;
vector<pii> a[N];
bool vis[N];
vector<pii> q;  // q[i].fi: query; q[i].se: answer
namespace center {
vector<int> rec;
int sz[N], maxx[N];
void dfs(int x, int fa = -1) {
    rec << x;
    sz[x] = 1;
    maxx[x] = 0;
    for (auto i : a[x]) {
        int p = i.fi;
        if (p != fa && !vis[p]) {
            dfs(p, x);
            sz[x] += sz[p];
            maxx[x] = max(maxx[x], sz[p]);
        }
    }
}
int get(int x) {  // get center
    rec.clear();
    dfs(x);
    int n = sz[x], ans = x;
    for (auto x : rec) {
        maxx[x] = max(maxx[x], n - sz[x]);
        if (maxx[x] < maxx[ans]) ans = x;
    }
    return ans;
}
}  // namespace center
vector<int> rec;
void getdist(int x, int dis, int fa = -1) {
    if (dis < 10000010) rec << dis;
    for (auto i : a[x]) {
        int p = i.fi;
        if (p != fa && !vis[p]) {
            getdist(p, dis + i.se, x);
        }
    }
}
unordered_set<int> bkt;
void dfs(int x) {
    x = center::get(x);
    bkt.clear();
    bkt.insert(0);
    vis[x] = 1;
    for (auto i : a[x]) {  // 这部分统计各个子树的信息并更新答案
        int p = i.fi;
        if (!vis[p]) {
            rec.clear();
            getdist(p, i.se);
            for (auto i : rec) {
                for (auto &j : q)
                    if (bkt.count(j.fi - i)) j.se = 1;
            }
            for (auto i : rec) bkt.insert(i);
        }
    }
    for (auto i : a[x]) {  // 这部分进一步分治
        int p = i.fi;
        if (!vis[p]) {
            dfs(p);
        }
    }
}
```

### 虚树

- $O(k\log n)$ 处理出指定 k 个点及其两两 lca 构成的树，原理是单调栈
- `pos[x]` 表示 DFS 序中 x 的位置，`lab[x]` 表示 x 是否为指定点
- `tr` 表示虚树，`v` 是指定点序列(input)
- 基于 lca，预处理为 `lca::init()` 以及 `pos[]`

```cpp
vector<int> e[N],v; // v: input
int pos[N];
vector<int> stk,rec;
vector<pii> tr[N];
bool lab[N];
#define r stk.rbegin()
void add(){
    tr[r[1]].push_back({r[0],dep[r[0]]-dep[r[1]]}); // tr[x][i].second is the length of the edge
    rec.push_back(r[0]);
    stk.pop_back();
}
void lastdfs(int x,int fa){
    ;
}
void vtree(){
    sort(v.begin(),v.end(),[](int x,int y){
        return pos[x]<pos[y];
    });
    stk.assign(1,1); rec.assign(1,1);
    for(auto i:v)lab[i]=1;
    for(auto i:v)if(i!=1){
        int l=lca(i,r[0]);
        while(pos[l]<pos[r[0]]){
            if(pos[l]>pos[r[1]])
                stk.insert(stk.end()-1,l);
            add();
        }
        stk.push_back(i);
    }
    while(stk.size()>1)add();
    // flag=true;
    lastdfs(1,-1);
    // if(flag)printf("%d\n",cost[1]); else puts("-1");
    for(auto i:rec){ // clear
        tr[i].clear();
        lab[i]=0; // cost[i]=0; up[i]=0;
    }
}
```

### 树上启发式合并

- 暴力方式处理子树问题
- 编号无限制，$O(n\log n)$

```cpp
vector<int> e[N]; int n;
int sz[N],son[N],dep[N]; bool vis[N];
ll ans[N],sum[N]; int num[N],top,c[N]; // not fixed
void initdfs(int x,int fa){
    dep[x]=dep[fa]+1; sz[x]=1;
    for(auto p:e[x])if(p!=fa){
        initdfs(p,x); sz[x]+=sz[p];
        if(sz[p]>sz[son[x]])son[x]=p;
    }
}
void update(int x,int fa,int op){
    sum[num[c[x]]]-=c[x]; num[c[x]]+=op; sum[num[c[x]]]+=c[x];
    if(sum[top+1])top++; if(!sum[top])top--;
    for(auto p:e[x])if(p!=fa && !vis[p])
        update(p,x,op);
}
void dfs(int x,int fa,int hs){
    for(auto p:e[x])if(p!=fa && p!=son[x])
        dfs(p,x,0);
    if(son[x])dfs(son[x],x,1),vis[son[x]]=1;
    update(x,fa,1);
    vis[son[x]]=0; ans[x]=sum[top];
    if(!hs)update(x,fa,-1);
}
// initdfs(s,-1); dfs(s,-1,1);
```

## 联通性相关

### 强联通分量 SCC + 缩点

Tarjan

- co: 染色结果（点 i 缩点后为 `co[i]`）。
- w: 点权（缩点后原地更新）。
- sz: 第 i 个颜色的点数。
- 编号从 0 开始，$O(V+E)$。

```cpp
vector<int> a[N];
stack<int> stk;
bool vis[N], instk[N];
int dfn[N], low[N], co[N], w[N];
vector<int> sz;
int n, dcnt;
void dfs(int x) { // Tarjan
    vis[x] = instk[x] = 1; stk.push(x);
    dfn[x] = low[x] = ++dcnt;
    for(auto p : a[x]) {
        if (!vis[p]) dfs(p);
        if (instk[p]) low[x] = min(low[x], low[p]);
    }
    if (low[x] == dfn[x]) {
        int t; sz.push_back(0);
        do {
            t = stk.top();
            stk.pop();
            instk[t] = 0;
            sz.back() += w[t];
            co[t] = sz.size() - 1;
        } while (t != x);
    }
}
void getscc() {
    fill(vis, vis + n, 0);
    sz.clear();
    repeat (i, 0, n) if (!vis[i]) dfs(i);
}
void shrink() { // result inplace
    static vector<pii> eset;
    eset.clear();
    getscc();
    repeat (i, 0, n)
    for (auto p : a[i])
    if (co[i] != co[p])
        eset.push_back({co[i], co[p]});
    n = sz.size();
    repeat (i, 0, n){
        a[i].clear();
        w[i] = sz[i];
    }
    for(auto i : eset){
        a[i.fi].push_back(i.se);
        // a[i.se].push_back(i.fi);
    }
}
```

- 例题：给一个有向图，连最少的边使其变为scc。解：scc缩点后输出 $\max(\sum\limits_i[indeg[i]=0],\sum\limits_i[outdeg[i]=0])$，特判只有一个scc的情况。

Kosaraju（缩点同上）

- 编号从 1 开始，$O(V+E)$。

```cpp
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

### 边双连通分量 using Tarjan

- 编号从0开始，$O(V+E)$

```cpp
void dfs(int x,int fa){ // Tarjan求边双联通分量
    vis[x]=instk[x]=1; stk.push(x);
    dfn[x]=low[x]=++dcnt;
    for(auto p:a[x])
    if(p!=fa){
        if(!vis[p])dfs(p,x);
        if(instk[p])low[x]=min(low[x],low[p]);
    }
    else fa=-1; // 处理重边
    if(low[x]==dfn[x]){
        int t; sz.push_back(0); // 记录
        do{
            t=stk.top();
            stk.pop();
            instk[t]=0;
            sz.back()+=w[t]; // 记录
            co[t]=sz.size()-1; // 染色
        }while(t!=x);
    }
}
void getscc(){
    fill(vis,vis+n,0);
    sz.clear();
    repeat(i,0,n)if(!vis[i])dfs(i,-1);
}
// 全局变量，shrink()同scc
```

### 割点 / 割顶

- Tarjan

```cpp
bool vis[N],cut[N]; // cut即结果，cut[i]表示i是否为割点
int dfn[N],low[N];
int dcnt; // 时间戳
void dfs(int x,bool isroot=1){
    if(vis[x])return; vis[x]=1;
    dfn[x]=low[x]=++dcnt;
    int ch=0; cut[x]=0;
    for(auto p:a[x]){
        if(!vis[p]){
            dfs(p,0);
            low[x]=min(low[x],low[p]);
            if(!isroot && low[p]>=dfn[x])
                cut[x]=1;
            ch++;
        }
        low[x]=min(low[x],dfn[p]);
    }
    if(isroot && ch>=2) // 根结点判断方法
        cut[x]=1;
}
```

## 2-Sat 问题

- 有 $2n$ 个顶点，其中顶点 $2i$ 和顶点 $2i+1$ 中能且仅能选一个，边 (u, v) 表示选了 u 就必须选 v，求一个可行解
- 暴力版，可以跑出字典序最小的解，编号从 0 开始，$O(VE)$，（~~但是难以跑到上界~~）

```cpp
struct twosat{ // 暴力版
    int n;
    vector<int> g[N*2];
    bool mark[N*2]; // mark即结果，表示是否选择了这个点
    int s[N],c;
    bool dfs(int x){ // private
        if(mark[x^1])return 0;
        if(mark[x])return 1;
        mark[s[c++]=x]=1;
        for(auto p:g[x])
        if(!dfs(p))
            return 0;
        return 1;
    }
    void init(int _n){
        n=_n;
        for(int i=0;i<n*2;i++){
            g[i].clear();
            mark[i]=0;
        }
    }
    void add(int x,int y){ // 这个函数随题意变化
        g[x*2].push_back(y*2+1); // 选了x*2就必须选y*2+1
        g[y*2].push_back(x*2+1); // 选了y*2就必须选x*2+1
    }
    bool solve(){ // 返回是否存在解
        for(int i=0;i<n*2;i+=2)
        if(!mark[i] && !mark[i^1]){
            c=0;
            if(!dfs(i)){
                while(c>0)mark[s[--c]]=0;
                if(!dfs(i^1))return 0;
            }
        }
        return 1;
    }
}ts;
```

- SCC 操作，编号从 1 开始，$O(V+E)$，注意 N 需要手动两倍

```cpp
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

## 支配树 using Lengauer-Tarjan 算法

- 有向图给定源点，若删掉 r，源点不可达 u，则称 r 是 u 的支配点
- 支配树即所有非源点的点与最近支配点(idom)连边形成的树（源点为根）
- input: a 邻接表，b 反图邻接表。
- 大约 $O(V+E)$。

```cpp
vector<int> a[N],b[N],tr[N]; // tr: result
int fa[N],dfn[N],dcnt,arcdfn[N];
int c[N],best[N],sm[N],im[N]; // im: result
void init(int n){
    dcnt=0;
    iota(c,c+n+1,0);
    repeat(i,1,n+1){
        tr[i].clear();
        a[i].clear();
        b[i].clear();
    }
    repeat(i,1,n+1)sm[i]=best[i]=i;
    fill(dfn,dfn+n+1,0);
}
void dfs(int u){
    dfn[u]=++dcnt; arcdfn[dcnt]=u;
    for(auto v:a[u])if(!dfn[v]){fa[v]=u; dfs(v);}
}
int find(int x){
    if(c[x]==x)return x;
    int &f=c[x],rt=find(f);
    if(dfn[sm[best[x]]]>dfn[sm[best[f]]])
        best[x]=best[f];
    return f=rt;
}
void solve(int s){
    dfs(s);
    repeat_back(i,2,dcnt+1){
        int x=arcdfn[i],mn=dcnt+1;
        for(auto u:b[x]){
            if(!dfn[u])continue;
            find(u); mn=min(mn,dfn[sm[best[u]]]);
        }
        c[x]=fa[x];
        tr[sm[x]=arcdfn[mn]].push_back(x);
        x=arcdfn[i-1];
        for(auto u:tr[x]){
            find(u);
            if(sm[best[u]]!=x)im[u]=best[u];
            else im[u]=x;
        }
        tr[x].clear();
    }
    repeat(i,2,dcnt+1){
        int u=arcdfn[i];
        if(im[u]!=sm[u])im[u]=im[im[u]];
        tr[im[u]].push_back(u);
    }
}
```

## 图上的NP问题

### 最大团 and 极大团计数

- 求最大团顶点数（和最大团），`g[][]` 编号从 0 开始，$O(\exp)$

```cpp
int g[N][N],f[N][N],v[N],Max[N],n,ans; // g[][]是邻接矩阵，n是顶点数
// vector<int> rec,maxrec; // maxrec是最大团
bool dfs(int x,int cur){
    if(cur==0)
        return x>ans;
    repeat(i,0,cur){
        int u=f[x][i],k=0;
        if(Max[u]+x<=ans)return 0;
        repeat(j,i+1,cur)
        if(g[u][f[x][j]])
            f[x+1][k++]=f[x][j];
        // rec.push_back(u);
        if(dfs(x+1,k))return 1;
        // rec.pop_back();
    }
    return 0;
}
void solve(){
    ans=0; // maxrec.clear();
    repeat_back(i,0,n){
        int k=0;
        repeat(j,i+1,n)
        if(g[i][j])
            f[1][k++]=j;
        // rec.clear(); rec.push_back(i);
        if(dfs(1,k)){
            ans++;
            // maxrec=rec;
        }
        Max[i]=ans;
    }
}
```

- 求极大团个数（和所有极大团），`g[][]` 的编号从 1 开始！$O(\exp)$

```cpp
int g[N][N],n;
// vector<int> rec; // 存当前极大团
int ans,some[N][N],none[N][N]; // some是未搜索的点，none是废除的点
void dfs(int d,int sn,int nn){
    if(sn==0 && nn==0)
        ans++; // 此时rec是其中一个极大图
    // if(ans>1000)return; // 题目要求_(:зゝ∠)_
    int u=some[d][0];
    for(int i=0;i<sn;++i){
        int v=some[d][i];
        if(g[u][v])continue;
        int tsn=0,tnn=0;
        for(int j=0;j<sn;++j)
        if(g[v][some[d][j]])
            some[d+1][tsn++]=some[d][j];
        for(int j=0;j<nn;++j)
        if(g[v][none[d][j]])
            none[d+1][tnn++]=none[d][j];
        // rec.push_back(v);
        dfs(d+1,tsn,tnn);
        // rec.pop_back();
        some[d][i]=0;
        none[d][nn++]=v;
    }
}
void solve(){ // 运行后ans即极大团数
    ans=0;
    for(int i=0;i<n;++i)
        some[0][i]=i+1;
    dfs(0,n,0);
}
```

### 最小染色数

- $O(\exp)$，`n=17` 可用

```cpp
int n,m;
int g[N]; // 二进制邻接矩阵
bool ind[1<<N]; // 是否为(极大)独立集
int dis[1<<N];
vector<int> a; // 存独立集
#define np (1<<n)
int bfs(){ // 重复覆盖简略版
    fill(dis,dis+np,inf); dis[0]=0;
    auto q=queue<int>(); q.push(0);
    while(!q.empty()){
        int x=q.front(); q.pop();
        for(auto i:a){
            int p=x|i;
            if(p==np-1)return dis[x]+1;
            if(dis[p]>dis[x]+1){
                dis[p]=dis[x]+1;
                q.push(p);
            }
        }
    }
    return 0;
}
int solve(){ // 返回最小染色数
    mst(g,0);
    for(auto i:eset){
        int x=i.fi,y=i.se;
        g[x]|=1<<y;
        g[y]|=1<<x;
    }
    // 求所有独立集
    ind[0]=1;
    repeat(i,1,np){
        int w=__lg(i); // 最高位
        if((g[w]&i)==0 && ind[i^(1<<w)])
            ind[i]=1;
    }
    // 删除所有不是极大独立集的独立集
    repeat(i,1,np)
    if(ind[i]){
        for(int j=1;j<np;j<<=1)
        if((i&j)==0 && ind[i|j]){
            ind[i]=0;
            break;
        }
        if(ind[i])
            a.push_back(i); // 记录极大独立集
    }
    return bfs();
}
```

## 仙人掌 using 圆方树

- 仙人掌：每条边至多属于一个简单环的无向联通图。
- 圆方树：原来的点称为圆点，每个环新建一个方点，环上的圆点都与方点连接。
- 子仙人掌：以 r 为根，点 p 的子仙人掌是删掉 p 到 r 的所有简单路径后 p 所在的联通块。这个子仙人掌就是圆方树中以 r 为根时，p 子树中的所有圆点。
- 仙人掌判定：DFS 树上差分。
- 仙人掌最短路径，基于树上倍增 LCA，编号从哪开始都可以，$O(n+m)$。

```cpp
vector<edge> a[N], b[N]; // a: input, b: the block forest
int n, bn; // n: input, bn = |B|
namespace cactus {
    bool vis[N];
    ll fa[N], lab[N], dep[N], dis[N];
    bool iscactus;
    ll cyclen[N];
    void dfs(int x) { // private
        vis[x] = 1;
        for (auto i : a[x]) {
            int p = i.to();
            if (p == fa[x]) continue;
            if (!vis[p]) {
                fa[p] = x; dep[p] = dep[x] + 1;
                dis[p] = dis[x] + i.dis();
                dfs(p);
                lab[x] += lab[p];
                if (lab[p] == 0) b[x].push_back({p, i.dis()});
            }
            else if (dep[p] < dep[x]) { // find a cycle
                lab[x]++; lab[p]--;
                // cycle size: dep[x] - dep[p] + 1
                // cycle length: dis[x] - dis[p] + i.second
                int u = bn++;
                cyclen[u] = dis[x] - dis[p] + i.dis();
                for (int k = x; k != p; k = fa[k]) {
                    ll d = dis[k] - dis[p];
                    b[u].push_back({k, min(d, cyclen[u] - d)});
                }
                b[p].push_back({u, 0});
            }
        }
        if (lab[x] >= 2) iscactus = 0;
    }
    void init(int s, int n) {
        bn = n;
        repeat (i, 0, n) vis[i] = false, lab[i] = dep[i] = dis[i] = 0;
        iscactus = 1; fa[s] = -1;
        dfs(s);
        lca::init(s);
    }
    ll length(int x, int y) {
        int l = lca::lca(x, y);
        ll ans = 0;
        if (l >= n) {
            ans += lca::qsumla(x, lca::dep[x] - lca::dep[l] - 1);
            x = lca::la(x, lca::dep[x] - lca::dep[l] - 1);
            ans += lca::qsumla(y, lca::dep[y] - lca::dep[l] - 1);
            y = lca::la(y, lca::dep[y] - lca::dep[l] - 1);
            ll t = abs(dis[x] - dis[y]);
            ans += min(t, cyclen[l] - t);
        } else {
            ans = lca::qsum(x, y);
        }
        return ans;
    }
}
```

## 二分图

### 二分图匹配 / 最大匹配

- 匈牙利 / hungarian，左右顶点编号从 0 开始，$O(VE)$

```cpp
vector<int> a[N]; // a: input, the left vertex x is connected to the right vertex a[x][i]
int dcnt,mch[N],dfn[N]; // mch: output, the right vertex p is connected to the left vertex mch[p]
bool dfs(int x){
    for(auto p:a[x]){
        if(dfn[p]!=dcnt){
            dfn[p]=dcnt;
            if(mch[p]==-1 || dfs(mch[p])){
                mch[p]=x;
                return 1;
            }
        }
    }
    return 0;
}
int hun(int n,int m){ // n,m: the number of the left/right vertexes. return max matching
    int ans=0;
    repeat(i,0,m)mch[i]=-1;
    repeat(i,0,n){
        dcnt++;
        if(dfs(i))ans++;
    }
    return ans;
}
```

- HK 算法 / Hopcroft-karp，左顶点编号从 0 开始，右顶点编号从 n 开始，$O(E\sqrt V)$

```cpp
vector<int> a[N]; // a: input, the left vertex x is connected to the right vertex a[x][i]
int mch[N*2],dep[N*2]; // mch: output, the vertex p is connected to the vertex mch[p] (p could be either left or right vertex)
bool bfs(int n,int m){
    static queue<int> q;
    fill(dep,dep+n+m,0);
    bool flag=0;
    repeat(i,0,n)if(mch[i]==-1)q.push(i);
    while(!q.empty()){
        int x=q.front(); q.pop();
        for(auto p:a[x]){
            if(!dep[p]){
                dep[p]=dep[x]+1;
                if(mch[p]==-1)flag=1;
                else dep[mch[p]]=dep[p]+1,q.push(mch[p]);
            }
        }
    }
    return flag;
}
bool dfs(int x){
    for(auto p:a[x]){
        if(dep[p]!=dep[x]+1) continue;
        dep[p]=0;
        if(mch[p]==-1 || dfs(mch[p])) {
            mch[x]=p; mch[p]=x;
            return 1;
        }
    }
    return 0;
}
int solve(int n,int m){ // n,m: the number of the left/right vertexes. return max matching
    int ans=0;
    fill(mch,mch+n+m,-1);
    while(bfs(n,m)){
        repeat(i,0,n)
        if(mch[i]==-1 && dfs(i))
            ans++;
    }
    return ans;
}
```

### 二分图最大权匹配 using KM

- 求满二分图的最大权匹配，如果没有边就建零边。
- 输入 n, g（$n\times n$ 邻接矩阵）。
- 编号从 0 开始，$O(n^3)$。

```cpp
int n;
ll g[N][N];
namespace km {
int mx[N], my[N], pre[N];
bool visx[N], visy[N];
ll lx[N], ly[N], slack[N];
queue<int> q;
bool check(int v) {
    visy[v] = true;
    if (my[v] != -1) {
        q.push(my[v]);
        visx[my[v]] = true;
        return false;
    }
    while (v != -1) {
        my[v] = pre[v];
        swap(v, mx[pre[v]]);
    }
    return true;
}
void bfs(int i) {
    while (!q.empty()) q.pop();
    q.push(i);
    visx[i] = true;
    while (1) {
        while (!q.empty()) {
            int x = q.front(); q.pop();
            repeat (y, 0, n) if (!visy[y]) {
                ll delta = lx[x] + ly[y] - g[x][y];
                if (slack[y] >= delta) {
                    pre[y] = x;
                    if (delta) {
                        slack[y] = delta;
                    } else if (check(y)) {
                        return;
                    }
                }
            }
        }
        ll a = INF;
        repeat (j, 0, n) if (!visy[j])
            a = min(a, slack[j]);
        repeat (j, 0, n) {
            if (visx[j]) lx[j] -= a;
            if (visy[j]) {
                ly[j] += a;
            } else {
                slack[j] -= a;
            }
        }
        repeat (j, 0, n)
        if (!visy[j] && slack[j] == 0 && check(j))
            return;
    }
}
ll solve() {
    ll res = 0;
    repeat (i, 0, n) {
        mx[i] = my[i] = -1;
        pre[i] = 0;
        lx[i] = -INF; ly[i] = 0;
        repeat (k, 0, n)
            lx[i] = max(lx[i], g[i][k]);
    }
    for (int i = 0; i < n; i++) {
        repeat (k, 0, n) {
            slack[k] = INF;
            visx[k] = visy[k] = 0;
        }
        bfs(i);
    }
    for (int i = 0; i < n; i++) {
        res += g[i][mx[i]];
    }
    return res;
}
};
```

### 稳定婚姻 using 延迟认可

- 稳定意味着不存在一对不是情侣的男女，都认为当前伴侣不如对方
- 编号从 0 开始，$O(n^2)$

```cpp
struct node{
    int s[N]; // s的值给定
        // 对男生来说是女生编号排序
        // 对女生来说是男生的分数
    int now; // 选择的伴侣编号
}a[N],b[N]; // 男生，女生
int tr[N]; // 男生尝试表白了几次
queue<int> q; // 单身狗（男）排队
bool match(int x,int y){ // 配对，返回是否成功
    int x0=b[y].now;
    if(x0!=-1){
        if(b[y].s[x]<b[y].s[x0])
            return 0; // 分数不够，竞争失败
        q.push(x0);
    }
    a[x].now=y;
    b[y].now=x;
    return 1;
}
void stable_marriage(){ // 运行后a[].now,b[].now即结果
    q=queue<int>();
    repeat(i,0,n){
        b[i].now=-1;
        q.push(i);
        tr[i]=0;
    }
    while(!q.empty()){
        int x=q.front(); q.pop();
        int y=a[x].s[tr[x]++]; // 下一个最中意女生
        if(!match(x,y))
            q.push(x); // 下次努力
    }
}
```

### 一般图最大匹配 using 带花树

- 对于一个无向图，找最多的边使得这些边两两无公共端点
- 编号从 1 开始，$O(n^3)$

```cpp
int n; DSU d;
deque<int> q; vector<int> e[N];
int mch[N],vis[N],dfn[N],fa[N],dcnt=0;
int lca(int x,int y){
    dcnt++;
    while(1){
        if(x==0)swap(x,y); x=d[x];
        if(dfn[x]==dcnt)return x;
        else dfn[x]=dcnt,x=fa[mch[x]];
    }
}
void shrink(int x,int y,int p){
    while(d[x]!=p){
        fa[x]=y; y=mch[x];
        if(vis[y]==2)vis[y]=1,q.push_back(y);
        if(d[x]==x)d[x]=p;
        if(d[y]==y)d[y]=p;
        x=fa[y];
    }
}
bool match(int s){
    d.init(n); fill(fa,fa+n+1,0);
    fill(vis,vis+n+1,0); vis[s]=1;
    q.assign(1,s);
    while(!q.empty()){
        int x=q.front(); q.pop_front();
        for(auto p:e[x]){
            if(d[x]==d[p] || vis[p]==2)continue;
            if(!vis[p]){
                vis[p]=2; fa[p]=x;
                if(!mch[p]){
                    for(int now=p,last,tmp;now;now=last){
                        last=mch[tmp=fa[now]];
                        mch[now]=tmp,mch[tmp]=now;
                    }
                    return 1;
                }
                vis[mch[p]]=1; q.push_back(mch[p]);
            }
            else if(vis[p]==1){
                int l=lca(x,p);
                shrink(x,p,l);
                shrink(p,x,l);
            }
        }
    }
    return 0;
}
int solve(){ // 返回匹配数，mch[] 是匹配结果（即匹配 x 和 mch[x]），==0 表示不匹配
    int ans=0; fill(mch,mch+n+1,0);
    repeat(i,1,n+1)ans+=(!mch[i] && match(i));
    return ans;
}
```

### 一般图最大权匹配 using 带权带花树

- ~~它带什么花跟我有什么关系。~~
- 编号从 1 开始，$O(n^3)$。

```cpp
struct edge{int u,v,w;};
int n,n_x;
edge g[N*2][N*2];
int lab[N*2];
int match[N*2],slack[N*2],st[N*2],pa[N*2];
int flower_from[N*2][N],S[N*2],vis[N*2];
vector<int> flower[N*2];
queue<int> q;
int e_delta(const edge &e){
    return lab[e.u]+lab[e.v]-g[e.u][e.v].w*2;
}
void update_slack(int u,int x){
    if(!slack[x] || e_delta(g[u][x])<e_delta(g[slack[x]][x]))slack[x]=u;
}
void set_slack(int x){
    slack[x]=0;
    repeat(u,1,n+1)
        if(g[u][x].w>0 && st[u]!=x && S[st[u]]==0)update_slack(u,x);
}
void q_push(int x){
    if(x<=n)q.push(x);
    else repeat(i,0,flower[x].size())q_push(flower[x][i]);
}
void set_st(int x,int b){
    st[x]=b;
    if(x>n)repeat(i,0,flower[x].size())
            set_st(flower[x][i],b);
}
int get_pr(int b,int xr){
    int pr=find(flower[b].begin(),flower[b].end(),xr)-flower[b].begin();
    if(pr%2==1){
        reverse(flower[b].begin()+1,flower[b].end());
        return (int)flower[b].size()-pr;
    }else return pr;
}
void set_match(int u,int v){
    match[u]=g[u][v].v;
    if(u>n){
        edge e=g[u][v];
        int xr=flower_from[u][e.u],pr=get_pr(u,xr);
        for(int i=0;i<pr;++i)set_match(flower[u][i],flower[u][i^1]);
        set_match(xr,v);
        rotate(flower[u].begin(),flower[u].begin()+pr,flower[u].end());
    }
}
void augment(int u,int v){
    while(1){
        int xnv=st[match[u]];
        set_match(u,v);
        if(!xnv)return;
        set_match(xnv,st[pa[xnv]]);
        u=st[pa[xnv]],v=xnv;
    }
}
int get_lca(int u,int v){
    static int t=0;
    for(++t;u || v;swap(u,v)){
        if(u==0)continue;
        if(vis[u]==t)return u;
        vis[u]=t;
        u=st[match[u]];
        if(u)u=st[pa[u]];
    }
    return 0;
}
void add_blossom(int u,int lca,int v){
    int b=n+1;
    while(b<=n_x && st[b])++b;
    if(b>n_x)++n_x;
    lab[b]=0,S[b]=0;
    match[b]=match[lca];
    flower[b].clear();
    flower[b].push_back(lca);
    for(int x=u,y;x!=lca;x=st[pa[y]])
        flower[b].push_back(x),flower[b].push_back(y=st[match[x]]),q_push(y);
    reverse(flower[b].begin()+1,flower[b].end());
    for(int x=v,y;x!=lca;x=st[pa[y]])
        flower[b].push_back(x),flower[b].push_back(y=st[match[x]]),q_push(y);
    set_st(b,b);
    repeat(x,1,n_x+1)g[b][x].w=g[x][b].w=0;
    repeat(x,1,n+1)flower_from[b][x]=0;
    repeat(i,0,flower[b].size()){
        int xs=flower[b][i];
        for(int x=1;x<=n_x;++x)
            if(g[b][x].w==0 || e_delta(g[xs][x])<e_delta(g[b][x]))
                g[b][x]=g[xs][x],g[x][b]=g[x][xs];
        for(int x=1;x<=n;++x)
            if(flower_from[xs][x])flower_from[b][x]=xs;
    }
    set_slack(b);
}
void expand_blossom(int b){ // S[b] == 1
    for(int i=0;i<(int)flower[b].size();++i)
        set_st(flower[b][i],flower[b][i]);
    int xr=flower_from[b][g[b][pa[b]].u],pr=get_pr(b,xr);
    for(int i=0;i<pr;i+=2){
        int xs=flower[b][i],xns=flower[b][i+1];
        pa[xs]=g[xns][xs].u;
        S[xs]=1,S[xns]=0;
        slack[xs]=0,set_slack(xns);
        q_push(xns);
    }
    S[xr]=1,pa[xr]=pa[b];
    for(int i=pr+1;i<(int)flower[b].size();++i){
        int xs=flower[b][i];
        S[xs]=-1,set_slack(xs);
    }
    st[b]=0;
}
bool on_found_edge(const edge &e){
    int u=st[e.u],v=st[e.v];
    if(S[v]==-1){
        pa[v]=e.u,S[v]=1;
        int nu=st[match[v]];
        slack[v]=slack[nu]=0;
        S[nu]=0,q_push(nu);
    }else if(S[v]==0){
        int lca=get_lca(u,v);
        if(!lca)return augment(u,v),augment(v,u),1;
        else add_blossom(u,lca,v);
    }
    return 0;
}
bool matching(){
    memset(S+1,-1,sizeof(int)*n_x);
    memset(slack+1,0,sizeof(int)*n_x);
    q=queue<int>();
    for(int x=1;x<=n_x;++x)
        if(st[x]==x && !match[x])pa[x]=0,S[x]=0,q_push(x);
    if(q.empty())return false;
    while(1){
        while(q.size()){
            int u=q.front(); q.pop();
            if(S[st[u]]==1)continue;
            repeat(v,1,n+1)
                if(g[u][v].w>0 && st[u]!=st[v]){
                    if(e_delta(g[u][v])==0){
                        if(on_found_edge(g[u][v]))return true;
                    }else update_slack(u,st[v]);
                }
        }
        int d=inf;
        for(int b=n+1;b<=n_x;++b)
            if(st[b]==b && S[b]==1)d=min(d,lab[b]/2);
        repeat(x,1,n_x+1)
            if(st[x]==x && slack[x]){
                if(S[x]==-1)d=min(d,e_delta(g[slack[x]][x]));
                else if(S[x]==0)d=min(d,e_delta(g[slack[x]][x])/2);
            }
        repeat(u,1,n+1){
            if(S[st[u]]==0){
                if(lab[u]<=d)return 0;
                lab[u]-=d;
            }else if(S[st[u]]==1)lab[u]+=d;
        }
        for(int b=n+1;b<=n_x;++b)
            if(st[b]==b){
                if(S[st[b]]==0)lab[b]+=d*2;
                else if(S[st[b]]==1)lab[b]-=d*2;
            }
        q=queue<int>();
        repeat(x,1,n_x+1)
            if(st[x]==x && slack[x] && st[slack[x]]!=x && e_delta(g[slack[x]][x])==0)
                if(on_found_edge(g[slack[x]][x]))return true;
        for(int b=n+1;b<=n_x;++b)
            if(st[b]==b && S[b]==1 && lab[b]==0)expand_blossom(b);
    }
    return false;
}
pair<ll,int> weight_blossom(){
    memset(match+1,0,sizeof(int)*n);
    n_x=n;
    int n_matches=0;
    ll tot_weight=0;
    for(int u=0;u<=n;++u)st[u]=u,flower[u].clear();
    int w_max=0;
    repeat(u,1,n+1)
    repeat(v,1,n+1){
        flower_from[u][v]=(u==v?u:0);
        w_max=max(w_max,g[u][v].w);
    }
    repeat(u,1,n+1)lab[u]=w_max;
    while(matching())++n_matches;
    repeat(u,1,n+1)
        if(match[u] && match[u]<u)
            tot_weight+=g[u][match[u]].w;
    return make_pair(tot_weight,n_matches);
}
void init_weight_graph(){
    repeat(u,1,n+1)
        repeat(v,1,n+1)
            g[u][v]=edge{u,v,0};
}
void Solve(){
    int m;
    scanf("%d%d",&n,&m);
    init_weight_graph();
    repeat(i,0,m){
        int u,v,w;
        scanf("%d%d%d",&u,&v,&w);
        g[u][v].w=g[v][u].w=w;
    }
    printf("%lld\n",weight_blossom().first);
    repeat(u,1,n+1)printf("%d ",match[u]); puts("");
}
```

## 网络流

### 最大流 using Dinic

- 编号从 0 开始，$O(V^2E)$。

```cpp
struct Flow {
    const int inf = numeric_limits<int>().max() >> 1;
    struct edge { int to, w, nxt; };
    vector<edge> a; int head[N], cur[N];
    int n, s, t, dep[N];
    queue<int> q; bool inque[N];
    void ae(int x, int y, int w) {  // add edge, private
        a.push_back({y, w, head[x]});
        head[x] = a.size() - 1;
    }
    bool bfs() {  // private
        fill(dep, dep + n, inf); dep[s] = 0;
        copy(head, head + n, cur);
        q = queue<int>(); q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop(); inque[x] = 0;
            for (int i = head[x]; i != -1; i = a[i].nxt) {
                int p = a[i].to;
                if (dep[p] > dep[x] + 1 && a[i].w) {
                    dep[p] = dep[x] + 1;
                    if (inque[p] == 0) {
                        inque[p] = 1;
                        q.push(p);
                    }
                }
            }
        }
        return dep[t] != inf;
    }
    int dfs(int x, int flow) {  // private
        int now, ans = 0;
        if (x == t) return flow;
        for (int &i = cur[x]; i != -1; i = a[i].nxt) {
            int p = a[i].to;
            if (a[i].w && dep[p] == dep[x] + 1)
                if ((now = dfs(p, min(flow, a[i].w)))) {
                    a[i].w -= now;
                    a[i ^ 1].w += now;
                    ans += now, flow -= now;
                    if (flow == 0) break;
                }
        }
        return ans;
    }
    void init(int _n) { // init with n nodes
        n = _n + 1; a.clear();
        fill(head, head + n, -1);
        fill(inque, inque + n, 0);
    }
    void add_dir(int x, int y, int w) { ae(x, y, w), ae(y, x, 0); }  // add directed edge
    void add_undir(int x, int y, int w) { ae(x, y, w), ae(y, x, w); } // add undirected edge
    int solve(int _s, int _t) {  // get max flow from s to t
        s = _s, t = _t;
        int ans = 0;
        while (bfs()) ans += dfs(s, inf);
        return ans;
    }
    void print() {
        repeat (x, 0, n) {
            cout << x << ":";
            for (int i = head[x]; i != -1; i = a[i].nxt) if (i % 2 == 0)
                cout << " (" << a[i].to << ", " << a[i ^ 1].w << " / " << a[i].w + a[i ^ 1].w << ")";
            cout << endl;
        }
    }
} flow;
// 先 init(n)，再 add_dir / add_undir 添边，最后 solve(s, t)
```

最小割的边集

```cpp
struct MinCut : Flow {
    bool vis[N];
    void dfs(int x) {
        vis[x] = 1;
        for (int i = head[x]; i != -1; i = a[i].nxt)
        if (a[i].w && a[i ^ 1].w && !vis[a[i].to])
            dfs(a[i].to);
    }
    vector<pii> get_cut() { // after solve(s, t)
        fill(vis, vis + n, 0);
        dfs(s);
        // vis[i] = true 表示 S 集合，否则 T 集合
        // 下面的代码不太清楚对不对
        vector<pii> ans;
        repeat (x, 0, n) if (vis[x]) {
            for (int i = head[x]; i != -1; i = a[i].nxt)
            if (i % 2 == 0 && a[i].w == 0 && !vis[a[i].to])
                ans.push_back({x, a[i].to});
        }
        return ans;
    }
} flow;
```

### 最小费用最大流 using MCMF

- 费用流一般指最小费用最大流（最大费用最大流把费用取反即可）
- 编号从 0 开始，$O(VE^2)$

```cpp
struct Flow {
    const int inf = INT_MAX >> 1;
    struct edge { int to, w, cost, nxt; };
    vector<edge> a; int head[N];
    int n, s, t, totcost;
    deque<int> q;
    bool inque[N];
    int dis[N];
    struct { int to, e; } pre[N];
    void ae(int x, int y, int w, int cost) {
        a.push_back({y, w, cost, head[x]});
        head[x] = a.size() - 1;
    }
    bool spfa() {
        fill(dis, dis + n, inf); dis[s] = 0;
        q.assign(1, s);
        while (!q.empty()) {
            int x = q.front(); q.pop_front();
            inque[x] = 0;
            for (int i = head[x]; i != -1; i = a[i].nxt) {
                int p = a[i].to;
                if (dis[p] > dis[x] + a[i].cost && a[i].w) {
                    dis[p] = dis[x] + a[i].cost;
                    pre[p] = {x, i};
                    if (inque[p] == 0) {
                        inque[p] = 1;
                        if (!q.empty()
                        && dis[q.front()] <= dis[p])
                            q.push_back(p);
                        else q.push_front(p);
                    }
                }
            }
        }
        return dis[t] != inf;
    }
    void init(int _n) {
        n = _n + 1; a.clear();
        fill(head, head + n, -1);
        fill(inque, inque + n, 0);
    }
    int solve(int _s, int _t) { // 返回最大流，费用存 totcost 里
        s = _s, t = _t;
        int ans = 0;
        totcost = 0;
        while (spfa()) {
            int fl = inf;
            for (int i = t; i != s; i = pre[i].to)
                fl = min(fl, a[pre[i].e].w);
            for (int i = t; i != s; i = pre[i].to){
                a[pre[i].e].w -= fl;
                a[pre[i].e ^ 1].w += fl;
            }
            totcost += dis[t] * fl;
            ans += fl;
        }
        return ans;
    }
} flow;
void add(int x, int y, int w, int cost) {
    flow.ae(x, y, w, cost), flow.ae(y, x, 0, -cost);
}
// 先 flow.init(n)，再 add 添边，最后 flow.solve(s,t)
```

### 上下界网络流

上下界最大 / 小流：两遍普通最大流。

```cpp
struct BoundFlow: public Flow { // 从 Dinic 板子里继承
    int degree[N];
    void add(int x, int y, int l, int r) { // x 到 y 的有向边，下界 l，上界 r
        degree[y] += l; degree[x] -= l;
        add_dir(x, y, r - l);
    }
    void init(int _n) { // 初始化
        Flow::init(_n + 2); // 新增两点：n - 1 附加源点，n - 2 附加汇点
        fill(degree, degree + n, 0);
    }
    bool ok() { // 无源汇可行流
        int s = n - 2, t = n - 1, sum = 0;
        repeat (i, 0, s) {
            if (degree[i] > 0)
                add_dir(s, i, degree[i]), sum += degree[i];
            else if (degree[i] < 0)
                add_dir(i, t, -degree[i]);
        }
        return sum == solve(s, t);
    }
    bool ok(int s, int t) { // 有源汇可行流
        add(t, s, 0, inf);
        return ok();
    }
    int max_flow(int s, int t) { // 有源汇最大流
        if (ok(s, t) == false) return -1;
        for (int i = head[n - 2]; i != -1; i = a[i].nxt)
            a[i].w = a[i ^ 1].w = 0;
        for (int i = head[n - 1]; i != -1; i = a[i].nxt)
            a[i].w = a[i ^ 1].w = 0;
        return solve(s, t); // 最小流这行改为 inf - solve(t, s)
    }
} flow;
```

## 图论杂项

### Kruskal 重构树

- 基于 Kruskal 算法构建的有根树。
- 在原图中从某一点出发，只走边权不超过 w 的边，可达的点集在重构树中是一棵子树，对应 DFS 序的一个区间。
- 构建过程：边权从小到大访问边 (x, y)。如果 x, y 不连通，新建点 s 连接 x, y 所在树的根，且 s 为新树的根，s 的点权为边 (x, y) 的边权。
- 查找 x 能访问的点时，树上倍增找到最远的点权不大于 w 的祖先。
- 性质：二叉树，大根堆。
- luogu P4197，编号从 1 开始，$O(E\log E)$。

```cpp
DSU d;
vector<int> a[N];
int h[N],w[N];
vector<array<int,3>> eset;
vector<int> rec; int l[N],r[N],fa[N][logN];
void dfs(int x){
    l[x]=rec.size(); rec.push_back(x);
    for(auto p:a[x])fa[p][0]=x,dfs(p);
    r[x]=rec.size()-1;
}
divtree tr; // 其中一行改为 repeat(i,1,n+1)tr[0][i]=a[i]=h[rec[i]];
void kru(){
    sort(eset.begin(),eset.end(),
        [](array<int,3> &a,array<int,3> &b){
            return a[2]<b[2];
        }
    );
    int s=n;
    for(auto i:eset){
        int x=d[i[0]],y=d[i[1]]; if(x==y)continue;
        ++s; a[s].push_back(x); a[s].push_back(y);
        w[s]=i[2];
        d[x]=d[y]=s;
    }
    ++s; repeat(i,1,s+1-1)if(d[i]==i)a[s].push_back(i);
    w[s]=inf;
}
void Solve(){
    int n=read(),m=read(),q=read(); d.init(n*2); rec.assign(1,0);
    repeat(i,1,n+1)h[i]=read();
    while(m--){
        int x=read(),y=read(),w=read();
        eset.push_back({x,y,w});
    }
    kru(); // get kruskal tree
    fa[s][0]=s; dfs(s); // get DFS order & fa[i][0]
    repeat(i,1,logN)
    repeat(x,1,s+1)
        fa[x][i]=fa[fa[x][i-1]][i-1];
    tr.init(s);
    while(q--){
        int x=read(),ww=read(),k=read();
        repeat_back(i,0,logN)
            if(w[fa[x][i]]<=ww)x=fa[x][i];
        if((r[x]-l[x]+1)/2+1<k)puts("-1");
        else printf("%d\n",tr.maxk(l[x],r[x],k)); 
    }
}
```

### DSU 重构树

- 离线处理连边和连通块询问
- 原理：DSU 重构树的 DFS 序保证任意时刻的任意连通块是一个区间
- 接口：先读入 ops，然后 build(n)，然后依次 merge(x, y)（要保证顺序不变）。询问连通块区间 query(x, l, r)，询问结点 x 在 DFS 序中的位置是 `red.l[x]`
- 编号从 1 开始

```cpp
struct dsu_rebuilder{
    int cnt,l[N],r[N];
    DSU d;
    vector<int> a[N];
    vector<pii> ops; // input
    void dfs(int x){ // private
        l[x]=r[x]=cnt++;
        for(auto p:a[x])dfs(p);
    }
    void build(int n){
        cnt=0; d.init(n);
        repeat(i,0,n+1)a[i].clear();
        for(auto i:ops){
            int x=i.first,y=i.second;
            x=d[x]; y=d[y];
            if(x!=y){
                a[x].push_back(y);
                d[y]=d[x];
            }
        }
        repeat(i,1,n+1)if(d[i]==i)a[0].push_back(i);
        dfs(0); d.init(n); ops.clear();
    }
    void merge(int x,int y){
        x=d[x]; y=d[y];
        if(x!=y)d[y]=d[x],r[x]=r[y];
    }
    void query(int x,int &L,int &R){
        x=d[x]; L=l[x]; R=r[x];
    }
}red;
```
