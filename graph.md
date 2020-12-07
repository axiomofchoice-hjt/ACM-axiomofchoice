<!-- TOC -->

- [图论](#图论)
	- [图论的一些概念](#图论的一些概念)
	- [图论基础](#图论基础)
		- [前向星](#前向星)
		- [拓扑排序×Toposort](#拓扑排序toposort)
		- [欧拉路径 欧拉回路](#欧拉路径-欧拉回路)
		- [dfs树 bfs树](#dfs树-bfs树)
	- [线段树优化建图](#线段树优化建图)
	- [最短路径](#最短路径)
		- [Dijkstra](#dijkstra)
		- [Floyd](#floyd)
		- [SPFA](#spfa)
		- [Johnson](#johnson)
		- [最小环](#最小环)
		- [差分约束](#差分约束)
		- [同余最短路](#同余最短路)
	- [最小生成树×MST](#最小生成树mst)
		- [Kruskal](#kruskal)
		- [Boruvka](#boruvka)
		- [最小树形图 | 朱刘算法](#最小树形图--朱刘算法)
		- [绝对中心+最小直径生成树×MDST](#绝对中心最小直径生成树mdst)
	- [树论](#树论)
		- [树的直径](#树的直径)
		- [树的重心](#树的重心)
		- [最近公共祖先×LCA](#最近公共祖先lca)
			- [树上倍增解法](#树上倍增解法)
			- [欧拉序列+st表解法](#欧拉序列st表解法)
			- [树链剖分解法](#树链剖分解法)
			- [Tarjan解法](#tarjan解法)
			- [一些关于lca的问题](#一些关于lca的问题)
		- [树链剖分](#树链剖分)
		- [虚树](#虚树)
		- [树分治](#树分治)
			- [点分治](#点分治)
		- [树哈希](#树哈希)
	- [联通性相关](#联通性相关)
		- [强联通分量scc+缩点 | Tarjan](#强联通分量scc缩点--tarjan)
		- [边双连通分量 | Tarjan](#边双连通分量--tarjan)
		- [割点×割顶](#割点割顶)
	- [2-sat问题](#2-sat问题)
	- [支配树 | Lengauer−Tarjan算法](#支配树--lengauertarjan算法)
	- [图上的NP问题](#图上的np问题)
		- [最大团+极大团计数](#最大团极大团计数)
		- [最小染色数](#最小染色数)
	- [弦图+区间图](#弦图区间图)
	- [仙人掌 | 圆方树](#仙人掌--圆方树)
	- [二分图](#二分图)
		- [二分图的一些概念](#二分图的一些概念)
		- [二分图匹配×最大匹配](#二分图匹配最大匹配)
		- [最大权匹配 | KM](#最大权匹配--km)
		- [稳定婚姻 | 延迟认可](#稳定婚姻--延迟认可)
		- [一般图最大匹配 | 带花树](#一般图最大匹配--带花树)
	- [网络流](#网络流)
		- [网络流的一些概念](#网络流的一些概念)
		- [最大流](#最大流)
			- [Dinic](#dinic)
			- [ISAP](#isap)
		- [最小费用最大流 | MCMF](#最小费用最大流--mcmf)
	- [图论杂项](#图论杂项)
		- [矩阵树定理](#矩阵树定理)
		- [Prufer序列](#prufer序列)
		- [LGV引理](#lgv引理)
		- [others of 图论杂项](#others-of-图论杂项)

<!-- /TOC -->

# 图论

## 图论的一些概念

***

- 基环图：树加一条边
- 简单图：不含重边和自环（默认）
- 完全图：顶点两两相连的无向图
- 竞赛图：顶点两两相连的有向图
- 点u到v可达：有向图中，存在u到v的路径
- 点u和v联通：无向图中，存在u到v的路径
- 生成子图：点集和原图相同
- 导出子图/诱导子图：选取一个点集，尽可能多加边
- 正则图：所有点的度均相同的无向图

***

- 强正则图：$\forall (u,v)\in E,|\omega(u)\cap \omega(v)|=const$，且 $\forall (u,v)\not\in E,|\omega(u)\cap \omega(v)|=const$ 的正则图 $\omega(u)$ 为 $u$ 的领域
- 强正则图的点数 $v$，度 $k$，相邻的点的共度 $\lambda$，不相邻的点的共度 $\mu$ 有 $k(k-1-\lambda)=\mu(v-1-k)$
- 强正则图的例子：所有完全图、所有nk顶点满n分图

***

- 点割集：极小的，把图分成多个联通块的点集
- 割点：自身就是点割集的点
- 边割基：极小的，把图分成多个联通块的边集
- 桥：自身就是边割集的边
- 点联通度：最小点割集的大小
- 边联通度：最小边割集的大小
- Whitney定理：点联通度≤边联通度≤最小度

***

- 最大团：最大完全子图
- 最大独立集：最多的两两不连接的顶点
- 最小染色数：相邻的点不同色的最少色数
- 最小团覆盖数：覆盖整个图的最少团数
- 最大独立集即补图最大团
- 最小染色数等于补图最小团覆盖数

***

- 哈密顿通路：通过所有顶点有且仅有一次的路径，若存在则为半哈密顿图/哈密顿图
- 哈密顿回路：通过所有顶点有且仅有一次的回路，若存在则为哈密顿图
- 完全图 $K_{2k+1}$ 的边集可以划分为 $k$ 个哈密顿回路
- 完全图 $K_{2k}$ 的边集去掉 $k$ 条互不相邻的边后可以划分为 $k-1$ 个哈密顿回路

***

- 连通块数 = 点数 - 边数

***

## 图论基础

### 前向星

```c++
struct edge{int to,w,nxt;}; //指向，权值，下一条边
vector<edge> a;
int head[N];
void addedge(int x,int y,int w){
	a.push_back({y,w,head[x]});
	head[x]=a.size()-1;
}
void init(int n){
	a.clear();
	fill(head,head+n,-1);
}
//for(int i=head[x];i!=-1;i=a[i].nxt) //遍历x出发的边(x,a[i].to)
```

### 拓扑排序×Toposort

- $O(V+E)$

```c++
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

### 欧拉路径 欧拉回路

- 若存在则路径为 $dfs$ 退出序（最后的序列还要再反过来）（如果for从小到大，可以得到最小字典序）
- （不记录点的 $vis$，只记录边的 $vis$）

### dfs树 bfs树

- 无向图dfs树：树边、返祖边
- 有向图dfs树：树边、返祖边、横叉边、前向边
- 无向图bfs树：树边、返祖边、横叉边
- 空缺

## 线段树优化建图

- 建两棵线段树，第一棵每个结点连向其左右儿子，第二棵每个结点连向其父亲，两棵树所有叶子对应连无向边
- add(x1,y1,x2,y2,w) 表示 $[x_1,y_1]$ 每个结点向 $[x_2,y_2]$ 每个结点连 $w$ 边
- $a[i+tr.n]$ 表示结点 $i$
- 建议 $10$ 倍内存，编号从 $0$ 开始，$O(n\log n)$

```c++
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

### Dijkstra

- 仅限正权，$O(E\log E)$

```c++
struct node{
	int to; ll dis;
	bool operator<(const node &b)const{
		return dis>b.dis;
	}
};
int n;
bool vis[N];
vector<node> a[N];
void dij(int s,ll dis[]){ //s是起点，dis是结果
	fill(vis,vis+n+1,0);
	fill(dis,dis+n+1,inf); dis[s]=0; //last[s]=-1;
	static priority_queue<node> q; q.push({s,0});
	while(!q.empty()){
		int x=q.top().to; q.pop();
		if(vis[x])continue; vis[x]=1;
		for(auto i:a[x]){
			int p=i.to;
			if(dis[p]>dis[x]+i.dis){
				dis[p]=dis[x]+i.dis;
				q.push({p,dis[p]});
				//last[p]=x; //last可以记录最短路（倒着）
			}
		}
	}
}
```

### Floyd

- $O(V^3)$

```c++
repeat(k,0,n)
repeat(i,0,n)
repeat(j,0,n)
	f[i][j]=min(f[i][j],f[i][k]+f[k][j]);
```

- 补充：`bitset` 优化（只考虑是否可达），$O(V^3)$

```c++
//bitset<N> g<N>;
repeat(i,0,n)
repeat(j,0,n)
if(g[j][i])
	g[j]|=g[i];
```

### SPFA

- SPFA搜索中，有一个点入队 $n+1$ 次即存在负环
- 编号从 $0$ 开始，$O(VE)$

```c++
int cnt[N]; bool vis[N]; ll h[N]; //h意思和dis差不多，但是Johnson里需要区分
int n;
struct node{int to; ll dis;};
vector<node> a[N];
bool spfa(int s){ //返回是否有负环（s为起点）
	repeat(i,0,n+1)
		cnt[i]=vis[i]=0,h[i]=inf;
	h[s]=0; //last[s]=-1;
	static deque<int> q; q.assign(1,s);
	while(!q.empty()){
		int x=q.front(); q.pop_front();
		vis[x]=0;
		for(auto i:a[x]){
			int p=i.to;
			if(h[p]>h[x]+i.dis){
				h[p]=h[x]+i.dis;
				//last[p]=x; //last可以记录最短路（倒着）
				if(vis[p])continue;
				vis[p]=1;
				q.push_back(p); //可以SLF优化
				if(++cnt[p]>n)return 1;
			}
		}
	}
	return 0;
}
bool negcycle(){ //返回是否有负环
	a[n].clear();
	repeat(i,0,n)
		a[n].push_back({i,0}); //加超级源点
	return spfa(n);
}
```

### Johnson

- SPFA+Dijkstra实现全源最短路，编号从 $0$ 开始，$O(VE\log E)$

```c++
ll dis[N][N];
bool jn(){ //返回是否成功
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

### 最小环

- 有向图最小环Dijkstra，$O(VE\log E)$：对每个点 $v$ 进行Dijkstra，到达 $v$ 的边更新答案，适用稀图
- 有向图最小环Floyd，$O(V^3)$：Floyd完之后，任意两点计算 $dis_{u,v}+dis_{v,u}$，适用稠图
- 无边权无向图最小环：以每个顶点为根生成bfs树（不是dfs），横叉边更新答案，$O(VE)$
- 有边权无向图最小环：上面的bfs改成Dijkstra，$O(VE \log E)$

```c++
//无边权无向图最小环
int dis[N],fa[N],n,ans;
vector<int> a[N];
queue<int> q;
void bfs(int s){ //求经过s的最小环（不一定是简单环）
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
	repeat(i,0,n)bfs(i); //只要遍历最小环可能经过的点即可
	return ans;
}
```

### 差分约束

- $a_i-a_j\le c$，建边 $(j,i,c)$

### 同余最短路

- $k$ 种木棍，每种木棍个数不限，长度分别为 $l_i$，求这些木棍可以拼凑出多少小于等于 $h$ 的整数（包括 $0$）
- 以任一木棍 $n=l_0$ 为剩余系，连边 $(i,(i+l_j)\%n,l_j),j>1$，跑最短路后 $dis[i]$ 表示 $dis[i]+tn,t∈\N$ 都可以被拼凑出来
- 编号从 $0$ 开始，$O(nk\log(nk))$

```c++
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

## 最小生成树×MST

### Kruskal

- 对边长排序，然后添边，并查集判联通，$O(E\log E)$，排序是瓶颈

```c++
DSU d;
struct edge{int u,v,dis;}e[200010];
ll kru(){
	ll ans=0,cnt=0; d.init(n);
	sort(e,e+m);
	repeat(i,0,m){
		int x=d[e[i].u],y=d[e[i].v];
		if(x==y)continue;
		d.join(x,y);
		ans+=e[i].dis;
		cnt++;
		if(cnt==n-1)break;
	}
	if(cnt!=n-1)return -1;
	else return ans;
}
```

### Boruvka

- 类似Prim算法，但是可以多路增广（~~名词迷惑行为~~），$O(E\log V)$

```c++
DSU d;
struct edge{int u,v,dis;}e[200010];
ll bor(){
	ll ans=0;
	d.init(n);
	e[m].dis=inf;
	vector<int> b; //记录每个联通块的增广路（名词迷惑行为）
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

### 最小树形图 | 朱刘算法

- 其实有更高级的Tarjan算法 $O(E+V\log V)$，~~但是学不会~~
- 编号从1开始，求的是叶向树形图，$O(VE)$

```c++
int n;
struct edge{int x,y,w;};
vector<edge> eset; //会在solve中被修改
ll solve(int rt){ //返回最小的边权和，返回-1表示没有树形图
	static int fa[N],id[N],top[N],minw[N];
	ll ans=0;
	while(1){
		int cnt=0;
		repeat(i,1,n+1)
			id[i]=top[i]=0,minw[i]=inf;
		for(auto &i:eset) //记录权最小的父亲
		if(i.x!=i.y && i.w<minw[i.y]){
			fa[i.y]=i.x;
			minw[i.y]=i.w;
		}
		minw[rt]=0;
		repeat(i,1,n+1){ //标记所有环
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
		if(cnt==0)return ans; //无环退出
		repeat(i,1,n+1)
		if(!id[i])
			id[i]=++cnt;
		for(auto &i:eset){ //缩点
			i.w-=minw[i.y];
			i.x=id[i.x],i.y=id[i.y];
		}
		n=cnt;
		rt=id[rt];
	}
}
```

### 绝对中心+最小直径生成树×MDST

- 绝对中心：到所有点距离最大值最小的点，可以在边上
- 最小直径生成树：直径最小的生成树，可构造绝对中心为根的最短路径树
- 返回绝对中心所在边，生成树直径为 $d[x][rk[x][n-1]]+d[y][rk[y][n-1]]-d[x][y]$
- 编号从 $0$ 开始，$O(n^3)$，$n=1000$ 勉强能过

```c++
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
//init: repeat(i,0,n)repeat(j,0,n)g[i][j]=inf*(i!=j);
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

```c++
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

### 最近公共祖先×LCA

#### 树上倍增解法

- 编号从哪开始都可以，初始化 $O(n\log n)$，查询 $O(\log n)$

```c++
vector<int> e[N]; int dep[N],fa[N][22];
#define log(x) (31-__builtin_clz(x))
void dfs(int x){
	repeat(i,1,log(dep[x])+1){
		fa[x][i]=fa[fa[x][i-1]][i-1];
		//dis[x][i]=U(dis[x][i-1],dis[fa[x][i-1]][i-1]);
	}
	for(auto p:e[x])
	if(fa[x][0]!=p){
		fa[p][0]=x,dep[p]=dep[x]+1,dfs(p);
		//dis[p][0]=f(x,p);
	}
}
int lca(int x,int y){
	if(dep[x]<dep[y])swap(x,y);
	while(dep[x]>dep[y])
		x=fa[x][log(dep[x]-dep[y])];
	if(x==y)return x;
	repeat_back(i,0,log(dep[x])+1)
	if(fa[x][i]!=fa[y][i])
		x=fa[x][i],y=fa[y][i];
	return fa[x][0];
}
void init(int s){fa[s][0]=s; dep[s]=0; dfs(s);}
/*
lf len2(int x,int y){ //y是x的祖先
	lf ans=0;
	while(dep[x]>dep[y]){
		ans=U(ans,dis[x][log(dep[x]-dep[y])]);
		x=fa[x][log(dep[x]-dep[y])];
	}
	return ans;
}
lf length(int x,int y){int l=lca(x,y); return U(len2(x,l),len2(y,l));} //无修查询链上信息
*/
```

#### 欧拉序列+st表解法

- 编号从 $0$ 开始，初始化 $O(n\log n)$，查询 $O(1)$

```c++
int n,m;
vector<int> a;
vector<int> e[500010];
bool vis[500010];
int pos[500010],dep[500010];
#define mininarr(a,x,y) (a[x]<a[y]?x:y)
struct RMQ{
	#define logN 21
	int f[N*2][logN],log[N*2];
	RMQ(){
		log[1]=0;
		repeat(i,2,N*2)
			log[i]=log[i/2]+1;
	}
	void build(){
		int n=a.size();
		repeat(i,0,n)
			f[i][0]=a[i];
		repeat(k,1,logN)
		repeat(i,0,n-(1<<k)+1)
			f[i][k]=mininarr(dep,f[i][k-1],f[i+(1<<(k-1))][k-1]);
	}
	int query(int l,int r){
		if(l>r)swap(l,r);//!!
		int s=log[r-l+1];
		return mininarr(dep,f[l][s],f[r-(1<<s)+1][s]);
	}
}rmq;
void dfs(int x,int d){
	if(vis[x])return;
	vis[x]=1;
	dep[x]=d;
	a.push_back(x);
	pos[x]=a.size()-1;
	repeat(i,0,e[x].size()){
		int p=e[x][i];
		if(vis[p])continue;
		dfs(p,d+1);
		a.push_back(x);
	}
}
int lca(int x,int y){
	return rmq.query(pos[x],pos[y]);
}
//初始化：dfs(s,1); rmq.build();
```

#### 树链剖分解法

- 编号从哪开始都可以，初始化 $O(n)$，查询 $O(\log n)$

```c++
vector<int> e[N];
int dep[N],son[N],sz[N],top[N],fa[N]; //son重儿子，top链顶
void dfs1(int x){ //标注dep,sz,son,fa
	sz[x]=1;
	son[x]=-1;
	dep[x]=dep[fa[x]]+1;
	for(auto p:e[x]){
		if(p==fa[x])continue;
		fa[p]=x; dfs1(p);
		sz[x]+=sz[p];
		if(son[x]==-1 || sz[son[x]]<sz[p])
			son[x]=p;
	}
}
void dfs2(int x,int tv){ //标注top
	top[x]=tv;
	if(son[x]==-1)return;
	dfs2(son[x],tv);
	for(auto p:e[x]){
		if(p==fa[x] || p==son[x])continue;
		dfs2(p,p);
	}
}
void init(int s){ //s是根
	fa[s]=s;
	dfs1(s);
	dfs2(s,s);
}
int lca(int x,int y){
	while(top[x]!=top[y])
		if(dep[top[x]]>=dep[top[y]])x=fa[top[x]];
		else y=fa[top[y]];
	return dep[x]<dep[y]?x:y;
}
```

#### Tarjan解法

- 离线算法，基于并查集
- qry 和 ans 编号从 $0$ 开始，$O(n+m)$，大常数（不看好）

```c++
vector<int> e[N]; vector<pii> qry,q[N]; //qry输入
DSU d; bool vis[N]; int ans[N]; //ans输出
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

```c++
int length(int x,int y){ //路径长度
	return dep[x]+dep[y]-2*dep[lca(x,y)];
}
```

```c++
int intersection(int x,int y,int xx,int yy){ //树上两条路径公共点个数
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

- 编号从 $0$ 开始，处理链 $O(\log^2 n)$，处理子树 $O(\log n)$

```c++
vector<int> e[N];
int dep[N],son[N],sz[N],top[N],fa[N];
int id[N],arcid[N],idcnt; //id[x]:结点x在树剖序中的位置，arcid相反
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
int lab[N]; //初始点权
seg tr[N*2],*pl; //if(l==r){a=lab[arcid[l]];return;}
void init(int s){
	idcnt=0; fa[s]=s;
	dfs1(s); dfs2(s,s);
	seginit(0,idcnt-1); //线段树的初始化
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

### 虚树

```c++
```

### 树分治

#### 点分治

- 每次找树的重心（最大子树最小的点），去掉它后对所有子树进行相同操作
- 一般 $O(n\log n)$
- 例：luogu P3806，带边权的树，询问长度为 $q_i$ 的路径是否存在

```c++
vector<pii> a[N];
bool vis[N];
vector<pii> q; //q[i].fi: query; q[i].se: answer
namespace center{
vector<int> rec;
int sz[N],maxx[N];
void dfs(int x,int fa=-1){
	rec<<x;
	sz[x]=1; maxx[x]=0;
	for(auto i:a[x]){
		int p=i.fi;
		if(p!=fa && !vis[p]){
			dfs(p,x);
			sz[x]+=sz[p];
			maxx[x]=max(maxx[x],sz[p]);
		}
	}
}
int get(int x){ //get center
	rec.clear(); dfs(x); int n=sz[x],ans=x;
	for(auto x:rec){
		maxx[x]=max(maxx[x],n-sz[x]);
		if(maxx[x]<maxx[ans])ans=x;
	}
	return ans;
}
}
vector<int> rec;
void getdist(int x,int dis,int fa=-1){
	if(dis<10000010)rec<<dis;
	for(auto i:a[x]){
		int p=i.fi;
		if(p!=fa && !vis[p]){
			getdist(p,dis+i.se,x);
		}
	}
}
unordered_set<int> bkt;
void dfs(int x){
	x=center::get(x);
	bkt.clear(); bkt.insert(0);
	vis[x]=1;
	for(auto i:a[x]){ //这部分统计各个子树的信息并更新答案
		int p=i.fi;
		if(!vis[p]){
			rec.clear(); getdist(p,i.se);
			for(auto i:rec){
				for(auto &j:q)
				if(bkt.count(j.fi-i))
					j.se=1;
			}
			for(auto i:rec)bkt.insert(i);
		}
	}
	for(auto i:a[x]){ //这部分进一步分治
		int p=i.fi;
		if(!vis[p]){
			dfs(p);
		}
	}
}
```

### 树哈希

- $\displaystyle Hash[u]=sz[u]\sum_{v_i} Hash[v_i]B^{i-1}$（$v_i$ 根据哈希值排序）
- $\displaystyle Hash[u]=\oplus(C\cdot Hash[v_i]+sz[v_i])$
- $\displaystyle Hash[u]=1+\sum_{v_i}Hash[v_i]\cdot prime[sz[v_i]]$

## 联通性相关

### 强联通分量scc+缩点 | Tarjan

- 编号从0开始，$O(V+E)$

```c++
vector<int> a[N];
stack<int> stk;
bool vis[N],instk[N];
int dfn[N],low[N],co[N],w[N]; //co:染色结果，w:点权
vector<int> sz; //sz:第i个颜色的点数
int n,m,dcnt;
void dfs(int x){ //Tarjan求强联通分量
	vis[x]=instk[x]=1; stk.push(x);
	dfn[x]=low[x]=++dcnt;
	for(auto p:a[x]){
		if(!vis[p])dfs(p);
		if(instk[p])low[x]=min(low[x],low[p]);
	}
	if(low[x]==dfn[x]){
		int t; sz.push_back(0); //记录
		do{
			t=stk.top();
			stk.pop();
			instk[t]=0;
			sz.back()+=w[t]; //记录
			co[t]=sz.size()-1; //染色
		}while(t!=x);
	}
}
void getscc(){
	fill(vis,vis+n,0);
	sz.clear();
	repeat(i,0,n)if(!vis[i])dfs(i);
}
void shrink(){ //缩点，在a里重构
	static set<pii> eset;
	eset.clear();
	getscc();
	repeat(i,0,n)
	for(auto p:a[i])
	if(co[i]!=co[p])
		eset.insert({co[i],co[p]});
	n=sz.size();
	repeat(i,0,n){
		a[i].clear();
		w[i]=sz[i];
	}
	for(auto i:eset){
		a[i.fi].push_back(i.se);
		//a[i.se].push_back(i.fi);
	}
}
```

- 例题：给一个有向图，连最少的边使其变为scc。解：scc缩点后输出 $\max(\sum\limits_i[indeg[i]=0],\sum\limits_i[outdeg[i]=0])$，特判只有一个scc的情况

### 边双连通分量 | Tarjan

- 编号从0开始，$O(V+E)$

```c++
void dfs(int x,int fa){ //Tarjan求边双联通分量
	vis[x]=instk[x]=1; stk.push(x);
	dfn[x]=low[x]=++dcnt;
	for(auto p:a[x])
	if(p!=fa){
		if(!vis[p])dfs(p,x);
		if(instk[p])low[x]=min(low[x],low[p]);
	}
	else fa=-1; //处理重边
	if(low[x]==dfn[x]){
		int t; sz.push_back(0); //记录
		do{
			t=stk.top();
			stk.pop();
			instk[t]=0;
			sz.back()+=w[t]; //记录
			co[t]=sz.size()-1; //染色
		}while(t!=x);
	}
}
void getscc(){
	fill(vis,vis+n,0);
	sz.clear();
	repeat(i,0,n)if(!vis[i])dfs(i,-1);
}
//全局变量，shrink()同scc
```

### 割点×割顶

- Tarjan

```c++
bool vis[N],cut[N]; //cut即结果，cut[i]表示i是否为割点
int dfn[N],low[N];
int dcnt; //时间戳
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
	if(isroot && ch>=2) //根结点判断方法
		cut[x]=1;
}
```

## 2-sat问题

<H3>可行解</H3>

- 有 $2n$ 个顶点，其中顶点 $2i$ 和顶点 $2i+1$ 中能且仅能选一个，边 $(u,v)$ 表示选了 $u$ 就必须选 $v$，求一个可行解
- 暴力版，可以跑出字典序最小的解，编号从 $0$ 开始，$O(VE)$，（~~但是难以跑到上界~~）

```c++
struct twosat{ //暴力版
	int n;
	vector<int> g[N*2];
	bool mark[N*2]; //mark即结果，表示是否选择了这个点
	int s[N],c;
	bool dfs(int x){
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
	void add(int x,int y){ //这个函数随题意变化
		g[x].push_back(y^1); //选了x就必须选y^1
		g[y].push_back(x^1); //选了y就必须选x^1
	}
	bool solve(){ //返回是否存在解
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

- SCC缩点版，$O(V+E)$，空缺
- 2-SAT计数
- 空缺（太恐怖了）

## 支配树 | Lengauer−Tarjan算法

- 有向图给定源点，若删掉 $r$，源点不可达 $u$，则称 $r$ 是 $u$ 的支配点
- 支配树即所有非源点的点与最近支配点(idom)连边形成的树（源点为根）

```c++
vector<int> a[N],b[N],tr[N]; //tr: result
int fa[N],dfn[N],dcnt,arcdfn[N];
int c[N],best[N],sm[N],im[N]; //im: result
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
		tr[sm[x]=arcdfn[mn]]<<x;
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
		tr[im[u]]<<u;
	}
}
```

## 图上的NP问题

### 最大团+极大团计数

- 求最大团顶点数（和最大团），`g[][]` 编号从 $0$ 开始，$O(\exp)$

```c++
int g[N][N],f[N][N],v[N],Max[N],n,ans; //g[][]是邻接矩阵，n是顶点数
//vector<int> rec,maxrec; //maxrec是最大团
bool dfs(int x,int cur){
	if(cur==0)
		return x>ans;
	repeat(i,0,cur){
		int u=f[x][i],k=0;
		if(Max[u]+x<=ans)return 0;
		repeat(j,i+1,cur)
		if(g[u][f[x][j]])
			f[x+1][k++]=f[x][j];
		//rec.push_back(u);
		if(dfs(x+1,k))return 1;
		//rec.pop_back();
	}
	return 0;
}
void solve(){
	ans=0; //maxrec.clear();
	repeat_back(i,0,n){
		int k=0;
		repeat(j,i+1,n)
		if(g[i][j])
			f[1][k++]=j;
		//rec.clear(); rec.push_back(i);
		if(dfs(1,k)){
			ans++;
			//maxrec=rec;
		}
		Max[i]=ans;
	}
}
```

- 求极大团个数（和所有极大团），`g[][]` 的编号从 $1$ 开始！$O(\exp)$

```c++
int g[N][N],n;
//vector<int> rec; //存当前极大团
int ans,some[N][N],none[N][N]; //some是未搜索的点，none是废除的点
void dfs(int d,int sn,int nn){
	if(sn==0 && nn==0)
		ans++; //此时rec是其中一个极大图
	//if(ans>1000)return; //题目要求_(:зゝ∠)_
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
		//rec.push_back(v);
		dfs(d+1,tsn,tnn);
		//rec.pop_back();
		some[d][i]=0;
		none[d][nn++]=v;
	}
}
void solve(){ //运行后ans即极大团数
	ans=0;
	for(int i=0;i<n;++i)
		some[0][i]=i+1;
	dfs(0,n,0);
}
```

### 最小染色数

- $O(\exp)$，`n=17` 可用

```c++
int n,m;
int g[N]; //二进制邻接矩阵
bool ind[1<<N]; //是否为(极大)独立集
int dis[1<<N];
vector<int> a; //存独立集
#define np (1<<n)
int bfs(){ //重复覆盖简略版
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
int solve(){ //返回最小染色数
	mst(g,0);
	for(auto i:eset){
		int x=i.fi,y=i.se;
		g[x]|=1<<y;
		g[y]|=1<<x;
	}
	//求所有独立集
	ind[0]=1;
	repeat(i,1,np){
		int w=63-__builtin_clzll(ll(i)); //最高位
		if((g[w]&i)==0 && ind[i^(1<<w)])
			ind[i]=1;
	}
	//删除所有不是极大独立集的独立集
	repeat(i,1,np)
	if(ind[i]){
		for(int j=1;j<np;j<<=1)
		if((i&j)==0 && ind[i|j]){
			ind[i]=0;
			break;
		}
		if(ind[i])
			a.push_back(i); //记录极大独立集
	}
	return bfs();
}
```

## 弦图+区间图

- 弦是连接环上不相邻点的边；弦图是所有长度大于3的环都有弦的无向图（类似三角剖分）
- 单纯点：所有与v相连的点构成一个团，则v是一个单纯点
- 完美消除序列：即点集的一个排列 $[v_1,v_2,...,v_n]$ 满足任意 $v_i$ 在 $[v_{i+1},...,v_n]$ 的导出子图中是一个单纯点
- 定理：无向图是弦图 $\Leftrightarrow$ 无向图存在完美消除序列
- 定理：最大团顶点数 $\le$ 最小染色数（弦图取等号）
- 定理：最大独立集顶点数 $\le$ 最小团覆盖（弦图取等号）

***

- 最大势算法MCS求完美消除序列：每次求出与 $[v_{i+1},...,v_n]$ 相邻点数最大的点作为 $v_i$
- `e[][]`点编号从 $1$ 开始！`rec` 下标从 $1$ 开始！桶优化，$O(V+E)$

```c++
vector<int> e[N];
int n,rec[N]; //rec[1..n]是结果
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

```c++
bool judge(){ //返回是否是完美消除序列（先要跑一遍MCS）
	static int s[N],rnk[N];
	repeat(i,1,n+1){
		rnk[rec[i]]=i;
		sort(e[i].begin(),e[i].end()); //方便二分查找，内存足够直接unmap
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

```c++
int color(){ //返回最大团点数/最小染色数
	return *max_element(lab+1,lab+n+1)+1;
	/* //以下求最大团
	static int rnk[N];
	repeat(i,1,n+1)rnk[rec[i]]=i;
	int x=max_element(lab+1,lab+n+1)-lab;
	rec2.push_back(x);
	for(auto p:e[x])
	if(rnk[x]<rnk[p])
		rec2.push_back(x);
	*/
}
int maxindset(){ //返回最大独立集点数/最小团覆盖数
	int ans=0;
	fill(vis,vis+n+1,0);
	repeat(i,1,n+1){
		int x=rec[i];
		if(!vis[x]){
			ans++; //rec2.push_back(x); //记录最大独立集
			for(auto p:e[x])
				vis[p]=1;
		}
	}
	return ans;
}
int cliquecnt(){ //返回极大团数
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

## 仙人掌 | 圆方树

- 仙人掌：每条边至多属于一个简单环的无向联通图
- 圆方树：原来的点称为圆点，每个环新建一个方点，环上的圆点都与方点连接
- 子仙人掌：以 $r$ 为根，点 $p$ 的子仙人掌是删掉 $p$ 到 $r$ 的所有简单路径后 $p$ 所在的联通块。这个子仙人掌就是圆方树中以 $r$ 为根时，$p$ 子树中的所有圆点
- 仙人掌的判定（dfs树上差分）编号从哪开始都可以，$O(n+m)$

```c++
vector<int> a[N]; //vector<int> rec; //rec存每个环的大小
bool vis[N]; int fa[N],lab[N],dep[N]; bool ans;
void dfs(int x){
	vis[x]=1;
	for(auto p:a[x])if(p!=fa[x]){
		if(!vis[p]){
			fa[p]=x; dep[p]=dep[x]+1;
			dfs(p); lab[x]+=lab[p];
		}
		else if(dep[p]<dep[x]){
			lab[x]++; lab[p]--;
			//rec.push_back(dep[x]-dep[p]+1);
		}
	}
	if(lab[x]>=2)ans=0;
}
bool iscactus(int s){
	fill(vis,vis+n+1,0);
	ans=1; fa[s]=-1; dfs(s); return ans;
}
```

## 二分图

### 二分图的一些概念

***

- 最小点覆盖（最小的点集，使所有边都能被覆盖） = 最大匹配
- 最大独立集 = 顶点数 - 最大匹配
- 最小路径覆盖 = （开点前）顶点数 - 最大匹配，右顶点未被匹配的都看作起点
- 最小带权点覆盖 = 点权之和 - 最大带权独立集（左式用最小割求）

***

- 霍尔定理：最大匹配 = 左顶点数 $\Leftrightarrow$ 所有左顶点子集 $S$ 都有 $|S|\le|\omega(S)|$ ，$\omega(S)$ 是 $S$ 的领域
- 运用：若在最大匹配中有 $t$ 个左顶点失配，因此最大匹配 = 左顶点数 - $t$
- 对任意左顶点子集 $S$ 都有 $|S|\le|\omega(S)|+t$，$t\ge|S|-|\omega(S)|$ ，求右式最大值即可求最大匹配

***

### 二分图匹配×最大匹配

- 匈牙利×hungarian，左右顶点编号从 $0$ 开始，$O(VE)$

```c++
vector<int> a[N]; //a: input, the left vertex x is connected to the right vertex a[x][i]
int dcnt,mch[N],dfn[N]; //mch: output, the right vertex p is connected to the left vertex mch[p]
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
int hun(int n,int m){ //n,m: the number of the left/right vertexes. return max matching
	int ans=0;
	repeat(i,0,m)mch[i]=-1;
	repeat(i,0,n){
		dcnt++;
		if(dfs(i))ans++;
	}
	return ans;
}
```

- HK算法×Hopcroft-karp，左顶点编号从 $0$ 开始，右顶点编号从 $n$开始，$O(E\sqrt V)$

```c++
vector<int> a[N]; //a: input, the left vertex x is connected to the right vertex a[x][i]
int mch[N*2],dep[N*2]; //mch: output, the vertex p is connected to the vertex mch[p] (p could be either left or right vertex)
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
int solve(int n,int m){ //n,m: the number of the left/right vertexes. return max matching
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

- 网络流建图，编号从 $0$ 开始，$O(E\sqrt V)$

```c++
int work(int n1,int n2,vector<pii> &eset){
	int n=n1+n2+2;
	int s=0,t=n1+n2+1;
	flow.init(n);
	repeat(i,1,n1+1)add(s,i,1);
	repeat(i,n1+1,n1+n2+1)add(i,t,1);
	for(const auto &i:eset){
		int x=i.fi,y=i.se;
		add(x+1,n1+y+1,1);
	}
	return flow.solve(s,t);
}
```

### 最大权匹配 | KM

- 求满二分图的最大权匹配
- 如果没有边就建零边，而且要求n<=m
- 编号从 $0$ 开始，$O(n^3)$

```c++
int e[N][N],n,m; //邻接矩阵，左顶点数，右顶点数
int lx[N],ly[N]; //顶标
int mch[N]; //右顶点i连接的左顶点编号
bool fx[N],fy[N]; //是否在增广路上
bool dfs(int i){
	fx[i]=1;
	repeat(j,0,n)
	if(lx[i]+ly[j]==e[i][j] && !fy[j]){
		fy[j]=1;
		if(mch[j]==-1 || dfs(mch[j])){
			mch[j]=i;
			return 1;
		}
	}
	return 0;
}
void update(){
	int fl=inf;
	repeat(i,0,n)if(fx[i])
	repeat(j,0,m)if(!fy[j])
		fl=min(fl,lx[i]+ly[j]-e[i][j]);
	repeat(i,0,n)if(fx[i])lx[i]-=fl;
	repeat(j,0,m)if(fy[j])ly[j]+=fl;
}
int solve(){ //返回匹配数
	repeat(i,0,n){
		mch[i]=-1;
		lx[i]=ly[i]=0;
		repeat(j,0,m)
			lx[i]=max(lx[i],e[i][j]);
	}
	repeat(i,0,n)
	while(1){
		repeat(j,0,m)
			fx[j]=fy[j]=0;
		if(dfs(i))break;
		else update();
	}
	int ans=0;
	repeat(i,0,m)
	if(mch[i]!=-1)
		ans+=e[mch[i]][i];
	return ans;
}
```

### 稳定婚姻 | 延迟认可

- 稳定意味着不存在一对不是情侣的男女，都认为当前伴侣不如对方
- 编号从 $0$ 开始，$O(n^2)$

```c++
struct node{
	int s[N]; //s的值给定
		//对男生来说是女生编号排序
		//对女生来说是男生的分数
	int now; //选择的伴侣编号
}a[N],b[N]; //男生，女生
int tr[N]; //男生尝试表白了几次
queue<int> q; //单身狗（男）排队
bool match(int x,int y){ //配对，返回是否成功
	int x0=b[y].now;
	if(x0!=-1){
		if(b[y].s[x]<b[y].s[x0])
			return 0; //分数不够，竞争失败
		q.push(x0);
	}
	a[x].now=y;
	b[y].now=x;
	return 1;
}
void stable_marriage(){ //运行后a[].now,b[].now即结果
	q=queue<int>();
	repeat(i,0,n){
		b[i].now=-1;
		q.push(i);
		tr[i]=0;
	}
	while(!q.empty()){
		int x=q.front(); q.pop();
		int y=a[x].s[tr[x]++]; //下一个最中意女生
		if(!match(x,y))
			q.push(x); //下次努力
	}
}
```

### 一般图最大匹配 | 带花树

- 对于一个无向图，找最多的边使得这些边两两无公共端点
- 编号从 $1$ 开始，$O(n^3)$

```c++
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
int solve(){ //返回匹配数，mch[]是匹配结果（即匹配x和mch[x]），==0表示不匹配
	int ans=0; fill(mch,mch+n+1,0);
	repeat(i,1,n+1)ans+=(!mch[i] && match(i));
	return ans;
}
```

- 例题：给定一个无向图和 $d_i$（$1\le d_i\le 2$），求是否能删去一些边后满足点 $i$ 的度刚好是 $d_i$

```c++
::n=n*2+m*2; //::n是带花树板子里的n
repeat(i,1,n+1)cnt+=deg[i]=read();
repeat(i,1,m+1){
	int x=read(),y=read();
	if(deg[x]==2 && deg[y]==2){ //(x,e)(x',e)(y,e')(y',e')(e,e')
		add(x,n*2+i),add(x+n,n*2+i),add(y,n*2+m+i),add(y+n,n*2+m+i),add(n*2+i,n*2+m+i);
		cnt+=2;
	}
	else{ //(x,y),度为2再添一条边
		add(x,y); if(deg[x]==2)add(x+n,y); if(deg[y]==2)add(x,y+n);
	}
}
puts(solve()*2==cnt?"Yes":"No");
```

## 网络流

### 网络流的一些概念

***

- $c(u,v)$ 为 $u$ 到 $v$ 的容量，$f(u,v)$ 为 $u$ 到 $v$ 的流量，$f(u,v)<c(u,v)$
- $c[X,Y]$ 为 $X$ 到 $Y$ 的容量和，不包括 $Y$ 到 $X$ 的容量；$f(X,Y)$ 为 $X$ 到 $Y$ 的流量和，要减去 $Y$ 到 $X$ 的流量

***

- 费用流（最小费用最大流）：保证最大流后的最小费用

***

- 割：割 $[S,T]$ 是点集的一个分割且 $S$ 包含源点，$T$ 包含汇点，称 $f(S,T)$ 为割的净流，$c[S,T]$ 为割的容量
- 最大流最小割定理：最大流即最小割容量
- 求最小割：在最大流残量网络中，令源点可达的点集为 $S$，其余的为 $T$ 即可（但是满流边不一定都在 $S,T$ 之间）

***

- 闭合子图：子图内所有点的儿子都在子图内。点权之和最大的闭合子图为最大闭合子图
- 求最大闭合子图：点权为正则s向该点连边，边权为点权，为负则向t连边，边权为点权绝对值，原图所有边的权设为inf，跑最小割。如果连s的边被割则不选这个点，若连t的边被割则选这个点

***

### 最大流

- 以下顶点编号均从 $0$ 开始

#### Dinic

- 多路增广，$O(V^2E)$

```c++
struct FLOW{
	struct edge{int to,w,nxt;};
	vector<edge> a; int head[N],cur[N];
	int n,s,t;
	queue<int> q; bool inque[N];
	int dep[N];
	void ae(int x,int y,int w){ //add edge
		a.push_back({y,w,head[x]});
		head[x]=a.size()-1;
	}
	bool bfs(){ //get dep[]
		fill(dep,dep+n,inf); dep[s]=0;
		copy(head,head+n,cur);
		q=queue<int>(); q.push(s);
		while(!q.empty()){
			int x=q.front(); q.pop(); inque[x]=0;
			for(int i=head[x];i!=-1;i=a[i].nxt){
				int p=a[i].to;
				if(dep[p]>dep[x]+1 && a[i].w){
					dep[p]=dep[x]+1;
					if(inque[p]==0){
						inque[p]=1;
						q.push(p);
					}
				}
			}
		}
		return dep[t]!=inf;
	}
	int dfs(int x,int flow){ //extend
		int now,ans=0;
		if(x==t)return flow;
		for(int &i=cur[x];i!=-1;i=a[i].nxt){
			int p=a[i].to;
			if(a[i].w && dep[p]==dep[x]+1)
			if((now=dfs(p,min(flow,a[i].w)))){
				a[i].w-=now;
				a[i^1].w+=now;
				ans+=now,flow-=now;
				if(flow==0)break;
			}
		}
		return ans;
	}
	void init(int _n){
		n=_n+1; a.clear();
		fill(head,head+n,-1);
		fill(inque,inque+n,0);
	}
	int solve(int _s,int _t){ //return max flow
		s=_s,t=_t;
		int ans=0;
		while(bfs())ans+=dfs(s,inf);
		return ans;
	}
}flow;
void add(int x,int y,int w){flow.ae(x,y,w),flow.ae(y,x,0);}
//先flow.init(n)，再add添边，最后flow.solve(s,t)
```

#### ISAP

- 仅一次bfs与多路增广，$O(V^2E)$，有锅！！

```c++
struct FLOW{
	struct edge{int to,w,nxt;};
	vector<edge> a; int head[N];
	int cur[N];
	int n,s,t;
	queue<int> q;
	int dep[N],gap[N];
	void ae(int x,int y,int w){
		a.push_back({y,w,head[x]});
		head[x]=a.size()-1;
	}
	bool bfs(){
		fill(dep,dep+n,-1); dep[t]=0;
		fill(gap,gap+n,0); gap[0]=1;
		q.push(t);
		while(!q.empty()){
			int x=q.front(); q.pop();
			for(int i=head[x];i!=-1;i=a[i].nxt){
				int p=a[i].to;
				if(dep[p]!=-1)continue;
				dep[p]=dep[x]+1;
				q.push(p);
				gap[dep[p]]++;
			}
		}
		return dep[s]!=-1;
	}
	int dfs(int x,int fl){
		int now,ans=0;
		if(x==t)return fl;
		for(int i=cur[x];i!=-1;i=a[i].nxt){
			cur[x]=i;
			int p=a[i].to;
			if(a[i].w && dep[p]+1==dep[x])
			if((now=dfs(p,min(fl,a[i].w)))){
				a[i].w-=now;
				a[i^1].w+=now;
				ans+=now,fl-=now;
				if(fl==0)return ans;
			}
		}
		gap[dep[x]]--;
		if(gap[dep[x]]==0)dep[s]=n;
		dep[x]++;
		gap[dep[x]]++;
		return ans;
	}
	void init(int _n){
		n=_n+1;
		a.clear();
		fill(head,head+n,-1);
	}
	int solve(int _s,int _t){ //返回最大流
		s=_s,t=_t;
		int ans=0;
		if(bfs())
		while(dep[s]<n){
			copy(head,head+n,cur);
			ans+=dfs(s,inf);
		}
		return ans;
	}
}flow;
void add(int x,int y,int w){flow.ae(x,y,w),flow.ae(y,x,0);}
//先flow.init(n)，再add添边，最后flow.solve(s,t)
```

### 最小费用最大流 | MCMF

- 费用流一般指最小费用最大流（最大费用最大流把费用取反即可）
- MCMF，单路增广，$O(VE^2)$

```c++
struct FLOW{
	struct edge{int to,w,cost,nxt;};
	vector<edge> a; int head[N];
	int n,s,t,totcost;
	deque<int> q;
	bool inque[N];
	int dis[N];
	struct{int to,e;}pre[N];
	void ae(int x,int y,int w,int cost){
		a.push_back((edge){y,w,cost,head[x]});
		head[x]=a.size()-1;
	}
	bool spfa(){
		fill(dis,dis+n,inf); dis[s]=0;
		q.assign(1,s);
		while(!q.empty()){
			int x=q.front(); q.pop_front();
			inque[x]=0;
			for(int i=head[x];i!=-1;i=a[i].nxt){
				int p=a[i].to;
				if(dis[p]>dis[x]+a[i].cost && a[i].w){
					dis[p]=dis[x]+a[i].cost;
					pre[p]={x,i};
					if(inque[p]==0){
						inque[p]=1;
						if(!q.empty()
						&& dis[q.front()]<=dis[p])
							q.push_back(p);
						else q.push_front(p);
					}
				}
			}
		}
		return dis[t]!=inf;
	}
	void init(int _n){
		n=_n+1;
		a.clear();
		fill(head,head+n,-1);
		fill(inque,inque+n,0);
	}
	int solve(int _s,int _t){ //返回最大流，费用存totcost里
		s=_s,t=_t;
		int ans=0;
		totcost=0;
		while(spfa()){
			int fl=inf;
			for(int i=t;i!=s;i=pre[i].to)
				fl=min(fl,a[pre[i].e].w);
			for(int i=t;i!=s;i=pre[i].to){
				a[pre[i].e].w-=fl;
				a[pre[i].e^1].w+=fl;
			}
			totcost+=dis[t]*fl;
			ans+=fl;
		}
		return ans;
	}
}flow;
void add(int x,int y,int w,int cost){
	flow.ae(x,y,w,cost),flow.ae(y,x,0,-cost);
}
//先flow.init(n)，再add添边，最后flow.solve(s,t)
```

## 图论杂项

### 矩阵树定理

<H4>无向图矩阵树定理</H4>

- 生成树计数

```c++
void matrix::addedge(int x,int y){
	a[x][y]--,a[y][x]--;
	a[x][x]++,a[y][y]++;
}
lf matrix::treecount(){
	//for(auto i:eset)addedge(i.fi,i.se); //加边
	n--,m=n; //a[n-1][n-1]的余子式（选任一结点均可）
	return get_det();
}
```

<H4>有向图矩阵树定理</H4>

- 根向树形图计数，每条边指向父亲
- （叶向树形图，即每条边指向儿子，只要修改一个地方）
- 如果要求所有根的树形图之和，就求逆的主对角线之和乘以行列式（$A^*=|A|A^{-1}$）

```c++
void matrix::addedge(int x,int y){
	a[x][y]--;
	a[x][x]++; //叶向树形图改成a[y][y]++;
}
ll matrix::treecount(){
	//for(auto i:eset)addedge(i.fi,i.se); //加边
	repeat(i,s,n) //s是根结点
	repeat(j,0,n)
		a[i][j]=a[i+1][j];
	repeat(i,0,n)
	repeat(j,s,n)
		a[i][j]=a[i][j+1];
	n--,m=n; //a[s][s]的余子式
	return get_det();
}
```

<H4>BSET定理</H4>

- 有向欧拉图的欧拉回路总数等于任意根的根向树形图个数乘以 $\Pi(deg(v)-1)!$（←阶乘）（$deg(v)$ 是 $v$ 的入度或出度，~~反正入度等于出度~~）

<H4>Enumerative properties of Ferrers graphs</H4>

- 二分图，左顶点连编号为 $1,2,...,a_i$ 的右顶点，则该图的生成树个数为 $\dfrac{\prod\limits_{i∈A}deg_i}{\max\limits_{i∈A}deg_i}\cdot\dfrac{\prod\limits_{i∈B}deg_i}{\max\limits_{i∈B}deg_i}$ 左顶点度之积（去掉度最大的）乘以右顶点度之积（去掉度最大的）

### Prufer序列

- $n$ 个点的无根树与长度 $n-2$ 值域 $[1,n]$ 的序列有双射关系，Prufer序列就是其中一种
- 性质：$i$ 出现次数等于节点 $i$ 的度 $-1$
- 无根树转Prufer：设无根树点数为 $n$，每次删除度为 $1$ 且编号最小的结点并把它所连接的点的编号加入Prufer序列，进行 $n-2$ 次操作
- Prufer转无根树：计算每个点的度为在序列中出现的次数加 $1$，每次找度为 $1$ 的编号最小的点与序列中第一个点连接，并将后者的度减 $1$
- Cayley定理：完全图 $K_n$ 有 $n^{n-2}$ 棵生成树
- 扩展：$k$ 个联通块，第 $i$ 个联通块有 $s_i$个点，则添加 $k-1$ 条边使整个图联通的方案数有 $n^{k-2}\Pi_{i=1}^k s_i$ 个

### LGV引理

- DAG上固定 $2n$ 个点 $[A_1,\cdots,A_n,B_1,\cdots,B_n]$，若有 $n$ 条路径 $[A_1→B_1,\cdots,A_n→B_n]$ 两两不相交，则方案数为
- $M=\left|\begin{array}{c}e(A_1,B_1)&\cdots &e(A_1,B_n)\\\vdots&\ddots&\vdots\\e(A_n,B_1)&\cdots&e(A_n,B_n)\end{array}\right|$
- 其中 $e(u,v)$ 表示 $u→v$ 的路径计数

### others of 图论杂项

<H3>Havel-Hakimi定理</H3>

- 给定一个度序列，反向构造出这个图
- 解：贪心，每次让剩余度最大的顶点 $k$ 连接其余顶点中剩余度最大的 $deg_k$ 个顶点
- （我认为二路归并比较快，可是找到的代码都用了`sort()`）

<H3>无向图三元环计数</H3>

- 无向图定向，$pii(deg_i,i)>pii(deg_j,j)\Leftrightarrow$ 建立有向边 $(i,j)$。然后暴力枚举 $u$，将 $u$ 的所有儿子 $\omega(u)$ 标记为 $dcnt$，暴力枚举 $v∈\omega(u)$，若 $v$ 的儿子被标记为 $dcnt$ 则 $ans++$，$O(E\log E)$
