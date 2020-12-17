- [unclassified](#unclassified)
	- [Notice](#notice)
	- [线段树分治](#线段树分治)
	- [反悔贪心](#反悔贪心)
- [计算几何](#计算几何)
	- [曼哈顿最小生成树](#曼哈顿最小生成树)
	- [struct of 整点直线](#struct-of-整点直线)
- [数据结构](#数据结构)
	- [吉老师线段树](#吉老师线段树)
	- [带懒标记treap](#带懒标记treap)
	- [可撤销种类并查集](#可撤销种类并查集)
	- [可删堆](#可删堆)
	- [主席树](#主席树)
	- [struct of 二维数组](#struct-of-二维数组)
	- [析合树](#析合树)
- [数学](#数学)
	- [超级卡特兰数](#超级卡特兰数)
	- [分治FFT](#分治fft)
	- [第二类斯特林数·行](#第二类斯特林数行)
	- [分数类](#分数类)
	- [线性基求交](#线性基求交)
	- [Miller-Rabin and Pollard_rho 更新](#miller-rabin-and-pollard_rho-更新)
- [图论](#图论)
	- [Kruskal重构树](#kruskal重构树)
	- [树哈希 补充](#树哈希-补充)
	- [dsu重构树](#dsu重构树)
	- [虚树](#虚树)
	- [树上启发式合并](#树上启发式合并)
- [其他](#其他)
	- [Java](#java)
	- [python3](#python3)
	- [字符串哈希模板更新](#字符串哈希模板更新)
- [斯坦纳树 from OIwiki](#斯坦纳树-from-oiwiki)
	- [例题](#例题)
- [高精度计算 from OIwiki](#高精度计算-from-oiwiki)
	- [封装类](#封装类)
- [类欧几里得算法 from OIwiki](#类欧几里得算法-from-oiwiki)
	- [引入](#引入)
	- [扩展](#扩展)
		- [推导 g](#推导-g)
		- [推导 h](#推导-h)
- [排列组合 from OIwiki](#排列组合-from-oiwiki)
	- [组合数性质](#组合数性质)

## unclassified

### Notice

- NTT中，$\omega_n=$ `qpow(G,(mod-1)/n))`
- 原根

```c++
if (m == 167772161) return 3;
if (m == 469762049) return 3;
if (m == 754974721) return 11;
if (m == 998244353) return 3;
```

- 矩阵行列式引理Matrix Determinant Lemma：$n\times n$ 可逆矩阵 $A$ 和 $n$ 维列向量 $u,v$ 有 $\det(A+uv^T)=\det(A)(1+v^TA^{-1}u)$
- 注意原线段树板子init里z没清空，注意啊啊
- 指令集优化

```c++
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,popcnt,tune=native")
```

### 线段树分治

- 支持添删边的离线操作
- 对时间建立线段树，每个结点开一个vector。对一条边，添加到删除的时间区间，插入到线段树中。最后对线段树dfs一遍统计答案，向下走即添边，向上走即撤销，用可撤销数据结构维护
- 复杂度为 $O(n\log n)$ 乘以可撤销数据结构复杂度

### 反悔贪心

- 例：第 $i$ 天要么获得 $A_i$ 元要么获得 $B_i$ 元和 $1$ 积分，$A_i\ge 0$，$B_i$ 可能 $<0$，问获得积分最大值
- 能拿 $B$ 就拿 $B$，不然就把之前某个 $B$ 换成 $A$ 然后拿这次的 $B$（如果更优的话）

```c++
priority_queue<int> q;
int n=read(),now=0,ans=0;
repeat(i,0,n){
	int A=read(),B=read();
	if(now+B>=0){
		q.push(A-B);
		ans++;
		now+=B;
	}
	else if(!q.empty() && now+q.top()+B>=0 && q.top()>A-B){
		now+=q.top()+B;
		q.pop();
		q.push(A-B);
	}
	else now+=A;
}
```

## 计算几何

### 曼哈顿最小生成树

- input: `n,a[i].x,a[i].y`，`a[i].p` 是没用的。编号从 $1$ 开始，$O(n\log n)$

```c++
DSU d;
int n,w[N],c[N];
struct node{
	int x,y,p;
}a[N],b[N];
vector<node> e;
int dist(int x,int y){
	return abs(a[x].x-a[y].x)+abs(a[x].y-a[y].y);
}
#define lb(x) (x&-x)
struct BIT{ //special
	int t[N];
	void init(){
		fill(t,t+n+1,0);
	}
	void insert(int x,int p){
		for(;x<=n;x+=lb(x))
		if(w[p]<=w[t[x]])
			t[x]=p;
	}
	int query(int x){
		int ans=0;
		for(;x!=0;x-=lb(x))
		if(w[t[x]]<=w[ans])
			ans=t[x];
		return ans;
	}
}bit; 
void work(){
	bit.init();
	repeat(i,1,n+1)c[i]=b[i].y; sort(c+1,c+n+1);
	sort(b+1,b+n+1,[](node a,node b){
		return pii(a.x,a.y)<pii(b.x,b.y);
	});
	repeat(i,1,n+1){
		int u=upper_bound(c+1,c+n+1,b[i].y)-c,j=bit.query(u);
		if(j)e.push_back({b[i].p,j,dist(b[i].p,j)});
		bit.insert(u,b[i].p);
	}
}
ll mmst(){
	w[0]=inf; e.clear(); d.init(n);
	repeat(i,1,n+1){
		b[i]={-a[i].x,a[i].x-a[i].y,i};
		w[i]=a[i].x+a[i].y;
	}
	work();
	repeat(i,1,n+1){
		b[i]={-a[i].y,a[i].y-a[i].x,i};
	}
	work();
	repeat(i,1,n+1){
		b[i]={a[i].y,-a[i].x-a[i].y,i};
		w[i]=a[i].x-a[i].y;
	}
	work();
	repeat(i,1,n+1){
		b[i]={-a[i].x,a[i].y+a[i].x,i};
	}
	work();
	sort(e.begin(),e.end(),[](node a,node b){
		return a.p<b.p;
	});
	ll ans=0;
	for(auto i:e)
	if(d[i.x]!=d[i.y]){
		d[i.x]=d[i.y],ans+=i.p;
	}
	return ans;
}
```

### struct of 整点直线

```c++
struct line{
	ll up,down,dx,dy; //y=(dy/dx)x+(up/down) or x=(up/down)
	void adjust(ll &x,ll &y){
		if(x<0)x=-x,y=-y;
		if(x==0)y=1;
		else if(y==0)x=1;
		else{
			ll d=abs(__gcd(x,y));
			x/=d; y/=d;
		}
	}
	line(ll x1,ll y1,ll x2,ll y2){
		dx=(x1-x2),dy=(y1-y2);
		adjust(dx,dy);
		if(dx!=0){
			up=-dy*x1+dx*y1;
			down=dx;
			adjust(up,down);
		}
		else{
			up=-dx*y1+dy*x1;
			down=dy;
			adjust(up,down);
		}
	}
	pii d(){return {dx,dy};} //斜率
	pii d2(){ //垂线斜率
		ll ddx=-dy,ddy=dx;
		adjust(ddx,ddy);
		return {ddx,ddy};
	}
	bool operator==(line b)const{
		return make_tuple(up,down,dx,dy)
			== make_tuple(b.up,b.down,b.dx,b.dy);
	}
};
struct h{ //Hash
	ll operator()(line a)const{
		return a.up+a.down*10000+a.dx*100000000+a.dy*1000000000000;
	}
};
```

## 数据结构

### 吉老师线段树

- 区间取min，区间max，区间和，$O(n\log n)$

```c++
int in[N],n;
struct seg{
	#define lc (u*2)
	#define rc (u*2+1)
	int mx[N<<2],se[N<<2],cnt[N<<2],tag[N<<2];
	ll sum[N<<2];
	void up(int u){ //private
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
	void pushtag(int u,int tg){ //private
		if(mx[u]<=tg)return;
		sum[u]+=(1ll*tg-mx[u])*cnt[u];
		mx[u]=tag[u]=tg;
	}
	void down(int u){ //private
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

- 区间取min，区间取max，区间加，区间min，区间max，区间和，$O(n\log^2 n)$

```c++
int in[N],n;
struct seg{
	#define lc (u*2)
	#define rc (u*2+1)
	struct node{
		int mx,mx2,mn,mn2,cmx,cmn,tmx,tmn,tad;
		ll sum;
	};
	node t[N<<2];
	void up(int u){ //private
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
	void push_add(int u,int l,int r,int v){ //private
		t[u].sum+=(r-l+1ll)* v;
		t[u].mx+=v,t[u].mn+=v;
		if(t[u].mx2!=-inf)t[u].mx2+=v;
		if(t[u].mn2!=inf)t[u].mn2+=v;
		if(t[u].tmx!=-inf)t[u].tmx+=v;
		if(t[u].tmn!=inf)t[u].tmn+=v;
		t[u].tad+=v;
	}
	void push_min(int u,int tg){ //private
		if(t[u].mx<=tg)return;
		t[u].sum+=(tg*1ll-t[u].mx)*t[u].cmx;
		if(t[u].mn2==t[u].mx)t[u].mn2=tg;
		if(t[u].mn==t[u].mx)t[u].mn=tg;
		if(t[u].tmx>tg)t[u].tmx=tg;
		t[u].mx=tg,t[u].tmn=tg;
	}
	void push_max(int u,int tg){ //private
		if(t[u].mn>tg)return;
		t[u].sum+=(tg*1ll-t[u].mn)*t[u].cmn;
		if(t[u].mx2==t[u].mn)t[u].mx2=tg;
		if(t[u].mx==t[u].mn)t[u].mx=tg;
		if(t[u].tmn<tg)t[u].tmn=tg;
		t[u].mn=tg,t[u].tmx=tg;
	}
	void down(int u,int l,int r){ //private
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

### 带懒标记treap

```c++
struct treap{
	struct node{
		int pri,sz; ll v,tag,s;
		node *l,*r;
		node(int _v){pri=(int)rnd(); v=s=_v; l=r=0; sz=1; tag=0;}
		node(){}
		friend int size(node *u){return u?u->sz:0;}
		friend ll sum(node *u){return u?(u->down(),u->s):0;}
		void up(){ //private
			sz=1+size(l)+size(r);
			s=v+sum(l)+sum(r);
		}
		void down(){ //private
			if(tag){
				v+=tag; s+=sz*tag;
				if(l)l->tag+=tag;
				if(r)r->tag+=tag;
				tag=0;
			}
		}
		friend pair<node *,node *> split(node *u,int key){ //private
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
		friend node *merge(node *x,node *y){ //private
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

### 可撤销种类并查集

- （维护是否有奇环）

```c++
namespace DSU{
	int a[N],r[N],sz[N];
	vector<pair<int *,int>> rec;
	void init(int n){repeat(i,0,n+1)a[i]=i,r[i]=0,sz[i]=1;}
	int plus(int a,int b){return a^b;}
	int inv(int a){return a;}
	int fa(int x){return a[x]==x?x:fa(a[x]);}
	int R(int x){return a[x]==x?r[x]:plus(r[x],R(a[x]));} //relation<x,fa(x)>
	int R(int x,int y){return plus(R(x),inv(R(y)));} //relation<x,y>
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

### 可删堆

- 原理：双堆模拟

```c++
struct heap{
	priority_queue<int> a,b;  //heap=a-b
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
	int top2(){
		int t=top(); pop();
		int ans=top(); push(t);
		return ans;
	}
	int size(){return a.size()-b.size();}
};
```

### 主席树

- 初始化 `init(l,r)`，版本复制 `his[i]=his[j]`（先复制再修改）
- 单点修改 `update(his[i],x,k)`，区间询问 `query(his[i],x,y)`
- 权值线段树 $his[i]\setminus his[j]$ 询问k小 `kth(his[i],his[j],k)`
- 静态区间k小：构造 $his[i]$ 为区间 $[1,i]$ 的权值线段树，`kth(his[r],his[l-1],k)` 即区间k小
- 修改询问 $O(\log n)$

```c++
namespace seg{
	struct{
		ll x; int l,r;
	}a[N<<5];
	int his[N],cnt,l0,r0;
	void init(int l,int r){
		l0=l,r0=r;
		cnt=0;
	}
	void update(int &u,int x,ll k,int l=l0,int r=r0){ //tr[u][x]+=k
		a[++cnt]=a[u]; u=cnt;
		if(l==r){a[u].x+=k; return;}
		int m=(l+r)/2;
		if(x<=m)update(a[u].l,x,k,l,m);
		else update(a[u].r,x,k,m+1,r);
		a[u].x=a[a[u].l].x+a[a[u].r].x;
	}
	ll query(int u,int x,int y,int l=l0,int r=r0){ //sum(tr[u][x..y])
		if(!u || x>r || y<l)return 0;
		if(x<=l && y>=r)return a[u].x;
		int m=(l+r)/2;
		return query(a[u].l,x,y,l,m)+query(a[u].r,x,y,m+1,r);
	}
	ll kth(int u,int v,int k,int l=l0,int r=r0){ //kth in (tr[u]-tr[v])[x..y]
		if(l==r)return l;
		int m=(l+r)/2,lv=a[a[u].l].x-a[a[v].l].x;
		if(k<=lv)return kth(a[u].l,a[v].l,k,l,m);
		else return kth(a[u].r,a[v].r,k-lv,m+1,r);
	}
}using namespace seg;
```

### struct of 二维数组

- （可以存储类似 $n\times m\le 2\times 10^5$ 的二维数组）

```c++
struct mat{
	ll a[N]; int n,m;
	void operator()(int _n,int _m){n=_n,m=_m;} //initialization
	ll *operator[](int x){
		return a+x*m;
	}
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
}a;
```

### 析合树

- 定义连续段为一个区间，区间内所有数排序后为连续正整数
- 析合树：每个节点对应一个连续段，合点由儿子顺序或倒序组成，析点为乱序
- 给定一个排列，询问包含给定区间的最短连续段
- 注意代码里N已经是两倍大小了，编号从 $1$ 开始，$O(n\log n)$

```c++
int n, m, a[N], st1[N], st2[N], tp1, tp2, rt;
int L[N], R[N], M[N], id[N], cnt, typ[N], st[N], tp;
#define log(x) (31 - __builtin_clz(x))
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
        int t = log(r - l + 1);
        return min(mn[l][t], mn[r - (1 << t) + 1][t]);
    }
    int qmax(int l, int r) {
        int t = log(r - l + 1);
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
    repeat(i, 1, log(dep[u]) + 1) fa[u][i] = fa[fa[u][i - 1]][i - 1];
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

## 数学

### 超级卡特兰数

- $(0,0)$ 走到 $(n,n)$ 方案数，只能往右、上、右上走，且满足 $y\le x$
- $S_{0..10}=1, 2, 6, 22, 90, 394, 1806, 8558, 41586, 206098, 1037718$
- $\displaystyle S_n=S_{n-1}+\sum_{k=0}^{n-1}S_kS_{n-1-k}$
- $F_0=S_0,2F_i=S_i,F_n=\dfrac{(6n-3)F_{n-1}-(n-2)F_{n-2}}{n+1}$
- 通项公式 $\displaystyle S_n=\dfrac{1}{n}\sum_{k=1}^n2^kC_n^kC_n^{k-1},n\ge1$

### 分治FFT

- 比如 $\displaystyle f[i]=\sum_{j=1}^if[i-j]g[i]$，把要求的多项式分成两边，先算 $f[0..n-1]$ 对自己的贡献（此时 $f[0..n-1]$ 已确定），然后算 $f[0..n-1]$ 对 $f[n..2n-1]$ 的贡献，再算 $f[n..2n-1]$ 对自己的贡献，$O(n\log^2n)$

```c++
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
//int n=polyinit(g,n1); fill(f,f+n,0); f[0]=1; work(0,n);
```

- 卡特兰数 $\displaystyle S_n=\sum_{k=0}^{n-1}S_kS_{n-1-k}$

```c++
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
//fill(f,f+n,0); f[0]=1; work(0,n);
```

- 超级卡特兰数 $\displaystyle S_n=S_{n-1}+\sum_{k=0}^{n-1}S_kS_{n-1-k}$

```c++
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
//fill(f,f+n,0); fill(f0,f0+n,0); f[0]=f0[0]=1; work(0,n);
```

### 第二类斯特林数·行
- $\displaystyle S(n,r)=[x^r](\sum_{i=0}^n\dfrac{(-1)^i}{i!}x^i)(\sum_{i=0}^{n}\dfrac{i^n}{i!}x^i)$

```c++
repeat(i,0,n1+1){
	a[i]=C.inv[i]; if(i%2==1)a[i]=-a[i];
	b[i]=qpow(i,n1)*C.inv[i]%mod;
}
int n=polyinit(a,n1+1); polyinit(b,n1+1);
conv(a,b,n,a);
```

### 分数类

- （可以直接哈希）（避免0/0，会当成0/1处理）

```c++
struct frac{
	ll x,y;
	frac(ll x=0,ll y=1):x(x),y(y){init();}
	void init(){
		if(y<0)x=-x,y=-y;
		if(x==0)y=1;
		else if(y==0)x=1;
		else{
			ll d=abs(__gcd(x,y));
			x/=d; y/=d;
		}
	}
	frac operator-()const{
		return frac(-x,y);
	}
	friend frac operator+(const frac &a,const frac &b){
		return frac(a.x*b.y+a.y*b.x,a.y*b.y);
	}
	friend frac operator-(const frac &a,const frac &b){
		return a+-b;
	}
	friend frac operator*(const frac &a,const frac &b){
		return frac(a.x*b.x,a.y*b.y);
	}
	friend frac operator/(const frac &a,const frac &b){
		return frac(a.x*b.y,a.y*b.x);
	}
	friend ostream &operator<<(ostream &cout,const frac &f){
		return cout<<f.x<<'/'<<f.y;
	}
	bool operator<(const frac &b)const{
		return x*b.y<y*b.x;
	}
	bool operator==(const frac &b)const{
		return x*b.y==y*b.x;
	}
};
```

### 线性基求交

- 未测试，$O(\log^2 W)$

```c++
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

### Miller-Rabin and Pollard_rho 更新

```c++
bool mr(ll x,ll b){ //private
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
ll pollard_rho(ll x){ //private
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

## 图论

### Kruskal重构树

- 通过Kruskal算法构建的有根树
- 在原图中从某一点出发，只走边权不超过 $w$ 的边，可达的点集在重构树中是一棵子树，对应dfs序的一个区间
- 构建过程：边权从小到大访问边 $(x,y)$。如果 $x,y$ 不连通，新建点 $s$ 连接 $x,y$ 所在树的根，且 $s$ 为新树的根，$s$ 的点权为边 $(x,y)$ 的边权
- 查找 $x$ 能访问的点时，树上倍增找到最远的点权不大于 $w$ 的祖先
- 性质：二叉树，大根堆
- luogu P4197，编号从 $1$ 开始，$O(E\log E)$，**更新划分树板子**

```c++
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
struct divtree{
	int a[N],pos[25][N],tr[25][N],n;
	void build(int l,int r,int dep){ //private
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
	int query(int ql,int qr,int k,int L,int R,int dep=0){ //private
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
	int mink(int l,int r,int k){ //k>=1, k<=r-l+1
		return query(l,r,k,1,n);
	}
	int maxk(int l,int r,int k){ //k>=1, k<=r-l+1
		return query(l,r,r-l+2-k,1,n);
	}
	void init(int _n){
		n=_n;
		repeat(i,1,n+1)tr[0][i]=a[i]=h[rec[i]];
		sort(a+1,a+n+1);
		build(1,n,0);
	}
}tr;
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
	kru(); //get kruskal tree
	fa[s][0]=s; dfs(s); //get dfs order & fa[i][0]
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

### 树哈希 补充

- 无根树哈希可以找重心为根（重心最多只有两个）
- 一种自创哈希方式

```c++
vector<int> a[N];
pii H[N];
void dfs(int x,int fa){ //the answer is H[rt]
	H[x]=pii(1,1);
	for(auto p:a[x])if(p!=fa)dfs(p,x);
//	sort(a[x].begin(),a[x].end(),[](int x,int y){
//		return pii(H[x].fi^H[x].se,H[x].fi)
//			<  pii(H[y].fi^H[y].se,H[y].fi);
//	});
	repeat(i,0,a[x].size()){
		H[x].fi^=H[a[x][i]].fi+H[a[x][i]].se;
		H[x].se+=H[a[x][i]].fi^H[a[x][i]].se;
	}
}
```

### dsu重构树

- 离线处理连边和连通块询问
- 原理：dsu重构树的dfs序保证任意时刻的任意连通块是一个区间
- 接口：先读入ops，然后build(n),然后依次merge(x,y)（要保证顺序不变）。询问连通块区间query(x,l,r)，询问结点x在dfs序中的位置是 `red.l[x]`
- 编号从 $1$ 开始

```c++
struct dsu_rebuilder{
	int cnt,l[N],r[N];
	DSU d;
	vector<int> a[N];
	vector<pii> ops; //input
	void dfs(int x){ //private
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

### 虚树

- $O(k\log n)$ 处理出指定 $k$ 个点及其两两lca构成的树，原理是单调栈
- `pos[x]` 表示 dfs 序中 $x$ 的位置，`lab[x]` 表示 $x$ 是否为指定点
- `tr` 表示虚树，`v` 是指定点序列(input)
- 基于 lca，预处理为 `lca::init()` 以及 `pos[]`

```c++
vector<int> e[N],v; //v: input
int pos[N];
vector<int> stk,rec;
vector<pii> tr[N];
bool lab[N];
#define r stk.rbegin()
void add(){
	tr[r[1]].push_back({r[0],dep[r[0]]-dep[r[1]]}); //tr[x][i].second is the length of the edge
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
	//flag=true;
	lastdfs(1,-1);
	//if(flag)printf("%d\n",cost[1]); else puts("-1");
	for(auto i:rec){ //clear
		tr[i].clear();
		lab[i]=0; //cost[i]=0; up[i]=0;
	}
}
```

### 树上启发式合并

- 暴力方式处理子树问题
- 编号无限制，$O(n\log n)$

```c++
vector<int> e[N]; int n;
int sz[N],son[N],dep[N]; bool vis[N];
ll ans[N],sum[N]; int num[N],top,c[N]; //not fixed
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
//initdfs(s,-1); dfs(s,-1,1);
```

## 其他

### Java

```java
import java.util.*;
import java.math.BigInteger;
import java.math.BigDecimal;
public class Main{
	static Scanner sc=new Scanner(System.in);
	public static void main(String[] args){
	}
}
```

```java
import java.util.*;
import java.io.*;
import java.math.BigInteger;
import java.math.BigDecimal;
public class Main{
	static Scanner cin=new Scanner(System.in);
	static PrintStream cout=System.out;
	public static void main(String[] args){
	}
}
```

- 编译运行 `java Main.java`
- 编译 `javac Main.java` //生成Main.class
- 运行 `java Main`

数据类型

```java
int //4字节有符号
long //8字节有符号
double,boolean,char,String
```

```java
final double PI=3.14; //final类似c++的const
var n=1; //var类似c++的auto
long 型常量结尾加 L，如 1L
```

数组

```java
int[] arr=new int[100]; //数组（可以是变量）
int[][] arr=new int[10][10]; //二维数组
arr.length; //数组长度，没有括号
Arrays.binarySearch(arr,l,r,x) //在arr[[l,r-1]]中二分查找x，若存在返回位置，不存在返回-lowerbound-1
Arrays.sort(arr,l,r); Arrays.sort(arr); //对arr[[l,r-1]]排序
Arrays.fill(arr,l,r,x); Arrays.fill(arr,x); //填充arr[[l,r-1]]为x
```

极其麻烦的结构体compareTo重载

```java
public class Main{
public static class node implements Comparable<node>{
	int x;
	public node(){}
	public int compareTo(node b){
		return x-b.x;
	}
}
static Scanner sc;
public static void main(String[] args){
	sc=new Scanner(System.in);
	int n=sc.nextInt();
	node[] a=new node[n];
	for(int i=0;i<n;i++){
		a[i]=new node();
		a[i].x=sc.nextInt();
	}
	Arrays.sort(a);
	for(node i:a)
		System.out.print(i.x+" ");
	System.out.println();
}
}
```

输出

```java
System.out.print(x);
System.out.println();
System.out.println(x);
System.out.printf("%.2f\n",d); //格式化
```

输入

```java
import java.util.Scanner;
Scanner sc=new Scanner(System.in); //初始化
String s=sc.nextline(); //读一行字符串
int n=sc.nextInt(); //读整数
double d=sc.nextDouble(); //读实数
sc.hasNext() //是否读完
```

String

```java
s1.equals(s2) //返回是否相等
s1.compareTo(s2) //s1>s2返回1，s1<s2返回-1，s1==s2返回0
s1.contains(s2) //返回是否包含子串s2
s1.indexOf(s2,begin=0) //查找子串位置
s1.substring(l,r) //返回子串[l,r-1]
s1.charAt(x) //类似c++的s1[x]
s1.length() //返回长度
s1+s2 //返回连接结果
String.format("%d",n) //返回格式化结果
```

StringBuffer/StringBuilder

```java
StringBuffer s1=new StringBuffer();
StringBuffer s1=new StringBuffer("A");
s1.append("A"); //类似c++的s1+="A";
s1.reverse(); //反转字符串
s1.replace(l,r,"A"); //将子串[l,r-1]替换为"A"(delete+insert)
s1.charAt(x); //类似c++的s1[x]
s1.setCharAt(x,c); //类似c++的s1[x]=c;
```

Math

```java
//不用import就能用下列函数
Math.{sqrt,sin,atan,abs,max,min,pow,exp,log,PI,E}
```

Random

```java
import java.util.Random;
Random rnd=new Random(); //已经把时间戳作为了种子
rnd.nextInt();
rnd.nextInt(n); //[0,n)
```

BigInteger

```java
import java.math.BigInteger;
BigInteger n=new BigInteger("0");
sc.nextBigInteger() //括号里可以写进制
BigInteger[] arr=new BigInteger[10];
n1.intValue() //转换为int
n1.longValue() //转换
n1.doubleValue() //转换
n1.add(n2) //加法
n1.subtract(n2) //减法
n1.multiply(n2) //乘法
n1.divide(n2) //除法
n1.mod(n2) //取模
BigInteger.valueOf(I) //int转换为BigInteger
n1.compareTo(n2) //n1>n2返回1，n1<n2返回-1，n1==n2返回0
n1.abs()
n1.pow(I)
n1.toString(I) //返回I进制字符串
//运算时n2一定要转换成BigInteger
```

BigDecimal

```java
import java.math.BigDecimal;
n1.divide(n2,2,BigDecimal.ROUND_HALF_UP) //保留两位（四舍五入）
//貌似没有sqrt等操作，都得自己实现qwq
```

### python3

读入（EOF停止）

```python
def Solve():
	a,b=map(int,input().split())
	print(a+b)
while True:
	try:
		Solve()
	except EOFError:
		break
```

读入（T组数据）

```python
def Solve():
	a,b=map(int,input().split())
	print(a+b)
T=int(input())
for ca in range(1,T+1):
    Solve()
```

输出（不回车）

```python
print(x,end="")
```

常量

```python
None #空值
True False #布尔值
```

字符串str

```python
eval(s); #表达式求值
ord(c); chr(c); #字符和编码的转换
int("123"); str(123); #数字与字符串的转换
```

列表list

```python
a=[]; a.append(x); a.pop(); a[x]; #最后一个不存在会报错
len(a); #返回size
a.sort(); #排序
```

字典dict

```python
- a={}; a={x:y}; a[x]=y; a[x]; #最后一个不存在会报错
- a.get(x); #不存在返回None
- a.pop(x); #不存在会报错
```

集合set

```python
a=set(); a.add(x); a.remove(x); #最后一个不存在会报错
a&b; a|b; a-b; a^b; #集合的交、并、差、对称差
```

### 字符串哈希模板更新

- 编号从 $0$ 开始

```c++
template<typename string>
struct Hash{
	template<int b,int mod,int x=101>
	struct hs{
		vector<int> a,p;
		hs(const string &s=""){
			a={0},p={1};
			for(auto c:s){
				a.push_back((1ll*a.back()*b+(c^x))%mod);
				p.push_back(1ll*p.back()*b%mod);
			}
		}
		ll q(int l,int r){
			return (a[r+1]-1ll*a[l]*p[r-l+1]%mod+mod)%mod;
		}
		ll q2(int l,int r){
			if(l<=r)return q(l,r); 
			return (a[r+1]+q(l,a.size()-2)*p[r+1])%mod;
		}
	};
	hs<257,1000000007> h1;
	hs<257,2147483647> h2;
	Hash(const string &s):h1(s),h2(s){}
	pii query(int l,int r){
		return {h1.q(l,r),h2.q(l,r)};
	}
	pii query2(int l,int r){ //循环字符串
		return {h1.q2(l,r),h2.q2(l,r)};
	}
};
```

## 斯坦纳树 from OIwiki

### 例题

首先以一道模板题来带大家熟悉最小斯坦纳树问题。见 [【模板】最小斯坦纳树](https://www.luogu.com.cn/problem/P6192) 。

题意已经很明确了，给定连通图 $G$ 中的 $n$ 个点与 $k$ 个关键点，连接 $k$ 个关键点，使得生成树的所有边的权值和最小。

结合上面的知识我们可以知道直接连接这 $k$ 个关键点生成的权值和不一定是最小的，或者这 $k$ 个关键点不会直接（相邻）连接。所以应当使用剩下的 $n-k$ 个点。

我们使用状态压缩动态规划来求解。用 $f(i,S)$ 表示以 $i$ 为根的一棵树，包含集合 $S$ 中所有点的最小边权值和。

考虑状态转移：

- 首先对连通的子集进行转移， $f(i,S)\leftarrow \min(f(i,S),f(i,T)+f(i,S-T))$ 。

- 在当前的子集连通状态下进行边的松弛操作， $f(i,S)\leftarrow \min(f(i,S),f(j,S)+w(j,i))$ 。在下面的代码中用一个 `tree[tot]` 来记录两个相连节点 $i,j$ 的相关信息。


```c++
#include <bits/stdc++.h>

using namespace std;

const int maxn = 510;
const int INF = 0x3f3f3f3f;
typedef pair<int, int> P;
int n, m, k;

struct edge {
	int to, next, w;
} e[maxn << 1];

int head[maxn << 1], tree[maxn << 1], tot;
int dp[maxn][5000], vis[maxn];
int key[maxn];
priority_queue<P, vector<P>, greater<P> > q;

void add(int u, int v, int w) {
	e[++tot] = edge{v, head[u], w};
	head[u] = tot;
}

void dijkstra(int s) {
	memset(vis, 0, sizeof(vis));
	while (!q.empty()) {
		P item = q.top();
		q.pop();
		if (vis[item.second]) continue;
		vis[item.second] = 1;
		for (int i = head[item.second]; i; i = e[i].next) {
			if (dp[tree[i]][s] > dp[item.second][s] + e[i].w) {
				dp[tree[i]][s] = dp[item.second][s] + e[i].w;
				q.push(P(dp[tree[i]][s], tree[i]));
			}
		}
	}
}

int main() {
	memset(dp, INF, sizeof(dp));
	scanf("%d %d %d", &n, &m, &k);
	int u, v, w;
	for (int i = 1; i <= m; i++) {
		scanf("%d %d %d", &u, &v, &w);
		add(u, v, w);
		tree[tot] = v;
		add(v, u, w);
		tree[tot] = u;
	}
	for (int i = 1; i <= k; i++) {
		scanf("%d", &key[i]);
		dp[key[i]][1 << (i - 1)] = 0;
	}
	for (int s = 1; s < (1 << k); s++) {
		for (int i = 1; i <= n; i++) {
			for (int subs = s & (s - 1); subs; subs = s & (subs - 1))
				dp[i][s] = min(dp[i][s], dp[i][subs] + dp[i][s ^ subs]);
			if (dp[i][s] != INF) q.push(P(dp[i][s], i));
		}
		dijkstra(s);
	}
	printf("%d\n", dp[key[1]][(1 << k) - 1]);
	return 0;
}
```

另外一道经典例题 [\[WC2008\]游览计划](https://www.luogu.com.cn/problem/P4294) 。

这道题是求点权和最小的斯坦纳树，用 $f(i,S)$ 表示以 $i$ 为根的一棵树，包含集合 $S$ 中所有点的最小点权值和。 $a_i$ 表示点权。

考虑状态转移：

-	$f(i,S)\leftarrow \min(f(i,S),f(i,T)+f(i,S-T)-a_i)$ 。由于此处合并时同一个点 $a_i$ ，会被加两次，所以减去。
-	$f(i,S)\leftarrow \min(f(i,S),f(j,S)+w(j,i))$ 。

可以发现状态转移与上面的模板题是类似的，麻烦的是对答案的输出，在 DP 的过程中还要记录路径。

用 `pre[i][s]` 记录转移到 $i$ 为根，连通状态集合为 $s$ 时的点与集合的信息。在 DP 结束后从 `pre[root][S]` 出发，寻找与集合里的点相连的那些点并逐步分解集合 $S$ ，用 ans 数组来记录被使用的那些点，当集合分解完毕时搜索也就结束了。

```c++
#include <bits/stdc++.h>

using namespace std;

#define mp make_pair
typedef pair<int, int> P;
typedef pair<P, int> PP;
const int INF = 0x3f3f3f3f;
const int dx[] = {0, 0, -1, 1};
const int dy[] = {1, -1, 0, 0};
int n, m, K, root;
int f[101][1111], a[101], ans[11][11];
bool inq[101];
PP pre[101][1111];
queue<P> q;

bool legal(P u) {
	if (u.first >= 0 && u.second >= 0 && u.first < n && u.second < m) {
		return true;
	}
	return false;
}

int num(P u) { return u.first * m + u.second; }

void spfa(int s) {
	memset(inq, 0, sizeof(inq));
	while (!q.empty()) {
		P u = q.front();
		q.pop();
		inq[num(u)] = 0;
		for (int d = 0; d < 4; d++) {
			P v = mp(u.first + dx[d], u.second + dy[d]);
			int du = num(u), dv = num(v);
			if (legal(v) && f[dv][s] > f[du][s] + a[dv]) {
				f[dv][s] = f[du][s] + a[dv];
				if (!inq[dv]) {
					inq[dv] = 1;
					q.push(v);
				}
				pre[dv][s] = mp(u, s);
			}
		}
	}
}

void dfs(P u, int s) {
	if (!pre[num(u)][s].second) return;
	ans[u.first][u.second] = 1;
	int nu = num(u);
	if (pre[nu][s].first == u) dfs(u, s ^ pre[nu][s].second);
	dfs(pre[nu][s].first, pre[nu][s].second);
}

int main() {
	memset(f, INF, sizeof(f));
	scanf("%d %d", &n, &m);
	int tot = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			scanf("%d", &a[tot]);
			if (!a[tot]) {
				f[tot][1 << (K++)] = 0;
				root = tot;
			}
			tot++;
		}
	}
	for (int s = 1; s < (1 << K); s++) {
		for (int i = 0; i < n * m; i++) {
			for (int subs = s & (s - 1); subs; subs = s & (subs - 1)) {
				if (f[i][s] > f[i][subs] + f[i][s ^ subs] - a[i]) {
					f[i][s] = f[i][subs] + f[i][s ^ subs] - a[i];
					pre[i][s] = mp(mp(i / m, i % m), subs);
				}
			}
			if (f[i][s] < INF) q.push(mp(i / m, i % m));
		}
		spfa(s);
	}
	printf("%d\n", f[root][(1 << K) - 1]);
	dfs(mp(root / m, root % m), (1 << K) - 1);
	for (int i = 0, tot = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (!a[tot++])
				putchar('x');
			else
				putchar(ans[i][j] ? 'o' : '_');
		}
		if (i != n - 1) printf("\n");
	}
	return 0;
}
```

## 高精度计算 from OIwiki

### 封装类

 [这里](https://paste.ubuntu.com/p/7VKYzpC7dn/) 有一个封装好的高精度整数类。

```cpp
#define MAXN 9999
// MAXN 是一位中最大的数字
#define MAXSIZE 10024
// MAXSIZE 是位数
#define DLEN 4
// DLEN 记录压几位
struct Big {
	int a[MAXSIZE], len;
	bool flag;	// 标记符号'-'
	Big() {
		len = 1;
		memset(a, 0, sizeof a);
		flag = 0;
	}
	Big(const int);
	Big(const char*);
	Big(const Big&);
	Big& operator=(const Big&);	// 注意这里operator有&，因为赋值有修改……
	// 由于OI中要求效率
	// 此处不使用泛型函数
	// 故不重载
	// istream& operator>>(istream&,	BigNum&);	 // 重载输入运算符
	// ostream& operator<<(ostream&,	BigNum&);	 // 重载输出运算符
	Big operator+(const Big&) const;
	Big operator-(const Big&) const;
	Big operator*(const Big&)const;
	Big operator/(const int&) const;
	// TODO: Big / Big;
	Big operator^(const int&) const;
	// TODO: Big ^ Big;

	// TODO: Big 位运算;

	int operator%(const int&) const;
	// TODO: Big ^ Big;
	bool operator<(const Big&) const;
	bool operator<(const int& t) const;
	inline void print() const;
};
// README::不要随随便便把参数都变成引用，那样没办法传值
Big::Big(const int b) {
	int c, d = b;
	len = 0;
	// memset(a,0,sizeof a);
	CLR(a);
	while (d > MAXN) {
		c = d - (d / (MAXN + 1) * (MAXN + 1));
		d = d / (MAXN + 1);
		a[len++] = c;
	}
	a[len++] = d;
}
Big::Big(const char* s) {
	int t, k, index, l;
	CLR(a);
	l = strlen(s);
	len = l / DLEN;
	if (l % DLEN) ++len;
	index = 0;
	for (int i = l - 1; i >= 0; i -= DLEN) {
		t = 0;
		k = i - DLEN + 1;
		if (k < 0) k = 0;
		g(j, k, i) t = t * 10 + s[j] - '0';
		a[index++] = t;
	}
}
Big::Big(const Big& T) : len(T.len) {
	CLR(a);
	f(i, 0, len) a[i] = T.a[i];
	// TODO:重载此处？
}
Big& Big::operator=(const Big& T) {
	CLR(a);
	len = T.len;
	f(i, 0, len) a[i] = T.a[i];
	return *this;
}
Big Big::operator+(const Big& T) const {
	Big t(*this);
	int big = len;
	if (T.len > len) big = T.len;
	f(i, 0, big) {
		t.a[i] += T.a[i];
		if (t.a[i] > MAXN) {
			++t.a[i + 1];
			t.a[i] -= MAXN + 1;
		}
	}
	if (t.a[big])
		t.len = big + 1;
	else
		t.len = big;
	return t;
}
Big Big::operator-(const Big& T) const {
	int big;
	bool ctf;
	Big t1, t2;
	if (*this < T) {
		t1 = T;
		t2 = *this;
		ctf = 1;
	} else {
		t1 = *this;
		t2 = T;
		ctf = 0;
	}
	big = t1.len;
	int j = 0;
	f(i, 0, big) {
		if (t1.a[i] < t2.a[i]) {
			j = i + 1;
			while (t1.a[j] == 0) ++j;
			--t1.a[j--];
			// WTF?
			while (j > i) t1.a[j--] += MAXN;
			t1.a[i] += MAXN + 1 - t2.a[i];
		} else
			t1.a[i] -= t2.a[i];
	}
	t1.len = big;
	while (t1.len > 1 && t1.a[t1.len - 1] == 0) {
		--t1.len;
		--big;
	}
	if (ctf) t1.a[big - 1] = -t1.a[big - 1];
	return t1;
}
Big Big::operator*(const Big& T) const {
	Big res;
	int up;
	int te, tee;
	f(i, 0, len) {
		up = 0;
		f(j, 0, T.len) {
			te = a[i] * T.a[j] + res.a[i + j] + up;
			if (te > MAXN) {
				tee = te - te / (MAXN + 1) * (MAXN + 1);
				up = te / (MAXN + 1);
				res.a[i + j] = tee;
			} else {
				up = 0;
				res.a[i + j] = te;
			}
		}
		if (up) res.a[i + T.len] = up;
	}
	res.len = len + T.len;
	while (res.len > 1 && res.a[res.len - 1] == 0) --res.len;
	return res;
}
Big Big::operator/(const int& b) const {
	Big res;
	int down = 0;
	gd(i, len - 1, 0) {
		res.a[i] = (a[i] + down * (MAXN + 1) / b);
		down = a[i] + down * (MAXN + 1) - res.a[i] * b;
	}
	res.len = len;
	while (res.len > 1 && res.a[res.len - 1] == 0) --res.len;
	return res;
}
int Big::operator%(const int& b) const {
	int d = 0;
	gd(i, len - 1, 0) d = (d * (MAXN + 1) % b + a[i]) % b;
	return d;
}
Big Big::operator^(const int& n) const {
	Big t(n), res(1);
	int y = n;
	while (y) {
		if (y & 1) res = res * t;
		t = t * t;
		y >>= 1;
	}
	return res;
}
bool Big::operator<(const Big& T) const {
	int ln;
	if (len < T.len) return 233;
	if (len == T.len) {
		ln = len - 1;
		while (ln >= 0 && a[ln] == T.a[ln]) --ln;
		if (ln >= 0 && a[ln] < T.a[ln]) return 233;
		return 0;
	}
	return 0;
}
inline bool Big::operator<(const int& t) const {
	Big tee(t);
	return *this < tee;
}
inline void Big::print() const {
	printf("%d", a[len - 1]);
	gd(i, len - 2, 0) { printf("%04d", a[i]); }
}

inline void print(Big s) {
	// s不要是引用，要不然你怎么print(a * b);
	int len = s.len;
	printf("%d", s.a[len - 1]);
	gd(i, len - 2, 0) { printf("%04d", s.a[i]); }
}
char s[100024];
```

## 类欧几里得算法 from OIwiki

类欧几里德算法由洪华敦在 2016 年冬令营营员交流中提出的内容，其本质可以理解为，使用一个类似辗转相除法来做函数求和的过程。

### 引入

设

$$
f(a,b,c,n)=\sum_{i=0}^n\left\lfloor \frac{ai+b}{c} \right\rfloor
$$

其中 $a,b,c,n$ 是常数。需要一个 $O(\log n)$ 的算法。

这个式子和我们以前见过的式子都长得不太一样。带向下取整的式子容易让人想到数论分块，然而数论分块似乎不适用于这个求和。但是我们是可以做一些预处理的。

如果说 $a\ge c$ 或者 $b\ge c$ ，意味着可以将 $a,b$ 对 $c$ 取模以简化问题：

$$
\begin{array}{llll}
f(a,b,c,n)&=\sum_{i=0}^n\left\lfloor \frac{ai+b}{c} \right\rfloor\\
&=\sum_{i=0}^n\left\lfloor
\frac{\left(\left\lfloor\frac{a}{c}\right\rfloor c+a\bmod c\right)i+\left(\left\lfloor\frac{b}{c}\right\rfloor c+b\bmod c\right)}{c}\right\rfloor\\
&=\frac{n(n+1)}{2}\left\lfloor\frac{a}{c}\right\rfloor+(n+1)\left\lfloor\frac{b}{c}\right\rfloor+
\sum_{i=0}^n\left\lfloor\frac{\left(a\bmod c\right)i+\left(b\bmod c\right)}{c}
\right\rfloor\\
&=\frac{n(n+1)}{2}\left\lfloor\frac{a}{c}\right\rfloor
+(n+1)\left\lfloor\frac{b}{c}\right\rfloor+f(a\bmod c,b\bmod c,c,n)
\end{array}
$$

那么问题转化为了 $a<c,b<c$ 的情况。观察式子，你发现只有 $i$ 这一个变量。因此要推就只能从 $i$ 下手。在推求和式子中有一个常见的技巧，就是条件与贡献的放缩与转化。具体地说，在原式 $\displaystyle f(a,b,c,n)=\sum_{i=0}^n\left\lfloor \frac{ai+b}{c} \right\rfloor$ 中， $0\le i\le n$ 是条件，而 $\left\lfloor \dfrac{ai+b}{c} \right\rfloor$ 是对总和的贡献。

要加快一个和式的计算过程，所有的方法都可以归约为 **贡献合并计算** 。但你发现这个式子的贡献难以合并，怎么办？ **将贡献与条件做转化** 得到另一个形式的和式。具体地，我们直接把原式的贡献变成条件：

$$
\sum_{i=0}^n\left\lfloor \frac{ai+b}{c} \right\rfloor
=\sum_{i=0}^n\sum_{j=0}^{\left\lfloor \frac{ai+b}{c} \right\rfloor-1}1\\
$$

现在多了一个变量 $j$ ，既然算 $i$ 的贡献不方便，我们就想办法算 $j$ 的贡献。因此想办法搞一个和 $j$ 有关的贡献式。这里有另一个家喻户晓的变换方法，笔者概括为限制转移。具体来说，在上面的和式中 $n$ 限制 $i$ 的上界，而 $i$ 限制 $j$ 的上界。为了搞 $j$ ，就先把 j 放到贡献的式子里，于是我们交换一下 $i,j$ 的求和算子，强制用 $n$ 限制 $j$ 的上界。

$$
\begin{array}{llll}
&=\sum_{j=0}^{\left\lfloor \frac{an+b}{c} \right\rfloor-1}
\sum_{i=0}^n\left[j<\left\lfloor \frac{ai+b}{c} \right\rfloor\right]\\
\end{array}
$$

这样做的目的是让 $j$ 摆脱 $i$ 的限制，现在 $i,j$ 都被 $n$ 限制，而贡献式看上去是一个条件，但是我们仍把它叫作贡献式，再对贡献式做变换后就可以改变 $i,j$ 的限制关系。于是我们做一些放缩的处理。首先把向下取整的符号拿掉

$$
j<\left\lfloor \frac{ai+b}{c} \right\rfloor
\Leftrightarrow j+1\leq \left\lfloor \frac{ai+b}{c} \right\rfloor
\Leftrightarrow j+1\leq \frac{ai+b}{c}\\
$$

然后可以做一些变换

$$
j+1\leq \frac{ai+b}{c} \Leftrightarrow jc+c\le ai+b \Leftrightarrow jc+c-b-1< ai
$$

最后一步，向下取整得到：

$$
jc+c-b-1< ai\Leftrightarrow \left\lfloor\frac{jc+c-b-1}{a}\right\rfloor< i
$$

这一步的重要意义在于，我们可以把变量 $i$ 消掉了！具体地，令 $m=\left\lfloor \frac{an+b}{c} \right\rfloor$ ，那么原式化为

$$
\begin{array}{llll}
f(a,b,c,n)&=\sum_{j=0}^{m-1}
\sum_{i=0}^n\left[i>\left\lfloor\frac{jc+c-b-1}{a}\right\rfloor \right]\\
&=\sum_{j=0}^{m-1}
n-\left\lfloor\frac{jc+c-b-1}{a}\right\rfloor\\
&=nm-f\left(c,c-b-1,a,m-1\right)
\end{array}
$$

这是一个递归的式子。并且你发现 $a,c$ 分子分母换了位置，又可以重复上述过程。先取模，再递归。这就是一个辗转相除的过程，这也是类欧几里德算法的得名。

容易发现时间复杂度为 $O(\log n)$ 。

### 扩展

理解了最基础的类欧几里德算法，我们再来思考以下两个变种求和式：

$$
g(a,b,c,n)=\sum_{i=0}^ni\left\lfloor \frac{ai+b}{c} \right\rfloor\\
h(a,b,c,n)=\sum_{i=0}^n\left\lfloor \frac{ai+b}{c} \right\rfloor^2
$$

#### 推导 g

我们先考虑 $g$ ，类似地，首先取模：

$$
g(a,b,c,n)
=g(a\bmod c,b\bmod c,c,n)+\left\lfloor\frac{a}{c}\right\rfloor\frac{n(n+1)(2n+1)}{6}+\left\lfloor\frac{b}{c}\right\rfloor\frac{n(n+1)}{2}
$$

接下来考虑 $a<c,b<c$ 的情况，令 $m=\left\lfloor\frac{an+b}{c}\right\rfloor$ 。之后的过程我会写得很简略，因为方法和上文略同：

$$
\begin{array}{llll}
&g(a,b,c,n)=\sum_{i=0}^ni\left\lfloor \frac{ai+b}{c} \right\rfloor\\
&=\sum_{j=0}^{m-1}
\sum_{i=0}^n\left[j<\left\lfloor\frac{ai+b}{c}\right\rfloor\right]\cdot i
\end{array}
$$

这时我们设 $t=\left\lfloor\frac{jc+c-b-1}{a}\right\rfloor$ ，可以得到

$$
\begin{array}{llll}
&=\sum_{j=0}^{m-1}\sum_{i=0}^n[i>t]\cdot i\\
&=\sum_{j=0}^{m-1}\frac{1}{2}(t+n+1)(n-t)\\
&=\frac{1}{2}\left[mn(n+1)-\sum_{j=0}^{m-1}t^2-\sum_{j=0}^{m-1}t\right]\\
&=\frac{1}{2}[mn(n+1)-h(c,c-b-1,a,m-1)-f(c,c-b-1,a,m-1)]
\end{array}
$$

#### 推导 h

同样的，首先取模：

$$
\begin{array}{llll}
h(a,b,c,n)&=h(a\bmod c,b\bmod c,c,n)\\
&+2\left\lfloor\frac{b}{c}\right\rfloor f(a\bmod c,b\bmod c,c,n)
+2\left\lfloor\frac{a}{c}\right\rfloor g(a\bmod c,b\bmod c,c,n)\\
&+\left\lfloor\frac{a}{c}\right\rfloor^2\frac{n(n+1)(2n+1)}{6}+\left\lfloor\frac{b}{c}\right\rfloor^2(n+1)
+\left\lfloor\frac{a}{c}\right\rfloor\left\lfloor\frac{b}{c}\right\rfloor n(n+1)
\end{array}
$$

考虑 $a<c,b<c$ 的情况， $m=\left\lfloor\dfrac{an+b}{c}\right\rfloor, t=\left\lfloor\dfrac{jc+c-b-1}{a}\right\rfloor$ .

我们发现这个平方不太好处理，于是可以这样把它拆成两部分：

$$
n^2=2\dfrac{n(n+1)}{2}-n=\left(2\sum_{i=0}^ni\right)-n
$$

这样做的意义在于，添加变量 $j$ 的时侯就只会变成一个求和算子，不会出现 $\sum\times \sum$ 的形式：

$$
\begin{array}{llll}
&h(a,b,c,n)=\sum_{i=0}^n\left\lfloor \frac{ai+b}{c} \right\rfloor^2
=\sum_{i=0}^n\left[\left(2\sum_{j=1}^{\left\lfloor \frac{ai+b}{c} \right\rfloor}j \right)-\left\lfloor\frac{ai+b}{c}\right\rfloor\right]\\
=&\left(2\sum_{i=0}^n\sum_{j=1}^{\left\lfloor \frac{ai+b}{c} \right\rfloor}j\right) -f(a,b,c,n)\\
\end{array}
$$

接下来考虑化简前一部分：

$$
\begin{array}{llll}
&\sum_{i=0}^n\sum_{j=1}^{\left\lfloor \frac{ai+b}{c} \right\rfloor}j\\
=&\sum_{i=0}^n\sum_{j=0}^{\left\lfloor \frac{ai+b}{c} \right\rfloor-1}(j+1)\\
=&\sum_{j=0}^{m-1}(j+1)
\sum_{i=0}^n\left[j<\left\lfloor \frac{ai+b}{c} \right\rfloor\right]\\
=&\sum_{j=0}^{m-1}(j+1)\sum_{i=0}^n[i>t]\\
=&\sum_{j=0}^{m-1}(j+1)(n-t)\\
=&\frac{1}{2}nm(m+1)-\sum_{j=0}^{m-1}(j+1)\left\lfloor \frac{jc+c-b-1}{a} \right\rfloor\\
=&\frac{1}{2}nm(m+1)-g(c,c-b-1,a,m-1)-f(c,c-b-1,a,m-1)
\end{array}
$$

因此

$$
h(a,b,c,n)=nm(m+1)-2g(c,c-b-1,a,m-1)\\-2f(c,c-b-1,a,m-1)-f(a,b,c,n)
$$

在代码实现的时侯，因为 $3$ 个函数各有交错递归，因此可以考虑三个一起整体递归，同步计算，否则有很多项会被多次计算。这样实现的复杂度是 $O(\log n)$ 的。

```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int P = 998244353;
int i2 = 499122177, i6 = 166374059;
struct data {
	data() { f = g = h = 0; }
	int f, g, h;
};	// 三个函数打包
data calc(int n, int a, int b, int c) {
	int ac = a / c, bc = b / c, m = (a * n + b) / c, n1 = n + 1, n21 = n * 2 + 1;
	data d;
	if (a == 0) {	// 迭代到最底层
		d.f = bc * n1 % P;
		d.g = bc * n % P * n1 % P * i2 % P;
		d.h = bc * bc % P * n1 % P;
		return d;
	}
	if (a >= c || b >= c) {	// 取模
		d.f = n * n1 % P * i2 % P * ac % P + bc * n1 % P;
		d.g = ac * n % P * n1 % P * n21 % P * i6 % P + bc * n % P * n1 % P * i2 % P;
		d.h = ac * ac % P * n % P * n1 % P * n21 % P * i6 % P +
					bc * bc % P * n1 % P + ac * bc % P * n % P * n1 % P;
		d.f %= P, d.g %= P, d.h %= P;

		data e = calc(n, a % c, b % c, c);	// 迭代

		d.h += e.h + 2 * bc % P * e.f % P + 2 * ac % P * e.g % P;
		d.g += e.g, d.f += e.f;
		d.f %= P, d.g %= P, d.h %= P;
		return d;
	}
	data e = calc(m - 1, c, c - b - 1, a);
	d.f = n * m % P - e.f, d.f = (d.f % P + P) % P;
	d.g = m * n % P * n1 % P - e.h - e.f, d.g = (d.g * i2 % P + P) % P;
	d.h = n * m % P * (m + 1) % P - 2 * e.g - 2 * e.f - d.f;
	d.h = (d.h % P + P) % P;
	return d;
}
int T, n, a, b, c;
signed main() {
	scanf("%lld", &T);
	while (T--) {
		scanf("%lld%lld%lld%lld", &n, &a, &b, &c);
		data ans = calc(n, a, b, c);
		printf("%lld %lld %lld\n", ans.f, ans.h, ans.g);
	}
	return 0;
}
```

## 排列组合 from OIwiki

### 组合数性质

由于组合数在 OI 中十分重要，因此在此介绍一些组合数的性质。

$$
\binom{n}{m}=\binom{n}{n-m}\tag{1}
$$

相当于将选出的集合对全集取补集，故数值不变。（对称性）

$$
\binom{n}{k} = \frac{n}{k} \binom{n-1}{k-1}\tag{2}
$$

由定义导出的递推式。

$$
\binom{n}{m}=\binom{n-1}{m}+\binom{n-1}{m-1}\tag{3}
$$

组合数的递推式（杨辉三角的公式表达）。我们可以利用这个式子，在 $O(n^2)$ 的复杂度下推导组合数。

$$
\binom{n}{0}+\binom{n}{1}+\cdots+\binom{n}{n}=\sum_{i=0}^n\binom{n}{i}=2^n\tag{4}
$$

这是二项式定理的特殊情况。取 $a=b=1$ 就得到上式。

$$
\sum_{i=0}^n(-1)^i\binom{n}{i}=0\tag{5}
$$

二项式定理的另一种特殊情况，可取 $a=1, b=-1$ 。

$$
\sum_{i=0}^m \binom{n}{i}\binom{m}{m-i} = \binom{m+n}{m}\ \ \ (n \geq m)\tag{6}
$$

拆组合数的式子，在处理某些数据结构题时会用到。

$$
\sum_{i=0}^n\binom{n}{i}^2=\binom{2n}{n}\tag{7}
$$

这是 $(6)$ 的特殊情况，取 $n=m$ 即可。

$$
\sum_{i=0}^ni\binom{n}{i}=n2^{n-1}\tag{8}
$$

带权和的一个式子，通过对 $(3)$ 对应的多项式函数求导可以得证。

$$
\sum_{i=0}^ni^2\binom{n}{i}=n(n+1)2^{n-2}\tag{9}
$$

与上式类似，可以通过对多项式函数求导证明。

$$
\sum_{l=0}^n\binom{l}{k} = \binom{n+1}{k+1}\tag{10}
$$

可以通过组合意义证明，在恒等式证明中较常用。

$$
\binom{n}{r}\binom{r}{k} = \binom{n}{k}\binom{n-k}{r-k}\tag{11}
$$

通过定义可以证明。

$$
\sum_{i=0}^n\binom{n-i}{i}=F_{n+1}\tag{12}
$$

其中 $F$ 是斐波那契数列。

$$
\sum_{l=0}^n \binom{l}{k} = \binom{n+1}{k+1}\tag{13}
$$

通过组合分析——考虑 $S={a_1, a_2, \cdots, a_{n+1}}$ 的 $k+1$ 子集数可以得证。
