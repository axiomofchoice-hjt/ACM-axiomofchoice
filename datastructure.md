<!-- TOC -->

- [数据结构](#数据结构)
	- [st表](#st表)
		- [<补充>猫树](#补充猫树)
	- [单调队列](#单调队列)
	- [树状数组](#树状数组)
	- [线段树](#线段树)
		- [<补充>高拓展性线段树](#补充高拓展性线段树)
		- [<补充>权值线段树（动态开点 线段树合并 线段树分裂）](#补充权值线段树动态开点-线段树合并-线段树分裂)
		- [<补充>zkw线段树](#补充zkw线段树)
		- [<补充>可持久化数组](#补充可持久化数组)
		- [<补充>李超线段树](#补充李超线段树)
	- [并查集](#并查集)
		- [<补充>种类并查集](#补充种类并查集)
		- [<补充>可持久化并查集](#补充可持久化并查集)
	- [左偏树](#左偏树)
	- [珂朵莉树×老司机树](#珂朵莉树×老司机树)
	- [K-D tree](#k-d-tree)
	- [划分树](#划分树)
	- [莫队](#莫队)
		- [普通莫队](#普通莫队)
		- [带修莫队](#带修莫队)
		- [回滚莫队](#回滚莫队)
	- [二叉搜索树](#二叉搜索树)
		- [不平衡的二叉搜索树](#不平衡的二叉搜索树)
		- [无旋treap](#无旋treap)
		- [<补充> 可持久化treap](#补充-可持久化treap)
	- [一些建议](#一些建议)

<!-- /TOC -->

# 数据结构

## st表

<H3>普通st表</H3>

- 编号从 $0$ 开始，初始化 $O(n\log n)$ 查询 $O(1)$

```c++
struct ST{
	#define logN 21
	#define U(x,y) max(x,y)
	ll a[N][logN];
	void init(int n){
		repeat(i,0,n)
			a[i][0]=in[i];
		repeat(k,1,logN)
		repeat(i,0,n-(1<<k)+1)
			a[i][k]=U(a[i][k-1],a[i+(1<<(k-1))][k-1]);
	}
	ll query(int l,int r){
		int s=31-__builtin_clz(r-l+1);
		return U(a[l][s],a[r-(1<<s)+1][s]);
	}
}st;
```

<H3>二维st表</H3>

- 编号从 $0$ 开始，初始化 $O(nm\log n\log m)$ 查询 $O(1)$

```c++
struct ST{ //注意logN=log(N)+2
	#define logN 9
	#define U(x,y) max(x,y)
	int f[N][N][logN][logN],log[N];
	ST(){
		log[1]=0;
		repeat(i,2,N)
			log[i]=log[i/2]+1;
	}
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
		int k=log[x1-x0+1],l=log[y1-y0+1];
		return U(U(U(
			f[x0][y0][k][l],
			f[x1-(1<<k)+1][y0][k][l]),
			f[x0][y1-(1<<l)+1][k][l]),
			f[x1-(1<<k)+1][y1-(1<<l)+1][k][l]);
	}
}st;
```

### <补充>猫树

- 编号从 $0$ 开始，初始化 $O(n\log n)$ 查询 $O(1)$

```c++
struct cat{
	#define U(a,b) max(a,b) //查询操作
	#define a0 0 //查询操作的零元
	#define logN 21
	vector<ll> a[logN];
	vector<ll> v;
	void init(){
		repeat(i,0,logN)a[i].clear();
		v.clear();
	}
	void push(ll in){
		v.push_back(in);
		int n=v.size()-1;
		repeat(s,1,logN){
			int len=1<<s; int l=n/len*len;
			if(n%len==len/2-1){
				repeat(i,0,len)a[s].push_back(a0);
				repeat_back(i,0,len/2)a[s][l+i]=U(a[s][l+i+1],v[l+i]);
			}
			if(n%len>=len/2)
				a[s][n]=(U(a[s][n-1],v[n]));
		}
	}
	ll query(int l,int r){ //区间查询
		if(l==r)return v[l];
		int s=32-__builtin_clz(l^r);
		return U(a[s][l],a[s][r]);
	}
}tr;
```

## 单调队列

- 求所有长度为k的区间中的最大值，线性复杂度

```c++
struct MQ{ //查询就用mq.q.front().first
	deque<pii> q; //first:保存的最大值; second:时间戳
	void init(){q.clear();}
	void push(int x,int k){
		static int T=0; T++;
		while(!q.empty() && q.back().fi<=x) //max
			q.pop_back();
		q.push_back({x,T});
		while(!q.empty() && q.front().se<=T-k)
			q.pop_front();
	}
	void work(function<int&(int)> a,int n,int k){ //原地保存，编号从0开始
		init();
		repeat(i,0,n){
			push(a(i),k);
			if(i+1>=k)a(i+1-k)=q.front().fi;
		}
	}
	void work(int a[][N],int n,int m,int k){ //原地保存，编号从0开始
		repeat(i,0,n){
			init();
			repeat(j,0,m){
				push(a[i][j],k);
				if(j+1>=k)a[i][j+1-k]=q.front().fi;
			}
		}
		m-=k-1;
		repeat(j,0,m){
			init();
			repeat(i,0,n){
				push(a[i][j],k);
				if(i+1>=k)a[i+1-k][j]=q.front().fi;
			}
		}
	}
}mq;
//求n*m矩阵中所有k*k连续子矩阵最大值之和 //编号从1开始
repeat(i,1,n+1)
	mq.work([&](int x)->int&{return a[i][x+1];},m,k);
repeat(j,1,m-k+2)
	mq.work([&](int x)->int&{return a[x+1][j];},n,k);
ll ans=0; repeat(i,1,n-k+2)repeat(j,1,m-k+2)ans+=a[i][j];
//或者
mq.work((int(*)[N])&(a[1][1]),n,m,k);
```

## 树状数组

<H3>普通树状数组</H3>

- 单点+区间，修改查询 $O(\log n)$

```c++
#define lb(x) (x&(-x))
struct BIT{
	ll t[N]; //一倍内存吧
	void init(int n){
		fill(t,t+n+1,0);
	}
	void add(int x,ll k){ //位置x加上k
		//x++;
		for(;x<N;x+=lb(x))
			t[x]+=k;
	}
	ll sum(int x){ //求[1,x]的和 //[0,x]
		//x++;
		ll ans=0;
		for(;x!=0;x-=lb(x))
			ans+=t[x];
		return ans;
	}
}bit;
```

- 大佬的第 $k$ 小（权值树状数组）

```c++
int findkth(int k){
	int ans=0,cnt=0;
	for (int i=20;i>=0;--i){
		ans+=1<<i;
		if (ans>=n || cnt+t[ans]>=k)ans-=1<<i;
		else cnt+=t[ans];
	}
	return ans+1;
}
```

<H3>超级树状数组</H3>

- 基于树状数组，基本只允许加法，区间+区间，$O(\log n)$

```c++
struct SPBIT{
	BIT a,a1;
	void init(){a.init();a1.init();}
	void add(ll x,ll y,ll k){
		a.add(x,k);
		a.add(y+1,-k);
		a1.add(x,k*(x-1));
		a1.add(y+1,-k*y);
	}
	ll sum(ll x,ll y){
		return y*a.sum(y)-(x-1)*a.sum(x-1)-(a1.sum(y)-a1.sum(x-1));
	}
}spbit;
```

<H3>二维超级树状数组</H3>

- 修改查询 $O(\log n\cdot\log m)$

```c++
int n,m;
#define lb(x) (x&(-x))
struct BIT{
	ll t[N][N]; //一倍内存吧
	void init(){
		mst(t,0);
	}
	void add(int x,int y,ll k){ //位置(x,y)加上k
		//x++,y++; //如果要从0开始编号
		for(int i=x;i<=n;i+=lb(i))
		for(int j=y;j<=m;j+=lb(j))
			t[i][j]+=k;
	}
	ll sum(int x,int y){ //求(1..x,1..y)的和
		//x++,y++; //如果要从0开始编号
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
	void add(int x0,int y0,int x1,int y1,int k){ //区间修改
		add(x0,y0,k);
		add(x0,y1+1,-k);
		add(x1+1,y0,-k);
		add(x1+1,y1+1,k);
	}
	ll sum(int x0,int y0,int x1,int y1){ //区间查询
		return sum(x1,y1)
			-sum(x0-1,y1)
			-sum(x1,y0-1)
			+sum(x0-1,y0-1);
	}
}spbit;
```

## 线段树

- 基本上适用于所有（线段树能实现的）区间+区间
- 我删了修改运算的零元，加了偷懒状态(state)，~~终于能支持赋值操作.jpg~~

```c++
struct seg{ //初始化init()修改查询tr->sth()
	#define U(x,y) (x+y) //查询运算
	#define a0 0 //查询运算的零元
	void toz(ll x){z+=x,state=1;} //加载到懒标记
	void toa(){a+=z*(r-l+1),z=0,state=0;} //懒标记加载到数据（z别忘了清空）
	ll a,z; bool state; //数据，懒标记，是否偷了懒
	int l,r; seg *lc,*rc;
	void init(int,int);
	void up(){a=U(lc->a,rc->a);}
	void down(){
		if(!state)return;
		if(l<r){lc->toz(z); rc->toz(z);}
		toa();
	}
	void update(int x,int y,ll k){
		if(x>r || y<l){down(); return;}
		if(x<=l && y>=r){toz(k); down(); return;}
		down();
		lc->update(x,y,k);
		rc->update(x,y,k);
		up();
	}
	ll query(int x,int y){
		if(x>r || y<l)return a0;
		down();
		if(x<=l && y>=r)return a;
		return U(lc->query(x,y),rc->query(x,y));
	}
}tr[N*2],*pl;
void seg::init(int _l,int _r){
	l=_l,r=_r; state=0; z=0; //z一定要清空啊啊
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

### <补充>高拓展性线段树

- 例：luogu P3373 线段树2，支持区间加、区间乘、区间和查询

```c++
struct Z{
	int x,y; explicit Z(int x=1,int y=0):x(x),y(y){}
	void push(Z b,int l,int r){
		x=(x*b.x)%mod;
		y=(y*b.x+b.y)%mod;
	}
};
struct A{
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

### <补充>权值线段树（动态开点 线段树合并 线段树分裂）

- 初始 $n$ 个线段树，支持对某个线段树插入权值、合并两个线段树、查询某个线段树第k小数
- 编号从 $1$ 开始，$O(n\log n)$

```c++
DSU d;
struct seg{
	seg *lc,*rc; int sz;
}tr[N<<5],*pl,*rt[N];
#define LL lc,l,m
#define RR rc,m+1,r
int size(seg *s){return s?s->sz:0;}
seg *newnode(){*pl=seg(); return pl++;}
void up(seg *s){s->sz=size(s->lc)+size(s->rc);}
void insert(seg *&s,int l,int r,int v,int num=1){ //insert v, (s=rt[d[x]])
	if(!s)s=newnode(); s->sz+=num;
	if(l==r)return;
	int m=(l+r)/2;
	if(v<=m)insert(s->LL,v,num);
	else insert(s->RR,v,num);
}
seg *merge(seg *a,seg *b,int l,int r){ //private, return the merged tree
	if(!a)return b; if(!b)return a;
	a->sz+=b->sz;
	if(l==r)return a;
	int m=(l+r)/2;
	a->lc=merge(a->lc,b->LL);
	a->rc=merge(a->rc,b->RR);
	return a;
}
void merge(int x,int y,int l,int r){ //merge tree x and y
	if(d[x]==d[y])return;
	rt[d[x]]=merge(rt[d[x]],rt[d[y]],l,r);
	d[y]=d[x];
}
int kth(seg *s,int l,int r,int k){ //kth in s, (k=1,2,...,sz, s=rt[d[x]])
	if(l==r)return l;
	int m=(l+r)/2,lv=size(s->lc);
	if(k<=lv)return kth(s->LL,k);
	else return kth(s->RR,k-lv);
}
int query(seg *s,int l,int r,int x,int y){ //count the numbers between [x,y] (s=rt[d[x]])
	if(!s || x>r || y<l)return 0;
	if(x<=l && y>=r)return s->sz;
	int m=(l+r)/2;
	return query(s->LL,x,y)+query(s->RR,x,y);
}
void split(seg *&s,int l,int r,int x,int y,seg *&t){ //the numbers between [x,y] trans from s to t, (s=rt[d[x]], t=rt[d[y]])
	if(!s || x>r || y<l)return;
	if(x<=l && y>=r){t=merge(s,t,l,r); s=0; return;}
	if(!t)t=newnode();
	int m=(l+r)/2;
	split(s->LL,x,y,t->lc);
	split(s->RR,x,y,t->rc);
	up(s); up(t);
}
void init(int n){ //create n trees
	pl=tr; d.init(n);
	fill(rt,rt+n+1,nullptr);
}
```

### <补充>zkw线段树

- 单点+区间，编号从0开始，建树 $O(n)$ 修改查询 $O(\log n)$
- 代码量和常数都和树状数组差不多

```c++
struct seg{
	#define U(a,b) max(a,b) //查询操作
	const ll a0=0; //查询操作的零元
	int n; ll a[1024*1024*4*2]; //内存等于2^k且大于等于两倍inn
	void init(int inn){ //建树
		for(n=1;n<inn;n<<=1);
		repeat(i,0,inn)a[n+i]=in[i];
		repeat(i,inn,n)a[n+i]=a0;
		repeat_back(i,1,n)up(i);
	}
	void up(int x){
		a[x]=U(a[x<<1],a[(x<<1)^1]);
	}
	void update(int x,ll k){ //位置x加上k
		a[x+=n]+=k; //也可以赋值等操作
		while(x>>=1)up(x);
	}
	ll query(int l,int r){ //区间查询
		ll ans=a0;
		for(l+=n-1,r+=n+1;l^r^1;l>>=1,r>>=1){
			if(~l & 1)ans=U(ans,a[l^1]); //l^1其实是l+1
			if(r & 1)ans=U(ans,a[r^1]); //r^1其实是r-1
		}
		return ans;
	}
}tr;
```

### <补充>可持久化数组

- 单点修改并创建新版本：`h[top]=update(h[i],x,v);`（每次 $O(\log n)$ 额外内存）
- 单点查询 `h[i]->query(x);`
- 初始化 $O(n)$，修改查询 $O(\log n)$

```c++
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
}pool[N*20],*h[N]; //h: versions
void init(int l,int r){
	segl=l,segr=r;
	pl=pool; pl->init(l,r); h[0]=pl;
}
```

### <补充>李超线段树

- 支持插入线段、查询所有线段与 $x=x_0$ 交点最高的那条线段
- 修改 $O(\log^2n)$，查询 $O(\log n)$

```c++
int funx; //这是y()的参数
struct func{
	lf k,b; int id;
	lf y()const{return k*funx+b;} //funx点处的高度
	bool operator<(const func &b)const{
		return make_pair(y(),-id)<make_pair(b.y(),-b.id);
	}
};
struct seg{ //初始化init()更新update()查询query()，func::y()是高度
	func a;
	int l,r;
	seg *ch[2];
	void init(int,int);
	void push(func d){
		funx=(l+r)/2;
		if(a<d)swap(a,d); //这个小于要用funx
		if(l==r)return;
		ch[d.k>a.k]->push(d);
	}
	void update(int x,int y,const func &d){ //更新[x,y]区间
		x=max(x,l); y=min(y,r); if(x>y)return;
		if(x==l && y==r)push(d);
		else{
			ch[0]->update(x,y,d);
			ch[1]->update(x,y,d);
		}
	}
	const func &query(int x){ //询问
		funx=x;
		if(l==r)return a;
		const func &b=ch[(l+r)/2<x]->query(x);
		return max(a,b); //这个max要用funx
	}
}tr[N*2],*pl;
void seg::init(int _l,int _r){
	l=_l,r=_r; a={0,-inf,-1}; //可能随题意改变
	if(l==r)return;
	int m=(l+r)/2;
	(ch[0]=++pl)->init(l,m);
	(ch[1]=++pl)->init(m+1,r);
}
void init(int l,int r){
	pl=tr;
	tr->init(l,r);
}
void add(int x0,int y0,int x1,int y1){ //线段处理并更新
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

## 并查集

- 合并查找 $O(α(n))$，可视为 $O(1)$

<H3>普通并查集</H3>

- 精简版，只有路径压缩

```c++
struct DSU{ //合并：d[x]=d[y]，查找：d[x]==d[y]
	int a[N];
	void init(int n){iota(a,a+n+1,0);}
	int fa(int x){return a[x]==x?x:a[x]=fa(a[x]);}
	int &operator[](int x){return a[fa(x)];}
}d;
```

- 普通版，路径压缩+启发式合并

```c++
struct DSU{
	int a[N],sz[N];
	void init(int n){
		iota(a,a+n+1,0);
		fill(sz,sz+n+1,1);
	}
	int fa(int x){
		return a[x]==x?x:a[x]=fa(a[x]);
	}
	bool query(int x,int y){ //查找
		return fa(x)==fa(y);
	}
	void join(int x,int y){ //合并
		x=fa(x),y=fa(y);
		if(x==y)return;
		if(sz[x]>sz[y])swap(x,y);
		a[x]=y; sz[y]+=sz[x];
	}
	int operator[](int x){return fa(x);}
}d;
```

### <补充>种类并查集

```c++
struct DSU{
	int a[N],r[N];
	void init(int n){
		repeat(i,0,n+1)a[i]=i,r[i]=0;
	}
	int plus(int a,int b){ //关系a+关系b，类似向量相加
		if(a==b)return -a;
		return a+b;
	}
	int inv(int a){ //关系a的逆
		return -a;
	}
	int fa(int x){ //返回根结点
		if(a[x]==x)return x;
		int f=a[x],ff=fa(f);
		r[x]=plus(r[x],r[f]);
		return a[x]=ff;
	}
	bool query(int x,int y){ //是否存在关系
		return fa(x)==fa(y);
	}
	int R(int x,int y){ //查找关系
		return plus(r[x],inv(r[y]));
	}
	void join(int x,int y,int r2){ //按r2关系合并
		r2=plus(R(y,x),r2);
		x=fa(x),y=fa(y);
		a[x]=y,r[x]=r2;
	}
}d;
```

### <补充>可持久化并查集

- 启发式合并，不能路径压缩，$O(\log^2 n)$

```c++
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
void join(seg *&a,seg *&sz,int x,int y){ //a=h[i].fi,sz=h[i].se
	x=fa(a,x); y=fa(a,y);
	if(x!=y){
		int sx=sz->query(x),sy=sz->query(y);
		if(sx<sy)swap(x,y),swap(sx,sy);
		a=update(a,y,x),sz=update(sz,x,sx+sy);
	}
}
```

## 左偏树

- 万年不用，$O(\log n)$
- ~~如果没有特殊要求一律平板电视~~

```c++
struct leftist{ //编号从1开始，因为空的左右儿子会指向0
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
		if(val[x]<val[y]) //大根堆
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
		dsu[x]=merge(lc,rc); //指向x的dsu也能正确指向top
	}
	#undef lc
	#undef rc
}lt;
//添加元素lt.add(v),位置是lt.val.size()-1
//是否未被pop：lt.exist(x)
//合并：lt.join(x,y)
//堆顶：lt.val[lt.top(x)]
//弹出：lt.pop(x)
```

```c++
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

## 珂朵莉树×老司机树

- 珂朵莉数以区间形式存储数据，~~非常暴力~~，适用于有区间赋值操作的题
- 均摊 $O(n\log\log n)$，但是~~很~~可能被卡

```c++
struct ODT{
	struct node{
		int l,r;
		mutable int v; //强制可修改
		bool operator<(const node &b)const{return l<b.l;}
	};
	set<node> a;
	void init(){
		a.clear();
		a.insert({-inf,inf,0});
	}
	set<node>::iterator split(int x){ //分裂区间
		auto it=--a.upper_bound({x,0,0});
		if(it->l==x)return it;
		int l=it->l,r=it->r,v=it->v;
		a.erase(it);
		a.insert({l,x-1,v});
		return a.insert({x,r,v}).first;
	}
	void assign(int l,int r,int v){ //区间赋值
		auto y=split(r+1),x=split(l);
		a.erase(x,y);
		a.insert({l,r,v});
	}
	int sum(int l,int r){ //操作示例：区间求和
		auto y=split(r+1),x=split(l);
		int ans=0;
		for(auto i=x;i!=y;i++){
			ans+=(i->r-i->l+1)*i->v;
		}
		return ans;
	}
}odt;
```

## K-D tree

- 例题：luogu P4148
- 支持在线在(x,y)处插入值、查询二维区间和
- 插入、查询 $O(\log n)$

```c++
struct node{
	int x,y,v;
}s[N];
bool cmp1(int a,int b){return s[a].x<s[b].x;}
bool cmp2(int a,int b){return s[a].y<s[b].y;}
struct kdtree{
	int rt,cur; //rt根结点
	int d[N],sz[N],lc[N],rc[N]; //d=1竖着砍，sz子树大小
	int L[N],R[N],D[N],U[N]; //该子树的界线
	int sum[N]; //维护的二维区间信息（二维区间和）
	int g[N],gt;
	void up(int x){ //更新信息
		sz[x]=sz[lc[x]]+sz[rc[x]]+1;
		sum[x]=sum[lc[x]]+sum[rc[x]]+s[x].v;
		L[x]=R[x]=s[x].x;
		D[x]=U[x]=s[x].y;
		if(lc[x]){
			L[x]=min(L[x],L[lc[x]]);
			R[x]=max(R[x],R[lc[x]]);
			D[x]=min(D[x],D[lc[x]]);
			U[x]=max(U[x],U[lc[x]]);
		}
		if(rc[x]){
			L[x]=min(L[x],L[rc[x]]);
			R[x]=max(R[x],R[rc[x]]);
			D[x]=min(D[x],D[rc[x]]);
			U[x]=max(U[x],U[rc[x]]);
		}
	}
	int build(int l,int r){ //以序列g[l..r]为模板重建树，返回根结点
		if(l>r)return 0;
		int mid=(l+r)>>1;
		lf ax=0,ay=0,sx=0,sy=0;
		for(int i=l;i<=r;i++)ax+=s[g[i]].x,ay+=s[g[i]].y;
		ax/=(r-l+1);
		ay/=(r-l+1);
		for(int i=l;i<=r;i++){
			sx+=(ax-s[g[i]].x)*(ax-s[g[i]].x);
			sy+=(ay-s[g[i]].y)*(ay-s[g[i]].y);
		}
		if(sx>sy)
			nth_element(g+l,g+mid,g+r+1,cmp1),d[g[mid]]=1;
		else
			nth_element(g+l,g+mid,g+r+1,cmp2),d[g[mid]]=2;
		lc[g[mid]]=build(l,mid-1);
		rc[g[mid]]=build(mid+1,r);
		up(g[mid]);
		return g[mid];
	}
	void pia(int x){ //将树还原成序列g
		if(!x)return;
		pia(lc[x]);
		g[++gt]=x;
		pia(rc[x]);
	}
	void ins(int &x,int v){
		if(!x){
			x=v;
			up(x);
			return;
		}
		#define ch(f) (f?rc:lc)
		if(d[x]==1)
			ins(ch(s[v].x>s[x].x)[x],v);
		else
			ins(ch(s[v].y>s[x].y)[x],v);
		up(x);
		if(0.725*sz[x]<=max(sz[lc[x]],sz[rc[x]])){
			gt=0;
			pia(x);
			x=build(1,gt);
		}
	}
	void insert(int x,int y,int v){ //在(x,y)处插入元素
		cur++;
		s[cur]={x,y,v};
		ins(rt,cur);
	}
	int x1,x2,y1,y2;
	int qry(int x){
		if(!x || x2<L[x] || x1>R[x] || y2<D[x] || y1>U[x])return 0;
		if(x1<=L[x] && R[x]<=x2 && y1<=D[x] && U[x]<=y2)return sum[x];
		int ret=0;
		if(x1<=s[x].x && s[x].x<=x2 && y1<=s[x].y && s[x].y<=y2)
			ret+=s[x].v;
		return qry(lc[x])+qry(rc[x])+ret;
	}
	int query(int _x1,int _x2,int _y1,int _y2){ //查询[x1,x2]×[y1,y2]的区间和
		x1=_x1; x2=_x2; y1=_y1; y2=_y2;
		return qry(rt);
	}
	void init(){
		rt=cur=0;
	}
}tr;
```

## 划分树

- 静态区间第 $k$ 小，可代替主席树
- 编号从 $1$ 开始，初始化 $O(n\log n)$，查询 $O(\log n)$

```c++
struct divtree{ //tr.query(l,r,k,1,n): kth in [l,r]
	int a[N],pos[25][N],tr[25][N];
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
	int query(int ql,int qr,int k,int L,int R,int dep=0){
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
	void init(int in[],int n){
		repeat(i,1,n+1)tr[0][i]=a[i]=in[i];
		sort(a+1,a+n+1);
		build(1,n,0);
	}
}tr;
```

## 莫队

- 离线（甚至在线）处理区间问题，~~猛得一批~~

### 普通莫队

- 移动指针 $l,r$ 来求所有区间的答案
- 块大小为 $\sqrt n$，$O(n^{\tfrac 3 2})$

```c++
int unit,n,bkt[N],a[N],final[N]; //bkt是桶
ll ans;
struct node{
	int l,r,id;
	bool operator<(const node &b)const{
		if(l/unit!=b.l/unit)return l<b.l; //按块排序
		if((l/unit)&1) //奇偶化排序
			return r<b.r;
		return r>b.r;
	}
};
vector<node> query; //查询区间
void update(int x,int d){
	int &b=bkt[a[x]];
	ans-=C(b,2); //操作示例
	b+=d;
	ans+=C(b,2); //操作示例
}
void solve(){ //final[]即最终答案
	fill(bkt,bkt+n+1,0);
	unit=sqrt(n)+1;
	sort(query.begin(),query.end());
	int l=1,r=0; ans=0; //如果原数组a编号从1开始
	for(auto i:query){
		while(l<i.l)update(l++,-1);
		while(l>i.l)update(--l,1);
		while(r<i.r)update(++r,1);
		while(r>i.r)update(r--,-1);
		final[i.id]=ans;
	}
}
//repeat(i,0,m)query.push_back({read(),read(),i}); //输入查询区间
```

### 带修莫队

- 相比与普通莫队，多了一个时间轴
- 块大小为 $\sqrt[3]{nt}$，$O(\sqrt[3]{n^4t})$

```c++
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

- 解决区间只能扩张不能收缩的问题
- 对于块内区间，直接暴力；对于所有左端点在一个块的区间，从块的右端点出发，先扩张右边界再扩张左边界再回滚左边界
- $O(n^{\tfrac 3 2})$

```c++
int n,unit;
ll a[N],ans[N];
typedef array<int,3> node; //l,r,id 
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

## 二叉搜索树

### 不平衡的二叉搜索树

- 左子树所有结点 $\le v <$ 右子树所有结点，目前仅支持插入，查询可以写一个 `map<int,TR *>`

```c++
struct TR{
	TR *ch[2],*fa; //ch[0]左儿子，ch[1]右儿子，fa父亲，根的父亲是inf
	int v,dep; //v是结点索引，dep深度，根的深度是1
	TR(TR *fa,int v,int dep):fa(fa),v(v),dep(dep){
		mst(ch,0);
	}
	void insert(int v2){ //tr->insert(v2)插入结点
		auto &c=ch[v2>v];
		if(c==0)c=new TR(this,v2,dep+1);
		else c->insert(v2);
	}
}*tr=new TR(0,inf,0);
//inf是无效结点，用tr->ch[0]来访问根结点
```

### 无旋treap

- 普通平衡树按v分裂，文艺平衡树按sz分裂
- insert,erase操作在普通平衡树中，push_back,output(dfs)在文艺平衡树中
- build(笛卡尔树线性构建)在普通平衡树中
- 普通平衡树

```c++
struct treap{
	struct node{
		int pri,v,sz;
		node *l,*r;
		node(int _v){pri=rnd(); v=_v; l=r=0; sz=1;}
		node(){}
		friend int size(node *u){return u?u->sz:0;}
		void up(){sz=1+size(l)+size(r);}
		friend pair<node *,node *> split(node *u,int key){ //按v分裂
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
		//top=0;
	}
	/*
	node *stk[N]; int top;
	void build(int pri,int v){ //v is nondecreasing
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
//if(opt==1)tr.insert(x);
//if(opt==2)tr.erase_one(x);
//if(opt==3)cout<<tr.order(x)+1<<endl; //x的排名
//if(opt==4)cout<<tr[x-1]<<endl; //排名为x
//if(opt==5)cout<<tr[tr.order(x)-1]<<endl; //前驱
//if(opt==6)cout<<tr.nxt(x)<<endl; //后继
```

- 文艺平衡树，tag表示翻转子树（区间）

```c++
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
		friend pair<node *,node *> split(node *u,int key){ //按sz分裂
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
	void add_tag(int l,int r){ //编号从0开始
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

### <补充> 可持久化treap

- 其他函数从treap板子里照搬（但是把rt作为参数）
- `h[i]=h[j]` 就是克隆
- 普通平衡树

```c++
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

## 一些建议

双头优先队列可以用multiset

区间众数：离线用莫队，在线用分块

支持插入、查询中位数可以用双堆

```c++
priority_queue<ll> h1; //大根堆
priority_queue< ll,vector<ll>,greater<ll> > h2; //小根堆
void insert(ll x){
	#define maintain(h1,h2,b) {h1.push(x); if(h1.size()>h2.size()+b)h2.push(h1.top()),h1.pop();}
	if(h1.empty() || h1.top()>x)maintain(h1,h2,1)
	else maintain(h2,h1,0);
}
//h1.size()+h2.size()为奇数时h1.top()为中位数，偶数看题目定义
```

双关键字堆可以用两个multiset模拟

```c++
struct HEAP{
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

- 以二维为例，t是维数
- 法一 $O(n^t2^t)$
- 法二 $O(n^tt)$

```c++
//<1>
for(int i=1;i<=n;i++)
for(int j=1;j<=m;j++)
	b[i][j]=b[i-1][j]+b[i][j-1]-b[i-1][j-1]+a[i][j];
//<2>
for(int i=1;i<=n;i++)
for(int j=1;j<=m;j++)
	a[i][j]+=a[i][j-1];
for(int i=1;i<=n;i++)
for(int j=1;j<=m;j++)
	a[i][j]+=a[i-1][j];
```

一个01串，支持把某位置的1改成0，查询某位置之后第一个1的位置，可以用并查集（删除 `d[x]=d[x+1]`，查询 `d[x]`）

手写deque很可能比stl deque慢（吸氧时）
