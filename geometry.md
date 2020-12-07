<!-- TOC -->

- [计算几何](#计算几何)
	- [struct of 向量](#struct-of-向量)
	- [平面几何基本操作](#平面几何基本操作)
		- [判断两条线段是否相交](#判断两条线段是否相交)
		- [others of 平面几何基本操作](#others-of-平面几何基本操作)
	- [二维凸包](#二维凸包)
		- [<补充>动态凸包](#补充动态凸包)
	- [旋转卡壳](#旋转卡壳)
	- [最大空矩形 | 扫描法](#最大空矩形--扫描法)
	- [平面最近点对 | 分治](#平面最近点对--分治)
	- [最小圆覆盖 | 随机增量法×RIA](#最小圆覆盖--随机增量法×ria)
	- [半面交 | S&I算法](#半面交--si算法)
	- [Delaunay三角剖分](#delaunay三角剖分)
	- [struct of 三维向量](#struct-of-三维向量)
	- [球面几何](#球面几何)
	- [三维凸包](#三维凸包)
	- [计算几何杂项](#计算几何杂项)
		- [正幂反演](#正幂反演)
		- [others of 计算几何杂项](#others-of-计算几何杂项)

<!-- /TOC -->

# 计算几何

## struct of 向量

- rotate()返回逆时针旋转后的点，left()返回朝左的单位向量
- trans()返回p沿a,b拉伸的结果，arctrans()返回p在坐标系<a,b>中的坐标
- 常量式写法，不要另加变量，需要加变量就再搞个struct
- 直线类在半面交里，其中包含线段交点

```c++
struct vec{
	lf x,y; vec(){} vec(lf x,lf y):x(x),y(y){}
	vec operator-(const vec &b){return vec(x-b.x,y-b.y);}
	vec operator+(const vec &b){return vec(x+b.x,y+b.y);}
	vec operator*(lf k){return vec(k*x,k*y);}
	lf len(){return hypot(x,y);}
	lf sqr(){return x*x+y*y;}
	vec trunc(lf k=1){return *this*(k/len());}
	vec rotate(double th){lf c=cos(th),s=sin(th); return vec(x*c-y*s,x*s+y*c);}
	vec left(){return vec(-y,x).trunc();}
	lf theta(){return atan2(y,x);}
	friend lf cross(vec a,vec b){return a.x*b.y-a.y*b.x;};
	friend lf cross(vec a,vec b,vec c){return cross(a-c,b-c);}
	friend lf dot(vec a,vec b){return a.x*b.x+a.y*b.y;}
	friend vec trans(vec p,vec a,vec b){
		swap(a.y,b.x);
		return vec(dot(a,p),dot(b,p));
	}
	friend vec arctrans(vec p,vec a,vec b){
		lf t=cross(a,b);
		return vec(-cross(b,p)/t,cross(a,p)/t);
	}
	void output(){printf("%.12f %.12f\n",x,y);}
}a[N];
```

- 整数向量

```c++
struct vec{
	int x,y; vec(){} vec(int x,int y):x(x),y(y){}
	vec operator-(const vec &b){return vec(x-b.x,y-b.y);}
	vec operator+(const vec &b){return vec(x+b.x,y+b.y);}
	void operator+=(const vec &b){x+=b.x,y+=b.y;}
	void operator-=(const vec &b){x-=b.x,y-=b.y;}
	vec operator*(lf k){return vec(k*x,k*y);}
	bool operator==(vec b)const{return x==b.x && y==b.y;}
	int sqr(){return x*x+y*y;}
	void output(){printf("%lld %lld\n",x,y);}
}a[N]; const vec dn[]={{1,0},{0,1},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
```

## 平面几何基本操作

### 判断两条线段是否相交

- 快速排斥实验：判断线段所在矩形是否相交（用来减小常数，可省略）
- 跨立实验：任一线段的两端点在另一线段的两侧

```c++
bool judge(vec a,vec b,vec c,vec d){ //线段ab和线段cd
	#define SJ(x) max(a.x,b.x)<min(c.x,d.x)\
	|| max(c.x,d.x)<min(a.x,b.x)
	if(SJ(x) || SJ(y))return 0;
	#define SJ2(a,b,c,d) cross(a-b,a-c)*cross(a-b,a-d)<=0
	return SJ2(a,b,c,d) && SJ2(c,d,a,b);
}
```

### others of 平面几何基本操作

<H3>点是否在线段上</H3>

```c++
bool onseg(vec p,vec a,vec b){
	return (a.x-p.x)*(b.x-p.x)<eps
	&& (a.y-p.y)*(b.y-p.y)<eps
	&& abs(cross(a-b,a-p))<eps;
}
```

<H3>多边形面积</H3>

```c++
lf area(vec a[],int n){
	lf ans=0;
	repeat(i,0,n)
		ans+=cross(a[i],a[(i+1)%n]);
	return abs(ans/2);
}
```

<H3>多边形的面积质心</H3>

```c++
vec centre(vec a[],int n){
	lf S=0; vec v=vec();
	repeat(i,0,n){
		vec &v1=a[i],&v2=a[(i+1)%n];
		lf s=cross(v1,v2);
		S+=s; v=v+(v1+v2)*s;
	}
	return v*(1/(3*S));
}
```

## 二维凸包

- 求上凸包，按坐标 $(x,y)$ 字典升序排序，从小到大加入栈，如果出现凹多边形情况则出栈。下凸包反着来
- $O(n\log n)$，排序是瓶颈

```c++
vector<vec> st;
void push(vec &v,int b){
	while((int)st.size()>b
	&& cross(*++st.rbegin(),st.back(),v)<=0) //会得到逆时针的凸包
		st.pop_back();
	st.push_back(v);
}
void convex(vec a[],int n){
	st.clear();
	sort(a,a+n,[](vec a,vec b){
		return make_pair(a.x,a.y)<make_pair(b.x,b.y);
	});
	repeat(i,0,n)push(a[i],1);
	int b=st.size();
	repeat_back(i,1,n-1)push(a[i],b); //repeat_back自动变成上凸包
}
```

### <补充>动态凸包

- 支持添加点、询问点是否在凸包内，$O(\log n)$

```c++
const lf eps=1e-7;
multiset<pair<lf,vec>> st; vec c;
typedef multiset<pair<lf,vec>>::iterator ptr;
void push(vec v){
	st.insert({v.theta(),v});
}
void dec(ptr &p){if(p==st.begin())p=st.end(); --p;}
void inc(ptr &p){++p; if(p==st.end())p=st.begin();}
ptr find(vec v){
	auto p=st.lower_bound({v.theta(),v}); dec(p);
	return p;
}
bool out(vec v){ //whether out of the convex
	v=v-c;
	auto l=find(v),r=l; inc(r);
	return cross(l->se,r->se,v)<-eps;
}
void init(vec v1,vec v2,vec v3){
	st.clear();
	c=(v1+v2+v3)*(1.0/3);
	push(v1-c); push(v2-c); push(v3-c);
}
void add(vec v){ //add a point to convex
	if(!out(v))return;
	v=v-c;
	auto l=find(v),r=l; inc(r);
	auto l2=l; dec(l2);
	while(cross(l->se,l2->se,v)>0){
		dec(l),dec(l2);
	}
	auto r2=r; inc(r2);
	while(cross(r->se,r2->se,v)<0){
		inc(r),inc(r2);
	}
	inc(l);
	if(*l<=*r)st.erase(l,r);
	else st.erase(l,st.end()),st.erase(st.begin(),r);
	st.insert({v.theta(),v});
}
```

## 旋转卡壳

- 每次找到凸包每条边的最远点，基于二维凸包，$O(n\log n)$

```c++
lf calipers(vec a[],int n){
	convex(a,n); //凸包算法
	repeat(i,0,st.size())a[i]=st[i]; n=st.size();
	lf ans=0; int p=1; a[n]=a[0];
	repeat(i,0,n){
		while(cross(a[p],a[i],a[i+1])<cross(a[p+1],a[i],a[i+1])) //必须逆时针凸包
			p=(p+1)%n;
		ans=max(ans,(a[p]-a[i]).len());
		ans=max(ans,(a[p+1]-a[i]).len()); //这里求了直径
	}
	return ans;
}
```

## 最大空矩形 | 扫描法

- 在范围 $(0,0)$ 到 $(l,w)$ 内求面积最大的不覆盖任何点的矩形面积，$O(n^2)$，$n$ 是点数
- 如果是 `lf` 就把 `vec` 结构体内部、`ans`、`u`和 `d` 的类型改一下

```c++
struct vec{
	int x,y; //可能是lf
	vec(int x,int y):x(x),y(y){}
};
vector<vec> a; //存放点
int l,w;
int ans=0;
void work(int i){
	int u=w,d=0;
	repeat(k,i+1,a.size())
	if(a[k].y>d && a[k].y<u){
		ans=max(ans,(a[k].x-a[i].x)*(u-d)); //更新ans
		if(a[k].y==a[i].y)return; //可行性剪枝
		(a[k].y>a[i].y?u:d)=a[k].y; //更新u和d
		if((l-a[i].x)*(u-d)<=ans)return; //最优性剪枝
	}
	ans=max(ans,(l-a[i].x)*(u-d)); //撞墙更新ans
}
int query(){
	a.push_back(vec(0,0));
	a.push_back(vec(l,w)); //加两个点方便处理
	//小矩形的左边靠着顶点的情况
	sort(a.begin(),a.end(),[](vec a,vec b){return a.x<b.x;});
	repeat(i,0,a.size())
		work(i);
	//小矩形的右边靠着顶点的情况
	repeat(i,0,a.size())a[i].x=l-a[i].x; //水平翻折
	sort(a.begin(),a.end(),[](vec a,vec b){return a.x<b.x;});
	repeat(i,0,a.size())
		work(i);
	//小矩形左右边都不靠顶点的情况
	sort(a.begin(),a.end(),[](vec a,vec b){return a.y<b.y;});
	repeat(i,0,(int)a.size()-1)
		ans=max(ans,(a[i+1].y-a[i].y)*l);
	return ans;
}
```

## 平面最近点对 | 分治

- $O(n\log n)$

```c++
lf ans;
bool cmp_y(vec a,vec b){return a.y<b.y;}
void work(int l,int r){
	#define upd(x,y) {ans=min(ans,(x-y).len());}
	if(r-l<=4){
		repeat(i,l,r)
		repeat(j,i+1,r)
			upd(a[i],a[j]);
		sort(a+l,a+r,cmp_y);
		return;
	}
	int m=(l+r)/2;
	lf midx=a[m].x;
	work(l,m); work(m,r);
	static vec b[N];
	inplace_merge(a+l,a+m,a+r,cmp_y);
	int t=0;
	repeat(i,l,r)
	if(abs(a[i].x-midx)<ans){
		repeat_back(j,0,t){
			if(a[i].y-b[j].y>ans)break;
			upd(a[i],b[j]);
		}
		b[t++]=a[i];
	}
}
lf nearest(int n){
	ans=1e20;
	sort(a,a+n,[](vec a,vec b){return a.x<b.x;});
	work(0,n);
	return ans;
}
```

## 最小圆覆盖 | 随机增量法×RIA

- eps可能要非常小。随机化，均摊 $O(n)$

```c++
struct cir{ //圆（结构体）
	vec v; lf r;
	bool out(vec b){ //点a在圆外
		return (v-b).len()>r+eps;
	}
	cir(vec a){v=a; r=0;}
	cir(vec a,vec b){v=(a+b)*0.5; r=(v-a).len();}
	cir(vec a,vec b,vec c){ //三个点的外接圆
		b=b-a,c=c-a;
		vec s=vec(b.sqr(),c.sqr())*0.5;
		lf d=1/cross(b,c);
		v=a+vec(s.x*c.y-s.y*b.y,s.y*b.x-s.x*c.x)*d;
		r=(v-a).len();
	}
};
cir RIA(vec a[],int n){
	repeat_back(i,2,n)swap(a[rand()%i],a[i]); //random_shuffle(a,a+n);
	cir c=cir(a[0]);
	repeat(i,1,n)if(c.out(a[i])){
		c=cir(a[i]);
		repeat(j,0,i)if(c.out(a[j])){
			c=cir(a[i],a[j]);
			repeat(k,0,j)if(c.out(a[k]))
				c=cir(a[i],a[j],a[k]);
		}
	}
	return c;
}
```

## 半面交 | S&I算法

- 编号从 $0$ 开始，$O(n\log n)$

```c++
struct line{
	vec p1,p2; lf th;
	line(){}
	line(vec p1,vec p2):p1(p1),p2(p2){
		th=(p2-p1).theta();
	}
	bool contain(vec v){
		return cross(v,p2,p1)<=eps;
	}
	vec PI(line b){ //point of intersection
		lf t1=cross(p1,b.p2,b.p1);
		lf t2=cross(p2,b.p2,b.p1);
		return vec((t1*p2.x-t2*p1.x)/(t1-t2),(t1*p2.y-t2*p1.y)/(t1-t2));
	}
};
vector<vec> ans; //ans: output, shows a convex hull
namespace half{
line a[N]; int n; //(a[],n): input, the final area will be the left of the lines
deque<line> q;
void solve(){
	a[n++]=line(vec(inf,inf),vec(-inf,inf));
	a[n++]=line(vec(-inf,inf),vec(-inf,-inf));
	a[n++]=line(vec(-inf,-inf),vec(inf,-inf));
	a[n++]=line(vec(inf,-inf),vec(inf,inf));
	sort(a,a+n,[](line a,line b){
		if(a.th<b.th-eps)return 1;
		if(a.th<b.th+eps && b.contain(a.p1)==1)return 1;
		return 0;
	});
	n=unique(a,a+n,[](line a,line b){return abs(a.th-b.th)<eps;})-a;
	q.clear();
	#define r q.rbegin()
	repeat(i,0,n){
		while(q.size()>1 && !a[i].contain(r[0].PI(r[1])))q.pop_back();
		while(q.size()>1 && !a[i].contain(q[0].PI(q[1])))q.pop_front();
		q.push_back(a[i]);
	}
	while(q.size()>1 && !q[0].contain(r[0].PI(r[1])))q.pop_back();
	while(q.size()>1 && !r[0].contain(q[0].PI(q[1])))q.pop_front();
	#undef r
	ans.clear();
	repeat(i,0,(int)q.size()-1)ans<<q[i].PI(q[i+1]);
	ans<<q[0].PI(q.back());
}
}
```

## Delaunay三角剖分

- 编号从 $0$ 开始，$O(n\log n)$

```c++
const lf eps=1e-8;
struct vec{
	lf x,y; int id;
	explicit vec(lf a=0,lf b=0,int c=-1):x(a),y(b),id(c){}
	bool operator<(const vec &a)const{
		return x<a.x || (abs(x-a.x)<eps && y<a.y);
	}
	bool operator==(const vec &a)const{
		return abs(x-a.x)<eps && abs(y-a.y)<eps;
	}
	lf dist2(const vec &b){
		return (x-b.x)*(x-b.x)+(y-b.y)*(y-b.y);
	}
};
struct vec3D{
	lf x,y,z;
	explicit vec3D(lf a=0,lf b=0,lf c=0):x(a),y(b),z(c){}
	vec3D(const vec &v){x=v.x,y=v.y,z=v.x*v.x+v.y*v.y;}
	vec3D operator-(const vec3D &a)const{
		return vec3D(x-a.x,y-a.y,z-a.z);
	}
};
struct edge{
	int id;
	list<edge>::iterator c;
	edge(int id=0){this->id=id;}
};
int cmp(lf v){return abs(v)>eps?(v>0?1:-1):0;}
lf cross(const vec &o,const vec &a,const vec &b){
	return(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
}
lf dot(const vec3D &a,const vec3D &b){return a.x*b.x+a.y*b.y+a.z*b.z;}
vec3D cross(const vec3D &a,const vec3D &b){
	return vec3D(a.y*b.z-a.z*b.y,-a.x*b.z+a.z*b.x,a.x*b.y-a.y*b.x);
}
vector<pii> ans; //三角剖分结果
struct DT{ //使用方法：直接solve()
	list<edge> a[N]; vec v[N]; int n;
	void solve(int _n,vec _v[]){
		n=_n;
		copy(_v,_v+n,v);
		sort(v,v+n);
		divide(0,n-1);
		ans.clear();
		for(int i=0;i<n;i++){
			for(auto p:a[i]){
				if(p.id<i)continue;
				ans.push_back({v[i].id,v[p.id].id});
			}
		}
	}
	int incircle(const vec &a,vec b,vec c,const vec &v){
		if(cross(a,b,c)<0)swap(b,c);
		vec3D a3(a),b3(b),c3(c),p3(v);
		b3=b3-a3,c3=c3-a3,p3=p3-a3;
		vec3D f=cross(b3,c3);
		return cmp(dot(p3,f));
	}
	int intersection(const vec &a,const vec &b,const vec &c,const vec &d){
		return cmp(cross(a,c,b))*cmp(cross(a,b,d))>0 &&
			cmp(cross(c,a,d))*cmp(cross(c,d,b))>0;
	}
	void addedge(int u,int v){
		a[u].push_front(edge(v));
		a[v].push_front(edge(u));
		a[u].begin()->c=a[v].begin();
		a[v].begin()->c=a[u].begin();
	}
	void divide(int l,int r){
		if(r-l<=2){
			for(int i=l;i<=r;i++)
				for(int j=i+1;j<=r;j++)addedge(i,j);
			return;
		}
		int mid=(l+r)/2;
		divide(l,mid); divide(mid+1,r);
		int nowl=l,nowr=r;
		for(int update=1;update;){
			update=0;
			vec vl=v[nowl],vr=v[nowr];
			for(auto i:a[nowl]){
				vec t=v[i.id];
				lf v=cross(vr,vl,t);
				if(cmp(v)>0 || (cmp(v)== 0 && vr.dist2(t)<vr.dist2(vl))){
					nowl=i.id,update=1;
					break;
				}
			}
			if(update)continue;
			for(auto i:a[nowr]){
				vec t=v[i.id];
				lf v=cross(vl,vr,t);
				if(cmp(v)<0 || (cmp(v)== 0 && vl.dist2(t)<vl.dist2(vr))){
					nowr=i.id,update=1;
					break;
				}
			}
		}
		addedge(nowl,nowr);
		while(1){
			vec vl=v[nowl],vr=v[nowr];
			int ch=-1,side=0;
			for(auto i:a[nowl])
			if(cmp(cross(vl,vr,v[i.id]))>0 && (ch==-1 || incircle(vl,vr,v[ch],v[i.id])<0)){
				ch=i.id,side=-1;
			}
			for(auto i:a[nowr])
			if(cmp(cross(vr,v[i.id],vl))>0 && (ch==-1 || incircle(vl,vr,v[ch],v[i.id])<0)){
				ch=i.id,side=1;
			}
			if(ch==-1)break;
			if(side==-1){
				for(auto it=a[nowl].begin();it!=a[nowl].end();){
					if(intersection(vl,v[it->id],vr,v[ch])){
						a[it->id].erase(it->c);
						a[nowl].erase(it++);
					}
					else it++;
				}
				nowl=ch;
				addedge(nowl,nowr);
			}
			else{
				for(auto it=a[nowr].begin();it!=a[nowr].end();){
					if(intersection(vr,v[it->id],vl,v[ch])){
						a[it->id].erase(it->c);
						a[nowr].erase(it++);
					}
					else it++;
				}
				nowr=ch;
				addedge(nowl,nowr);
			}
		}
	}
}dt;
```

- 可以求最小生成树

```c++
vec a[N]; DSU d;
vector<int> e[N]; //最小生成树结果
void mst(){ //求最小生成树
	dt.solve(n,a);
	sort(ans.begin(),ans.end(),[](const pii &A,const pii &B){
		return a[A.fi].dist2(a[A.se])<a[B.fi].dist2(a[B.se]);
	});
	d.init(n);
	for(auto i:ans)
	if(d[i.fi]!=d[i.se]){
		e[i.fi].push_back(i.se);
		e[i.se].push_back(i.fi);
		d[i.fi]=d[i.se];
	}
}
```

## struct of 三维向量

- `trunc(K)` 返回 `K` 在 `*this` 上的投影向量
- `rotate(P,L,th)` 返回点 `P` 绕轴 `(O,L)` 旋转 `th` 弧度后的点
- `rotate(P,L0,L1,th)` 返回点 `P` 绕轴 `(L0,L1)` 旋转 `th` 弧度后的点

```c++
struct vec{
	lf x,y,z; vec(){} vec(lf x,lf y,lf z):x(x),y(y),z(z){}
	vec operator-(vec b){return vec(x-b.x,y-b.y,z-b.z);}
	vec operator+(vec b){return vec(x+b.x,y+b.y,z+b.z);}
	vec operator*(lf k){return vec(k*x,k*y,k*z);}
	bool operator<(vec b)const{return make_tuple(x,y,z)<make_tuple(b.x,b.y,b.z);}
	lf sqr(){return x*x+y*y+z*z;}
	lf len(){return sqrt(x*x+y*y+z*z);}
	vec trunc(lf k=1){return *this*(k/len());}
	vec trunc(vec k){return *this*(dot(*this,k)/sqr());}
	friend vec cross(vec a,vec b){
		return vec(
			a.y*b.z-a.z*b.y,
			a.z*b.x-a.x*b.z,
			a.x*b.y-a.y*b.x);
	}
	friend lf dot(vec a,vec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
	friend vec rotate(vec p,vec l,lf th){
		struct four{
			lf r; vec v;
			four operator*(four b){
				return {r*b.r-dot(v,b.v),v*b.r+b.v*r+cross(v,b.v)};
			}
		};
		l=l.trunc();
		four P={0,p};
		four Q1={cos(th/2),l*sin(th/2)};
		four Q2={cos(th/2),vec()-l*sin(th/2)};
		return ((Q1*P)*Q2).v;
	}
	friend vec rotate(vec p,vec l0,vec l1,lf th){
		return rotate(p-l0,l1-l0,th)+l0;
	}
	void output(){printf("%.12f %.12f %.12f\n",x,y,z);}
};
```

## 球面几何

```c++
vec to_vec(lf lng,lf lat){ //lng经度，lat纬度，-90<lat<90
	lng*=pi/180,lat*=pi/180;
	lf z=sin(lat),m=cos(lat);
	lf x=cos(lng)*m,y=sin(lng)*m;
	return vec(x,y,z);
};
lf to_lng(vec v){return atan2(v.y,v.x)*180/pi;}
lf to_lat(vec v){return asin(v.z)*180/pi;}
lf angle(vec a,vec b){return acos(dot(a,b));}
```

## 三维凸包

- 将所有凸包上的面放入面集 `f` 中，其中 `face::p[i]` 作为 `a` 的下标，$O(n^2)$

```c++
const lf eps=1e-9;
struct vec{
	lf x,y,z;
	vec(lf x=0,lf y=0,lf z=0):x(x),y(y),z(z){};
	vec operator-(vec b){return vec(x-b.x,y-b.y,z-b.z);}
	lf len(){return sqrt(x*x+y*y+z*z);}
	void shake(){ //微小扰动
		x+=(rand()*1.0/RAND_MAX-0.5)*eps;
		y+=(rand()*1.0/RAND_MAX-0.5)*eps;
		z+=(rand()*1.0/RAND_MAX-0.5)*eps;
	}
}a[N];
vec cross(vec a,vec b){
	return vec(
		a.y*b.z-a.z*b.y,
		a.z*b.x-a.x*b.z,
		a.x*b.y-a.y*b.x);
}
lf dot(vec a,vec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
struct face{
	int p[3];
	vec normal(){ //法向量
		return cross(a[p[1]]-a[p[0]],a[p[2]]-a[p[0]]);
	}
	lf area(){return normal().len()/2.0;}
};
vector<face> f;
bool see(face f,vec v){
	return dot(v-a[f.p[0]],f.normal())>0;
}
void convex(vec a[],int n){
	static vector<face> c;
	static bool vis[N][N];
	repeat(i,0,n)a[i].shake(); //防止四点共面
	f.clear();
	f.push_back((face){0,1,2});
	f.push_back((face){0,2,1});
	repeat(i,3,n){
		c.clear();
		repeat(j,0,f.size()){
			bool t=see(f[j],a[i]);
			if(!t) //加入背面
				c.push_back(f[j]);
			repeat(k,0,3){
				int x=f[j].p[k],y=f[j].p[(k+1)%3];
				vis[x][y]=t;
			}
		}
		repeat(j,0,f.size())
		repeat(k,0,3){
			int x=f[j].p[k],y=f[j].p[(k+1)%3];
			if(vis[x][y] && !vis[y][x]) //加入新面
				c.push_back((face){x,y,i});
		}
		f.swap(c);
	}
}
```

## 计算几何杂项

### 正幂反演

- 给定反演中心 $O$ 和反演半径 $R$。若直线上的点 $OPQ$ 满足 $|OP|\cdot|OQ|=R^2$，则 $P$ 和 $Q$ 互为反演点（令 $R=1$ 也可）
- 不经过反演中心的圆的反演图形是圆（计算时取圆上靠近/远离中心的两个点）
- 经过反演中心的圆的反演图形是直线（计算时取远离中心的点，做垂线）

### others of 计算几何杂项

<H3>曼哈顿、切比雪夫距离</H3>

- 曼：`mdist=|x1-x2|+|y1-y2|`
- 切：`cdist=max(|x1-x2|,|y1-y2|)`
- 转换：
	- `mdist((x,y),*)=cdist((x+y,x-y),**)`
	- `cdist((x,y),*)=mdist(((x+y)/2,(x-y)/2),**)`

<H3>Pick定理</H3>

- 可以用Pick定理求多边形内部整点个数，其中一条线段上的点数为 $\gcd(|x_1-x_2|,|y_1-y_2|)+1$
- 正方形点阵：`面积 = 内部点数 + 边上点数 / 2 - 1`
- 三角形点阵：`面积 = 2 * 内部点数 + 边上点数 - 2`

<H3>圆的面积并</H3>

- 求每个圆未被其他圆覆盖的圆弧，逆时针作它的弦的向量，答案为这些弓形面积加上弦向量组成的多边形有向面积
- $O(n^2\log n)$

<H3>圆的扫描线</H3>

- empty

<H3>几何公式</H3>

- 三角形面积 $S=\sqrt{P(P-a)(P-b)(P-c)}$，$P$ 为半周长
- 斯特瓦尔特定理：$BC$ 上一点 $P$，有 $AP=\sqrt{AB^2\cdot \dfrac{CP}{BC}+AC^2\cdot \dfrac{BP}{BC}-BP\cdot CP}$
- 三角形内切圆半径 $r=\dfrac {2S} C$，外接圆半径 $R=\dfrac{a}{2\sin A}=\dfrac{abc}{4S}$
- 四边形有 $a^2+b^2+c^2+d^2=D_1^2+D_2^2+4M^2$，$D_1,D_2$ 为对角线，$M$ 为对角线中点连线
- 圆内接四边形有 $ac+bd=D_1D_2$，$S=\sqrt{(P-a)(P-b)(P-c)(P-d)}$，$P$ 为半周长
- 棱台体积 $V=\dfrac 13(S_1+S_2+\sqrt{S_1S_2})h$，$S_1,S_2$ 为上下底面积
- 正棱台侧面积 $\dfrac 1 2(C_1+C_2)L$，$C_1,C_2$ 为上下底周长，$L$ 为斜高（上下底对应的平行边的距离）
- 球全面积 $S=4\pi r^2$，体积 $V=\dfrac 43\pi r^3$，
- 球台(球在平行平面之间的部分)有 $h=|\sqrt{r^2-r_1^2}\pm\sqrt{r^2-r_2^2}|$，侧面积 $S=2\pi r h$，体积 $V=\dfrac{1}{6}\pi h[3(r_1^2+r_2^2)+h^2]$，$r_1,r_2$ 为上下底面半径
- 正三角形面积 $S=\dfrac{\sqrt 3}{4}a^2$，正四面体面积 $S=\dfrac{\sqrt 2}{12}a^3$
- 四面体体积公式

```c++
lf sqr(lf x){return x*x;}
lf V(lf a,lf b,lf c,lf d,lf e,lf f){ //a,b,c共顶点
	lf A=b*b+c*c-d*d;
	lf B=a*a+c*c-e*e;
	lf C=a*a+b*b-f*f;
	return sqrt(4*sqr(a*b*c)-sqr(a*A)-sqr(b*B)-sqr(c*C)+A*B*C)/12;
}
```
