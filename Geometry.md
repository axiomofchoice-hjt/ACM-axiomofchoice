# 计算几何

- [1. struct of 向量](#1-struct-of-向量)
- [2. struct of 直线](#2-struct-of-直线)
- [3. struct of 圆](#3-struct-of-圆)
- [4. 平面几何基本操作](#4-平面几何基本操作)
  - [4.1. 判断两条线段是否相交](#41-判断两条线段是否相交)
  - [4.2. 点是否在线段上](#42-点是否在线段上)
  - [4.3. 多边形面积](#43-多边形面积)
  - [4.4. 多边形的面积质心](#44-多边形的面积质心)
  - [4.5. 凸包切线](#45-凸包切线)
  - [4.6. 凸包与圆的面积交](#46-凸包与圆的面积交)
- [5. 二维凸包](#5-二维凸包)
  - [5.1. \<补充\> 动态凸包](#51-补充-动态凸包)
- [6. 旋转卡壳](#6-旋转卡壳)
- [7. 最大空矩形 using 扫描法](#7-最大空矩形-using-扫描法)
- [8. 平面最近点对 using 分治](#8-平面最近点对-using-分治)
- [9. 最小圆覆盖 using 随机增量法](#9-最小圆覆盖-using-随机增量法)
- [10. 半面交 using S\&I 算法](#10-半面交-using-si-算法)
- [11. struct of 整点直线](#11-struct-of-整点直线)
- [12. 整点向量线性基](#12-整点向量线性基)
- [13. 曼哈顿最小生成树](#13-曼哈顿最小生成树)
- [14. 圆的离散化](#14-圆的离散化)
- [15. Delaunay 三角剖分](#15-delaunay-三角剖分)
- [16. struct of 三维向量](#16-struct-of-三维向量)
- [17. 三维凸包](#17-三维凸包)

## 1. struct of 向量

- `rotate()` 返回逆时针旋转后的点，`left()` 返回朝左的单位向量
- `trans()` 返回 p 沿 a, b 拉伸的结果，`arctrans()` 返回 p 在坐标系 `<a, b>` 中的坐标
- 常量式写法，不要另加变量，需要加变量就再搞个 struct
- 直线类在半面交里，其中包含线段交点

```cpp
struct vec {
    lf x, y; vec() {} vec(lf x, lf y) : x(x), y(y) {}
    vec operator-(const vec &b) { return vec(x - b.x, y - b.y); }
    vec operator+(const vec &b) { return vec(x + b.x, y + b.y); }
    vec operator*(lf k) { return vec(k * x, k * y); }
    lf len() { return hypot(x, y); }
    lf sqr() { return x * x + y * y; }
    vec trunc(lf k = 1) { return *this * (k / len()); }
    vec rotate(double th) {
        lf c = cos(th), s = sin(th);
        return vec(x * c - y * s, x * s + y * c);
    }
    vec left() { return vec(-y, x).trunc(); }
    lf theta() { return atan2(y, x); }
    friend lf cross(vec a, vec b) { return a.x * b.y - a.y * b.x; }
    friend lf cross(vec a, vec b, vec c) { return cross(a - c, b - c); }
    friend lf dot(vec a, vec b) {return a.x * b.x + a.y * b.y; }
    friend lf cos(vec a, vec b) { return dot(a, b) / sqrt(a.sqr() * b.sqr()); }
    friend vec trans(vec p, vec a, vec b) {
        swap(a.y, b.x);
        return vec(dot(a, p), dot(b, p));
    }
    friend vec arctrans(vec p, vec a, vec b) {
        lf t = cross(a, b);
        return vec(-cross(b, p) / t, cross(a, p) / t);
    }
    void output() { printf("%.12f %.12f\n", x, y); }
} a[N];
vec projection(vec v, vec a, vec b) { // v 在 line(a,b) 上的投影
    vec d = b - a;
    return a + d * (dot(v - a, d) / d.sqr());
}
```

- 整数向量

```cpp
struct vec{
    ll x, y; vec() {} vec(ll x, ll y): x(x), y(y) {}
    vec operator-(const vec &b) { return vec(x - b.x, y - b.y); }
    vec operator+(const vec &b) { return vec(x +b .x, y + b.y); }
    vec operator*(ll k) { return vec(k * x, k * y); }
    bool operator==(vec b) const { return x == b.x && y == b.y; }
    friend ll cross(vec a, vec b) { return a.x * b.y - a.y * b.x; }
    friend ll cross(vec a, vec b, vec c) { return cross(a - c, b - c); }
    friend ll dot(vec a, vec b) {return a.x * b.x + a.y * b.y; }
    ll sqr_dist(vec b) { return sqr(x - b.x) + sqr(y - b.y); }
    // ll sqr() { return x * x + y * y; }
    void output() { printf("%lld %lld\n", x, y); }
} a[N];
```

## 2. struct of 直线

```cpp
struct line{
    vec p1,p2; lf th;
    line(){}
    line(vec p1,vec p2):p1(p1),p2(p2){
        th=(p2-p1).theta();
    }
    bool contain(vec v){ // 直线是否包含给定点
        return cross(v,p2,p1)<=eps;
    }
    vec PI(line b){ // 两条直线的交点
        lf t1=cross(p1,b.p2,b.p1);
        lf t2=cross(p2,b.p2,b.p1);
        return vec((t1*p2.x-t2*p1.x)/(t1-t2),(t1*p2.y-t2*p1.y)/(t1-t2));
    }
};
```

## 3. struct of 圆

```cpp
struct cir{
    vec v; lf r;
    void PI(vec a,vec b,vec &A,vec &B){ // 与直线 (a, b) 的交点
        vec H=projection(v,a,b);
        vec D=(a-b).trunc(sqrt(r*r-(v-H).sqr()));
        A=H+D; B=H-D;
    }
    void PI(cir b,vec &A,vec &B){ // 与圆 c 的交点
        vec d=b.v-v;
        lf dis=abs(b.r*b.r-r*r-d.sqr())/(2*d.len());
        vec H=v+d.trunc(dis);
        vec D=d.left().trunc(sqrt(r*r-dis*dis));
        A=H+D; B=H-D;
    }
};
```

## 4. 平面几何基本操作

### 4.1. 判断两条线段是否相交

- 快速排斥实验：判断线段所在矩形是否相交（用来减小常数，可省略）
- 跨立实验：任一线段的两端点在另一线段的两侧

```cpp
bool judge(vec a,vec b,vec c,vec d){ // 线段 ab 和线段 cd
    #define SJ(x) max(a.x,b.x)<min(c.x,d.x)\
    || max(c.x,d.x)<min(a.x,b.x)
    if(SJ(x) || SJ(y))return 0;
    #define SJ2(a,b,c,d) cross(a-b,a-c)*cross(a-b,a-d)<=0
    return SJ2(a,b,c,d) && SJ2(c,d,a,b);
}
```

### 4.2. 点是否在线段上

```cpp
bool onseg(vec p,vec a,vec b){
    return (a.x-p.x)*(b.x-p.x)<eps
    && (a.y-p.y)*(b.y-p.y)<eps
    && abs(cross(a-b,a-p))<eps;
}
```

### 4.3. 多边形面积

```cpp
lf area(vec a[],int n){
    lf ans=0;
    repeat(i,0,n)
        ans+=cross(a[i],a[(i+1)%n]);
    return abs(ans/2);
}
```

### 4.4. 多边形的面积质心

```cpp
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

### 4.5. 凸包切线

- 先 4 次三分求最近点和极远点（有多个，求其中一个），然后 2 次二分求切点。
- （这是多校随手写的，事实上可以有更好的算法）

```cpp
int left = 0, right = 0;
repeat (i, 0, n) { // 读入凸包和预处理
    a[i].x = read(), a[i].y = read(); a[i + n] = a[i];
    if (a[i].x < a[left].x) left = i;
    if (a[i].x > a[right].x) right = i;
}
vec v; v.x = read(), v.y = read(); // 读入凸包外一点
int l = left, r = right; if (l > r) r += n;
while (l < r) { // 上凸包三分极远点
    int x = (l + r) / 2, y = x + 1;
    if (a[x].dist(v) < a[y].dist(v)) l = x + 1; else r = y - 1;
}
int mx = l % n;
l = right, r = left; if (l > r) r += n;
while (l < r) { // 下凸包三分极远点
    int x = (l + r) / 2, y = x + 1;
    if (a[x].dist(v) < a[y].dist(v)) l = x + 1; else r = y - 1;
}
if (a[l % n].dist(v) > a[mx].dist(v)) mx = l % n;
l = left, r = right; if (l > r) r += n;
while (l < r) { // 上凸包三分最近点
    int x = (l + r) / 2, y = x + 1;
    if (a[x].dist(v) > a[y].dist(v)) l = x + 1; else r = y - 1;
}
int mn = l % n;
l = right, r = left; if (l > r) r += n;
while (l < r) { // 下凸包三分最近点
    int x = (l + r) / 2, y = x + 1;
    if (a[x].dist(v) > a[y].dist(v)) l = x + 1; else r = y - 1;
}
if (a[l % n].dist(v) < a[mn].dist(v)) mn = l % n;
auto see = [](vec v, vec a, vec b) { return cross(b, v, a) < 0; };
l = mn, r = mx - 1; if (l > r) r += n;
while (l <= r) { // 二分切线（其一）
    int mid = (l + r) / 2;
    if (see(v, a[mid], a[mid + 1])) l = mid + 1;
    else r = mid - 1;
}
R = (r + n) % n + 1;
l = mx, r = mn - 1; if (l > r) r += n;
while (l <= r) { // 二分切线（其二）
    int mid = (l + r) / 2;
    if (!see(v, a[mid], a[mid + 1])) l = mid + 1;
    else r = mid - 1;
}
L = (l + n) % n + 1;
// L, R + 1 为切点，边 (L, L + 1) ... (R, R + 1) 可以被看见。
```

### 4.6. 凸包与圆的面积交

```cpp
const double PI=acos(-1);
const double eps=1e-10;
inline int sgn(double x){return fabs(x)<eps?0:(x<0?-1:1);}
inline double mysqrt(double x){return sqrt(max(0.0,x));}
struct point{
    double x,y;
    point(double a=0,double b=0):x(a),y(b){}
    point operator +(const point &A)const{
        return point(x+A.x,y+A.y);
    }
    point operator -(const point &A)const{
        return point(x-A.x,y-A.y);
    }
    point operator *(const double v)const{
        return point(x*v,y*v);
    }
    double norm(){
        return sqrt(x*x+y*y);
    }
};
double dot(point a,point b){
    return a.x*b.x+a.y*b.y;
}
double det(point a,point b){
    return a.x*b.y-a.y*b.x;
}
bool point_on_segment(point p,point s,point t){
    return sgn(det(p-s,p-t))==0&&sgn(dot(p-s,p-t))<=0;
}
vector<point>circle_cross_line(point a,point b,point o,double r){
    double dx=b.x-a.x,dy=b.y-a.y;
    double A=dx*dx+dy*dy;
    double B=2*dx*(a.x-o.x)+2*dy*(a.y-o.y);
    double C=(a.x-o.x)*(a.x-o.x)+(a.y-o.y)*(a.y-o.y)-r*r;
    double delta=B*B-4*A*C;
    vector<point>vi;
    if(sgn(delta)>=0){
        double t1=(-B+mysqrt(delta))/(2*A);
        double t2=(-B-mysqrt(delta))/(2*A);
        vi.push_back(point(a.x+t1*dx,a.y+t1*dy));
        if(sgn(delta)>0)vi.push_back(point(a.x+t2*dx,a.y+t2*dy));
    }
    return vi;
}
double sector_area(point a,point b,double r){
    double ang=atan2(a.y,a.x)-atan2(b.y,b.x);
    if(ang<0)ang+=2*PI;
    if(ang>2*PI)ang-=2*PI;
    return r*r*min(ang,2*PI-ang)/2;
}
double circle_cross_triangle(point a,point b,double r){
    int ina=sgn(a.norm()-r)<0;
    int inb=sgn(b.norm()-r)<0;
    if(ina&&inb)return fabs(det(a,b))/2.0;
    vector<point>p=circle_cross_line(a,b,point(0,0),r);
    if(ina){
        if(point_on_segment(p[0],a,b)==0)swap(p[0],p[1]);
        return sector_area(b,p[0],r)+fabs(det(a,p[0]))/2.0;
    }
    else if(inb){
        if(point_on_segment(p[0],a,b)==0)swap(p[0],p[1]);
        return sector_area(p[0],a,r)+fabs(det(p[0],b))/2.0;
    }
    else{
        if(p.size()==2&&point_on_segment(p[0],a,b)&&point_on_segment(p[1],a,b)){
            if((a-p[0]).norm()>(a-p[1]).norm())swap(p[0],p[1]);
            return sector_area(a,p[0],r)+sector_area(p[1],b,r)+fabs(det(p[0],p[1]))/2.0;
        }
        else return sector_area(a,b,r);
    }
}
double circle_cross_polygon(int n,point *p,point o,double r){
    double res=0;
    p[n+1]=p[1];
    for(int i=1;i<=n;i++){
        int tmp=sgn(det(p[i]-o,p[i+1]-o));
        if(tmp)res+=tmp*circle_cross_triangle(p[i]-o,p[i+1]-o,r);
    }
    return fabs(res);
}
```

## 5. 二维凸包

- 求上凸包，按坐标 (x, y) 字典升序排序，从小到大加入栈，如果出现凹多边形情况则出栈。下凸包反着来
- $O(n\log n)$，排序是瓶颈

```cpp
vector<vec> st;
void push(vec &v, int b) {
    while ((int)st.size() > b
    && cross(st.end()[-2], st.back(), v) <= 0) // 会得到逆时针的凸包
        st.pop_back();
    st.push_back(v);
}
void convex(vec a[], int n) {
    st.clear();
    sort(a, a + n, [](vec a, vec b) {
        return make_pair(a.x, a.y) < make_pair(b.x, b.y);
    });
    repeat (i, 0, n) push(a[i], 1);
    int b = st.size();
    repeat_back (i, 0, n - 1) push(a[i], b); // repeat_back自动变成上凸包
    st.pop_back(); // 可能要对 st.size() <= 2 特判
}
```

### 5.1. <补充> 动态凸包

- 支持添加点、询问点是否在凸包内，$O(\log n)$

```cpp
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
bool out(vec v){ // whether out of the convex
    v=v-c;
    auto l=find(v),r=l; inc(r);
    return cross(l->se,r->se,v)<-eps;
}
void init(vec v1,vec v2,vec v3){
    st.clear();
    c=(v1+v2+v3)*(1.0/3);
    push(v1-c); push(v2-c); push(v3-c);
}
void add(vec v){ // add a point to convex
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
    if(l<=r)st.erase(l,r);
    else st.erase(l,st.end()),st.erase(st.begin(),r);
    st.insert({v.theta(),v});
}
```

## 6. 旋转卡壳

- 每次找到凸包每条边的最远点，基于二维凸包，$O(n\log n)$

```cpp
lf calipers(vec a[],int n){
    convex(a,n); // 凸包算法
    repeat(i,0,st.size())a[i]=st[i]; n=st.size();
    lf ans=0; int p=1; a[n]=a[0];
    repeat(i,0,n){
        while(cross(a[p],a[i],a[i+1])<cross(a[p+1],a[i],a[i+1])) // 必须逆时针凸包
            p=(p+1)%n;
        ans=max(ans,(a[p]-a[i]).len());
        ans=max(ans,(a[p+1]-a[i]).len()); // 这里求了直径
    }
    return ans;
}
```

## 7. 最大空矩形 using 扫描法

- 在范围 (0, 0) 到 (l, w) 内求面积最大的不覆盖任何点的矩形面积，$O(n^2)$，n 是点数
- 如果是 `lf` 就把 `vec` 结构体内部、`ans`、`u`和 `d` 的类型改一下

```cpp
struct vec{
    int x,y; // 可能是lf
    vec(int x,int y):x(x),y(y){}
};
vector<vec> a; // 存放点
int l,w;
int ans=0;
void work(int i){
    int u=w,d=0;
    repeat(k,i+1,a.size())
    if(a[k].y>d && a[k].y<u){
        ans=max(ans,(a[k].x-a[i].x)*(u-d)); // 更新ans
        if(a[k].y==a[i].y)return; // 可行性剪枝
        (a[k].y>a[i].y?u:d)=a[k].y; // 更新u和d
        if((l-a[i].x)*(u-d)<=ans)return; // 最优性剪枝
    }
    ans=max(ans,(l-a[i].x)*(u-d)); // 撞墙更新ans
}
int query(){
    a.push_back(vec(0,0));
    a.push_back(vec(l,w)); // 加两个点方便处理
    // 小矩形的左边靠着顶点的情况
    sort(a.begin(),a.end(),[](vec a,vec b){return a.x<b.x;});
    repeat(i,0,a.size())
        work(i);
    // 小矩形的右边靠着顶点的情况
    repeat(i,0,a.size())a[i].x=l-a[i].x; // 水平翻折
    sort(a.begin(),a.end(),[](vec a,vec b){return a.x<b.x;});
    repeat(i,0,a.size())
        work(i);
    // 小矩形左右边都不靠顶点的情况
    sort(a.begin(),a.end(),[](vec a,vec b){return a.y<b.y;});
    repeat(i,0,(int)a.size()-1)
        ans=max(ans,(a[i+1].y-a[i].y)*l);
    return ans;
}
```

## 8. 平面最近点对 using 分治

- $O(n\log n)$

```cpp
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

## 9. 最小圆覆盖 using 随机增量法

- eps可能要非常小。随机化，均摊 $O(n)$

```cpp
struct cir{ // 圆（结构体）
    vec v; lf r;
    bool out(vec b){ // 点a在圆外
        return (v-b).len()>r+eps;
    }
    cir(vec a){v=a; r=0;}
    cir(vec a,vec b){v=(a+b)*0.5; r=(v-a).len();}
    cir(vec a,vec b,vec c){ // 三个点的外接圆
        b=b-a,c=c-a;
        vec s=vec(b.sqr(),c.sqr())*0.5;
        lf d=1/cross(b,c);
        v=a+vec(s.x*c.y-s.y*b.y,s.y*b.x-s.x*c.x)*d;
        r=(v-a).len();
    }
};
cir RIA(vec a[],int n){
    repeat_back(i,2,n)swap(a[rand()%i],a[i]); // random_shuffle(a,a+n);
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

## 10. 半面交 using S&I 算法

- 编号从 0 开始，$O(n\log n)$

```cpp
vector<vec> ans; // ans: output, shows a convex hull
namespace half{
line a[N]; int n; // (a[],n): input, the final area will be the left of the lines
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

## 11. struct of 整点直线

```cpp
struct line{
    ll up,down,dx,dy; // y=(dy/dx)x+(up/down) or x=(up/down)
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
    pii d(){return {dx,dy};} // 斜率
    pii d2(){ // 垂线斜率
        ll ddx=-dy,ddy=dx;
        adjust(ddx,ddy);
        return {ddx,ddy};
    }
    bool operator==(line b)const{
        return make_tuple(up,down,dx,dy)
            == make_tuple(b.up,b.down,b.dx,b.dy);
    }
};
struct h{ // Hash
    ll operator()(line a)const{
        return a.up+a.down*10000+a.dx*100000000+a.dy*1000000000000;
    }
};
```

## 12. 整点向量线性基

- n 个向量的集合 $\{(x_i,y_i)\}$ 可以构造线性等价的两个向量的集合 $\{(a_1,b_1),(a_2,b_2)\},(b_2=0)$，即 $\displaystyle\{\sum_{i=1}^n t_i(x_i,y_i)\mid t\in \mathbb{Z}^n\}=\{t_1(a_1,b_1)+t_2(a_2,b_2)\mid t\in \mathbb{Z}^2\}$
- `linear::push(x,y)`: 添加向量
- `linear::query(x,y)`: 询问向量能否被线性表示

```cpp
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

## 13. 曼哈顿最小生成树

- input: `n,a[i].x,a[i].y`，`a[i].p` 是没用的。编号从 1 开始，$O(n\log n)$

```cpp
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
struct BIT{ // special
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

## 14. 圆的离散化

- refer to CTSC 07 高逸涵
- 若干圆，任意两圆不相切，求未被圆覆盖的闭合图形个数
- 将圆的上下顶点和两两圆的交点的y作为事件，取相邻事件中点 $e[i]$，分析其状态，对相邻的 $e[i]$ 用并查集判连通

```cpp
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

## 15. Delaunay 三角剖分

- 编号从 0 开始，$O(n\log n)$

```cpp
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
vector<pii> ans; // 三角剖分结果
struct DT{ // 使用方法：直接solve()
    list<edge> a[N]; vec v[N]; int n;
    void solve(int _n,vec _v[]){
        n=_n;
        repeat (i, 0, n) a[i].clear();
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

```cpp
vec a[N]; DSU d;
vector<int> e[N]; // 最小生成树结果
void MST(){ // 求最小生成树
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

## 16. struct of 三维向量

```cpp
struct vec {
    lf x, y, z;
    vec(){} vec(lf x, lf y, lf z): x(x), y(y), z(z) {}
    vec operator-(vec b) { return vec(x - b.x, y - b.y, z - b.z); }
    vec operator+(vec b) { return vec(x + b.x, y + b.y, z + b.z); }
    vec operator*(lf k) { return vec(k * x, k * y, k * z); }
    lf sqr() { return x * x + y * y + z * z; }
    lf len() { return sqrt(x * x + y * y + z * z); }
    vec trunc(lf k = 1) { return *this * (k / len()); } // 截取
    friend vec cross(vec a, vec b) { // 叉积
        return vec(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
    friend lf dot(vec a, vec b) { // 点积
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    void scan() { scanf("%lf%lf%lf", &x, &y, &z); }
    void print() { printf("%.12f %.12f %.12f", x, y, z); }
};
vec rotate(vec p, vec l, lf th) { // p 绕轴 (O, l) 旋转 th 弧度
    l = l.trunc();
    return p * cos(th) + cross(l, p) * sin(th)
        + l * dot(l, p) * (1 - cos(th));
}
vec rotate(vec p, vec l0, vec l1, lf th) { // p 绕轴 (l0, l1) 旋转 th 弧度
    return rotate(p - l0, l1 - l0, th) + l0;
}
void inter_ff(vec p1, vec dir1, vec p2, vec dir2, vec &res1, vec &res2) { // 面与面的交线
    vec e = cross(dir1, dir2), v = cross(dir1, e);
    lf d = dot(dir2, v); if (abs(d) < 1e-9) return;
    vec q = p1 + v * (dot(dir2, p2 - p1) / d);
    res1 = q;
    res2 = q + e;
}
lf dist_pp(vec p1, vec p2) { // 点与点的距离
    return (p2 - p1).len();
}
lf dist_pl(vec p, vec l1, vec l2) { // 点与线的距离
    return cross(l2 - l1, p - l1).len() / (l2 - l1).len();
}
vec perpendicular_pl(vec p, vec l1, vec l2) { // 点到线的垂足
    return l1 + (l2 - l1) * (dot(l2 - l1, p - l1) / (l2 - l1).sqr());
}
void inter_lo(vec l1, vec l2, vec o, lf r, vec &res1, vec &res2) { // 直线与球的交点
    lf dis = dist_pl(o, l1, l2); if (dis > r) return;
    vec delta = (l2 - l1).trunc(sqrt(r * r - dis * dis));
    vec mid = perpendicular_pl(o, l1, l2);
    res1 = mid + delta;
    res2 = mid - delta;
}
```

## 17. 三维凸包

- 将所有凸包上的面放入面集 `f` 中，其中 `face::p[i]` 作为 `a` 的下标，$O(n^2)$

```cpp
const lf eps=1e-9;
struct vec{
    lf x,y,z;
    vec(lf x=0,lf y=0,lf z=0):x(x),y(y),z(z){};
    vec operator-(vec b){return vec(x-b.x,y-b.y,z-b.z);}
    lf len(){return sqrt(x*x+y*y+z*z);}
    void shake(){ // 微小扰动
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
    vec normal(){ // 法向量
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
    repeat(i,0,n)a[i].shake(); // 防止四点共面
    f.clear();
    f.push_back((face){0,1,2});
    f.push_back((face){0,2,1});
    repeat(i,3,n){
        c.clear();
        repeat(j,0,f.size()){
            bool t=see(f[j],a[i]);
            if(!t) // 加入背面
                c.push_back(f[j]);
            repeat(k,0,3){
                int x=f[j].p[k],y=f[j].p[(k+1)%3];
                vis[x][y]=t;
            }
        }
        repeat(j,0,f.size())
        repeat(k,0,3){
            int x=f[j].p[k],y=f[j].p[(k+1)%3];
            if(vis[x][y] && !vis[y][x]) // 加入新面
                c.push_back((face){x,y,i});
        }
        f.swap(c);
    }
}
```
