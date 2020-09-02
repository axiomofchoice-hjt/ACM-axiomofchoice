<!-- TOC -->

- [语法](#语法)
	- [C++11](#c11)
		- [初始代码](#初始代码)
		- [如果没有万能头](#如果没有万能头)
		- [容器](#容器)
		- [其他语法](#其他语法)
		- [神奇特性](#神奇特性)
	- [Java](#java)
	- [python3](#python3)
- [常规算法](#常规算法)
	- [算法基础](#算法基础)
	- [离散化](#离散化)
	- [01分数规划](#01分数规划)
	- [任务规划 | Livshits-Kladov定理](#任务规划--livshits-kladov定理)
	- [分治](#分治)
		- [逆序数×二维偏序](#逆序数×二维偏序)
	- [最大空矩阵 | 悬线法](#最大空矩阵--悬线法)
	- [搜索](#搜索)
		- [舞蹈链×DLX](#舞蹈链×dlx)
		- [启发式算法](#启发式算法)
	- [动态规划](#动态规划)
		- [多重背包](#多重背包)
		- [最长不降子序列×LIS](#最长不降子序列×lis)
		- [数位dp](#数位dp)
		- [换根dp](#换根dp)
		- [斜率优化](#斜率优化)
		- [四边形优化](#四边形优化)
- [计算几何](#计算几何)
	- [struct of 向量](#struct-of-向量)
	- [平面几何基本操作](#平面几何基本操作)
		- [判断两条线段是否相交](#判断两条线段是否相交)
		- [others of 平面几何基本操作](#others-of-平面几何基本操作)
	- [二维凸包](#二维凸包)
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
- [数据结构](#数据结构)
	- [st表](#st表)
		- [<补充>猫树](#补充猫树)
	- [单调队列](#单调队列)
	- [树状数组](#树状数组)
	- [线段树](#线段树)
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
	- [二叉搜索树](#二叉搜索树)
		- [不平衡的二叉搜索树](#不平衡的二叉搜索树)
		- [无旋treap](#无旋treap)
		- [<补充> 可持久化treap](#补充-可持久化treap)
	- [一些建议](#一些建议)
- [图论](#图论)
	- [图论的一些概念](#图论的一些概念)
	- [图论基础](#图论基础)
		- [前向星](#前向星)
		- [拓扑排序×Toposort](#拓扑排序×toposort)
		- [欧拉路径 欧拉回路](#欧拉路径-欧拉回路)
		- [dfs树 bfs树](#dfs树-bfs树)
	- [最短路径](#最短路径)
		- [Dijkstra](#dijkstra)
		- [Floyd](#floyd)
		- [SPFA](#spfa)
		- [Johnson](#johnson)
		- [最小环](#最小环)
	- [最小生成树×MST](#最小生成树×mst)
		- [Kruskal](#kruskal)
		- [Boruvka](#boruvka)
		- [最小树形图 | 朱刘算法](#最小树形图--朱刘算法)
	- [树论](#树论)
		- [树的直径](#树的直径)
		- [树的重心](#树的重心)
		- [最近公共祖先×LCA](#最近公共祖先×lca)
			- [树上倍增解法](#树上倍增解法)
			- [欧拉序列+st表解法](#欧拉序列st表解法)
			- [树链剖分解法](#树链剖分解法)
			- [Tarjan解法](#tarjan解法)
			- [一些关于lca的问题](#一些关于lca的问题)
		- [树链剖分](#树链剖分)
		- [树分治](#树分治)
			- [点分治](#点分治)
	- [联通性相关](#联通性相关)
		- [强联通分量scc+缩点 | Tarjan](#强联通分量scc缩点--tarjan)
		- [边双连通分量 | Tarjan](#边双连通分量--tarjan)
		- [割点×割顶](#割点×割顶)
	- [2-sat问题](#2-sat问题)
	- [图上的NP问题](#图上的np问题)
		- [最大团+极大团计数](#最大团极大团计数)
		- [最小染色数](#最小染色数)
	- [弦图+区间图](#弦图区间图)
	- [仙人掌 | 圆方树](#仙人掌--圆方树)
	- [二分图](#二分图)
		- [二分图的一些概念](#二分图的一些概念)
		- [二分图匹配×最大匹配](#二分图匹配×最大匹配)
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
- [字符串](#字符串)
	- [哈希×Hash](#哈希×hash)
		- [字符串哈希](#字符串哈希)
		- [质因数哈希](#质因数哈希)
	- [字符串函数](#字符串函数)
		- [前缀函数×kmp](#前缀函数×kmp)
		- [z函数×exkmp](#z函数×exkmp)
		- [马拉车×Manacher](#马拉车×manacher)
		- [最小表示法](#最小表示法)
		- [后缀数组×SA](#后缀数组×sa)
		- [height数组](#height数组)
	- [自动机](#自动机)
		- [字典树×Trie](#字典树×trie)
		- [AC自动机](#ac自动机)
		- [后缀自动机×SAM](#后缀自动机×sam)
- [杂项](#杂项)
	- [位运算](#位运算)
		- [位运算函数](#位运算函数)
		- [枚举二进制子集](#枚举二进制子集)
	- [浮点数](#浮点数)
	- [常数优化](#常数优化)
		- [快读快写](#快读快写)
		- [STL手写内存分配器](#stl手写内存分配器)
	- [在TLE边缘试探](#在tle边缘试探)
	- [对拍](#对拍)
	- [战术分析 坑点](#战术分析-坑点)
- [数学](#数学)
	- [数论](#数论)
		- [基本操作](#基本操作)
			- [模乘 模幂 模逆 扩欧](#模乘-模幂-模逆-扩欧)
			- [阶乘 组合数](#阶乘-组合数)
			- [防爆模乘](#防爆模乘)
			- [最大公约数](#最大公约数)
		- [高级模操作](#高级模操作)
			- [同余方程组 | CRT+extra](#同余方程组--crtextra)
			- [离散对数 | BSGS+extra](#离散对数--bsgsextra)
			- [阶与原根](#阶与原根)
			- [N次剩余](#n次剩余)
		- [数论函数的生成](#数论函数的生成)
			- [单个欧拉函数](#单个欧拉函数)
			- [线性递推乘法逆元](#线性递推乘法逆元)
			- [线性筛](#线性筛)
				- [筛素数](#筛素数)
				- [筛欧拉函数](#筛欧拉函数)
				- [筛莫比乌斯函数](#筛莫比乌斯函数)
				- [筛其他函数](#筛其他函数)
			- [min_25筛](#min_25筛)
		- [素数约数相关](#素数约数相关)
			- [唯一分解](#唯一分解)
			- [素数判定 | 朴素 or Miller-Rabin](#素数判定--朴素-or-miller-rabin)
			- [大数分解 | Pollard-rho](#大数分解--pollard-rho)
			- [单个约数个数函数](#单个约数个数函数)
			- [反素数生成](#反素数生成)
		- [数论杂项](#数论杂项)
			- [数论分块](#数论分块)
			- [二次剩余](#二次剩余)
			- [莫比乌斯反演](#莫比乌斯反演)
			- [杜教筛](#杜教筛)
			- [斐波那契数列](#斐波那契数列)
			- [佩尔方程×Pell](#佩尔方程×pell)
	- [组合数学](#组合数学)
		- [组合数取模 | Lucas+extra](#组合数取模--lucasextra)
		- [康托展开+逆 编码与解码](#康托展开逆-编码与解码)
		- [置换群计数](#置换群计数)
		- [组合数学的一些结论](#组合数学的一些结论)
	- [博弈论](#博弈论)
		- [SG函数 SG定理](#sg函数-sg定理)
		- [Nim游戏](#nim游戏)
		- [删边游戏×Green Hachenbush](#删边游戏×green-hachenbush)
		- [翻硬币游戏](#翻硬币游戏)
		- [高维组合游戏 | Nim积](#高维组合游戏--nim积)
		- [不平等博弈 | 超现实数](#不平等博弈--超现实数)
		- [其他博弈结论](#其他博弈结论)
	- [代数结构](#代数结构)
		- [置换群](#置换群)
		- [多项式](#多项式)
			- [拉格朗日插值](#拉格朗日插值)
			- [快速傅里叶变换+任意模数](#快速傅里叶变换任意模数)
			- [多项式的一些概念](#多项式的一些概念)
		- [矩阵](#矩阵)
			- [矩阵乘法 矩阵快速幂](#矩阵乘法-矩阵快速幂)
			- [矩阵高级操作](#矩阵高级操作)
			- [异或方程组](#异或方程组)
			- [线性基](#线性基)
			- [线性规划 | 单纯形法](#线性规划--单纯形法)
			- [矩阵的一些结论](#矩阵的一些结论)
	- [数学杂项](#数学杂项)
		- [主定理](#主定理)
		- [质数表](#质数表)
		- [struct of 自动取模](#struct-of-自动取模)
		- [struct of 高精度](#struct-of-高精度)
		- [表达式求值](#表达式求值)
		- [一些数学结论](#一些数学结论)
			- [约瑟夫问题](#约瑟夫问题)
			- [格雷码 汉诺塔](#格雷码-汉诺塔)
			- [Stern-Brocot树 Farey序列](#stern-brocot树-farey序列)
			- [浮点与近似计算](#浮点与近似计算)
			- [others of 数学杂项](#others-of-数学杂项)

<!-- /TOC -->

# 语法

## C++11

### 初始代码

```c++
#include <bits/stdc++.h>
using namespace std;
#define repeat(i,a,b) for(int i=(a),_=(b);i<_;i++)
#define repeat_back(i,a,b) for(int i=(b)-1,_=(a);i>=_;i--)
#define mst(a,x) memset(a,x,sizeof(a))
#define fi first
#define se second
mt19937 rnd(chrono::high_resolution_clock::now().time_since_epoch().count());
int cansel_sync=(ios::sync_with_stdio(0),cin.tie(0),0);
const int N=200010; typedef long long ll; const int inf=~0u>>2; const ll INF=~0ull>>2; ll read(){ll x; if(scanf("%lld",&x)==-1)exit(0); return x;} typedef double lf; const lf pi=acos(-1.0); lf readf(){lf x; if(scanf("%lf",&x)==-1)exit(0); return x;} typedef pair<ll,ll> pii; template<typename T> void operator<<(vector<T> &a,T b){a.push_back(b);}
const ll mod=(1?1000000007:998244353); ll mul(ll a,ll b,ll m=mod){return a*b%m;} ll qpow(ll a,ll b,ll m=mod){ll ans=1; for(;b;a=mul(a,a,m),b>>=1)if(b&1)ans=mul(ans,a,m); return ans;}
#define int ll
void Solve(){
}
signed main(){
	//freopen("data.txt","r",stdin);
	int T=1; T=read();
	repeat(ca,1,T+1){
		Solve();
	}
	return 0;
}
```

放在本地的内容（比如可以放进 `bits/stdc++.h`）（当然修改了头文件还要编译一下）

```c++
template<typename A,typename B>
std::ostream &operator<<(std::ostream &o,const std::pair<A,B> &x){
	return o<<'('<<x.first<<','<<x.second<<')';
}
#define qwq [&]{cerr<<"qwq"<<endl;}()
#define orz(x) [&]{cerr<<#x": "<<x<<endl;}()
#define orzarr(a,n) [&]{cerr<<#a": "; repeat(__,0,n)cerr<<(a)[__]<<" "; cerr<<endl;}()
#define orzeach(a) [&]{cerr<<#a": "; for(auto __:a)cerr<<__<<" "; cerr<<endl;}()
#define pause [&]{system("pause");}()
```

### 如果没有万能头

```c++
#include<cstdio>
#include<cmath>
#include<cstring>
#include<cstdlib>
#include<ctime>
#include<cctype>
#include<iostream>
#include<algorithm>
//#include<chrono>
#include<vector>
#include<list>
#include<queue>
#include<string>
#include<set>
#include<map>
//#include<unordered_set>
//#include<unordered_map>
```

其他定义

```c++
#pragma GCC optimize(2) //(3),("Ofast")
#define lll __int128
#define inline __inline __attribute__((always_inline))
//struct name{bool operator()(const type &x,const type &y){return func(x,y);}}
#define vector basic_string
#define sortunique(a) ({sort(a.begin(),a.end()); a.erase(unique(a.begin(),a.end()),a.end());})
#define gets(s) (scanf("%[^\n]",s)+1)
template<typename T> T sqr(const T &x){return x*x;}
typedef long double lf;
template<typename A,typename B>void operator<<(A &a,B b){a.push_back(b);}
#define endl "\n"
```

### 容器

平板电视红黑树

```c++
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
tree<pii,null_type,less<pii>,rb_tree_tag,tree_order_statistics_node_update> t; //红黑树
t.insert({x,i+1}); //----------------- 插入x，用独特的正整数i+1标注（因为erase太辣鸡）
t.erase(t.lower_bound({x,0})); //----- 删除x（删除单个元素）
t.order_of_key({x,0})+1; //----------- x的排名（小于x的元素个数+1）
t.find_by_order(x-1)->first; //------- 排名为x的元素（第x小的数）
prev(t.lower_bound({x,0}))->first; //- x的前驱（小于x且最大）
t.lower_bound({x+1,0})->first; //----- x的后继（大于x且最小）
t.join(t2); //------------------------ 将t2并入t，t2清空，前提是取值范围不相交
t.split(v,t2); //--------------------- 小于等于v的元素属于t，其余的属于t2
```

平板电视优先队列

- `pairing_heap_tag` 配对堆，应该是可并堆里最快的
- `thin_heap_tag` 斐波那契堆
- `std::priority_queue` 不合并就很快

```c++
#include<ext/pb_ds/priority_queue.hpp>
using namespace __gnu_pbds;
__gnu_pbds::priority_queue<int,less<int>,pairing_heap_tag> h; //大根堆
h.push(x); h.top(); h.pop();
h.join(h2); //将h2并入h，h2清空
```

rope

- 可能是可分裂平衡树

```c++
#include <ext/rope>
using namespace __gnu_cxx;
rope<int> r; //块状链表
r.push_back(n);
r.insert(pos,n); //插入一个元素
r.erase(pos,len); //区间删除
r.copy(pos,len,x); //区间赋值到x
r.replace(pos,x); //相当于r[pos]=x;
r.substr(pos,len); //这是啥不会用
r[pos] //只能访问不能修改
r.clear();
rope<int> *his[N]; his[0]=new rope<int>(); his[i]=new rope<int>(*his[i-1]); //据说O(1)拷贝，一行可持久化
```

### 其他语法

STL

```c++
a=move(b); //容器移动（a赋值为b，b清空）
priority_queue<int>(begin,end) //O(n)建堆
```

```c++
a.find(key) //set,map查找，没找到返回a.end()
a.lower_bound(key) //set,map限制最小值
a.insert(b.begin(),b.end()); //set,map合并（时间复杂度极高）
```

```c++
complex<lf> c; complex<lf> c(1,2);//复数
c.real(),c.imag() //实部、虚部
```

```c++
bitset<32> b; //声明一个32位的bitset
b[n]; b[n]=1; //访问和修改
b.none(); //返回是否为空
b.count(); //返回1的个数
b.to_ullong(); b.to_string(); //转换
```

unordered容器手写hash

```c++
struct myhash{
	typedef unsigned long long ull;
	ull f(ull x)const{
		x+=0x321354564536; //乱敲
		x=(x^(x>>30))*0x3212132123; //乱敲
		return x^(x>>31);
	}
	ull operator()(pii x)const{
		static ull t=chrono::steady_clock::now().time_since_epoch().count();
		return f(x.fi+t)^f(x.se+t*2);
	}
};
unordered_set<pii,myhash> a;
```

cmath

```c++
fmod(x) //浮点取模
tgamma(x) //计算Γ(x)
atan2(x,y) //计算坐标(x,y)的极角
hypot(x,y) //计算sqrt(x^2+y^2)
```

scanf字符串正则化

```c++
scanf("%ns",str); //读入n个字符
scanf("%[a-z]",str); //遇到非小写字母停止
scanf("%[^0-9]",str); //遇到数字停止，^表示非
scanf("%*[a-z]"); //也是遇到非小写字母停止，只不过不读入字符串
```

### 神奇特性

- 命名空间rel_ops：之后只定义小于就能用其他所有次序关系符号
- raw strings：`R"(abc\n)"` 相当于 `"abc\\n"`
- 定义数字开头的变量：`type operator ""_name(type number){/*...*/}`（之后 `1_name` 即把1带入上述函数中，参数类型只能是`ull,llf,char,(const char *,size_t)`）
- 高级宏：`__VA_ARGS__`是参数列表(对应`...`)，`__LINE__` 是当前行数，`__FUNCTION__`是当前函数名，`__COUNTER__`是宏展开次数-1
- 位域：`struct{int a:3;};` 表示struct里a占3 bit，可以节省空间
- %n：`scanf,printf` 中 %n 将读入/输出的字符个数写入变量

## Java

```java
import java.util.*;
import java.math.BigInteger;
import java.math.BigDecimal;
public class Main{
static Scanner sc;
public static void main(String[] args){
	sc=new Scanner(System.in);
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

- `final double PI=3.14; //final => (c++) const`
- `var n=1; //var => (c++) auto`
- `long` 型常量结尾加 `L`，如 `1L`

数组

```java
int[] arr=new int[100]; //数组
int[][] arr=new int[10][10]; //二维数组
Array.sort(arr,l,r); //对arr[l..(r-1)]排序（import java.util.Arrays;）
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
s1.indexOf(s2,begin=0) //返回子串位置
s1.substring(2,4) //返回子串，首末坐标[2,4)
s1.charAt(3) //返回第4个字符，就像c++的s1[3]
s1.length() //返回长度
s1+s2 //返回连接结果
String.format("%d",n) //返回格式化结果
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

## python3

eval 表达式求值

# 常规算法

## 算法基础

- STL自带算法

```c++
fill(begin,end,element); //填充
fill_n(begin,n,element); //填充
iota(begin,end,t); //递增填充（赋值为t,t+1,t+2,...）
copy(a_begin,a_end,b_begin); //复制（注意会复制到b里）
reverse(begin,end); //翻转
```

```c++
nth_element(begin,begin+k,end); //将第k+1小置于位置k，平均O(n)
binary_search(begin,end,key,[less]) //返回是否存在
upper_bound(begin,end,key,[less]) //返回限制最小值地址
lower_bound(begin,end,key,[less]) //返回严格限制最小值地址
merge(a_begin,a_end,b_begin,b_end,c_begin,[less]); //归并a和b（结果存c）
inplace_merge(begin,begin+k,end); //归并（原地保存）
next_permutation(begin,end); prev_permutation(begin,end); //允许多重集，返回不到底，使用方法 do{/*...*/}while(next_permutation(begin,end));
min_element(begin,end); max_element(begin,end); //返回最值的指针
for_each(begin,end,work); //每个元素进行work操作
```

```c++
auto it=back_inserter(a); //it=x表示往a.push_back(x)
```

- 进制转换

```c++
strtol(str,0,base),strtoll //返回字符数组的base进制数
stol(s,0,base),stoll //返回字符串的base进制数
sscanf,sprintf //十进制
to_string(n) //十进制
```

- 随机数

```c++
#include<random>
mt19937 rnd(time(0));
//巨佬定义：mt19937 rnd(chrono::high_resolution_clock::now().time_since_epoch().count());
cout<<rnd()<<endl; //范围是unsigned int
rnd.max() //返回最大值
lf rndf(){return rnd()*1.0/rnd.max();}
lf rndf2(){return rndf()*2-1;}
```

- 手写二分/三分

```c++
while(l<=r){
	int mid=(l+r)/2; //l+(r-l)/2
	if(ok(mid))l=m+1; else r=m-1; //小的值容易ok
}
//此时r是ok的右边界
```

```c++
while(l<r){
	int x=(l+r)/2,y=x+1; //l+(r-l)/2
	if(work(x)<work(y))l=x+1; else r=y-1; //最大值
}
//此时l和r均为极值点
```

```c++
#define f(x) (-x*x+23*x)
const lf ph=(sqrt(5)-1)/2; //0.618
lf dfs(lf l,lf x,lf r,lf fx){
	if(abs(l-r)<1e-9)return x;
	lf y=l+ph*(r-l),fy=f(y);
	if(fx<fy)return dfs(x,y,r,fy);
	else return dfs(y,x,l,fx);
}
lf search(lf l,lf r){
	lf x=r-ph*(r-l);
	return dfs(l,x,r,f(x));
}
```

- 次大值

```c++
int m1=-inf,m2=-inf;
repeat(i,0,n){m1=max(m1,a[i]); if(m1>m2)swap(m1,m2);}
//m1即次大值
```

## 离散化

- 从小到大标号并赋值，$O(n\log n)$，~~是个好东西~~

```c++
void disc(int a[],int n){
	vector<int> b(a,a+n);
	sort(b.begin(),b.end());
	b.erase(unique(b.begin(),b.end()),b.end());
	repeat(i,0,n)
		a[i]=lower_bound(b.begin(),b.end(),a[i])-b.begin(); //从0开始编号
}
```

```c++
void disc(int a[],int n,int d){ //把距离>d的拉近到d
	vector<int> b(a,a+n);
	sort(b.begin(),b.end());
	b.erase(unique(b.begin(),b.end()),b.end());
	vector<int> c(b.size()); c[0]=0; //从0开始编号
	repeat(i,1,b.size())
		c[i]=c[i-1]+min(d,b[i]-b[i-1]);
	repeat(i,0,n)
		a[i]=c[lower_bound(b.begin(),b.end(),a[i])-b.begin()];
}
```

```c++
struct Disc{ //离散化后a[]的值互不相同，但是a[i]与a[j]∈[d.pre[a[i]],d.nxt[a[i]]]在离散化前是相同的
	int b[N],pre[N],nxt[N];
	void init(int a[],int n){
		copy(a,a+n,b); sort(b,b+n);
		pre[0]=0;
		repeat(i,1,n){
			if(b[i]==b[i-1])pre[i]=pre[i-1];
			else pre[i]=i;
		}
		nxt[n-1]=n-1;
		repeat_back(i,0,n-1){
			if(b[i]==b[i+1])nxt[i]=nxt[i+1];
			else nxt[i]=i;
		}
		repeat(i,0,n){
			a[i]=lower_bound(b,b+n,a[i])-b;
			b[a[i]]--;
		}
	}
}d;
```

## 01分数规划

- $n$ 个物品，都有两个属性 $a_i$ 和 $b_i$，任意取 $k$ 个物品使它们的 $\dfrac {\sum a_j}{\sum b_j}$ 最大
- 解：二分答案
- $m$ 是否满足条件即判断 $\dfrac {\sum a_j}{\sum b_j}\ge m$，即 $\sum(a_j-mb_j)\ge 0$
- 因此计算 $c_i=a_i-mb_i$ ，取前 $k$ 个最大值看它们之和是否 $\ge 0$
- 如果限制条件是 $\sum b_j\ge W$，则将 $b_j$ 看成体积，$a_j-mb_j$ 看成价值，转换为背包dp

```c++
int n,k; lf a[N],b[N],c[N];
bool check(lf mid){
	repeat(i,0,n)c[i]=a[i]-mid*b[i];
	nth_element(c,c+k,c+n,greater<lf>());
	lf sum=0; repeat(i,0,k)sum+=c[i];
	return sum>=0;
}
lf solve(){
	lf l=0,r=1;
	while(r-l>1e-9){
		lf mid=(l+r)/2;
		if(check(mid))l=mid; else r=mid;
	}
	return l;
}
```

## 任务规划 | Livshits-Kladov定理

- 给出 $n$ 个任务，第 $i$ 个任务花费 $t_i$ 时间，该任务开始之前等待 $t$ 时间的代价是 $f_i(t)$ 个数，求一个任务排列方式，最小化代价 $\sum\limits_{i=1}^n f_j(\sum\limits_{j=1}^{i-1}t_i)$
- Livshits-Kladov定理：当 $f_i(t)$ 是一次函数 / 指数函数 / 相同的单增函数时，最优解可以用排序计算
- 一次函数：$f_i(t)=c_it+d_i$，按 $\dfrac {c_i}{t_i}$ 升序排列
- 指数函数：$f_i(t)=c_ia^t+d_i$，按 $\dfrac{1-a^{t_i}}{c_i}$ 升序排列
- 相同的单增函数：按 $t_i$ 升序排序

## 分治

### 逆序数×二维偏序

- $O(n\log n)$

```c++
void merge(int l,int r){ //归并排序
	//对[l,r-1]的数排序
	if(r-l<=1)return;
	int mid=l+(r-l)/2;
	merge(l,mid);
	merge(mid,r);
	int p=l,q=mid,s=l;
	while(s<r){
		if(p>=mid || (q<r && a[p]>a[q])){
			t[s++]=a[q++];
			ans+=mid-p; //统计逆序数
		}
		else
			t[s++]=a[p++];
	}
	for(int i=l;i<r;++i)a[i]=t[i];
}
```

## 最大空矩阵 | 悬线法

- 求01矩阵中全是0的最大连续子矩阵（面积最大）$O(nm)$
- 此处障碍物是正方形。如果障碍只是一些整点，答案从 $ab$ 变为 $(a+1)(b+1)$

```c++
int n,m,a[N][N],l[N][N],r[N][N],u[N][N];
int getlm(){
	int ans=0;
	repeat(i,0,n)
	repeat(k,0,m)
		l[i][k]=r[i][k]=u[i][k]=(a[i][k]==0);
	repeat(i,0,n){
		repeat(k,1,m)
		if(a[i][k]==0)
			l[i][k]=l[i][k-1]+1; //可以向左延伸几格
		repeat_back(k,0,m-1)
		if(a[i][k]==0)
			r[i][k]=r[i][k+1]+1; //可以向右延伸几格
		repeat(k,0,m)
		if(a[i][k]==0){
			if(i!=0 && a[i-1][k]==0){
				u[i][k]=u[i-1][k]+1; //可以向上延伸几格
				l[i][k]=min(l[i][k],l[i-1][k]);
				r[i][k]=min(r[i][k],r[i-1][k]); //如果向上延伸u格，lr对应的修改
			}
			ans=max(ans,(l[i][k]+r[i][k]-1)*u[i][k]);
		}
	}
	return ans;
}
```

## 搜索

### 舞蹈链×DLX

<H4>精确覆盖</H4>

- 在01矩阵中找到某些行，它们两两不相交，且它们的并等于全集
- xy编号从 $1$ 开始！$O(\exp)$，节点数 $<5000$

```c++
int n,m;
vector<int> rec; //dance后存所有选中的行的编号
struct DLX{
	#define rep(i,i0,a) for(int i=a[i0];i!=i0;i=a[i])
	int u[N],d[N],l[N],r[N],x[N],y[N]; //N=10010
	int sz[N],h[N];
	int top;
	void init(){
		top=m;
		repeat(i,0,m+1){
			sz[i]=0; u[i]=d[i]=i;
			l[i]=i-1; r[i]=i+1;
		}
		l[0]=m; r[m]=0;
		repeat(i,0,n+1)h[i]=-1;
		rec.clear();
	}
	void add(int x0,int y0){
		top++; sz[y0]++;
		x[top]=x0; y[top]=y0;
		u[top]=u[y0]; d[top]=y0;
		u[d[top]]=d[u[top]]=top;
		if(h[x0]<0)
			h[x0]=l[top]=r[top]=top;
		else{
			l[top]=h[x0]; r[top]=r[h[x0]];
			l[r[h[x0]]]=top; r[h[x0]]=top;
		}
	}
	void remove(int c){
		l[r[c]]=l[c]; r[l[c]]=r[c];
		rep(i,c,d)rep(j,i,r){
			u[d[j]]=u[j]; d[u[j]]=d[j];
			sz[y[j]]--;
		}
	}
	void resume(int c){
		rep(i,c,d)rep(j,i,r){
			u[d[j]]=d[u[j]]=j;
			sz[y[j]]++;
		}
		l[r[c]]=r[l[c]]=c;
	}
	bool dance(int dep=1){ //返回是否可行
		if(r[0]==0)return 1;
		int c=r[0];
		rep(i,0,r)if(sz[c]>sz[i])c=i;
		remove(c);
		rep(i,c,d){
			rep(j,i,r)remove(y[j]);
			if(dance(dep+1)){rec.push_back(x[i]);return 1;}
			rep(j,i,l)resume(y[j]);
		}
		resume(c);
		return 0;
	}
}dlx;
```

<H4>重复覆盖</H4>

- 在01矩阵中找到最少的行，它们的并等于全集
- xy编号还是从 $1$ 开始！$O(\exp)$，节点数可能 $<3000$

```c++
struct DLX{
	#define rep(i,d,s) for(node* i=s->d;i!=s;i=i->d)
	struct node{
		node *l,*r,*u,*d;
		int x,y;
	};
	static const int M=2e5;
	node pool[M],*h[M],*R[M],*pl;
	int sz[M],vis[M],ans,clk;
	void init(int n,int m){ //行和列
		clk=0; ans=inf; pl=pool; ++m;
		repeat(i,0,max(n,m)+1)
			R[i]=sz[i]=0,vis[i]=-1;
		repeat(i,0,m)
			h[i]=new(pl++)node;
		repeat(i,0,m){
			h[i]->l=h[(i+m-1)%m];
			h[i]->r=h[(i+1)%m];
			h[i]->u=h[i]->d=h[i];
			h[i]->y=i;
		}
	}
	void link(int x,int y){
		sz[y]++;
		auto p=new(pl++)node;
		p->x=x; p->y=y;
		p->u=h[y]->u; p->d=h[y];
		p->d->u=p->u->d=p;
		if(!R[x])R[x]=p->l=p->r=p;
		else{
			p->l=R[x]; p->r=R[x]->r;
			p->l->r=p->r->l=p;
		}
	}
	void remove(node* p){
		rep(i,d,p)i->l->r=i->r,i->r->l=i->l;
	}
	void resume(node* p){
		rep(i,u,p)i->l->r=i->r->l=i;
	}
	int eval(){
		++clk; int ret=0;
		rep(i,r,h[0])
		if(vis[i->y]!=clk){
			++ret;
			vis[i->y]=clk;
			rep(j,d,i)rep(k,r,j)vis[k->y]=clk;
		}
		return ret;
	}
	void dfs(int d){
		if(h[0]->r==h[0]){ans=min(ans,d); return;}
		if(eval()+d>=ans)return;
		node* c; int m=inf;
		rep(i,r,h[0])
			if(sz[i->y]<m){m=sz[i->y]; c=i;}
		rep(i,d,c){
			remove(i); rep(j,r,i)remove(j);
			dfs(d+1);
			rep(j,l,i)resume(j); resume(i);
		}
	}
	int solve(){ //返回最优解
		ans=inf; dfs(0); return ans;
	}
}dlx;
```

### 启发式算法

<H4>A-star</H4>

- 定义 $g(v)$ 是 $s$ 到 $v$ 的实际代价，$h(v)$ 是 $v$ 到 $t$ 的估计代价
- 定义估价函数 $f(v)=g(v)+h(v)$
- 每次从堆里取出 $f(v)$ 最小的点进行更新
- 如果满足 $h(v_1)+w(v_2,v_1)\ge h(v_2)$ （存疑）则不需要重复更新同一点，可以用set标记，已标记的不入堆

<H4>模拟退火</H4>

- 以当前状态 $X$ 为中心，半径为温度 $T$ 的圆（或球）内选一个新状态 $Y$
- 计算 $D=E(Y)-E(X)$ 新状态势能减去当前状态势能
- 如果 $D<0$ 则状态转移（势能 $E$ 越小越优）
- 否则状态转移的概率是 $\exp(-\dfrac{KD}{T})$（Metropolis接受准则，~~学不会~~）
- 最后温度乘以降温系数，返回第一步
- 需要调 $3$ 个参数：初始温度，终止温度，降温系数（？）

注意点：

- 让运行时间在TLE边缘试探
- 多跑几次退火
- 多交几次（注意风险）
- 可以先不用某准则，输出中间过程后再调参

```c++
lf rndf(){return rnd()*1.0/rnd.max();}
vec rndvec(){return vec(rndf()*2-1,rndf()*2-1);}
//lf E(vec); //计算势能
struct state{ //表示一个状态
	vec v; lf e; //位置和势能
	state(vec v=vec()):v(v),e(E(v)){}
	operator lf(){return e;}
};
state getstate(){
	state X; lf T=1000;
	auto work=[&](){
		state Y=X.v+rndvec()*T;
		if(Y<X /*|| rndf()<exp(-K*(Y.e-X.e)/T)*/){X=Y; return 1;}
		return 0;
	};
	while(T>1e-9){
		if(work()){work(); work(); T*=1.1;}
		T*=0.99992;
	}
	return X;
}

void solve(){
	state X;
	repeat(i,0,6){
		state Y=getstate();
		if(X>Y)X=Y;
	}
	printf("%.10f\n",lf(X));
}
```

## 动态规划

### 多重背包

- 二进制版，$O(nV\log num)$，$V$ 是总容量

```c++
int n,V; ll dp[N];
void push(int val,int v,int c){ //处理物品(价值=val,体积=v,个数=c)
	for(int b=1;c;c-=b,b=min(b*2,c)){
		ll dv=b*v,dval=b*val;
		repeat_back(j,dv,V+1)
			dp[j]=max(dp[j],dp[j-dv]+dval);
	}
}
//初始化fill(dp,dp+V+1,0)，结果是dp[V]
```

- 单调队列版，$O(nV)$，$V$ 是总容量

```c++
int n,V; ll dp[N];
void push(int val,int v,int c){ //处理物品(价值=val,体积=v,个数=c)
	static deque< pair<int,ll> > q; //单调队列，fi是位置，se是价值
	if(v==0){
		repeat(i,0,V+1)dp[i]+=val*c;
		return;
	}
	c=min(c,V/v);
	repeat(d,0,v){
		q.clear();
		repeat(j,0,(V-d)/v+1){
			ll t=dp[d+j*v]-j*val;
			while(!q.empty() && t>=q.back().se)
				q.pop_back();
			q.push_back({j,t});
			while(q.front().fi<j-c)
				q.pop_front();
			dp[d+j*v]=max(dp[d+j*v],q.front().se+j*val);
		}
	}
}
//初始化fill(dp,dp+V+1,0)，结果是dp[V]
```

### 最长不降子序列×LIS

- 二分查找优化，$O(n\log n)$

```c++
const int inf=1e9;
repeat(i,0,n+1)dp[i]=inf; //初始化为inf
repeat(i,0,n)
	*lower_bound(dp,dp+n,a[i])=a[i];
return lower_bound(dp,dp+n,inf)-dp;
```

### 数位dp

- 记忆化搜索，dfs的参数lim表示是否被限制，lz表示当前位的前一位是不是前导零
- 复杂度等于状态数
- 如果每个方案贡献不是1，dp可能要变成struct数组(cnt,sum,....)

```c++
ll dp[20][*][2],bit[20]; //这个[2]表示lz状态，如果lz被使用了的话就需要记录
ll dfs(int pos,ll *,bool lim=1,bool lz=1){
	if(pos==-1)return *; //返回该状态是否符合要求(0或1)
	ll &x=dp[pos][*];
	if(!lim && x!=-1)return x;
	ll ans=0;
	int maxi=lim?bit[pos]:9;
	repeat(i,0,maxi+1){
		...//状态转移
		if(lz && i==0)...//可能要用lz，其他地方都不用
		ans+=dfs(pos-1,*,
			lim && i==maxi,
			lz && i==0);
	}
	if(!lim)x=ans; //不限制的时候才做存储
	return ans;
}
ll solve(ll n){
	int len=0;
	while(n)bit[len++]=n%10,n/=10;
	return dfs(len-1,*);
}
signed main(){
	mst(dp,-1); //在很多时候dp值可以反复使用
	ll t=read();
	while(t--){
		ll l=read(),r=read();
		printf("%lld\n",solve(r)-solve(l-1));
	}
	return 0;
}
```

### 换根dp

- 两次dfs
- 第一次求所有点所在子树的答案 $dp_v$，此时 $dp_{rt}$ 是 $rt$ 的最终答案
- 第二次将根转移来算其他点的最终答案，回溯时复原即可

```c++
void dfs1(int x,int fa=-1){
	for(auto p:a[x])
	if(p!=fa){
		dfs1(p,x);
		dp[x]+=op(dp[p]);
	}
}
void dfs2(int x,int fa=-1){
	ans[x]=dp[x];
	for(auto p:a[x])
	if(p!=fa){
		dp[x]-=op(dp[p]);
		dp[p]+=op(dp[x]);
		dfs2(p,x);
		dp[p]-=op(dp[x]);
		dp[x]+=op(dp[p]);
	}
}
```

### 斜率优化

- 例：HDOJ3507
- $dp_i=\min\limits_{j=0}^{i-1}[dp_j+(s_i-s_j)^2+M]$
- 考虑 $k<j<i$，$j$ 决策优于 $k\Leftrightarrow dp_j+(s_i-s_j)^2<dp_k+(s_i-s_k)^2$
- 一通操作后 $\dfrac{s_j^2+dp_j-s_k^2-dp_k}{2(s_j-s_k)}<s_i$，即寻找点集 $\{(2s_j,s_j^2+dp_j)\}$ 的下凸包中斜率刚好 $>s_i$ 的线段的左端点，作为决策点

### 四边形优化

- $dp(l,r)=\min\limits_{k=l}^{r-1}[dp(l,k)+dp(k+1,r)]+w(l,r)$
- 其中 $w(l,r)$ 满足
	- 区间包含单调性：任意 $l \le l' \le r' \le r$ 有 $w(l',r')\le w(l,r)$
	- 四边形不等式：任意 $a \le b \le c \le d$ 有 $w(a,c)+w(b,d)\le w(a,d)+w(b,c)$（若等号恒成立则满足四边形恒等式）
- 决策单调性：令 $m(l,r)$ 为最优决策点（满足 $dp(l,r)=dp(l,m)+dp(m+1,r)+w(l,r)$），则有 $m(l,r-1) \le m(l,r) \le m(l+1,r)$，遍历这个区间可以优化至 $O(n^2)$

```c++
repeat(i,0,n)dp[i][i]=0,m[i][i]=i;
repeat(len,2,n+1)
for(int l=0,r=len-1;r<n;l++,r++){
	dp[l][r]=inf;
	repeat(k,m[l][r-1],min(m[l+1][r]+1,r))
	if(dp[l][r]>dp[l][k]+dp[k+1][r]+w(l,r)){
		dp[l][r]=dp[l][k]+dp[k+1][r]+w(l,r);
		m[l][r]=k;
	}
}
```

- $dp(i)=\min\limits_{k=1}^{i-1}w(k,i)$，$w(l,r)$ 满足四边形不等式
- 决策单调性：令 $m_i$ 为最优决策点（满足 $dp(i)=w(m,i)$），则 $m_{i-1}\le m_i$，因此可以分治优化成 $O(n\log n)$

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

- $O(n\log n)$，可能有锅

```c++
lf ans;
bool cmp_y(vec a,vec b){return a.y<b.y;}
void rec(int l,int r){ //左闭右开区间
	#define upd(x,y) {ans=min(ans,(x-y).len());}
	if(r-l<4){
		repeat(i,l,r)
		repeat(j,i+1,r)
			upd(a[i],a[j]);
		sort(a+l,a+r,cmp_y); //按y排序
		return;
	}
	int m=(l+r)/2;
	lf midx=a[m].x;
	rec(l,m),rec(m,r);
	static vec b[N];
	merge(a+l,a+m,a+m,a+r,b+l,cmp_y); //逐渐按y排序
	copy(b+l,b+r,a+l);
	int t=0;
	repeat(i,l,r)
	if(abs(a[i].x-midx)<ans){
		repeat_back(j,0,t){
			if(a[i].y-b[i].y>ans)break;
			upd(a[i],b[j]);
		}
		b[t++]=a[i];
	}
}
lf nearest(){
	ans=1e20;
	sort(a,a+n); //按x排序
	rec(0,n);
	return ans;
}
```

## 最小圆覆盖 | 随机增量法×RIA

- eps可能要非常小。随机化，均摊 $O(n)$

```c++
struct cir{ //圆（结构体）
	vec v; lf r;
	bool out(vec a){ //点a在圆外
		return (v-a).len()>r+eps;
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

<H3>公式</H3>

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
	void add(ll x,ll k){ //位置x加上k
		//x++;
		for(;x<N;x+=lb(x))
			t[x]+=k;
	}
	ll sum(ll x){ //求[1,x]的和 //[0,x]
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
		x=max(x,l); y=min(y,r); if(x>y){down();return;}
		if(x==l && y==r){toz(k); down(); return;}
		down();
		lc->update(x,y,k);
		rc->update(x,y,k);
		up();
	}
	ll query(int x,int y){
		x=max(x,l); y=min(y,r); if(x>y)return a0;
		down();
		if(x==l && y==r)return a;
		return U(lc->query(x,y),rc->query(x,y));
	}
}tr[N*2],*pl;
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
		for(n=1;n<inn;n<<=1); repeat(i,inn,n)in[i]=a0;
		repeat(i,0,n)a[n+i]=in[i];
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
	int fa(int x){
		return a[x]==x?x:a[x]=fa(a[x]);
	}
	int &operator[](int x){
		return a[fa(x)];
	}
}d;
```

- 普通版，路径压缩+启发式合并

```c++
struct DSU{
	int a[10010],sz[10010];
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
}d;
```

### <补充>种类并查集

```c++
struct DSU{
	int a[50010],r[50010];
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
	int fa(int x){ //返回根节点
		if(a[x]==x)return x;
		int f=a[x],ff=fa(f);
		r[x]=plus(r[x],r[f]);
		return a[x]=ff;
	}
	bool query(int x,int y){ //是否存在关系
		return fa(x)==fa(y);
	}
	int getr(int x,int y){ //查找关系
		return plus(r[x],inv(r[y]));
	}
	void join(int x,int y,int r2){ //按r2关系合并
		r2=plus(plus(inv(r[x]),r2),r[y]);
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

- 万年不用，$O(?)$
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
	int rt,cur; //rt根节点
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
	int build(int l,int r){ //以序列g[l..r]为模板重建树，返回根节点
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

<H3>普通莫队</H3>

- 移动指针 $l,r$ 来求所有区间的答案
- 块大小为 $\sqrt n$，$O(n^{\tfrac 3 2})$

```c++
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
int unit,n,bkt[N],a[N],final_ans[N]; //bkt是桶
ll ans;
void update(int x,int d){
	int &b=bkt[a[x]];
	ans-=C(b,2); //操作示例
	b+=d;
	ans+=C(b,2); //操作示例
}
void solve(){ //final_ans[]即最终答案
	fill(bkt,bkt+n+1,0);
	unit=int(ceil(sqrt(n)));
	sort(query.begin(),query.end());
	int l=1,r=0; ans=0; //如果原数组a编号从1开始
	for(auto i:query){
		while(l<i.l)update(l++,-1);
		while(l>i.l)update(--l,1);
		while(r<i.r)update(++r,1);
		while(r>i.r)update(r--,-1);
		final_ans[i.id]=ans;
	}
}
//repeat(i,0,m)query.push_back({read(),read(),i}); //输入查询区间
```

<H3>带修莫队</H3>

- 相比与普通莫队，多了一个时间轴
- 块大小为 $\sqrt[3]{nt}$，$O(\sqrt[3]{n^4t})$
- 空缺

## 二叉搜索树

### 不平衡的二叉搜索树

- 左子树所有结点 $\le v <$ 右子树所有节点，目前仅支持插入，查询可以写一个 `map<int,TR *>`

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
//inf是无效节点，用tr->ch[0]来访问根节点
```

### 无旋treap

- 普通平衡树按v分裂，文艺平衡树按sz分裂
- insert,erase操作在普通平衡树中，push_back,output(dfs)在文艺平衡树中
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
	}
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

支持插入、查询中位数可以用双堆

区间众数：离线用莫队，在线用分块

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
	void init(){a[0].clear();a[1].clear();}
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

# 图论

## 图论的一些概念

***

- 基环图：树加一条边
- 简单图：不含重边和自环
- 完全图：顶点两两相连的无向图
- 竞赛图：顶点两两相连的有向图
- 点u到v可达：有向图中，存在u到v的路径
- 点u和v联通：无向图中，存在u到v的路径
- 生成子图：点集和原图相同
- 导出子图/诱导子图：选取一个点集，尽可能多加边
- 正则图：所有点的度均相同的无向图

***

+ 强正则图：与任意两个相邻的点相邻的点数相同，与任意两个不相邻的点相邻的点数相同的正则图
+ 强正则图的点数 $v$，度 $k$，相邻的点的共度 $\lambda$，不相邻的点的共度 $\mu$ 有 $k(k-1-\lambda)=\mu(v-1-k)$
+ 强正则图的例子：所有完全图、所有nk顶点满n分图

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

+ 哈密顿通路：通过所有顶点有且仅有一次的路径，若存在则为半哈密顿图/哈密顿图
+ 哈密顿回路：通过所有顶点有且仅有一次的回路，若存在则为哈密顿图
+ 完全图 $K_{2k+1}$ 的边集可以划分为 $k$ 个哈密顿回路
+ 完全图 $K_{2k}$ 的边集去掉 $k$ 条互不相邻的边后可以划分为 $k-1$ 个哈密顿回路

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

## 最小生成树×MST

### Kruskal

- 对边长排序，然后添边，并查集判联通，$O(E\log E)$，排序是瓶颈

```c++
DSU d;
struct edge{int u,v,dis;}e[200010];
ll kru(){
	ll ans=0,cnt=0;
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

## 树论

### 树的直径

- 直径：即最长路径
- 求直径：以任意一点出发所能达到的最远节点为一个端点，以这个端点出发所能达到的最远节点为另一个端点（也可以树上dp）

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
	if(isroot && ch>=2) //根节点判断方法
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

+ 最大势算法MCS求完美消除序列：每次求出与 $[v_{i+1},...,v_n]$ 相邻点数最大的点作为 $v_i$
+ `e[][]`点编号从 $1$ 开始！`rec` 下标从 $1$ 开始！桶优化，$O(V+E)$

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
- 仙人掌的判定（树上差分）编号从哪开始都可以，$O(n+m)$

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

+ 霍尔定理：最大匹配 = 左顶点数 $\Leftrightarrow$ 所有左顶点子集 $S$ 都有 $|S|\le|\omega(S)|$ ，$\omega(S)$ 是 $S$ 的领域
+ 运用：若在最大匹配中有 $t$ 个左顶点失配，因此最大匹配 = 左顶点数 - $t$
+ 对任意左顶点子集 $S$ 都有 $|S|\le|\omega(S)|+t$，$t\ge|S|-|\omega(S)|$ ，求右式最大值即可求最大匹配

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

+ 费用流（最小费用最大流）：保证最大流后的最小费用

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
struct FLOW{ //MCMF费用流
	struct edge{int to,w,cost,nxt;}; //指向，限流，费用，下一条边
	vector<edge> a; int head[N]; //前向星
	int n,s,t,totcost; //点数，源点，汇点，总费用
	deque<int> q;
	bool inque[N]; //在队里的不需要入队
	int dis[N]; //费用
	struct{int to,e;}pre[N]; //路径的前一个点，这条边的位置
	void ae(int x,int y,int w,int cost){
		a.push_back((edge){y,w,cost,head[x]});
		head[x]=a.size()-1;
	}
	bool spfa(){ //已死的算法
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
						//松弛，或者直接q.push_back(p);
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
	n--,m=n; //a[n-1][n-1]的余子式（选任一节点均可）
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
	repeat(i,s,n) //s是根节点
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

# 字符串

- （~~我字符串是最菜的~~）
- 寻找模式串p在文本串t中的所有出现

## 哈希×Hash

### 字符串哈希

- 如果不需要区间信息，可以调用 `hash<string>()(s)` 获得ull范围的hash值
- 碰撞概率：单哈希 $10^6$ 次比较大约有 $\dfrac 1 {1000}$ 概率碰撞
- 支持查询子串hash值，初始化 $O(n)$，子串查询 $O(1)$

```c++
const int hashxor=rnd()%1000000000; //如果不是cf可以不用hashxor
struct Hash{
	vector<ll> a[2],p[2];
	const ll b=257,m[2]={1000000007,998244353};
	Hash(){repeat(i,0,2)a[i]={0},p[i]={1};}
	void push(const string &s){
		repeat(i,0,2)
		for(auto c:s){
			a[i]+=(a[i].back()*b+(c^hashxor))%m[i];
			p[i]+=p[i].back()*b%m[i];
		}
	}
	pair<ll,ll> get(int l,int r){
		#define q(i) (a[i][r+1]-a[i][l]*p[i][r-l+1]%m[i]+m[i])%m[i]
		return {q(0),q(1)};
		#undef q
	}
	int size(){return a[0].size()-1;}
	pair<ll,ll> prefix(int len){return get(0,len-1);}
	pair<ll,ll> suffix(int len){return get(size()-len,size()-1);}
}h;
```

### 质因数哈希

```c++
int fac(int n,int c,int mod,const function<int(int)> &f){
	int p=c*c%mod,ans=0;
	for(int i=2;i*i<=n;i++){
		int cnt=0;
		while(n%i==0)n/=i,cnt++;
		ans=(ans+p*f(cnt))%mod;
		p=p*c%mod;
	}
	if(n>1)ans=(ans+qpow(c,n,mod)*f(1))%mod;
	return ans;
}
//例：匹配乘积为x^k（x任意）的两个数
pii hash1(int n){
	return pii(
		fac(n,101,2147483647,[](int x){return x%k;}),
		fac(n,103,1000000007,[](int x){return x%k;})
	);
}
pii hash2(int n){
	return pii(
		fac(n,101,2147483647,[](int x){return (k-x%k)%k;}),
		fac(n,103,1000000007,[](int x){return (k-x%k)%k;})
	);
}
```

## 字符串函数

### 前缀函数×kmp

- $p[x]$ 表示满足 `s.substr(0,k)==s.substr(x-k,k)` 且 $x\not=k$ 的 $k$ 的最大值，$p[0]=0$
- 线性复杂度

```c++
int p[N];
void kmp(const string &s){ //求s的前缀函数
	p[0]=0; int k=0;
	repeat(i,1,s.length()){
		while(k>0 && s[i]!=s[k])k=p[k-1];
		if(s[i]==s[k])k++;
		p[i]=k;
	}
}
void solve(string s1,string s2){ //模拟s1.find(s2)
	kmp(s2+'#'+s1);
	repeat(i,s2.size()+1,s.size())
	if(p[i]==(int)s2.size())
		ans.push_back(i-2*s2.size()); //编号从0开始的左端点
}
```

```c++
struct KMP{ //kmp自动机
	string s; int k;
	vector<int> p;
	int get(char c){
		while(k>0 && c!=s[k])k=p[k-1];
		if(c==s[k])k++;
		return k;
	}
	KMP(const string &_s){
		p.push_back(k=0);
		s=_s+'#'; repeat(i,1,s.size())p.push_back(get(s[i]));
	}
	int size(){return s.size()-1;}
};
void solve(string s1,string s2){ //模拟s1.find(s2)
	KMP kmp(s2);
	repeat(i,0,s1.size())
	if(kmp.get(s1[i])==kmp.size())
		ans.push_back(i+1-kmp.size()); //编号从0开始的左端点
	kmp.k=0; //清空（如果下次还要用的话）
}
```

### z函数×exkmp

- $z[x]$ 表示满足 `s.substr(0,k)==s.substr(x,k)` 的 $k$ 的最大值，$z[0]=0$
- 线性复杂度

```c++
int z[N];
void exkmp(const string &s){ //求s的z函数
	fill(z,z+s.size(),0); int l=0,r=0;
	repeat(i,1,s.size()){
		if(i<=r)z[i]=min(r-i+1,z[i-l]);
		while(i+z[i]<(int)s.size() && s[z[i]]==s[i+z[i]])z[i]++;
		if(i+z[i]-1>r)l=i,r=i+z[i]-1;
	}
}
```

### 马拉车×Manacher

- 预处理为 `"#*A*A*A*A*A*"`
- 线性复杂度

```c++
int len[N*2]; char s[N*2]; //两倍内存
int manacher(char s1[]){ //s1可以是s
	int n=strlen(s1)*2+1;
	repeat_back(i,0,n)s[i+1]=(i%2==0?'*':s1[i/2]);
	n++; s[0]='#'; s[n++]=0;
	len[0]=0;
	int mx=0,id=0,ans=0;
	repeat(i,1,n-1){
		if(i<mx)len[i]=min(mx-i,len[2*id-i]);
		else len[i]=1;
		while(s[i-len[i]]==s[i+len[i]])len[i]++;
		if(len[i]+i>mx)mx=len[i]+i,id=i;
		ans=max(ans,len[i]-1); //最长回文串长度
	}
	return ans;
}
```

### 最小表示法

- 求 $s$ 重复无数次的字符串最小后缀的左端点
- 线性复杂度

```c++
int minstr(const string &s){
	int k=0,i=0,j=1,n=s.size();
	while(max(k,max(i,j))<n){
		if(s[(i+k)%n]==s[(j+k)%n])k++;
		else{
			s[(i+k)%n]>s[(j+k)%n]?i+=k+1:j+=k+1;
			if(i==j)i++;
			k=0;
		}
	}
	return min(i,j);
}
```

### 后缀数组×SA

- $sa[i]$ 表示所有后缀中第 $i$ 小的后缀是 `s.substr(sa[i],-1)`
- $rk[i]$ 表示所有后缀中 `s.substr(i,-1)` 是第 $rk[i]$ 小
- 编号从 $1$ 开始！$O(n\log n)$

```c++
int sa[N],rk[N]; //sa,rk即结果
void get_sa(const string &S){
	static int pre[N*2],id[N],px[N],cnt[N];
	int n=S.length(),m=256;
	const char *const s=S.c_str()-1; //为了编号从1开始
	for(int i=1;i<=n;i++)cnt[rk[i]=s[i]]++;
	for(int i=1;i<=m;i++)cnt[i]+=cnt[i-1];
	for(int i=n;i>=1;i--)sa[cnt[rk[i]]--]=i;
	for(int w=1;w<n;w<<=1){
		int t=0;
		for(int i=n;i>n-w;i--)id[++t]=i;
		for(int i=1;i<=n;i++)
			if(sa[i]>w)id[++t]=sa[i]-w;
		mst(cnt,0);
		for(int i=1;i<=n;i++)cnt[px[i]=rk[id[i]]]++;
		for(int i=1;i<=m;i++)cnt[i]+=cnt[i-1];
		for(int i=n;i>=1;i--)sa[cnt[px[i]]--]=id[i];
		memcpy(pre,rk,sizeof(rk));
		int p=0;
		static auto pp=[&](int x){return pii(pre[x],pre[x+w]);};
		for(int i=1;i<=n;i++)
			rk[sa[i]]=pp(sa[i])==pp(sa[i-1])?p:++p;
		m=p; //优化计数排序值域
	}
}
```

- sa可以在同一文本串中在线多次查找模式串（二分查找）

### height数组

- 定义 $lcp(i,j)=$ 后缀 $i$ 和后缀 $j$ 的最长公共前缀长度
- 定义 $height[i]=lcp(sa[i],sa[i-1])$，$height[1]=0$
- $height[rk[i]]\ge height[rk[i-1]]-1$
- 编号从 $1$ 开始，$O(n)$

```c++
for(int i=1,k=0;i<=n;i++){
	if(k)k--;
	while(s[i+k]==s[sa[rk[i]-1]+k])k++;
	ht[rk[i]]=k;
}
```

- 不相等的子串个数为 $\dfrac{n(n+1)}{2}-\sum\limits_{i=2}^{n}height[i]$

## 自动机

### 字典树×Trie

- 线性复杂度

```c++
struct trie{
	int a[N][26],cnt[N],t;
	void init(){
		t=0; add();
	}
	int add(){
		mst(a[t],0);
		cnt[t]=0;
		return t++;
	}
	void insert(const char s[]){
		int k=0;
		for(int i=0;s[i];i++){
			int c=s[i]-'a'; //小写字母
			if(!a[k][c])a[k][c]=add();
			k=a[k][c];
			//son[k]++; //如果要记录子树大小
		}
		cnt[k]++;
	}
	int query(const char s[]){
		int k=0;
		for(int i=0;s[i];i++){
			int c=s[i]-'a'; //小写字母
			if(!a[k][c])return 0;
			k=a[k][c];
		}
		return cnt[k];
	}
}t;
```

### AC自动机

- 先构建字典树，再构建fail树和字典图
- 线性复杂度

```c++
struct AC{
	static const int sigma=26,c0='a'; //小写字母
	struct node{
		int to[sigma],fail,trie_cnt,cnt;
		int &operator[](int x){return to[x];}
		//vector<int> p; //指向模式串集合
	}a[N];
	int t;
	vector<int> q; //存了bfs序
	void init(){t=0; a[t++]=node(); q.clear();}
	void insert(const char s[]/*,int ptr*/){ //将模式串插入字典树
		int k=0;
		for(int i=0;s[i];i++){
			int c=s[i]-c0;
			if(!a[k][c])a[k][c]=t,a[t++]=node();
			k=a[k][c];
		}
		a[k].trie_cnt++;
		//a[k].p.push_back(ptr);
	}
	void build(){ //构建fail树，将字典树扩展为图
		int tail=0;
		repeat(i,0,sigma)
		if(a[0][i])
			q.push_back(a[0][i]);
		while(tail!=(int)q.size()){
			int k=q[tail++];
			repeat(i,0,sigma)
			if(a[k][i])
				a[a[k][i]].fail=a[a[k].fail][i],q.push_back(a[k][i]);
			else
				a[k][i]=a[a[k].fail][i];
		}
	}
	void query(const char s[]){ //记录文本串中模式串出现次数
		int k=0;
		for(int i=0;s[i];i++){
			int c=s[i]-c0;
			k=a[k][c];
			a[k].cnt++; //fail树上差分
			//for(int kk=k;kk!=0;kk=a[kk].fail)
			//for(auto p:a[kk].p)
				//ans[p]++; //代替下面的反向遍历，我也不知道什么时候用
		}
		repeat_back(i,0,q.size()){ //反向遍历bfs序
			a[a[q[i]].fail].cnt+=a[q[i]].cnt; //差分求和
			//for(auto p:a[q[i]].p)ans[p]=a[q[i]].cnt; //反馈答案
		}
	}
}ac;
```

### 后缀自动机×SAM

- 给定字符串中所有子串对应了SAM中从源点出发的一条路径
- SAM是一个DAG，至多有 $2n-1$ 个结点和 $3n-4$ 条边
- 每个结点表示一个endpos等价类，对应子串长度区间 $[a[a[i].fa].len+1,a[i].len]$
- 构造 $O(n)$，编号从 $1$ 开始（$a[1]$ 表示源点）

```c++
struct SAM{
	static const int sigma=26,c0='a'; //小写字母
	struct node{
		int to[sigma],fa,len;
		int &operator[](int x){return to[x];}
	}a[N*2];
	int last,tot;
	void init(){last=tot=1;}
	int add(){a[++tot]=node(); return tot;}
	void push(int c){
		c-=c0;
		int x=last,nx=last=add();
		a[nx].len=a[x].len+1;
		for(;x && a[x][c]==0;x=a[x].fa)a[x][c]=nx;
		if(x==0)a[nx].fa=1;
		else{
			int p=a[x][c];
			if(a[p].len==a[x].len+1)a[nx].fa=p;
			else{
				int np=add();
				a[np]=a[p]; a[np].len=a[x].len+1;
				a[p].fa=a[nx].fa=np;
				for(;x && a[x][c]==p;x=a[x].fa)a[x][c]=np;
			}
		}
	}
}sam;
//构造：for(auto i:s)sam.push(i);
```

# 杂项

## 位运算

<H3>位运算巨佬操作</H3>

- 中点向下取整 `(x+y)/2: (x & y) + ((x ^ y) >> 1)`
- 中点向上取整 `(x+y+1)/2: (x | y) - ((x ^ y) >> 1)`
	- 一般来说用 `x + (y - x >> 1)`
- `abs(n): (n ^ (n >> 31)) - (n >> 31)`
- `max(a,b): b & ((a - b) >> 31) | a & (~(a - b) >> 31)`
- `min(a,b): a & ((a - b) >> 31) | b & (~(a - b) >> 31)`

```c++
#define B(x,i) ((x>>i)&1) //返回x的第i位
#define Bswap(x,i,j) (B(x,i)^B(x,j)) && (x^=(1ll<<i)|(1ll<<j)) //交换x的i位和j位
#define Bset(x,i,b) (B(x,i)^b) && (x^=(1ll<<i)) //将x的第i位赋值为b
```

### 位运算函数

- （不需要头文件）

```c++
__builtin_ctz(x),__builtin_ctzll(x) //返回x后导0的个数，x是0则返回32 or 64
__builtin_clz(x),__builtin_clzll(x) //返回x前导0的个数，x是0则返回32 or 64
__builtin_popcount(x),__builtin_popcountll(x) //返回x中1的个数
__builtin_parity(x),__builtin_parityll(x) //返回x中1的个数是否为奇数
```

### 枚举二进制子集

- 枚举二进制数m的非空子集

```c++
for(int s=m;s;s=(s-1)&m){
	work(s);
}
```

- 枚举n个元素的大小为k的二进制子集（要保证k不等于0）

```c++
int s=(1<<k)-1;
while(s<(1<<n)){
	work(s);
	int x=s&-s,y=s+x;
	s=((s&~y)/x>>1)|y; //这里有一个位反~
}
```

## 浮点数

<H3>浮点数操作</H3>

```c++
const lf eps=1e-11;
if(abs(x)<eps)x=abs(x); //输出浮点数的预处理
```

<H3>浮点数常量</H3>

```c++
float 1e38, 有效数字6
double 1e308, 有效数字15
long double 1e4932, 有效数字18
```

## 常数优化

<H3>估计函数用时</H3>

- `clock()` 可以获取时刻，单位毫秒，运行函数前后的时间之差即为用时
- 一些巨佬测出来的结论：

```
- 整数加减：1（个时间单位，下同）
- 整数位运算：1
- 整数乘法：2
- 整数除法：21
- 浮点加减：3
- 浮点除法：35
- 浮点开根：60
```

### 快读快写

```c++
ll read(){
	ll x=0,tag=1; char c=getchar();
	for(;!isdigit(c);c=getchar())if(c=='-')tag=-1;
	for(; isdigit(c);c=getchar())x=x*10+c-48;
	return x*tag;
}
void write(ll x){ //可能不比printf快
	if(x<0)x=-x,putchar('-');
	if(x>=10)write(x/10);
	putchar(x%10^48);
}
```

```c++
char getc(){ //代替getchar，用了这个就不能用其他读入函数如scanf
    static char now[1<<16],*S,*T;
    if(T==S){T=(S=now)+fread(now,1,1<<16,stdin); if(T==S)return EOF;}
	return *S++;
}
```

### STL手写内存分配器

```c++
static char space[10000000],*sp=space;
template<typename T>
struct allc:allocator<T>{
	allc(){}
	template<typename U>
	allc(const allc<U> &a){}
	template<typename U>
	allc<T>& operator=(const allc<U> &a){return *this;}
	template<typename U>
	struct rebind{typedef allc<U> other;};
	T* allocate(size_t n){
		T *res=(T*)sp;
		sp+=n*sizeof(T);
		return res;
	}
	void deallocate(T* p,size_t n){}
};
vector< int,allc<int> > a;
```

<H3>吸氧气</H3>

```c++
#pragma GCC optimize(2) //(3),("Ofast")
```

<H3>其他优化</H3>

```
//（都是听说的）
1ll*a 比 (ll)a 快
取模：x%mod 优化为 x<mod?x:x%mod
减少浮点除法：a/b+c/d 优化为 (a*d+b*c)/(b*d)
精度足够时用ll代替浮点类型
多路并行运算，如 (a+b)+(c+d) 比 ((a+b)+c)+d 快
加上inline，以及强制内联__inline __attribute__((always_inline))
多重for循环时，修改for的顺序保证内存连续访问
多使用局部变量
```

## 在TLE边缘试探

```c++
while(clock()<0.9*CLOCKS_PER_SEC){
	//反复更新最优解
}
```

## 对拍

```c++
#include<bits/stdc++.h>
using namespace std;
int main(){
	for(int i=0;;i++){
		if(i%10==0)cerr<<i<<endl;
		system("gen.exe > test.in");
		system("test1.exe < test.in > a.out");
		system("test2.exe < test.in > b.out");
		if(system("fc a.out b.out")){
			system("pause");
			return 0;
		}
	}
}
```

备选

```c++
#include<bits/stdc++.h>
using namespace std;
ifstream a,b;
int main(){
	for(int i=0;;i++){
		if(i%10==0)cerr<<i<<endl;
		system("datamaker.exe > data.txt");
		system("A.exe < data.txt > a.out");
		system("B.exe < data.txt > b.out");
		a.open("a.out");
		b.open("b.out");
		while(a.good() || b.good()){
			if(a.get()!=b.get()){
				system("pause");
				return 0;
			}
		}
		a.close(),b.close();
	}
}
```

## 战术分析 坑点

（~~我真的真的真的太南了~~）

```
ll t; 1<<t返回int，必须是1ll<<t
int x; x<<y的y会先对32取模
operator<的比较内容一定要写完整
试一试输入^Z能否结束
无向图输入要给两个值赋值g[x][y]=g[x][y]=1
多组输入时，图记得初始化
建模的转换函数的宏定义一定要加括号，或者写成函数
多想想极端数据！！
islower()等函数返回值不一定是0或1
多用相空间角度思考问题
内存比我想象的要大一些（有时候1e7可以塞下）
在64位编译器（我的编译器）中set每个元素需要额外32字节内存
struct里放大数组，最好用vector代替
deque占用很大很大的内存
```

# 数学

## 数论

### 基本操作

#### 模乘 模幂 模逆 扩欧

```c++
ll mul(ll a,ll b,ll m=mod){return a*b%m;} //模乘
ll qpow(ll a,ll b,ll m=mod){ //快速幂
	ll ans=1;
	for(;b;a=mul(a,a,m),b>>=1)
		if(b&1)ans=mul(ans,a,m);
	return ans;
}
void exgcd(ll a,ll b,ll &d,ll &x,ll &y){ //ax+by=gcd(a,b), d=gcd
	if(!b)d=a,x=1,y=0;
	else exgcd(b,a%b,d,y,x),y-=x*(a/b);
}
ll gcdinv(ll v,ll m=mod){ //扩欧版逆元
	ll d,x,y;
	exgcd(v,m,d,x,y);
	return (x%m+m)%m;
}
ll getinv(ll v,ll m=mod){ //快速幂版逆元，m必须是质数!!
	return qpow(v,m-2,m);
}
ll qpows(ll a,ll b,ll m=mod){
	if(b>=0)return qpow(a,b,m);
	else return getinv(qpow(a,-b,m),m);
}
```

#### 阶乘 组合数

- $O(n)$ 初始化，$O(1)$ 查询

```c++
struct CC{
	static const int N=100010;
	ll fac[N],inv[N];
	CC(){
		fac[0]=1;
		repeat(i,1,N)fac[i]=fac[i-1]*i%mod;
		inv[N-1]=qpow(fac[N-1],mod-2,mod);
		repeat_back(i,1,N)inv[i-1]=inv[i]*i%mod;
	}
	ll operator()(ll a,ll b){ //a>=b
		if(a<b || b<0)return 0;
		return fac[a]*inv[a-b]%mod*inv[b]%mod;
	}
	ll A(ll a,ll b){ //a>=b
		if(a<b || b<0)return 0;
		return fac[a]*inv[a-b]%mod;
	}
}C;
```

#### 防爆模乘

```c++
//int128版本
ll mul(ll a,ll b,ll m=mod){return (__int128)a*b%m;}
//long double版本（欲防爆，先自爆）（注意在测试的时候不知道为什么有锅）
ll mul(ll a,ll b,ll m=mod){return (a*b-ll((long double)a/m*b)*m+m)%m;}
//每位运算一次版本，注意这是真·龟速乘，O(logn)
ll mul(ll a,ll b,ll m=mod){
	ll ans=0;
	while(b){
		if(b&1)ans=(ans+a)%m;
		a=(a+a)%m;
		b>>=1;
	}
	return ans;
}
//把b分成两部分版本，要保证m小于1<<42（约等于4e12），a,b<m
ll mul(ll a,ll b,ll m=mod){
	a%=m,b%=m;
	ll l=a*(b>>21)%m*(1ll<<21)%m;
	ll r=a*(b&(1ll<<21)-1)%m;
	return (l+r)%m;
}
```

#### 最大公约数

```c++
__gcd(a,b) //内置gcd，推荐
ll gcd(ll a,ll b){return b==0?a:gcd(b,a%b);} //不推荐233，比内置gcd慢
ll gcd(ll a,ll b){ //卡常gcd来了！！
	#define tz __builtin_ctzll
	if(!a || !b)return a|b;
	int t=tz(a|b);
	a>>=tz(a);
	while(b){
		b>>=tz(b);
		if(a>b)swap(a,b);
		b-=a;
	}
	return a<<t;
	#undef tz
}
```

- 实数gcd

```c++
lf fgcd(lf a,lf b){return abs(b)<1e-5?a:fgcd(b,fmod(a,b));}
```

### 高级模操作

#### 同余方程组 | CRT+extra

```c++
//CRT，m[i]两两互质
ll crt(ll a[],ll m[],int n){ //ans%m[i]==a[i]
	repeat(i,0,n)a[i]%=m[i];
	ll M=1,ans=0;
	repeat(i,0,n)
		M*=m[i];
	repeat(i,0,n){
		ll k=M/m[i],t=gcdinv(k%m[i],m[i]); //扩欧!!
		ans=(ans+a[i]*k*t)%M; //两个乘号可能都要mul
	}
	return (ans+M)%M;
}
//exCRT，m[i]不需要两两互质，基于扩欧exgcd和龟速乘mul
ll excrt(ll a[],ll m[],int n){ //ans%m[i]==a[i]
	repeat(i,0,n)a[i]%=m[i]; //根据情况做适当修改
	ll M=m[0],ans=a[0],g,x,y; //M是m[0..i]的最小公倍数
	repeat(i,1,n){
		ll c=((a[i]-ans)%m[i]+m[i])%m[i];
		exgcd(M,m[i],g,x,y); //Ax=c(mod B)
		if(c%g)return -1;
		ans+=mul(x,c/g,m[i]/g)*M; //龟速乘
		M*=m[i]/g;
		ans=(ans%M+M)%M;
	}
	return (ans+M)%M;
}
```

#### 离散对数 | BSGS+extra

- 求 $a^x \equiv b \pmod m$ ，$O(\sqrt m)$

```c++
//BSGS，a和mod互质
ll bsgs(ll a,ll b,ll mod){ //a^ans%mod==b
	a%=mod,b%=mod;
	static unordered_map<ll,ll> m; m.clear();
	ll t=(ll)sqrt(mod)+1,p=1;
	repeat(i,0,t){
		m[mul(b,p,mod)]=i; //p==a^i
		p=mul(p,a,mod);
	}
	a=p; p=1;
	repeat(i,0,t+1){
		if(m.count(p)){ //p==a^i
			ll ans=t*i-m[p];
			if(ans>0)return ans;
		}
		p=mul(p,a,mod);
	}
	return -1;
}
//exBSGS，a和mod不需要互质，基于BSGS
ll exbsgs(ll a,ll b,ll mod){ //a^ans%mod==b
	a%=mod,b%=mod;
	if(b==1)return 0;
	ll ans=0,c=1,g;
	while((g=__gcd(a,mod))!=1){
		if(b%g!=0)return -1;
		b/=g,mod/=g;
		c=mul(c,a/g,mod);
		ans++;
		if(b==c)return ans;
	}
	ll t=bsgs(a,mul(b,getinv(c,mod),mod),mod); //必须扩欧逆元!!
	if(t==-1)return -1;
	return t+ans;
}
```

#### 阶与原根

- 判断是否有原根：若 $m$ 有原根，则 $m$ 一定是下列形式：$2,4,p^a,2p^a$（ $p$ 是奇素数， $a$ 是正整数）
- 求所有原根：若 $g$ 为 $m$ 的一个原根，则 $g^s\space(1\le s\le\varphi(m),\gcd(s,\varphi(m))=1)$ 给出了 $m$ 的所有原根。因此若 $m$ 有原根，则 $m$ 有 $\varphi(\varphi(m))$ 个原根
- 求一个原根，$O(n\log\log n)$ ~~实际远远不到~~

```c++
ll getG(ll n){ //求n最小的原根
	static vector<ll> a; a.clear();
	ll k=n-1;
	repeat(i,2,sqrt(k+1)+1)
	if(k%i==0){
		a.push_back(i); //a存放(n-1)的质因数
		while(k%i==0)k/=i;
	}
	if(k!=1)a.push_back(k);
	repeat(i,2,n){ //枚举答案
		bool f=1;
		for(auto j:a)
		if(qpow(i,(n-1)/j,n)==1){
			f=0;
			break;
		}
		if(f)return i;
	}
	return -1; 
}
```

#### N次剩余

- 求 $x^a \equiv b \pmod m$ ，基于BSGS、原根

```c++
//只求一个
ll residue(ll a,ll b,ll mod){ //ans^a%mod==b
	ll g=getG(mod),c=bsgs(qpow(g,a,mod),b,mod);
	if(c==-1)return -1;
	return qpow(g,c,mod);
}
//求所有N次剩余
vector<ll> ans;
void allresidue(ll a,ll b,ll mod){ //ans^a%mod==b
	ll g=getG(mod),c=bsgs(qpow(g,a,mod),b,mod);
	ans.clear();
	if(b==0){ans.push_back(0);return;}
	if(c==-1)return;
	ll now=qpow(g,c,mod);
	ll step=(mod-1)/__gcd(a,mod-1);
	ll ps=qpow(g,step,mod);
	for(ll i=c%step;i<mod-1;i+=step,now=mul(now,ps,mod))
		ans.push_back(now);
	sort(ans.begin(),ans.end());
}
```

### 数论函数的生成

#### 单个欧拉函数

- $\varphi(n)=$ 小于 `n` 且与 `n` 互质的正整数个数
- 令 `n` 的唯一分解式 $n=Π({p_k}^{a_k})$，则 $\varphi(n)=n\cdot Π(1-\dfrac 1 {p_k})$
- $O(\sqrt n)$

```c++
int getphi(int n){
	int ans=n;
	repeat(i,2,sqrt(n)+2)
	if(n%i==0){
		while(n%i==0)n/=i;
		ans=ans/i*(i-1);
	}
	if(n>1)ans=ans/n*(n-1);
	return ans;
}
```

#### 线性递推乘法逆元

求1..(n-1)的逆元，$O(n)$

```c++
void get_inv(int n,int m=mod){
	inv[1]=1;
	repeat(i,2,n)inv[i]=m-m/i*inv[m%i]%m;
}
```

求a[1..n]的逆元，离线，$O(n)$

```c++
void get_inv(int a[],int n){ //求a[1..n]的逆元，存在inv[1..n]中
	static int pre[N];
	pre[0]=1;
	repeat(i,1,n+1)
		pre[i]=(ll)pre[i-1]*a[i]%mod;
	int inv_pre=qpow(pre[n],mod-2);
	repeat_back(i,1,n+1){
		inv[i]=(ll)pre[i-1]*inv_pre%mod;
		inv_pre=(ll)inv_pre*a[i]%mod;
	}
}
```

#### 线性筛

- 定理：求出 $f(p)$（$p$ 为质数）的复杂度不超过 $O(\log p)$ 的积性函数可以被线性筛

##### 筛素数

- `a[i]` 表示第 $i+1$ 个质数，`vis[i]==0` 表示 $i$ 是素数，`rec[i]` 为 $i$ 的最小质因数
- $O(n)$

```c++
bool vis[N]; int rec[N]; vector<int> a;
void get_prime(){
	vis[1]=1;
	repeat(i,2,N){
		if(!vis[i])a.push_back(i),rec[i]=i;
		for(auto j:a){
			if(i*j>=N)break;
			vis[i*j]=1; rec[i*j]=j;
			if(i%j==0)break;
		}
	}
}
```

##### 筛欧拉函数

- 线性版，$O(n)$

```c++
bool vis[N]; int phi[N]; vector<int> a;
void get_phi(){
	vis[1]=1; phi[1]=1;
	repeat(i,2,N){
		if(!vis[i])a.push_back(i),phi[i]=i-1;
		for(auto j:a){
			if(i*j>=N)break;
			vis[i*j]=1;
			if(i%j==0){phi[i*j]=phi[i]*j; break;}
			phi[i*j]=phi[i]*(j-1);
		}
	}
}
```

- 不是线性但节省力气和空间版，$O(n\log\log n)$

```c++
void get_phi(){
	phi[1]=1; //其他的值初始化为0
	repeat(i,2,N)if(!phi[i])
	for(int j=i;j<N;j+=i){
		if(!phi[j])phi[j]=j;
		phi[j]=phi[j]/i*(i-1);
	}
}
```

##### 筛莫比乌斯函数

- $O(n)$

```c++
bool vis[N]; int mu[N]; vector<int> a;
void get_mu(){
	vis[1]=1; mu[1]=1;
	repeat(i,2,N){
		if(!vis[i])a.push_back(i),mu[i]=-1;
		for(auto j:a){
			if(i*j>=N)break;
			vis[i*j]=1;
			if(i%j==0){mu[i*j]=0; break;}
			mu[i*j]=-mu[i];
		}
	}
}
```

##### 筛其他函数

- 筛约数个数

```c++
bool vis[N]; int d[N]; vector<int> a;
void get_d(){
	vector<int> c(N); vis[1]=1; d[1]=1,c[1]=0;
	repeat(i,2,N){
		if(!vis[i])a.push_back(i),d[i]=2,c[i]=1;
		for(auto j:a){
			if(i*j>=N)break;
			vis[i*j]=1;
			if(i%j==0){
				d[i*j]=d[i]/(c[i]+1)*(c[i]+2);
				c[i*j]=c[i]+1;
				break;
			}
			d[i*j]=d[i]*2,c[i*j]=1;
		}
	}
}
```

- 筛gcd

```c++
int gcd[N][N];
void get_gcd(int n,int m){
	repeat(i,1,n+1)
	repeat(j,1,m+1)
	if(!gcd[i][j])
	repeat(k,1,min(n/i,m/j)+1)
		gcd[k*i][k*j]=k;
}
```

#### min_25筛

- 求 $[1,n]$ 内的素数个数

```c++
#include<cstdio>
#include<math.h>
#define ll long long
const int N = 316300;
ll n, g[N<<1], a[N<<1];
int id, cnt, sn, prime[N];
inline int Id(ll x){return x<=sn?x:id-n/x+1;}
int main() {
	scanf("%lld", &n), sn=sqrt(n);
	for(ll i=1; i<=n; i=a[id]+1) a[++id]=n/(n/i), g[id]=a[id]-1;
	for(int i=2; i<=sn; ++i) if(g[i]!=g[i-1]){
		// 这里 i 必然是质数，因为 g[] 是前缀质数个数
		// 当 <i 的质数的倍数都被筛去，让 g[] 发生改变的位置只能是下一个质数
		// 别忘了 i<=sn 时，ID(i) 就是 i。
		prime[++cnt]=i;
		ll sq=(ll)i*i;
		for(int j=id; a[j]>=sq; --j) g[j]-=g[Id(a[j]/i)]-(cnt-1);
	}
	return printf("%lld\n", g[id]), 0;
}
```

### 素数约数相关

#### 唯一分解

- 用数组表示数字唯一分解式的素数的指数，如 $50=\{1,0,2,0,…\}$
- 可以用来计算阶乘和乘除操作

```c++
void fac(int a[],ll n){
	repeat(i,2,(int)sqrt(n)+2)
	while(n%i==0)a[i]++,n/=i;
	if(n>1)a[n]++;
}
```

- set维护版

```c++
struct fac{
	#define facN 1010
	ll a[facN]; set<ll> s;
	fac(){mst(a,0); s.clear();}
	void lcm(ll n){ //self=lcm(self,n)
		repeat(i,2,facN)
		if(n%i==0){
			ll cnt=0;
			while(n%i==0)cnt++,n/=i;
			a[i]=max(a[i],cnt); //改成a[i]+=cnt就变成了乘法
		}
		if(n>1)s.insert(n);
	}
	ll value(){ //return self%mod
		ll ans=1;
		repeat(i,2,facN)
			if(a[i])ans=ans*qpow(i,a[i],mod)%mod;
		for(auto i:s)ans=ans*i%mod;
		return ans;
	}
}f;
```

#### 素数判定 | 朴素 or Miller-Rabin

- 朴素算法，$O(\sqrt n)$

```c++
bool isprime(int n){
	if(n<=3)return n>=2;
	if(n%2==0 || n%3==0)return 0;
	repeat(i,1,int(sqrt(n)+1.5)/6+1)
		if(n%(i*6-1)==0 || n%(i*6+1)==0)return 0;
	return 1;
}
```

- Miller-Rabin素性测试，$O(10\cdot\log^3 n)$

```c++
bool isprime(ll n){
	if(n<4)return n>1;
	ll a=n-1,b=0;
	while(a%2==0)a/=2,++b;
	repeat(i,0,10){
		ll x=rnd()%(n-2)+2,v=qpow(x,a,n);
		if(v==1 || v==n-1)continue;
		repeat(j,0,b+1){
			v=mul(v,v,n); //mul要防爆
			if(v==n-1)break;
		}
		if(v!=n-1)return 0;
	}
	return 1;
}
```

#### 大数分解 | Pollard-rho

- $O(n^{\tfrac 1 4})$，基于MR素性测试（很遗憾的是，我不擅长卡常因此这个板子过不了洛谷P4718）

```c++
ll pollard_rho(ll c,ll n){
	ll i=1,x,y,k=2,d;
	x=y=rnd()%n;
	while(1){
		d=__gcd(n+y-x,n);
		if(d>1 && d<n)
			return d;
		if(++i==k)y=x,k*=2;
		x=(mul(x,x,n)+n-c)%n; //mul要防爆
		if(y==x)return n;
	}
}
vector<ll> ans; //存结果（质因数，无序）
void rho(ll n){ //分解n
	if(isprime(n)){
		ans.push_back(n);
		return;
	}
	ll t;
	do{t=pollard_rho(rnd()%(n-1)+1,n);}while(t>=n);
	rho(t);
	rho(n/t);
}
```

#### 单个约数个数函数

```c++
int get_divisor(int n){ //求约数个数
	int ans=0;
	for(int i=1;i<n;i=n/(n/(i+1)))
	if(n%i==0)
		ans++; //v.push_back(i); //记录约数
	return ans+1; //v.push_back(n); //记录约数
}
```

#### 反素数生成

- 求因数最多的数（因数个数一样则取最小）
- 性质：$M = {p_1}^{k_1}{p_2}^{k_2}...$ 其中，$p_i$ 是从 $2$ 开始的连续质数，$k_i-k_{i+1}∈\{0,1\}$
- 先打出质数表再 $dfs$，枚举 $k_n$，$O(\exp)$

```c++
int pri[16]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
ll n; //范围
pair<ll,ll> ans; //ans是结果，ans.fi是最大反素数，ans.se是反素数约数个数
void dfs(ll num=1,ll cnt=1,int *p=pri,int pre=inf){ //注意ans要初始化
	if(make_pair(cnt,-num)>make_pair(ans.se,-ans.fi))
		ans={num,cnt};
	num*=*p;
	for(int i=1;i<=pre && num<=n;i++,num*=*p)
		dfs(num,p+1,i,cnt*(i+1));
}
```

- $n$ 以内约数个数最大值是 $O(n^{\tfrac {1.066}{\ln\ln n}})$

|      范围      | 1e4  |  1e5  |  1e6   |    1e9    |       1e16       |
| :------------: | :--: | :---: | :----: | :-------: | :--------------: |
|   最大反素数   | 7560 | 83160 | 720720 | 735134400 | 8086598962041600 |
| 反素数约数个数 |  64  |  128  |  240   |   1344    |      41472       |

### 数论杂项

#### 数论分块

- $n=(k-n\%k)(n/k)+(n\%k)(n/k+1)$
- 将 $\lfloor \dfrac{n}{x}\rfloor=C$ 的 $[x_{\min},x_{\max}]$ 作为一块，其中区间内的任一整数 $x_0$ 满足 $x_{\max}=n/(n/x_0)$

```c++
for(int l=l0,r;l<=r0;l=r+1){
	r=min(r0,n/(n/l));
	//c=n/l;
	//len=r-l+1;
}
```

- 将 $\lceil \dfrac{n}{x}\rceil=C$ 的 $[x_{\min},x_{\max}]$ 作为一块：

```c++
for(int l=l0,r;l<=r0;l=r+1){
	r=min(r0,n/(n/l)); if(n%r==0)r=max(r-1,l);
	//c=(n+l-1)/l;
	//len=l-r+1;
}
```

#### 二次剩余

- 对于奇素数模数 $p$，存在 $\frac {p-1} 2$ 个二次剩余 $\{1^2,2^2,...,(\frac {p-1} 2)^2\}$，和相同数量的二次非剩余
- 对于奇素数模数 $p$，如果 $n^{\frac{p-1}2}\equiv1\pmod{p}$ ，则 $n$ 是一个二次剩余；如果 $n^{\frac{p-1}2}\equiv-1\pmod{p}$，则 $n$ 是一个二次非剩余
- 对于奇素数模数 $p$，二次剩余的乘积是二次剩余，二次剩余与非二次剩余乘积为非二次剩余，非二次剩余乘积是二次剩余
- 费马-欧拉素数定理：$(4n+1)$ 型素数只能用一种方法表示为一个范数（两个完全平方数之和），$(4n+3)$ 型素数不能表示为一个范数
- 二次互反律：记 $p^{\frac{q-1}2}$ 的符号为 $(\dfrac p q)$ ，则对奇素数 $p,q$ 有 $(\dfrac p q)\cdot(\dfrac q p)=(-1)^{\tfrac{p-1}2\cdot\tfrac{q-1}2}$
- 求二次剩余，要求 mod 是质数，sqrtmod() 返回其中一个 sqrt，另一个为 mod 减这个返回值（如果 mod=2 就没有第二个）；返回 $-1$ 表示无解

```c++
ll w;
struct vec{ //x+y*sqrt(w)
	ll x,y;
	vec operator*(vec b){
		return {(x*b.x+y*b.y%mod*w)%mod,(x*b.y+y*b.x)%mod};
	}
};
vec qpow(vec a,ll b){
	vec ans={1,0};
	for(;b;a=a*a,b>>=1)
		if(b&1)ans=ans*a;
	return ans;
}
ll Leg(ll a){return qpow(a,(mod-1)>>1,mod)!=mod-1;}
ll sqrtmod(ll b){
	if(mod==2)return 1;
	if(!Leg(b))return -1;
	ll a;
	do{a=rnd()%mod; w=((a*a-b)%mod+mod)%mod;}while(Leg(w));
	return qpow({a,1},(mod+1)>>1).x;
}
```

#### 莫比乌斯反演

- 引理1：$\lfloor \dfrac{a}{bc}\rfloor=\lfloor \dfrac{\lfloor \dfrac{a}{b}\rfloor}{c}\rfloor$；引理2：$n$ 的因数个数 $≤\lfloor 2\sqrt n \rfloor$
- 狄利克雷卷积：$(f*g)(n)=\sum\limits_{d|n}f(d)g(\dfrac n d)$，有交换律、结合律、对加法的分配律
- 积性函数：$\{f(n)|\gcd(n,m)=1\Rightarrow f(nm)=f(n)f(m)\}$
- 单位函数：$\varepsilon(n)=[n=1]$ 为狄利克雷卷积的单位元
- 恒等函数：$id(n)=n$
- 约数个数：$d(n)=1*1$
- 约数之和：$\sigma(n)=1*id$
- 莫比乌斯函数性质：$\mu(n)=\begin{cases} 1&n=1\\0&n含有平方因子\\(-1)^k&k为n的质因数个数\end{cases}$
- 结论：$(\forall f)(f*\varepsilon=f),\mu*1=\varepsilon,\varphi*1=id,d*\mu=id$
- 莫比乌斯反演：若$f=g*1$，则$g=f*\mu$；或者，若$f(n)=\sum\limits_{d|n}g(d)$，则$g(n)=\sum\limits_{d|n}\mu(d)f(\dfrac n d)$

***

- 例题：求模意义下的 $\sum\limits_{i=1}^n \sum\limits_{j=1}^m \dfrac{i\cdot j}{\gcd(i,j)}$
- $ans=\sum\limits_{i=1}^n\sum\limits_{j=1}^m\sum\limits_{d|i,d|j,\gcd(\frac i d,\frac j d)=1}\dfrac{i\cdot j}d$
- 非常经典的化法：
- $ans=\sum\limits_{d=1}^n d\cdot\sum\limits_{i=1}^{\lfloor\frac nd\rfloor}\sum\limits_{j=1}^{\lfloor\frac md\rfloor}[\gcd(i,j)=1]i\cdot j$
- 设 $sum(n,m)=\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m}[\gcd(i,j)=1]i\cdot j$
- $sum(n,m)=\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m}\sum\limits_{c|i,c|j}{\mu(c)}\cdot i\cdot j$
- 设 $i'=\dfrac i c,j'=\dfrac j c$
- $sum(n,m)=\sum\limits_{c=1}^n\mu(c)\cdot c^2\cdot\sum\limits_{i'=1}^{\lfloor\frac nc\rfloor}\sum\limits_{j'=1}^{\lfloor\frac mc\rfloor} i'\cdot j'$
- 易得 $\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m} i\cdot j=\dfrac 1 4 n(n+1) m(m+1)$

#### 杜教筛

- $g(1)S(n)=\sum\limits_{i=1}^n(f*g)(i)-\sum\limits_{i=2}^n g(i)S(\lfloor\dfrac n i \rfloor),S(n)=\sum\limits_{i=1}^nf(i)$
- 如果能找到合适的 $g(n)$，能快速计算 $\sum\limits_{i=1}^n(f*g)(i)$，就能快速计算 $S(n)$
- $f(n)=\mu(n),g(n)=1,(f*g)(n)=[n=1]$
- $f(n)=\varphi(n),g(n)=1,(f*g)(n)=n$
- $f(n)=n\cdot\varphi(n),g(n)=n,(f*g)(n)=n^2$
- $f(n)=d(n),g(n)=\mu(n),(f*g)(n)=1$
- $f(n)=\sigma(n),g(n)=\mu(n),(f*g)(n)=n$（但是用公式 $\sum\limits_{i=1}^{n}\sigma(n)=\sum\limits_{i=1}^{n}i\cdot\lfloor\dfrac n i\rfloor$ 更好）
- $O(n^{\tfrac 2 3})$，注意有递归的操作就要记忆化

```c++
struct DU{
	static const int N=2000010;
	int sum[N];
	DU(){
		vector<int> a,mu(N,1),vis(N,0);
		repeat(i,2,N){
			if(!vis[i])a.push_back(i),mu[i]=-1;
			for(auto j:a){
				if(i*j>=N)break;
				vis[i*j]=1;
				if(i%j==0){mu[i*j]=0; break;}
				mu[i*j]=-mu[i];
			}
		}
		repeat(i,1,N)sum[i]=sum[i-1]+mu[i];
	}
	ll sum_mu(ll n){
		if(n<N)return sum[n];
		static map<ll,ll> rec; if(rec.count(n))return rec[n];
		ll ans=1;
		for(ll l=2,r;l<=n;l=r+1){
			r=n/(n/l);
			ans-=sum_mu(n/l)*(r-l+1);
		}
		return rec[n]=ans;
	}
	ll sum_phi(ll n){
		ll ans=0;
		for(ll l=1,r;l<=n;l=r+1){
			r=n/(n/l);
			ans+=(sum_mu(r)-sum_mu(l-1))*(n/l)*(n/l);
		}
		return ((ans-1)>>1)+1;
	}
	ll sum_d(ll n){
		static map<ll,ll> rec; if(rec.count(n))return rec[n];
		ll ans=n;
		for(ll l=2,r;l<=n;l=r+1){
			r=n/(n/l);
			ans-=(sum_mu(r)-sum_mu(l-1))*sum_d(n/l);
		}
		return rec[n]=ans;
	}
	ll sum_sigma(ll n){
		ll ans=0;
		for(ll l=1,r;l<=n;l=r+1){
			r=n/(n/l);
			ans+=(l+r)*(r-l+1)/2*(n/l);
		}
		return ans;
	}
}du;
```

#### 斐波那契数列

- 定义：$F_0=0,F_1=1,F_n=F_{n-1}+F_{n-2}$
- $F_n=\dfrac 1 {\sqrt{5}} [(\dfrac{1+\sqrt 5}2)^n-(\dfrac{1-\sqrt 5}2)^n)]$ （公式中若 $5$ 是二次剩余则可以化简，比如 $\sqrt 5\equiv 383008016\pmod {1000000009}$）
- $F_{a+b-1}=F_{a-1}F_{b-1}+F_aF_b$ （重要公式）
- $F_{n-1}F_{n+1}-F_n^2=(-1)^n$ （卡西尼性质）
- $F_{n}^2+F_{n+1}^2=F_{2n+1}$
- $F_{n+1}^2-F_{n-1}^2=F_{2n}$ （由上一条写两遍相减得到）
- $F_1+F_3+F_5+...+F_{2n-1}=F_{2n}$ （奇数项求和）
- $F_2+F_4+F_6+...+F_{2n}=F_{2n+1}-1$ （偶数项求和）
- $F_1^2+F_2^2+F_3^2+...+F_n^2=F_nF_{n+1}$
- $F_1+2F_2+3F_3+...+nF_n=nF_{n+2}-F_{n+3}+2$
- $-F_1+F_2-F_3+...+(-1)^nF_n=(-1)^n(F_{n+1}-F_n)+1$
- $F_{2n-2m-2}(F_{2n}+F_{2n+2})=F_{2m+2}+F_{4n-2m}$
- $F_a \mid F_b \Leftrightarrow a \mid b$
- $\gcd(F_a,F_b)=F_{\gcd(a,b)}$
- 当 $p$ 为 $5k\pm 1$ 型素数时，$\begin{cases} F_{p-1}\equiv 0\pmod p \\ F_p\equiv 1\pmod p \\ F_{p+1}\equiv 1\pmod p \end{cases}$
- 当 $p$ 为 $5k\pm 2$ 型素数时，$\begin{cases} F_{p-1}\equiv 1\pmod p \\ F_p\equiv -1\pmod p \\ F_{p+1}\equiv 0\pmod p \end{cases}$
- $F_{n+2}$ 为集合 `{1,2,3,...,n-2}` 中不包含相邻正整数的子集个数（包括空集）
- `F(n)%m` 的周期 $\le 6m$（$m=2\times 5^k$ 取等号）
- 齐肯多夫定理：任何正整数都可以表示成若干个不连续的斐波那契数（$F_2$ 开始）可以用贪心实现

快速倍增法求$F_n$，返回二元组$(F_n,F_{n+1})$ ，$O(\log n)$

```c++
pii fib(ll n){ //fib(n).fi即结果
	if(n==0)return {0,1};
	pii p=fib(n>>1);
	ll a=p.fi,b=p.se;
	ll c=a*(2*b-a)%mod;
	ll d=(a*a+b*b)%mod;
	if(n&1)return {d,(c+d)%mod};
	else return {c,d};
}
```

#### 佩尔方程×Pell

- $x^2-dy^2=1$，$d$ 是正整数
- 若 $d$ 是完全平方数，只有平凡解 $(\pm 1,0)$，其余情况总有非平凡解
- 若最小正整数解 $(x_1,y_1)$，则递推公式
- $\begin{cases}x_n=x_1x_{n-1}+dy_1y_{n-1}\\y_n=y_1x_{n-1}+x_1y_{n-1}\end{cases}$
- $\left[\begin{array}{c}x_n\\y_n\end{array}\right]=\left[\begin{array}{cc}x_1 & dy_1\\y_1 & x_1\end{array}\right]\left[\begin{array}{c}x_{n-1}\\y_{n-1}\end{array}\right]$
- 最小解（可能溢出）

```c++
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

## 组合数学

### 组合数取模 | Lucas+extra

- Lucas定理用来求模意义下的组合数
- 真·Lucas，$p$ 是质数（~~后面的exLucas都不纯正~~）

```c++
ll lucas(ll a,ll b,ll p){ //a>=b
	if(b==0)return 1;
	return mul(C(a%p,b%p,p),lucas(a/p,b/p,p),p);
}
```

- 特例：如果p=2，可能lucas失效（？）

```c++
ll C(ll a,ll b){ //a>=b，p=2的情况
	return (a&b)==b;
}
```

- 快速阶乘和exLucas
- $qfac.A(x),qfac.B(x)$ 满足 $A\equiv \dfrac{x!}{p^B}\pmod {p^k}$
- $qfac.C(a,b)\equiv C_a^b \pmod {p^k}$
- $exlucas(a,b,m)\equiv C_a^b \pmod m$，函数内嵌中国剩余定理

```c++
struct Qfac{
	ll s[2000010];
	ll p,m;
	ll A(ll x){ //快速阶乘的A值
		if(x==0)return 1;
		ll c=A(x/p);
		return s[x%m]*qpow(s[m],x/m,m)%m*c%m;
	}
	ll B(ll x){ //快速阶乘的B值
		int ans=0;
		for(ll i=x;i;i/=p)ans+=i/p;
		return ans;
	}
	ll C(ll a,ll b){ //组合数，a>=b
		ll k=B(a)-B(b)-B(a-b);
		return A(a)*gcdinv(A(b),m)%m
			*gcdinv(A(a-b),m)%m
			*qpow(p,k,m)%m;
	}
	void init(ll _p,ll _m){ //一定要满足m=p^k
		p=_p,m=_m;
		s[0]=1;
		repeat(i,1,m+1)
			if(i%p)s[i]=s[i-1]*i%m;
			else s[i]=s[i-1];
	}
}qfac;
ll exlucas(ll a,ll b,ll mod){
	ll ans=0,m=mod;
	for(ll i=2;i<=m;i++) //不能repeat
	if(m%i==0){
		ll p=i,k=1;
		while(m%i==0)m/=i,k*=i;
		qfac.init(p,k);
		ans=(ans+qfac.C(a,b)*(mod/k)%mod*gcdinv(mod/k,k)%mod)%mod;
	}
	return (ans+mod)%mod;
}
```

### 康托展开+逆 编码与解码

<H4>康托展开+逆</H4>

- 康托展开即排列到整数的映射
- 排列里的元素都是从1到n

```c++
//普通版，O(n^2)
int cantor(int a[],int n){
	int f=1,ans=1; //假设答案最小值是1
	repeat_back(i,0,n){
		int cnt=0;
		repeat(j,i+1,n)cnt+=a[j]<a[i];
		ans=(ans+f*cnt%mod)%mod; //ans+=f*cnt;
		f=f*(n-i)%mod; //f*=(n-i);
	}
	return ans;
}
//树状数组优化版，基于树状数组，O(nlogn)
int cantor(int a[],int n){
	static BIT t; t.init(); //树状数组
	ll f=1,ans=1; //假设答案最小值是1
	repeat_back(i,0,n){
		ans=(ans+f*t.sum(a[i])%mod)%mod; //ans+=f*t.sum(a[i]);
		t.add(a[i],1);
		f=f*(n-i)%mod; //f*=(n-i);
	}
	return ans;
}
//逆展开普通版，O(n^2)
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
		repeat(i,0,q)it++; //第q+1小的数
		ans[i]=*it;
		s.erase(it);
	}
	return ans;
}
```

<H4>编码与解码问题</H4>

<1>

- 给定一个字符串，求出它的编号
- 例，输入acab，输出5（aabc,aacb,abac,abca,acab,...）
- 用递归，令d(S)是小于S的排列数，f(S)是S的全排列数
- 小于acab的第一个字母只能是a，所以d(acab)=d(cab)
- 第二个字母是a,b,c，所以d(acab)=f(bc)+f(ac)+d(ab)
- d(ab)=0
- 因此d(acab)=4，加1之后就是答案

<2>

- 给定编号求字符串，对每一位进行尝试即可

### 置换群计数

Polya定理

- 例：立方体 $n=6$ 个面，每个面染上 $m=3$ 种颜色中的一种
- 两个染色方案相同意味着两个立方体经过旋转可以重合
- 其染色方案数为：$\dfrac{\sum m^{k_i}}{|k|}$（$k_i$ 为某一置换可以拆分的循环置换数，$|k|$ 为所有置换数）

```
不旋转，{U|D|L|R|F|B}，k=6，共1个
对面中心连线为轴的90度旋转，{U|D|L R F B}，k=3，共6个
对面中心连线为轴的180度旋转，{U|D|L R|F B}，k=4，共3个
对棱中点连线为轴的180度旋转，{U L|D R|F B}，k=3，共6个
对顶点连线为轴的120度旋转，{U L F|D R B}，k=2，共8个
```

- 因此 $\dfrac{3^6+3^3 \cdot 6+3^4 \cdot 3+3^3 \cdot 6+3^2 \cdot 8}{1+6+3+6+8}=57$
- 例题（poj1286），n个点连成环，染3种颜色，允许旋转和翻转

```c++
ans=0,cnt=0;
//只考虑旋转，不考虑翻转
repeat(i,1,n+1)
	ans+=qpow(3,__gcd(i,n));
cnt+=n;
//考虑翻转
if(n%2==0)ans+=(qpow(3,n/2+1)+qpow(3,n/2))*(n/2);
else ans+=qpow(3,(n+1)/2)*n;
cnt+=n;
cout<<ans/cnt<<endl;
```

### 组合数学的一些结论

***

<H4> 组合数 </H4>

- `C(n,k)=(n-k+1)*C(n,k-1)/k​`

```c++
const int N=20;
repeat(i,0,N){
	C[i][0]=C[i][i]=1;
	repeat(j,1,i)C[i][j]=C[i-1][j]+C[i-1][j-1];
}
```

- 二项式反演
- $\displaystyle f_n=\sum_{i=0}^n{n\choose i}g_i\Leftrightarrow g_n=\sum_{i=0}^n(-1)^{n-i}{n\choose i}f_i$
- $\displaystyle f_k=\sum_{i=k}^n{i\choose k}g_i\Leftrightarrow g_k=\sum_{i=k}^n(-1)^{i-k}{i\choose k}f_i$

***

- 卡塔兰/卡特兰数×Catalan，$H_n=\dfrac{\binom{2n}n}{n+1}$，$H_n=\dfrac{H_{n-1}(4n-2)}{n+1}$

***

- 贝尔数×Bell，划分n个元素的集合的方案数

```c++
B[0]=B[1]=1;
repeat(i,2,N){
	B[i]=0;
	repeat(j,0,i)
		B[i]=(B[i]+C(i-1,j)*B[j]%mod)%mod;
}
```

***

- 错排数，$D_n=n![\dfrac 1{0!}-\dfrac 1{1!}+\dfrac 1{2!}-...+\dfrac{(-1)^n}{n!}]$

```c++
D[0]=1;
repeat(i,0,N-1){
	D[i+1]=D[i]+(i&1?C.inv[i+1]:mod-C.inv[i+1]);
	D[i]=1ll*D[i]*fac[i]%mod;
}
```

***

<H4>第一类斯特林数×Stirling</H4>

- 多项式 $x(x-1)(x-2) \cdots (x-n+1)$ 展开后 $x^r$ 的系数绝对值记作 $s(n,r)$ （系数符号 $(-1)^{n+r}$）
- 也可以表示 $n$ 个元素分成 $r$ 个环的方案数
- 递推式 $s(n,r) = (n-1)s(n-1,r)+s(n-1,r-1)$
- $\displaystyle n!=\sum_{i=0}^n s(n,i)$
- $\displaystyle A_x^n=\sum_{i=0}^n s(n,i)(-1)^{n-i}x^i$
- $\displaystyle A_{x+n-1}^n=\sum_{i=0}^n s(n,i)x^i$

***

<H4>第二类斯特林数×Stirling</H4>

- $n$ 个不同的球放入 $r$ 个相同的盒子且无空盒的方案数，记作 $S(n,r)$ 或 $S_n^r$
- 递推式 $S(n,r) = r S(n-1,r) + S(n-1,r-1)$
- 通项公式 $\displaystyle S(n,r)=\frac{1}{r!}\sum_{i=0}^r(-1)^i{r\choose i}(r-i)^n$
- $\displaystyle m^n=\sum_{i=0}^mS(n,i)A_m^i$
- $\displaystyle \sum_{i=1}^n i^k=\sum_{i=0}^kS(k,i)i!{n+1\choose i+1}$
- 斯特林反演
- $\displaystyle f(n)=\sum_{i=1}^n S(n,i)g(i)\Leftrightarrow g(n)=\sum_{i=0}^n(-1)^{n-i}s(n,i)f(i)$

***

- $a$ 个相同的球放入 $b$ 个不同的盒子，方案数为 $C_{a+b-1}^{b-1}$（隔板法）

***

- 一个长为 $n+m$ 的数组，$n$ 个 $1$，$m$ 个 $-1$，限制前缀和最大为 $k$，则方案数为 $C_{n+m}^{m+k}-C_{n+m}^{m+k+1}$

***

- $2n$ 个带标号的点两两匹配，方案数为 $(2n-1)!!=\dfrac{(2n)!}{2^nn!}$
- $1,2,...,n$ 中无序地选择 $r$ 个互不相同且互不相邻的数字，则这 $r$ 个数字之积对所有方案求和的结果为 $C_{n+1}^{2r}(2r-1)!!=\dfrac{C_{n+1}^{2r}(2r)!}{2^rr!}$（问题可以转换为，$(n+1)$ 个点无序匹配 $r$ 对点的方案数）

```c++
int M(int a,int b){
	static const int inv2=qpow(2,mod-2);
	return C(a+1,2*b)*C.fac[2*b]%mod*qpow(inv2,b)%mod*C.inv[b]%mod;
}
```

***

## 博弈论

### SG函数 SG定理

- 有向无环图中，两个玩家轮流推多颗棋子，不能走的判负
- 假设 $x$ 的后继状态为 $y_1,y_2,...,y_k$
- 则 $SG[x]=mex\{SG[y_i]\}$，$mex(S)$ 表示不属于集合 $S$ 的最小自然数
- 当且仅当所有起点SG值的异或和为 $0$ 时先手必败
- （如果只有一个起点，SG的值可以只考虑01）
- 例题：拿 $n$ 堆石子，每次只能拿一堆中的斐波那契数颗石子

```c++
void getSG(int n){
	mst(SG,0);
	repeat(i,1,n+1){
		mst(S,0);
		for(int j=0;f[j]<=i && j<=N;j++)
			S[SG[i-f[j]]]=1;
		for(int j=0;;j++)
		if(!S[j]){
			SG[i]=j;
			break;
		}
	}
}
```

### Nim游戏

***

Nim

- $n$ 堆石子 $a_1,a_2,...,a_n$，每次选择 $1$ 堆石子拿任意非空的石子，拿不了的人失败
- $SG_i=a_i,NimSum=\oplus\{SG_i\}$，先手必败当且仅当 $NimSum=0$
- 注：先手必胜策略是找到满足 `(a[i]>>(63-__builtin_clzll(NimSum)))&1` 的 $a[i]$，并取走 $a[i]-a[i]\oplus NimSum$ 个石子
- Bash Game：一堆石子 $n$，最多取 $k$ 个，$SG=n\%(k+1)$

***

Moore's Nimk

- $n$ 堆石子，每次最多选取 $k$ 堆石子，选中的每一堆都取走任意非空的石子
- 先手必胜当且仅当
	- 存在 $t$ 使得 `sum{(a[i]>>t)&1}%(k+1)!=0`

***

扩展威佐夫博弈×Extra Wythoff's Game

- 两堆石子，分别为 $a,b$，每次取一堆的任意非空的石子或者取两堆数量之差的绝对值小于等于 $k$ 的石子
- 解：假设 $a\le b$，当且仅当存在自然数 $n$ 使得 $a=\lfloor n\dfrac{\sqrt{(k+1)^2+4}-(k-1)}2\rfloor,b=a+n(k+1)$，先手必败
- Betty定理与Betty数列：$\alpha,\beta$ 为正无理数且 $\dfrac 1 {\alpha}+\dfrac 1 {\beta}=1$，数列 $\{\lfloor \alpha n\rfloor\},\{\lfloor \beta n\rfloor\},n=1,2,...$ 无交集且覆盖正整数集合

***

斐波那契博弈×Fibonacci Nim

- 一堆石子 $n,n\ge 2$，先手第一次只能取 $[1,n-1]$，之后每次取的石子数不多于对手刚取的石子数的 $2$ 倍且非空
- 先手必败当且仅当 $n$ 是Fibonacci数

***

阶梯Nim×Staircase Nim

- $n$ 堆石子，每次选择一堆取任意非空的石子放到前一堆，第 $1$ 堆的石子可以放到第 $0$ 堆
- 先手必败当且仅当奇数堆的石子数异或和为 $0$

***

Lasker's Nim

- $n$ 堆石子，每次可以选择一堆取任意非空石子，或者选择某堆至少为 $2$，分成两堆非空石子
- $SG(0)=0,SG(4k+1)=4k+1,SG(4k+2)=4k+2,SG(4k+3)=4k+4,SG(4k+4)=4k+3$

***

k倍动态减法博弈

- 一堆石子 $n,n\ge 2$，先手第一次只能取 $[1,n-1]$，之后每次取的石子数不多于对手刚取的石子数的 $k$ 倍且非空

```c++
int calc(ll n,int k){ //n<=1e8,k<=1e5
	static ll a[N],b[N],ans; //N=750010
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
	else{
		while(n){
			while(a[t]>n)t--;
			n-=a[t];
			ans=a[t];
		}
		return ans;
	}
}
```

***

Anti-SG | SJ定理

- $n$ 个游戏，移动不了的人获胜
- 先手必胜当且仅当
	- $(\forall i)SG_i\le 1$ 且 $NimSum=0$
	- $(\exist i)SG_i>1$ 且 $NimSum\not=0$

***

Every-SG

- $n$ 个游戏，每次都要移动所有可移动的游戏
- 对于先手来说，必胜态的游戏要越长越好，必败态的游戏要越短越好
- u是终止态，step(u)=0
- u->v,SG(u)=0,SG(v)>0，step(u)=max(step(v))+1
- u->v,SG(v)=0，step(u)=min(step(v))+1
- 先手必胜当且仅当所有游戏的step的最大值为奇数

### 删边游戏×Green Hachenbush

- 树上删边游戏
	- 一棵有根树，每次可以删除一条边并移除不和根连接的部分
	- 叶子的 $SG$ 为 $0$，非叶子的 $SG$ 为(所有儿子的 $SG$ 值 $+1$)的异或和
- 无向图删边游戏
	- 奇环可以缩为一个点加一条边，偶环可以缩为一点，变为树上删边游戏

### 翻硬币游戏

- $n$ 枚硬币排成一排，玩家的操作有一定约束，并且翻动的硬币中，最右边的必须是从正面翻到反面，不能操作的玩家失败
- 定理：局面的 $SG$ 值等于所有正面朝上的硬币单一存在时的 $SG$ 值的异或和（把这个硬币以外的所有硬币翻到反面后的局面的 $SG$ 值）
- 编号从 $1$ 开始
	- 每次翻一枚或两枚硬币 $SG(n)=n$
	- 每次翻转连续的 $k$ 个硬币 $SG(n)=[n\%k=0]$
	- Ruler Game，每次翻转一个区间的硬币，$SG(n)=lowbit(n)$
	- Mock Turtles Game，每次翻转不多于 $3$ 枚硬币 $SG(n)=2n-1-popcount(n-1)\%2$

### 高维组合游戏 | Nim积

- Nim和与Nim积的关系类似加法与乘法
- Tartan定理：对于一个高维的游戏（多个维度的笛卡尔积），玩家的操作也是笛卡尔积的形式，那么对每一维度单独计算SG值，最终的SG值为它们的Nim积
- 比如，在 $n\times m$ 硬币中翻转 $4$ 个硬币，$4$ 个硬币构成一个矩形，这个矩形是每一维度（翻转两个硬币）的笛卡尔积
- $O(\log^2 n)$

```c++
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
//int x=read(),y=read(),z=read();
//ans^=nim.f(SG(x),nim.f(SG(y),SG(z)));
```

### 不平等博弈 | 超现实数

***

- 超现实数(Surreal Number)
- 超现实数由左右集合构成，是最大的兼容四则运算的全序集合，包含实数集和“无穷大”
- 博弈局面的值可以看作左玩家比右玩家多进行的次数，独立的局面可以相加
- 如果值 $>0$ 则左玩家必胜，$<0$ 则右玩家必胜，$=0$ 则后手必胜
- 一个博弈局面，$L$ 为左玩家操作一次后的博弈局面的最大值，$R$ 为右玩家操作一次后的博弈局面的最小值，那么该博弈局面的值 $G=\dfrac A {2^B},L<G<R$，并且 $B$ 尽可能小（$B=0$ 则 $|A|$ 尽可能小）
- 如果存在 $L=R$ 需要引入Irregular surreal number就不讨论了（比如两个玩家能进行同一操作即Nim）

***

- Blue-Red Hackenbush string
- 若干个 BW 串，player-W 只能拿 W，player-B 只能拿 B，每次拿走一个字符后其后缀也会消失，最先不能操作者输
- 对于每个串计算超现实数(Surreal Number)并求和，若 $> 0$ 则 W 必胜；若 $= 0$ 则后手必胜；若 $< 0$ 则 B 必胜

```c++
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

```c++
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
- 对于 W 点，先求所有儿子的值之和 $x$。如果 $x \ge 0$，那么直接加一即可。否则 $x$ 变为 $x$ 的小数部分加一，乘以 $2^{-\lfloor|x|\rfloor}$

***

- Alice's Game
- $x\times y$ 方格，如果 $x>1$ Alice可以水平切，如果 $y>1$ Bob可以垂直切，超现实数计算如下

```c++
ll calc(int x,int y){ //get surreal number
	while(x>1 && y>1)x>>=1,y>>=1;
	return x-y;
}
```

***

### 其他博弈结论

***

欧几里得的游戏

- 两个数 $a,b$，每次对一个数删去另一个数的整数倍，出现 $0$ 则失败
- $a\ge 2b$ 则先手必胜，否则递归处理

***

无向点地理问题×Undirected vertex geography problem

- 二分图上移动棋子，不能经过重复点
- 先手必败当且仅当存在一个不包含起点的最大匹配

***

Untitled

- 1到n，每次拿一个数或差值为1的两个数
- 先手必胜，第一步拿最中间的1/2个数，之后对称操作

***

## 代数结构

### 置换群

- 求 $A^x$，编号从 $0$ 开始，$O(n)$

```c++
void qpow(int a[],int n,int x){
	static int rec[N],c[N];
	static bool vis[N];
	fill(vis,vis+n,0);
	repeat(i,0,n)if(!vis[i]){
		int cnt=0; rec[cnt++]=i;
		for(int p=a[i];p!=i;p=a[p])
			rec[cnt++]=p,vis[p]=1;
		repeat(J,0,cnt)
			c[rec[J]]=a[rec[(J+x-1)%cnt]];
		repeat(J,0,cnt)
			a[rec[J]]=c[rec[J]];
	}
}
```

- $A^k=B$ 求任一 $A$，编号从 $0$ 开始，$O(n)$（暂无判断有解操作）

```c++
repeat(i,0,n){
	a[read()-1]=i;
	vis[i]=0;
}
repeat(i,0,n)if(!vis[i]){
	int cnt=0; rec[cnt++]=i;
	for(int p=a[i];p!=i;p=a[p])
		rec[cnt++]=p,vis[p]=1;
	repeat(J,0,cnt)
		c[1ll*J*k%cnt]=rec[J];
	repeat(J,0,cnt)
		ans[c[(J+1)%cnt]]=c[J];
}
```

### 多项式

#### 拉格朗日插值

- 函数曲线通过n个点 $(x_i,y_i)$，求 $f(k)$
- 拉格朗日插值：$f(x)=\sum\limits_{i=1}^n[y_i\Pi_{j!=i}\dfrac{x-x_j}{x_i-x_j}]$
- $O(n^2)$

```c++
repeat(i,0,n)x[i]%=mod,y[i]%=mod;
repeat(i,0,n){
	s1=y[i];
	s2=1;
	repeat(j,0,n)
	if(i!=j){
		s1=s1*(k-x[j])%mod;
		s2=s2*(x[i]-x[j])%mod;
	}
	ans=(ans+s1*getinv(s2)%mod+mod)%mod;
}
```

#### 快速傅里叶变换+任意模数

- 求两个多项式的卷积，$O(n\log n)$

```c++
struct FFT{
	static const int N=1<<20;
	struct cp{
		long double a,b;
		cp(){}
		cp(const long double &a,const long double &b):a(a),b(b){}
		cp operator+(const cp &t)const{return cp(a+t.a,b+t.b);}
		cp operator-(const cp &t)const{return cp(a-t.a,b-t.b);}
		cp operator*(const cp &t)const{return cp(a*t.a-b*t.b,a*t.b+b*t.a);}
		cp conj()const{return cp(a,-b);}
	};
	cp wn(int n,int f){
		static const long double pi=acos(-1.0);
		return cp(cos(pi/n),f*sin(pi/n));
	}
	int g[N];
	void dft(cp a[],int n,int f){
		repeat(i,0,n)if(i>g[i])swap(a[i],a[g[i]]);
		for(int i=1;i<n;i<<=1){
			cp w=wn(i,f);
			for(int j=0;j<n;j+=i<<1){
				cp e(1,0);
				for(int k=0;k<i;e=e*w,k++){
					cp x=a[j+k],y=a[j+k+i]*e;
					a[j+k]=x+y,a[j+k+i]=x-y;
				}
			}
		}
		if(f==-1){
			cp Inv(1.0/n,0);
			repeat(i,0,n)a[i]=a[i]*Inv;
		}
	}
	#ifdef CONV
	cp a[N],b[N];
	vector<ll> conv(const vector<ll> &u,const vector<ll> &v){ //一般fft
	    const int n=(int)u.size()-1,m=(int)v.size()-1;
	    const int k=32-__builtin_clz(n+m+1),s=1<<k;
		g[0]=0; repeat(i,1,s)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
		repeat(i,0,s){
			a[i]=cp(i<=n?u[i]:0,0);
			b[i]=cp(i<=m?v[i]:0,0);
		}
	    dft(a,s,1); dft(b,s,1);
		repeat(i,0,s)a[i]=a[i]*b[i];
	    dft(a,s,-1);
	    vector<ll> ans;
	    repeat(i,0,n+m+1)ans+=llround(a[i].a);
	    return ans;
	}
	#endif
	#ifdef CONV_MOD
	cp a[N],b[N],Aa[N],Ab[N],Ba[N],Bb[N];
	vector<ll> conv_mod(const vector<ll> &u,const vector<ll> &v,ll mod){ //任意模数fft
		const int n=(int)u.size()-1,m=(int)v.size()-1,M=sqrt(mod)+1;
		const int k=32-__builtin_clz(n+m+1),s=1<<k;
		g[0]=0; repeat(i,1,s)g[i]=(g[i/2]/2)|((i&1)<<(k-1));
		repeat(i,0,s){
			a[i]=i<=n?cp(u[i]%mod%M,u[i]%mod/M):cp();
			b[i]=i<=m?cp(v[i]%mod%M,v[i]%mod/M):cp();
		}
		dft(a,s,1); dft(b,s,1);
		repeat(i,0,s){
			int j=(s-i)%s;
			cp t1=(a[i]+a[j].conj())*cp(0.5,0);
			cp t2=(a[i]-a[j].conj())*cp(0,-0.5);
			cp t3=(b[i]+b[j].conj())*cp(0.5,0);
			cp t4=(b[i]-b[j].conj())*cp(0,-0.5);
			Aa[i]=t1*t3,Ab[i]=t1*t4,Ba[i]=t2*t3,Bb[i]=t2*t4;
		}
		repeat(i,0,s){
			a[i]=Aa[i]+Ab[i]*cp(0,1);
			b[i]=Ba[i]+Bb[i]*cp(0,1);
		}
		dft(a,s,-1); dft(b,s,-1);
		vector<ll> ans;
		repeat(i,0,n+m+1){
			ll t1=llround(a[i].a)%mod;
			ll t2=llround(a[i].b)%mod;
			ll t3=llround(b[i].a)%mod;
			ll t4=llround(b[i].b)%mod;
			ans+=(t1+(t2+t3)*M%mod+t4*M*M)%mod;
		}
		return ans;
	}
	#endif
}fft;
```

#### 多项式的一些概念

***

- 生成函数：$A(x)=a_0+a_1x+a_2x^2+...$
- 组合对象：x
- 组合对象的大小：x的指数i
- 方案数：系数
- 有 $1+x+x^2+...=\dfrac{1}{1-x}$

***

+ 指数生成函数：无序排列
+ 有 $1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+...=e^x$

***

* 严重空缺

***

### 矩阵

#### 矩阵乘法 矩阵快速幂

矩乘 $O(n^3)$，矩快 $O(n^3\log b)$

```c++
struct mat{
	static const int N=110;
	ll a[N][N];
	explicit mat(ll e=0){
		repeat(i,0,n)
		repeat(j,0,n)
			a[i][j]=e*(i==j);
	}
	mat operator*(const mat &b)const{ //矩阵乘法
		mat ans(0);
		repeat(i,0,n)
		repeat(j,0,n){
			ll &t=ans.a[i][j];
			repeat(k,0,n)
				t=(t+a[i][k]*b.a[k][j])%mod;
		}
		return ans;
	}
	ll *operator[](int x){return a[x];}
	const ll *operator[](int x)const{return a[x];}
};
mat qpow(mat a,ll b){ //矩阵快速幂
	mat ans(1); //mat ans; repeat(i,0,n)ans[i][i]=1;
	while(b){
		if(b&1)ans=ans*a;
		a=a*a; b>>=1;
	}
	return ans;
}
```

#### 矩阵高级操作

- 行列式、逆矩阵（luogu P3389 && luogu P4783）
- $O(n^3)$

```c++
int n,m;
#define T ll
struct mat{
	static const int N=110;
	vector< vector<T> > a;
	mat():a(N,vector<T>(N*2)){} //如果要求逆这里乘2
	T det;
	void r_div(int x,T k){ //第x行除以实数k
		T r=qpow(k,mod-2);
		repeat(i,0,m) //从x开始也没太大关系（对求det来说）
			a[x][i]=a[x][i]*r%mod;
		det=det*k%mod;
	}
	void r_plus(int x,int y,T k){ //第x行加上第y行的k倍
		repeat(i,0,m)
			a[x][i]=(a[x][i]+a[y][i]*k)%mod;
	}
	/*
	void r_div(int x,T k){ //lf版
		T r=1/k;
		repeat(i,0,m)a[x][i]*=r;
		det*=k;
	}
	void r_plus(int x,int y,T k){ //lf版
		repeat(i,0,m)a[x][i]+=a[y][i]*k;
	}
	*/
	bool gauss(){ //返回是否满秩，注意必须n<=m
		det=1;
		repeat(i,0,n){
			int t=-1;
			repeat(j,i,n)
			if(abs(a[j][i])>eps){t=j; break;}
			if(t==-1){det=0; return 0;}
			if(t!=i){a[i].swap(a[t]); det=-det;}
			r_div(i,a[i][i]);
			repeat(j,0,n) //如果只要det可以从i+1开始
			if(j!=i && abs(a[j][i])>eps)
				r_plus(j,i,-a[j][i]);
		}
		return 1;
	}
	T get_det(){gauss(); return det;} //返回行列式
	bool get_inv(){ //把自己变成逆矩阵，返回是否成功
		if(n!=m)return 0;
		repeat(i,0,n)
		repeat(j,0,n)
			a[i][j+n]=i==j; //生成增广矩阵
		m*=2; bool t=gauss(); m/=2;
		repeat(i,0,n)
		repeat(j,0,n)
			a[i][j]=a[i][j+n];
		return t;
	}
	//vector<T> &operator[](int x){return a[x];}
	//const vector<T> &operator[](int x)const{return a[x];}
}a;
```

- 任意模数行列式（HDOJ 2827）
- $O(n^3\log C)$

```c++
int n;
struct mat{
	static const int N=110;
	vector< vector<ll> > a;
	mat():a(N,vector<ll>(N)){}
	ll det(int n){
		ll ans=1;
		repeat(i,0,n){
			repeat(j,i+1,n)
			while(a[j][i]){
				ll t=a[i][i]/a[j][i];
				repeat(k,i,n)a[i][k]=(a[i][k]-a[j][k]*t)%mod;
				swap(a[i],a[j]);
				ans=-ans;
			}
			ans=ans*a[i][i]%mod;
			if(!ans)return 0;
		}
		return (ans+mod)%mod;
	}
}a;
```

#### 异或方程组

- 编号从 $0$ 开始，高斯消元部分 $O(n^3)$（luogu P2962）

```c++
bitset<N> a[N]; bool l[N];
int n,ans;
bool gauss(){ //返回是否有唯一解
	bool flag=1;
	repeat(i,0,n){
		int t=-1;
		repeat(j,i,n)if(a[j][i]){t=j; break;}
		if(t==-1){flag=0; continue;}
		if(t!=i)swap(a[i],a[t]);
		repeat(j,0,n)
		if(i!=j && a[j][i])
			a[j]^=a[i];
	}
	return flag;
}
void dfs(int x=n-1,int num=0){
	if(num>ans)return;
	if(x==-1){ans=num; return;}
	if(a[x][x]){
		bool v=a[x][n];
		repeat(i,x+1,n)
		if(a[x][i])
			v^=l[i];
		dfs(x-1,num+v);
	}
	else{
		dfs(x-1,num);
		l[x]=1;
		dfs(x-1,num+1);
		l[x]=0;
	}
}
int solve(){ //返回满足方程组的sum(xi)最小值
	ans=inf; gauss(); dfs(n-1,0);
	return ans;
}
```

#### 线性基

- 线性基是一系列线性无关的基向量组成的集合

<H4>异或线性基</H4>

- 结论：$basis.exist(a^b)$ 等价于 $a,b$ 在 $basis$ 里消去关键位后相等（要求是最简线性基，即第一个板子）
- 插入、查询 $O(\log M)$

```c++
struct basis{
	static const int n=63;
	#define B(x,i) ((x>>i)&1)
	ll a[n],sz;
	bool failpush; //是否线性相关
	void init(){mst(a,0); sz=failpush=0;}
	void push(ll x){ //插入元素
		repeat(i,0,n)if(B(x,i))x^=a[i];
		if(x!=0){
			int p=63-__builtin_clzll(x); sz++;
			repeat(i,p+1,n)if(B(a[i],p))a[i]^=x;
			a[p]=x;
		}
		else failpush=1;
	}
	ll top(){ //最大值
		ll ans=0;
		repeat(i,0,n)ans^=a[i];
		return ans;
	}
	bool exist(ll x){ //是否存在
		repeat_back(i,0,n)
		if((x>>i)&1){
			if(a[i]==0)return 0;
			else x^=a[i];
		}
		return 1;
	}
	ll kth(ll k){ //第k小，不存在返回-1
		if(failpush)k--; //如果认为0是可能的答案就加这句话
		if(k>=(1ll<<sz))return -1;
		ll ans=0;
		repeat(i,0,n)
		if(a[i]!=0){
			if(k&1)ans^=a[i];
			k>>=1;
		}
		return ans;
	}
}b;
basis operator+(basis a,const basis &b){ //将b并入a
	repeat(i,0,a.n)
	if(b.a[i])a.push(b.a[i]);
	a.failpush|=b.failpush;
	return a;
}
```

- 这个版本中求kth需要rebuild $O(\log^2 n)$

```c++
struct basis{
	//...
	void push(ll x){ //插入元素
		repeat_back(i,0,n)
		if((x>>i)&1){
			if(a[i]==0){a[i]=x; sz++; return;}
			else x^=a[i];
		}
		failpush=1;
	}
	ll top(){ //最大值
		ll ans=0;
		repeat_back(i,0,n)
			ans=max(ans,ans^a[i]);
		return ans;
	}
	void rebuild(){ //求第k小的前置操作
		repeat_back(i,0,n)
		repeat_back(j,0,i)
		if((a[i]>>j)&1)
			a[i]^=a[j];
	}
}b;
```

<H4>实数线性基</H4>

- 编号从 $0$ 开始，插入、查询 $O(n^2)$

```c++
struct basis{
	lf a[N][N]; bool f[N]; int n; //f[i]表示向量a[i]是否被占
	void init(int _n){
		n=_n;
		fill(f,f+n,0);
	}
	bool push(lf x[]){ //返回0表示可以被线性表示，不需要插入
		repeat(i,0,n)
		if(abs(x[i])>1e-5){ //这个值要大一些
			if(f[i]){
				lf t=x[i]/a[i][i];
				repeat(j,0,n)x[j]-=t*a[i][j];
			}
			else{
				f[i]=1;
				repeat(j,0,n)a[i][j]=x[j];
				return 1;
			}
		}
		return 0;
	}
}b;
```

#### 线性规划 | 单纯形法

- 声明：还没学会
- $\left[\begin{array}{ccccccc} a & a & a & a & a & a & b \\ a & a & a & a & a & a & b \\ a & a & a & a & a & a & b \\ c & c & c & c & c & c & v \end{array}\right]$
- 每行表示一个约束，$\sum ax\le b$，并且所有 $x\ge 0$，求 $\sum cx$ 的最大值
- 对偶问题：每列表示一个约束，$\sum ax\ge c$，并且所有 $x\ge 0$，求 $\sum bx$ 的最小值
- 先找 $c[y]>0$ 的 $y$，再找 $b[x]>0$ 且 $\dfrac {b[x]}{a[x][y]}$ 最小的x（找不到 $y$ 则 $v$，找不到 $x$ 则 INF），用行变换将 $a[x][y]$ 置 $1$，将其他 $a[i][y]$ 和 $c[y]$ 置 $0$
- 编号从 $1$ 开始，$O(n^3)$，缺init

```c++
const int M=1010; const lf eps=1e-6;
int n,m;
lf a[N][M],b[N],c[M],v; //a[1..n][1..m],b[1..n],c[1..m]
void pivot(int x,int y){
	b[x]/=a[x][y];
	repeat(j,1,m+1)if(j!=y)
		a[x][j]/=a[x][y];
	a[x][y]=1/a[x][y];
	repeat(i,1,n+1)
	if(i!=x && abs(a[i][y])>eps){
		b[i]-=a[i][y]*b[x];
		repeat(j,1,m+1)if(j!=y)
			a[i][j]-=a[i][y]*a[x][j];
		a[i][y]=-a[i][y]*a[x][y];
	}
	v+=c[y]*b[x];
	repeat(j,1,m+1)if(j!=y)
		c[j]-=c[y]*a[x][j];
	c[y]=-c[y]*a[x][y];
}
lf simplex(){ //返回INF表示无限制，否则返回答案
	while(1){
		int x,y;
		for(y=1;y<=m;y++)if(c[y]>eps)break;
		if(y==m+1)return v;
		lf mn=INF;
		repeat(i,1,n+1)
		if(a[i][y]>eps && mn>b[i]/a[i][y])
			mn=b[i]/a[i][y],x=i;
		if(mn==INF)return INF; //unbounded
		pivot(x,y);
	}
}
void init(){v=0;}
```

#### 矩阵的一些结论

***

- $n\times n$ 方阵 $A$ 有：$\left[\begin{array}{c}A&E\\O&E\end{array}\right]^{k+1}=\left[\begin{array}{c}A^k&E+A+A^2+...+A^k\\O&E\end{array}\right]$

***

- 线性递推转矩快
- $f_{n+3}=af_{n+2}+bf_{n+1}+cf_{n}\\\Leftrightarrow\left[\begin{array}{c}a&b&c\\1&0&0\\0&1&0\end{array}\right]^n \left[\begin{array}{c}f_2\\f_1\\f_0\end{array}\right]=\left[\begin{array}{c}f_{n+2}\\f_{n+1}\\f_{n}\end{array}\right]$

***

## 数学杂项

### 主定理

- 对于 $T(n)=aT(\dfrac nb)+n^k$ （要估算 $n^k$ 的 $k$ 值）
- 若 $\log_ba>k$，则 $T(n)=O(n^{\log_ba})$
- 若 $\log_ba=k$，则 $T(n)=O(n^k\log n)$
- 若 $\log_ba<k$（有省略），则 $T(n)=O(n^k)$

### 质数表

42737, 46411, 50101, 52627, 54577, 191677, 194869, 210407, 221831, 241337, 578603, 625409, 713569, 788813, 862481, 2174729, 2326673, 2688877, 2779417, 3133583, 4489747, 6697841, 6791471, 6878533, 7883129, 9124553, 10415371, 11134633, 12214801, 15589333, 17148757, 17997457, 20278487, 27256133, 28678757, 38206199, 41337119, 47422547, 48543479, 52834961, 76993291, 85852231, 95217823, 108755593, 132972461, 171863609, 173629837, 176939899, 207808351, 227218703, 306112619, 311809637, 322711981, 330806107, 345593317, 345887293, 362838523, 373523729, 394207349, 409580177, 437359931, 483577261, 490845269, 512059357, 534387017, 698987533, 764016151, 906097321, 914067307, 954169327

1572869, 3145739, 6291469, 12582917, 25165843, 50331653 （适合哈希的素数）

19260817   原根15，是某个很好用的质数
1000000007 原根5
998244353  原根3

- NTT素数表， $g$ 是模 $(r \cdot 2^k+1)$ 的原根

```c++
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

### struct of 自动取模

- 不好用，别用了

```c++
struct mint{
	ll v;
	mint(ll _v){v=_v%mod;}
	mint operator+(const mint &b)const{return v+b.v;}
	mint operator-(const mint &b)const{return v-b.v;}
	mint operator*(const mint &b)const{return v*b.v;}
	explicit operator ll(){return (v+mod)%mod;}
};
```

### struct of 高精度

- 加、减、乘、单精度取模、小于号和等于号（其他不等号用rel_ops命名空间）
- 如果涉及除法，~~那就完蛋~~，用java吧；如果不想打这么多行也用java吧（~~一定要让队友会写java~~）

```c++
struct big{
	vector<ll> a;
	static const ll k=1000000000,w=9;
	int size()const{return a.size();}
	explicit big(const ll &x=0){ //接收ll
		*this=big(to_string(x));
	}
	explicit big(const string &s){ //接收string
		static ll p10[9]={1};
		repeat(i,1,w)p10[i]=p10[i-1]*10;
		int len=s.size();
		int f=(s[0]=='-')?-1:1;
		a.resize(len/w+1);
		repeat(i,0,len-(f==-1))
			a[i/w]+=f*(s[len-1-i]-48)*p10[i%w];
		adjust();
	}
	int sgn(){return a.back()>=0?1:-1;} //这个只能在强/弱调整后使用
	void shrink(){ //收缩（内存不收缩）
		while(size()>1 && a.back()==0)a.pop_back();
	}
	void adjust(){ //弱调整
		repeat(i,0,3)a.push_back(0);
		repeat(i,0,size()-1){
			a[i+1]+=a[i]/k;
			a[i]%=k;
		}
		shrink();
	}
	void final_adjust(){ //强调整
		adjust();
		int f=sgn();
		repeat(i,0,size()-1){
			ll t=(a[i]+k*f)%k;
			a[i+1]+=(a[i]-t)/k;
			a[i]=t;
		}
		shrink();
	}
	explicit operator string(){ //转换成string
		static char s[N]; char *p=s;
		final_adjust();
		if(sgn()==-1)*p++='-';
		repeat_back(i,0,size())
			sprintf(p,i==size()-1?"%lld":"%09lld",abs(a[i])),p+=strlen(p);
		return s;
	}
	const ll &operator[](int n)const{ //访问
		return a[n];
	}
	ll &operator[](int n){ //弹性访问
		repeat(i,0,n-size()+1)a.push_back(0);
		return a[n];
	}
};
big operator+(big a,const big &b){
	repeat(i,0,b.size())a[i]+=b[i];
	a.adjust();
	return a;
}
big operator-(big a,const big &b){
	repeat(i,0,b.size())a[i]-=b[i];
	a.adjust();
	return a;
}
big operator*(const big &a,const big &b){
	big ans;
	repeat(i,0,a.size()){
		repeat(j,0,b.size())
			ans[i+j]+=a[i]*b[j];
		ans.adjust();
	}
	return ans;
}
void operator*=(big &a,ll b){ //有时被卡常
	big ans;
	repeat(i,0,a.size())a[i]*=b;
	a.adjust();
}
ll operator%(const big &a,ll mod){
	ll ans=0,p=1;
	repeat(i,0,a.size()){
		ans=(ans+p*a[i])%mod;
		p=(p*a.k)%mod;
	}
	return (ans+mod)%mod;
}
bool operator<(big a,big b){
	a.final_adjust();
	b.final_adjust();
	repeat_back(i,0,max(a.size(),b.size()))
		if(a[i]!=b[i])return a[i]<b[i];
	return 0;
}
bool operator==(big a,big b){
	a.final_adjust();
	b.final_adjust();
	repeat_back(i,0,max(a.size(),b.size()))
		if(a[i]!=b[i])return 0;
	return 1;
}
```

### 表达式求值

```c++
inline int lvl(const string &c){ //运算优先级，小括号要排最后
	if(c=="*")return 2;
	if(c=="(" || c==")")return 0;
	return 1;
}
string convert(const string &in) { //中缀转后缀
	stringstream ss;
	stack<string> op;
	string ans,s;
	repeat(i,0,in.size()-1){
		ss<<in[i];
		if(!isdigit(in[i]) || !isdigit(in[i+1])) //插入空格
			ss<<" ";
	}
	ss<<in.back();
	while(ss>>s){
		if(isdigit(s[0]))ans+=s+" ";
		else if(s=="(")op.push(s);
		else if(s==")"){
			while(!op.empty() && op.top()!="(")
				ans+=op.top()+" ",op.pop();
			op.pop();
		}
		else{
			while(!op.empty() && lvl(op.top())>=lvl(s))
				ans+=op.top()+" ",op.pop();
			op.push(s);
		}
	}
	while(!op.empty())ans+=op.top()+" ",op.pop();
	return ans;
}
ll calc(const string &in){ //后缀求值
	stack<ll> num;
	stringstream ss;
	ss<<in;
	string s;
	while(ss>>s){
		char c=s[0];
		if(isdigit(c))
			num.push((stoll(s))%mod);
		else{
			ll b=num.top(); num.pop();
			ll a=num.top(); num.pop();
			if(c=='+')num.push((a+b)%mod);
			if(c=='-')num.push((a-b)%mod);
			if(c=='*')num.push((a*b)%mod);
			//if(c=='^')num.push(qpow(a,b));
		}
	}
	return num.top();
}
```

### 一些数学结论

#### 约瑟夫问题

- n个人编号0..(n-1)，每次数到k出局，求最后剩下的人的编号
- 线性算法，$O(n)$

```c++
int jos(int n,int k){
	int res=0;
	repeat(i,1,n+1)res=(res+k)%i;
	return res; //res+1，如果编号从1开始
}
```

- 对数算法，适用于k较小情况，$O(k\log n)$

```c++
int jos(int n,int k){
	if(n==1 || k==1)return n-1;
	if(k>n)return (jos(n-1,k)+k)%n; //线性算法
	int res=jos(n-n/k,k)-n%k;
	if(res<0)res+=n; //mod n
	else res+=res/(k-1); //还原位置
	return res; //res+1，如果编号从1开始
}
```

#### 格雷码 汉诺塔

<H5>格雷码</H5>

- 一些性质：
- 相邻格雷码只变化一次
- `grey(n-1)` 到 `grey(n)` 修改了二进制的第 `(__builtin_ctzll(n)+1)` 位
- `grey(0)..grey(2^k-1)` 是k维超立方体顶点的哈密顿回路，其中格雷码每一位代表一个维度的坐标
- 格雷码变换，正 $O(1)$，逆 $O(logn)$

```c++
ll grey(ll n){ //第n个格雷码
	return n^(n>>1);
}
ll degrey(ll n){ //逆格雷码变换，法一
	repeat(i,0,63) //or 31
		n=n^(n>>1);
	return n;
}
ll degrey(ll n){ //逆格雷码变换，法二
	int ans=0;
	while(n){
		ans^=n;
		n>>=1;
	}
	return ans;
}
```

<H5>汉诺塔</H5>

- 假设盘数为n，总共需要移动 `(1<<n)-1` 次
- 第k次移动第 `i=__builtin_ctzll(n)+1` 小的盘子
- 该盘是第 `(k>>i)+1` 次移动
- （可以算出其他盘的状态：总共移动了 `((k+(1<<(i-1)))>>i)` 次）
- 该盘的移动顺序是：
	`A->C->B->A（当i和n奇偶性相同）`
	`A->B->C->A（当i和n奇偶性不同）`

```c++
cin>>n; //层数
repeat(k,1,(1<<n)){
	int i=__builtin_ctzll(k)+1;
	int p1=(k>>i)%3; //移动前状态
	int p2=(p1+1)%3; //移动后状态
	if(i%2==n%2){
		p1=(3-p1)%3;
		p2=(3-p2)%3;
	}
	cout<<"move "<<i<<": "<<"ABC"[p1]<<" -> "<<"ABC"[p2]<<endl;
}
```

- 4个柱子的汉诺塔情况：令 $k=\lfloor n+1-\sqrt{2n+1}+0.5\rfloor$，让前k小的盘子用4个柱子的方法移到2号柱，其他盘子用3个柱子的方法移到4号柱，最后再移一次前k小，最短步数 $f(n)=2f(k)+2^{n-k}-1$

#### Stern-Brocot树 Farey序列

- 分数序列：在 $[\dfrac 0 1,\dfrac 1 0]$ 中不断在 $\dfrac a b$ 和 $\dfrac c d$ 之间插入 $\dfrac {a+c}{b+d}$
- 性质：所有数都是既约分数、可遍历所有既约分数、保持单调递增
- Stern-Brocot树：二叉树，其第 $k$ 行是分数序列第 $k$ 次操作新加的数
- Farey序列：$F_n$ 是所有分子分母 $\le n$ 的既约分数按照分数序列顺序排列后的序列
- $F_n$ 的长度 $=1+\sum\limits_{i=1}^n\varphi(i)$

#### 浮点与近似计算

<H5>数值积分 | 自适应辛普森法</H5>

- 求 $\int_{l}^{r}f(x)\mathrm{d}x$ 的近似值

```c++
lf raw(lf l,lf r){ //辛普森公式
	return (f(l)+f(r)+4*f((l+r)/2))*(r-l)/6;
}
lf asr(lf l,lf r,lf eps,lf ans){
	lf m=(l+r)/2;
	lf x=raw(l,m),y=raw(m,r);
	if(abs(x+y-ans)<=15*eps)
		return x+y-(x+y-ans)/15;
	return asr(l,m,eps/2,x)+asr(m,r,eps/2,y);
}
//调用方法：asr(l,r,eps,raw(l,r))
```

<H5>牛顿迭代法</H5>

- 求 $f(x)$ 的零点：$x_{n+1}=x_n-\dfrac{f(x)}{f'(x)}$
- 检验 $x_{n+1}=g(x_n)$ 多次迭代可以收敛于 $x_0$ 的方法：看 $|g'(x_0)|\le1$ 是否成立

```c++
lf newton(lf n){ //sqrt
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

<H5>others of 浮点与近似计算</H5>

- $\lim\limits_{n\rightarrow\infty}\dfrac{错排(n)}{n!}=\dfrac 1 e,e\approx 2.718281828459045235360287471352$
- $\lim\limits_{n\rightarrow\infty}(\sum\frac 1 n-\ln n)=\gamma\approx 0.577215664901532860606$

#### others of 数学杂项

***

- 埃及分数Engel展开
- 待展开的数为 $x$，令 $u_1=x, u_{i+1}=u_i\times\lceil\dfrac 1 {u_i}\rceil-1$（到0为止）
- 令 $a_i=\lceil\dfrac 1 {u_i}\rceil$
- 则 $x=\dfrac 1{a_1}+\dfrac 1{a_1a_2}+\dfrac 1{a_1a_2a_3}+...$

***

- 三个水杯容量为 $a,b,c$（正整数），$a=b+c$，初始 $a$ 装满水，则得到容积为 $\dfrac a 2$ 的水需要倒 $\dfrac a{\gcd(b,c)}-1$ 次水

***

- 兰顿蚂蚁（白色异或右转，黑色异或左转），约一万步后出现周期为104步的无限重复（高速公路）

***

- 任意勾股数能由复数 $(a+bi)^2\space(a,b∈\Z)$ 得到

***

- 任意正整数 $a$ 都存在正整数 $b,c$ 使得 $a<b<c$ 且 $a^2,b^2,c^2$ 成等差数列：构造 $b=5a,c=7a$

***

- 拉格朗日四平方和定理：每个正整数都能表示为4个整数平方和
- 对于偶素数 $2$ 有 $2=1^2+1^2+0^2+0^2$
- 对于奇素数 $p$ 有 $p=a^2+b^2+1^2+0^2$ （容斥可证）
- 对于所有合数 $n$ 有 $n=z_1^2+z_2^2+z_3^2+z_4^2=(x_1^2+x_2^2+x_3^2+x_4^2)\cdot(y_1^2+y_2^2+y_3^2+y_4^2)$
- 其中 $\begin{cases} z_1=x_1y_1+x_2y_2+x_3y_3+x_4y_4 \\ z_2=x_1y_2-x_2y_1-x_3y_4+x_4y_3 \\ z_3=x_1y_3-x_3y_1+x_2y_4-x_4y_2 \\ z_4=x_1y_4-x_4y_1-x_2y_3+x_3y_2\end{cases}$

***

- 基姆拉尔森公式
- 已知年月日，返回星期几

```c++
int week(int y,int m,int d){
	if(m<=2)m+=12,y--;
	return (d+2*m+3*(m+1)/5+y+y/4-y/100+y/400)%7+1;
}
```

***

求 sum(n/i)

```c++
int f(int n){
	int ans=0;
	int t=sqrt(n);
	repeat(i,1,t+1)ans+=n/i;
	return ans*2-t*t;
}
```