# 杂项

- [杂项](#杂项)
  - [语言](#语言)
    - [C++11](#c11)
      - [初始代码](#初始代码)
      - [如果没有万能头](#如果没有万能头)
      - [容器](#容器)
      - [其他语法](#其他语法)
      - [神奇特性](#神奇特性)
    - [Java](#java)
    - [Python](#python)
  - [常规算法](#常规算法)
    - [算法基础](#算法基础)
    - [离散化](#离散化)
    - [01 分数规划](#01-分数规划)
    - [任务规划](#任务规划)
      - [Livshits-Kladov 定理](#livshits-kladov-定理)
      - [Johnson 规则](#johnson-规则)
    - [分治](#分治)
      - [逆序数 using 二维偏序](#逆序数-using-二维偏序)
    - [反悔贪心](#反悔贪心)
    - [线段树分治](#线段树分治)
    - [最大空矩阵 using 悬线法](#最大空矩阵-using-悬线法)
    - [搜索](#搜索)
      - [舞蹈链 / DLX](#舞蹈链--dlx)
      - [启发式算法](#启发式算法)
    - [动态规划](#动态规划)
      - [多重背包](#多重背包)
      - [最长不降子序列 / LIS](#最长不降子序列--lis)
      - [数位 dp](#数位-dp)
      - [换根 dp](#换根-dp)
      - [斜率优化](#斜率优化)
      - [四边形优化](#四边形优化)
    - [哈希 / Hash](#哈希--hash)
      - [字符串哈希](#字符串哈希)
      - [质因数哈希](#质因数哈希)
  - [字符串](#字符串)
    - [字符串函数](#字符串函数)
      - [前缀函数 using kmp](#前缀函数-using-kmp)
      - [z 函数 using exkmp](#z-函数-using-exkmp)
      - [马拉车 / Manacher](#马拉车--manacher)
      - [最小表示法](#最小表示法)
      - [后缀数组 / SA](#后缀数组--sa)
      - [height数组](#height数组)
    - [自动机](#自动机)
      - [字典树 / Trie](#字典树--trie)
      - [AC 自动机](#ac-自动机)
      - [后缀自动机 / SAM](#后缀自动机--sam)
  - [技巧](#技巧)
    - [位运算](#位运算)
      - [位运算操作](#位运算操作)
      - [枚举二进制子集](#枚举二进制子集)
    - [浮点数](#浮点数)
    - [常数优化 / 卡常](#常数优化--卡常)
      - [指令集优化](#指令集优化)
      - [快读快写](#快读快写)
      - [STL 手写内存分配器](#stl-手写内存分配器)
      - [取模卡常](#取模卡常)
      - [其他优化](#其他优化)
    - [扩栈](#扩栈)
    - [卡时暴力](#卡时暴力)
    - [对拍 / bat](#对拍--bat)
    - [战术分析](#战术分析)

## 语言

### C++11

#### 初始代码

```cpp
// #pragma GCC optimize(3)
#include <bits/stdc++.h>
using namespace std;
#define repeat(i, a, b) for (int i = (a), ib = (b); i < ib; i++)
#define repeat_back(i, a, b) for (int i = (b) - 1, ib = (a);i >= ib; i--)
#define fi first
#define se second
// #define int ll
namespace start {
    typedef long long ll; const int inf = INT_MAX >> 1; const ll INF = INT64_MAX >> 1;
    typedef double lf; const lf pi = acos(-1); typedef pair<int, int> pii;
    mt19937 rnd(chrono::high_resolution_clock::now().time_since_epoch().count());
    const int dx[8] = {0, 0, 1, -1, 1, 1, -1, -1}, dy[8] = {1, -1, 0, 0, 1, -1, 1, -1};
        template <int fast = 1> ll read() { // will detect EOF, set fast = 0 in Hangdian OJ
            if (!fast) { ll x; if (scanf("%lld", &x) != 1) exit(0); return x; }
            ll x = 0, tag = 1; int c = getchar(); for (; !isdigit(c); c = getchar()) { if (c == '-') tag = -1; if (c == EOF) exit(0); }
            for (; isdigit(c); c = getchar()) { x = x * 10 + c - 48; } if (c != EOF) ungetc(c, stdin);
            return x * tag;
        }
        lf readf() { double x; if (scanf("%lf", &x) != 1) exit(0); return x; } // will detect EOF
        template <class T> void write(T x) { if (x < 0) { x = -x, putchar('-'); } if (x >= 10) { write(x / 10); } putchar(x % 10 + 48); }
        void write(double x) { printf("%.12f", x); } void write(long double x) { printf("%.12f", double(x)); }
        template <class A, class B> void write(const pair<A, B> &x) { write(x.first); putchar(' '); write(x.second); }
        template <class T> void print(T x, int e = 1) { write(x); putchar(" \n"[e]); }
        template <class T> void print(const initializer_list<T> &a, int e = 1) { for (auto i = a.begin(), last = prev(a.end()); i != a.end(); i++) print(*i, e && i == last); }
        template <class T> void printArray(const T &a, int e = 1) { for (auto i = a.begin(), last = prev(a.end()); i != a.end(); i++) print(*i, e && i == last); }
        template <class T, int TempN, class TempT> void mst(TempT (&a)[TempN], T el) { fill_n(reinterpret_cast<T *>(a), sizeof(a) / sizeof(T), el); }
        template <class T> T &MAX(T &a, const T &b) { if (a < b) a = b; return a; } template <class T> T &MIN(T &a, const T &b) { if (b < a) a = b; return a; }
        template <class T> vector<T> &operator<<(vector<T> &a, const T &b) { a.push_back(b); return a; }
        template <class T> T sqr(const T &x) { return x * x; }
        template <class T> void sortunique(T &a) { sort(a.begin(), a.end()); a.erase(unique(a.begin(), a.end()), a.end()); }
        void OK(const char *msg = "Yes") { puts(msg); throw 0; } void GG(const char *msg = "No") { puts(msg); throw 0; }
        // int cansel_sync = (ios::sync_with_stdio(0), cin.tie(0), 0);
    const int N = 500010;
    const lf eps = 1e-9;
    const int mod = (1 ? 1000000007 : 998244353);
        int D(int x, int m = mod) { return x >= m ? x - m : x; }
        int mul(int a, int b, int m = mod) { return 1ll * a * b % m; }
        template <class T> int qpow(int a, T b, int m = mod) {
            int ans = 1;
            for (; b; a = mul(a, a, m), b >>= 1) if (b & 1)
                ans = mul(ans, a, m);
            return ans;
        }
} using namespace start;
const int debug = 1;
void Solve() {
}
signed main() {
    // freopen("data.txt", "r", stdin);
    int T = 1; // T = read();
    repeat (ca, 1, T + 1) {
        try { Solve(); } catch (signed) { }
    }
    return 0;
}
```

```cpp
struct Comb {
    static const int N = 1000010;
    int fac[N], inv[N];
    Comb() {
        fac[0] = 1;
        repeat (i, 1, N)
            fac[i] = 1ll * fac[i - 1] * i % mod;
        inv[N - 1] = qpow(fac[N - 1], mod - 2, mod);
        repeat_back (i, 1, N)
            inv[i - 1] = 1ll * inv[i] * i % mod;
    }
    int operator()(int a, int b) { // a >= b
        if (a < b || b < 0) return 0;
        return 1ll * fac[a] * inv[a - b] % mod * inv[b] % mod;
    }
    int A(int a, int b) { // a >= b
        if (a < b || b < 0) return 0;
        return 1ll * fac[a] * inv[a - b] % mod;
    }
} C;
struct Pow {
    int p[N];
    Pow(int a) {
        p[0] = 1;
        repeat (i, 1, N) p[i] = 1ll * p[i - 1] * a % mod;
    }
    int operator[](int n) { return p[n]; }
};
```

#### 如果没有万能头

```cpp
#include<cstdio>
#include<cmath>
#include<cstring>
#include<cstdlib>
#include<ctime>
#include<cctype>
#include<iostream>
#include<algorithm>
#include<vector>
#include<array>
#include<list>
#include<queue>
#include<string>
#include<set>
#include<map>
// #include<chrono>
// #include<unordered_set>
// #include<unordered_map>
```

其他定义

```cpp
#define inline __inline __attribute__((always_inline))
// struct name{bool operator()(const type &x,const type &y){return func(x,y);}}
typedef long double llf;
#define endl "\n"
```

#### 容器

平板电视红黑树

- 其实就是一个支持排名的 set/map，改造成 multiset/multimap 需要加个时间戳
- 第二个参数 null_type 表示这是个 set，如果想要 map 可以换成其他类型
- 第四个参数可以改成 splay_tree_tag（然后就不是红黑树了）
- 第五个参数表示支持排名（tree_policy.hpp 只找到这一个类）

```cpp
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
tree<pii,null_type,less<pii>,rb_tree_tag,tree_order_statistics_node_update> t; // 定义
t.insert({x,i+1}); // ----------------- 插入 x，用时间戳 i+1 标注（因为不是 multiset）
t.erase(t.lower_bound({x,0})); // ----- 删除 x（删除单个元素）
t.order_of_key({x,0})+1; // ----------- x 的排名（小于 x 的元素个数 + 1）
t.find_by_order(x-1)->first; // ------- 排名为 x 的元素（第 x 小的数）
prev(t.lower_bound({x,0}))->first; // - x 的前驱（小于x且最大）
t.lower_bound({x+1,0})->first; // ----- x 的后继（大于x且最小）
t.join(t2); // ------------------------ 将 t2 并入 t， t2 清空，前提是取值范围不相交
t.split(v,t2); // --------------------- 小于等于 v 的元素属于 t，其余的属于 t2
```

平板电视优先队列

- `pairing_heap_tag` 配对堆，应该是可并堆里最快的
- `thin_heap_tag` 斐波那契堆
- `std::priority_queue` 不合并就很快

```cpp
#include<ext/pb_ds/priority_queue.hpp>
using namespace __gnu_pbds;
__gnu_pbds::priority_queue<int,less<int>,pairing_heap_tag> h; // 大根堆
h.push(x); h.top(); h.pop();
h.join(h2); // 将h2并入h，h2清空
```

rope

- 可能是可分裂平衡树

```cpp
#include <ext/rope>
using namespace __gnu_cxx;
rope<int> r; // 块状链表
r.push_back(n);
r.insert(pos,n); // 插入一个元素
r.erase(pos,len); // 区间删除
r.copy(pos,len,x); // 区间赋值到x
r.replace(pos,x); // 相当于r[pos]=x;
r.substr(pos,len); // 这是啥不会用
r[pos] // 只能访问不能修改
r.clear();
rope<int> *his[N]; his[0]=new rope<int>(); his[i]=new rope<int>(*his[i-1]); // 据说O(1)拷贝，一行可持久化
```

#### 其他语法

STL

```cpp
a=move(b); // 容器移动（a赋值为b，b清空）
priority_queue<int>(begin,end) // O(n)建堆
```

```cpp
a.find(key) // set,map查找，没找到返回a.end()
a.lower_bound(key) // set,map限制最小值
a.insert(b.begin(),b.end()); // set,map合并（时间复杂度极高）
a.emplace_hint(it,b); // 把插入b，如果位置恰好为it就会很快
```

```cpp
complex<lf> c; complex<lf> c(1,2);// 复数
c.real(),c.imag() // 实部、虚部
```

```cpp
bitset<32> b; // 声明一个32位的bitset
b[n]; b[n]=1; // 访问和修改
b.none(); // 返回是否为空
b.count(); // 返回1的个数
b.to_ullong(); b.to_string(); // 转换
b._Find_first(); // 最低位 1 的下标
b._Find_next(pos); // 位置严格大于 pos 的最低位 1 的下标（不存在就返回 size()）
```

unordered 容器手写 hash 函数

```cpp
struct myhash{
    typedef unsigned long long ull;
    ull f(ull x)const{
        x+=0x321354564536; // 乱敲
        x=(x^(x>>30))*0x3212132123; // 乱敲
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

```cpp
fmod(x) // 浮点取模
tgamma(x) // 计算Γ(x)
atan2(x,y) // 计算坐标(x,y)的极角
hypot(x,y) // 计算sqrt(x^2+y^2)
```

读入

```cpp
getline(cin, [string]); // 读一行
cin.getline(s, sizeof s); // 读一行
cin.putback([char]); ungetc([char], stdin); // 将字符放入缓冲区
cin.unget(); // 退回一个字符
void gets(char s[]) {
    for (int i = 0;; i++) {
        s[i] = getchar();
        if (s[i] == '\n' || s[i] == -1) { s[i] = 0; return; }
    }
}
```

scanf字符串正则化

```cpp
scanf("%ns",str); // 读入n个字符
scanf("%[a-z]",str); // 遇到非小写字母停止
scanf("%[^0-9]",str); // 遇到数字停止，^表示非
scanf("%*[a-z]"); // 也是遇到非小写字母停止，只不过不读入字符串
```

#### 神奇特性

- 在64位编译器（我的编译器）中set每个元素需要额外32字节内存。
- 命名空间rel_ops：之后只定义小于就能用其他所有次序关系符号
- raw strings：`R"(abc\n)"` 相当于 `"abc\\n"`
- 定义数字开头的变量：`type operator ""_name(type number){/*...*/}`（之后 `1_name` 即把1带入上述函数中，参数类型只能是`ull,llf,char,(const char *,size_t)`）
- 高级宏：`__VA_ARGS__`是参数列表(对应`...`)，`__LINE__` 是当前行数，`__FUNCTION__`是当前函数名，`__COUNTER__`是宏展开次数-1
- 位域：`struct{int a:3;};` 表示struct里a占3 bit，可以节省空间
- %n：`scanf,printf` 中 %n 将读入/输出的字符个数写入变量

### Java

```java
import java.util.*;
import java.math.BigInteger;
import java.math.BigDecimal;
public class Main {
    static Scanner sc = new Scanner(System.in);
    public static void main(String[] args) {
    }
}
```

```java
import java.util.*;
import java.io.*;
import java.math.BigInteger;
import java.math.BigDecimal;
public class Main {
    static Scanner cin = new Scanner(System.in);
    static PrintStream cout = System.out;
    public static void main(String[] args) {
    }
}
```

- 编译运行 `java Main.java`
- 编译 `javac Main.java` // 生成 Main.class
- 运行 `java Main`

数据类型

```java
int // 4字节有符号
long // 8字节有符号
double, boolean, char, String
```

```java
final double PI = 3.14; // final 类似 c++ 的 const
var n = 1; // var 类似 c++ 的 auto
long 型常量结尾加 L，如 1L
```

数组

```java
int[] arr = new int[100]; // 数组（可以是变量）
int[][] arr = new int[10][10]; // 二维数组
arr.length; // 数组长度，没有括号
Arrays.binarySearch(arr, l, r, x) // 在 arr[[l,r-1]] 中二分查找 x，若存在返回位置，不存在返回 -lowerbound-1
Arrays.sort(arr, l, r); Arrays.sort(arr); // 对 arr[[l, r-1]] 排序
Arrays.fill(arr, l, r, x); Arrays.fill(arr, x); // 填充 arr[[l, r-1]] 为x
```

极其麻烦的结构体 compareTo 重载

```java
public class Main {
public static class node implements Comparable<node> {
    int x;
    public node() {}
    public int compareTo(node b) {
        return x - b.x;
    }
}
static Scanner sc;
public static void main(String[] args) {
    sc = new Scanner(System.in);
    int n = sc.nextInt();
    node[] a = new node[n];
    for (int i = 0; i < n; i++) {
        a[i] = new node();
        a[i].x = sc.nextInt();
    }
    Arrays.sort(a);
    for (node i : a)
        System.out.print(i.x + " ");
    System.out.println();
}
}
```

输出

```java
System.out.print(x);
System.out.println();
System.out.println(x);
System.out.printf("%.2f\n", d); // 格式化
```

输入

```java
import java.util.Scanner;
Scanner sc = new Scanner(System.in); // 初始化
String s = sc.nextLine(); // 读一行字符串
int n = sc.nextInt(); // 读整数
double d = sc.nextDouble(); // 读实数
sc.hasNext() // 是否读完
```

String

```java
s1.equals(s2) // 返回是否相等
s1.compareTo(s2) // s1 > s2 返回 1，s1 < s2 返回 -1，s1 == s2 返回 0
s1.contains(s2) // 返回是否包含子串 s2
s1.indexOf(s2, begin = 0) // 查找子串位置
s1.substring(l, r) // 返回子串 [l,r-1]
s1.charAt(x) // 类似 c++ 的 s1[x]
s1.length() // 返回长度
s1 + s2 // 返回连接结果
String.format("%d", n) // 返回格式化结果
```

StringBuffer / StringBuilder

```java
StringBuffer s1 = new StringBuffer();
StringBuffer s1 = new StringBuffer("A");
s1.append("A"); // 类似 c++ 的 s1+="A";
s1.reverse(); // 反转字符串
s1.replace(l, r, "A"); // 将子串 [l, r - 1] 替换为 "A" (delete + insert)
s1.charAt(x); // 类似 c++ 的 s1[x]
s1.setCharAt(x, c); // 类似 c++ 的 s1[x] = c;
```

Math

```java
// 不用 import 就能用下列函数
Math.{sqrt, sin, atan, abs, max, min, pow, exp, log, PI, E}
```

Random

```java
import java.util.Random;
Random rnd = new Random(); // 已经把时间戳作为了种子
rnd.nextInt();
rnd.nextInt(n); // [0, n)
```

BigInteger

```java
import java.math.BigInteger;
BigInteger n;
new BigInteger("0"); // String 转换为 BigInteger
BigInteger.valueOf(I) // int / long 转换为 BigInteger
BigInteger.ZERO / ONE / TEN // 0 / 1 / 10
sc.nextBigInteger() // 读入，括号里可以写进制
BigInteger[] arr = new BigInteger[10];
n1.intValue / longValue() // 转换为 int / long（太大直接丢失高位，不报异常，报异常可以后面加 Exact）
n1.add / subtract / multiply(n2) // 加 减 乘
n1.divide / mod(n2) // 整除 取模
n1.compareTo(n2) // n1 > n2 返回 1，n1 < n2 返回 -1，n1 == n2 返回 0
n1.equals(n2) // 判断相等
n1.abs() n1.pow(I) n1.gcd(n2) n1.max / min(n2) // int I
n1.modPow(n2,n3) n1.modInverse(n2) // 快速幂 逆元(gcd(n1,n2)!=1)
n1.toString(I) // 返回 I 进制字符串
n1.and / or / xor / shiftLeft / shiftRight(n2) // 位运算
n1.testBit(I) n1.setBit / clearBit / flipBit(I) // 取二进制第 I 位，操作第 I 位
n1.bitCount() // 二进制 1 的个数
// 运算时 n2 一定要转换成 BigInteger
```

BigDecimal

```java
import java.math.BigDecimal;
n1.divide(n2, 2, BigDecimal.ROUND_HALF_UP) // 保留 2 位（四舍五入）
// 貌似没有 sqrt 等操作，都得自己实现 qwq
```

### Python

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
None # 空值
True False # 布尔值
```

字符串str

```python
eval(s); # 表达式求值
ord(c); chr(c); # 字符和编码的转换
int("123"); str(123); # 数字与字符串的转换
```

列表list

```python
a=[]; a.append(x); a.pop(); a[x]; # 最后一个不存在会报错
len(a); # 返回 size
a.sort(); # 排序
a.sort(reverse=True); # 降序排序
a.sort(key=str); # 按照指定函数转换后比较
```

字典dict

```python
- a={}; a={x:y}; a[x]=y; a[x]; # 最后一个不存在会报错
- a.get(x); # 不存在返回 None
- a.pop(x); # 不存在会报错
```

集合set

```python
a=set(); a.add(x); a.remove(x); # 最后一个不存在会报错
a&b; a|b; a-b; a^b; # 集合的交、并、差、对称差
```

高级操作

```python
enumerate(<list>) # 返回 list 的每个元素变成 (下标, 元素) 后的生成器
zip(<list>, <list>) # 返回两个 list 对应元素组成二元组的生成器
```

## 常规算法

### 算法基础

- STL自带算法

```cpp
fill(begin,end,element); // 填充
fill_n(begin,n,element); // 填充
iota(begin,end,t); // 递增填充（赋值为t,t+1,t+2,...）
copy(a_begin,a_end,b_begin); // 复制（注意会复制到b里）
reverse(begin,end); // 翻转
```

```cpp
nth_element(begin,begin+k,end); // 将第k+1小置于位置k，平均O(n)
binary_search(begin,end,key,[less]) // 返回是否存在
upper_bound(begin,end,key,[less]) // 返回限制最小值地址
lower_bound(begin,end,key,[less]) // 返回严格限制最小值地址
merge(a_begin,a_end,b_begin,b_end,c_begin,[less]); // 归并a和b（结果存c）
inplace_merge(begin,begin+k,end); // 归并（原地保存）
next_permutation(begin,end); prev_permutation(begin,end); // 允许多重集，返回不到底，使用方法 do{/*...*/}while(next_permutation(begin,end));
min_element(begin,end); max_element(begin,end); // 返回最值的指针
for_each(begin,end,work); // 每个元素进行work操作
```

```cpp
auto it=back_inserter(a); // it=x表示往a.push_back(x)
```

- 进制转换

```cpp
strtol(str,0,base),strtoll // 返回字符数组的base进制数
stol(s,0,base),stoll // 返回字符串的base进制数
sscanf,sprintf // 十进制
to_string(n) // 十进制
```

- 随机数

```cpp
#include<random>
mt19937 rnd(time(0));
// 巨佬定义：mt19937 rnd(chrono::high_resolution_clock::now().time_since_epoch().count());
cout<<rnd()<<endl; // 范围是unsigned int
rnd.max() // 返回最大值
lf rndf(){return rnd()*1.0/rnd.max();}
lf rndf2(){return rndf()*2-1;}
```

- 手写二分/三分

```cpp
while(l<=r){
    int mid=(l+r)/2; // l+(r-l)/2
    if(ok(mid))l=m+1; else r=m-1; // 小的值容易ok
}
// 此时r是ok的右边界
```

```cpp
while (l < r) {
    int x = (l + r) / 2, y = x + 1; // l + (r - l) / 2
    if (work(x) < work(y)) l = x + 1; else r = y - 1; // 最大值
}
// 此时l和r均为极值点
```

```cpp
#define f(x) (-x*x+23*x)
const lf ph=(sqrt(5)-1)/2; // 0.618
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

```cpp
int m1=-inf,m2=-inf;
repeat(i,0,n){m1=max(m1,a[i]); if(m1>m2)swap(m1,m2);}
// m1即次大值
```

- 朴素 mex

```cpp
struct Mex {
    int vis[N], dcnt;
    Mex() { dcnt = 1; }
    void push(int x) { vis[x] = dcnt; }
    int calc() { for (int i = 0; ; i++) if (vis[i] != dcnt) return dcnt++, i; }
} mex;
```

- 快速排序 / 快排

```cpp
void qsort(int l, int r) {
    if (l >= r) return;
    swap(a[l], a[l + rnd() % (r - l)]);
    int x = l, y = r, mid = a[l];
    while (x < y) {
        while (x < y && a[y] >= mid) y--;
        swap(a[x], a[y]);
        while (x < y && a[x] < mid) x++;
        swap(a[x], a[y]);
    }
    qsort(l, x - 1);
    qsort(x + 1, r);
}
```

### 离散化

- 从小到大标号并赋值，$O(n\log n)$，~~是个好东西~~。

```cpp
void disc(int a[],int n){
    vector<int> b(a,a+n);
    sort(b.begin(),b.end());
    b.erase(unique(b.begin(),b.end()),b.end());
    repeat(i,0,n)
        a[i]=lower_bound(b.begin(),b.end(),a[i])-b.begin(); // 从0开始编号
}
```

```cpp
void disc(int a[],int n,int d){ // 把距离>d的拉近到d
    vector<int> b(a,a+n);
    sort(b.begin(),b.end());
    b.erase(unique(b.begin(),b.end()),b.end());
    vector<int> c(b.size()); c[0]=0; // 从0开始编号
    repeat(i,1,b.size())
        c[i]=c[i-1]+min(d,b[i]-b[i-1]);
    repeat(i,0,n)
        a[i]=c[lower_bound(b.begin(),b.end(),a[i])-b.begin()];
}
```

```cpp
struct Disc{ // 离散化后a[]的值互不相同，但是a[i]与a[j]∈[d.pre[a[i]],d.nxt[a[i]]]在离散化前是相同的
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

### 01 分数规划

- n 个物品，都有两个属性 $a_i$ 和 $b_i$，任意取 k 个物品使它们的 $\dfrac {\sum a_j}{\sum b_j}$ 最大
- 解：二分答案
- m 是否满足条件即判断 $\dfrac {\sum a_j}{\sum b_j}\ge m$，即 $\sum(a_j-mb_j)\ge 0$
- 因此计算 $c_i=a_i-mb_i$ ，取前 k 个最大值看它们之和是否 $\ge 0$
- 如果限制条件是 $\sum b_j\ge W$，则将 $b_j$ 看成体积，$a_j-mb_j$ 看成价值，转换为背包dp

```cpp
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

### 任务规划

#### Livshits-Kladov 定理

- 给出 n 个任务，第 i 个任务花费 $t_i$ 时间，该任务开始之前等待 t 时间的代价是 $f_i(t)$ 个数，求一个任务排列方式，最小化代价 $\sum_{i=1}^n f_j(\sum_{j=1}^{i-1}t_i)$
- Livshits-Kladov定理：当 $f_i(t)$ 是一次函数 / 指数函数 / 相同的单增函数时，最优解可以用排序计算
- 一次函数：$f_i(t)=c_it+d_i$，按 $\dfrac {c_i}{t_i}$ 升序排列
- 指数函数：$f_i(t)=c_ia^t+d_i$，按 $\dfrac{1-a^{t_i}}{c_i}$ 升序排列
- 相同的单增函数：按 $t_i$ 升序排序

#### Johnson 规则

- n 个任务和两个机器，每个任务必须先在 A 上做 $a_i$ 分钟再在 B 上做 $b_i$ 分钟，求最小时间。

```cpp
sort(p,p+n,[](int x,int y){
    int xx=a[x]>b[x],yy=a[y]>b[y];
    if(xx!=yy)return xx<yy;
    if(xx==0)return a[x]<a[y];
    else return b[x]>b[y];
});
```

### 分治

#### 逆序数 using 二维偏序

- $O(n\log n)$

```cpp
void merge(int l, int r) { // include, exclude
    static int t[N];
    if (r - l <= 1) return;
    int mid = l + (r - l) / 2;
    merge(l, mid);
    merge(mid, r);
    int p = l, q = mid, s = l;
    while (s < r) {
        if (p >= mid || (q < r && a[p] > a[q])) {
            t[s++] = a[q++];
            ans += mid - p; // here
        }
        else
            t[s++] = a[p++];
    }
    repeat (i, l, r) a[i] = t[i];
}
```

### 反悔贪心

- 例：第 i 天要么获得 $A_i$ 元要么获得 $B_i$ 元和 1 积分，$A_i\ge 0$，$B_i$ 可能 $<0$，问获得积分最大值
- 能拿 B 就拿 B，不然就把之前某个 B 换成 A 然后拿这次的 B（如果更优的话）

```cpp
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

### 线段树分治

- 支持添删边的离线操作
- 对时间建立线段树，每个结点开一个vector。对一条边，添加到删除的时间区间，插入到线段树中。最后对线段树 DFS 一遍统计答案，向下走即添边，向上走即撤销，用可撤销数据结构维护
- 复杂度为 $O(n\log n)$ 乘以可撤销数据结构复杂度
- 如果元素对询问的贡献可以分开计算，那么只要对线段树每个节点进行统计即可，不需要可撤销数据结构

```cpp
struct seg {
    vector<pii> a;
    int l, r; seg *lc, *rc;
    void init(int, int);
    void update(int x, int y, pii xx) {
        if (x > r || y < l) return;
        if (x <= l && y >= r) { a.push_back(xx); return; }
        lc->update(x, y, xx);
        rc->update(x, y, xx);
    }
    void dfs() {
        for (auto i : a) d.join(i.fi, i.se);
        if (l == r) ans[l] = now;
        if (l != r) { lc->dfs(); rc->dfs(); }
        repeat (i, 0, a.size()) d.undo();
    }
} tr[N * 2], *pl;
void seg::init(int _l, int _r) {
    l = _l, r = _r;
    if (l == r) { return; }
    int m = (l + r) >> 1;
    lc = ++pl; lc->init(l, m);
    rc = ++pl; rc->init(m + 1, r);
}
void init(int l, int r) {
    pl = tr; tr->init(l, r);
}
```

### 最大空矩阵 using 悬线法

- 求01矩阵中全是0的最大连续子矩阵（面积最大）$O(nm)$
- 此处障碍物是正方形。如果障碍只是一些整点，答案从 $ab$ 变为 $(a+1)(b+1)$

```cpp
int n,m,a[N][N],l[N][N],r[N][N],u[N][N];
int getlm(){
    int ans=0;
    repeat(i,0,n)
    repeat(k,0,m)
        l[i][k]=r[i][k]=u[i][k]=(a[i][k]==0);
    repeat(i,0,n){
        repeat(k,1,m)
        if(a[i][k]==0)
            l[i][k]=l[i][k-1]+1; // 可以向左延伸几格
        repeat_back(k,0,m-1)
        if(a[i][k]==0)
            r[i][k]=r[i][k+1]+1; // 可以向右延伸几格
        repeat(k,0,m)
        if(a[i][k]==0){
            if(i!=0 && a[i-1][k]==0){
                u[i][k]=u[i-1][k]+1; // 可以向上延伸几格
                l[i][k]=min(l[i][k],l[i-1][k]);
                r[i][k]=min(r[i][k],r[i-1][k]); // 如果向上延伸u格，lr对应的修改
            }
            ans=max(ans,(l[i][k]+r[i][k]-1)*u[i][k]);
        }
    }
    return ans;
}
```

### 搜索

#### 舞蹈链 / DLX

精确覆盖

- 在 01 矩阵中找到某些行，它们两两不相交，且它们的并等于全集。
- xy 编号从 1 开始！$O(\exp)$，结点数 $<5000$。

```cpp
int n,m;
vector<int> rec; // dance后存所有选中的行的编号
struct DLX{
    #define rep(i,i0,a) for(int i=a[i0];i!=i0;i=a[i])
    int u[N],d[N],l[N],r[N],x[N],y[N]; // N=10010
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
    bool dance(int dep=1){ // 返回是否可行
        if(r[0]==0)return 1;
        int c=r[0];
        rep(i,0,r)if(sz[c]>sz[i])c=i;
        remove(c);
        rep(i,c,d){
            rep(j,i,r)remove(y[j]);
            if(dance(dep+1)){rec.push_back(x[i]); return 1;}
            rep(j,i,l)resume(y[j]);
        }
        resume(c);
        return 0;
    }
}dlx;
```

- 解数独，一般数独有 $k=9$。

```cpp
int k,sqrtk; 
int f(int x,int y,int co){return (x*k+y)*k+co;}
int ord(char c){return c=='.'?0:isdigit(c)?c-'0':c-'A'+10;}
char chr(int x){return x<10?x+'0':x-10+'A';}
int a[16][16];
void Solve(){
    sqrtk=read(); k=sqrtk*sqrtk;
    n=k*k*k,m=k*k*4; dlx.init();
    repeat(x,0,k){
        static char str[20];
        scanf("%s",str);
        repeat(y,0,k){
            a[x][y]=ord(str[y]);
            int l,r; if(a[x][y])l=r=a[x][y]; else l=1,r=k;
            repeat(co,l,r+1){
                dlx.add(f(x,y,co),x*k+y+1);
                dlx.add(f(x,y,co),x*k+co+k*k);
                dlx.add(f(x,y,co),y*k+co+k*k*2);
                dlx.add(f(x,y,co),(x/sqrtk*sqrtk+y/sqrtk)*k+co+k*k*3);
            }
        }
    }
    dlx.dance();
    repeat(i,0,rec.size()){
        int s=rec[i]-1,x=s/(k*k),y=s/k%k,co=s%k+1;
        a[x][y]=co;
    }
    repeat(i,0,k){
        repeat(j,0,k)printf("%c",chr(a[i][j]));
        puts("");
    }
}
```

重复覆盖

- 在01矩阵中找到最少的行，它们的并等于全集。
- xy编号还是从 1 开始！$O(\exp)$，结点数可能 $<3000$。

```cpp
struct DLX{
    #define rep(i,d,s) for(node* i=s->d;i!=s;i=i->d)
    struct node{
        node *l,*r,*u,*d;
        int x,y;
    };
    static const int M=2e5;
    node pool[M],*h[M],*R[M],*pl;
    int sz[M],vis[M],ans,clk;
    void init(int n,int m){ // 行和列
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
    int solve(){ // 返回最优解
        ans=inf; dfs(0); return ans;
    }
}dlx;
```

#### 启发式算法

A-star

- 定义 $g(v)$ 是 s 到 v 的实际代价，$h(v)$ 是 v 到 t 的估计代价。
- 定义估价函数 $f(v)=g(v)+h(v)$。
- 每次从堆里取出 $f(v)$ 最小的点进行更新。
- 如果满足 $h(v_1)+w(v_2,v_1)\ge h(v_2)$ （存疑）则不需要重复更新同一点，可以用set标记，已标记的不入堆。

模拟退火

- 以当前状态 X 为中心，半径为温度 T 的圆（或球）内选一个新状态 Y。
- 计算 $D=E(Y)-E(X)$ 新状态势能减去当前状态势能。
- 如果 $D<0$ 则状态转移（势能 E 越小越优）。
- 否则状态转移的概率是 $\exp(-\dfrac{KD}{T})$（Metropolis接受准则，~~学不会~~）。
- 最后温度乘以降温系数，返回第一步。
- 需要调 3 个参数：初始温度，终止温度，降温系数（？）。
- 注意点：
  - 让运行时间在TLE边缘试探。
  - 多跑几次退火。
  - 多交几次（注意风险）。
  - 可以先不用某准则，输出中间过程后再调参。

```cpp
lf rndf(){return rnd()*1.0/rnd.max();}
vec rndvec(){return vec(rndf()*2-1,rndf()*2-1);}
// lf E(vec); // 计算势能
struct state{ // 表示一个状态
    vec v; lf e; // 位置和势能
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

### 动态规划

#### 多重背包

- 二进制版，$O(nV\log V)$，V 是总容量。

```cpp
int n,V; ll dp[N];
void push(int val,int v,int c){ // 处理物品(价值=val,体积=v,个数=c)
    for(int b=1;c;c-=b,b=min(b*2,c)){
        ll dv=b*v,dval=b*val;
        repeat_back(j,dv,V+1)
            dp[j]=max(dp[j],dp[j-dv]+dval);
    }
}
// 初始化fill(dp,dp+V+1,0)，结果是dp[V]
```

- 单调队列版，$O(nV)$，V 是总容量。

```cpp
int n,V; ll dp[N];
void push(int val,int v,int c){ // 处理物品(价值=val,体积=v,个数=c)
    static deque< pair<int,ll> > q; // 单调队列，fi是位置，se是价值
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
// 初始化fill(dp,dp+V+1,0)，结果是dp[V]
```

#### 最长不降子序列 / LIS

- 二分查找优化，$O(n\log n)$。

```cpp
fill(dp, dp + n + 1, INF);
repeat (i, 0, n)
    *lower_bound(dp, dp + n, a[i]) = a[i];
return lower_bound(dp, dp + n, INF) - dp;
```

- 最大权不下降子序列
- 即把 $\langle x,v\rangle$ 拆成 v 个 x 后求一遍 LIS。

```cpp
void insert(map<ll,ll> &mp,vector<pii> &push,vector<pii> &pop){ // pii = <x,v>
    for(auto i:push){
        int key=i.fi,val=i.se;
        auto r=mp.lower_bound(key);
        if(r->fi==key)
            r->se+=val,++r;
        else
            mp.emplace_hint(r,key,val);
        auto s=r; int sum=0;
        while(sum+s->se<=val){
            pop.push_back(*s);
            sum+=s->se;
            ++s;
        }
        if(s->fi!=INF)pop.push_back({s->fi,val-sum});
        s->se-=val-sum;
        mp.erase(r,s);
    }
    push.clear();
}
// init: mp.clear(),mp[INF]=INF;
// query: INF-mp[i].rbegin()->se
```

#### 数位 dp

- 记忆化搜索，DFS的参数lim表示是否被限制，lz表示当前位的前一位是不是前导零。
- 复杂度等于状态数。
- 如果每个方案贡献不是1，dp可能要变成struct数组(cnt,sum,....)。

```cpp
ll dp[20][*][2],bit[20]; // 这个[2]表示lz状态，如果lz被使用了的话就需要记录
ll dfs(int pos,ll *,bool lim=1,bool lz=1){
    if(pos==-1)return *; // 返回该状态是否符合要求(0或1)
    ll &x=dp[pos][*];
    if(!lim && x!=-1)return x;
    ll ans=0;
    int maxi=lim?bit[pos]:9;
    repeat(i,0,maxi+1){
        ...// 状态转移
        if(lz && i==0)...// 可能要用lz，其他地方都不用
        ans+=dfs(pos-1,*,
            lim && i==maxi,
            lz && i==0);
    }
    if(!lim)x=ans; // 不限制的时候才做存储
    return ans;
}
ll solve(ll n){
    int len=0;
    while(n)bit[len++]=n%10,n/=10;
    return dfs(len-1,*);
}
signed main(){
    mst(dp,-1); // 在很多时候dp值可以反复使用
    ll t=read();
    while(t--){
        ll l=read(),r=read();
        printf("%lld\n",solve(r)-solve(l-1));
    }
    return 0;
}
```

#### 换根 dp

- 两次 DFS。
- 第一次求所有点所在子树的答案 $dp_v$，此时 $dp_{rt}$ 是 $rt$ 的最终答案。
- 第二次将根转移来算其他点的最终答案，回溯时复原即可。

```cpp
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

#### 斜率优化

- 例：HDOJ3507
- $dp_i=\min\limits_{j=0}^{i-1}[dp_j+(s_i-s_j)^2+M]$
- 考虑 $k<j<i$，j 决策优于 $k\Leftrightarrow dp_j+(s_i-s_j)^2<dp_k+(s_i-s_k)^2$
- 一通操作后 $\dfrac{s_j^2+dp_j-s_k^2-dp_k}{2(s_j-s_k)}<s_i$，即寻找点集 $\{(2s_j,s_j^2+dp_j)\}$ 的下凸包中斜率刚好 $>s_i$ 的线段的左端点，作为决策点

#### 四边形优化

- $dp(l,r)=\min\limits_{k=l}^{r-1}[dp(l,k)+dp(k+1,r)]+w(l,r)$
- 其中 $w(l,r)$ 满足
  - 区间包含单调性：任意 $l \le l' \le r' \le r$ 有 $w(l',r')\le w(l,r)$
  - 四边形不等式：任意 $a \le b \le c \le d$ 有 $w(a,c)+w(b,d)\le w(a,d)+w(b,c)$（若等号恒成立则满足四边形恒等式）
- 决策单调性：令 $m(l,r)$ 为最优决策点（满足 $dp(l,r)=dp(l,m)+dp(m+1,r)+w(l,r)$），则有 $m(l,r-1) \le m(l,r) \le m(l+1,r)$，遍历这个区间可以优化至 $O(n^2)$

```cpp
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

### 哈希 / Hash

#### 字符串哈希

- 如果不需要区间信息，可以调用 `hash<string>()(s)` 获得 ull 范围的 hash 值。
- 碰撞概率：单哈希 $10^6$ 次比较大约有 $\dfrac 1 {1000}$ 概率碰撞。
- 更新生日悖论：n 个字符串，两个串 Hash 值相同概率 $P\approx 1-(1-\tfrac{1}{mod})^{\tfrac{n(n-1)}{2}}$。
- 支持查询子串 hash 值，编号从 0 开始，初始化 $O(n)$，子串查询 $O(1)$。

```cpp
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
        ll q2(int l,int r){ // 循环字符串
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
    pii query2(int l,int r){ // 循环字符串
        return {h1.q2(l,r),h2.q2(l,r)};
    }
};
```

#### 质因数哈希

```cpp
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
// 例：匹配乘积为x^k（x任意）的两个数
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

## 字符串

- （~~我字符串是最菜的~~）
- 寻找模式串p在文本串t中的所有出现

### 字符串函数

#### 前缀函数 using kmp

- $p[x]$ 表示满足 `s.substr(0,k)==s.substr(x-k,k)` 且 $x\not=k$ 的 k 的最大值，$p[0]=0$
- 线性复杂度

```cpp
int p[N];
void kmp(const string &s) {
    p[0] = 0; int k = 0;
    repeat (i, 1, s.length()){
        while (k > 0 && s[i] != s[k]) k = p[k - 1];
        if (s[i] == s[k]) k++;
        p[i] = k;
    }
}
void solve(string s1,string s2){ // 模拟 s1.find(s2)
    kmp(s2+'#'+s1);
    repeat(i,s2.size()+1,s.size())
    if(p[i]==(int)s2.size())
        ans.push_back(i-2*s2.size()); // 编号从 0 开始的左端点
}
```

```cpp
struct KMP{
    string s;
    vector<int> p;
    int get(int k,char c){
        while(k>0 && c!=s[k])k=p[k-1];
        if(c==s[k])k++;
        return k;
    }
    void init(const string &_s){
        p.assign(1,0); int k=0; s=_s+'#';
        repeat(i,1,s.size())p.push_back(k=get(k,s[i]));
    }
    int size(){return s.size()-1;}
};
```

#### z 函数 using exkmp

- $z[x]$ 表示满足 `s.substr(0,k)==s.substr(x,k)` 的 k 的最大值，$z[0]=0$
- 线性复杂度

```cpp
int z[N];
void exkmp(const string &s){ // 求s的z函数
    fill(z,z+s.size(),0); int l=0,r=0;
    repeat(i,1,s.size()){
        if(i<=r)z[i]=min(r-i+1,z[i-l]);
        while(i+z[i]<(int)s.size() && s[z[i]]==s[i+z[i]])z[i]++;
        if(i+z[i]-1>r)l=i,r=i+z[i]-1;
    }
}
```

#### 马拉车 / Manacher

- 预处理为 `"#*A*A*A*A*A*"`
- 线性复杂度

```cpp
int len[N*2]; char s[N*2]; // 两倍内存
int manacher(char s1[]){ // s1可以是s
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
        ans=max(ans,len[i]-1); // 最长回文串长度
    }
    return ans;
}
```

#### 最小表示法

- 求 s 重复无数次的字符串最小后缀的左端点
- 线性复杂度

```cpp
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

#### 后缀数组 / SA

- $sa[i]$ 表示所有后缀中第 i 小的后缀是 `s.substr(sa[i],-1)`
- $rk[i]$ 表示所有后缀中 `s.substr(i,-1)` 是第 $rk[i]$ 小
- 编号从 1 开始！$O(n\log n)$

```cpp
int sa[N],rk[N]; // sa,rk即结果
void get_sa(const string &S){
    static int pre[N*2],id[N],px[N],cnt[N];
    int n=S.length(),m=256;
    const char *const s=S.c_str()-1; // 为了编号从1开始
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
        m=p; // 优化计数排序值域
    }
}
```

- sa可以在同一文本串中在线多次查找模式串（二分查找）

#### height数组

- 定义 $lcp(i,j)=$ 后缀 i 和后缀 j 的最长公共前缀长度
- 定义 $height[i]=lcp(sa[i],sa[i-1])$，$height[1]=0$
- $height[rk[i]]\ge height[rk[i-1]]-1$
- 编号从 1 开始，$O(n)$

```cpp
for(int i=1,k=0;i<=n;i++){
    if(k)k--;
    while(s[i+k]==s[sa[rk[i]-1]+k])k++;
    ht[rk[i]]=k;
}
```

- 不相等的子串个数为 $\dfrac{n(n+1)}{2}-\sum\limits_{i=2}^{n}height[i]$

### 自动机

#### 字典树 / Trie

- 线性复杂度

```cpp
struct trie{
    struct node{
        int to[26],cnt;
        int &operator[](int n){return to[n];}
    }a[N];
    int t;
    void init(){t=0; a[t++]=node();}
    void insert(const char s[]){
        int k=0;
        for(int i=0;s[i];i++){
            int c=s[i]-'a'; // small letter
            if(!a[k][c])a[k][c]=t,a[t++]=node();
            k=a[k][c];
            // a[k].son++; // 子树大小
        }
        a[k].cnt++;
    }
    int query(const char s[]){
        int k=0;
        for(int i=0;s[i];i++){
            int c=s[i]-'a'; // small letter
            if(!a[k][c])return 0;
            k=a[k][c];
        }
        return a[k].cnt;
    }
}t;
```

- 最小异或生成树，01字典树
- 归并，$O(n\log^2n)$

```cpp
struct trie{
    static const int B=30;
    struct node{
        int to[2];
        int &operator[](int n){return to[n];}
    }a[N];
    int t;
    void init(){t=0; a[t++]=node();}
    void insert(ll s){
        int k=0;
        repeat_back(i,0,B){
            int c=(s>>i)&1;
            if(!a[k][c])a[k][c]=t,a[t++]=node();
            k=a[k][c];
        }
    }
    ll query(ll s){ // the min value in {s^t | t in trie}
        int k=0; ll ans=0;
        repeat_back(i,0,B){
            int c=(s>>i)&1;
            if(!a[k][c])c^=1,ans^=1ll<<i;
            k=a[k][c];
        }
        return ans;
    }
}t;
int a[N],n; ll ans;
void merge(int l,int r,ll s){
    if(s==0 || l>=r-1)return;
    int m=lower_bound(a+l,a+r,s)-a;
    if(l<m && m<r){
        t.init(); ll mn=INF;
        repeat(i,l,m)t.insert(a[i]);
        repeat(i,m,r)mn=min(mn,t.query(a[i]));
        ans+=mn;
    }
    repeat(i,m,r)a[i]^=s;
    merge(l,m,s>>1); merge(m,r,s>>1);
}
void solve(){
    sort(a,a+n);
    merge(0,n,1ll<<31);
}
```

- 可持久化字典树，查询与 s 异或后的序列的区间最大值

```cpp
struct trie{
    static const int B=30;
    struct node{
        int to[2]; int lst;
        int &operator[](int n){return to[n];}
    }a[N];
    int t;
    int clone(int k){a[t]=a[k]; return t++;}
    int init(){t=1; return ins(0,0,0);}
    int ins(int rt,ll s,int lst){
        int k=rt=clone(rt);
        repeat_back(i,0,B){
            int c=(s>>i)&1;
            a[k][c]=clone(a[k][c]);
            k=a[k][c];
            a[k].lst=max(a[k].lst,lst);
        }
        return rt;
    }
    ll q(int rt,ll s,int lst){ // the max value in {s^t | t in trie}
        int k=rt; ll ans=0;
        repeat_back(i,0,B){
            int c=(s>>i)&1; c^=1,ans^=1ll<<i;
            if(!a[k][c] || a[a[k][c]].lst<lst)c^=1,ans^=1ll<<i;
            k=a[k][c];
        }
        return ans;
    }
}tr;
int h[N],top;
int query(int l,int r,int s){
    return tr.q(h[r],s,l);
}
void push_back(int s){
    top++; h[top]=tr.ins(h[top-1],s,top);
}
void init(){
    top=0; h[0]=tr.init();
}
```

#### AC 自动机

- 先构建字典树，再构建fail树和字典图
- 线性复杂度

```cpp
struct AC{
    static const int sigma=26,c0='a'; // 小写字母
    struct node{
        int to[sigma],fail,trie_cnt,cnt;
        int &operator[](int x){return to[x];}
        // vector<int> p; // 指向模式串集合
    }a[N];
    int t;
    vector<int> q; // 存了BFS序
    void init(){t=0; a[t++]=node(); q.clear();}
    void insert(const char s[]/*,int ptr*/){ // 将模式串插入字典树
        int k=0;
        for(int i=0;s[i];i++){
            int c=s[i]-c0;
            if(!a[k][c])a[k][c]=t,a[t++]=node();
            k=a[k][c];
        }
        a[k].trie_cnt++;
        // a[k].p.push_back(ptr);
    }
    void build(){ // 构建fail树，将字典树扩展为图
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
    void query(const char s[]){ // 记录文本串中模式串出现次数
        int k=0;
        for(int i=0;s[i];i++){
            int c=s[i]-c0;
            k=a[k][c];
            a[k].cnt++; // fail树上差分
            // for(int kk=k;kk!=0;kk=a[kk].fail)
            // for(auto p:a[kk].p)
                // ans[p]++; // 代替下面的反向遍历，我也不知道什么时候用
        }
        repeat_back(i,0,q.size()){ // 反向遍历BFS序
            a[a[q[i]].fail].cnt+=a[q[i]].cnt; // 差分求和
            // for(auto p:a[q[i]].p)ans[p]=a[q[i]].cnt; // 反馈答案
        }
    }
}ac;
```

#### 后缀自动机 / SAM

- 给定字符串中所有子串对应了SAM中从源点出发的一条路径
- SAM是一个DAG，至多有 $2n-1$ 个结点和 $3n-4$ 条边
- 每个结点表示一个endpos等价类，对应子串长度区间 $[a[a[i].fa].len+1,a[i].len]$
- 构造 $O(n)$，编号从 1 开始（$a[1]$ 表示源点）

```cpp
struct SAM{
    static const int sigma=26,c0='a'; // 小写字母
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
// 构造：for(auto i:s)sam.push(i);
```

## 技巧

### 位运算

#### 位运算操作

- 中点向下取整 `(x+y)/2: (x & y) + ((x ^ y) >> 1)`
- 中点向上取整 `(x+y+1)/2: (x | y) - ((x ^ y) >> 1)`
  - 一般来说用 `x + (y - x >> 1)`
- `abs(n): (n ^ (n >> 31)) - (n >> 31)`
- `max(a,b): b & ((a - b) >> 31) | a & (~(a - b) >> 31)`
- `min(a,b): a & ((a - b) >> 31) | b & (~(a - b) >> 31)`

```cpp
template<typename T> T getbit(T x,int pos){return x>>pos&1;}
template<typename T> void setbit(T &x,int pos,bool z){z?x|=T(1)<<pos:x&=~(T(1)<<pos);}
```

- 位运算函数（不需要头文件）

```cpp
__builtin_ctz(x),__builtin_ctzll(x) // 返回x后导0的个数，x是0则返回32 or 64
__builtin_clz(x),__builtin_clzll(x) // 返回x前导0的个数，x是0则返回32 or 64
__builtin_popcount(x),__builtin_popcountll(x) // 返回x中1的个数
__builtin_parity(x),__builtin_parityll(x) // 返回x中1的个数是否为奇数
```

#### 枚举二进制子集

- 枚举二进制数m的非空子集

```cpp
for(int s=m;s;s=(s-1)&m){
    work(s);
}
```

- 枚举n个元素的大小为k的二进制子集（要保证k不等于0）

```cpp
int s=(1<<k)-1;
while(s<(1<<n)){
    work(s);
    int x=s&-s,y=s+x;
    s=((s&~y)/x>>1)|y; // 这里有一个位反~
}
```

### 浮点数

浮点数操作

```cpp
const lf eps=1e-11;
if(abs(x)<eps)x=0; // 输出浮点数的预处理
```

浮点数常量

```cpp
float 1e38, 有效数字6
double 1e308, 有效数字15
long double 1e4932, 有效数字18
```

### 常数优化 / 卡常

估计函数用时

- `clock()` 可以获取时刻，单位毫秒，运行函数前后的时间之差即为用时。
- 一些巨佬测出来的结论：

```text
- 整数加减：1（个 CPU 周期，下同）
- 整数位运算：1
- 整数乘法：2
- 整数除法：21
- 浮点加减：3
- 浮点除法：35
- 浮点开根：60
```

#### 指令集优化

```cpp
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,popcnt,tune=native")
```

#### 快读快写

```cpp
ll readfast(){
    ll x=0,tag=1; char c=getchar();
    for(;!isdigit(c);c=getchar())if(c=='-')tag=-1;
    for(; isdigit(c);c=getchar())x=x*10+c-48;
    // ungetc(c,stdin);
    return x*tag;
}
template<typename T>
void write(T x){ // slow
    if(x<0)x=-x,putchar('-');
    if(x>=10)write(x/10);
    putchar(x%10^48);
}
```

```cpp
char getc(){ // 代替 getchar，用了这个就不能用其他读入函数如 scanf
    static char now[1<<21],*S,*T;
    if(T==S){T=(S=now)+fread(now,1,1<<21,stdin); if(T==S)return EOF;}
    return *S++;
}
```

#### STL 手写内存分配器

```cpp
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

#### 取模卡常

大量对同一个数取模的优化。（其实根本用不到）

```cpp
struct fastmod {
    using u64 = uint64_t;
    using u128 = __uint128_t;
    int f, l; u64 m, d;
    fastmod(u64 d): d(d) {
        l = 64 - __builtin_clzll(d - 1);
        const u128 one = 1;
        u128 M = ((one << (64 + l)) + (one << l)) / d;
        if(M < (one << 64)) f = 1, m = M;
        else f = 0, m = M - (one << 64);
    }
    friend u64 operator/(u64 n, const fastmod &m) { // get n / d
        if (m.f) return u128(n) * m.m >> 64 >> m.l;
        else {
            u64 t = u128(n) * m.m >> 64;
            return (((n - t) >> 1) + t) >> (m.l - 1);
        }
    }
    friend u64 operator%(u64 n, const fastmod &m) { // get n % d
        return n - n / m * m.d;
    }
};
```

#### 其他优化

```cpp
int mul(int a,int b,int m=mod){ // 汇编，a * b % mod
    int ret;
    __asm__ __volatile__ (
        "\tmull %%ebx\n\tdivl %%ecx\n"
        : "=d"(ret)
        : "a"(a),"b"(b),"c"(m)
    );
    return ret;
}
int bitrev(int x){ // 反转二进制 x
    int z;
    #define op(i) \
        z = ~0u / (1 << i | 1); \
        x = (x & z) << i | (x >> i & z)
    op(1); op(2); op(4); op(8); op(16);
    #undef op
    return x;
}
int kthbit(int x,int k){ // 从低到高第 k 个 1 的位置
    int ans=0,z;
    #define op(b) \
        z=__builtin_popcount(x & ((1 << b) - 1)); \
        if(z<k)k-=z,ans|=b,x>>=b;
    op(16); op(8); op(4); op(2); op(1);
    #undef op
    return ans;
}
```

查表法反转二进制位

```cpp
u32 rev_tbl[65537];
void bitrevinit() {
    rev_tbl[0] = 0;
    for (int i=1; i<65536; ++i)
    rev_tbl[i] = (rev_tbl[i>>1]>>1) | ((i&1)<<15);
}
u32 bitrev(u32 x) {
    return (rev_tbl[x & 65535] << 16) | rev_tbl[x >> 16];
}
```

```text
// （都是听说的）
1ll*a 比 (ll)a 快
取模：x%mod 优化为 x<mod?x:x%mod
减少浮点除法：a/b+c/d 优化为 (a*d+b*c)/(b*d)
精度足够时用ll代替浮点类型
多路并行运算，如 (a+b)+(c+d) 比 ((a+b)+c)+d 快
加上inline，以及强制内联__inline __attribute__((always_inline))
多重for循环时，修改for的顺序保证内存连续访问
多使用局部变量
排序后依次比较可以减少分支预测失败率
```

### 扩栈

```cpp
int size = 128 << 20; // 申请 128M 栈空间 
// 32位 windows
__asm__("movl %0, %%esp\n"::"r"((char*)malloc(size) + size));
// Linux, 64位 windows
__asm__("movq %0, %%rsp\n"::"r"((char*)malloc(size) + size));
// 要用 exit(0) 的方式退出，不然 RE
```

### 卡时暴力

- 在 TLE 的边缘试探。

```cpp
while(clock()<0.9*CLOCKS_PER_SEC){
    // 反复更新最优解
}
```

### 对拍 / bat

```cpp
#include<bits/stdc++.h>
using namespace std;
int main(){
    for(int i=0;;i++){
        if(i%10==0)cerr<<i<<endl;
        system("gen.exe > test.in");
        system("test1.exe < test.in > a.out");
        system("test2.exe < test.in > b.out");
        if(system("fc a.out b.out")){ // linux: diff
            system("pause");
            return 0;
        }
    }
}
```

备选

```cpp
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

### 战术分析

交题前~~给爷~~好好考虑：

- 数据范围。
- 初始化，并测试一下。
- 考虑极端数据！！
- 试一试 `ctrl+Z / ctrl+D` 能否结束。

坑点

- `ll t;` `1<<t` 返回int，必须是 `1ll<<t`。
- `int x;` `x<<y` 的 y 会先对 32 取模。
- `operator<` 的比较内容要写完整。
- 无向图输入时每条边要给两个值赋值 `g[x][y]=g[x][y]=1`。
- 建模的转换函数的宏定义一定要加括号，或者写成函数。
- `islower()` 等函数返回值不一定是 0 或 1。
- 多用相空间角度思考问题。
- 内存比想象的要大一些（有时候 `1e7` 可以塞下）。
- 字符串 `find()` 最好与 `string::npos` 比较，或者 `(signed)s.find(c)==-1`。
- 多年来多项式常数大的原因找到了，加后取模改成 `int D(int x){return x>=mod?x-mod:x;}`（ll 改 int 以及两个分支改一个），减法改成 `D(x-y+mod)`；mod 等等也改成 int 类型。
- 临时变量一定要初始化，特别是 struct。
- 滑动窗口可能要写 `r = max(l, r);` 强制区间为合法区间
