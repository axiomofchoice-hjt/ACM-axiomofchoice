# ACM-Axiomofchoice

## 简介

这是一个 ACM-XCPC 竞赛的技能树、代码仓库，由 HDU 吾有一數名之曰誒 (int a)、吾有一數名之曰嗶 (int b) 队员 Axiomofchoice 维护。

## 导航

|                                                   文件名                                                    |                介绍                |
| :---------------------------------------------------------------------------------------------------------: | :--------------------------------: |
|          [**Math.md**](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/Math.md)          |                数学                |
|         [**Graph.md**](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/Graph.md)         |                图论                |
|      [**Geometry.md**](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/Geometry.md)      |              计算几何              |
| [**Datastructure.md**](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/Datastructure.md) |              数据结构              |
|        [**Others.md**](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/Others.md)        | 搜索、动态规划、字符串、编程技巧等 |
|    [**Conclusion.md**](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice/blob/master/Conclusion.md)    |                结论                |

## 代码风格

- 之前：OI 风格（随便起的名），非必要不用空格、到处压行。
- 现在：Google Style，4 缩进且有压行。
- 模板里两种风格共存。

代码中的预定义：

- 循环宏，`repeat (i, a, b)` 表示 `i` 从 `a` 循环到 `b - 1`，`repeat_back (i, a, b)` 表示 `i` 从 `b - 1` 反着循环到 `a`。

```cpp
#define repeat(i, a, b) for (int i = (a), _ = (b); i < _; i++)
#define repeat_back(i, a, b) for (int i = (b) - 1, _ = (a); i >= _; i--)
```

- 宏 `fi` 表示 `first`，`se` 表示 `second`。
- 类型 `ll` 表示 `long long`，`lf` 表示 `double`，`pii` 表示 `pair<int, int>`。
- `rnd()` 会生成一个 64 位无符号整数范围内的随机数。
- 宏 `mst(a, x)` 表示 `memset(a, x, sizeof(a))`。
- 以前图方便用 `v << e` 表示 `v.push_back(e)`，正在逐渐减少这种写法。
