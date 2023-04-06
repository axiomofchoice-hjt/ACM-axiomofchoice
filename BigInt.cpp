#include <bits/stdc++.h>
using namespace std;

/**
 * @brief 二进制大数类
 * 支持与十进制字符串的互相转换、加法、减法、乘法。
 * （useless，需要普通的高精度板子转到 Math.md > 数学杂项 > struct of 高精度）
 *
 */
struct big {
    using u32 = unsigned;
    using ll = long long;
    using u64 = uint64_t;
    using vtr = vector<u32>;
    /// string 转换为 uint
    static u32 parseInt(const string &s) {
        u32 result = 0;
        for (char ch : s) result = result * 10 + ch - '0';
        return result;
    }
    /// 高精度乘单精度
    static void mulInplace(vtr &x, u32 y) {
        u64 carry = 0;
        for (u32 &i : x) {
            carry = (u64)y * i + carry;
            i = carry;
            carry >>= 32;
        }
        adjust(x, carry);
    }
    /// 高精度加单精度
    static void addInplace(vtr &x, u32 y) {
        u64 carry = y;
        for (u32 &i : x) {
            carry += i;
            i = carry;
            carry >>= 32;
        }
        adjust(x, carry);
    }
    /// 若 carry 非 0，最高位进位 carry；否则去除前导 0
    static void adjust(vtr &x, u32 carry = 0) {
        if (carry)
            x.push_back(carry);
        else
            while (x.size() && x.back() == 0) x.pop_back();
    }
    /// 高精度整除单精度，返回余数
    static u32 divModInplace(vtr &x, u32 m) {
        u64 remain = 0;
        for (int i = x.size() - 1; i >= 0; i--) {
            remain = remain << 32 | x[i];
            x[i] = remain / m;
            remain %= m;
        }
        adjust(x);
        return remain;
    }
    /// 高精度加减高精度，sign 为 -1 表示减法，且必须满足 x > y
    static void addInplace(vtr &x, vtr y, int sign = 1) {
        if (sign == 1 && x.size() < y.size()) swap(x, y);
        ll carry = 0;
        for (u32 i = 0; i < y.size(); i++) {
            carry += x[i] + (ll)sign * y[i];
            x[i] = carry;
            carry >>= 32;
        }
        for (u32 i = y.size(); i < x.size(); i++) {
            carry += x[i];
            x[i] = carry;
            carry >>= 32;
        }
        adjust(x, carry);
    }
    /// 单精度比较
    static int compare(u32 x, u32 y) { return (x == y ? 0 : x > y ? 1 : -1); }
    /// 高精度比较
    static int compare(const vtr &x, const vtr &y) {
        if (x.size() != y.size()) return compare(x.size(), y.size());
        if (x.size() == 0) return 0;
        for (u32 i = x.size() - 1; i != 0; i--)
            if (x[i] != y[i]) return compare(x[i], y[i]);
        return compare(x[0], y[0]);
    }
    /// 高精度乘法 Grade-School Algorithm
    static vtr mul(const vtr &x, const vtr &y) {
        vtr z(x.size() + y.size());
        for (u32 i = 0; i < x.size(); i++) {
            u64 carry = 0;
            for (u32 j = 0; j < y.size(); j++) {
                carry += (u64)x[i] * y[j] + z[i + j];
                z[i + j] = carry;
                carry >>= 32;
            }
            z[i + y.size()] = carry;
        }
        adjust(z);
        return z;
    }
    /// 左移 32 * nInts 位
    static void shiftLeft32Inplace(vtr &mag, u32 nInts) {
        mag.resize(mag.size() + nInts, 0);
        move_backward(mag.begin(), mag.end() - nInts, mag.end());
        fill(mag.begin(), mag.begin() + nInts, 0);
    }
    /// 高精度乘法 Karatsuba Algorithm
    static vtr Karatsuba(const vtr &x, const vtr &y) {
        if (x.size() * y.size() <= 512) {
            return mul(x, y);
        }
        u64 half = (max(x.size(), y.size()) + 1) / 2;
        vtr xl(x.begin(), x.begin() + min(half, x.size()));
        vtr xh(x.begin() + min(half, x.size()), x.end());
        vtr yl(y.begin(), y.begin() + min(half, y.size()));
        vtr yh(y.begin() + min(half, y.size()), y.end());
        vtr p1 = Karatsuba(xh, yh);
        vtr p2 = Karatsuba(xl, yl);
        addInplace(xh, xl);
        addInplace(yh, yl);
        vtr p3 = Karatsuba(xh, yh);  // 接下来计算答案 p1 = p1 * 2^(64h) + (p3 -
                                     // p1 - p2) * 2^(32h) + p2
        addInplace(p3, p1, -1);
        addInplace(p3, p2, -1);
        shiftLeft32Inplace(p1, half);
        addInplace(p1, p3);
        shiftLeft32Inplace(p1, half);
        addInplace(p1, p2);
        return p1;
    }

    /// 符号，1 是正数，-1 是负数，0 是等于 0
    int sign;
    /// 存放绝对值
    vtr mag;
    big() { sign = 0; }
    /// 用 ll 构造
    big(ll val) {
        sign = val > 0 ? 1 : val < 0 ? -1 : 0;
        if (sign == -1) val = -val;
        mag = {u32(val), u32((u64)val >> 32)};
        adjust(mag);
    }
    /// 用 string 构造
    explicit big(string val) {
        const u32 len = val.size();
        sign = (val[0] == '-' ? -1 : 1);
        u32 cursor = (val[0] == '-');
        u32 groupLen = (len - cursor - 1) % 9 + 1;
        while (cursor < len) {
            string group = val.substr(cursor, groupLen);
            cursor += groupLen;
            mulInplace(mag, 1000000000);
            addInplace(mag, parseInt(group));
            groupLen = 9;
        }
        if (mag.size() == 0) sign = 0;
    }
    /// 转换为 string
    string toString() const {
        if (sign == 0) return "0";
        string result;
        vtr t = mag;
        while (t.size()) {
            string k = to_string(divModInplace(t, 1000000000));
            if (t.size()) k = string(9 - k.size(), '0') + k;
            reverse(k.begin(), k.end());
            result += k;
        }
        if (sign == -1) result += '-';
        reverse(result.begin(), result.end());
        return result;
    }
    /// 加法
    friend big operator+(big a, big b) {
        if (a.sign == 0) return b;
        if (b.sign == 0) return a;
        if (a.sign == b.sign) return addInplace(a.mag, b.mag), a;
        int c = compare(a.mag, b.mag);
        if (c == 0) return big();
        if (c > 0)
            return addInplace(a.mag, b.mag, -1), a;
        else
            return addInplace(b.mag, a.mag, -1), b;
    }
    /// 减法
    friend big operator-(big a, big b) {
        b.sign = -b.sign;
        return a + b;
    }
    /// 乘法
    friend big operator*(const big &a, const big &b) {
        big z;
        z.sign = a.sign * b.sign;
        z.mag = Karatsuba(a.mag, b.mag);
        return z;
    }
    /// 小于
    bool operator<(const big &b) const {
        if (sign != b.sign) return sign < b.sign;
        return (compare(mag, b.mag) == -1) ^ (sign == -1);
    }
    /// 等于
    bool operator==(const big &b) const {
        return sign == b.sign && mag == b.mag;
    }
};

signed main() {
    cout << (big("1111111") * big("1111111")).toString() << endl;
    return 0;
}