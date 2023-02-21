#include <bits/stdc++.h>
using namespace std;

/**
 * 大数类，支持与十进制字符串的互相转换、加法、减法、乘法。
 */

struct big {
    using uint = unsigned;
    using ll = long long;
    using ull = uint64_t;
    using vtr = vector<uint>;
    static uint parseInt(const string &s) { // string 转换为 uint
        uint result = 0;
        for (char ch : s) result = result * 10 + ch - '0';
        return result;
    }
    static void mulInplace(vtr &x, uint y) { // 高精度乘单精度
        ull carry = 0;
        for (uint &i : x) {
            carry = (ull)y * i + carry;
            i = carry;
            carry >>= 32;
        }
        adjust(x, carry);
    }
    static void addInplace(vtr &x, uint y) { // 高精度加单精度
        ull carry = y;
        for (uint &i : x) {
            carry += i;
            i = carry;
            carry >>= 32;
        }
        adjust(x, carry);
    }
    static void adjust(vtr &x, uint carry = 0) { // 若 carry 非 0，最高位进位 carry；否则去除前导 0
        if (carry) x.push_back(carry);
        else while (x.size() && x.back() == 0) x.pop_back();
    }
    static uint divModInplace(vtr &x, uint m) { // 高精度整除单精度，返回余数
        ull remain = 0;
        for (int i = x.size() - 1; i >= 0; i--) {
            remain = remain << 32 | x[i];
            x[i] = remain / m;
            remain %= m;
        }
        adjust(x);
        return remain;
    }
    static void addInplace(vtr &x, vtr y, int sign = 1) { // 高精度加减高精度，sign 为 -1 表示减法，且必须满足 x > y
        if (sign == 1 && x.size() < y.size()) swap(x, y);
        ll carry = 0;
        for (uint i = 0; i < y.size(); i++) {
            carry += x[i] + (ll)sign * y[i];
            x[i] = carry;
            carry >>= 32;
        }
        for (uint i = y.size(); i < x.size(); i++) {
            carry += x[i];
            x[i] = carry;
            carry >>= 32;
        }
        adjust(x, carry);
    }
    static int compare(uint x, uint y) { // 单精度比较
        return (x == y ? 0 : x > y ? 1 : -1);
    }
    static int compare(const vtr &x, const vtr &y) { // 高精度比较
        if (x.size() != y.size()) return compare(x.size(), y.size());
        if (x.size() == 0) return 0;
        for (uint i = x.size() - 1; i != 0; i--) if (x[i] != y[i])
            return compare(x[i], y[i]);
        return compare(x[0], y[0]);
    }
    static vtr mul(const vtr &x, const vtr &y) { // 高精度乘法 Grade-School Algorithm
        vtr z(x.size() + y.size());
        for (uint i = 0; i < x.size(); i++){
            ull carry = 0;
            for (uint j = 0; j < y.size(); j++) {
                carry += (ull)x[i] * y[j] + z[i + j];
                z[i + j] = carry;
                carry >>= 32;
            }
            z[i + y.size()] = carry;
        }
        adjust(z);
        return z;
    }
    static void shiftLeft32Inplace(vtr &mag, uint nInts) { // 左移 32 * nInts 位
        mag.resize(mag.size() + nInts, 0);
        move_backward(mag.begin(), mag.end() - nInts, mag.end());
        fill(mag.begin(), mag.begin() + nInts, 0);
    }
    static vtr Karatsuba(const vtr &x, const vtr &y) { // 高精度乘法 Karatsuba Algorithm
        if (x.size() * y.size() <= 512) { return mul(x, y); }
        ull half = (max(x.size(), y.size()) + 1) / 2;
        vtr xl(x.begin(), x.begin() + min(half, x.size()));
        vtr xh(x.begin() + min(half, x.size()), x.end());
        vtr yl(y.begin(), y.begin() + min(half, y.size()));
        vtr yh(y.begin() + min(half, y.size()), y.end());
        vtr p1 = Karatsuba(xh, yh);
        vtr p2 = Karatsuba(xl, yl);
        addInplace(xh, xl); addInplace(yh, yl);
        vtr p3 = Karatsuba(xh, yh); // 接下来计算答案 p1 = p1 * 2^(64h) + (p3 - p1 - p2) * 2^(32h) + p2
        addInplace(p3, p1, -1); addInplace(p3, p2, -1);
        shiftLeft32Inplace(p1, half);
        addInplace(p1, p3);
        shiftLeft32Inplace(p1, half);
        addInplace(p1, p2);
        return p1;
    }

    int sign; // 1 表示正数，-1 表示负数，0 表示等于 0
    vtr mag; // 存放绝对值
    big() { sign = 0; }
    big(ll val) { // 用 ll 构造
        sign = val > 0 ? 1 : val < 0 ? -1 : 0;
        if (sign == -1) val = -val;
        mag = {uint(val), uint((ull)val >> 32)};
        adjust(mag);
    }
    explicit big(string val) { // 用 string 构造
        const uint len = val.size();
        sign = (val[0] == '-' ? -1 : 1);
        uint cursor = (val[0] == '-');
        uint groupLen = (len - cursor - 1) % 9 + 1;
        while (cursor < len) {
            string group = val.substr(cursor, groupLen);
            cursor += groupLen;
            mulInplace(mag, 1000000000);
            addInplace(mag, parseInt(group));
            groupLen = 9;
        }
        if (mag.size() == 0) sign = 0;
    }
    string toString() const { // 转换为 string
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
    friend big operator+(big a, big b) { // 加法
        if (a.sign == 0) return b;
        if (b.sign == 0) return a;
        if (a.sign == b.sign) return addInplace(a.mag, b.mag), a;
        int c = compare(a.mag, b.mag);
        if (c == 0) return big();
        if (c > 0) return addInplace(a.mag, b.mag, -1), a;
        else return addInplace(b.mag, a.mag, -1), b;
    }
    friend big operator-(big a, big b) { // 减法
        b.sign = -b.sign;
        return a + b;
    }
    friend big operator*(const big &a, const big &b) { // 乘法
        big z;
        z.sign = a.sign * b.sign;
        z.mag = Karatsuba(a.mag, b.mag);
        return z;
    }
    bool operator<(const big &b) const { // 小于
        if (sign != b.sign) return sign < b.sign;
        return (compare(mag, b.mag) == -1) ^ (sign == -1);
    }
    bool operator==(const big &b) const { // 等于
        return sign == b.sign && mag == b.mag;
    }
};

signed main() {
    cout << (big("1111111") * big("1111111")).toString() << endl;
    return 0;
}