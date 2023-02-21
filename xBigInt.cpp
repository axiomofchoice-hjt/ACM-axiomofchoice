#include <bits/stdc++.h>
using namespace std;

/**
 * 在 BigInt 的加、减、乘基础上，添加位运算功能。
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
        for (size_t i = x.size() - 1; ~i; i--) {
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
        for (size_t i = 0; i < y.size(); i++) {
            carry += x[i] + (ll)sign * y[i];
            x[i] = carry;
            carry >>= 32;
        }
        for (size_t i = y.size(); i < x.size(); i++) {
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
        for (size_t i = x.size() - 1; i != 0; i--) if (x[i] != y[i])
            return compare(x[i], y[i]);
        return compare(x[0], y[0]);
    }
    static vtr GradeSchool(const vtr &x, const vtr &y) { // 高精度乘法 Grade-School Algorithm
        vtr z(x.size() + y.size());
        for (size_t i = 0; i < x.size(); i++){
            ull carry = 0;
            for (size_t j = 0; j < y.size(); j++) {
                carry += (ull)x[i] * y[j] + z[i + j];
                z[i + j] = carry;
                carry >>= 32;
            }
            z[i + y.size()] = carry;
        }
        adjust(z);
        return z;
    }
    static void shiftLeft32Inplace(vtr &mag, size_t nInts) { // 左移 32 * nInts 位
        mag.resize(mag.size() + nInts, 0);
        move_backward(mag.begin(), mag.end() - nInts, mag.end());
        fill(mag.begin(), mag.begin() + nInts, 0);
    }
    static vtr Karatsuba(const vtr &x, const vtr &y) { // 高精度乘法 Karatsuba Algorithm
        if (x.size() < 80 || y.size() < 80) { return GradeSchool(x, y); }
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
        mag = { uint(val), uint((ull)val >> 32) };
        adjust(mag);
    }
    explicit big(string val) { // 用十进制字符串构造
        const size_t len = val.size();
        sign = (val[0] == '-' ? -1 : 1);
        size_t cursor = (val[0] == '-');
        size_t groupLen = (len - cursor - 1) % 9 + 1;
        while (cursor < len) {
            string group = val.substr(cursor, groupLen);
            cursor += groupLen;
            mulInplace(mag, 1000000000);
            addInplace(mag, parseInt(group));
            groupLen = 9;
        }
        if (mag.size() == 0) sign = 0;
    }
    string toString() const { // 得到十进制字符串
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
        return compare(mag, b.mag) == -1;
    }
    bool operator==(const big &b) const { // 等于
        return sign == b.sign && mag == b.mag;
    }

    bool getBit(size_t n) const { // 二进制第 n 位
        return mag[n >> 5] >> (n & 31) & 1;
    }
    void setBit(size_t n, bool val) { // 二进制第 n 位赋值为 val
        if (val == 1) mag[n >> 5] |= 1u << (n & 31);
        else mag[n >> 5] &= ~1u << (n & 31);
    }
    void fromBinary(string val) { // 用二进制字符串构造
        mag.assign((val.size() + 31) >> 5, 0);
        for (size_t i = 0; i < val.size(); i++) if (val[i] == '1')
            setBit(val.size() - 1 - i, 1);
        adjust(mag);
        sign = bool(mag.size());
        if (val[0] == '-') sign = -sign;
    }
    string toBinary() const { // 得到二进制字符串
        if (sign == 0) return "0";
        size_t cursor = mag.size() * 32 - 1;
        while (~cursor && getBit(cursor) == 0) cursor--;
        string result; if (sign == -1) result += '-';
        while (~cursor) result += char('0' + getBit(cursor--));
        return result;
    }
    string toComplBinary() const { // 得到补码二进制字符串
        if (sign >= 0) return toBinary();
        big x = *this;
        complImplace(x);
        size_t cursor = mag.size() * 32 - 1;
        while (~cursor && x.getBit(cursor) == 1) cursor--;
        string result = "1...11";
        while (~cursor) result += char('0' + x.getBit(cursor--));
        return result;
    }
    static size_t getHighBit(const vtr &mag) { // 最高位的位置，无视符号
        if (mag.size() == 0) return -1;
        return __lg(mag.back()) + (mag.size() - 1) * 32;
    }
    static size_t getLowBit(const vtr &mag) { // 最低位的位置，无视符号
        for (size_t i = 0; i < mag.size(); i++) if (mag[i])
            return __builtin_ctz(mag[i]) + i * 32;
        return -1;
    }
    size_t countBits() const { // 统计二进制 1 的个数，无视符号
        size_t result = 0;
        for (uint i : mag) result += __builtin_popcount(i);
        return result;
    }
    static void leftShiftInplace(vtr &mag, size_t n) { // 左移
        if ((n & 31) == 0) return shiftLeft32Inplace(mag, n >> 5);
        mag.resize(mag.size() + (n >> 5) + 1, 0);
        for (size_t i = mag.size() - 1; i >= (n >> 5) + 1; i--) {
            mag[i] = mag[i - (n >> 5)] << (n & 31) | mag[i - (n >> 5) - 1] >> (32 - (n & 31));
        }
        mag[n >> 5] = mag[0] << (n & 31);
        fill(mag.begin(), mag.begin() + (n >> 5), 0);
        adjust(mag);
    }
    static void rightShiftInplace(vtr &mag, size_t n) { // 右移
        for (size_t i = 0; i < mag.size() - (n >> 5) - 1; i++) {
            mag[i] = mag[i + (n >> 5)] >> (n & 31) | mag[i + (n >> 5) + 1] << (32 - (n & 31));
        }
        mag[mag.size() - (n >> 5) - 1] = mag[mag.size() - 1] >> (n & 31);
        fill(mag.end() - (n >> 5), mag.end(), 0);
        adjust(mag);
    }
    friend big operator<<(big x, size_t n) { // 左移，无视符号
        leftShiftInplace(x.mag, n);
        return x;
    }
    friend big operator>>(big x, size_t n) { // 右移，无视符号
        rightShiftInplace(x.mag, n);
        if (x.mag.size() == 0) x.sign = 0;
        return x;
    }
    friend void complImplace(big &x) { // 符号不变，mag 转换为补码
        if (x.sign >= 0) return;
        uint delta = 1;
        for (size_t i = 0; i < x.mag.size(); i++) {
            x.mag[i] = (~x.mag[i]) + delta;
            if (x.mag[i]) delta = 0;
        }
    }
    template <class Op>
    static big bitOperation(big x, big y) { // 位运算
        Op op;
        big z; z.mag.assign(max(x.mag.size(), y.mag.size()), 0);
        complImplace(x);
        complImplace(y);
        for (size_t i = 0; i < z.mag.size(); i++)
            z.mag[i] = op(i < x.mag.size() ? x.mag[i] : x.sign >> 1, i < y.mag.size() ? y.mag[i] : y.sign >> 1);
        adjust(z.mag);
        z.sign = op(x.sign, y.sign) < 0 ? -1 : 1;
        if (z.mag.size() == 0) z.sign = 0;
        complImplace(z);
        return z;
    }
    friend big operator&(big x, big y) { // 高精度位与
        return bitOperation<bit_and<uint>>(x, y);
    }
    friend big operator|(big x, big y) { // 高精度位或
        return bitOperation<bit_or<uint>>(x, y);
    }
    friend big operator^(big x, big y) { // 高精度异或
        return bitOperation<bit_xor<uint>>(x, y);
    }
    friend big operator~(big x) { // 高精度按位取反
        if (x.sign == 0) return big(-1);
        complImplace(x);
        for (size_t i = 0; i < x.mag.size(); i++) x.mag[i] = ~x.mag[i];
        adjust(x.mag);
        x.sign = x.sign >= 0 ? -1 : 1;
        if (x.mag.size() == 0) x.sign = 0;
        complImplace(x);
        return x;
    }

    static vtr Knuth(vtr &x, vtr y) {
        if (y.size() == 1) {
            return { divModInplace(x, y[0]) };
        }
        if (x.size() >= 6) {
            size_t ctz = min(getLowBit(x), getLowBit(y));
            if (ctz >= 3 * 32) {
                rightShiftInplace(x, ctz);
                rightShiftInplace(y, ctz);
                vtr r = Knuth(x, y);
                leftShiftInplace(r, ctz);
                return r;
            }
        }
        // 还未实现
    }
};
signed main() {
    big a, b;
    a.fromBinary("101101101000000000000000000000000000000000000000000000000000000");
    cout << (a >> 60).toBinary() << endl;
    cout << (a << 120 >> 180).toBinary() << endl;
    return 0;
}