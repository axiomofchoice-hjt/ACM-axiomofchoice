#include <algorithm>
#include <bitset>
#include <cstdint>
#include <deque>
#include <forward_list>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace std {
// template <typename...>
// using to_void = void;

// template <typename T, typename = void>
// struct is_container : false_type {};

// template <typename T>
// struct is_container<T, to_void<decltype(begin(T()))>> : std::true_type {};

inline string toString(const bool &v) { return v ? "true" : "false"; }
inline string toString(const char &v) { return (string) "'" + v + "'"; }
inline string toString(const long double &v) { return to_string(double(v)); }
inline string toString(const string &v) { return "\"" + v + "\""; }
inline string toString(const char *const &v) {
    return (string) "\"" + v + "\"";
}
inline string toString(__uint128_t value) {
    if (value == 0) {
        return "0";
    }
    string res;
    while (value != 0) {
        res += char(value % 10 + '0');
        value /= 10;
    }
    reverse(res.begin(), res.end());
    return res;
}

inline string toString(__int128_t value) {
    if (value < 0) {
        return '-' + toString((__uint128_t)-value);
    } else {
        return toString((__uint128_t)value);
    }
}

template <typename T>
string toString(T value, decltype(to_string(T())) * = nullptr) {
    return to_string(value);
}

template <typename T>
string toString(const T &value, decltype(begin(T())) * = nullptr) {
    string ans = "{";
    bool flag = 0;
    for (const auto &i : value) {
        if (flag) {
            ans += ", ";
        } else {
            flag = 1;
        }
        ans += toString(i);
    }
    ans += "}";
    return ans;
}

// pair
template <typename A, typename B>
string toString(const pair<A, B> &v) {
    return "(" + toString(v.first) + ", " + toString(v.second) + ")";
}
// tuple
template <class Tuple, size_t N>
struct __TuplePrinter {
    static string __to_string(const Tuple &t) {
        return __TuplePrinter<Tuple, N - 1>::__to_string(t) + ", " +
               toString(get<N - 1>(t));
    }
};
template <class Tuple>
struct __TuplePrinter<Tuple, 1> {
    static string __to_string(const Tuple &t) { return toString(get<0>(t)); }
};
template <class... Args>
string toString(const std::tuple<Args...> &t) {
    return "(" + __TuplePrinter<decltype(t), sizeof...(Args)>::__to_string(t) +
           ")";
}
// bitset
template <size_t N>
string toString(const bitset<N> &container) {
    return "<" + container.to_string() + ">";
}
// log
inline void log_rest() { cerr << endl; }
template <typename T, typename... Args>
void log_rest(const T &a, Args... x) {
    cerr << ", " << toString(a);
    log_rest(x...);
}
template <typename T, typename... Args>
void log_first(const T &a, Args... x) {
    cerr << toString(a);
    log_rest(x...);
}
template <typename T>
T log_first(const T &a) {
    cerr << toString(a) << endl;
    return a;
}
#define qwq [] { std::cerr << "qwq" << endl; }()
// #define log(x) [&] { cerr << #x ": " << x << endl; return x; } ()
#define log(...)                            \
    [&] {                                   \
        std::cerr << #__VA_ARGS__ ": ";     \
        return std::log_first(__VA_ARGS__); \
    }()
template <typename T>
void __logarr(const T &a, int n) {
    for (int i = 0; i < n; i++) {
        if (i != 0) cerr << ", ";
        cerr << toString(a[i]);
    }
}
#define logarr(a, n)                   \
    [&] {                              \
        std::cerr << #a ": [";         \
        __logarr(a, n);                \
        std::cerr << "]" << std::endl; \
    }()
template <typename T>
void __logmat(const T &a, int n, int m) {
    for (int i = 0; i < n; i++) {
        cerr << " | ";
        __logarr(a[i], m);
        cerr << " |" << endl;
    }
}
#define logmat(a, n, m)                        \
    [&] {                                      \
        std::cerr << #a ": mat(" << std::endl; \
        __logmat(a, n, m);                     \
        cerr << ")" << endl;                   \
    }()
#define pause                                      \
    [] {                                           \
        std::cerr << "pause at line " << __LINE__; \
        getchar();                                 \
    }()
}  // namespace std
