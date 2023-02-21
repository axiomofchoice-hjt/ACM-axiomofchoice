// C++ includes used for precompiling -*- C++ -*-

// Copyright (C) 2003-2018 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/** @file stdc++.h
 *  This is an implementation file for a precompiled header.
 */

// 17.4.1.2 Headers

// C
#ifndef _GLIBCXX_NO_ASSERT
#include <cassert>
#endif
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if __cplusplus >= 201103L
#include <ccomplex>
#include <cfenv>
#include <cinttypes>
#include <cstdalign>
#include <cstdbool>
#include <cstdint>
#include <ctgmath>
#include <cuchar>
#include <cwchar>
#include <cwctype>
#endif

// C++
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>

#if __cplusplus >= 201103L
#include <array>
#include <atomic>
#include <chrono>
#include <codecvt>
#include <condition_variable>
#include <forward_list>
#include <future>
#include <initializer_list>
#include <mutex>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#endif

#if __cplusplus >= 201402L
#include <shared_mutex>
#endif

#if __cplusplus >= 201703L
#include <charconv>
// #include <filesystem>
#endif

namespace std {
// bool
string toString(bool v) { return v ? "true" : "false"; }
// char
string toString(char v) { return (string) "'" + v + "'"; }
// int8, int16, int32, int64
string toString(int8_t v) { return to_string(v); }
string toString(int16_t v) { return to_string(v); }
string toString(int32_t v) { return to_string(v); }
string toString(int64_t v) { return to_string(v); }
string toString(uint8_t v) { return to_string(v); }
string toString(uint16_t v) { return to_string(v); }
string toString(uint32_t v) { return to_string(v); }
string toString(uint64_t v) { return to_string(v); }
// float, double, long double
string toString(float v) { return to_string(v); }
string toString(double v) { return to_string(v); }
string toString(long double v) { return to_string(double(v)); }
// int128
string toString(__int128_t v) {
    if (v < 0) return '-' + toString(-v);
    return (v >= 10 ? toString(v / 10) : string()) + char(v % 10 + '0');
}
string toString(__uint128_t v) {
    return (v >= 10 ? toString(v / 10) : string()) + char(v % 10 + '0');
}
// void *
string to_hex(unsigned v) {
    static const char *s = "0123456789abcdef";
    return string() + s[v >> 28 & 15] + s[v >> 24 & 15] + s[v >> 20 & 15] +
           s[v >> 16 & 15] + s[v >> 12 & 15] + s[v >> 8 & 15] + s[v >> 4 & 15] +
           s[v & 15];
}
string to_hex(unsigned long long v) {
    return to_hex(unsigned(v >> 32)) + to_hex(unsigned(v));
}
string toString(const void *const v) { return "0x" + to_hex((size_t)v); }
// string
string toString(const string &v) { return "\"" + v + "\""; }
string toString(const char *const v) { return (string) "\"" + v + "\""; }
// pair
template <typename A, typename B>
string toString(const pair<A, B> &v) {
    return "(" + toString(v.first) + ", " + toString(v.second) + ")";
}
// tuple
template <class Tuple, size_t N>
struct TuplePrinter {
    static string __to_string(const Tuple &t) {
        return TuplePrinter<Tuple, N - 1>::__to_string(t) + ", " +
               toString(get<N - 1>(t));
    }
};
template <class Tuple>
struct TuplePrinter<Tuple, 1> {
    static string __to_string(const Tuple &t) { return toString(get<0>(t)); }
};
template <class... Args>
string toString(const std::tuple<Args...> &t) {
    return "(" + TuplePrinter<decltype(t), sizeof...(Args)>::__to_string(t) +
           ")";
}
// [container]
template <typename Container>
string __toString(const Container &container) {
    string ans;
    bool flag = 0;
    for (const auto &i : container) {
        if (flag)
            ans += ", ";
        else
            flag = 1;
        ans += toString(i);
    }
    return ans;
}
// vector
template <typename Value>
string toString(const vector<Value> &container) {
    return "[" + __toString(container) + "]";
}
// deque
template <typename Value>
string toString(const deque<Value> &container) {
    return "[" + __toString(container) + "]";
}
// array
template <typename Value, size_t N>
string toString(const array<Value, N> &container) {
    return "[" + __toString(container) + "]";
}
// bitset
template <size_t N>
string toString(const bitset<N> &container) {
    return "<" + container.to_string() + ">";
}
// list
template <typename Value>
string toString(const list<Value> &container) {
    return "[" + __toString(container) + "]";
}
// forward_list
template <typename Value>
string toString(const forward_list<Value> &container) {
    return "[" + __toString(container) + "]";
}
// set
template <typename Key, typename Compare>
string toString(const set<Key, Compare> &container) {
    return "{" + __toString(container) + "}";
}
template <typename Key, typename Compare>
string toString(const multiset<Key, Compare> &container) {
    return "{" + __toString(container) + "}";
}
template <typename Key, typename Hash, typename Equal>
string toString(const unordered_set<Key, Hash, Equal> &container) {
    return "{" + __toString(container) + "}";
}
template <typename Key, typename Hash, typename Equal>
string toString(const unordered_multiset<Key, Hash, Equal> &container) {
    return "{" + __toString(container) + "}";
}
// map
template <typename Key, typename Value, typename Compare>
string toString(const map<Key, Value, Compare> &container) {
    return "{" + __toString(container) + "}";
}
template <typename Key, typename Value, typename Compare>
string toString(const multimap<Key, Value, Compare> &container) {
    return "{" + __toString(container) + "}";
}
template <typename Key, typename Value, typename Hash, typename Equal>
string toString(const unordered_map<Key, Value, Hash, Equal> &container) {
    return "{" + __toString(container) + "}";
}
template <typename Key, typename Value, typename Hash, typename Equal>
string toString(const unordered_multimap<Key, Value, Hash, Equal> &container) {
    return "{" + __toString(container) + "}";
}
// orz
void orz_second() { cerr << endl; }
template <typename Arg, typename... Args>
void orz_second(const Arg &a, Args... x) {
    cerr << ", " << toString(a);
    orz_second(x...);
}
template <typename Arg, typename... Args>
void orz_first(const Arg &a, Args... x) {
    cerr << toString(a);
    orz_second(x...);
}
template <typename Arg>
Arg orz_first(const Arg &a) {
    cerr << toString(a) << endl;
    return a;
}
#define qwq [&] { std::cerr << "qwq" << endl; }()
// #define orz(x) [&] { cerr << #x ": " << x << endl; return x; } ()
#define orz(x...)                 \
    [&] {                         \
        std::cerr << #x ": ";     \
        return std::orz_first(x); \
    }()
template <typename Type>
void __orzarr(const Type &a, int n) {
    for (int i = 0; i < n; i++) {
        if (i != 0) cerr << ", ";
        cerr << toString(a[i]);
    }
}
#define orzarr(a, n)                   \
    [&] {                              \
        std::cerr << #a ": [";         \
        __orzarr(a, n);                \
        std::cerr << "]" << std::endl; \
    }()
template <typename Type>
void __orzmat(const Type &a, int n, int m) {
    for (int i = 0; i < n; i++) {
        cerr << " | ";
        __orzarr(a[i], m);
        cerr << " |" << endl;
    }
}
#define orzmat(a, n, m)                        \
    [&] {                                      \
        std::cerr << #a ": mat(" << std::endl; \
        __orzmat(a, n, m);                     \
        cerr << ")" << endl;                   \
    }()
#define orzeach(a)                                       \
    [&] {                                                \
        std::cerr << #a ": ";                            \
        for (const auto &__each_i : a)                   \
            std::cerr << std::toString(__each_i) << " "; \
        std::cerr << endl;                               \
    }()
#define pause                                    \
    [&] {                                        \
        std::cerr << "Line " << __LINE__ << " "; \
        system("pause");                         \
    }()
}  // namespace std
