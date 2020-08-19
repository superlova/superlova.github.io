---
title: 【竞赛打卡】leetcode打卡：分治算法
date: 2020-08-19 18:21:29
index_img: /img/datawhale.jpg
math: true
tags: ['Datawhale', 'leetcode', 'algorithm', 'divide-and-conquer']
categories: 
- notes
---
见多了优秀的文章，再写博客的时候就会感叹自己的学识浅薄。
<!--more--->
## leetcode 50 Pow(x,n)

**题目描述**

实现 pow(x, n) ，即计算 x 的 n 次幂函数。

**算法描述**：

Pow算法有快速幂实现方法。

快速幂，二进制取幂（Binary Exponentiation，也称平方法），是一个在 $O(\log(n))$ 的时间内计算 $a^n$ 的小技巧，而暴力的计算需要 $O(n)$ 的时间。而这个技巧也常常用在非计算的场景，因为它可以应用在任何具有结合律的运算中。其中显然的是它可以应用于模意义下取幂、矩阵幂等运算。

计算a的n次方表示将n个a连乘在一起。然而当a和n太大的时候，这种方法就不太适用了。

不过我们知道，$a^{b+c}=a^b\cdot a^c$，$a^{2b}=(a^b)^2$。

快速幂的想法是，我们将取幂的任务按照指数的 **二进制表示** 来分割成更小的任务。

我们将 n 表示为 2 进制，举一个例子：

$3^{13}=3^{(1101)_2}=3^8\cdot 3^4\cdot 3^1$

因此只需把n转化成二进制，然后分解成对应的权值即可简化计算。

为什么这样能简化计算？因为n的二进制形式长度最长只有$O(\log(n))$。原问题被我们转化成了形式相同的子问题的乘积。

**实现**：

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        if (n < 0) {
            return 1 / myPow(x, -n);
        }
        double base = x;
        double res = 1.0;
        for (; n != 0; n >>= 1) {
            if (n & 0x1) res *= base;
            base *= base;
        }
        return res;
    }
};
```

上面的代码在循环的过程中将二进制位为 1 时对应的幂累乘到答案中。
> https://oi-wiki.org/

## leetcode 53 最大子序和

**题目描述**

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**算法描述**：

给定一个数组，下标从start到end，即[start,end]。要求这其中的最大连续子数组之和。

分解成如下子问题：查找[start,mid]中的最大连续子数组之和，查找[mid,end]中的最大连续子数组之和，最后比较二者哪个更大。

但是最大连续子数组可能是跨越mid的数组，所以递归的时候要额外计算mid及其周围元素之和的最大值，用此值与前面两个区间的值比较。

**实现**

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        return find(nums, 0, nums.size()-1);
    }

    int find(vector<int>& nums, int start, int end) {
        if (start == end) return nums[start];
        if (start > end) return INT_MIN;

        int mid = start + (end - start) / 2;
        int left_max = 0, right_max = 0, ml = 0, mr = 0;

        left_max = find(nums, start, mid-1);
        right_max = find(nums, mid+1, end);

        for (int i = mid-1, sum = 0; i >= start; --i) {
            sum += nums[i];
            if (sum > ml) ml = sum;
        }
        for (int i = mid+1, sum = 0; i <= end; ++i) {
            sum += nums[i];
            if (sum > mr) mr = sum;
        }

        return max(max(left_max, right_max), ml + mr + nums[mid]);
    }
};
```

在代码中，`left_max`为[start,mid)区间内的最大连续子数组和，`right_max`为(mid,end]区间内的最大连续子数组和。

而跨越中心mid的计算方法，则是通过两个for循环，从mid开始一个往前遍历得到最大值`ml`，一个往后遍历得到`mr`，最后得到`ml + mr + nums[mid]`即可。

结果为三者的最大值。

> https://www.bilibili.com/video/BV19t411k7jR

## leetcode 169 多数元素

**题目描述**

给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**算法描述**：

如果数 a 是数组 nums 的众数，如果我们将 nums 分成两部分，那么 a 必定是至少一部分的众数。

这样一来，我们就可以使用分治法解决这个问题：将数组分成左右两部分，分别求出左半部分的众数 a1 以及右半部分的众数 a2，随后在 a1 和 a2 中选出正确的众数。

**实现**

遍历法：

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int res = -1;
        int count = 0;
        for (auto c : nums) {
            if (!count) {
                res = c;
            }
            if (res == c) {
                ++count;
            } else {
                --count;
            }
        }
        return res;
    }
};
```

分治法：

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        return majorityElement_inrange(nums, 0, nums.size()-1);
    }

private:
    int majorityElement_inrange(vector<int>& nums, int lo, int hi) {
        if (lo == hi) return nums[lo];
        if (lo > hi) return -1;
        int mid = lo + (hi - lo) / 2;
        int left_maj = majorityElement_inrange(nums, lo, mid);
        int right_maj = majorityElement_inrange(nums, mid+1, hi);
        return (count_in_range(nums, lo, hi, left_maj) > count_in_range(nums, lo, hi, right_maj)) ? left_maj : right_maj;
    }

    int count_in_range(vector<int>& nums, int lo, int hi, int val) {
        int count = 0;
        for (int i = lo; i <= hi; ++i) {
            if (nums[i] == val) count++;
        }
        return count;
    }
};
```

> https://leetcode-cn.com/problems/majority-element/solution/duo-shu-yuan-su-by-leetcode-solution/