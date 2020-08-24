---
title: 【竞赛打卡】leetcode打卡：查找算法
date: 2020-08-24 08:23:08
index_img: /img/datawhale.jpg
math: true
tags: ['Datawhale', 'leetcode', 'algorithm', 'searching']
categories: 
- notes
---
见多了优秀的文章，再写博客的时候就会感叹自己的学识浅薄。
<!--more--->
## leetcode 35 搜索插入位置

给定一个无重复元素的排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

思路：简单的二分搜索。注意边界条件。注意初始化条件是`L = 0, R = nums.size()`。

代码：

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int L = 0, R = nums.size();
        if (nums.empty() || target < nums[L]) return 0;
        if (nums[R-1] < target) return R;

        while (L <= R) {
            int i = (L + R) / 2;
            if (nums[i] < target) {
                L = i + 1;
            } else if (target < nums[i]) {
                R = i - 1;
            } else if (target == nums[i]){
                return i;
            } //else 
        }
        return L;
    }
};
```

## leetcode 202 快乐数

先分享一下直观的解法


检查一个数是快乐数，就不断执行`n=compute(n)`这一步，然后检查n是否为1就行了。但是一旦一个数不是快乐数，则必定是陷于某个数字循环中。比如2这个非快乐数，它的计算过程如下：
```
2
4
16
37
58
89
145
42
20
4 <- 注意这里的4已经出现过
```
我的思路很简单。只需将出现过的n都保存在一个字典中，如果新计算的n已经存在于字典中了，那就意味着陷入了计算循环，非快乐数。

```cpp
class Solution {
public:
    bool isHappy(int n) {
        unordered_map<int, int> hash;
        while (n != 1) {
            n = compute(n);
            auto iter = hash.find(n);
            if (iter != hash.end()) return false;
            ++hash[n];
        }
        return true;
    }

    int compute(int n) {
        int res = 0, bit = 0;
        while (n) {
            bit = n % 10;
            n = n / 10;
            res += bit * bit; 
        }
        return res;
    }
};
```
这个问题还可以转化为检测链表是否存在环路的问题。就可以使用快慢指针法。`compute`函数不变，只需把主函数部分变成：

```cpp
class Solution {
public:
    bool isHappy(int n) {
        int slow = n;
        int fast = compute(n);
        while (slow != fast && fast != 1) {
            slow = compute(slow);
            fast = compute(compute(fast));
        }
        return fast == 1;
    }

    int compute(int n) {
        int res = 0, bit = 0;
        while (n) {
            bit = n % 10;
            n = n / 10;
            res += bit * bit; 
        }
        return res;
    }
};
```

## leetcode 205 同构字符串

将两个字符串翻译为数字，最后比较数字是否相同即可。

```cpp
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        s = translate(s);
        t = translate(t);
        return s == t;
    }

    string translate(string s) {
        int count = 0;
        unordered_map<char, int> hash;
        string res = "";
        for (auto c : s) {
            auto iter = hash.find(c);
            if (iter != hash.end()) res += std::to_string(iter->second);
            else hash[c] = count++;
        }
        return res;
    }
};
```

## leetcode 242 有效的字母异位词

总体思路还是哈希表，保存两个字符串出现的字符类别和次数，如若相等则true。

可以进一步优化，即使用一个哈希表，遍历s的时候构建哈希，遍历t的时候删减对应哈希的元素，如果哈希表的数值低于0，就说明为false。

万一删减不到零呢？其实这种情况是不会出现的，因为我们在循环伊始，检查两字符串的长度必须相同。

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.size() != t.size()) return false;
        vector<int> table(26, 0);
        for (auto c : s) {
            ++table[c - 'a'];
        }
        for (auto c : t) {
            --table[c - 'a'];
            if (table[c - 'a'] < 0) return false;
        }
        return true;
    }
};
```

## leetcode 290 单词规律

还是将其翻译成中间表示，然后比较中间表示是否同一。

```cpp
#include<regex>
#include <iterator>
class Solution {
public:
    bool wordPattern(string pattern, string str) {
        vector<string> str_array;
        std::regex r("\\s+");
        std::sregex_token_iterator pos(str.cbegin(), str.cend(), r, -1); // -1代表你对正则表达式匹配的内容不感兴趣
        std::sregex_token_iterator end;
        for (; pos != end; ++pos) {
            str_array.push_back(*pos);
        }

        if (pattern.size() != str_array.size()) return false;

        unordered_map<char, int> hash_char;
        unordered_map<string, int> hash_string;

        for (int i = 0; i < pattern.size(); ++i) {
            auto iter_char = hash_char.find(pattern[i]);
            auto iter_string = hash_string.find(str_array[i]);
            if (iter_char != hash_char.end() && iter_string != hash_string.end()) {
                if (iter_char->second != iter_string->second) return false;
            } else if (iter_char == hash_char.end() && iter_string == hash_string.end()) {
                hash_char[pattern[i]] = i;
                hash_string[str_array[i]] = i;
            } else return false;
        }

        return true;
    }
};
```

## leetcode 349 两个数组的交集

显然是用hash。

```cpp
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        if (nums1.empty() || nums2.empty()) return {};
        unordered_map<int, int> hash;
        vector<int> res;
        for (auto c : nums1) {
            hash[c] = 1;
        }
        for (auto c : nums2) {
            auto iter = hash.find(c);
            if (iter != hash.end()) hash[c] = 0;
        }
        for (auto iter = hash.begin(); iter != hash.end(); ++iter) {
            if (iter->second == 0) res.push_back(iter->first);
        }
        return res;
    }
};
```


## leetcode 350 两个数组的交集 II

带重复元素了。由于hash本身就可以记录每个元素出现的次数，那么我们每当发现一个元素，执行的不是`hash[nums[i]] = 1`，而是`hash[nums[i]]++`。

```cpp
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        if (nums1.empty() || nums2.empty()) return {};
        unordered_map<int, int> hash;
        vector<int> res;
        for (int i = 0; i < nums1.size(); ++i) {
            ++hash[nums1[i]];
        }
        for (int i = 0; i < nums2.size(); ++i) {
            auto iter = hash.find(nums2[i]);
            if (iter != hash.end() && iter->second != 0) {
                res.push_back(nums2[i]);
                --iter->second;
            }
        }
        return res;
    }
};
```

## leetcode 451 根据字符出现频率排序

```cpp
class Solution {
public:
    string frequencySort(string s) {
        if (s.empty()) return "";
        unordered_map<char, int> hash;

        for (auto c : s) {
            ++hash[c];
        }
        sort(s.begin(), s.end(), [&hash](char lhs, char rhs) {
            return hash[lhs] > hash[rhs] || (hash[lhs] == hash[rhs] && lhs < rhs);
        });
        return s;
    }
};
```

## leetcode 540 有序数组中的单一元素

```cpp
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        if (nums.size() == 1) return nums[0];
        return helper(nums, 0, nums.size()-1);
    }

    int helper(vector<int>& nums, int start, int end) {
        if (end == start) return nums[start];
        int mid = start + (end - start) / 2;
        if (nums[mid-1] == nums[mid]) { // 中点左边相同，须删除中点和左边元素
            int left_len = mid - start - 1;
            int right_len = end - mid;
            if (left_len % 2 != 0) { // 如果删除后左边长度为奇数则递归左边
                return helper(nums, start, mid-2);
            } else { // 如果删除后右边长度为奇数则递归右边
                return helper(nums, mid+1, end);
            }
        } else if (nums[mid] == nums[mid+1]) {  // 中点右边相同，须删除中点和右边元素
            int left_len = mid - start;
            int right_len = end - mid - 1;
            if (left_len % 2 != 0) { // 如果删除后左边长度为奇数则递归左边
                return helper(nums, start, mid-1);
            } else { // 如果删除后右边长度为奇数则递归右边
                return helper(nums, mid+2, end);
            }
        } else return nums[mid];
    }
};
```

## leetcode 410 分割数组的最大值

