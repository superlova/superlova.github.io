---
title: 【竞赛打卡】leetcode打卡：动态规划
date: 2020-08-23 07:52:25
index_img: /img/datawhale.jpg
math: true
tags: ['Datawhale', 'leetcode', 'algorithm', 'Dynamic Programming']
categories: 
- notes
---
见多了优秀的文章，再写博客的时候就会感叹自己的学识浅薄。
<!--more--->

## leetcode 198 打家劫舍

动态规划

设置`dp[i]`为前i个元素中打劫所得最高金额

构建状态转移方程：
```cpp
dp[i] = max(dp[i-1], dp[i-2]+nums[i]);
```
边界条件：
```cpp
dp[0] = nums[0];
dp[1] = max(nums[0], nums[1]);
```

代码：

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.empty()) return 0;
        if (nums.size() == 1) return nums[0];
        vector<int> dp(nums.size(), 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);

        for (int i = 2; i < nums.size(); ++i) {
            dp[i] = max(dp[i-1], dp[i-2]+nums[i]);
        }

        return dp[nums.size()-1];
    }
};
```

## leetcode 674 最长连续递增子序列

不必使用动态规划，直接一遍遍历，碰到`nums[i] < nums[i+1]`就递增计数器，保留计数器最大值即可：

```cpp
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        if (nums.empty()) return 0;
        int count = 1;
        int maxCount = 1;
        for (int i = 0; i < nums.size() - 1; ++i) {
            if (nums[i] < nums[i+1]) {
                ++count;
                maxCount = max(count, maxCount);
            } else {
                count = 1;
            }
        }
        return maxCount;
    }
};
```

后来我悟了，这就是动态规划，只不过我利用`maxCount`来代替了dp数组。我真是个天才（误）！

## leetcode 5 最长回文子串

本来是想定义dp[i][j]，表达字符串从i到j的子串中的最长回文子串的。后来想想这样定义不合适，不如把dp定义为一个bool数组，用来标记从i到j子串是否为回文串即可。

状态转移关系：dp[i][j] = dp[i+1][j-1] && (s[i] == s[j])

即s[i:j]为回文串的条件为s[i+1][j-1]为回文串，且s[i] == s[j]

边界条件：
dp[i][j] = true if i == j
dp[i][j] = false if i > j
dp[i][i+1] = (s[i] == s[i+1])

最后遍历所有dp[i][j]=true的项，返回最长的子串即可

需要注意的一点：我们在遍历双层循环的时候，应该j在外，i在内。想想为什么？如果循环结构是这样的：

```cpp
for (int i = 0; i < s.size(); ++i) {
    for (int j = i+1; j < s.size(); ++j) {
        // TODO
    }
}
```

那 i、j的变化为：
0,0
0,1
0,2
0,3->这里就不对了，因为dp[0][3]需要用到dp[1][2]的值。而i=1时的所有dp都还没求呢。

代码：

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        if (s.empty()) return "";
        int size = s.size();
        vector<vector<bool>> dp(size, vector<bool>(size, false));
        string ans = "";
        for (int j = 0; j < size; ++j) {
            for (int i = 0; i <= j; ++i) {
                if (j == i) dp[i][j] = true;
                else if (j == i+1) dp[i][j] = (s[i] == s[j]);
                else dp[i][j] = (dp[i+1][j-1]) && (s[i] == s[j]);

                if (dp[i][j] && ans.size() < j-i+1) ans = s.substr(i, j-i+1);
            }
        }
        return ans;
    }
};
```

## leetcode 213 打家劫舍2

打家劫舍升级版，贼不能同时打劫头尾。

也好办，拆分成两个动态规划，一个规定不能打劫nums[0]，另一个规定不能打劫nums[size-1]，最后返回更大的那个即可。

代码：

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.empty()) return 0;
        int size = nums.size();
        if (size == 1) return nums[0];
        if (size == 2) return max(nums[0], nums[1]);

        vector<int> dp_robfirst(size, 0);
        vector<int> dp_roblast(size, 0);

        dp_robfirst[0] = nums[0];
        dp_robfirst[1] = max(nums[0], nums[1]);
        for (int i = 2; i < size-1; ++i) {
            dp_robfirst[i] = max(dp_robfirst[i-1], dp_robfirst[i-2] + nums[i]);
        }

        dp_roblast[0] = 0;
        dp_roblast[1] = nums[1];
        for (int i = 2; i < size; ++i) {
            dp_roblast[i] = max(dp_roblast[i-1], dp_roblast[i-2] + nums[i]);
        }

        return max(dp_robfirst[size-2], dp_roblast[size-1]);
    }
};
```

## leetcode 516 最长回文子序列

这次的dp含义为从i到j子串中最长的回文序列长度。

转移方程：

dp[i][j] = dp[i+1][j-1] if s[i] == s[j]
dp[i][j] = max(dp[i+1][j], dp[i][j-1]) if s[i] != s[j]

注意，i从大遍历到小，j从小遍历到大。最后返回dp[0][size-1]

边界条件：dp[i][j] = 1 if i == j

代码：

```cpp
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        if (s.empty()) return 0;
        int size = s.size();
        vector<vector<int>> dp(size, vector<int>(size, 0));

        for (int i = size-1; i >= 0; --i) {
            for (int j = i; j < size; ++j) {
                if (i == j) dp[i][j] = 1;
                else if (s[i] == s[j]) dp[i][j] = dp[i+1][j-1] + 2;
                else dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
            }
        }

        return dp[0][size-1];
    }
};
```

## leetcode 72 编辑距离

代码：

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        int M = word1.size();
        int N = word2.size();
        // if (word1.empty() || word2.empty()) return abs(M-N);
        vector<vector<int>> dp(M+1, vector<int>(N+1, 0));
        //initial
        for (int i = 0; i <= M; ++i) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= N; ++i) {
            dp[0][i] = i;
        }
        //dp
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                if (word1[i-1] == word2[j-1]) {
                    dp[i][j] = min(dp[i - 1][j - 1], 1 + dp[i - 1][j]);
                    dp[i][j] = min(dp[i][j], 1 + dp[i][j - 1]);
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]);
                    dp[i][j] = 1 + min(dp[i][j], dp[i][j - 1]);
                }
            }
        }
        return dp[M][N];
    }
};
```