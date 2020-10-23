---
title: 【竞赛打卡】leetcode打卡：查找算法2
date: 2020-08-27 19:32:22
index_img: /img/datawhale.jpg
math: true
tags: ['Datawhale', 'leetcode', 'algorithm', 'search']
categories: 
- notes
---
见多了优秀的文章，再写博客的时候就会感叹自己的学识浅薄。
<!--more--->
## 1 两数之和

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        if (nums.empty()) return {};
        unordered_map<int, int> hash;
        vector<int> res;

        // construct dict
        for (int i = 0; i < nums.size(); ++i) {
            hash[nums[i]] = i;
        }

        for (int i = 0; i < nums.size(); ++i) {
            int temp = target - nums[i];
            auto iter = hash.find(temp);
            if (iter != hash.end() && iter->second != i) {
                res.push_back(i);
                res.push_back(iter->second);
                break;
            }
        }
        return res;
    }
};
```

思想：用hash把待查数组保存起来，这样再次查找的时间就是O(1)了。

## 15 三数之和

```py
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        size = len(nums)
        if size < 3: return res

        nums.sort()

        for i in range(size-2):
            if i > 0 and nums[i] == nums[i-1]: continue
            j = i + 1
            k = size - 1
            while j < k:
                ans = nums[i] + nums[j] + nums[k]
                if (ans > 0): k = k - 1
                elif (ans < 0): j = j + 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    while j < size and nums[j] == nums[j-1]: j += 1
                    k -= 1
                    while k >= 0 and nums[k] == nums[k+1]: k -= 1

        return res
```

思想：用三个下标`i,j,k`遍历所有可能。首先排序，然后不断缩小i、j和k的区间。

## 16 最接近的三数之和

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        const int size = nums.size();
        if (size <= 3) return std::accumulate(nums.begin(), nums.end(), 0); // 0是累加的初值

        std::sort(nums.begin(), nums.end());

        int result = nums[0] + nums[1] + nums[2]; // 初值
        for (int i = 0; i < size - 2; ++i) {
            int j = i + 1;
            int k = size - 1;
            while (j < k) {
                int temp = nums[i] + nums[j] + nums[k];
                if (std::abs(target - temp) < std::abs(target - result)) {
                    result = temp;
                }
                if (result == target) { // 直接找到了
                    return result;
                }
                if (temp > target) {
                    --k; // temp太大，需要缩小右边界
                } else {
                    ++j; // temp太小，需要缩小左边界
                }
            }
        }
        return result;
    }
};
```

思路：还是利用三个下标`i,j,k`遍历全部数组。中途不断保存和target最近的temp值。

## 18 四数之和

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(),nums.end());
        vector<vector<int> > res;
        if(nums.size()<4) return res;
        
        int a,b,c,d,_size=nums.size();
        for(a=0;a<_size-3;a++){
            if(a>0&&nums[a]==nums[a-1]) continue;      //确保nums[a] 改变了
            for(b=a+1;b<_size-2;b++){
                if(b>a+1&&nums[b]==nums[b-1])continue;   //确保nums[b] 改变了
                c=b+1,d=_size-1;
                while(c<d){
                    if(nums[a]+nums[b]+nums[c]+nums[d]<target)
                        c++;
                    else if(nums[a]+nums[b]+nums[c]+nums[d]>target)
                        d--;
                    else{
                        res.push_back({nums[a],nums[b],nums[c],nums[d]});
                        while(c<d&&nums[c+1]==nums[c])      //确保nums[c] 改变了
                            c++;
                        while(c<d&&nums[d-1]==nums[d])      //确保nums[d] 改变了
                            d--;
                        c++;
                        d--;
                    }
                }
            }
        }
        return res;
    }
};
```

思路：四指针。