---
title: lintcode-138 子数组求和问题
date: 2020-03-24 17:30:31
index_img: /img/algorithm.png
tags: ['algorithm', 'lintcode', 'array']
categories: 
- code_exercises
---
- 给定一个整数数组，找到和为零的子数组。
- 你的代码应该返回满足要求的子数组的起始位置和结束位置
<!-- more -->
# lintcode 138：子数组之和
## 题目描述

给定一个整数数组，找到和为零的子数组。你的代码应该返回满足要求的子数组的起始位置和结束位置
**样例 1:**
输入: [-3, 1, 2, -3, 4]
输出: [0,2] 或 [1,3]	
样例解释： 返回任意一段和为0的区间即可。
**样例 2:**
输入: [-3, 1, -4, 2, -3, 4]
输出: [1,5]
**注意事项**
至少有一个子数组的和为 0

## 解题思路

子数组之和问题。看看哪个区间段，段内所有元素加起来刚好等于0（或者某个值）。类似这种求区间段，段内元素满足什么条件的问题。

关键是下面这个结论：

准备一个数组array，其中第i个元素保存num[0]~num[i]之和。那么index_value中一旦出现两个元素其值相同，就说明这两个下标之间所有元素加起来等于0。

$$ \begin{aligned}& if & \sum_{i=0}^{\operatorname{index_1}}nums(i) = \sum_{i=0}^{\operatorname{index_2}}nums(i) \\ & then\quad & return \left[ \operatorname{index_1}+1, \operatorname{index_2} \right] \end{aligned}
$$

举个例子：对于数组`num = [-3, 1, 2, -3, 4]`，我们可以构建array数组如下：

| index | nums[index] | $\sum_{i=0}^{index}nums(i)$ |
| --- | --- | --- |
| 0 | -3 | -3 |
| 1 | 1 | -2 |
| 2 | 2 | 0 |
| 3 | -3 | -3 |
| 4 | 4 | 1 |

返回 [0, 2] 或 [1, 3]

在代码实现中，当我们采用数组实现array时，会受限于查询array内元素的线型时间复杂度，为了找某个值对应的下标，遍历array数组的过程，可能耗费线性复杂度的时间，导致代码TLE超时。

因此我们采用散列，将散列的key设置为前i个元素的和值，value为该值对应的下标位置。

在Python中查找元素，用**字典**可以大大加快查找速度。

## 代码

```python
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        index_value = {}
        accumulator = 0
        for i in range(len(nums)):
            accumulator += nums[i]
            if accumulator in index_value:
                return [index_value[accumulator] + 1, i]
            else:
                index_value[accumulator] = i
        else:
            if accumulator == 0:
                return [0, i]
        return [0, 0]
```

## 变种：子数组元素之和等于k

$$ 
\begin{aligned}
& if & \sum_{i=0}^{\operatorname{index_1}}nums(i) - \bold{k} = \sum_{i=0}^{\operatorname{index_2}}nums(i) \\ 
& then\quad & return \left[ \operatorname{index_1}+1, \operatorname{index_2} \right] \end{aligned}
$$

```python
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """

    def subarraySum(self, nums, obj_num):
        index_value = {}
        accumulator = 0
        for i in range(len(nums)):
            accumulator += nums[i]
            if accumulator - obj_num in index_value:
                return [index_value[accumulator - obj_num] + 1, i]
            else:
                index_value[accumulator] = i
        else:
            if accumulator == 0:
                return [0, i]
        return [0, 0]
```