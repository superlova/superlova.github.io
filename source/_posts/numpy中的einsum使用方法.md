---
title: numpy中的einsum使用方法
date: 2020-05-19 14:52:52
index_img: /img/Einstein.jpg
tags: ['numpy', 'Python', 'einsum']
categories: 
- notes
---
本文将介绍爱因斯坦求和约定，以及在numpy中的使用
<!--more--->

numpy里面有很多奇技淫巧，爱因斯坦求和约定就是其中之一。

爱因斯坦求和约定能够很方便和简介地表示点积、外积、转置、矩阵-向量乘法、矩阵-矩阵乘法等，这在深度学习公式推导中的用处很大。

其实我不认为einsum在numpy中用处很大，我认为其顶多就是一种统一的矩阵运算写法罢了。这种技巧，是在牺牲可读性基础上，对代码的简化。而且由于numpy对其他运算也有进行优化，所以仅凭借爱因斯坦乘数法还不一定能提升代码执行效率。

可能是我还没有体会到高维张量相互计算时的痛苦吧。

先看一下einsum的api：

```python
np.einsum(equation, *arr)
```

最开始需要一个字符串，用以描述想要完成的计算。后面是计算需要的操作数，也就是你的矩阵等。

来看具体的例子：

### 对于向量

```python
arr1 = np.arange(5) # 0,1,2,3,4
arr2 = np.arange(5) # 0,1,2,3,4

```

1. 计算向量所有分量的和，即`np.sum(arr)`。如何利用einsum完成？

```python
np.einsum("i->", arr) # 10
```

在数学上相当于：

$$
c = \sum_{i} a_i,\quad i = 1, 2, \dots
$$

2. 计算两向量内积，即`np.dot(arr1, arr2)`或`np.inner(arr1, arr2)`

```python
# 0*0 + 1*1 + 2*2 + 3*3 + 4*4
np.einsum("i,i->", arr1, arr2) # 30
```

在数学上相当于：

$$
c = \sum_{i} a_i \times b_i,\quad i = 1, 2, \dots
$$

3. 计算两向量逐元素乘积，即`arr1 * arr2`

```python
np.einsum("i,i->i", arr1, arr2) # 0,1,4,9,16
```

在数学上相当于：

$$
c_i = a_i \times b_i,\quad i = 1, 2, \dots
$$

4. 计算两向量外积，即`np.outer(arr1, arr2)`

```python
[[ 0  0  0  0  0]
 [ 0  1  2  3  4]
 [ 0  2  4  6  8]
 [ 0  3  6  9 12]
 [ 0  4  8 12 16]]

np.einsum("i,j->ij", arr1, arr2)
```

在数学上相当于：

$$
c_{i,j} = a_i \times b_j,\quad i,j = 1, 2, \dots
$$

### 对于矩阵

```python
A = np.arange(4).reshape(2,2)
B = np.arange(4,8).reshape(2,2)

[[0 1]
 [2 3]]

[[4 5]
 [6 7]]
```

5. 计算矩阵转置，即`A.T`

```python
[[0 2]
 [1 3]]

print(np.einsum("ij->ji", A))
```

在数学上相当于：

$$
c_{i,j} = a_{j,i},\quad i,j = 1, 2, \dots
$$

6. 计算矩阵各元素求和，即`np.sum(A)`

```python
6

print(np.einsum("ij->", A))
```

在数学上相当于：

$$
c = \sum_{i}\sum_{j}a_{i,j},\quad i,j = 1, 2, \dots
$$

7. 计算矩阵按列求和，即`np.sum(A, axis=0)`

```python
[2 4]

print(np.einsum("ij->j", A))
```

在数学上相当于：

$$
c_{j} = \sum_{i}a_{i,j},\quad i,j = 1, 2, \dots
$$

8. 计算矩阵按行求和，即`np.sum(A, axis=1)`

```python
[1 5]

print(np.einsum("ij->i", A))
```

在数学上相当于：

$$
c_{i} = \sum_{j}a_{i,j},\quad i,j = 1, 2, \dots
$$

9. 求矩阵对角线元素，即`np.diag(A)`

```python
[0 3]

print(np.einsum("ii->i", A))
```

在数学上相当于：

$$
c_{i} = a_{i,i},\quad i = 1, 2, \dots
$$

10. 计算矩阵的迹，即对角线元素和，即`np.trace(A)`

```python
3

print(np.einsum("ii->", A))
```

在数学上相当于：

$$
c = \sum_{i}a_{i,i},\quad i = 1, 2, \dots
$$

11. 计算两矩逐元素乘积，即`A*B`

```python
[[ 0  5]
 [12 21]]

 print(np.einsum("ij,ij->ij", A, B))
```

在数学上相当于：

$$
c_{i,j} = a_{i,j} \times b_{i,j}, i,j = 1, 2, \dots
$$

12. 计算`A*B.T`

```python
[[ 0  6]
 [10 21]]

print(np.einsum("ij,ji->ij", A, B))
```

在数学上相当于：

$$
c_{i,j} = a_{i,j} \times b_{j,i}, i,j = 1, 2, \dots
$$

13. 计算两矩阵乘积`np.dot(A, B)`

```python
[[ 6  7]
 [26 31]]

print(np.einsum("ij,jk->ik", A, B))
```

在数学上相当于：

$$
c_{i,k} = a_{i,j} \times b_{j,k}, i,j = 1, 2, \dots
$$

停一下，停一下。

![](numpy中的einsum使用方法/2020-05-20-09-45-48.png)

你们懂了吗？反正我没有。网上的文章指望着我们光看例子就能学会，这是把我们都当成模型训练了吗？

仔细看一下上面的两个例子，其实每个equation都拥有一个箭头`->`。对应数学公式不难得出，箭头左边对应数学公式右边，箭头右边对应数学公式左边。

比如这个式子：

```python
np.einsum("ij,ji->i", A, B)
```

`"ij,ji->i"`解释成自然语言：将A中第`{i,j}`个元素与B中第`{j,i}`个元素相乘（逗号理解成相乘），结果中没有j分量，只有i分量，所以所有j分量求和。

就是对应这个数学公式：

$$
c_i = \sum_{j}a_{i,j}\times b_{j,i}
$$

实际含义代表：`np.sum(A*B.T, axis=1)`