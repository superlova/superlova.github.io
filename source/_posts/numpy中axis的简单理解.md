---
title: numpy中axis的简单理解
date: 2020-05-19 14:52:25
index_img: /img/numpy.png
tags: ['numpy', 'Python', 'axis']
categories: 
- notes
---
本文将介绍numpy中的axis
<!--more--->

我对于numpy中的axis的理解，一直处于似懂非懂、似是而非的状态。看到网上大神的文章，也只能点个赞之后，该不会还是不会。每次看完博客，都会觉得自己懂了；但是每次使用的时候，又要想老半天才行。因此今天我想借此机会，彻底扫清使用numpy时，axis的障碍。

在numpy中，数据的基本类型是array。array有个基本的数据属性，是它的维度。

比如下面的这个array，在逻辑上来看这就是个2维的数据，是一个矩阵。

```python
A = np.random.randint(0, 19, 9).reshape(3, 3)
print(A)
[[12 15  0]
 [ 3  3  7]
 [ 9 18  4]]
```

接下来我要对其中的元素进行求和。

```python
print(np.sum(A))
print(np.sum(A, axis=0))
print(np.sum(A, axis=1))

71
[24 36 11]
[27 13 31]
```
显然，第一个sum是对所有元素累加。第二个参数为axis=0的求和，则是这样计算的：

`A[0][X] + A[1][X] + A[2][X]`
`--|---------|---------|----`

也就是说，axis=0意味着在求和的过程中，只有A的第0个分量会变化，将第0个分量的所有情况穷举出来，再作为被操作元素，求和之。

第0个分量的元素计算完毕、得到一个结果时，计算并没有结束，因为我们的X还有很多种可能。

同理，axis=1时，变化的只有A的第1个（从逻辑上讲是第二个）分量有变化：

`A[X][0] + A[X][1] + A[X][2]`
`-----|---------|---------|-`

把该结论推广到更高维度的数据也不会有问题。我们看一个4维的张量是如何指定axis求和的：

```python
np.random.seed(0)
A = np.random.randint(0, 9, 16).reshape(2, 2, 2, 2)
print("orignal A", A)

orignal A [[[[5 0]
   [3 3]]

  [[7 3]
   [5 2]]]


 [[[4 7]
   [6 8]]

  [[8 1]
   [6 7]]]]
```

```python
print(np.sum(A))
75
```
```python
print(np.sum(A, axis=0))
# 相当于
print(A[0,:,:,:]+A[1,:,:,:])
[[[ 9  7]
  [ 9 11]]

 [[15  4]
  [11  9]]]
```

```python
print(np.sum(A, axis=1))
# 相当于
print(A[:,0,:,:] + A[:,1,:,:])
[[[12  3]
  [ 8  5]]

 [[12  8]
  [12 15]]]
```

```python
print(np.sum(A, axis=2))
# 相当于
print(A[:,:,0,:] + A[:,:,1,:])
[[[ 8  3]
  [12  5]]

 [[10 15]
  [14  8]]]
```


```python
print(np.sum(A, axis=3))
# 相当于
print(A[:,:,:,0]+A[:,:,:,1])
[[[ 5  6]
  [10  7]]

 [[11 14]
  [ 9 13]]]
```