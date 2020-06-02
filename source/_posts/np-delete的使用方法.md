---
title: numpy中delete的使用方法
date: 2020-05-31 13:10:08
math: false
index_img: /img/numpy.png
tags: ['numpy', 'Python', 'slice']
categories: 
- notes
---
本文将介绍np.delete中的参数及使用方法
<!--more--->

## Python中列表元素删除

在列表中删除元素，我们可以：

```python
list_a = [1,2,3,4,5]
list_a.pop(-1)
print(list_a) # [1,2,3,4]

del list_a[0]
print(list_a) # [2,3,4]

del list[1:]
print(list_a) # [2]
```

## 在numpy的ndarray中删除元素

numpy中的数组ndarray是定长数组，对ndarray的处理不像对python中列表的处理那么方便。想要删除ndarray中的元素，我们往往只能退而求其次，返回一个没有对应元素的副本。在numpy中我们一般使用delete函数。此外，numpy的delete是可以删除数组的整行和整列的。

简单介绍一下`np.delete`：

```python
numpy.delete(arr, obj, axis=None)
```
* arr：输入数组
* obj：切片，整数，表示哪个子数组要被移除
* axis：删除子数组的轴
* 返回：一个新的子数组

下面是使用举例：

```python
A = np.arange(15).reshape((3,5))
print(A)
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]

B = np.delete(A, 1) # 先把A给ravel成一维数组，再删除第1个元素。
C = np.delete(A, 1, axis=0) # axis=0代表按行操作
D = np.delete(A, 1, axis=1) # axis=1代表按列操作

print(A) # 并没有改变，delete不会操作原数组。
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]

print(B) # 先把A给ravel成一维数组，再删除第1个元素。
[ 0  2  3  4  5  6  7  8  9 10 11 12 13 14]

print(C) # axis=0代表按行操作
[[ 0  1  2  3  4]
 [10 11 12 13 14]]

print(D) # axis=1代表按列操作
[[ 0  2  3  4]
 [ 5  7  8  9]
 [10 12 13 14]]
```

不了解axis的读者可以看我写的[这篇文章](https://superlova.github.io/2020/05/19/numpy%E4%B8%ADaxis%E7%9A%84%E7%AE%80%E5%8D%95%E7%90%86%E8%A7%A3/)。

## 在np.delete的index参数中应用切片操作

index参数必须是个由整数元素组成的列表，内部存放着的整数代表着目标array的下标。

当我想实现删除从第5个到第100个之间的所有元素时，不能使用slice，这就比较尴尬了。

```python
In [5]: np.delete(x, [3:6])
  File "<ipython-input-215-0a5bf5cc05ba>", line 1
    np.delete(x, [3:6])
                   ^
SyntaxError: invalid syntax
```

我们没办法在函数参数部分让其接受slice。怎么解决呢？我们可以把参数从`[start:end]`换成`A[start:end]`吗？

```python
A = np.arange(10)*2
print(A)
[ 0  2  4  6  8 10 12 14 16 18]

B = np.delete(A, A[1:4]) # 搞错了吧！预期结果：0 8 10 12 14 16 18
print(B)
[ 0  2  6 10 14 16 18]
```

我们这段代码能够执行，但是不是我们想要的结果。什么原因呢？是因为np.delete的index参数接受的是下标数组，而A[1:4]=[2,4,6]，那么np.delete就忠实地执行了删除第2、4、6个元素的任务。但我们的本意只是想删除下标从1到4的元素而已。

```python
D = np.delete(A, [1,2,3])
print(D) # [ 0  8 10 12 14 16 18]
```

要想使用slice，可以采用下列方式：1. `slice`函数或者`range`函数；2. `np.s_`

```python
C = np.delete(A, slice(1,4))
print(C) # [ 0  8 10 12 14 16 18]

E = np.delete(A, np.s_[1:4])
print(E) # [ 0  8 10 12 14 16 18]
```

其实`np.s_[1:4]`只不过是很方便产生slice(1,4)的一种方式而已。

## 其他实用的方法

除此之外，我们还可以采用mask的方式选择原数组中的元素组成新数组

```python
mask = np.ones((len(A),), dtype=bool)
mask[[1,2,3]] = False
print(A[mask]) # [ 0  8 10 12 14 16 18]
```

或者干脆采用数组拼合的方式

```python
G = np.empty(len(A)-len(A[1:4]), dtype=int)
G[0:1] = A[0:1]
G[1:len(G)] = A[4:]
print(G) # [ 0  8 10 12 14 16 18]
```

后两种方法不像我们想象的那么没用，反而很常见，尤其是mask方法。