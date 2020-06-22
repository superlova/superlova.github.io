---
title: 【经验分享】IMDb数据集的预处理
date: 2020-06-09 22:48:03
index_img: /img/imdb.png
tags: ['IMDb', 'Python', 'preprocessing']
categories: 
- record
---
IMDb从官网下载与从keras直接调用的处理方法是不同的。
<!--more--->

## 一、IMDb数据集的处理方法

### 1. 官网下载法

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
```

```shell
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxvf aclImdb_v1.tar.gz
!ls
```

对于 v1.0 版数据，其训练集大小是 75 000，而不是 25 000，因为其中还包含 50 000 个用于无监督学习的无标签文档。

在进行后续操作之前，建议先将这 50 000 个无标签文档从训练集中剔除。

```bash
!mkdir aclImdb/train_unlabel
!mv aclImdb/train/unsupBow.feat aclImdb/train_unlabel
!mv aclImdb/train/urls_unsup.txt aclImdb/train_unlabel
!mv aclImdb/train/unsup aclImdb/train_unlabel
```

```python
reviews_train = load_files("aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = load_files("aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
# 删掉HTML换行符
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

print("type of text_train: {}".format(type(text_train))) # 查看训练集类型：list
print("length of text_train: {}".format(len(text_train))) # 查看训练集大小
print("text_train[1]:\n{}".format(text_train[1])) # 查看第二段文本
print("Samples per class (training): {}".format(np.bincount(y_train))) # 查看数据集是否均等
#------------------------------------------------#
type of text_train: <class 'list'>
length of text_train: 25000
text_train[1]:
b'Words can\'t describe how bad this movie is. I can\'t explain it by writing only. You have too see it for yourself to get at grip of how horrible a movie really can be. Not that I recommend you to do that. There are so many clich\xc3\xa9s, mistakes (and all other negative things you can imagine) here that will just make you cry. To start with the technical first, there are a LOT of mistakes regarding the airplane. I won\'t list them here, but just mention the coloring of the plane. They didn\'t even manage to show an airliner in the colors of a fictional airline, but instead used a 747 painted in the original Boeing livery. Very bad. The plot is stupid and has been done many times before, only much, much better. There are so many ridiculous moments here that i lost count of it really early. Also, I was on the bad guys\' side all the time in the movie, because the good guys were so stupid. "Executive Decision" should without a doubt be you\'re choice over this one, even the "Turbulence"-movies are better. In fact, every other movie in the world is better than this one.'
Samples per class (training): [12500 12500]
```

采用词袋模型整理数据

```python
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))
#---------------------------#
X_train:
<25000x74849 sparse matrix of type '<class 'numpy.int64'>'
with 3431196 stored elements in Compressed Sparse Row format>
```

X_train 是训练数据的词袋表示，其形状为 25 000×74 849，这表示词表中包含 74 849 个元素。数据被保存为 SciPy 稀疏矩阵。

访问词表的另一种方法是使用向量器（vectorizer）的 get_feature_name 方法，它将返回一个列表，每个元素对应于一个特征：

feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th feature:\n{}".format(feature_names[::2000]))

Number of features: 74849
First 20 features:
['00', '000', '0000000000001', '00001', '00015', '000s', '001', '003830',
'006', '007', '0079', '0080', '0083', '0093638', '00am', '00pm', '00s','01', '01pm', '02']
Features 20010 to 20030:
['dratted', 'draub', 'draught', 'draughts', 'draughtswoman', 'draw', 'drawback',
'drawbacks', 'drawer', 'drawers', 'drawing', 'drawings', 'drawl',
'drawled', 'drawling', 'drawn', 'draws', 'draza', 'dre', 'drea']
Every 2000th feature:
['00', 'aesir', 'aquarian', 'barking', 'blustering', 'beête', 'chicanery',
'condensing', 'cunning', 'detox', 'draper', 'enshrined', 'favorit', 'freezer',
'goldman', 'hasan', 'huitieme', 'intelligible', 'kantrowitz', 'lawful',
'maars', 'megalunged', 'mostey', 'norrland', 'padilla', 'pincher',
'promisingly', 'receptionist', 'rivals', 'schnaas', 'shunning', 'sparse',
'subset', 'temptations', 'treatises', 'unproven', 'walkman', 'xylophonist']

词表的前 10 个元素都是数字。所有这些数字都出现在评论中的某处，因此被提取为单词。

### 2. 使用keras自带的IMDb数据集

```python
from tensorflow.keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) # 仅保留训练数据中前10000个最经常出现的单词，低频单词被舍弃

print('len of X_train: {}'.format(len(X_train)))
print('shape of X_train: {}'.format(X_train.shape))
print('first of X_train: {}'.format(X_train[0]))
print('training sample per class: {}'.format(np.bincount(y_train)))
#-------------------#
len of X_train: 25000
shape of X_train: (25000,)
first of X_train: [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
training sample per class: [12500 12500]
```

可以看到keras已经把IMDb数据集给提前整理过了。此处每条数据都是一个向量，每个数值代表一个单词。数值的大小代表了该单词在单词表中的位置。显然，每条数据向量的长度不一定相同。

为了方便处理，我们可以规定每条文档的长度为maxlen

```python
from tensorflow.keras.preprocessing import sequence
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
#-------------------#
Pad sequences (samples x time)
x_train shape: (25000, 80)
x_test shape: (25000, 80)
```

训练集中一共25000条文档，其中12500个正类，12500个负类。每个文档都是由80个数字组成的向量。测试集亦然。