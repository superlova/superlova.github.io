---
title: NLP学习笔记1
date: 2019-06-21 21:02:07
tags: ['NLP','deep learning']
categories: 
- notes
---
# IMDB数据集探索

实验是在Google Colab上面做的，机器也是用的谷歌云。

```python
# keras.datasets.imdb is broken in 1.13 and 1.14, by np 1.16.3
!pip install tf_nightly
```

安装tensorflow

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)
```

导入相关的包

```python
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

导入IMDB数据集，将其分成四部分，分别是训练集、训练集答案、测试集、测试集答案。

```python
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])
```
探索数据集。可以发现，训练集的每一个训练样本是一个由数字组成的列表。我还纳闷呢，这不应该是文本序列吗？

后来我发现，每个数字对应着不同的单词。比如1对应的是The，4对应的是film。之后只要根据字典mapping一下就好。

```python
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])
```
提取字典。并尝试利用字典还原一个样本的本来面貌。

```python
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
len(train_data[0]), len(train_data[1])
print(train_data[0])
```
简化训练集和测试集，每个样本之提取最多256个单次，不够的就以0来凑。

```python
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
```
安排网络，输入层、池化层、全连接、softmax层。

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
```


```python
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
```
训练！

```python
results = model.evaluate(test_data, test_labels)

print(results)
```
测试结果是准确率87%。