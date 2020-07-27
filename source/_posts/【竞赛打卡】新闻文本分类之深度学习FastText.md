---
title: 【竞赛打卡】新闻文本分类之深度学习FastText
date: 2020-07-27 21:56:03
index_img: /img/datawhale.jpg
math: true
tags: ['Datawhale', 'Deep Learning', 'Classification', 'FastText']
categories: 
- notes
---
当你写东西或讲话的时候，始终要想到使每个普通工人都懂得，都相信你的号召，都决心跟着你走。要想到你究竟为什么人写东西，向什么人讲话。——《反对党八股》
<!--more--->

在上一章节，我们使用传统机器学习算法来解决了文本分类问题，从本章开始我们将尝试使用深度学习方法。与传统机器学习不同，深度学习既提供特征提取功能，也可以完成分类的功能。

本次学习我们主要介绍FastText。

fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：
1、fastText在保持高精度的情况下加快了训练速度和测试速度
2、fastText不需要预训练好的词向量，fastText会自己训练词向量
3、fastText两个重要的优化：层级 Softmax、N-gram

```python
import fasttext
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2, verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in df_train.iloc[-5000:]['text']]
print(f1_score(df_train['label'].values[-5000:].astype(str), val_pred, average='macro'))
0.8256254253081777
```