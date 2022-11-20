---
title: 【学习笔记】NLP数据增强方法记录
date: 2022-11-20 16:41:35
index_img: /img/nlp_data_aug.png
tags: ['Data Augmentation', 'NLP']
mathjax: true
math: true
categories: 
- notes
---


<!--more--->

## 1. 基于规则的数据增强

https://mp.weixin.qq.com/s?__biz=Mzk0NzMwNjU5Nw==&mid=2247484450&idx=1&sn=953dd0856ee087d52a9c229f68281eb3&chksm=c379ad28f40e243eade7677e16fb4146a257c32c39d8b364952c061bdba0af02b6a49a8d5cb0&scene=178&cur_album_id=2592122537619554305#rd

对于同一个句子，可以采用如下的数据增强方法：

- 随机删除一个词
- 随机选择一个词，用它的同义词替换
- 随机选择两个词，然后交换它们的位置
- 随机选择一个词，然后随机选择一个它的近义词，然后随机插入句子的任意位置

重点是 "同义词替换"，在问题中选择一个不是停止词的词，并用一个随机选择的同义词来替换它。我们可以使用nltk WordNet来产生同义词，生成与原问题等价的新问题。这样就得到了一条新训练数据。

或者我们识别实体后，通过反义词进行替换，则得到负例。

先使用规则的方法令可回答和不可回答的问题平衡；在二者之间达到平衡之后，数据增强即可结束。

## 2. 使用模型生成数据

我们希望得到一个生成模型，将每个篇章与对应的问题以及答案，输入生成模型，模型输出新的问题，这个问题与篇章很相关；或者直接输出新的问题和新的答案（这个比较困难）。

UniLM 模型：

https://xv44586.github.io/2020/08/22/qa-augmentation/

GPT-2模型：

https://zhuanlan.zhihu.com/p/146382050

Pair-to-Sequence 模型：

https://zhuanlan.zhihu.com/p/74514486

## 3. 使用对抗训练方法扰动输入数据


