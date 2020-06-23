---
title: 【论文阅读笔记】RNN-Test Adversarial Testing Framework for Recurrent Neural Network Systems 
date: 2020-03-25 10:37:16
math: true
index_img: /img/testing.png
tags: ['RNN', 'testing', 'RNN-Test']
categories: 
- paper
---

<!--more--->

近年来对于对抗样本的研究主要集中于图像领域，少部分集中在音频领域。而这两种领域的突变攻击，都局限于DNN模型，RNN模型的对抗样本生成则鲜有研究（当时）。

RNN的对抗测试面临某些挑战，概括为三方面。

首先，没有明显的类别标签，就没有规则来识别对抗性输入。对于没有应用于分类的顺序输出，没有标准将输出确定为关于变化程度的错误输出。

其次，对诸如文本之类的顺序输入进行突变很难确保微小的扰动。将扰动应用于离散空间中的单词总是无法获得合法的输入，并且显式修改对于人类是可区分的。

第三，现有的基于CNN的基于神经元的覆盖度量未能考虑RNN结构的特征，因此无法直接采用。
