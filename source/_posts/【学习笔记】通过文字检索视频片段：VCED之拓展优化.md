---
title: 【学习笔记】通过文字检索视频片段：VCED之拓展优化
date: 2022-11-27 23:36:31
index_img: /img/datawhale.jpg
tags: ['VCED', 'multimodal']
mathjax: false
math: false
categories: 
- notes
---

本篇文章是跨模态检索工具 VCED 介绍的第六篇。介绍完 VCED 的核心机制与前后端，本次记录下可以优化的点。

<!--more--->

## 1. 目前向量检索使用的是最简单的暴力搜索，所以向量检索花费的时间很慢，这里可以优化成以 Faiss 实现的高性能检索库；

Faiss是Facebook AI团队开源的针对聚类和相似性搜索库，为稠密向量提供高效相似度搜索和聚类，支持十亿级别向量的搜索。Faiss用C++编写，并提供 Numpy 风格的 Python 接口。

为了实现 Faiss 的高效检索功能，我们需要改写 SimpleIndexer 类：

索引初始化时，默认是从本地 sqlite 中进行读取，可以更改为以下读取方法：

```py
if cls is DocumentArray:
    if storage == 'memory':
        from docarray.array.memory import DocumentArrayInMemory

        instance = super().__new__(DocumentArrayInMemory)
    elif storage == 'sqlite':
        from docarray.array.sqlite import DocumentArraySqlite

        instance = super().__new__(DocumentArraySqlite)
    elif storage == 'annlite':
        from docarray.array.annlite import DocumentArrayAnnlite

        instance = super().__new__(DocumentArrayAnnlite)
    elif storage == 'weaviate':
        from docarray.array.weaviate import DocumentArrayWeaviate

        instance = super().__new__(DocumentArrayWeaviate)
    elif storage == 'qdrant':
        from docarray.array.qdrant import DocumentArrayQdrant

        instance = super().__new__(DocumentArrayQdrant)
    elif storage == 'elasticsearch':
        from docarray.array.elastic import DocumentArrayElastic

        instance = super().__new__(DocumentArrayElastic)
    elif storage == 'redis':
        from .redis import DocumentArrayRedis

        instance = super().__new__(DocumentArrayRedis)

    else:
        raise ValueError(f'storage=`{storage}` is not supported.')
```

在 Faiss 里有不同的索引类型可供选择：

IndexFlatL2、IndexFlatIP、IndexHNSWFlat、IndexIVFFlat、IndexLSH、IndexScalarQuantizer、IndexPQ、IndexIVFScalarQuantizer、IndexIVFPQ、IndexIVFPQR等。


## 2. 目前跨模态模型这里使用了比较大众的模型，文本与视频的匹配度有待提升，这里可以优化成更加优秀的模型；

TBC


## 3. 目前 VCED 项目仅能够处理对单个视频的检索，需要对项目改造来实现对多个视频片段的检索。

TBC