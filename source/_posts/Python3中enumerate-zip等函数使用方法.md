---
title: Python3中enumerate/zip等函数使用方法
date: 2019-05-18 15:24:47
tags: ['python']
categories:
 - notes
---
最近正在研读《Python机器学习基础教程》（Introduction to Machine Learning with Python）这本书。书中的Python3代码、对于numpy、pandas、matplotlib以及scikit-learn库的使用都让人叹为观止。作为Python初学者，这本书不仅可以让人入门机器学习，更可以让人的Python技巧得到提升。

下面的代码使用sklearn自带数据集moon以及sklearn的随机森林模型构建由5棵树组成的随机森林，并利用matplotlib库可视化。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn # 需要额外下载

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                                    random_state=42)
# 创建5棵树组成的随机森林
forest = RandomForestClassifier(n_estimators=5, random_state=2)
# 对训练集进行拟合
forest.fit(X_train, y_train)
# 生成两行三列的六张图，宽20高10
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
```

`forest.estimators_`是一个列表，保存五棵树的信息
```python
print(len(forest.estimators_))
print(type(forest.estimators_[0]))
5
<class 'sklearn.tree.tree.DecisionTreeClassifier'>
```

下面的for遍历用法是我之前很少接触的，尤其是对于enumerate与zip的使用，在此记录下来。

```python
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                               alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
```

首先zip函数就像大型相亲现场，接受两个可迭代集合：一波男性和一波女性，将前一个集合中的元素与后一个集合中的元素一一配对，返回两两结合的对象，即返回一大群元素组成的集合，集合中元素都是一男一女配对。配对方法就是粗暴的第i个男生-第i个女生，如果有男生or女生多了咋办？zip不管配不上对的元素，只挑选配对成功的组合。

经过zip函数处理，axes里面的六张图配上了五棵树，最后一张图我们留到最后处理。

接下来是enumerate，enumerate接受可迭代对象，不仅仅输出对象元素，还附带输出该元素所在的位置。enumerate本身就是“枚举”的意思嘛。