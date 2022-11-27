---
title: 【学习笔记】机器学习之XGBoost、LightGBM、CatBoost
date: 2022-11-27 08:09:26
index_img: /img/algorithm.png
tags: ['Machine Learning', 'XGBoost', 'LightGBM', 'CatBoost']
categories:
  - notes
mathjax: true
math: true
---

本文会记录 XGBoost、LightGBM、CatBoost 的原理、区别和使用方法。

<!--more--->

## XGBoost

XGBoost 是 GBDT 算法的一种高效实现方式。

### XGBoost 的原理

原始的 GBDT 算法基于经验损失函数的负梯度来构造新的决策树，构造完成后再进行后剪枝。XGBoost 添加了预剪枝操作，它在目标函数中添加了正则项。

$$
L_t=\sum_i L(y_i, F_{t-1}(x_i)+f_t(x_i)) + \Omega(f_t)
$$

其中：

$$
\Omega(f_t)=\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w_{j}^2
$$

其中 T 为这棵树的叶子结点个数，$w_j$表示第 j 个叶子结点的预测值。这是一个正则项，该值能够在一定程度上代表一棵树的复杂程度。因此在建树时，我们可以调整超参数，达到预剪枝的效果， $\gamma$ 越大则不容易分叉，较小则容易分叉。XGBoost 添加此项的目的在于控制不要生成过于复杂的子树结构。

要想优化该目标函数，但是 f 本身不是数值型的函数，不能直接使用梯度下降进行优化。因此在这里对损失函数进行二阶泰勒展开，用作函数的近似：

$$
\begin{aligned}
L_t &=\sum_i L(y_i, F_{t-1}(x_i)+f_t(x_i)) + \Omega(f_t) \\
&= \sum_i [L(y_i, F_{t-1}(x_i)) + g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)] + \Omega(f_t)
\end{aligned}
$$

其中 $ g_i=\frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)} $，$ h_i=\frac{\partial^2 L(y_i, F_{t-1}(x_i))}{\partial^2 F_{t-1}(x_i)} $，分别为f 的一阶导数和二阶导数

对第 t 次迭代来说，$ L(y_i, F_{t-1}(x_i)) $ 是个常数，因此目标函数可进一步简化为：

$$
L_t=\sum_i [g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)] + \Omega(f_t)
$$

这样做，只需知道每个样本的损失函数的一阶导数、二阶导数，每个样本在第 t 个分类器的预测结果，叶子结点的个数，叶子结点的输出值，就知道第 t 步的损失函数值了。

上面的公式，有两个 $\sum$ ，在数学上不好处理：左边的 $\sum$ 是逐样本累加，右边的 $\sum$ 是逐叶子累加，需要把他们合并起来。

如果我们把某个叶子结点 j 上面所有样本的损失函数的一阶导数之和记作 G，二阶导数之和记作 H，则上式可简化。具体地，如果叶子结点 j 上面的样本集合为 $I_j=\{q(x_i)=j\}$ （q(x) 是一个函数，能输入一个样本，输出该样本在树上的节点的下标），则 G、H 的定义如下：

$$
G_j=\sum_{i\in I_j}g_i \\
H_j=\sum_{i\in I_j}h_i
$$

然后我们使用如下变换：

$$
\begin{aligned}
L_t&=\sum_{i=1}^{N} [g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)] + \gamma T + \frac{1}{2}\lambda\sum^{T}_{j=1}w^2_j \\
&=\sum_{i=1}^N [g_iw_{q(x_i)}+\frac{1}{2}h_iw^2_{q(x_i)}] + \gamma T + \frac{1}{2}\lambda\sum^{T}_{j=1}w^2_j \\
&=\sum_{j=1}^T [\sum_{i\in I_j}g_iw_j+\frac{1}{2}\sum_{i\in I_j}h_iw_j^2] + \gamma T + \frac{1}{2}\lambda\sum^{T}_{j=1}w^2_j \\
&=\sum_{j=1}^T(G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2)+\gamma T
\end{aligned}
$$

这就是目标函数的最终结果。

接下来我们希望能够优化该目标函数，计算第 t 轮时使目标函数最小的叶节点的输出分数 w。我们直接对 w 进行求导并令导数为 0，此时的 w 为：

$$
w_j=-\frac{G_j}{H_j+\lambda}
$$

最终得到损失函数形式：

$$
L_t=-\frac{1}{2}\sum_{j=1}^T (\frac{G^2_j}{H_j + \lambda}) + \gamma T
$$

该值越小，目标函数越小。

XGBoost 在分裂时，会遍历该特征的所有取值，尝试计算分裂前的损失和分裂后的损失。

$$
\text{Gain}=\frac{1}{2}[\frac{G^2_L}{H_L + \lambda}+\frac{G^2_R}{H_R + \lambda}-\frac{(G_L + G_R)^2}{H_L + H_R + \lambda}] - \gamma
$$

其中第一项是左子树分数，第二项是右子树分数，第三项是分裂前的分数，最后的 gamma 是新叶子结点的复杂度。

gain 值越大，说明分裂后能够更多地让损失函数减小，也就是越好。

### XGBoost 自定义损失函数

XGBoost 能够解决二元分类、多元分类、回归问题。对于二元分类问题，最终的预测值会经过 sigmoid 函数后比较输出与 0 或 1 的接近程度；对于多元分类问题，最终的预测值会经过 softmax 函数后得到某一类型的最大概率，输出某一类型的值。

二元分类器的目标函数：

$$
sigmoid=\frac{1}{1+e^{-x}}
$$

```py
def sigmoid(x):
    return 1.0 / 1 + np.exp(-x)
```

损失函数：

$$
logloss=-(y\log(p)+(1-y)\log(1-p))
$$

将损失函数代入 sigmoid：

$$
logloss(y_i,\hat{y_i})=(y\ln(1+e^{-\hat{y_i}})+(1-y_i)\ln(1+e^{\hat{y_i}}))
$$

损失函数的一阶导数和二阶导数：

$$
g_i=p_i^{t-1}-y_i \\
h_i=p^{t-1}_i (1-p^{t-1}_i)
$$

多元分类器的目标函数：

$$
\text{softmax}=\frac{e^{\hat{y_i}}}{\sum_j e^{\hat{y_i}}}
$$

```py
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
```

损失函数：

$$
logloss=-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^M y_{ij}\log(p_{i,j})
$$

其中 N 为样本数，M 为类别数。 $y_{ij}$ 代表第 i 个样本分类为 j 时则为 1，否则为 0； $p_{i,j}$ 代表第 i 个样本被预测为第 j 类的概率。

损失函数的一阶导数和二阶导数，其整体形式仍与二分类一致.

如果我们想自定义损失函数，比如在二分类问题上，我们对数据较少的那个类别予以更多惩罚，就可以这样实现：

```py
def weighted_binary_cross_entropy(pred, dtrain, imbalance_alpha=10):
    label = dtrain.get_label()
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # 一阶导数
    grad = -(imbalance_alpha ** label) * (label - sigmoid_pred)
    # 二阶导数
    hessian = (imbalance_alpha ** label) * sigmoid_pred* （1.0 - sigmoid_pred）

    return grad, hessian
```

### XGBoost 候选分裂点选取策略

### XGBoost 在工程上的优化

XGBoost 在选择分裂切分点时，不会一个一个地遍历，而是选择分位数点。

XGBoost 在生成下一棵树时，所有样本的权重 g 和 h 可以通过当前树的预测值和实际值得到。因此要确定最佳分割点，需要提前对特征值进行排序。xgboost可以在训练下一棵树前预先计算好每个特征在不同切分点的增益，并对数据进行排序，保存为block的结构，迭代中重复使用这个结构，因此各个特征的增益计算就可以多线程并行进行。

### XGBoost 与 GBDT 的区别

1. 基分类器不同。XGBoost 目前支持三种基分类器，分别是 gbtree，gblinear 和 dart，可以使用 **booster** 参数修改。一般情况下 gbtree 就是基于 CART 回归树的 XGBoost，gblinear 是逻辑回归线性分类器，dart 是用了 dropout 的树分类器。

2. 导数信息不同，损失函数也不同。XGBoost 对损失函数做了二阶泰勒展开，GBDT 只用了一阶导数信息。

XGBoost 没有定义损失函数本身，只要该损失函数能够进行二阶泰勒展开即可，因此基学习器可以更换。

XGBoost 默认使用 CART 回归树用作基学习器，但是可以更换为其他学习器，比如线性模型。此时的损失函数需要更换。

比如分类问题，损失函数一般为 logistics 损失函数

3. XGBoost 在每轮迭代时，可以使用全部数据，也可以使用自助采样的数据。

这里的思想类似随机森林，每次节点分裂并没有比较所有特征的好坏，而是先随机圈定一部分特征，只使用这部分特征进行比较，从而在生成树时引入了随机性。

列采样有两种方式，一种是层内随机，一层的所有节点都使用随机圈定的特征；另一种是整颗树随机，即建树伊始就使用随机圈定的特征。

另外 XGBoost 在决定特征的分裂点是，并没有逐个比较所有情况，而是选用了二阶梯度分位点。

4. XGBoost 能够自动学习出缺失值的处理策略。

缺失值处理策略相对简单，即分别尝试把所有缺失该特征的数据放在左子树和右子树，分别计算增益，来决定缺失数据的归宿。这也就决定了 XGBoost 无法应对 OneHot 编码这种大部分都为 0 的特征。

5. XGBoost 在工程实现上的优化

在选择分裂点时，XGBoost 会首先将样本按照特征进行排序。排序的这个步骤可以进行优化。

## LightGBM

## CatBoost
## LambdaMART

## 参考

https://www.jianshu.com/p/bac6a0dfac2c

https://blog.csdn.net/zwqjoy/article/details/109311133