---
title: 【学习笔记】使用tf.data对预处理过程优化
date: 2020-08-15 23:18:43
math: false
index_img: /img/tensorflow.png
tags: ['Tensorflow', 'preprocessing']
categories: 
- notes
---
本文是关于 `tf.data` 介绍的第二篇，主要介绍一些数据预处理方面的优化方法，诸如并行化预处理映射函数、使用缓存等。
<!--more--->

构建一个机器学习模型时，由于数据预处理过程不能使用GPU进行加速，因此格外耗时。背后的原因可能是CPU、网络或者缓存等复杂的因素。因此要研究如何提升数据预处理的效率，首先需要控制实验的变量。想实现这一点，构造一个虚假的数据集比较可行。

通过构建一个虚假的数据集，从`tf.data.Dataset`继承的类，称为`ArtificialDataset`。该数据集模拟三件事：

1. 生成`num_samples`样本（默认为3）
2. 在第一个模拟打开文件的项目之前睡眠一段时间
3. 在产生每个项目以模拟从文件读取数据之前先休眠一段时间

```py
class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)
        
        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)
            
            yield (sample_idx,)
    
    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )
```

构建 `benchmark` ，通过模拟训练的方式，计算该数据预处理模式的耗时：

```py
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)
```

我们先来不加任何优化地运行一次benchmark：
```py
benchmark(ArtificialDataset())

Execution time: 0.33306735700000445
```

此时模型的执行时间图如图所示：
![Naive](https://www.tensorflow.org/guide/images/data_performance/naive.svg)

时间消耗是这样的：先是打开文件，然后从文件中获取数据项，然后使用数据进行训练。这种执行方式，当数据进行预处理，模型就空闲；当模型开始训练，管道又空闲下来了。预处理和训练这两部分明显可以重叠。

`tf.data` API提供了`tf.data.Dataset.prefetch`转换。它可以用于将数据生成时间与数据消耗时间分开。转换使用后台线程和内部缓冲区预取元素。要预取的元素数量应等于（或可能大于）单个训练步骤消耗的批次数量。将预取的元素数量设置为`tf.data.experimental.AUTOTUNE` ，这将提示`tf.data`运行时在运行时动态调整值。
```py
benchmark(
    ArtificialDataset()
    .prefetch(tf.data.experimental.AUTOTUNE)
)
Execution time: 0.20504431599999862
```
![Prefetched](https://www.tensorflow.org/guide/images/data_performance/prefetched.svg)

时间有了明显优化，因为数据的生产和消费有了些许重叠。

在实际工作中，输入数据可以远程存储在其他计算机上。在本地和远程存储之间存在以下差异：

1. 到达第一个字节的时间：从远程存储读取文件的第一个字节所花费的时间要比从本地存储中读取文件的时间长几个数量级。
2. 读取吞吐量：虽然远程存储通常提供较大的聚合带宽，但是读取单个文件可能只能使用此带宽的一小部分。

此外，一旦将原始字节加载到内存中，可能还需要对数据进行反序列化和/或解密，这需要进行额外的计算。不管数据是本地存储还是远程存储，都存在这种开销，但是**如果数据没有有效地预取，则在远程情况下会更糟**。

可以使用`tf.data.Dataset.interleave`转换来**并行化数据加载步骤**， `cycle_length` 表明可以一起处理的数据集数量， `num_parallel_calls` 则是并行度。

```py
benchmark(
    tf.data.Dataset.range(2)
    .interleave(
        ArtificialDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
Execution time: 0.18243273299958673
```
![Parallel interleave](https://www.tensorflow.org/guide/images/data_performance/parallel_interleave.svg)

该图可以显示`interleave`变换的行为，从两个可用的数据集中获取样本。这次，两个数据集的读取并行进行，从而减少了全局数据处理时间

## 并行预处理操作

在准备数据时，可能需要对输入元素进行预处理。可以使用`tf.data.Dataset.map(f)`转换，其含义为将某个转换`f`作用于数据集`Dataset`中的每个元素。这里有个很重要的前提条件，由于输入元素彼此独立，因此预处理可以跨多个CPU内核并行化。因此`map`转换也提供`num_parallel_calls`参数来指定并行度。关于并行度的选择上，`map`转换支持`tf.data.experimental.AUTOTUNE`，而不必人工定义。

首先定义伪操作：

```py
def mapped_function(s):
    # Do some hard pre-processing
    tf.py_function(lambda: time.sleep(0.03), [], ())
    return s
```

我们来测试伪操作，此时没有任何并行优化：

```py
benchmark(
    ArtificialDataset()
    .map(mapped_function)
)
Execution time: 0.4592052289999913
```
![Sequential mapping](https://www.tensorflow.org/guide/images/data_performance/sequential_map.svg)

现在，使用相同的预处理功能，但将其并行应用于多个样本。

```py
benchmark(
    ArtificialDataset()
    .map(
        mapped_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
)
Execution time: 0.3045882669994171
```
![Parallel mapping](https://www.tensorflow.org/guide/images/data_performance/parallel_map.svg)

可以在图上看到预处理步骤重叠，从而减少了单次迭代的总时间。

`tf.data.Dataset.cache`转换可以在内存中或本地存储上缓存数据集。这样可以避免在每个epoch执行某些重复性操作（例如打开文件和读取数据）。
```py
benchmark(
    ArtificialDataset()
    .map(  # Apply time consuming operations before cache
        mapped_function
    ).cache(
    ),
    5
)
Execution time: 0.3795637040002475
```

![Cached dataset](https://www.tensorflow.org/guide/images/data_performance/cached_dataset.svg)

第一个epoch执行一次cache之前的转换（例如文件打开和数据读取）。下一个epoch将重用cache转换所缓存的数据。

这里涉及到一个`map`和`cache`操作谁先谁后的问题。有一个原则，如果`map`操作很复杂、昂贵，那么先`map`再`cache`，下次不用`map`了。如果`cache`过大而无法放入缓冲区，则先`cache`后`map`，或者试图采用一些数据预处理方法以减少资源使用。

## 向量化数据预处理操作

所谓向量化，即使得`mapping`操作能够一次处理一`batch`数据。这样做肯定可以加速，因为避免了繁杂的数据读取时间。对用户定义的函数进行向量化处理，并且对数据集应用`batch`转换再进入`mapping`。在某种情况下，这个做法非常有用。

首先定义一个数据集操作`increment`，负责把每个元素的值+1。另外之前的例子里面使用了毫秒级别的`sleep`操作，这会掩盖我们优化的结果。这次我们把它拿掉。

下面是未经向量化优化的`increment`操作耗时：

```py
fast_dataset = tf.data.Dataset.range(10000)

def fast_benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for _ in tf.data.Dataset.range(num_epochs):
        for _ in dataset:
            pass
    tf.print("Execution time:", time.perf_counter() - start_time)
    
def increment(x):
    return x+1

fast_benchmark(
    fast_dataset
    # Apply function one item at a time
    .map(increment)
    # Batch
    .batch(256)
)
Execution time: 0.7625284370005829
```

![Scalar map](https://www.tensorflow.org/guide/images/data_performance/scalar_map.svg)

与之对比，经过向量化后，耗时明显减少：
```py
fast_benchmark(
    fast_dataset
    .batch(256)
    # Apply function on a batch of items
    # The tf.Tensor.__add__ method already handle batches
    .map(increment)
)
Execution time: 0.04735958700075571
```
![Vectorized map](https://www.tensorflow.org/guide/images/data_performance/vectorized_map.svg)

## 减少内存占用

许多转换（包括interleave ， prefetch和shuffle ）各自维护内部缓冲区。如果传递给map转换的用户定义函数更改了元素的大小，则映射转换的顺序以及缓冲元素的转换会影响内存使用。

通常，我们建议选择导致内存占用减少的顺序，除非需要不同的顺序才能提高性能。

对于缓存，我们建议除非转换后的数据难以保存到缓冲区，否则一律先`map`再`cache`。如果你有两个`map`，其中一个比较耗时`time_consuming_mapping`，另一个比较耗内存`memory_consuming_mapping`，那么其实你可以将其拆分成两部分；
```py
dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)
```
这样，耗时部分仅在第一个epoch执行，并且避免了使用过多的缓存空间。

## 总结

使用`tf.data`，并采用合理的优化手段，就能让你的数据预处理过程节约很多时间。这些手段有：

* 使用`prefetch`转换可以使生产者和消费者的工作重叠。
* 使用`interleave`变换并行化数据读取变换。
* 通过设置`num_parallel_calls`参数来并行化`map`转换 。
* 在第一个epoch使用`cache`转换将数据缓存在内存中
* 向量化传递给`map`转换的用户定义函数
* 应用`interleave` ， `prefetch`和`shuffle`转换时， 逐渐减少内存使用 。