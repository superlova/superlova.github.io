---
title: 【经验分享】TensorFlow模型训练和保存
date: 2020-06-03 22:35:56
index_img: /img/tensorflow.png
tags: ['TensorFlow', 'Python', 'SaveModel']
categories: 
- record
---
使用LSTM训练最简单的IMDB影评分类任务，总结文本分类任务常见流程。
<!--more--->
## 1. 模型训练和保存

### 1.1 训练结束时保存

训练模型，使用fit函数。fit函数的参数如下。
```python
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
```
x：训练数据
y：训练标签
batch_size：批次大小，默认为32
validation_data：在每个epoch结束之时计算loss等其他模型性能指标，不用做训练。
epoch：训练轮次
verbose：输出的详细程度，为1则输出进度条，表明每个epoch训练完成度；为0则什么也不输出，为2则很啰嗦地输出所有信息

最后保存模型用`model.save('xxx.h5')`，这里模型格式为HDF5，因此结尾为h5。

```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epoch=10, batch_size=64) 
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save('models/sentiment-lstm.h5')
```

### 1.2 在训练期间保存模型（以 checkpoints 形式保存）

您可以使用训练好的模型而无需从头开始重新训练，或在您打断的地方开始训练，以防止训练过程没有保存。` tf.keras.callbacks.ModelCheckpoint` 允许在训练的过程中和结束时回调保存的模型。

```python
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调函数
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 使用新的回调函数训练模型
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）是防止过时使用，可以忽略。
```

这将创建一个 TensorFlow checkpoint 文件集合，这些文件在每个 epoch 结束时更新

```
cp.ckpt.data-00001-of-00002
cp.ckpt.data-00000-of-00002  
cp.ckpt.index
```

默认的 tensorflow 格式仅保存最近的5个 checkpoint 。

### 1.3 手动保存权重

不必等待epoch结束，通过执行`save_weights`就可以生成ckpt文件。

```python
# 保存权重
model.save_weights('./checkpoints/my_checkpoint')
```

## 2. 模型加载

### 2.1 从h5文件中恢复
```python
# 重新创建完全相同的模型
model=load_model('models/sentiment-lstm.h5')
# 加载后重新编译模型，否则您将失去优化器的状态
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
model.summary()
```

加载模型的时候，损失函数等参数需要重新设置。

### 2.2 从ckpt文件中断点续训

仅恢复模型的权重时，必须具有与原始模型具有相同网络结构的模型。

```python
# 这个模型与ckpt保存的一样架构，只不过没经过fit训练
model = create_model()
# 加载权重
model.load_weights(checkpoint_path)
```

我们可以对回调函数增加一些新的设置，之前的回调函数每个epoch都覆盖掉之前的ckpt，现在我们想每过5个epoch保存一个新的断点：

```python
# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)
```

利用新的回调训练，并随后选择最新的断点文件：

```python
# 使用新的回调训练模型
model.fit(train_images, 
              train_labels,
              epochs=50, 
              callbacks=[cp_callback],
              validation_data=(test_images,test_labels),
              verbose=0)
# 选择新的断点
latest = tf.train.latest_checkpoint(checkpoint_dir)
>>> 'training_2/cp-0050.ckpt'

# 加载以前保存的权重
model.load_weights(latest)
```
