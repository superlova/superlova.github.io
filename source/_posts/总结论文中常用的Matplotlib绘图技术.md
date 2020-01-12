---
title: 总结论文中常用的Matplotlib绘图技术——长期更新
date: 2020-01-11 21:47:26
tags: ["matplotlib", "seaborn", "论文"]
categories:
 - 学习笔记
---

# 使用matplotlib绘制图像

matplotlib是一个Python的数据可视化2D图形库。matplotlib的特点是可以采用面向对象的方法，模仿MATLAB中的图形命令。matplotlib经常与numpy、pandas等库结合起来使用。
matplotlib可以采用MATLAB的命令风格使用，也可以采用面向对象的风格使用。

![](总结论文中常用的Matplotlib绘图技术/2020-01-11-23-58-29.png)

## matplotlib的图像中各组件名称

![](总结论文中常用的Matplotlib绘图技术/2020-01-11-23-42-39.png)

## 新建图像

```python
fig, axes = plt.subplots(2,1,figsize=(5,10)) #两行一列组成一张图，图像大小宽5高10
```
上面的语句创建了一个figure，由两个ax组成。把它想象成一张画布上面的两个贴画，会比较容易理解。

plt.figure()函数的前两个参数是设置figure是由几行几列的ax组成。figure(2,1)说明figure是由两行一列的ax一共两个ax组成。

后面的figsize参数设置画布的宽和高，单位为英寸。
