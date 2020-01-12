---
title: 如何使用VS Code编写github pages博客
date: 2020-01-11 21:55:17
tags: ["GitHub Pages", "hexo", "Visual Studio Code", "markdown"]
categories: 
  - 经验教训
---

使用VS Code写博客，需要你按照我之前写的两篇博客，将github pages平台搭建起来。

[配置hexo+GitHub Pages纪实](https://superlova.github.io/2019/04/14/%E9%85%8D%E7%BD%AEhexo+GitHub%20Pages%E7%BA%AA%E5%AE%9E/)
[hexo图片加载失败解决方案](https://superlova.github.io/2019/04/25/hexo%E5%9B%BE%E7%89%87%E5%8A%A0%E8%BD%BD%E5%A4%B1%E8%B4%A5%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/)

之后我们安装VSCode。接下来介绍我一直使用的几个插件，和它们的配置小技巧。

第一个是**Markdown Preview Enhanced**，有了该插件，就可以提前预览markdown文件的渲染效果。方法是使用VSCode打开以md后缀名结尾的文件，右键点击**Markdown Preview Enhanced： Open Preview To The Side**，即可在侧边栏生成即时渲染的md效果文件。

第二个是**Markdown PDF**，该插件可以令写好的md文件打印成pdf格式。该插件需要安装chromium内核。

第三个是****Paste Image****插件，可以很方便地在md文章中粘贴位于剪切板的图片。

粘贴的快捷键是Ctrl+Alt+V。

在Paste Image插件的Path设置部分，改成如下所示：
![](如何使用VS-Code编写github-pages博客/2020-01-11-23-28-36.png)
这样图片粘贴的位置就变成了**当前文章目录下，与该文章同名的文件夹内**，方便我们进行进一步整理。