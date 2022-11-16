---
title: 【学习笔记】通过文字检索视频片段：VCED
date: 2022-11-14 22:12:07
index_img: /img/datawhale.jpg
tags: ['VCED', 'multimodal']
mathjax: false
math: false
categories: 
- notes
---

第一篇：环境部署

<!--more--->

VCED: Video Clip Extraction by description, 可以通过你的文字描述来自动识别视频中相符合的片段进行视频剪辑。基于跨模态搜索与向量检索技术搭建。

本项目参考自 [Datawhale 的 VCED 学习教程](https://github.com/datawhalechina/vced)。

环境为 Mac Monterey, Apple M1 Pro 芯片，内存 16GB。

首先需要安装 docker，在 mac 上安装 docker 只需去官网下载客户端：

https://docs.docker.com/desktop/install/mac-install/

![](【学习笔记】通过文字检索视频片段：VCED/mac安装docker.png)

注意，在 mac 上安装 docker 需要提前安装 rosetta。

安装 docker 完成后，为了方便下载镜像，我们先修改下源。

![](【学习笔记】通过文字检索视频片段：VCED/mac修改源.png)

在如图所示的文本框内按照 json 格式添加如下文本：

```json
"registry-mirrors": [
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
```

之后使用现有的镜像文件即可部署：

```sh
docker pull nil01/vced
docker run -itd -p 8501:8501 -p 45679:45679 --name vced_arm nil01/vced
```

最大的文件有 2GB，需要等待一会儿下载。部署完成后，访问 `localhost:8501` 即可。

![](【学习笔记】通过文字检索视频片段：VCED/浏览器交互页面.png)

启动 docker 进程之后，输入指令 `docker ps` 查看运行中的 container，输入 `docker ps <CONTAINER ID>` 即可结束该 container。

![](【学习笔记】通过文字检索视频片段：VCED/docker操作.png)


# 参考

https://github.com/datawhalechina/vced/blob/main/README.md

https://docs.jina.ai/get-started/install/windows/