---
title: 【学习笔记】Docker数据管理、数据卷和挂载主机目录
date: 2021-04-16 21:25:46
math: false
index_img: /img/docker.jpg
tags: ['Docker']
categories: 
- notes
---
Datawhale Docker学习笔记第三篇
<!--more--->

# 数据卷
- 创建数据卷
```
docker volume create datawhale
```
查看所有的数据卷
```
docker volume ls
```
- 启动一个挂载数据卷的容器

在用 docker run 命令的时候，使用 --mount 标记来将数据卷挂载到容器里。在一次 docker run 中可以挂载多个 数据卷。

- 查看数据卷的具体信息

在主机里使用以下命令可以查看 web 容器的信息
```
docker inspect web
```
- 删除数据卷
```
docker volume rm datawhale  #datawhale为卷名
```
无主的数据卷可能会占据很多空间，要清理请使用以下命令
```
docker volume prune
```
# 挂载主机目录

- 挂载一个主机目录作为数据卷
```
docker run -d -P \
    --name web \
    --mount type=bind,source=/src/webapp,target=/usr/share/nginx/html \
    nginx:alpine
```
使用 --mount 标记可以指定挂载一个本地主机的目录到容器中去。

- 查看数据卷的具体信息

在主机里使用以下命令可以查看 web 容器的信息
```
docker inspect web
```
- 挂载一个本地主机文件作为数据卷

--mount 标记也可以从主机挂载单个文件到容器中
```
docker run --rm -it \
   --mount type=bind,source=$HOME/.bash_history,target=/root/.bash_history \
   ubuntu:18.04 \
   bash
```