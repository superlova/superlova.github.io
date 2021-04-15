---
title: 【学习笔记】Docker镜像与容器
date: 2021-04-15 22:57:55
math: false
index_img: /img/docker.jpg
tags: ['Docker']
categories: 
- notes
---
Datawhale Docker学习笔记第二篇
<!--more--->

# Docker镜像

- 获取镜像


docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]


- 列出镜像

docker image ls

- 删除本地镜像

docker image rm [选项] <镜像1> [<镜像2> ...]

- Dockerfile构建镜像

如何可以生成 image 文件？如果你要推广自己的软件，势必要自己制作 image 文件。

这就需要用到 Dockerfile 文件。它是一个文本文件，用来配置 image。Docker 根据 该文件生成二进制的 image 文件。

在项目的根目录下，新建一个文本文件 Dockerfile

入下面的内容。


FROM node:8.4
COPY . /app
WORKDIR /app
RUN npm install --registry=https://registry.npm.taobao.org
EXPOSE 3000

上面代码一共五行，含义如下。

FROM node:8.4：该 image 文件继承官方的 node image，冒号表示标签，这里标签是8.4，即8.4版本的 node。
COPY . /app：将当前目录下的所有文件（除了.dockerignore排除的路径），都拷贝进入 image 文件的/app目录。
WORKDIR /app：指定接下来的工作路径为/app。
RUN npm install：在/app目录下，运行npm install命令安装依赖。注意，安装后所有的依赖，都将打包进入 image 文件。
EXPOSE 3000：将容器 3000 端口暴露出来， 允许外部连接这个端口。

有了 Dockerfile 文件以后，就可以使用docker image build命令创建 image 文件了

# Docker容器

容器是独立运行的一个或一组应用，以及它们的运行态环境。

- 新建并启动容器

使用 ubuntu 输出一个 “Hello World”，之后终止容器。

docker run ubuntu:18.04 /bin/echo 'Hello world'
Hello world

启动一个 bash 终端，允许用户进行交互

docker run -t -i ubuntu:18.04 /bin/bash
root@af8bae53bdd3:/#


其中，-t 选项让Docker分配一个伪终端（pseudo-tty）并绑定到容器的标准输入上， -i 则让容器的标准输入保持打开。

当利用 docker run 来创建容器时，Docker 在后台运行的标准操作包括：

检查本地是否存在指定的镜像，不存在就从registry下载
利用镜像创建并启动一个容器
分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
从地址池配置一个 ip 地址给容器
执行用户指定的应用程序
执行完毕后容器被终止

- 启动已终止的容器

可以利用 docker container start 命令，直接将一个已经终止（exited）的容器启动运行。

- 停止容器

docker stop可以停止运行的容器。理解：容器在docker host中实际上是一个进程，docker stop命令本质上是向该进程发送一个SIGTERM信号。如果想要快速停止容器，可使用docker kill命令，其作用是向容器进程发送SIGKILL信号。

docker ps 列出容器，默认列出只在运行的容器；加-a可以显示所有的容器

- 重启容器

对于已经处于停止状态的容器，可以通过docker start重新启动。docker start会保留容器的第一次启动时的所有参数。docker restart可以重启容器，其作用就是依次执行docker stop和docker start。

- 后台运行容器

添加 -d 参数来实现后台运行容器。在使用 -d 参数时，容器启动后会进入后台，启动完容器之后会停在host端；某些时候需要进入容器进行操作，包括使用 docker attach 命令或 docker exec 命令

docker exec 后边可以跟多个参数，这里主要说明 -i -t 参数。

只用 -i 参数时，由于没有分配伪终端，界面没有我们熟悉的 Linux 命令提示符，但命令执行结果仍然可以返回。

当 -i -t 参数一起使用时，则可以看到我们熟悉的 Linux 命令提示符。

attach和exec的区别
attach和exec的区别： （1）attach直接进入容器启动命令的终端，不会启动新的进程； （2）exec则是在容器中打开新的终端，并且可以启动新的进程； （3）如果想直接在终端中查看命令的输出，用attach，其他情况使用exec；

- 删除容器

可以使用 docker container rm 来删除一个处于终止状态的容器。

docker container rm trusting_newton
trusting_newton

如果要删除一个运行中的容器，可以添加 -f 参数。Docker 会发送 SIGKILL 信号给容器。

用 docker container ls -a 命令可以查看所有已经创建的包括终止状态的容器，如果数量太多要一个个删除可能会很麻烦，用下面的命令可以清理掉所有处于终止状态的容器。

docker container prune

批量删除所有已经退出的容器

docker rm -v $(docker ps -aq -f status=exited)

- 导出容器

如果要导出本地某个容器，可以使用 docker export 命令。

这样将导出容器快照到本地文件。

- 导入容器


可以使用 docker import 从容器快照文件中再导入为镜像

$ cat ubuntu.tar | docker import - test/ubuntu:v1.0
$ docker image ls
REPOSITORY          TAG                 IMAGE ID            CREATED              VIRTUAL SIZE
test/ubuntu         v1.0                9d37a6082e97        About a minute ago   171.3 MB


用户既可以使用 docker load 来导入镜像存储文件到本地镜像库，也可以使用 docker import 来导入一个容器快照到本地镜像库。这两者的区别在于容器快照文件将丢弃所有的历史记录和元数据信息（即仅保存容器当时的快照状态），而镜像存储文件将保存完整记录，体积也要大。此外，从容器快照文件导入时可以重新指定标签等元数据信息。