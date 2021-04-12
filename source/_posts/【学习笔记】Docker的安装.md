---
title: 【学习笔记】Docker的安装
date: 2021-04-13 00:34:56
math: false
index_img: /img/docker.jpg
tags: ['Docker']
categories: 
- notes
---
Datawhale Docker学习笔记第一篇
<!--more--->

为了今后的方便，我选择将Docker安装在实验室的电脑上，服务器的操作系统为 Ubuntu 18.04 LTS 。整个安装过程参考这篇文章(https://vuepress.mirror.docker-practice.com/install/ubuntu)。

我曾在一款老旧的笔记本电脑上尝试过安装 Docker ，但是最终失败了，原因是 Docker 不支持 32 位的操作系统。震惊！ Docker竟然不支持 32位操作系统！

由于我没有安装过老版本，因此不需要执行卸载旧版本的语句。直接执行

```sh
$ sudo apt-get update

$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

这里还出现了一些小插曲，当我执行完上面的 install 语句后，我与服务器建立的 ssh 连接断掉了，之后我试图重新连接居然提示密码错误。最后我重启虚拟机，修改 /etc/ssh/sshd_config 中的 PermitRootLogin 字段为 yes 解决了该问题。

添加软件源的 GPG 密钥，下载并安装，这一部分不再赘述。

```sh
$ curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

$ echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

$ sudo apt-get update

$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

之后启动 Docker 服务：

```sh
$ sudo systemctl enable docker
$ sudo systemctl start docker
```

激动人心的时刻到了，测试下 Docker 是否安装成功，执行一个 Hello World 看看：

```sh
$ sudo docker run --rm hello-world
```

![](【学习笔记】Docker的安装/2021-04-13-00-48-07.png)

大功告成！收工睡觉~