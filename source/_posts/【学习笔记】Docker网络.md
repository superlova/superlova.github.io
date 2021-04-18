---
title: 【学习笔记】Docker网络
date: 2021-04-18 21:53:06
math: false
index_img: /img/docker.jpg
tags: ['Docker']
categories: 
- notes
---
Datawhale Docker学习笔记第四篇
<!--more--->

# Docker 基础网络介绍

## 外部访问容器

容器中可以运行一些网络应用，要让外部也可以访问这些应用，可以通过-P或-p参数来指定端口映射。

当使用-P标记时，Docker会随机映射一个端口到内部容器开放的网络端口。 使用docker container ls可以看到，本地主机的 32768 被映射到了容器的 80 端口。此时访问本机的 32768 端口即可访问容器内 NGINX 默认页面。

```
$ docker run -d -P nginx:alpine

$ docker container ls -l
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                   NAMES
fae320d08268        nginx:alpine        "/docker-entrypoint.…"   24 seconds ago      Up 20 seconds       0.0.0.0:32768->80/tcp   bold_mcnulty
```

同样的，可以通过docker logs命令来查看访问记录。

```
$ docker logs fa
172.17.0.1 - - [25/Aug/2020:08:34:04 +0000] "GET / HTTP/1.1" 200 612 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0" "-"
```

-p则可以指定要映射的端口，并且，在一个指定端口上只可以绑定一个容器。支持的格式有`ip:hostPort:containerPort | ip::containerPort | hostPort:containerPort`.

## 容器互联

下面先创建一个新的 Docker网络。

```
$ docker network create -d bridge my-net
```

-d参数指定Docker网络类型，有bridge overlay,其中overlay网络类型用于Swarm mode，在本小节中你可以忽略它。

运行一个容器并连接到新建的my-net网络

```
$ docker run -it --rm --name busybox1 --network my-net busybox sh
```

打开新的终端，再运行一个容器并加入到 my-net网络

```
$ docker run -it --rm --name busybox2 --network my-net busybox sh
```

再打开一个新的终端查看容器信息

```
$ docker container ls

CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
b47060aca56b        busybox             "sh"                11 minutes ago      Up 11 minutes                           busybox2
8720575823ec        busybox             "sh"                16 minutes ago      Up 16 minutes                           busybox1
```

下面通过 ping来证明busybox1容器和busybox2容器建立了互联关系。 在busybox1容器输入以下命令

```
/ # ping busybox2
PING busybox2 (172.19.0.3): 56 data bytes
64 bytes from 172.19.0.3: seq=0 ttl=64 time=0.072 ms
64 bytes from 172.19.0.3: seq=1 ttl=64 time=0.118 ms
```

用ping来测试连接busybox2容器，它会解析成 172.19.0.3。 同理在busybox2容器执行ping busybox1，也会成功连接到。

```
/ # ping busybox1
PING busybox1 (172.19.0.2): 56 data bytes
64 bytes from 172.19.0.2: seq=0 ttl=64 time=0.064 ms
64 bytes from 172.19.0.2: seq=1 ttl=64 time=0.143 ms
```

这样，busybox1 容器和 busybox2 容器建立了互联关系。

Docker Compose 如果你有多个容器之间需要互相连接，推荐使用DockerCompose。

## 配置DNS

如何自定义配置容器的主机名和 DNS 呢？秘诀就是Docker利用虚拟文件来挂载容器的 3个相关配置文件。

在容器中使用 mount命令可以看到挂载信息：

```
$ mount
/dev/disk/by-uuid/1fec...ebdf on /etc/hostname type ext4 ...
/dev/disk/by-uuid/1fec...ebdf on /etc/hosts type ext4 ...
tmpfs on /etc/resolv.conf type tmpfs ...
```

这种机制可以让宿主主机 DNS 信息发生更新后，所有Docker容器的 DNS 配置通过 /etc/resolv.conf文件立刻得到更新。

配置全部容器的 DNS ，也可以在 /etc/docker/daemon.json 文件中增加以下内容来设置。

```
{
  "dns" : [
    "114.114.114.114",
    "8.8.8.8"
  ]
}
```

这样每次启动的容器 DNS 自动配置为 114.114.114.114 和8.8.8.8。使用以下命令来证明其已经生效。
```
$ docker run -it --rm ubuntu:18.04  cat etc/resolv.conf

nameserver 114.114.114.114
nameserver 8.8.8.8
```

如果用户想要手动指定容器的配置，可以在使用docker run命令启动容器时加入如下参数： -h HOSTNAME或者--hostname=HOSTNAME设定容器的主机名，它会被写到容器内的/etc/hostname 和 /etc/hosts。但它在容器外部看不到，既不会在docker container ls中显示，也不会在其他的容器的/etc/hosts看到。

--dns=IP_ADDRESS添加 DNS 服务器到容器的/etc/resolv.conf中，让容器用这个服务器来解析所有不在 /etc/hosts 中的主机名。

--dns-search=DOMAIN设定容器的搜索域，当设定搜索域为.example.com时，在搜索一个名为host的主机时，DNS 不仅搜索 host，还会搜索host.example.com。

**注意：**如果在容器启动时没有指定最后两个参数，Docker会默认用主机上的/etc/resolv.conf来配置容器。

# Docker的网络模式

可以通过docker network ls查看网络，默认创建三种网络。

```
[root@localhost ~]# docker network ls
NETWORK ID          NAME                DRIVER              SCOPE
688d1970f72e        bridge              bridge              local
885da101da7d        host                host                local
f4f1b3cf1b7f        none                null                local
```

常见网络的含义：

网络模式|简介
---|---
Bridge | 为每一个容器分配、设置 IP 等，并将容器连接到一个 docker0 虚拟网桥，默认为该模式。
Host|容器将不会虚拟出自己的网卡，配置自己的 IP 等，而是使用宿主机的 IP 和端口。
None|容器有独立的 Network namespace，但并没有对其进行任何网络设置，如分配 veth pair 和网桥连接，IP 等。
Container|新创建的容器不会创建自己的网卡和配置自己的 IP，而是和一个指定的容器共享 IP、端口范围等。

## Bridge 模式

当Docker进程启动时，会在主机上创建一个名为docker0的虚拟网桥，此主机上启动的Docker容器会连接到这个虚拟网桥上，附加在其上的任何网卡之间都能自动转发数据包。虚拟网桥的工作方式和物理交换机类似，这样主机上的所有容器就通过交换机连在了一个二层网络中。从docker0子网中分配一个 IP 给容器使用，并设置 docker0 的 IP 地址为容器的默认网关。在主机上创建一对虚拟网卡veth pair设备，Docker 将 veth pair 设备的一端放在新创建的容器中，并命名为eth0（容器的网卡），另一端放在主机中，以vethxxx这样类似的名字命名，并将这个网络设备加入到 docker0 网桥中。可以通过brctl show命令查看。

## Host 模式

host 网络模式需要在创建容器时通过参数 --net host 或者 --network host 指定；
采用 host 网络模式的 Docker Container，可以直接使用宿主机的 IP 地址与外界进行通信，若宿主机的 eth0 是一个公有 IP，那么容器也拥有这个公有 IP。同时容器内服务的端口也可以使用宿主机的端口，无需额外进行 NAT 转换；
host 网络模式可以让容器共享宿主机网络栈，这样的好处是外部主机与容器直接通信，但是容器的网络缺少隔离性。

## None 模式

none 网络模式是指禁用网络功能，只有 lo 接口 local 的简写，代表 127.0.0.1，即 localhost 本地环回接口。在创建容器时通过参数 --net none 或者 --network none 指定；
none 网络模式即不为 Docker Container 创建任何的网络环境，容器内部就只能使用 loopback 网络设备，不会再有其他的网络资源。可以说 none 模式为 Docke Container 做了极少的网络设定，但是俗话说得好“少即是多”，在没有网络配置的情况下，作为 Docker 开发者，才能在这基础做其他无限多可能的网络定制开发。这也恰巧体现了 Docker 设计理念的开放。

## Container 模式

Container 网络模式是 Docker 中一种较为特别的网络的模式。在创建容器时通过参数 --net container:已运行的容器名称|ID 或者 --network container:已运行的容器名称|ID 指定；
处于这个模式下的 Docker 容器会共享一个网络栈，这样两个容器之间可以使用 localhost 高效快速通信。

Container 网络模式即新创建的容器不会创建自己的网卡，配置自己的 IP，而是和一个指定的容器共享 IP、端口范围等。同样两个容器除了网络方面相同之外，其他的如文件系统、进程列表等还是隔离的。
