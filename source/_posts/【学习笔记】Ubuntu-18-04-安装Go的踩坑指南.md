---
title: 【学习笔记】Ubuntu 18.04 安装Go的踩坑指南
date: 2020-10-02 11:26:10
index_img: /img/go.png
tags: ['Golang', '安装']
categories: 
- notes
---
Go安装中遇到的坑
<!--more--->

gvm是第三方开发的Go多版本管理工具，利用gvm下载和安装go。

执行以下代码时，你应该确保自己安装有curl

```
bash < <(curl -s -S -L https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer)
```

在执行上述curl时，我的ubuntu 18.04报错

curl: (7) Failed to connect to [raw.githubusercontent.com](http://raw.githubusercontent.com/) port 443: Connection refused

经过[这个]([https://github.com/hawtim/blog/issues/10](https://github.com/hawtim/blog/issues/10))帖子的指引，我在hosts中添加了如下几行：

199.232.68.133 raw.githubusercontent.com

199.232.68.133 user-images.githubusercontent.com

199.232.68.133 avatars2.githubusercontent.com

199.232.68.133 avatars1.githubusercontent.com

curl便可以正常下载了。

安装完成gvm后我们就可以安装go了：

```
gvm install go1.15.2
gvm use go1.15.2
```

这个时候出现错误

zyt@ubuntu:~$ gvm install go1.15.2
Installing go1.15.2...

- Compiling...
/home/zyt/.gvm/scripts/install: line 84: go: command not found
ERROR: Failed to compile. Check the logs at /home/zyt/.gvm/logs/go-go1.15.2-compile.log
ERROR: Failed to use installed version

经查询，Go版本在1.5以上，需要在指令最后加上-B

[https://github.com/moovweb/gvm#a-note-on-compiling-go-15](https://github.com/moovweb/gvm#a-note-on-compiling-go-15)

zyt@ubuntu:~$ gvm install go1.15.2 -B
Installing go1.15.2 from binary source

```
gvm use go1.15.2
export GOROOT_BOOTSTRAP=$GOROOT
gvm install go1.5

zyt@ubuntu:~$ gvm list

gvm gos (installed)

=> go1.15.2

zyt@ubuntu:~$ gvm listall

gvm gos (available)
```