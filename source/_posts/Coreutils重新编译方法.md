---
title: Coreutils重新编译方法
date: 2019-04-14 10:18:30
tags: ['compiler']
categories:
- record
---

# Coreutils重新编译方法

1. 下载coreutils，在Linux系统下解压

![下载coreutils解压](Coreutils重新编译方法\下载coreutils解压.png)

2. 运行指令 ` ./configure `

![运行指令](Coreutils重新编译方法\运行指令.png)

3. 运行 `make`

4. 进入src文件夹，挑选您要修改的文件，我以pwd.c为例，将其复制到我的个人文件夹

5. 修改pwd.c，将其内部所有带“VERSION”的行全部注释掉

6. 运行指令1

   ```
   $ gcc -E -I ~/MyCode/coreutils-8.30/lib/ -I ~/MyCode/coreutils-8.30/ -I ~/MyCode/coreutils-8.30/src pwd.c -o pwd.i 
   ```

7. 运行指令2

   ```
   $ gcc -c pwd.i -o pwd.o
   ```

8. 运行指令3

   ```
   $ gcc -L ~/MyCode/coreutils-8.30/lib/ -L /usr/lib/ pwd.o -o pwd -lcoreutils -lcrypt
   ```

9. 执行`./pwd`
   其他文件同理