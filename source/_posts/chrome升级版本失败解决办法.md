---
title: chrome升级版本失败解决办法
date: 2020-05-11 11:46:55
index_img: /img/chrome.png
tags: ['chrome']
categories: 
- record
---

## 错误描述：

在Win7电脑上试图将Chrome从32位的72版本升级到64位的80版本时发生问题，升级进度到62%报错：
Chrome安装 未知错误导致安装失败  "0x80040902"

从chrome官网下载“chromesetup.exe”，打开梯子之后下载成功，在安装过程中也出现未知错误。
从Chrome官网下载“Chromestandalonesetup64.exe”，即离线安装包，最后也出现同样的错误。
重新启动、进入安全模式、试图结束所有有关google的进程的方法对我都没用。

## 最后有效的方法：

把原来的Chrome从控制面板的“添加删除程序”中卸载；

按住windows+R，在“开始”运行中输入“regedit”，打开注册表编辑器，依次进入HKEY_CURRENT_USER\Software\Google\Chrome；

把Chrome这一项删除，然后重启。再安装就不会存在问题了。