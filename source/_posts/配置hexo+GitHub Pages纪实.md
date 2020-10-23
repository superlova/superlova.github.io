---
title: 配置hexo+GitHub Pages纪实
date: 2019-04-14 09:32:13
tags: ["github pages", "hexo"]
categories:
- record
---

# 配置hexo+GitHub Pages纪实

想发博客不容易，折腾了大半天总算打通了一条路。

我一直有将自己匮乏的知识付诸于纸面的冲动。首先我在与FTK同学交流的过程中了解到同性交友网站GitHub可以搭建个人博客，于是今天试着探索了一下子。怎么说呢，没想到坑这么多？？？

本文旨在记录我从零开始建立个人博客的过程，还是有一定门槛的，如果对大家能有一点点帮助那就太好了。

# 获得个人网站域名方法

注册github即可，官网`www.github.com`，之后新建repository，名字为“你的用户名.github.io”。所以说用户名很重要，你要起个qq号做github名字那我估计没人能记住你的博客名，毕竟个人域名只能和用户名一样。记住项目一定要设置成公有public啊，博客是给大家看的嘛，孤芳自赏就不好玩了。

刚开始你的项目（repository）里面除了README.md都没有，但是不用怕。

# 安装nodejs、git、hexo方法

先下载Git，Git是什么？Git是代码托管工具。Git的特点是能够让多人一起开发一款软件。具体怎么实现的略去不表，总之Git的思想就是大家一起贡献代码，通过合理的组织方式保证代码提交的有序和质量的稳定，同时保存历史的代码记录防止现在的版本崩坏。但是Git的用途早已不限于程序员圈子了，起码在我看来Git用来写书、翻译、写文档等等都是极佳的，任何需要多人远程协作的项目都应该尝试使用Git。目前GitHub是全网最大的代码托管平台。自从GitHub背后的金主微软爸爸（一点也不微，一点也不软）财大气粗的宣布可以免费托管私人项目后，托管平台的隐私性得到了保证，我们更没有理由不试一试GitHub啦！

   OS: Windows 10 ~ 7均可

   [Hexo](https://hexo.io/zh-tw/): 3.3.1

   [Node.js](https://nodejs.org/en/): 6.10.2 LTS

   [Github Desktop](https://desktop.github.com/)

   [Git](https://git-scm.com/download/win)选择64位的安装包，下载后安装

   新增一个仓库(Repositories)，仓库名称为 ``yourname.github.io`` [yourname是你的账号]。
   开启刚安装好的 Github Desktop ，并将刚创好的仓库存到本地端。
   然后右键刚拉下来的仓库，选取 Open in Git Shell 打开 Git bash(option可选)，执行指令将 Github 上的仓库拉到本地端。

```
$ git pull origin master
```

Github Desktop 右上的 Sync 按钮具有 pull/push 功能，不想打指令可以多尝试。
使用 Github Desktop 的好处是不必像其他教学一样，不需要配置 SSH Key 和 设定 origin 位置路径，省了两个步骤。
安装好 Node.js 后，就能使用 npm 安装 hexo。

```
$ npm install -g hexo-cli
```
输入以下指令可查看版本。
```
$ hexo version
```
接下来，依序输入以下指令，初始化我们的 Blog。
```
# git bash 上面的路径大概长这样
You-PC@You  /e/Documents/GitHub/yourname.github.io (master)
$ hexo init		# 初始化 blog
# 输入上面那个指令后 hexo 会产生新的 .git盖掉旧的 .git
$ git init              # 所以就重新产生一个 .git
$ npm install		# 安装相关套件
$ hexo g		# 产生静态页面
$ hexo s		# 启动本地服务器
```
网址列输入 ``http://localhost:4000`` ，就能观看 Blog 了。
预设会有一个 Hello World 文章。
到我们本地 yourname.github.io 的文件夹中能找到 _config.yml 文件。
这是 Hexo 的全局配置文件。

```
# Deployment
deploy:
  type: git
  repository: git@github.com:yourname/yourname.github.io.git
  branch: master
```

这边是 YAML 语法，冒号后面记得空一格，照上面的设定输入，仓库的 SSH 地址如下图可获得。
执行以下命令安装 hexo-deployer-git，没安装套件前输入 hexo d 会出现 Error。
```
$ npm install hexo-deployer-git --save
```
产生静态页面后，部署到 Github。
```
$ hexo d -g     # 等同输入 hexo g 和 hexo d 指令
```
再来就可以上 ``https://yourname.github.io/`` 查看 Blog 了。


## 常用 Hexo 指令
写新文章
```
$ hexo new "postName" 		# 产生新的文章
$ hexo new page "pageName"	# 产生新的页面
```
Hexo提供了常用命令的简写
```
$ hexo n == hexo new   		# 产生新的 post/page/draft
$ hexo g == hexo generate  	# 产生静态页面
$ hexo s == hexo server		# 启动本地浏览
$ hexo d == hexo deploy		# 部署文件至 Github 上
```
指令组合
```
$ hexo d -g	# 产生静态文件后，部署 blog
$ hexo s -g	# 产生静态文件后，预览 blog
```
# 各种坑的处理方法

## Hexo 无法生成 index.html

在刚初始化一个项目后， 你运行 `hexo g`，有时候 hexo 并不会生成 `index.html` 和其他一些静态文件。 这一般是没有初始化完全的原因, 有些插件没有安装

1. 查看 npm 插件缺失情况

```
$ npm ls --depth 0
```

这时一般会提醒你有插件没有装。

```
npm ERROR! missing xxx
```

2. 安装缺失插件

如果你的插件都在 `packages.json` 里， 可以简单通过如下命令安装

```
$ npm install --save
```

要是没有， 就依次将所有缺失的插件安装上

```
$ npm install --save jquery jsdom [xxx ...]
```

3. 重新生成静态文件

安装好后，执行 `hexo g` 命令应该就可以正常生成完整博客了。

## hexo deploy不上去，明明本地能看但是联网后死活404


![github-page-404](github-page-404.png)

   1. 首先有可能是环境没安装完全。

   2. 其次，可能是config.yml中deploy不完全，应为https链接，此时删除根目录下.deploy_git文件夹后重新hexo deploy即可
![config-yml-deploy-not-ready](config-yml-deploy-not-ready.png)
![config-yml-deploy-ready](config-yml-deploy-ready.png)

   3. 再次，未使用git的bash，会提示权限不够

   4. 最后，没有CNAME，CNAME应位于source文件夹内，新建文本文件CNAME，内容为你的域名，username.github.io即可。CNAME不应该有后缀名。其实这一步应该是多余的，因为域名都会自动跳转。

   5. 另外，目前GitHub还没被墙，但不排除将来被隔离的可能。

   ## hexo next主题下出现德文

   根目录theme文件夹，你正在使用的主题内的config.yml内language词条加属性zh-CN，不知道哪个language就全加。

   ## 登录自己的域名，发现优先以文本方式显示README.md

   README.md应该位于source内，并且根目录config.yml中skip_render为README.md



![list-of-hexo-on-github](list-of-hexo-on-github.png)

# 需要的环境：

1. GitHub账号
2. 安装nodejs环境
3. 安装Git环境
4. windows10 or 7

# 2020年1月11日更新：

许久不写博客，今日发现重装系统后的新电脑无法正常发布和更新博客内容。事实上我的内容都上传到我的repository上了，我只需要调整好下载的分支就可以了。具体请看[参考文献](https://www.zhihu.com/question/21193762)。具体流程如下：

首先确认你的哪个分支是最全的。我的是hexo分支保存所有博文，master分支只适用于网页渲染。因此我迁移电脑需要下载hexo分支。

调整到hexo分支的本地目录，在git bash下执行
`npm install hexo`
`npm install` # 下载和更新需要的库
`npm install hexo-deployer-git`
不需要执行hexo init。

然后下载，输入
`git pull origin hexo`

如果提示报错，显示你本地已经有修改，则输入下列命令放弃修改：
`git reset --hard`
`git pull origin hexo`

现在你的本地hexo分支内应该下载了已经上传过的所有博客。

# 2020年10月23日更新：

npm下载慢的话，可以先
`npm install cnpm`
然后将所有npm指令替代为cnpm