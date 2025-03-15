## VS

### 一些基础认识

#### 文件->新建->项目

设定名称以及位置

生成该项目的目标程序.exe：【生成】—【生成解决方案】

通常倒数第二行->后面的一串字符就是生成的目标程序路径，在本机上，可以直接将该路径复制到命令行运行


   * 当然，在VS这样强大的IDE里显然不需要这样复制粘贴来执行程序（但新手应该知道这种没有图形界面的程序本质上是如何运行的！），我们只需选择菜单栏中的【调试】→【开始执行(不调试)】（快捷键Ctrl+F5）


   * IDE将自动启动一个控制台并执行由该项目生成的程序


   * 注意，如果你尚未生成目标程序而直接选择执行，VS将自动替你完成生成的步骤，也就是说前面生成解决方案的步骤是可以省略的

解决方案资源管理器（视图）

- 
  #### debug


生成的可执行文件未经优化，大而慢，含开发人员添加的调试信息，不会投入市场应用

- 
  #### release


生成的可执行文件已经优化，小而快，可投入市场应用

2017的最新版本也在项目引导中去掉了“安全开发生命周期检查”的复选框，我的建议是要么干脆就学习下scanf_s这类函数的用法，要么实在想直接用scanf，就在所有代码文件开头加一句

#define _CRT_SECURE_NO_WARNINGS即可，或者到项目属性页里面将“SDL检查”改为“否"

VS默认将多项目解决方案中第一个添加的作为启动项目，每次执行会固定从它开始启动。这就给我们分别调试不同的项目带来了不便，解决的方法为

右键解决方案的图标，选择【属性】，在解决方案的属性页中，选择左侧【通用属性】→【启动项目】，再选择“当前选定内容”作为启动项目。确定回到代码页，在我们当前RudiCalc项目下，快捷键Ctrl+F5执行，此时其他无关项目被自动忽略


#### 用todo:写任务列表

### 直观的内部循环工作流

可以通过分支执行多项任务并试验代码

#### 提交Git

当你操作时，Git 会跟踪存储库中的文件更改，并将存储库中的文件分为三类:

- **未修改的文件**：自上次提交以来，这些文件未更改。
- **已修改的文件**：自上次提交以来，这些文件已更改，但尚未被暂存以用于下一次提交。
- **已暂存的文件**：这些文件已更改并将添加到下一次提交中。

#### 使用 git 提取、拉取、推送和同步进行版本控制

- 推送前始终拉取。 第一次拉取时，可以防止上游[合并冲突](https://learn.microsoft.com/zh-cn/visualstudio/version-control/git-resolve-conflicts?view=vs-2022)。
- “传出”文本表示尚未推送到远程库的提交数，而“传入”文本表示已提取但尚未从远程库拉取的提交数(outgoing/incoming)

#### 浏览和管理 Git 存储库

要详细了解如何使用 Visual Studio 中的“Git 存储库”窗口浏览和管理 Git 存储库，请参阅以下页面：

- [浏览存储库](https://learn.microsoft.com/zh-cn/visualstudio/version-control/git-browse-repository?view=vs-2022)
- [管理存储库](https://learn.microsoft.com/zh-cn/visualstudio/version-control/git-manage-repository?view=vs-2022)

#### 处理合并冲突

在[解决合并冲突](https://learn.microsoft.com/zh-cn/visualstudio/version-control/git-resolve-conflicts?view=vs-2022)页面了解详细信息

### github


#### git

我们先进到（我们定义的）Git仓库的最顶层文件目录下，然后从此目录进入Git Bash,这后操作才能顺利进行（这样会自动定位位置）


##### 常用命令

git config --global user.name "yourname"

git config --global user.email "your_email@youremail.com"

输入git status命令，查看仓库的状态


   * 在每个git操作之后，我们基本都会输入git status命令，查看仓库状态。

输入git init命令，初始化 Git 仓库

git remote add origin git@github.com:xxx.git

git remote add origin https:xxx.git

git pull origin master


   * git pull origin master

输入git add hit.txt（文件名）命令，将hit.txt文件添加到 Git 仓库


   * 在这里，需要声明一点，那就是：git add命令并没有把文件提交到 Git 仓库，而是把文件添加到了「临时缓冲区」，这个命令有效防止了我们错误提交的可能性。

输入git commit -m "text commit"命令，将hit.txt文件提交到 Git 仓库


   * 其中commit表示提交，-m表示提交信息，提交信息写在双引号""内


   * Git will not create a master branch until you commit something.

git push origin master 　　 将本地仓库的文件push到远程仓库(若 push 不成功，可加 -f 进行强推操作)

输入git log"命令，打印 Git 仓库提交日志

输入git branch命令，查看 Git 仓库的分支情况


   * 输入命令git branch a，再输入命令git branch，我们创建了一个名为a的分支，并且当前的位置仍然为主分支

输入git checkout a命令，切换到a分支


   * 我们也可以在创建分支的同时，直接切换到新分支，命令为git checkout -b，例如输入git checkout -b b命令


   * 我们在a分支下创建b分支（b为a的分支），并直接切换到b分支

切换到master分支，然后输入git merge a命令，将a分支合并到master分支

git branch -d & git branch -D


   * 输入git branch -d a命令，删除a分支


      * 不过有的时候，通过git branch -d命令可以出现删除不了现象，例如分支a的代码没有合并到主分支等，这时如果我们一定要删除该分支，那么我们可以通过命令git branch -D进行强制删除。

git tag


   * 输入git tag v1.0命令，为当前分支添加标签


      * 我们为当前所在的a分支添加了一个v1.0标签。通过命令git tag即可查看标签记录


      * 通过命令git checkout v1.0即可切换到该标签下的代码状态


#### ssh(安全外壳协议)

因为在 GitHub 上，一般都是通过 SSH 来授权的，而且大多数 Git 服务器也会选择使用 SSH 公钥来进行授权，所以想要向 GitHub 提交代码，首先就得在 GitHub 上添加 SSH key配置。


#### 提交代码

git push origin master

一般情况下，我们在push操作之前都会先进行pull操作，这样不容易造成冲突

git pull origin master

starting Oct. 2020, any new repository is created with the default branch main, not master

第一种：本地没有 Git 仓库，这时我们就可以直接将远程仓库clone到本地。通过clone命令创建的本地仓库，其本身就是一个 Git 仓库了，不用我们再进行init初始化操作啦，而且自动关联远程仓库。我们只需要在这个仓库进行修改或者添加等操作，然后commit即可

第二种：本地有 Git 仓库，并且我们已经进行了多次commit操作。(见知乎)


#### 使用

如果你想参与某个开源项目，你首先要做的是先了解这个项目，最好的方式是先仔细阅读它的 README

##### 编程初学者如何在 GitHub 寻找适合自己的小项目

HelloGitHub，致力于分享 GitHub 上有趣，入门级的开源项目，对于编程新手而言十分友好。地址：https://github.com/521xueweihan

这个新手项目还不够？那在推荐一个类似但更有趣的开源项目：GitHubDaily。链接：https://github.com/GitHubDaily/GitHubDailyGitHubDaily 每日不定时推送一批 GitHub 上优秀的开源项目给开发者, 帮助开发者们发现当下最火的开源项目, 令开发者们得以掌控技术脉搏, 扩大自己的技术视野, 并从开源项目的学习中获得技术能力的提升。

#### issue

提问+留言

#### Advanced search

高级搜索可视化

### 编译步骤

- 预处理：宏定义展开、头文件展开、条件编译等，删除注释.c——.i

- 编译：检查语法，将预处理后的文件生成汇编文件.i——.s

- 汇编：将汇编文件生成目标文件（二进制文件）.s——.o

- 链接：C语言写的程序使需要依赖各种库的，所以编译后还需要把库链接到最终的可执行程序中去.o——.exe


### 常见coding技巧

- 使用assert

- 尽量使用const

- 养成良好的编码风格

- 添加必要的注释

- 避免编码的陷阱

###  快捷键

- ctrl Q，搜索其他快捷键
- 