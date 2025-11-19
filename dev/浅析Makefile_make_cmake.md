# Main Takeaway

编译链

<!--more-->

# 简介

cmake生成makefile文件给make执行

- makefile中设置你想要的编译规则，你想要编译哪些文件，哪些文件不需要编译等等都可以体现在Makefile中，而且支持多线程并发操作，可以减少编译的时间。

- make就是根据Makefile中写的内容进行编译和链接

  > Tips:make只会编译我们修改过的文件，没有修改过的就不用重新编译，这样我们debug了一个小bug后重新编译就不用花费大量的编译时间

- cmake跨平台项目管理的工具生成Makefile文件给make去执行，实现跨平台。cmake它仍然是目标、依赖之类的抽象的东西，根据一个叫CMakeLists.txt的文件（还是自己手写）生成Makefile的

# Makefile

### 关于程序的编译和链接

源文件首先会编译(compile)成中间目标文件（在Windows下也就是 .obj 文件，UNIX下是 .o 文件，即object file ），然后再把大量的Object File合成执行文件，这个动作叫作链接（link）。在编译时，编译器只检测程序语法，和函数、变量是否被声明。如果函数未被声明， 编译器会给出一个警告，但可以生成Object File。而在链接程序时，链接器会在所有的Object File中找寻函数的实现，如果找不到，那到就会报链接错误码（Linker Error）

.lib（windows下Library File） .a（linux下Archive File）库文件（打包形成，因为文件太多）

### 作用

makefile关系到了整个工程的编译规则

makefile定义了一系列的规则来指定，哪些文件需要先编译，哪些文件需要后编译，哪些文件需要重新编译，甚至于进行更复杂的功能操作，因为makefile就像一个Shell 脚本一样，其中也可以执行操作系统的命令

### 好处

makefile带来的好处就是——“自动化编译”，一旦写好，只需要一个make命令，整个工程完全自动编译，极大的提高了软件开发的效率

大部分IDE中都有

更省事、准确。对于大的工程（文件多，一个.c对应了一个.o文件，最后还涉及.o文件和.a文件链接）编译汇编工作如果手动操作就会变得繁琐且重复，那么makefile出现能极大的简化这个重复繁琐的过程，而且对于不同的操作系统，也需要让编译自动适应

### Makefile介绍

make命令执行时，需要一个 Makefile 文件，以告诉make命令需要怎么样的去编译和链接程序。

说白一点就是说，prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行。这就是 Makefile的规则。也就是Makefile中最核心的内容

#### Makefile规则

##### 举例

foo.o : foo.c defs.h # foo模块
cc -c -g foo.c

1、文件的依赖关系，foo.o依赖于foo.c和defs.h的文件，如果foo.c和defs.h的文件日期要比foo.o文件日期要新，或是foo.o不存在，那么依赖关系发生。
2、如果生成（或更新）foo.o文件。也就是那个cc命令，其说明了，如何生成foo.o这个文件。（当然foo.c文件include了defs.h文件）

##### 语法

target ... : prerequisites（*n.*先决条件，前提*adj.*先决的，必备的） ...
command
...
...

或是这样：

targets : prerequisites ; command
command
...

- target也就是一个目标文件，可以是Object File，也可以是执行文件。还可以是一个标签（Label）
- prerequisites就是，要生成那个target所需要的文件或是目标
- command也就是make需要执行的命令。（任意的Shell命令）如果不与target在同一行则必须以[Tab键]开头

这是一个文件的依赖关系，也就是说，target这一个或多个的目标文件依赖于prerequisites中的文件，其生成规则定义在command中。 说白一点就是说，prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行。这就是 Makefile的规则。也就是Makefile中最核心的内容。

##### 通配符

如果我们想定义一系列比较类似的文件，我们很自然地就想起使用通配符。make支持三各通配符：“*”，“?”和“[...]”。这是和Unix的B-Shell是相同的。

波浪号（“~”）字符在文件名中也有比较特殊的用途。如果是“ ~/test”，这就表示当前用户的$HOME目录下的test目录。而 “~hchen/test”则表示用户hchen的宿主目录下的test目录。（这些都是Unix下的小知识了，make也支持）而在Windows或是 MS-DOS下，用户没有宿主目录，那么波浪号所指的目录则根据环境变量“HOME”而定。

通配符代替了你一系列的文件，如“*.c”表示所以后缀为c的文件。

$符号表示取变量的值，当变量名多于一个字符时，使用"( )"

$符的其他用法

$^ 表示所有的依赖文件

$@ 表示生成的目标文件

$< 代表第一个依赖文件

通配符代替了你一系列的文件，如“*.c”表示所以后缀为c的文件

##### 伪目标

因为，我们并不生成“clean”这个文件。“伪目标”并不是一个文件，只是一个标签，由于“伪目标”不是文件，所以make无法生成它的依赖关系和决定 它是否要执行。我们只有通过显示地指明这个“目标”才能让其生效。当然，“伪目标”的取名不能和文件名重名，不然其就失去了“伪目标”的意义了。

当然，为了避免和文件重名的这种情况，我们可以使用一个特殊的标记“.PHONY”来显示地指明一个目标是“伪目标”，向make说明，不管是否有这个文件，这个目标就是“伪目标”。

.PHONY : clean

只要有这个声明，不管是否有“clean”文件，要运行“clean”这个目标，只有“make clean”这样。于是整个过程可以这样写：

.PHONY: clean
clean:
rm *.o temp

伪目标一般没有依赖的文件。但是，我们也可以为伪目标指定所依赖的文件。伪目标同样可以作为“默认目标”，只要将其放在第一个。一个示例就是，如果你的 Makefile需要一口气生成若干个可执行文件，但你只想简单地敲一个make完事，并且，所有的目标文件都写在一个Makefile中，那么你可以使 用“伪目标”这个特性：

all : prog1 prog2 prog3
.PHONY : all

prog1 : prog1.o utils.o
cc -o prog1 prog1.o utils.o

prog2 : prog2.o
cc -o prog2 prog2.o

prog3 : prog3.o sort.o utils.o
cc -o prog3 prog3.o sort.o utils.o

我们知道，Makefile中的第一个目标会被作为其默认目标。我们声明了一个“all”的伪目标，其依赖于其它三个目标。由于伪目标的特性是，总是被执行的，所以其依赖的那三个目标就总是不如“all”这个目标新。所以，其它三个目标的规则总是会被决议。也就达到了我们一口气生成多个目标的目的。 “.PHONY : all”声明了“all”这个目标为“伪目标”。

随便提一句，从上面的例子我们可以看出，目标也可以成为依赖。所以，伪目标同样也可成为依赖。看下面的例子：

.PHONY: cleanall cleanobj cleandiff

cleanall : cleanobj cleandiff
rm program

cleanobj :
rm *.o

cleandiff :
rm *.diff

“make clean”将清除所有要被清除的文件。“cleanobj”和“cleandiff”这两个伪目标有点像“子程序”的意思。我们可以输入“make cleanall”和“make cleanobj”和“make cleandiff”命令来达到清除不同种类文件的目的。

##### 多目标

Makefile的规则中的目标可以不止一个，其支持多目标，有可能我们的多个目标同时依赖于一个文件，并且其生成的命令大体类似。于是我们就能把其合并 起来。当然，多个目标的生成规则的执行命令是同一个，这可能会可我们带来麻烦，不过好在我们的可以使用一个自动化变量“$@”

#### Makefile内容

Makefile里主要包含了五个东西：显式规则、隐晦规则、变量定义、文件指示和注释。

1、显式规则。显式规则说明了，如何生成一个或多的的目标文件。这是由Makefile的书写者明显指出，要生成的文件，文件的依赖文件，生成的命令。

2、隐晦规则。由于我们的make有自动推导的功能，所以隐晦的规则可以让我们比较粗糙地简略地书写Makefile，这是由make所支持的。

3、变量的定义。在Makefile中我们要定义一系列的变量，变量一般都是字符串，这个有点你C语言中的宏，当Makefile被执行时，其中的变量都会被扩展到相应的引用位置上。

4、文件指示。其包括了三个部分，一个是在一个Makefile中引用另一个Makefile，就像C语言中的include一样；另一个是指根据某些情况指定Makefile中的有效部分，就像C语言中的预编译#if一样；还有就是定义一个多行的命令。

5、注释。Makefile中只有行注释，和UNIX的Shell脚本一样，其注释是用“#”字符，这个就像C/C++中的“//”一样。如果你要在你的Makefile中使用“#”字符，可以用反斜框进行转义，如：“\#”。

规则包含两个部分，一个是依赖关系，一个是生成目标的方法

在Makefile中，规则的顺序是很重要的，因为，Makefile中只应该有一个最终目标，其它的目标都是被这个目标所连带出来的，所以一定要让 make知道你的最终目标是什么。一般来说，定义在Makefile中的目标可能会有很多，但是第一条规则中的目标将被确立为最终的目标。如果第一条规则中的目标有很多个，那么，第一个目标会成为最终的目标。make所完成的也就是这个目标。

#### 清空目标文件的规则

一般的风格都是：

```
clean:
rm edit $(objects)
```

更为稳健的做法是：

```
.PHONY : clean  (phony,<非正式>虚伪的，做作的)
clean :
-rm edit $(objects)
```

#### MakeFile怎么运作

MakeFile里面大致是先处理一些环境变量或者参数，然后从某一个位置开始执行命令集。

1、读入所有的Makefile。

2、读入被include的其它Makefile。

3、初始化文件中的变量。

4、推导隐晦规则，并分析所有规则。

5、为所有的目标文件创建依赖关系链。

6、根据依赖关系，决定哪些目标要重新生成。

7、执行生成命令。

1-5步为第一个阶段，6-7为第二个阶段。第一个阶段中，如果定义的变量被使用了，那么，make会把其展开在使用的位置。但make并不会完全马上展 开，make使用的是拖延战术，如果变量出现在依赖关系的规则中，那么仅当这条依赖被决定要使用了，变量才会在其内部展开。

# 常用命令行

- cc、gcc、g++、CC的区别概括：[(17条消息) cc、gcc、g++、CC的区别概括_如小丧的博客-CSDN博客](<https://blog.csdn.net/liaoshengshi/article/details/40424397#:~:text=cc是Unix系统的C> Compiler，而gcc则是GNU Compiler Collection，GNU编译器套装。.,gcc原名为Gun C语言编译器，因为它原本只能处理C语言，但gcc很快地扩展，包含很多编译器（C、C%2B%2B、Objective-C、Ada、Fortran、Java）。. 因此，它们是不一样的，一个是古老的C编译器，一个是GNU编译器集合，gcc里面的C编译器比cc强大多了，因此没必要用cc。. 下载不到cc的原因在于：cc来自于昂贵的Unix系统，cc是商业软件。.)
- 超全整理！Linux shell及常用36类命令汇总：[超全整理！Linux shell及常用36类命令汇总 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/50448669)

-g 可执行程序包含调试信息，为了调试用的：加个-g 是为了gdb 用，不然gdb用不到

-o 指定输出文件名（o：output）
-c 只编译不链接：产生.o文件，就是obj文件，不产生执行文件（c : compile）

-o output_filename，确定输出文件的名称为output_filename，同时这个名称不能和源文件同名。如果不给出这个选项，gcc就给出预设的可执行文件a.out。
一般语法：
gcc filename.c -o filename
上面的意思是如果你不打 -o filename（直接gcc filename.c ）；那么默认就是输出a.out.这个-o就是用来控制输出文件的。 ------用./a.out执行文件

-c 只编译不链接

产生.o文件，就是obj文件，不产生执行文件

# Cmake

# References

- [浅析Makefile、make、cmake - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/431118510)

- [Makefile详解（超级好）【mingw吧】_百度贴吧 (baidu.com)](https://tieba.baidu.com/p/591519800?red_tag=2203856275)
- [(17条消息) 静态库与动态库的区别与优缺点_静态库和动态库的优缺点_雨荔@秋垣的博客-CSDN博客](https://blog.csdn.net/weixin_51483516/article/details/120837316)

- 【GNU Makefile编译C/C++教程（Linux系统、VSCODE)】<https://www.bilibili.com/video/BV1EM41177s1?p=22&vd_source=93bb338120537438ee9180881deab9c1>!!!
