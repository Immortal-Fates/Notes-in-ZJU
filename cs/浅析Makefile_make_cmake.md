# Main Takeaway



# 简介

cmake生成makefile文件给make执行

- makefile中设置你想要的编译规则，你想要编译哪些文件，哪些文件不需要编译等等都可以体现在Makefile中，而且支持多线程并发操作，可以减少编译的时间。

- make就是根据Makefile中写的内容进行编译和链接

  > Tips:make只会编译我们修改过的文件，没有修改过的就不用重新编译，这样我们debug了一个小bug后重新编译就不用花费大量的编译时间

- cmake跨平台项目管理的工具生成Makefile文件给make去执行，实现跨平台。cmake它仍然是目标、依赖之类的抽象的东西，根据一个叫CMakeLists.txt的文件（还是自己手写）生成Makefile的

# References

- [浅析Makefile、make、cmake - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/431118510)

