# 课程

- 平时作业（40%——8次，每次5分）：源代码+运行截图交在学在浙大
- 大作业（60%）任选一个：实验报告（两个部分，第一部分写必做的，第二部分写可选部分；即先做特殊部分——感觉就是仿真（不要调用其他奇怪的库），然后做附加）+完整源代码

# Makefile

- g++与gcc是不同的。g++是C++编译器。gcc是C编译器

- C++模板类编译有问题：必须同时包含.h和.cpp文件才能实现不报错（不然会报undefined）

  所以模板类就写一个.cpp文件就好了，全写在这个里面







# 绪论

- 过程式：程序=算法+数据结构

## 对象

对象技术（OT）——有边界，能判断

- 对象，是现实世界中某个实体在计算机逻辑中的映射和描述。对象具有标识(identity)，在一个明确的边界(boundary)里封装了状态(state)和行为(behavior)。

  State is represented by attributes(属性) and relationships.

  Behavior is represented by operations,methods, and state machines.

## 类

和struct差不多，就是在结构中添加了函数

- A class is a description of a set of objects that share the same attributes operations,relationships, and semantics(语义).
  - An object is an instance of a class.

- 类的UML表示

  ![image-20231113151033532](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231113151033532.png)

## OO的四个基本原理

Object Orientation面向对象技术

- Abstraction抽象

- Encapsulation封装：Clients depend on interface.

  <img src="https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231113151648086.png" alt="image-20231113151648086" style="zoom:50%;" />

- Modularity模块化

- Hierarchy层次



## 多态性和泛化

- 多态性Polymorphism：多态性是能够把多种不同的实现隐藏在一个单一的接口后面的能力

- 泛化Generalization：类与类之间的关系，一个类共享其它类的结构和行为

  继承（单继承和多继承）：重用接口

  - 添加全新的函数 is-like-a
  - 只改变基类中函数的行为，即“重载”函数 is-a



# 初探

[C++ 基本语法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/cplusplus/cpp-basic-syntax.html)

- Type safety类型安全性
- classes类
- templates模板

![image-20231120133519097](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231120133519097.png)

- 两种赋值方法：

  ```
  int sum(0);
  int sum=0;//两种在赋初值的时候是等价的
  ```

  

- iosteam

  ```
  #include <iostream>
  using namespace std;
  int main() {
  cout << "Hello, world" << endl; //函数的链式调用，一个<<调用一次
  //endl，这将在每一行后插入一个换行符
  }  
  ```

> C++输入输出是类型安全的输入输出（会自动检查）

- new and delete 来代替malloc/free

  ```
  int* ip = new int(7);		//分配一个空间来放整数7（自动计算空间大小）
  delete ip;
  int* iap = new int[10];		//分配10个整数数组的空间
  delete []iap;
  ```

  用new分配的是堆空间，自己创建的是栈空间

  delete释放堆空间（这里与C不一样，C++需要告诉他是数组[ ]）



- 三字符组

  | 三字符组 | 替换 |
  | :------- | :--- |
  | ??=      | #    |
  | ??/      | \    |
  | ??'      | ^    |
  | ??(      | [    |
  | ??)      | ]    |
  | ??!      | \|   |
  | ??<      | {    |
  | ??>      | }    |
  | ??-      | ~    |

  想保留??要用转义?\?



## 类型转换

- 静态转换（static cast）:

  静态转换是将一种数据类型的值强制转换为另一种数据类型的值。

  静态转换通常用于比较类型相似的对象之间的转换，例如将 int 类型转换为 float 类型。

  静态转换不进行任何运行时类型检查，因此可能会导致运行时错误。

  ```
  int i = 10;
  float f = static_cast<float>(i); // 静态将int类型转换为float类型
  ```

- 动态转换（Dynamic Cast）

  动态转换通常用于将一个基类指针或引用转换为派生类指针或引用。动态转换在运行时进行类型检查，如果不能进行转换则返回空指针或引发异常

  ```
  class Base {};
  class Derived : public Base {};
  Base* ptr_base = new Derived;
  Derived* ptr_derived = dynamic_cast<Derived*>(ptr_base); // 将基类指针转换为派生类指针
  ```

- 常量转换（Const Cast）

  常量转换用于将 const 类型的对象转换为非 const 类型的对象。

  常量转换只能用于转换掉 const 属性，不能改变对象的类型。

  ```
  const int i = 10;
  int& r = const_cast<int&>(i); // 常量转换，将const int转换为int
  ```

- 重新解释转换（Reinterpret Cast）

  重新解释转换将一个数据类型的值重新解释为另一个数据类型的值，通常用于在不同的数据类型之间进行转换。

  重新解释转换不进行任何类型检查，因此可能会导致未定义的行为。

  ```
  int i = 10;
  float f = reinterpret_cast<float&>(i); // 重新解释将int类型转换为float类型
  ```



## 对象

automatic initialization/cleanup构造和析构函数，构造函数/析构函数名和类名相同，析构函数前面还要加一个~

```
class StackOfInt {
public:
	StackOfInt(int); //构造函数，生成对象时自动执行
	~StackOfInt(); //析构函数，撤销对象时自动执行
//
protected:
	string m_name; //外部不可见
```



## Templates

模板：参数化的类

generic programming：泛型编程

```
template<class T>	//typename和class都可以用
class Stack{
	T pop();
}

template<class T>
Stack<T>::pop()
{
}
//使用时换掉T即可
Stack<float> stk(5);
```

> card=rank+suit（花色） 



## virtual

基类的函数调用如果有virtual则根据多态性调用派生类的，如果没有virtual则是正常的静态函数调用，还是调用基类的。

```
virtual bool IsHitting() const=0;  //纯虚函数
```



# C++中的C

## C++的词法及词法规则

- 标识符

- 关键字

  ![image-20231127142603960](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231127142603960.png)

- 作用域分辨符

  ```
  class_name::member
  ::name	//全局作用域分辨符
  
  int x;
  void f()
  {
  	int x;
  	x=2;
  	::x=10;
  }
  
  ```

- 堆分配/回收运算符：new,delete

```
char a[10]="hello";	//不会报错，这里=不是赋值运算符，相当于char a[10]("hello");
//给创建函数传了参
```

## 基本数据类型

- 修饰符：
  - volatile：告诉编译程序，该变量值可能按程序中没有显示说明的方式改变，防止编译器作不正确的的优化
  - opertor(重载)：[C++编程语言中重载运算符（operator）介绍_c++ operator-CSDN博客](https://blog.csdn.net/liitdar/article/details/80654324)

> 注： C的规范(C++继承)并不说明每一个内部类型必须有多少位，而是规定内部类型必须能存储的最大值和最小值。因此，内部类型的字长跟硬件相关  

- bool类型：C++: 内部常量true表示“真”， false表示“假”  （不能用0和！0来表示）

- 指针

  [C/C++函数指针与指针函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/206424665)

  ```
  //函数指针
  int (*fun)(int x，int y) //函数指针的定义
  fun = &Function          //函数指针的赋值方式1
  fun = Function           //函数指针的赋值方式2
  x = (*fun)()             //函数指针的调用方式1
  x = fun()                //函数指针的调用方式2
  ```

  

- 引用（Reference）:引用是通过**别名**直接访问某个变量，对引用的操作就是对被引用的变量的操作

  ```
  int a=10;
  int &b=a; // b是a的别名；
  b=b+10; // 对b的改变，实际上改变的是a；
  cout << a;
  ```

  > 引用定义时必须初始化；初始化后，不能改变引用的“指向”

- typedef创建类型的别名

- 引用和指针的选择
  - 引用不会造成内存泄漏
  - 引用不适合动态内存管理
  - 链表等数据结构必须配合指针和动态内存使用
  - 尽量只在动态内存管理时使用指针



## 程序结构和编译环境

由类和函数组成

![image-20231127161403709](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231127161403709.png)

- C++提供一下类用于文件I/O:
  - ofstream：写入文件
  - ifstream：读出文件
  - fstream：读写文件

```
#include <fstream>
int main(){
    std::ofstream square_file; //建立文件操作流对象
    square_file.open(“squares.txt”);
    for (int i=0;i<10;++i)
    	square_file<<i<<“^2=”<<i*i<<std::endl;
    square_file.close();
}
```

## 错误处理

- 错误处理：

  - assert

    ```
    #include <cassert>
    double square_root(double x)
    {
        check(x>=0);
        ………
        assert(result>=0); //如果不成立，程序立刻终止
        return result;
    }
    ```

  - 用于对程序的各种细节进行充分的测试（debug）

    然后在正式版本中无效化debug代码，可以提高程序速度

    ```
    #define NDEBUG
    #include <cassert>
    ```

    

- 异常处理

  - try-catch模式

    ```
    struct cannot_open_file {}; //自定义异常类型
    void read_matrix_file(const char* fname,…)
    {
        fstream f(fname);
        if (!f.is_open())
        throw cannot_open_file{};
    }
    ```

    • C++几乎允许程序员抛出任何类型的异常

    • 大型程序开发通常需要建立自己的异常类型架构，一般从标准库std::exception派生而来

    • 异常抛出后，可以选择即时处理或者延后处理

    ```
    bool keep_trying=true;
    do {
    char fname[80];
    cin>>fname;
    try{
        A=read_matrix_file(fname);
        …
        keep_trying=false; //没有异常抛出，修改flag
    } catch (cannot_open_file& e){
    	cout<<“Could not open the file. Try again!\n”;
    } catch (…) { // 捕获所有其它异常
    …
    }
    } while (keep_trying);
    
    
    //e:err就是里面throw出来的错误
    
    ```

- 智能指针：C++的智能指针主要方便内存管理，定义在头文件<memory>

  Unique Pointer（对内存独占的控制权）

  ```
  #include <memory>
  int main(){
      unique_ptr<double> dp{new double};
      *dp=7;
  }
  ```

  - 智能指针在生命周期结束时自动释放对象内存（因此智能指针指向的内存地址必须是动态分配的！）

  - unique_ptr指针不能改变类型，也不能赋值给另一个unique_ptr

  - shared pointer：

    • shared_ptr 利用引用计数（额外的内存占用）实现内存的自动管理。每当复制一个shared_ptr，引用计数+1；一个shared_ptr离开作用域时，引用计数-1。引用计数清零时，自动delete内存

    • 适用领域：并发和多线程

## 文件I/O

```
#include<ofstream>  //写入文件
```



# 变量

## 引用

references must be initialized when defined

- can`t be null
- 没有引用的引用

```
int x=3;
int& y=x;//通过y可以修改x
int& z=y//错误！！！
const int& z=x;//通过z不能修改x
```

## 变量

- 寄存器变量：

  在函数中频繁使用的局部变量可以定义为寄存器变量(用register修饰)，只要还有寄存器可用，编译器会尽可能将其放入寄存器中，提高执行速度

  ```
  register int i=0;
  ```



- ???

  ```
  namespace lib1{ int a; }
  namespace lib2{ int a; }
  lib1::a
  ```

  



# 函数

参数化的共享代码

定义在先，调用在后，调用前可以不必声明；

定义在后，调用在前，调用前必须声明。

## 传递

- 值传递

- 指针传递

- 引用传递：

  ```
  void swap3(int &v1,int &v2)
  {
  	int temp = v2; v2 =v1; v1 = temp;
  }
  void main(){
  	int a=5,b=9; swap3(a,b);
  }
  ```

  函数定义：形参声明为引用量；函数体直接对引用量操作；

  函数直接调用，修改会影响到实参

  > 用指针要处理空间（从现在开始少用）；简洁，简单

传引用使用时不需要加&,只需要在定义时加上&

- 

## 内联函数

inline function——对宏替换的改进  

告诉编译器，在编译时用函数代码的“拷贝替换函数调用

定义和声明时都需要加inline关键字

```
inline int power_int(int x)
{
	return x*x;
}
```

> Notes:在内联函数内不允许用循环语句和开关语句
>
> 内联函数的定义必须出现在每一个调用该函数的源文件之中；(由于要进行源代码替换)
>
> -- 编译器不保证所有被定义为inline 的函数编译成内联函数。Eg:递归函数

> Tips:类内实现的函数自动内联



## 默认参数

设置默认值：——只能写在声明，定义的地方不重写了

```
int setclock(TIME t, int hour=12,int minute=0,int
second=0){
    t->h = (hour <= 23 && hour >=0)? hour : 12;
    t->m=(minute<=59 &&minute>=0)? minute:0;
    t->s=(second<=59 && second >=0)?second:0;
}
```

- 若有实际参数值，则缺省值无效

> Tips:带有缺省值的参数**必须全部放置在参数的最后**，即在带有缺省值的参数的右边不再出现无缺省值的参数  

![image-20231204143359390](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231204143359390.png)

> 声明时定义了缺省参数，定义时就不要再写了

- 占位符参数：

  函数声明时，参数可以没有标识符，称这种参数为“占位符参数” ，或称“哑元”

  ```
  // 声明
  void f(int x, int = 0 , float = 1.1);
  ```

## 函数重载

优于占位符，实际上是多个同名函数

重载：把多个功能类似的函数定义成同名函数

> 怎么知道调用哪个函数：根据参数类型、个数的不同来判断；——避免二义性

- 编译器如何处理重载函数：

  ![image-20231204144316231](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231204144316231.png)

  ```
  extern "C"
  max(a,b);
  //强迫C++编译器按照C的函数命名规则去连接相应的目标代码
  ```



## 标准模板库

容器定义在C++标准模板库中

std::vector是一种常用的容器，功能与数组相似

- 标准模板库（STL）提供标准容器和算法

  方便，大量预定义的数据结构和封装好的算法  

- 迭代器（Iterator）用于连接容器和算法



# 类

- 数据抽象：数据成员and成员函数
- 实现隐藏（访问控制）
- 自动初始化和清除（构造函数和析构函数）

> 不定义构造函数也是OK的，编译器会创建一个默认的（但是大概率不是我们想要的）

- 显示调用析构函数

  如果只想执行析构函数中的执行的操作，而不释放对象的空间，则可以显式调用析构函数



- 拷贝构造函数？

## 友元

[C++ 友元函数 | 菜鸟教程 (runoob.com)](https://www.runoob.com/cplusplus/cpp-friend-functions.html)

- 友元不是类的成员，只是一个声明。
- 友元可以是外部全局函数、类或类的成员函数；
- 友元是一个“特权”函数，破坏了封装。当类的实现发生变化后，友元也要跟着变动

```
/* 声明类 ClassTwo 的所有成员函数作为类 ClassOne 的友元，需要在类 ClassOne 的定义中放置如下声明： */

friend class ClassTwo;
```



## 组合

拿对象拼成一个新的类

组合有两种：**fully** （一个对象在另一个对象内部）和 **by reference** （一个对象通过指针引用另一个对象）。fully 和 reference 是两种不同的**内存模型（在内存中保存如何存储的）**

- ![image-20231225150541656](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231225150541656.png)

  这就是fully

  ![image-20231225150916277](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231225150916277.png)

  尽管在body里面，但也不能随机使用

- 如果是指针就是by reference

## 继承

- 所有数据都放在private
- public给所有人用
- protected给子类用，别人不能用

inheritance

拿类拼成一个新的类

example:

```
class Student1: public Student
{
public:
	void display_1( )
    { display();
    cout<<″age: ″<<age<<endl;
    cout<<″address: ″<<addr<<endl;}
private:
    int age;
    string addr;
}
Student1 stu1;
stu1.display_1();
```

继承方式：

```
class 派生类名: ［继承方式］ 基类名1，［继承方式］ 基类名2
{派生类新增加的成员} ;
//可以同时继承多个父类
```

> 类的默认继承方式是私有的

```
public:
    Teacher(const string& name, int age, const string& title)
    : Person(name, age), m_title(title){}
```

> 可以直接用：来写

C++中支持三种不同的继承方式：

- public继承：父类成员在子类中保持原有访问级别。
- private继承：父类成员在子类中变为私有成员。
- protected继承：父类中的公有成员变为保护成员，其它成员保持不变

> 继承方式不影响对基类接口的访问权限

- public继承

  ```
  class Teacher : public Person
  ```

- private继承

  ```
  class Teacher : private Person
  ```


- protected继承

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200210211449294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1Nsb3dJc0Zhc3RMZW1vbg==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200210211449294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1Nsb3dJc0Zhc3RMZW1vbg==,size_16,color_FFFFFF,t_70)

- 同名覆盖——本质上是作用域的问题

  - 取消覆盖的方法

  - 利用作用域运算符

    ```
    //.h
    class Derived: public Base{
    public:
    	virtual void func1(); //纯虚函数重写后，这样才能实例化。
    	void func3(int a,int b){ Base::func3(a);...} //方式1：作用域运算符
    	virtual void func1(int a){Base::func1(a);} // 派生类声明同名函数，函数中调用基类同名函数，这被称为转交函数
    }
    
    ```

  - 利用using声明

    ```
    //.h
    class Derived: public Base{
    public:
    	using Base::func1; //让基类中名为func1的所有函数都可用，无论特征标是什么
    	using Base::func3; //让基类中名为func3的所有函数都可用，无论特征标是什么
    	virtual void func1(); //纯虚函数重写后，这样才能实例化。
    	void func3(int a,int b){ Base::func3(a);...} //方式1：作用域运算符
    }
    
    //.cpp
    Derived d;
    d.func1(2); //正确，调用基类Base中func1(int)。
    d.func3(5); //正确，调用基类Base中func3(int)。
    d.func3(); //正确，调用基类Base中func3()。
    
    ```

- 析构函数不能继承——派生类的构造函数只需要对新增的成员进行初始化即可，对所有从基类继承来的成员，其初始化工作还是由基类的构造函数完成

![image-20231211085218766](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231211085218766.png)

- 调用基类含参构造函数
- ![image-20231211085457316](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231211085457316.png)

- 为避免二义性，需要利用作用域限定符::，把基类的成员与下一层基类关联起来

- 虚基类(virtual inheritance)——类层次结构中虚基类的成员只出现一次，即基类的一个副本被所有派生类对象所共享

  > C++只执行最后的派生类对虚基类的构造函数的调用，而忽略虚基类的其他派生类(如类B和类C) 对虚基类的构造函数的调用

![image-20231211090321266](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231211090321266.png)

### Problem

C++继承了两个类都调用了相同的函数，如何让它只调用一次：

**虚拟继承**是C++中用于实现虚基类的机制。 通过虚拟继承（virtual inheritance），可以确保在多重继承中只有一个共享的虚基类实例。 虚拟继承使用关键字"virtual"来声明基类，以指示这是一个虚基类。 虚拟继承的主要目的是解决多重继承中的菱形继承问题和冗余数据问题。

[C++多重继承重复调用的解决_c++ 多继承 父类 方法相同-CSDN博客](https://blog.csdn.net/qq_15029743/article/details/79418795)

- 父类有相同函数名的函数（重载）；子类若有与父类相同的函数（函数名+参数表）——则父类该函数名函数全部隐藏


> C++函数的重载之间没关系，其他语言有

## 多态

两种多态

![image-20231211090444058](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231211090444058.png)

- 虚函数的作用是允许在派生类中重新定义与基类同名的函数，并且可以通过基类指针或引用来访问基类和派生类中的同名函数

  > C++不是通过不同的对象名去调用不同派生层次中的同名函数，而是通过指针调用它们

```
virtual <类型说明符><函数名>(<参数表>) ;
```

有一个virtual，最好把其他全部写成virtual

- 纯虚函数

```
virtual <类型说明符><函数名>(<参数表>) =0;
```

> 有了virtual子类和父类的同名函数才有联系
>
> 只要一个父类那个是virtual，后代（子、孙）这个函数无论加不加virtual都是virtual
>
> 变成动态绑定（不再是静态绑定）

纯虚函数的特点：

- 只有声明，没有实现/定义
- 含有纯虚函数的类称为**抽象类**，抽象类不能被实例化
- 抽象类的派生类如果想成为具体的类（能够被实例化），则必须重写纯虚函数。（关于函数重写与函数重载，见后文介绍）

虚函数的特点：

- 必须实现/被定义
- 虚函数所在类可以被实例化

> 一个类有虚函数，那么它的析构函数也应该是虚的???

- 重写(overreide):

  指派生类中存在重新定义的函数。其函数名，参数列表，返回值类型，所有都必须同基类中被重写的函数一致。只有函数体不同（花括号内），派生类调用时会调用派生类的重写函数，不会调用被重写函数。基类中被重写的函数必须是虚函数/纯虚函数。

- 重载(overload):

  指同一可访问区内被声明的几个具有不同参数列表（参数的类型，个数，顺序不同）的同名函数，根据参数列表确定调用哪个函数，重载不关心函数返回类型。

有virtual，则不同类之间第一个地方存的是vtable指针，指向它自己对应的函数

> 注意子类和父类可以直接赋值（最好不要这么干）

![image-20231228160256508](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231228160256508.png)





## 运算符重载

运算符重载是通过定义函数来实现。运算符重载实质是函数的重载

operator专门用于定义重载运算符函数

![image-20231211093124375](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231211093124375.png)

- 成员函数

```
Complex::Complex operator+ (int &x)
{return Complex(real+x,imag);}
```

- 友元函数

> 一般将单目运算符重载为成员函数，将双目运算符重载为友元函数

```
Complex operator+ (int x,Complex &c)
{return Complex(x+c.real,c.imag);}
```

![image-20231211094005296](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20231211094005296.png)

为什么c++重载运算符时返回值为类的对象或者返回对象的引用？

最终的目的是为了进行连续的运算

```
classA& operator+(classA &a, classA &b)
{
	classA temp;
	//加法运算
	return *temp;
}
```





## 拷贝构造函数

使用：

```
Student stu2(stu1);
```

定义方法：

```
Student(Student&);
Student(const Student&);//const防止修改原来引用对象的值
//可以多参数，但第二个参数开始必须有默认值
```

> 当对象的引用作为参数时，**可以直接打点访问该对象的私有成员**

- 浅拷贝指向相同空间



可以使用delete指定不生成拷贝构造函数和赋值

```
class Student
{
    public:
        Student(const Student& p) = delete;
        Student& operator=(const Student& p) = delete;
    private:
        unsigned id;
        string name;
};
```

```
class MyStr str2; 
str2 = str1; 
//注意这两种方式不同, 前者是赋值运算, 后者是拷贝构造
class MyStr str3 = str2;
```

## this指针

- 通过返回*this的引用，可以实现成员函数的链式调用

  ![image-20231218145745588](C:\Users\Immortal\AppData\Roaming\Typora\typora-user-images\image-20231218145745588.png)

- 如果是自身赋值——需要处理

```
if(this == &s) {
    error(“自身赋值”);
    return(*this);
}
```





# Bonus

- namespace命名空间——防止名称冲突

  [详解c++的命名空间namespace - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/126481010)

  ```
  namespace A {
      int a = 100;
  }
  namespace B {
      int a = 200;
  }
  A::a=10;
  
  ```

  - 命名空间只能全局范围内定义
  - 可以嵌套

- [c++11新特性，所有知识点都在这了！](https://zhuanlan.zhihu.com/p/139515439)




















