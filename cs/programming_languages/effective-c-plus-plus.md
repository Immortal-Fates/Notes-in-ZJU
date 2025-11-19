# Main Takeaway

# eff_C++

1. 将 C++ 视为 federation of languages（语言联合体）

2. 用 consts, enums 和 inlines 取代 #defines

   Things to Remember

   - 对于 simple constants（简单常量），用 const objects（const 对象）或 enums（枚举）取代 #defines。
   - 对于 function-like macros（类似函数的宏），用 inline functions（内联函数）取代 #defines。

   ![image-20240626215137739](assets/eff_C++.assets/image-20240626215137739.png)

   - 函数会不会修改成员加const说明——语义

     ```C++
     const char& operator[](std::size_t pos) const
     {}
     ```

   - 一个是const 一个是non-const ，但是内容除了能否被修改差不多

     让non-const调用const版本是避免代码重复的安全方法

     ![image-20240626215817037](assets/eff_C++.assets/image-20240626215817037.png)

     > 返回值去掉const，调用时加上const

3. 确保 objects（对象）在使用前被初始化

   - 使用前就初始化：特别是对于built-in types的non member objects

     > 数组不一定保证其内容初始化，而vector的内容必须初始化

   - 在类调用时在调用 constructor（构造函数）的函数体之前，它们的 default constructors（缺省的构造函数）已经被自动调用——即构造函数仅仅是赋值，而不是被初始化

     不太好的写法：

     ```c++
     ABEntry::ABEntry(const std::string& name, const std::string& address,
                      const std::list<PhoneNumber>& phones)

     {

       theName = name;                       // these are all assignments,

       theAddress = address;                 // not initializations

       thePhones = phones;

       numTimesConsulted = 0;

     }
     ```

     更好的写法：

     ```c++
     ABEntry::ABEntry(const std::string& name, const std::string& address,
                      const std::list<PhoneNumber>& phones)

     : theName(name),
       theAddress(address),                  // these are now all initializations
       thePhones(phones),
       numTimesConsulted(0)

     {}                                      // the ctor body is now empty
     ```

     > 这是拷贝构造

   - 将非局部静态对象替换为局部静态对象

     ```c++
     FileSystem& tfs() //这将替换tfs对象
     {
      static FileSystem fs; //初始化一个局部静态对象
      return fs;    //返回对它的引用
     }
     ```

     > 防止tfs先调用而fs还没初始化，没有注册

4. 构造、析构、赋值运算

   如果不自己声明，编译器可能（不同编译器不同）会替你声明拷贝构造函数、拷贝赋值运算符和析构函数

   ![image-20240626221246653](assets/eff_C++.assets/image-20240626221246653.png)

   如果不想使用，则放在private里面，为了防止成员函数和友元函数调用，只提供声明，不提供定义

   ![image-20240626222928316](assets/eff_C++.assets/image-20240626222928316.png)

# References
