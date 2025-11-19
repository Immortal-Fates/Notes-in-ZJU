# Main Takeaway

《视觉SLAM十四讲》——每讲包括理论+实验

博客园 Jessica&jie有每一讲的文章，可以看看[高翔slam](https://zzk.cnblogs.com/s?w=高翔slam)

<!--more-->

# CH1_引言

![image-20240928212032513](markdown-img/SLAM十四讲.assets/image-20240928212032513.png)

推书：

![image-20240928212254451](markdown-img/SLAM十四讲.assets/image-20240928212254451.png)

![image-20240928212435644](markdown-img/SLAM十四讲.assets/image-20240928212435644.png)

code

<https://github.com/gaoxiang12/slambook>

# CH2_初识SLAM

自主运动两大基本问题：定位+建图——相互耦合的两个问题

sensors:

![image-20240928214537284](markdown-img/SLAM十四讲.assets/image-20240928214537284.png)

> 环境中的传感器限制了应用环境

![image-20240928221656708](markdown-img/SLAM十四讲.assets/image-20240928221656708.png)

**视觉SLAM框架**

![image-20240928221950985](markdown-img/SLAM十四讲.assets/image-20240928221950985.png)

VO：知道局部环境的运动

不可避免又飘移的现象——需要后端进行全局优化

然后回环检测（检测回到原本的地方）

![image-20240928222635858](markdown-img/SLAM十四讲.assets/image-20240928222635858.png)

**SLAM问题的数学描述**

![image-20240928222816591](markdown-img/SLAM十四讲.assets/image-20240928222816591.png)

![image-20240928223054839](markdown-img/SLAM十四讲.assets/image-20240928223054839.png)

![image-20240928223430940](markdown-img/SLAM十四讲.assets/image-20240928223430940.png)

## 实验

介绍ubuntu下的cmd and cmake

# CH3_三维空间的刚体运动

# CH4_李群和李代数

sophus库——eigen库的扩展

![image-20240929101217274](markdown-img/SLAM十四讲.assets/image-20240929101217274.png)

![image-20240929101301523](markdown-img/SLAM十四讲.assets/image-20240929101301523.png)

![image-20241002135000975](markdown-img/SLAM十四讲.assets/image-20241002135000975.png)

## Basic

要让每个李代数都对应一个李群

![image-20240929101703436](markdown-img/SLAM十四讲.assets/image-20240929101703436.png)
$$
旋转矩阵SO(3),齐次变换矩阵SE(3)
$$
群(group)：集合+运算

![image-20240929101807796](markdown-img/SLAM十四讲.assets/image-20240929101807796.png)

![image-20240929102147309](markdown-img/SLAM十四讲.assets/image-20240929102147309.png)

流形：光滑的表面——可用微积分

李群（Lie Group）——既是群也是流形

> 圆是一维流形，圆环是二维流形

![image-20240929102330678](markdown-img/SLAM十四讲.assets/image-20240929102330678.png)

李代数——李群单位元处的正切空间

> 因此我们可以通过指数映射和对数映射在两者之间相互转化

![image-20240929103035068](markdown-img/SLAM十四讲.assets/image-20240929103035068.png)

对左右两边求导得

![image-20240929103843827](markdown-img/SLAM十四讲.assets/image-20240929103843827.png)

可见$\dot{R}(t)R(t)^T$是反对称矩阵，把反对称符号记为如下形式

![image-20240929103931702](markdown-img/SLAM十四讲.assets/image-20240929103931702.png)

![image-20240929104345937](markdown-img/SLAM十四讲.assets/image-20240929104345937.png)

![image-20240929104739046](markdown-img/SLAM十四讲.assets/image-20240929104739046.png)

李代数（Lie Algebra）——理解为一个向量空间

![image-20240929112037271](markdown-img/SLAM十四讲.assets/image-20240929112037271.png)

> 自反性？雅可比等价？

李括号运算：表达了两个元素的差异

![image-20240929112833849](markdown-img/SLAM十四讲.assets/image-20240929112833849.png)

![image-20240929113417812](markdown-img/SLAM十四讲.assets/image-20240929113417812.png)

## 指数映射和对数映射

指数映射

![image-20240929115952662](markdown-img/SLAM十四讲.assets/image-20240929115952662.png)

> a是单位长度的向量，$\theta$为长度
>
> 求指数运算：Taylor展开

![image-20240929120355428](markdown-img/SLAM十四讲.assets/image-20240929120355428.png)

![image-20240929120828164](markdown-img/SLAM十四讲.assets/image-20240929120828164.png)

## 李代数求导与扰动模型

解决李群上没有加法的问题，李群只有乘法

problem：李代数上做了加法后，李群上发生了什么事情？

![image-20240929121338439](markdown-img/SLAM十四讲.assets/image-20240929121338439.png)

很遗憾上式在$\phi$为矩阵的情况下不成立

![image-20240929121720007](markdown-img/SLAM十四讲.assets/image-20240929121720007.png)

这时可以得到BCH的线性近似形式

![image-20240929122122163](markdown-img/SLAM十四讲.assets/image-20240929122122163.png)

李群上面的乘法为李代数的以下式子

李代数的加法为李群的以下式子

![image-20240929122428515](markdown-img/SLAM十四讲.assets/image-20240929122428515.png)

![image-20241002153836728](markdown-img/SLAM十四讲.assets/image-20241002153836728.png)

对李群导数的定义有两种模型——导数模型+扰动模型

![image-20241002154045299](markdown-img/SLAM十四讲.assets/image-20241002154045299.png)

![image-20241002155807018](markdown-img/SLAM十四讲.assets/image-20241002155807018.png)

扰动模型结果更简洁

下面介绍一些扰动模型

![image-20241002160035538](markdown-img/SLAM十四讲.assets/image-20241002160035538.png)

# CH5_相机与图像

![image-20241002215727282](markdown-img/SLAM十四讲.assets/image-20241002215727282.png)

## 相机模型

相机成像的过程

![image-20241005134722545](markdown-img/SLAM十四讲.assets/image-20241005134722545.png)

![image-20241005135503918](markdown-img/SLAM十四讲.assets/image-20241005135503918.png)

下面会介绍相机模型和相机内外参

### 单目模型

普通相机可以用针孔模型很好地近似

![image-20241002220227421](markdown-img/SLAM十四讲.assets/image-20241002220227421.png)

![image-20241005133729321](markdown-img/SLAM十四讲.assets/image-20241005133729321.png)

![image-20241005133921215](markdown-img/SLAM十四讲.assets/image-20241005133921215.png)

> K为内参矩阵
>
> 传统习惯说明：改变Z时，投影点仍是同一个

![image-20241005134405715](markdown-img/SLAM十四讲.assets/image-20241005134405715.png)

> 因为SLAM下相机会动，所以要将世界坐标系下的坐标转换到相机坐标系下
>
> 因此这里R,t,T称为外参

**畸变**

相机前一般有透镜

![image-20241005135043631](markdown-img/SLAM十四讲.assets/image-20241005135043631.png)

径向畸变：跟像离中心距离有关系

切向畸变：跟像离中心夹角有关系

![image-20241005135215303](markdown-img/SLAM十四讲.assets/image-20241005135215303.png)

均用多项式来描述

> 标定的时候会将这些参数都标出来

### 双目模型

![image-20241005135915827](markdown-img/SLAM十四讲.assets/image-20241005135915827.png)

> problem：两个相机如何知道某个像素对应同一个点

### RGB-D相机

![image-20241005140337136](markdown-img/SLAM十四讲.assets/image-20241005140337136.png)

## 图像

![image-20241005140639352](markdown-img/SLAM十四讲.assets/image-20241005140639352.png)

> opencv默认彩色图为BGR

## 基本图像处理

cv::mat是浅拷贝

## 点云拼接

pcl点云库，.pcd文件

# CH6_非线性优化

[非线性优化（高翔slam---第六讲 ） - Jessica&jie - 博客园 (cnblogs.com)](https://www.cnblogs.com/Jessica-jie/p/7153014.html)——包括ceres and g20的基本使用方法

[贝叶斯---最大似然估计（高翔slam---第六讲 ） - Jessica&jie - 博客园 (cnblogs.com)](https://www.cnblogs.com/Jessica-jie/p/8846493.html)

![image-20241006190339316](markdown-img/SLAM十四讲.assets/image-20241006190339316.png)

## 状态估计问题

![image-20241006190717065](markdown-img/SLAM十四讲.assets/image-20241006190717065.png)

# CH7_视觉里程计

从本讲开始介绍SLAM系统的重要算法：visual odometry，VO

![image-20241007151235137](markdown-img/SLAM十四讲.assets/image-20241007151235137.png)

## 特征点法

pose——landmark

![image-20241007151459033](markdown-img/SLAM十四讲.assets/image-20241007151459033.png)

> 其中一种方式是视觉SLAM中，可以利用图像特征点作为SLAM中的路标

### 特征提取

![image-20241007160148055](markdown-img/SLAM十四讲.assets/image-20241007160148055.png)

> 角点、边缘、区块

特征点：关键点+描述子（周围信息，区分不同特征点）

ORB是视觉SLAM中常用的方法

![image-20241007160634462](markdown-img/SLAM十四讲.assets/image-20241007160634462.png)

FAST是一种提取关键点的方法（快）

Oriented Fast类似于计算重心，计算旋转指向重心

![image-20241007161330314](markdown-img/SLAM十四讲.assets/image-20241007161330314.png)

BRIEF是一种二进制的描述子，每一位表示附近一对点(一对pattern a\b，a>b,a<b分别取0，1的描述子)

ORB就是BRIEF加上旋转

### 特征匹配

![image-20241007162103769](markdown-img/SLAM十四讲.assets/image-20241007162103769.png)

- 暴力匹配——比较图一特征点a和图二特征点的距离
- 快速最近邻（FLANN）

### 2D-2D：对极几何

通过对极约束求解相机运动——用于相机初始化

特征匹配后，我们就得到了特征点之间的对应关系

![image-20241007163536388](markdown-img/SLAM十四讲.assets/image-20241007163536388.png)

![image-20241007164112325](markdown-img/SLAM十四讲.assets/image-20241007164112325.png)

$O_2P$在图一上的投影$e_1p_1$为极线，我们要求$T_{12}$——算相机的运动，即定位和建图

![image-20241007164657920](markdown-img/SLAM十四讲.assets/image-20241007164657920.png)

> 中间推导化简得到右下角的式子

![image-20241007165132743](markdown-img/SLAM十四讲.assets/image-20241007165132743.png)

> 最后两步计算位姿：左下角
>
> 按理五点法即可解决（麻烦），一般用八点法

![image-20241007165420741](markdown-img/SLAM十四讲.assets/image-20241007165420741.png)

![image-20241007165654231](markdown-img/SLAM十四讲.assets/image-20241007165654231.png)

> 没懂

![image-20241007170152595](markdown-img/SLAM十四讲.assets/image-20241007170152595.png)

单目的尺度不确定性

![image-20241007170555001](markdown-img/SLAM十四讲.assets/image-20241007170555001.png)

> 推导较复杂，直接调API即可

![image-20241007170833950](markdown-img/SLAM十四讲.assets/image-20241007170833950.png)

三角法

## 3D-2D：PNP

![image-20241008104444889](markdown-img/SLAM十四讲.assets/image-20241008104444889.png)

### 代数解法

加了噪声后结果不够鲁棒

- DLT：直接线性变换

  ![image-20241008105207835](markdown-img/SLAM十四讲.assets/image-20241008105207835.png)

  > 一个点提供两个方程，需要六对点

  ![image-20241008105542953](markdown-img/SLAM十四讲.assets/image-20241008105542953.png)

  上述求解忽略了旋转矩阵R本身的约束，所以需要将结果投影回$SO(3)$

- P3P：利用三对点求相机外参

  ![image-20241008110040527](markdown-img/SLAM十四讲.assets/image-20241008110040527.png)

  ![image-20241008110028327](markdown-img/SLAM十四讲.assets/image-20241008110028327.png)

  得到二元二次方程：使用吴氏消元法可得x,y解析解（用数值算也可

  ![image-20241008110455585](markdown-img/SLAM十四讲.assets/image-20241008110455585.png)

### 优化解法

Bundle Adjustment

[Bundle Adjustment---即最小化重投影误差（高翔slam---第七讲](https://www.cnblogs.com/Jessica-jie/p/7739775.html)

> 使用g2o——非线性优化求解器

![](markdown-img/SLAM十四讲.assets/image-20241008111425707.png)

![image-20241008112035956](markdown-img/SLAM十四讲.assets/image-20241008112035956.png)

![image-20241008112214818](markdown-img/SLAM十四讲.assets/image-20241008112214818.png)

![image-20241008112500915](markdown-img/SLAM十四讲.assets/image-20241008112500915.png)

## 3D-3D：ICP

ICP：iterative closest point迭代最近点

[三维点云匹配，ICP算法详解](https://zhuanlan.zhihu.com/p/397926700)

以下介绍线性解法（BA也可解）

![image-20241008121054214](markdown-img/SLAM十四讲.assets/image-20241008121054214.png)

> 根据配对好的3D点求解R,t，定义误差e进行最小二乘问题的求解

- 带匹配的点

  ![image-20241008121513784](markdown-img/SLAM十四讲.assets/image-20241008121513784.png)

  ![image-20241008140159081](markdown-img/SLAM十四讲.assets/image-20241008140159081.png)

  实际上就是先求质心然后相减得到去质心后的$q_i,q\prime_i$，利用上面公式求得W，

  使用SVD分解得到U，V，得到R。t可以根据R的值得到

  ```
      Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
  ```

![image-20241008140642830](markdown-img/SLAM十四讲.assets/image-20241008140642830.png)

![image-20241113221800925](markdown-img/SLAM十四讲.assets/image-20241113221800925.png)

## 总结

![image-20241009161444501](markdown-img/SLAM十四讲.assets/image-20241009161444501.png)

# CH8_视觉里程计（2）

提取特征点太耗时了——能否不提取特征计算VO

![image-20241009161725351](markdown-img/SLAM十四讲.assets/image-20241009161725351.png)

## 光流法

本质上估计像素在不同时刻图像中的运动

分为稀疏光流（类似于选特征点）和稠密光流

![image-20241009161936761](markdown-img/SLAM十四讲.assets/image-20241009161936761.png)

选角点，不用再加入周围的描述了

### 问题描述

![image-20241009162104230](markdown-img/SLAM十四讲.assets/image-20241009162104230.png)

整个问题基于一个灰度不变假设

![image-20241009162158249](markdown-img/SLAM十四讲.assets/image-20241009162158249.png)

### 数学推导

爱看不看

![image-20241009162333296](markdown-img/SLAM十四讲.assets/image-20241009162333296.png)

> $\frac{\partial I}{\partial t}$已知——两张图像对比

![image-20241009162525268](markdown-img/SLAM十四讲.assets/image-20241009162525268.png)

> 定义窗口得到多个方程获得超定方程。超定方程如何解——矩阵论

![image-20241009162731648](markdown-img/SLAM十四讲.assets/image-20241009162731648.png)

像素距离太远就G了，往往需要迭代多次

实践看例子

## 直接法

光流仅估计了像素间的平移，but

![image-20241009165404059](markdown-img/SLAM十四讲.assets/image-20241009165404059.png)

目标使用直接法优化相机位姿——已经有一个先验估计

### 数学推导

![image-20241009165731500](markdown-img/SLAM十四讲.assets/image-20241009165731500.png)

假设知道深度

现有一个估计的R,t但是不准确。根据计算后的$p_2$和$p_1$进行光度误差计算

> 假设光度不变

![image-20241009165919206](markdown-img/SLAM十四讲.assets/image-20241009165919206.png)

![image-20241009170318535](markdown-img/SLAM十四讲.assets/image-20241009170318535.png)

线性化，泰勒展开只取第一项。然后找三个偏导的物理意义

![image-20241009170419421](markdown-img/SLAM十四讲.assets/image-20241009170419421.png)

### 结果

![image-20241009170505440](markdown-img/SLAM十四讲.assets/image-20241009170505440.png)

希望梯度较大一点，对优化贡献大一点——所以找角点

直接法也不适合处理图像间差异太大的问题

![image-20241009181351736](markdown-img/SLAM十四讲.assets/image-20241009181351736.png)

优缺点小结：

![image-20241009181456572](markdown-img/SLAM十四讲.assets/image-20241009181456572.png)

### 实践：RGB-D直接法

# CH9_工程

讲了一个工程。自己看书

# CH10_后端

![image-20241010100450464](markdown-img/SLAM十四讲.assets/image-20241010100450464.png)

## EKF形式的后端

什么是后端——从带噪声的数据估计内在状态，状态估计问题

![image-20241010100913865](markdown-img/SLAM十四讲.assets/image-20241010100913865.png)

主流两种做法

![image-20241010100929076](markdown-img/SLAM十四讲.assets/image-20241010100929076.png)

- 渐进式以EKF为主
- 批量式以优化为主

![image-20241010101254373](markdown-img/SLAM十四讲.assets/image-20241010101254373.png)

### 数学描述

KF数学描述自己看，小结如下：

![image-20241010103920353](markdown-img/SLAM十四讲.assets/image-20241010103920353.png)

EKF：KF的非线性扩展

![image-20241010104328357](markdown-img/SLAM十四讲.assets/image-20241010104328357.png)

即将非线性函数f,h在工作点附近进行一阶Taylor展开

EKF优缺点：

![image-20241010104701870](markdown-img/SLAM十四讲.assets/image-20241010104701870.png)

## BA与图优化

![image-20241010105612666](markdown-img/SLAM十四讲.assets/image-20241010105612666.png)

![image-20241010110412624](markdown-img/SLAM十四讲.assets/image-20241010110412624.png)

因为不存在相机与相机/路标与路标之间的关联所以，H有一定的特殊结构

![image-20241010110341679](markdown-img/SLAM十四讲.assets/image-20241010110341679.png)

镐子形矩阵

![image-20241010110824243](markdown-img/SLAM十四讲.assets/image-20241010110824243.png)

对这个稀疏结构进行加速求解，以下为加速求解：

![image-20241010111042401](markdown-img/SLAM十四讲.assets/image-20241010111042401.png)

> C为约当块矩阵，求逆简单

![image-20241010111224803](markdown-img/SLAM十四讲.assets/image-20241010111224803.png)

以上做法称为Schur消元

![image-20241010111709877](markdown-img/SLAM十四讲.assets/image-20241010111709877.png)

## 实践

G2O/ceres的BA

# References

- 【【视觉SLAM十四讲】全书讲解！】<https://www.bilibili.com/video/BV1ZC4y1J7EZ?vd_source=93bb338120537438ee9180881deab9c1>
