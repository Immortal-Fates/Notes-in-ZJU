**数学没有物理是瞎子，物理没有数学是跛子**

[高等数学导数公式、微分公式和积分公式大全_常用微分公式 知乎-CSDN博客](<https://blog.csdn.net/weixin_43148062/article/details/106302533#:~:text=总结了> 高等数学 微 积分常用公式 ，适合快速查阅，本资源为word版本，方便进一编辑排版： 一、基本 导数公式 二、,6.万能 公式 7.平方关系 8.倒数关系 9.商数关系 十五、几种常见的 微分 方程)

## 数列极限

[数学分析笔记（第三章）：数列极限 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/570450925)

## 级数

### 数项级数的敛散性判别

[数学分析笔记（第四章）：级数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/570587740)

- 积分判别法，

  <img src="https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20230621174029369.png" alt="image-20230621174029369" style="zoom:50%;" />

- 比较判别法（极限形式，看阶）

- 比值判别法，根值判别法算出来的$\rho$一样（先用根值，一般根值比较强）+if $\rho=1则 \space lima_n=0$

- 斯特林公式：$n!≈\sqrt{2\pi n}(\frac{n}{e})^n$,n很大时可以用这样来估计大小

- stolz定理：[(1 封私信 / 8 条消息) Stolz定理是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/432246847)

- 对数判别法：
  $$
  lim\frac{ln\frac{1}{u_n}}{lnn}=q,q>1收敛，q<1发散,a_n=\frac{1}{(f(n))^{g(n)}}可用
  $$

### 幂级数及其和函数

- 基本概念：[级数知识点小结2-幂级数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/291823132)
- 一些题型：[考研数学三级数~求幂级数的收敛域、和函数、幂级数展开式问题总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/181727876)

- **注意**：可能存在收敛域改变的情况

  ![img](https://picx.zhimg.com/v2-3717fbeca8f3992a255ec64e57f0d986_r.jpg?source=1940ef5c)

- 武器库（其他级数全部裂开再套进去就行了）

  <img src="https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20230621200440569.png" alt="image-20230621200440569" style="zoom:50%;" />

- 构造微分方程

- 注意分母为0

### 函数的傅里叶展开

- 不难，背公式带入就行

$$
f(x)=\frac{a_0}{2}+\sum_{n=1}^{\infty}({a_ncos\frac{n\pi x}{l}}+{b_nsin\frac{n\pi x}{l}}),T=2l
$$

$$
a_n=\frac{1}{l}\int_{-l}^{l}f(x)cos\frac{n\pi x}{l}dx(n=0,1,2 ...)
$$

$$
b_n=\frac{1}{l}\int_{-l}^{l}f(x)sin\frac{n\pi x}{l}dx(n=0,1,2 ...)
$$

- 迪利克雷定理：函数在断点的傅里叶级数的值为两端点函数值的平均值
- 要求余弦级数则偶延拓，正弦级数则奇延拓

## 含参量积分（不会）

- 欧拉积分：
  $$
  第二类欧拉积分\Gamma(s)=\int_{0}^{\infty}t^{s-1}e^{-t}dt
  $$

  $$
  \Gamma(\frac{1}{2})=\pi^\frac{1}{2}
  $$

  $$
  \Gamma(z+1)=z\Gamma(z)
  $$

## 矢量代数与空间解析几何

- 基本概念：[高数(上)笔记——向量代数与空间解析几何 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/336699646)
- 一些题型：[高数(上)笔记——空间解析几何相关习题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/343252758)

### 数量积

$$
a\cdot b=a_1b_1+a_2b_2+a_3b_3
$$

### 矢量积

$$
\vec{a}×\vec{b}
$$

叉乘交换顺序会改变符号

叉乘，点乘均可以打开括号

### 混合积

$$
(\vec{a}*\vec{b})\cdot \vec{c}
$$

混合积为0则向量共面

3个的顺序可以随便交换

几何意义：以a,b,c为棱的平行六面体的体积

### 一些证明

- 证明两直线异面：$(v_1×v_2)\cdot P_1P_2!=0$,异面直线的距离$d=\frac{|(v_1×v_2)\cdot P_1P_2|}{|v_1×v_2|}$
- 求直线在平面上的投影：用平面束方程把投影直线表示为两平面的交线

## 多元函数微分学

- 牛逼：[考研高数笔记V1.0——第九章 多元函数微分法及其应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/101694412)

- 两个偏导数连续=>可微=>函数在该点连续

  ​                                       =>函数在该点偏导数存在

- 方向导数
  $$
  方向导数的定义：\frac{\partial u}{\partial l}|_{P_0}=\lim_{\rho->0}\frac{u(P)-u(P_0)}{\rho}
  $$

- 点集E的全体边界点构成E的边界记作$\partial E$

### 多元函数的极值

- 极值的必要条件：在点$P(x_0,y_0)$的一阶偏导均为0

- 极值的充分条件：
  $$
  if \space f\prime_x(x_0,y_0)=0,f\prime_y(x_0,y_0)=0
  $$

  $$
  A=f\prime\prime_{xx}(x_0,y_0)=0,B=f\prime\prime_{xy}(x_0,y_0)=0,C=f\prime\prime_{yy}(x_0,y_0)=0
  $$

  $$
  B^2-AC<0,f(x_0,y_0)一定为极值，A>0极小值，A<0极大值
  $$

  $$
  B^2-AC>0,f(x_0,y_0)一定不为极值
  $$

  $$
  B^2-AC=0,f(x_0,y_0)该方法失效
  $$

- 条件极值：拉格朗日乘数法

### 偏导数在几何上的应用

- 空间曲线的切线与法平面
  $$
  切矢量和法平面的法矢量均为:(x\prime(t_0),y\prime(t_0),z\prime(t_0))
  $$

- 空间曲面的切平面与法线
  $$
  法矢量：n=(F\prime_x,F\prime_y,F\prime_z)
  $$

### 多元函数的一些技巧

- [多元函数的泰勒展开式 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/33316479)
- 求函数区域D内的极值，最大最小值，先找驻点：一阶偏导均=0，再看边界点

## 重积分

### 二重积分

- 先画图像，分匀和精
- 直角坐标系下：X，Y型区域，不规则则分块
- 极坐标系下：$$先\rho后\theta$$(常用)，角型区域均可用极坐标

$$
\iint f(r,\theta)rd\theta dr
$$

- 对称性化简!

- 积分换序

- 换元法（经典变换$$xy=u,\frac{y}{x}=v$$）
  $$
  x=x(u,v),y=y(u,v)，变量换元，乘上二阶的雅可比矩阵即可（x,y对于u,v的，别搞反了）
  $$

### 三重积分

- 先一后二(投影法)，先二后一(平面截割补法)，画出积分区域很重要！

- 柱面坐标变换（本质x,y的极坐标变换）：一般都先求出$$\sigma_{xy}$$，确定上曲面和下曲面，再坐标变换

- 球面坐标变换（椭球面的坐标变换相似）：
  $$
  x=\rho sin\varphi cos\theta,y=\rho sin\varphi sin\theta,z=\rho cos\varphi,0\leq\varphi\leq\pi
  $$

  $$
  V=\iiint{f\rho^2sin\varphi}d\rho d\varphi d\theta
  $$

- 对称性化简!：奇偶性，轮换对称性

## 曲线曲面积分

### 第一型曲线积分

- 直接化为参数方程，带公式即可(注意是平面曲线，还是空间曲线)

$$
\int_L{f(x,y)}ds=\int{f}\cdot{\sqrt {x^{\prime2}+y^{\prime2}}}dt
$$

$$
\int_L{f(x,y)}ds=\int{f}\cdot{\sqrt {r^2+r^{\prime2}}}d\theta
$$

- 利用轮换对称性

### 第二型曲线积分

- 和第一型的一样，也就是直接代入，$dx=x\prime(t)dt$

$$
\int_L{(P(x,y)}cos\alpha+{Q(x,y)}cos\beta)ds=\int_L{P(x,y)}dx+{Q(x,y)}dy=\int_{\alpha}^{\beta}{P}x\prime (t)+Qy\prime(t) dt
$$

- 有方向!

- 找找是否有原函数，直接代入就很简单

- 积分与路径无关的判定与应用（这是平面，空间同理）

  ![image-20230612193757867](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20230612193757867.png)

- **Green公式**：第二型曲线积分与二重积分的互换(证明需要掌握：将二重积分分段写为累次积分)
  $$
  \iint_D(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y})dxdy=\oint Pdx+Qdy
  $$

  - 利用Green公式求面积：$S=\frac{1}{2}\oint_\Gamma-ydx+xdy$

### 第一型曲面积分

- 直接化为参数方程，带公式即可

$$
\iint_S{f(x,y,z)}dS=\iint_{\sigma_{xy}}{f}\cdot{\sqrt {1+z_x^{\prime2}+z_y^{\prime2}}}dxdy
$$

- 灵活利用对称性（即使对称点不在原点）

### 第二型曲面积分

- 投影变化如下：

$$
\iint_SR(x,y,z)cos\gamma dS=\iint_SR(x,y,z)dxdy=\pm\iint_DR(x,y,z(x,y))dxdy
$$

$$
\pm由S的方向向量与z轴正向的夹角来确定:dxdy=sgn(\frac{\pi}{2}-\gamma)\mid cos\gamma \mid dS=sgn(\frac{\pi}{2}-\gamma)d\sigma
$$

- **Gauss公式**：第二曲面积分与三重积分的互换

$$
\iiint_V(\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z})dxdydz=\oiint_SPdxdy+Qdzdx+Rdxdy，note:这dxdy可能为正、负、零，与二重积分的dxdy恒正不同
$$

$$
\iiint_VdivAdV=\oiint_SA\cdot dS
$$

- 散度场：

$$
divA=\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}
$$

- 可以先化为全部$dS$再用将$dS$投影到一个面上，化简计算
- 投影角度的计算：曲面上一点的法矢量与坐标轴的夹角（dydz则求对x轴的夹角），所以曲面的上侧还是下侧就会影响夹角的正负

### 斯托克斯公式

Green公式的推广

- 第二型曲线积分与第二型曲面积分的互换

$$
\iint_S(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z})dydz+(\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x})dydz+(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial Y})dydz=\oint_LPdx+Qdy+Rdz
$$

$$
dS=e_Tds称为弧长元素向量，e_n为法线,\iint_SrotA\cdot dS=\iint_SrotA\cdot e_ndS=\oint_LA\cdot ds=\oint_LA\cdot e_Tds
$$

- 旋度场

$$
A(x,y,z)=(P(x,y,z)，Q(x,y,z)，R(x,y,z))
$$

$$
rotA=\left[
\matrix{
  i & j & k\\
  \frac{\partial}{\partial x} & \frac{\partial}{\partial y} &\frac{\partial}{\partial z}\\
  P & Q &  R
}\right]，旋度和坐标轴的选取无关
$$

$$
(rotA\cdot e_n)_{M_0}=\lim_{D->M_0}\frac{\oint_LA\cdot ds}{D}
$$

### 点函数的物理应用

- 质心
  $$
  average(x)=\frac{\int_\sigma\mu(P)xd\sigma}{M}
  $$

  $$
  \mu为常数，则average(x)=\frac{\int_\sigma xd\sigma}{\sigma}
  $$

- 转动惯量
  $$
  当L为Oz轴时，I_z=\int_\sigma(x^2+y^2)\mu(P)d\sigma
  $$

- 引力
  $$
  F_x=Gm\int_\sigma\frac{\mu(P)(x-x_0)}{r^3}d\sigma
  $$

- $$!!!dS=\frac{1}{cos\gamma}d\sigma$$

### 梯度，旋度，散度

- 理解：[(14 封私信 / 80 条消息) 散度和旋度的物理意义是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/21912411)
- 计算：[基础篇2: 梯度、散度与旋度 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/136836187)
- 哈密顿算子$\nabla$,$\nabla\cdot A=divA;\nabla×A=rotA,\nabla_u=gradu$

> **数学没有物理是瞎子，物理没有数学是跛子**

## 不足

- 一些重要级数的展开
- 求解偏微分方程

## 一些总结+拓展

- $arcsinx$的范围是从$[-\frac{\pi}{2},\frac{\pi}{2}]$
- 傅里叶级数的计算，大胆分部积分
- 求函数的连续性遇到上下次数不一致，可以令$y=x^n$使次数一致
- 难以说明则用反证法，求证的式子中有min,max一般很难入手，从另一侧入手
- 级数条件是等式试试构造递推，是$lim$极限形式则一般只能用定义(有时可能用保号性)，要有目标！
- 想要建立$f与f\prime，f\prime\prime$的的关系，只有积分中值定理+泰勒展开式（多元函数的泰勒展开也需要掌握）
- 函数的连续性，最小值最大值导数的判断（想象一下曲面图）；区域连续必能够取到极大极小值（偏导均为0！对0敏感一点）

- $|\int_cPdx+Qdy|<=LM,L为积分路径长度，M=max\sqrt{P^2+Q^2}$

- 单位外法矢量一般写为：$n=${$cos\alpha+cos\beta+cos\eta$}

- 比值判别法，根值判别法（先用根值，一般根值比较强）+$lima_n=0$

- 一些常见的积分!
  $$
  lim\frac{\sqrt[n]{n!}}{n}=\frac{1}{e},斯特林公式or取对数
  $$
  $1^p+2^P......+n^P$~$\frac{n^{P+1}}{P+1}$

- 见到根号差就用有理化

- 角形区域——极坐标

- 没有积分与路径无关时进行拆项，

  构造出积分与路径无关的情况简化计算
