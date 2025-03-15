# ODE

## 线性微分方程

1. 先算齐次方程的通解
2. 再算非齐次方程的特解

### 一阶

- 

- 左右同时乘上积分因子乘上integrating factor积分因子
  $$
  \frac{dy}{dt}+a(t)y=b(t)
  $$

  $$
  \mu(t)\frac{dy}{dt}+\mu(t)a(t)y=\mu(t)b(t)
  $$

  $$
  so:I\space want\space \frac{d\mu(t)y}{dt}=\mu(t)\frac{dy}{dt}+\mu(t)a(t)y=>\frac{d\mu}{dt}=a(t)\mu(t)
  $$

  $$
  then\space we\space get:\frac{d\mu(t)y}{dt}=\mu(t)b(t)
  $$

- 一阶的齐次性/非齐次性的判断

A diferential equation $y\prime=f(t,y)$ is homogeneous if 
$$
f(kt,ky)=f(t,y)
$$
for every real number k

- Exact equations全微分方程，即第二类曲线积分

  if the equation is not exact ,use the integrating factor

$$
M\frac{\partial\mu}{\partial y}+\mu\frac{\partial M}{\partial y}=N\frac{\partial\mu}{\partial t}+\mu\frac{\partial N}{\partial t}
$$

$$
I\space want\space \mu\space is \space of\space t \space or\space y\space alone
$$

### 二阶

- principle of superposition：y1,y2的朗斯基方程！=0则y1,y2是二阶方程的通解

- 知一个解求另一个解
  $$
  y_2(t)=v(t)y_1(t)
  $$

  $$
  \mu(t)=\frac{c\space exp(-\int p(t)dt)}{y^2_1(t)}=\frac{dv}{dt}
  $$

- 

## 常系数线性微分方程

![e52dae461bf6aa7023ff0d85222094de](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/e52dae461bf6aa7023ff0d85222094de.jpg)

$$
y^*=u_1(t)y_1(t)+u_2(t)y_2(t)
$$

### the method of judicious guessing

一些特殊的情况：（sin,cos可以堪称e的虚部和实部）
$$
方程左边=P_n(t)e^{\lambda t}
$$

$$
Q\prime\prime(t)+(2\lambda+p)Q\prime(t)+(\lambda^2+p\lambda+q)Q(t)=P_n(t),
$$

$$
令Q（t）是与P_n同阶的多项式
$$

$$
\lambda^2+p\lambda+q=0,是特征方程的根，则Q（t）需要×t,
$$

$$
在此基础上，2\lambda+p=0（即上式的导数仍未0)，还要再×t
$$



## 常微分方程的常数变易法

https://zhuanlan.zhihu.com/p/387631989

## 常系数齐次线性方程组

### 欧拉方程（the degree of x is equal to the degree of the prime of x）

$$
let\space x=e^t(if\space x<0,x=-e^t),y对x的一阶导=\frac{1}{x}D（D为y对t的导）
$$

$$
y对x的二阶导=\frac{1}{x^2}(D^2-D)
$$

$$
y对x的一阶导=\frac{1}{x^3}(D^3-3D^2+2D)
$$



### 特征根为单根

含复数根（成对），将$$e^{a+bi}$$展开，然后将特征向量写为实数+复数的形式再相乘，即得实数部分为一个解，复数部分为一个解（不含i）
$$
e^{at}(pcos\beta t-qsin\beta t),e^{at}(psin\beta t+qcos\beta t)
$$


### 特征根为重根

#### 空间分解法

![image-20230609185332206](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20230609185332206.png)

Notes:先求广义特征向量

#### 待定系数法

$$
A_{2*2}  有2个线性无关的解则和对角阵相似，仅有一个则与约当标准型相似
$$

#### 我的结论

在有重根的方程中找到x，1次特征方程不为0，但二次为0（两重）

多重则依次找，前面低次不为0，高次为0得$$v_2$$(括号内为泰勒展开，依次找就行)
$$
x^{(1)}=e^{\lambda}v_1,
x^{(2)}=e^{At}v_2=e^{\nu\lambda}[I+t(A-\lambda I)]\nu_2
$$

![QQ图片20230617105516](E:\notes\课程\常微分方程\常微分复习\QQ图片20230617105516.jpg)

![QQ图片20230617105541](E:\notes\课程\常微分方程\常微分复习\QQ图片20230617105541.jpg)



## 常系数非齐次线性方程组

#### 常数变易法：

$$
let\space x(t)=X(t)u(t),X\prime(t)=AX(t)
$$

$$
so\space u\prime(t)=X^{-1}(t)f(t)
$$

$$
e^{At}=X(t)X^{-1}(0)
$$

![image-20230609213002498](C:\Users\杨逍宇\AppData\Roaming\Typora\typora-user-images\image-20230609213002498.png)
$$
当给定初值条件x(t_0)=x_0时，c=X^{-1}(t_0)x_0代入上式
$$

- 当n=2时用消参法
- 当n>=3时用常数变易法（直接写然后算就行了）

![image-20230612222710615](https://raw.githubusercontent.com/Immortal-Fates/image_host/main/blog/image-20230612222710615.png)

然后对$u\prime$积个分就行了

#### 带入消参法Method of elimiantion

## some tricks(aim to make it standard)

- change of variable

- reverse the fraction

- 欧拉公式
  $$
  e^{i\theta}=cos\theta +isin\theta
  $$

- 已经学过的方法不知道怎么做的问题，一般只能通过观察方程结构来尝试变量替换并求解（一阶微分方程就是各个方法一个个试）
- 当全微分甚至连配凑都不能变成与路径无关时，直接同除dx or dy（$\frac{dx}{dy}or\frac{dy}{dx}$也是一种trick）
- 一边有根号则把另一边变成一个整体（换元消参），然后就可以平方了

## some examples

![lQDPJxf1evkOWC7NBQDNA8CwEzduNUFOO2cEfjHS-EDBAA_960_1280](C:\Users\Immortal\AppData\Roaming\DingTalk\64689221_v2\ImageFiles\f4\lQDPJxf1evkOWC7NBQDNA8CwEzduNUFOO2cEfjHS-EDBAA_960_1280.jpg)

- [高数笔记-常系数非齐次微分方程的解法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/64558412)

## some words

- dependent varible 因变量
- substitution 替换
- denote 标志，表示
- lemma 引理
