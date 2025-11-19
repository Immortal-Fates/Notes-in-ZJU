# 非线性规划

[TOC]

# 非线性规划问题

## Concepts

> 介绍一些概念

- 强制函数：当$||x||\to \infty,f(x)\to \infty$
- 下水平集合：对$\forall ~constant ~\alpha ,L_{\alpha} = \{x\in R^n, f(x\le \alpha \}$
- 集合紧等价于闭且有界
- 强制函数等价于所有下水平集紧

## Model

- 标准模型
  $$
  \min\ f(x)\ \ \text{s.t.}\ \ g_j(x)\le 0,\ \ h_i(x)=0
  其中 $$x\in\mathbb{R}^n,\ f,g_j,h_i$$ 连续可微
  $$

- 最优解存在性的直觉
  连续目标在“**非空且有界闭**”的可行域上常能取得极小（Weierstrass 思路）；无约束时若 $$f(x)\to +\infty\ ( \|x\|\to\infty )$$（强制/coercive），也常存在极小。

# 最优性条件

## 无约束优化：一/二阶条件

- **一阶必要条件（驻点）**
  局部极小 $$x^*$$ 必满足 $$\nabla f(x^*)=0$$。

- **二阶必要/充分条件**
  设 Hessian $$\nabla^2 f(x^*)$$：
  必要：$$\nabla f(x^*)=0,\ \nabla^2 f(x^*)\succeq 0$$；
  充分：$$\nabla f(x^*)=0,\ \nabla^2 f(x^*)\succ 0\Rightarrow x^*$$ 为严格局部极小。
  矩阵正定性可用特征值、主子式或 Cholesky 判定。

- **几何直觉**
  沿任意方向 $$p$$ 的二次近似 $$f(x^*+p)\approx f(x^*)+\tfrac12 p^\top \nabla^2 f(x^*) p$$；正定意味着所有方向上“碗口向上”。

---

## 等式约束：拉格朗日乘子法

- **拉格朗日函数**
  $$
  L(x,\nu)=f(x)+\nu^\top h(x)
  $$

- **一阶条件**
  可行点 $$x^*$$ 若为局部极小，常伴随乘子 $$\nu^*$$ 使$\nabla_x L(x^*,\nu^*)=0,\ \ h(x^*)=0$。

## 不等式约束与可行方向

- **活跃约束与可行方向**
  在边界点，活跃集 $\mathcal{A}(x)=\{j\mid g_j(x)=0\}$$。
  向量 $$d$ 为线性化可行方向需满足
  $\nabla h_i(x)^\top d=0,\ \ \nabla g_j(x)^\top d\le 0\ (j\in\mathcal{A}(x))$。
  若不存在“可行的下降方向”（使 $$\nabla f(x)^\top d<0$$），则是局部极小的直观必要性。

---

## FJ 与 KKT：一阶最优性条件的统一框架

- **Fritz John（FJ）必要条件**
  对局部极小 $$x^*$$，存在不全为零的乘子$\mu_0\ge 0,\ \mu_j\ge 0,\ \nu_i\in\mathbb{R}$，使
  $$
  \mu_0\nabla f(x^*)+\sum_j \mu_j \nabla g_j(x^*)+\sum_i \nu_i \nabla h_i(x^*)=0,
  $$

  $$
  g_j(x^*)\le 0,\ \ \mu_j g_j(x^*)=0,\ \ h_i(x^*)=0
  $$

  其中 $$\mu_0=0$$ 的“退化”情形提示约束奇异或资格条件不足。

- **Karush–Kuhn–Tucker（KKT）条件**
  若再满足合适的**约束资格（CQ）**，可令 $$\mu_0=1$$ 归一化，得到
  $$
  \nabla f(x^\*)+\sum_j \lambda_j^\* \nabla g_j(x^\*)+\sum_i \nu_i^\* \nabla h_i(x^\*)=0, \\
  g_j(x^\*)\le 0,\ \ \lambda_j^\*\ge 0,\ \ \lambda_j^\* g_j(x^\*)=0,\ \ h_i(x^\*)=0
  $$
  ——这是最常用的一阶**必要条件**。

## 约束资格（CQ）：KKT 必要性的保障

- **常见 CQ**
  - **LICQ**（线性无关）：活跃约束的梯度向量线性无关。
  - **MFCQ**：存在方向 $$d$$ 使 $$\nabla h_i^\top d=0$$ 且活跃不等式满足 $$\nabla g_j^\top d<0$$。
  - **Slater 条件**（凸问题）：存在严格可行点 $$g_j(x)<0,\ h_i(x)=0$$。
- CQ 的作用是：
  - 排除“奇异点”
  - 保证拉格朗日乘子存在且有意义
  - 保证可行方向锥的结构良好
  - 保证 KKT 为最优性的必要条件

## 二阶最优性条件（含约束）

- **临界锥 $$C(x^*,\lambda^*,\nu^*)$$**
  由线性化可行方向且满足驻点互补关系的“可疑方向”构成，是二阶分析的舞台。

- **二阶必要条件**
  对任意 $$p\in C(x^*,\lambda^*,\nu^*)$$，有
  $$
  p^\top \nabla_{xx}^2 L(x^*,\lambda^*,\nu^*)\, p\ \ge\ 0
  $$

- **二阶充分条件**
  若对所有非零 $p\in C(x^*,\lambda^*,\nu^*)$ 有
  $$
  p^\top \nabla_{xx}^2 L(x^*,\lambda^*,\nu^*)\, p\ >\ 0
  $$
  则 $x^*$ 为严格局部极小。
  直觉：在一阶“平衡”后，拉氏 Hessian 在可行“自由度”方向上要“向上弯”。

# 对偶理论
