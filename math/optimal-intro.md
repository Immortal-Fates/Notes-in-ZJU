# Main Takeaway

研究生课，何衍《优化方法及应用》

<!--more-->

# 考评方法

期末成绩=50%大作业成绩+30%期末课内测试成绩+20%平时作业成绩说明：

1. 大作业：单独完成，按质量和规范性打分。
2. 期末课内测试：30分钟开卷测试（第八周），具体考察：
   - I.对基本概念、分析方法的掌握程度；
   - II.对基础理论、典型算法的理解程度。
3. 奖励分：课堂发言、课后讨论、教学反馈、大作业分享。

# 应用软件

- **直接求解器调用**
  Matlab Optimization Toolbox、IBM CPLEX…

- **优化建模语言编程**
  1) **CVX**: matlab\Julia\Python
     - 免费求解器：SDPT3、SeDuMi
     - 商业求解器：MOSEK（稀疏问题）、Gurobi（MIP）

  2) **YALMIP** (Yet Another LMI Matlab Toolbox & Interior-Point algorithm)：
     支持多数求解器，在电网、机器学习等领域应用广泛。

  3) **AMPL**（A Mathematical Programming Language）：
     大规模优化问题，支持大多数求解器。

- **代码生成器**
  **CVXGEN**：c代码生成器，4核Xeon仅用0.4ms求解110个变量、111个约束的组合投资问题（SDPT3需350ms）。
