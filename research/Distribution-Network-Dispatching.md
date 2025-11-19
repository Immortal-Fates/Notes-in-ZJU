# Distribution Network Dispatching

- **An Auxiliary Decision Model for Distribution Network Dispatching Based on Scientific Computing**. Li Qiang et.al. **No journal**, **2024-7-27**,  ([pdf](..\..\papers\Dispatching\An_Auxiliary_Decision_Model_for_Distribution_Network_Dispatching_Based_on_Scientific_Computing.pdf))([link](https://doi.org/10.1109/icnc-fskd64080.2024.10702264)).

  - 双层优化架构：
    - **上层模型**：以220kV输电网络为对象，目标为**最小化输电成本**（涵盖发电成本、线损成本、输电线路扩建成本），约束包括功率平衡、安全运行限值等；
    - **下层模型**：以110kV高压配电网为对象，目标为**最小化负荷削减成本**（保障停电损失最小），约束涵盖配电网辐射性结构、安全运行规则等。
  - **求解方法**：采用**遗传算法**实现双层模型交互迭代——上层为下层提供输电功率约束，下层反馈负荷转供与储能调度结果，通过多轮迭代逼近全局最优解，提升负荷转供策略的**计算效率与最优性**。
  - LLaMA - CRF的意图理解

- **Metaheuristic search in smart grid: A review with emphasis on planning, scheduling and power flow optimization applications**. Papadimitrakis M. et.al. **Renewable and Sustainable Energy Reviews**, **2021-7**, ([pdf](..\..\papers\Dispatching\Metaheuristic_search.pdf))([link](https://doi.org/10.1016/j.rser.2021.111072)).

  - review on 元启发式算法（如粒子群PSO、差分进化DE、遗传算法GA）

    > 感觉挺复杂的

- **Research and Application of Knowledge Graph in Distribution Network Dispatching and Control Aided Decision Making**. Wang Jundong et.al. **No journal**, **2021-7-2**, ([pdf](..\..\papers\Dispatching\Research_and_Application_of_Knowledge_Graph_in_Distribution_Network_Dispatching_and_Control_Aided_Decision_Making.pdf))([link](https://doi.org/10.1109/bdai52447.2021.9515217)).**

- **Research and Application of Power Grid Fault Diagnosis and Auxiliary Decision-making System Based on Artificial Intelligence Technology**. Qiu Chenguang et.al. **No journal**, **2022-9-23**, ([link](https://doi.org/10.1109/icpre55555.2022.9960335)).

- **Enhancement of Power Equipment Management Using Knowledge Graph**. Tang Yachen et.al. **No journal**, **2019-5**, ([pdf](..\..\papers\Dispatching\Enhancement_of_Power_Equipment_Management_Using_Knowledge_Graph.pdf))([link](https://doi.org/10.1109/isgt-asia.2019.8881348)).

- **Optimal operation of smart distribution networks: A review of models, methods and future research**. Evangelopoulos Vasileios A. et.al. **Electric Power Systems Research**, **2016-11**, ([pdf](..\..\papers\Dispatching\Optimal_operation.pdf))([link](https://doi.org/10.1016/j.epsr.2016.06.035)).

  - 数值方法
  - 启发式方法
  - 随机方法

# Cmd

```
set PYTHONUTF8=1
autoliter -i ./Distribution_Network_Dispatching.md -o ../../papers/Dispatching
```
