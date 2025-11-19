# Main Takeaway

本意做自动兑矿，尝试控制Mobile Manipulator

<!--more-->

# Mathmatics

设 $\Phi_d = \{\eta_d, \epsilon_d\}$，$\Phi_e = \{\eta_e, \epsilon_e\}$ 为单位四元数，分别表示期望的末端执行器方向和测量的方向。如果期望的末端执行器框架和测量的框架重合，则必须满足方程 $\Delta \Phi = \Phi_d \star \Phi_e^{-1} = \{1, 0\}$，其中符号 $\star$ 表示四元数乘法运算。这给出了方向误差的以下形式：

$$
e_o = \eta_e \epsilon_d - \eta_d \epsilon_e - [\epsilon_d]^\times \epsilon_e
$$

> Tips: 这里$[\cdot]^\times $

# References

- [manuscript.pdf (ethz.ch)](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/353053/manuscript.pdf?sequence=1&isAllowed=y)
