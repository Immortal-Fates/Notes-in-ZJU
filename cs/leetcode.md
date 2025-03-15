# Main Takeaway

刷leetcode心得

<!--more-->

# 动态规划DP

[超详细！动态规划详解分析（典型例题分析和对比，附源码）_动态规划案例解析-CSDN博客](https://blog.csdn.net/qq_44398094/article/details/111318003)

动态规划算法需要存储各子问题的解——以空间换时间



# 常见数据类型

- list：
- deque：双向队列[C++中deque的用法（超详细，入门必看）_c++ deque-CSDN博客](https://blog.csdn.net/H1727548/article/details/130959610)

- priority_queue：优先队列——在队列中添加了内部排序[c++优先队列(priority_queue)用法详解](https://www.cnblogs.com/huashanqingzhu/p/11040390.html)



# 常见套路

- 使用`unorder_map`创建一个空的哈希表，用于记录每个元素出现的次数

  ```
  unordered_map<int, int> m;
  m[nums[i]]++;
  ```

- 原地哈希：将数组视为哈希表/哈希表其实本身也是一个数组

  相当于自己将数组按照一定的哈希规则排序

- 原地算法（in place）

- 矩阵旋转90°：先转置，再水平翻转

  