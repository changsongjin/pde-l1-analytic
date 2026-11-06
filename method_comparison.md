# PDE 反问题方法对照记录

## PDE-FIND

PDE-FIND 的反问题是：给定 (u(x,t)) 的时空观测，识别

\[
u_t=N(u,u_x,u_{xx},\ldots,x,\mu).
\]

原论文把候选项放进 (Theta(U,Q))，再用稀疏回归选择非零项；多条不同初值轨迹用于消除行波数据造成的等价方程。标准实现是 STRidge，即岭回归加硬阈值，不是 L1。

来源：[Rudy et al., Data-driven discovery of partial differential equations](https://pmc.ncbi.nlm.nih.gov/articles/PMC5406137/)。

## DLGA-PDE

DLGA-PDE 仍然解决“数据到控制 PDE”的反问题，但先训练神经网络拟合 (u(x,t))，再用自动微分生成导数，遗传算法搜索候选项的乘积和。它允许初始候选库不完整，目标函数包含拟合误差和项长度的 (ell_0) 惩罚。原论文测试了 KdV、Burgers、波动和 Chafee–Infante，并报告了有限数据和噪声实验。

来源：[Xu, Chang & Zhang, DLGA-PDE](https://www.sciencedirect.com/science/article/pii/S0021999120303582)。本项目使用 [KD 的开源重实现](https://github.com/Scientific-Artificial-Intelligence-Lab/kd)，并在 JSON 中记录 surrogate、种群和代数预算。

## SGA-PDE

SGA-PDE 把每个 PDE 项编码为符号树，把整条方程表示成森林，再用交叉、变异和替换搜索开放形式。它可以生成预定义库之外的复合函数和分式项；原始代码默认使用有限差分，当前候选森林中的系数组合还调用 STRidge。

来源：[Chen et al., Symbolic genetic algorithm for discovering open-form PDEs](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.023174)；[原始代码](https://github.com/YuntianChen/SGA-PDE)。

## 本项目的对照方法

本项目的 GML-L1 使用弱形式、语法闭合的 113 列微分多项式库和 KKT 列生成。固定权重时求解精确 L1，IRL1 只改变 L1 权重；PDE-FIND/STRidge 作为非 L1 对照；SGA-PDE 使用已有独立进程结果；DLGA-PDE 使用 KD 的神经 surrogate + 遗传搜索。

对比时必须记录数据处理链：GML-L1 和 PDE-FIND 对照使用同一个弱形式矩阵；SGA 使用其原生有限差分；DLGA 使用神经网络自动微分。因此只比较“支撑随噪声的曲线”时可以并列，直接比较运行时间时必须保留各自预算和预处理时间。
