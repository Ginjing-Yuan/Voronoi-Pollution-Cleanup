#  EcoAgents: Multi-Agent Pollution Neutralization Simulation

本项目基于 Python 和 NumPy 深度复现了论文 **《A New Coordinated Control Strategy of Multi-Agent System for Pollution Neutralization》** 的核心控制算法。

项目构建了一个高保真的物理仿真环境，模拟了多移动机器人（智能体）在风场对流、浓度扩散与自然衰减的复杂环境下，协同清理区域污染物的全过程。

##  核心亮点 (Key Features)

- **📐 Voronoi 动态空间划分**：实时计算并可视化智能体的“领地”边界（泰森多边形），完美解决多机协同中的任务冲突与重复劳动问题。
- **🎯 新型广义质心目标函数**：创新性引入 **“初始浓度记忆”机制**。将实时浓度场与初始污染快照结合，引导智能体“直捣黄龙”锁定污染源头，避免被风吹散的扩散尾迹带偏。
- **🤝 全局协同与跨区支援**：彻底打破传统算法中“各扫门前雪”的局部最优陷阱。当智能体清理完自身区域后，会自动跨越边界，前往全局污染最严重的区域进行支援。
- **🎨 极具视觉冲击力的动态仿真**：包含风场动态变化、污染物晕染扩散模型，提供平滑的热力图、动态演化的细胞边界以及智能体清理轨迹的实时动画。

## ️ 技术栈 (Tech Stack)

- **Python 3.x**
- **NumPy** (用于高效的矩阵运算与向量化 Voronoi 划分)
- **Matplotlib** (用于动态动画渲染与数据可视化)
- **Pillow** (用于 GIF 动画导出)

## 🚀 快速运行

```bash
# 安装依赖
pip install numpy matplotlib pillow

# 运行仿真
python pollution_monitoring.py

```
## 🎬 演示效果

下图展示了多智能体在风场和扩散作用下的协同清理过程。可以看到 Voronoi 边界（白色虚线）随智能体移动而动态变形，智能体根据广义质心直捣污染源头。


<p align="center">
  <img src="pollution_diffusion_2.gif" alt="污染清理仿真预览" width="800">
</p>
