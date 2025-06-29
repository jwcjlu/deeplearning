import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 修复负号显示为方块的问题
# 定义 x 轴范围
x = np.linspace(-4, 4, 500)

# 计算 tanh(x) 及其导数
tanh_x = np.tanh(x)
tanh_derivative = 1 - np.tanh(x)**2

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制 tanh(x)
plt.plot(x, tanh_x, label='tanh(x)', color='blue', linewidth=2)

# 绘制 tanh 的导数
plt.plot(x, tanh_derivative, label='dy/dx', color='red', linestyle='--', linewidth=2)

# 添加标题和标签
plt.title('tanh(x)及导数', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例和水平/垂直线
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.legend(fontsize=12)

# 显示图形
plt.show()