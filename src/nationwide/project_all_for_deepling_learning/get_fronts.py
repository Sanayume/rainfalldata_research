import matplotlib.pyplot as plt

# --- 字体设置 开始 ---
# 在 Windows 上，优先尝试 SimHei 或 Microsoft YaHei
plt.rcParams['font.sans-serif'] = ['SimHei'] # 尝试使用黑体
# 或者，如果更喜欢微软雅黑的显示效果，可以尝试：
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
print("已尝试设置中文字体 (SimHei 或 Microsoft YaHei)。")
# --- 字体设置 结束 ---

# --- 你的绘图代码 ---
# 例如:
plt.figure()
plt.title("中文标题测试")
plt.xlabel("X轴 (单位)")
plt.ylabel("Y轴 (比率)")
plt.plot([1,2,3],[1,4,9])
plt.text(1.5, 5, "测试点 -5")
plt.show()
# --- 绘图代码结束 ---