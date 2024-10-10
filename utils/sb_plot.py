import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成样本数据
np.random.seed(0)
x = np.random.normal(50, 5, 1000)
y = np.random.normal(55, 7, 1000)

# 创建带有核密度估计和直方图的 jointplot
g = sns.jointplot(x=x, y=y, kind="kde", fill=True, cmap="coolwarm", levels=15, thresh=0)

# 添加边际直方图
g.plot_marginals(sns.histplot, kde=True, color="coral", alpha=0.6)

# 显示图形
plt.show()
