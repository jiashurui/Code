import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# 生成一些模拟的时间序列数据
np.random.seed(0)
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
sales = 100 + 10 * np.sin(2 * np.pi * date_rng.month / 12) + np.random.normal(0, 5, len(date_rng))
time_series = pd.Series(sales, index=date_rng)

# 使用 STL 进行分解
stl = STL(time_series, seasonal=13)
result = stl.fit()

# 绘制分解结果
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(result.observed)
axes[0].set_title('Observed')
axes[1].plot(result.trend)
axes[1].set_title('Trend')
axes[2].plot(result.seasonal)
axes[2].set_title('Seasonal')
axes[3].plot(result.resid)
axes[3].set_title('Residual')

plt.tight_layout()
plt.show()
