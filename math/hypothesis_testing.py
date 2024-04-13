#hypothesis testing
from scipy.stats import binom

# 样本数据
n = 100  # 总人数
k = 5   # 加班人数
p = 0.7 # 加班率

# 在原假设下，计算二项分布的累积概率（小于等于 k 的概率）
p_value = binom.cdf(k, n, p)

print("P值为:", p_value)
