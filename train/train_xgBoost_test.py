import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 生成模拟数据
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练 BayesianRidge 模型
model = BayesianRidge()
model.fit(X_train, y_train)

# 使用模型进行预测，获取预测均值和标准差
y_mean, y_std = model.predict(X_test, return_std=True)

# 可视化预测值和不确定性
plt.figure(figsize=(10, 6))
plt.errorbar(X_test.squeeze(), y_mean, yerr=y_std, fmt='o', ecolor='r', capsize=5, label='Predictions with uncertainty')
plt.scatter(X_train.squeeze(), y_train, color='blue', alpha=0.5, label='Training data')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Bayesian Ridge Predictions with Uncertainty')
plt.legend()
plt.show()
