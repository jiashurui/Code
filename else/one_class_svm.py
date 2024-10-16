import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 生成训练数据 (正常数据)
x_train = 0.3 * np.random.randn(100, 10)
x_train = np.r_[x_train + 2, x_train - 2]

# 生成测试数据 (包含异常点)
x_test = 0.3 * np.random.randn(20, 10)
x_test = np.r_[x_test + 2, x_test - 2]
x_outliers = np.random.uniform(low=-4, high=4, size=(20, 10))

# 训练 One-Class SVM 模型
model = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
model.fit(x_train)

# 预测
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
outlier_pred = model.predict(x_outliers)

# 可视化结果 (仅限前两个维度)
plt.title("One-Class SVM for Anomaly Detection (First 2 Dimensions)")
plt.scatter(x_train[:, 0], x_train[:, 1], c='white', s=20, edgecolor='k', label="Training Data")
plt.scatter(x_test[:, 0], x_test[:, 1], c='blue', s=20, edgecolor='k', label="Test Data")
plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='red', s=20, edgecolor='k', label="Outliers")

# 绘制决策边界 (仅限前两个维度)
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
z = model.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 8))])
z = z.reshape(xx.shape)
plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='black')

plt.legend()
plt.show()