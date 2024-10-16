import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from utils.uci_datareader import get_data_1d_uci_all_data

dataset_name = 'uci'
if dataset_name == 'uci':
    train_normal, train_abnormal, test_normal, test_abnormal = get_data_1d_uci_all_data()

# tensor data --> numpy
train_normal = train_normal.numpy().reshape(-1, train_normal.shape[2])
train_abnormal = train_abnormal.numpy().reshape(-1, train_abnormal.shape[2])
test_normal = test_normal.numpy().reshape(-1, test_normal.shape[2])
test_abnormal = test_abnormal.numpy().reshape(-1, test_abnormal.shape[2])



# 训练 One-Class SVM 模型
model = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
model.fit(train_normal)

# 预测
train_normal_pred = model.predict(train_normal)
train_abnormal_pred = model.predict(train_abnormal)
test_normal_pred = model.predict(test_normal)
test_abnormal_pred = model.predict(test_abnormal)


print()
# 可视化结果 (仅限前两个维度)
# plt.title("One-Class SVM for Anomaly Detection")
# plt.scatter(x_train[:, 0], x_train[:, 1], c='white', s=20, edgecolor='k', label="Training Data")
# plt.scatter(x_test[:, 0], x_test[:, 1], c='blue', s=20, edgecolor='k', label="Test Data")
# plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='red', s=20, edgecolor='k', label="Outliers")
#
# # 绘制决策边界 (仅限前两个维度)
# xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# z = model.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 8))])
# z = z.reshape(xx.shape)
# plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='black')
#
# plt.legend()
# plt.show()