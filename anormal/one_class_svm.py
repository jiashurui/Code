import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.metrics import confusion_matrix

from utils.uci_datareader import get_data_1d_uci_all_data

dataset_name = 'uci'
if dataset_name == 'uci':
    train_normal, train_abnormal, test_normal, test_abnormal = get_data_1d_uci_all_data()

# tensor data --> numpy
train_normal = train_normal.cpu().numpy().reshape(-1, train_normal.shape[2])
train_abnormal = train_abnormal.cpu().numpy().reshape(-1, train_abnormal.shape[2])
test_normal = test_normal.cpu().numpy().reshape(-1, test_normal.shape[2])
test_abnormal = test_abnormal.cpu().numpy().reshape(-1, test_abnormal.shape[2])

# 训练 One-Class SVM 模型
model = svm.OneClassSVM(kernel='sigmoid', gamma=0.5, nu=0.1)
model.fit(train_normal)

# 预测
train_normal_pred = model.predict(train_normal)
train_abnormal_pred = model.predict(train_abnormal)
test_normal_pred = model.predict(test_normal)
test_abnormal_pred = model.predict(test_abnormal)


# 统计结果 训练集
train_tp = np.sum(train_normal_pred == 1)
train_tn = np.sum(train_abnormal_pred == -1)

train_fp = np.sum(train_abnormal_pred == 1)
train_fn = np.sum(train_normal_pred == -1)

# 统计结果 测试集
test_tp = np.sum(test_normal_pred == 1)
test_tn = np.sum(test_abnormal_pred == -1)

test_fp = np.sum(test_abnormal_pred == 1)
test_fn = np.sum(test_normal_pred == -1)

# 合并测试集的预测结果和真实标签
train_true = np.concatenate([np.ones(len(train_normal)), -np.ones(len(train_abnormal))])
train_pred = np.concatenate([train_normal_pred, train_abnormal_pred])

test_true = np.concatenate([np.ones(len(test_normal)), -np.ones(len(test_abnormal))])
test_pred = np.concatenate([test_normal_pred, test_abnormal_pred])

# 计算混淆矩阵
cm_train = confusion_matrix(train_true, train_pred, labels=[1, -1])
cm_test = confusion_matrix(test_true, test_pred, labels=[1, -1])

# 可视化训练集和测试集的混淆矩阵
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Training Set')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')

plt.tight_layout()
plt.show()