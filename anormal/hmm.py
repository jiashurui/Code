from hmmlearn import hmm
import numpy as np

from utils.uci_datareader import get_data_1d_uci_all_data

# 假设你有加速度和角速度数据

dataset_name = 'uci'
if dataset_name == 'uci':
    train_normal, train_abnormal, test_normal, test_abnormal = get_data_1d_uci_all_data()

# tensor data --> numpy
train_normal = train_normal.cpu().numpy().reshape(-1, train_normal.shape[2])
train_abnormal = train_abnormal.cpu().numpy().reshape(-1, train_abnormal.shape[2])
test_normal = test_normal.cpu().numpy().reshape(-1, test_normal.shape[2])
test_abnormal = test_abnormal.cpu().numpy().reshape(-1, test_abnormal.shape[2])


# 定义 HMM 模型 (GaussianHMM 适用于连续数据)
model = hmm.GaussianHMM(n_components=6, covariance_type='diag', n_iter=1000)

# 拟合 HMM 模型
model.fit(train_normal)

# 预测状态序列
states = model.predict(test_normal)

# 生成新的序列
log_likelihood_train_normal = model.score(train_normal)
log_likelihood_normal = model.score(test_normal)
log_likelihood_abnormal = model.score(test_abnormal)

print(log_likelihood_train_normal)
print(log_likelihood_normal)
print(log_likelihood_abnormal)

