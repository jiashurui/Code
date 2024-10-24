from hmmlearn import hmm
import numpy as np

from datareader.realworld_datareader import get_realworld_raw_for_abnormal
from utils.uci_datareader import get_data_1d_uci_all_data

# 加载数据

train_normal, train_abnormal, test_normal, test_abnormal = get_data_1d_uci_all_data()
real_normal,  real_abnormal = get_realworld_raw_for_abnormal(128, 6)

# 将tensor数据转换为numpy
train_normal = train_normal.cpu().numpy()
train_abnormal = train_abnormal.cpu().numpy()
test_normal = test_normal.cpu().numpy()
test_abnormal = test_abnormal.cpu().numpy()
real_normal = real_normal.cpu().numpy()
real_abnormal = real_abnormal.cpu().numpy()
# 定义 HMM 模型
n_components = 6  # 假设有6个隐状态
model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100)


# 展平数据并训练 HMM
def train_hmm(train_data, model):
    # train_data 的形状是 (n, 128, 6)
    lengths = [sequence.shape[0] for sequence in train_data]  # 每段数据长度（128）

    # 将所有序列展平为一个大序列
    train_data_flattened = np.concatenate(train_data, axis=0)

    # 训练 HMM 模型
    model.fit(train_data_flattened, lengths)

# 计算对数似然
def calculate_average_log_likelihood(data, model):
    total_log_likelihood = 0
    for sequence in data:
        log_likelihood = model.score(sequence)
        total_log_likelihood += log_likelihood
    return total_log_likelihood / len(data)


# 训练 HMM 模型
train_hmm(train_normal, model)

# 对 test_normal 和 test_abnormal 数据集进行预测并计算平均似然
train_normal_result = calculate_average_log_likelihood(train_normal, model)
train_abnormal_result = calculate_average_log_likelihood(train_abnormal, model)
test_normal_result = calculate_average_log_likelihood(test_normal, model)
test_abnormal_result = calculate_average_log_likelihood(test_abnormal, model)


real_normal_result = calculate_average_log_likelihood(real_normal, model)
real_abnormal_result = calculate_average_log_likelihood(real_abnormal, model)


print(f"Train Normal 平均似然: {train_normal_result}")
print(f"Train Abnormal 平均似然: {train_abnormal_result}")

print(f"Test Normal 平均似然: {test_normal_result}")
print(f"Test Abnormal 平均似然: {test_abnormal_result}")

print(f"Real Normal 平均似然: {real_normal_result}")
print(f"Real Abnormal 平均似然: {real_abnormal_result}")

