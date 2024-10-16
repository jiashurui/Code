from datetime import datetime

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from torch import nn

from anormal.t_SNE import plot_data_pca, plot_data_tSNE
from prototype.constant import Constant
from prototype.model import LSTM
from utils import show, report
from utils.uci_datareader import get_data_1d_uci_all_features, get_uci_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slide_window_length = 128  # 序列长度
learning_rate: float = 0.0001
batch_size = 64
epochs = 1
in_channel = 561
out_channel = 6
model_path = '../model/LSTM.pth'

# UCI HAR
model = LSTM(input_size=in_channel, output_size=out_channel).to(device)
model_load = LSTM(input_size=in_channel, output_size=out_channel).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
loss_function = nn.CrossEntropyLoss()
label_map = Constant.UCI.action_map


def train_model():
    train_data, train_labels, test_data, test_labels = get_data_1d_uci_all_features(slide_window_length)

    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    # 5. 创建并训练SVM模型
    svm_classifier = SVC(kernel='rbf', C=0.5, gamma='scale', random_state=42)  # 使用RBF核
    svm_classifier.fit(train_data, train_labels)

    # 测试集评估
    y_pred_train = svm_classifier.predict(train_data)
    train_accuracy = accuracy_score(train_labels, y_pred_train)
    print(f'Train Accuracy: {train_accuracy:.2f}')

    # 6. 模型评估
    # 在测试集上进行预测
    y_pred = svm_classifier.predict(test_data)

    # 计算准确率
    accuracy = accuracy_score(test_labels, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')

    # 打印分类报告
    print('Classification Report:')
    print(classification_report(test_labels, y_pred))

    # 打印混淆矩阵
    print('Confusion Matrix:')
    print(confusion_matrix(test_labels, y_pred))

    # 保存模型到文件
    joblib.dump(svm_classifier, '../model/uci_svm_model.pkl')

    # TODO
    origin_train,origin_test,_,_ = get_uci_data()
    origin_train = origin_train.reshape(-1, origin_train.shape[2])
    origin_test = np.tile(origin_test.reshape(-1, 1), 128).ravel()  # 假设你保存了标签到 .npy 文件中

    plot_data_pca(origin_train, origin_test, Constant.UCI.action_map_reverse)
    plot_data_tSNE(origin_train, origin_test, Constant.UCI.action_map_reverse)


    # 7. PCA降维并可视化

    plot_data_pca(train_data, train_labels, Constant.UCI.action_map_reverse)
    plot_data_tSNE(train_data, train_labels, Constant.UCI.action_map_reverse)

model_load_flag = False

def apply_lstm(test_data):
    global model_load_flag
    model_apply = LSTM(input_size=in_channel,output_size=out_channel).to(device)

    if not model_load_flag:
        model_apply.load_state_dict(torch.load(model_path, map_location=device))
        model_apply.eval()

    start_time = datetime.now()
    # 归一化(128, 9)
    # test_data = standlize(test_data)
    tensor_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    data = tensor_data.unsqueeze(0)[:, :, :in_channel]
    outputs = model_apply(data)
    pred = outputs.argmax(dim=1, keepdim=True)
    print(f"Model predict finished, start: {start_time} , end: {datetime.now()}")
    return pred

if __name__ == '__main__':
    train_model()