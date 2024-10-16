from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from torch import nn

from prototype.constant import Constant
from prototype.model import LSTM
from utils import show, report
from utils.uci_datareader import get_data_1d_uci_all_features

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