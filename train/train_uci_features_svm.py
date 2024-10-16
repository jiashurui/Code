import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from anormal.t_SNE import plot_data_pca, plot_data_tSNE
from prototype.constant import Constant
from utils.uci_datareader import get_data_1d_uci_all_features

slide_window_length = 128  # 序列长度
in_channel = 561
out_channel = 6
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

    # TODO 对原始数据进行降维花的时间太长了
    # origin_train,origin_test,_,_ = get_uci_data()
    # origin_train = origin_train.reshape(-1, origin_train.shape[2])
    # origin_test = np.tile(origin_test.reshape(-1, 1), 128).ravel()  # 假设你保存了标签到 .npy 文件中
    #
    # plot_data_pca(origin_train, origin_test, Constant.UCI.action_map_reverse)
    # plot_data_tSNE(origin_train, origin_test, Constant.UCI.action_map_reverse)


    # 7. PCA降维并可视化
    plot_data_pca(train_data, train_labels, Constant.UCI.action_map_reverse)
    plot_data_tSNE(train_data, train_labels, Constant.UCI.action_map_reverse)

model_load_flag = False

# TODO apply SVM
def apply_svm(test_data):
    print()

if __name__ == '__main__':
    train_model()