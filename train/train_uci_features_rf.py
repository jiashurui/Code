import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from utils.uci_datareader import get_data_1d_uci_all_features

from sklearn.tree import export_graphviz
import graphviz
slide_window_length = 128  # 序列长度
in_channel = 561
out_channel = 6
def train_model():
    train_data, train_labels, test_data, test_labels = get_data_1d_uci_all_features(slide_window_length)

    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)

    # 创建决策树分类器
    clf = RandomForestClassifier(random_state=3407, max_depth=5, n_estimators=10)

    # 训练模型
    clf.fit(train_data, train_labels)

    # 进行预测
    train_pred = clf.predict(train_data)
    test_pred = clf.predict(test_data)

    # 评估模型
    print("Train Accuracy:", accuracy_score(train_labels, train_pred))
    print("Test Accuracy:", accuracy_score(test_labels, test_pred))
    print("Classification Report:\n", classification_report(test_labels, test_pred))

    #
    # # 导出决策树为dot格式
    # dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=df['activity'].unique(),
    #                            filled=True)
    #
    # # 使用graphviz绘制决策树
    # graph = graphviz.Source(dot_data)
    # graph.render("decision_tree")  # 生成的决策树图像将保存在当前目录下
    # graph.view()  # 显示决策树

if __name__ == '__main__':
    train_model()