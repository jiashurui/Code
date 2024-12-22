import numpy as np
from joblib import dump
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datareader.datareader_stu_1111 import simple_get_stu_1111_all_features, get_features

# Param
slice_length = 20

# 读取并处理数据
origin_data = simple_get_stu_1111_all_features(slice_length, type='df', with_rpy=True)
origin_data_np = np.array(origin_data)

# 提取特征
features_list, features_name = get_features(origin_data)
train_data = np.array(features_list)

# 标签处理（四舍五入并转换为整数）
label = np.round(origin_data_np[:, 0, 9]).astype(int)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.3, random_state=0)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义 SVM 模型和参数搜索范围，包含多种核函数
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf'],
    'coef0': [0.0, 0.1]
}
svc = SVC()
clf = GridSearchCV(svc, param_grid, scoring='accuracy', cv=10)

# 执行网格搜索以找到最佳参数
clf.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best Param:", clf.best_params_)
print("Best Score (Cross Validation):", clf.best_score_)

# 用最佳模型预测训练集和测试集
best_model = clf.best_estimator_

# 训练集评估
y_pred_train = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
train_cm = confusion_matrix(y_train, y_pred_train)
print("\nTrain Confusion Matrix:\n", train_cm)
print("Train Accuracy:", train_accuracy)

# 测试集评估
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test, average='weighted')
test_f1 = f1_score(y_test, y_pred_test, average='weighted')
test_precision = precision_score(y_test, y_pred_test, average='weighted')
test_cm = confusion_matrix(y_test, y_pred_test)

print("\nTest Confusion Matrix:\n", test_cm)
print("Test Accuracy:", test_accuracy)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print("Test Precision:", test_precision)

# 保存最佳模型
dump(best_model, '../model/svm_stu_1111_optimized.joblib')
print("最佳模型已保存。")
