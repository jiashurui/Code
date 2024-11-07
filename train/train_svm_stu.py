import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datareader.datareader_stu import simple_get_stu_all_features, get_features
from prototype import constant
from joblib import dump

# Param
slice_length = 40
filtered_label = [2, 3]
mapping = constant.Constant.simple_action_set.mapping_stu

# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_stu_all_features(slice_length, type='df',
                                          filtered_label=filtered_label,
                                          mapping_label=mapping, with_rpy=True)
origin_data_np = np.array(origin_data)
# 抽取特征
features_list = get_features(origin_data)

train_data = np.array(features_list)

# np round 是因为,标签在转换过程中出现了浮点数,导致astype int的时候,标签错误
label = np.round(origin_data_np[:, 0, 9]).astype(int)

clf = SVC(kernel='rbf', C=1.0, gamma='scale')

X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.3, random_state=0)

clf = clf.fit(X_train, y_train)

train_accuracy = clf.score(X_train, y_train)
y_pred_train = clf.predict(X_train)

cm_train = confusion_matrix(y_train, y_pred_train)
print("Train Confusion Matrix:\n", cm_train)
print(f'Train Accuracy: {train_accuracy}')

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Test Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Test Accuracy:", accuracy)

dump(clf, '../model/svm_stu.joblib')