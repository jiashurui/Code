import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from datareader.datareader_stu import simple_get_stu_all_features, get_features
from prototype import constant
from joblib import dump

# Param
slice_length = 20
filtered_label = [2, 3, 4]
mapping = constant.Constant.simple_action_set.mapping_stu

# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_stu_all_features(slice_length, type='df', with_rpy=True)
origin_data_np = np.array(origin_data)
# 抽取特征
features_list = get_features(origin_data)

train_data = np.array(features_list)

# np round 是因为,标签在转换过程中出现了浮点数,导致astype int的时候,标签错误
label = np.round(origin_data_np[:, 0, 9]).astype(int)


X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.3, random_state=0)

# Define Random Forest and parameter grid for Grid Search
param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
clf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, scoring='accuracy', cv=10)

clf = clf.fit(X_train, y_train)


# Output the best parameters and cross-validation score
print("Best parameters found:", clf.best_params_)
print("Best cross-validation accuracy:", clf.best_score_)


# Evaluate on the training set with the best model
best_model = clf.best_estimator_
train_accuracy = best_model.score(X_train, y_train)
y_pred_train = best_model.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print("Training Confusion Matrix:\n", cm_train)
print(f"Training Accuracy: {train_accuracy}")

# Test set evaluation
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Test Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Test Accuracy:", accuracy)
print("Test Recall:", recall)
print("Test F1 Score:", f1)
dump(clf, '../model/random_forest_stu.joblib')