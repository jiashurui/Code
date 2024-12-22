import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from joblib import dump

from datareader.datareader_stu_1111 import simple_get_stu_1111_all_features, get_features
from prototype import constant

# Parameters
slice_length = 20

# Load student data (global transformation based on frame)
origin_data = simple_get_stu_1111_all_features(slice_length, type='df', with_rpy=True)
origin_data_np = np.array(origin_data)

# Feature extraction
features_list, features_name = get_features(origin_data)
train_data = np.array(features_list)

# Label processing (rounding to avoid float errors)
label = np.round(origin_data_np[:, 0, 9]).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.3, random_state=0)

# Define decision tree and parameter grid
param_grid = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Perform Grid Search for hyperparameter tuning
clf = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, scoring='accuracy', cv=10)
clf.fit(X_train, y_train)

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

# Save the best model
dump(best_model, '../model/decision_tree_stu_1111.joblib')
print("Best model has been saved.")

# # 可视化决策树
# plt.figure(figsize=(12, 8))
# plot_tree(best_model, filled=True, feature_names=, class_names=constant.Constant.uStudent_1111.action_map_en_reverse.values(), rounded=True)
# plt.show()