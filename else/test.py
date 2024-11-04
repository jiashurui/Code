import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Init Model
lda = LinearDiscriminantAnalysis()
svm = SVC(kernel="linear", random_state=0)
log_reg = LogisticRegression(random_state=0,max_iter=1000)
#################################################################
# LDA
lda.fit(X_train, y_train)

# LDAの予測値を取得
y_train_lda = lda.predict(X_train)
y_pred_lda = lda.predict(X_test)

# Accuracyを計算
accuracy_train_lda = accuracy_score(y_train, y_train_lda)
accuracy_test_lda = accuracy_score(y_test, y_pred_lda)

print("LDA Train Accuracy:", accuracy_train_lda)
print("LDA Test Accuracy:", accuracy_test_lda)

# 混同行列を計算(Test集)
conf_matrix_lda = confusion_matrix(y_test, y_pred_lda)
print(f"LDA Confusion Matrix:\n{conf_matrix_lda}")

#################################################################
# SVM
svm.fit(X_train, y_train)

# LDAの予測値を取得
y_train_svm = svm.predict(X_train)
y_pred_svm = svm.predict(X_test)

# Accuracyを計算
accuracy_train_svm = accuracy_score(y_train, y_train_svm)
accuracy_test_svm = accuracy_score(y_test, y_pred_svm)

print("SVM Train Accuracy:", accuracy_train_svm)
print("SVM Test Accuracy:", accuracy_test_svm)

# 混同行列を計算(Test集)
conf_matrix_lda = confusion_matrix(y_test, y_pred_svm)
print(f"SVM Confusion Matrix:\n{conf_matrix_lda}")

#################################################################
# LogicRegression
log_reg.fit(X_train, y_train)

# LDAの予測値を取得
y_train_log = log_reg.predict(X_train)
y_pred_log = log_reg.predict(X_test)

# Accuracyを計算
accuracy_train_log = accuracy_score(y_train, y_train_log)
accuracy_test_log = accuracy_score(y_test, y_pred_log)

print("LogisticRegression Train Accuracy:", accuracy_train_log)
print("LogisticRegression Test Accuracy:", accuracy_test_log)

# 混同行列を計算(Test集)
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
print(f"LogisticRegression Confusion Matrix:\n{conf_matrix_log}")
