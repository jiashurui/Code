import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.svm import SVC
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch

# 共通の緯度・経度グリッドを定義します
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)


file_path = '../data/weather/Data.csv'
df = pd.read_csv(file_path, index_col=0)
df.head()
selected_lat = common_lat[(common_lat >= 24.0) & (common_lat <= 28.0)]
selected_lon = common_lon[(common_lon >= 125.5) & (common_lon <= 129.5)]

# build select index
selected_index = ['沖縄の天気','沖縄の降水量']
for lat in selected_lat:
    for lon in selected_lon:
        selected_index.append(f"{lat}_{lon}")

df = df[selected_index]

Y1 = (df.iloc[:, 0] / 100).astype(int)
Y2 = df.iloc[:, 1]
X = df.iloc[:, 2:]

df[df.iloc[:, 0] >= 500].iloc[:, :2]
Y1[Y1 == 5] = 3
Y1[Y1 == 7] = 3

y1_train = Y1.iloc[:365]
y1_test = Y1.iloc[365:]
y2_train = Y2.iloc[:365]
y2_test = Y2.iloc[365:]
x_train = X.iloc[:365]
x_test = X.iloc[365:]


clf = SVC(kernel='rbf', C=1.0, gamma='scale')

clf = clf.fit(x_train, y1_train)

train_accuracy = clf.score(x_train, y1_train)
y_pred_train = clf.predict(x_train)

cm_train = confusion_matrix(y1_train, y_pred_train)
print("Train Confusion Matrix:\n", cm_train)
print(f'Train Accuracy: {train_accuracy}')

y_pred = clf.predict(x_test)
cm = confusion_matrix(y1_test, y_pred)
print("Test Confusion Matrix:\n", cm)

accuracy = accuracy_score(y1_test, y_pred)
recall = recall_score(y1_test, y_pred, average='weighted')
f1 = f1_score(y1_test, y_pred, average='weighted')
print("Test Accuracy:", accuracy)
