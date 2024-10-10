# 导入必要的库
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 第一步：使用PyTorch生成数据
X, y = make_classification(n_samples=100,
                           n_features=10,  # 总特征数
                           n_informative=5,  # 信息性特征数
                           n_redundant=0,  # 冗余特征数
                           n_repeated=0,  # 重复特征数
                           n_classes=2,
                           random_state=42)

# 第二步：将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 第三步：使用scikit-learn中的随机森林训练模型
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

# 预测并评估模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率: {accuracy:.2f}')




# 第五步：展示决策边界
# plot_decision_boundary(X_test, y_test, clf)

importances = clf.feature_importances_  # 提取特征重要性
indices = np.argsort(importances)[::-1]  # 根据重要性排序

# 第五步：绘制特征重要性图
plt.figure()
plt.title("features importance")
plt.barh(range(X.shape[1]), importances[indices], align="center")
plt.yticks(range(X.shape[1]), indices)
plt.ylim([-1, X.shape[1]])
plt.ylabel("features")
plt.xlabel("importances")
plt.show()