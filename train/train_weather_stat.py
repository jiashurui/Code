import math

import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AutoReg

# 0.共通の緯度・経度グリッドを定義します
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)

# 1. データを読み
file_path = '../data/weather/Data.csv'
df = pd.read_csv(file_path, index_col=0)
df.head()

# 2. 沖縄周辺を絞り（11x11 Grid）
selected_lat = common_lat[(common_lat >= 24.0) & (common_lat <= 29.0)]
selected_lon = common_lon[(common_lon >= 125.0) & (common_lon <= 130.0)]
area_size = len(selected_lat)

selected_index = ['沖縄の天気', '沖縄の降水量']
for lat in selected_lat:
    for lon in selected_lon:
        selected_index.append(f"{lat}_{lon}")
df = df[selected_index]

# 3.ラベル整形
Y1 = (df.iloc[:, 0] / 100).astype(int)
Y2 = df.iloc[:, 1]
X = df.iloc[:, 2:]

# 3.1.「5と7」のラベルを「3」に整形
df[df.iloc[:, 0] >= 500].iloc[:, :2]
Y1[Y1 == 5] = 3
Y1[Y1 == 7] = 3

# 3.2. データセット分割
y1_train = Y1.iloc[:365]
y1_test = Y1.iloc[365:]
y2_train = Y2.iloc[:365]
y2_test = Y2.iloc[365:]
x_train = X.iloc[:365]
x_test = X.iloc[365:]

# 3.3 np配列に調整
x_train = np.array(x_train)
x_test = np.array(x_test)


# 4 関数定義

# 4.1 モデルの採点関数
def clf_score(accuracy):
    return print("clf_score:", int(accuracy * 10 + 0.5))


def reg_score(mae, y2_test):
    score = 1 - mae / 15
    return print("reg_score:", max(0, int(score * 10 + 0.5)))


# 4.2 スライディングウィンドウ
# @Param:{arr: 入力の時系列配列, Eg. np(365, 169)}
# @Result:{Eg. np(365, 8, 169)}
# slide の step は　1
def slide_window(arr, window_size):
    num_windows = len(arr) - window_size + 1
    result = np.zeros((len(arr), window_size, arr.shape[1]))

    for i in range(num_windows):
        result[i + window_size - 1] = arr[i:i + window_size]

    for j in range(window_size - 1):
        result[j, :] = arr[0]
    return result


# 4.3 気圧に対して、近隣気圧Gridの差分、及び交互作用項を求め（上、下、左、右）
# @Param:{pressures: 毎日気圧のグリッド , Eg: np(365, 13, 13)}
# @Result:{total_differences : 近隣Grid差分の和　sum(abs(k1 - k2) + abs(k1 - k3))}
# @Result:{total_differences_interact : 近隣Gridの交互作用項 sqrt(k1 * k2 + k1 * k3 ...)}
def differences_nearby(pressures):
    d, m, n = pressures.shape
    total_differences = np.zeros(d)
    total_differences_interact = np.zeros(d)

    for k in range(d):
        layer_difference = 0
        layer_difference_interact = 0
        for i in range(m):
            for j in range(n):
                current_value = pressures[k, i, j]

                if i > 0:
                    layer_difference += abs(current_value - pressures[k, i - 1, j])
                    layer_difference_interact += math.sqrt(current_value * pressures[k, i - 1, j])
                if i < m - 1:
                    layer_difference += abs(current_value - pressures[k, i + 1, j])
                    layer_difference_interact += math.sqrt(current_value * pressures[k, i + 1, j])
                if j > 0:
                    layer_difference += abs(current_value - pressures[k, i, j - 1])
                    layer_difference_interact += math.sqrt(current_value * pressures[k, i, j - 1])

                if j < n - 1:
                    layer_difference += abs(current_value - pressures[k, i, j + 1])
                    layer_difference_interact += math.sqrt(current_value * pressures[k, i, j + 1])

        total_differences[k] = layer_difference
        total_differences_interact[k] = layer_difference_interact
    return total_differences, total_differences_interact


# 4.4 過去５、７日間のデータに（同じのGridは過去の自分自身と比較）、自己回帰モデルを適用、その係数を特徴量として返す
# @Param:{data: あるグリッド過去5日また7日間の気圧配列 Eg. np(10050, 10150, 10250 ...)}
# @Result:{自己回帰モデルの係数　AR: X(t) = c + Sum(b *  X(t-1)) + eps}
def calculate_ar_coefficients(data, lags=1):
    if len(set(data)) == 1 or np.var(data) == 0:
        return np.array([0])
    model = AutoReg(data, lags=lags, old_names=False)
    model_fit = model.fit()
    return model_fit.params[1:]


# 4.5 統計の特徴量
# @Param:{data: 1日の気圧データ}
# @Result:{features: 気圧の統計特徴量配列}
def calc_feature(data):
    day_mean = np.mean(data, axis=1)  # 平均値(Mean)
    day_min = np.min(data, axis=1)  # 最小値(Min)
    day_max = np.max(data, axis=1)  # 最大値(Min)
    day_range = day_max - day_min  # 変動範囲(Range) : Max - Min
    day_median = np.median(data, axis=1)  # 中央値(Median)
    day_std = np.std(data, axis=1)  # 標準偏差(standard Deviation)
    day_var = np.var(data, axis=1)  # 分散(Variance)
    day_skewness = skew(data, axis=1)  # 歪度(Skewness)
    day_kurt = kurtosis(data, axis=1)  # 尖度(Kurtosis)

    # 近隣Gridの差分和、交互作用項を求め
    average_differences, average_differences_interact = differences_nearby(
        data.reshape(len(data), area_size, area_size))

    features = np.stack(
        [day_mean, day_min, day_max, day_range, day_median, day_median, day_std, day_var, day_skewness, day_kurt,
         average_differences, average_differences_interact], axis=1)
    return features


# 4.6 基準日から過去３、５、７日の特徴量の平均を求め、気圧が時間経過の関係性を表す
# @Param:{data: 1日の気圧データ}
# @Result:{features: 気圧の統計特徴量配列}
def hist_day_features(data, days):
    arr = slide_window(data, window_size=days)

    avg_n_days_features_list = []
    # 過去の数日間データを計算し、それぞれの特徴を、さらに平均値の取得する
    for i in range(arr.shape[0]):
        n_days_features = calc_feature(arr[i])
        avg_n_days_features = np.mean(n_days_features, axis=0)
        avg_n_days_features_list.append(avg_n_days_features)

    result = np.array(avg_n_days_features_list)

    feature_vector = np.zeros(arr.shape[0])

    # ARモデルは過去５、7日間のデータを計算、(3日の場合、特徴量は0に設定)
    if days > 3:
        for day in range(arr.shape[0]):
            ar_values = []  # 用于保存当天所有空间特征的 AR 系数
            for space_idx in range(arr.shape[1]):
                ar = calculate_ar_coefficients(arr[day, :, space_idx], lags=1)
                ar_values.append(ar)

            feature_vector[day] = np.mean(ar_values)
    result = np.concatenate((result, feature_vector.reshape(-1, 1)), axis=1)
    return result

# 4.7 離散ウェーブレット変換
def haar_dwt(data):
    flattened_features = []
    for feature in data:
        cA, (cH, cV, cD) = pywt.dwt2(feature, 'haar')
        flattened = np.concatenate([c.flatten() for c in (cA, cH, cV, cD)])
        flattened_features.append(flattened)

    return np.array(flattened_features)


# 5. 処理の主幹
# 5.1 過去３、５、７日間の統計特徴を計算
# （過去のデータがない場合（例：train_data:2022年1月1日は、その日のデータをコピーして計算する））
# （test_data　例：train_data:2023年1月1日の過去データは、2022年分のデータを使用しないこと）

three_days_avg_features_train = hist_day_features(x_train, days=3)
three_days_avg_features_test = hist_day_features(x_test, days=3)

five_days_avg_features_train = hist_day_features(x_train, days=5)
five_days_avg_features_test = hist_day_features(x_test, days=5)

seven_days_avg_features_train = hist_day_features(x_train, days=7)
seven_days_avg_features_test = hist_day_features(x_test, days=7)

res = haar_dwt(x_train.reshape(len(x_train), area_size, area_size))
res_test = haar_dwt(x_test.reshape(len(x_test), area_size, area_size))

# 5.2 毎日の気圧統計特徴を計算
today_feature_train = calc_feature(x_train)
today_feature_test = calc_feature(x_test)

X_train = np.concatenate(
    (today_feature_train,
     three_days_avg_features_train,
     five_days_avg_features_train,
     seven_days_avg_features_train,
     res
     ),
    axis=1)

X_test = np.concatenate((today_feature_test,
                         three_days_avg_features_test,
                         five_days_avg_features_test,
                         seven_days_avg_features_test,res_test), axis=1)




# 5.3 0 ~ 1に正規化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=0.90, random_state=3407)
pca_result = pca.fit_transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("PCA 维度:", len(explained_variance))
print("方差解释率:", explained_variance)
print("方差累计解释率:", np.sum(explained_variance))


# 5.4 分類モデルRF
rf = RandomForestClassifier(random_state=3047, n_estimators=50, max_depth=20,
                            min_samples_split=2)


# 5.5 OneVsRestの分類機構を導入する
clf = OneVsRestClassifier(estimator=rf,)

# 5.6 モデルトレーニング
clf.fit(X_train, y1_train)

# 5.7 10-foldの交差検証でトレーニングの安定さを評価
scores = cross_val_score(clf, X_train, y1_train, cv=KFold(n_splits=10, shuffle=True, random_state=3047))

# 5.8 分類の精度＆混同行列
y_pred_test = clf.predict(X_test)
test_accuracy = accuracy_score(y1_test, y_pred_test)


# 5.9 分類の結果
print("Test Accuracy:", test_accuracy)
clf_score(test_accuracy)
cm = confusion_matrix(y1_test, y_pred_test)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y1_test), yticklabels=np.unique(y1_test))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 训练集评估
print("Accuracy scores for each fold:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation of accuracy:", scores.std())



# 6. 回帰問題
br_model = BayesianRidge()

# 6.1 モデルの学習
br_model.fit(X_train, y2_train)

# 6.2 降水量を予測する
y2_pred = br_model.predict(X_test)

# 6.3 MSEで誤差評価
mae = mean_absolute_error(y2_test, y2_pred)

# 6.4 回帰の結果
print(f"Mean Absolute Error: {mae}")
reg_score(mae, y2_test)

plt.figure(figsize=(12, 6))
plt.plot(y2_test, label='True Value', marker='o')
plt.plot(y2_pred, label='Prediction', marker='x')
plt.title('True Values vs Predictions')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()


