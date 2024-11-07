import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from datareader.mh_datareader import simple_get_mh_all_features
from statistic.stat_common import calc_df_features, spectral_centroid, dominant_frequency, calculate_ar_coefficients, \
    calc_fft_spectral_energy, spectral_entropy, calc_acc_sma
from joblib import dump, load

# K = 6 に設定する
K = 12
simpling = 50
features_number = 9
slice_length = 256
# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_mh_all_features(slice_length, type='df', with_rpy= True)
origin_data_np = np.array(origin_data)

features_list = []
for d in origin_data:
    df_features, _ = calc_df_features(d.iloc[:, :9])

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='arm_x', acc_y_name='arm_y', acc_z_name='arm_z', T=simpling)
    gex,gey,gez,get = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='gyro_arm_x', acc_y_name='gyro_arm_y', acc_z_name='gyro_arm_z', T=simpling)
    mex,mey,mez,met = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='magnetometer_arm_x', acc_y_name='magnetometer_arm_y', acc_z_name='magnetometer_arm_z', T=simpling)
    df_features['fft_spectral_energy'] = [aex,aey,aez,gex,gey,gez,mex,mey,mez]

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = spectral_entropy(d.iloc[:, :9], acc_x_name='arm_x', acc_y_name='arm_y', acc_z_name='arm_z', T=simpling)
    gex,gey,gez,get = spectral_entropy(d.iloc[:, :9], acc_x_name='gyro_arm_x', acc_y_name='gyro_arm_y', acc_z_name='gyro_arm_z', T=simpling)
    mex,mey,mez,met = spectral_entropy(d.iloc[:, :9], acc_x_name='magnetometer_arm_x', acc_y_name='magnetometer_arm_y', acc_z_name='magnetometer_arm_z', T=simpling)
    df_features['fft_spectral_entropy'] = [aex,aey,aez,gex,gey,gez,mex,mey,mez]

    centroid_arr = []
    dominant_frequency_arr = []
    ar_co_arr = []
    for i in (range(features_number)):
        centroid_feature = spectral_centroid(d.iloc[:, i].values, sampling_rate=10)
        dominant_frequency_feature = dominant_frequency(d.iloc[:, i].values, sampling_rate=10)
        ar_coefficients = calculate_ar_coefficients(d.iloc[:, i].values)

        centroid_arr.append(centroid_feature)
        dominant_frequency_arr.append(dominant_frequency_feature)
        ar_co_arr.append(ar_coefficients)

    df_features['fft_spectral_centroid'] = np.array(centroid_arr)
    df_features['fft_dominant_frequency'] = np.array(dominant_frequency_arr)

    # 舍弃掉磁力数据(结果会变坏)
    # df_features = df_features.iloc[:6, :]

    # 特征打平
    flatten_val = df_features.values.flatten()
    # 单独一维特征
    # 加速度XYZ
    acc_sma = calc_acc_sma(d.iloc[:, 0], d.iloc[:, 1], d.iloc[:, 2])
    roll_avg = d.iloc[:, 10].mean()
    pitch_avg = d.iloc[:, 11].mean()
    yaw_avg = d.iloc[:, 12].mean()

    flatten_val = np.append(flatten_val, acc_sma)
    flatten_val = np.append(flatten_val, roll_avg)
    flatten_val = np.append(flatten_val, pitch_avg)
    flatten_val = np.append(flatten_val, yaw_avg)
    features_list.append(flatten_val)

train_data = np.array(features_list)

# np round 是因为,标签在转换过程中出现了浮点数,导致astype int的时候,标签错误
label = np.round(origin_data_np[:, 0, 9]).astype(int)

clf = DecisionTreeClassifier(max_depth=5, random_state=0)

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

# 保存模型到文件
dump(clf, '../model/decision_tree_mHealth.joblib')



