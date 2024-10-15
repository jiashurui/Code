import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from prototype.constant import Constant
from prototype.global_tramsform import transform_sensor_data_to_df, transform_sensor_data

def radians_to_degree(radians):
    return np.degrees(radians)

# 数据分布(X,Y,Z 加速度数据)
def show_child_hist_stat():
    file_list = glob.glob('../data/child/2023_03/merged_data/*.csv')
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    # acceleration xyz, roll/pitch/yaw
    big_df = big_df.iloc[:,list(range(1, 10)) + list(range(12, 18)) + [23]]
    xyz_rpy = transform_sensor_data(big_df)
    df_trans = pd.DataFrame(xyz_rpy, columns=['trans_x','trans_y','trans_z','my_roll','my_pitch','my_yaw'])
    df_trans['my_roll'] = df_trans['my_roll'].apply(radians_to_degree)
    df_trans['my_pitch'] = df_trans['my_pitch'].apply(radians_to_degree)
    df_trans['my_yaw'] = df_trans['my_yaw'].apply(radians_to_degree)


    df = pd.concat([big_df, df_trans],axis=1)

    print()

if __name__ == '__main__':
    show_child_hist_stat()