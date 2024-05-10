# classify realworld data use 1D-CNN
# Euler Angle Rotation 欧拉角变换,转换全局坐标系
# 数据开始是静止状态

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

path = '../data/realworld/1/'

filename = 'acc_jumping_head.csv'
simple_num = 80

# 读取CSV文件
column_names = ['id', 'attr_time', 'attr_x', 'attr_y', 'attr_z']
df = pd.read_csv(path + filename, names=column_names, nrows=simple_num, skiprows=1)
# format date
df['attr_time'] = pd.to_datetime(df['attr_time'])

list_x = df['attr_x']
list_y = df['attr_y']
list_z = df['attr_z']

x_avg = list_x.mean()
y_avg = list_y.mean()
z_avg = list_z.mean()

# roll angle
r = np.degrees(np.arctan(y_avg / z_avg))

# pitch angle
p = np.degrees(np.arctan(-x_avg / np.sqrt(y_avg ** 2 + z_avg ** 2)))

print(r)
print(p)

list_r = []
list_p = []


for data in df.itertuples():
    x = data.attr_x
    y = data.attr_y
    z = data.attr_z

    list_r.append(np.degrees(np.arctan(y/z)))
    list_p.append(np.degrees(np.arctan(-x / np.sqrt(y ** 2 + z ** 2))))

r2, p2 = np.mean(list_r), np.mean(list_p)

print(r2)
print(p2)