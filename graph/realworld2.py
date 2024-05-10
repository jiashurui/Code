import glob

import pandas as pd
import matplotlib.pyplot as plt

path = '../data/realworld/*/'
filename = 'acc_walking_waist.csv'
full_path = glob.glob(path + filename)

# 读取CSV文件
column_names = ['id', 'attr_time', 'attr_x', 'attr_y', 'attr_z']
df = [pd.read_csv(file, names=column_names, nrows=300, skiprows=400) for file in full_path]
fig, axs = plt.subplots(1,2)

# dataFrame index
index = 0
for ax in axs:
    # 设置图例
    ax.legend()

    # 设置标题和标签
    ax.set_title(filename.split('.')[0])
    ax.set_xlabel('time')
    ax.set_ylabel('value')

    # 绘制每条线
    axs[index].plot(df[index].index, df[index]['attr_x'], label='X')
    axs[index].plot(df[index].index, df[index]['attr_y'], label='Y')
    axs[index].plot(df[index].index, df[index]['attr_z'], label='Z')

    # next df
    index += 1

fig.autofmt_xdate()
# 显示图形
plt.show()
