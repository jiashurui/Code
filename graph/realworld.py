import pandas as pd
import matplotlib.pyplot as plt

path = '../data/realworld/1/'
filename = 'acc_sitting_head.csv'

# 读取CSV文件
column_names = ['id', 'attr_time', 'attr_x', 'attr_y', 'attr_z']
df = pd.read_csv(path + filename, names=column_names, nrows=500, skiprows=1)
# format date
df['attr_time'] = pd.to_datetime(df['attr_time'])

fig, ax = plt.subplots()

# 绘制每条线
ax.plot(df.index, df['attr_x'], label='X')
ax.plot(df.index, df['attr_y'], label='Y')
ax.plot(df.index, df['attr_z'], label='Z')

# 设置图例
ax.legend()

# 设置标题和标签
ax.set_title(filename.split('.')[0])
ax.set_xlabel('time')
ax.set_ylabel('value')

fig.autofmt_xdate()
# 显示图形
plt.show()
