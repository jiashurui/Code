import glob

from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from utils.unix_time import datetime_to_unixtime
# 行動順：　１、立つ
# 　　　　　２、フラフラ
# 　　　　　３、しゃがむ
# 　　　　　４、飛ぶ
# 　　　　　５、歩く
# 　　　　　６、走る

# １　賈：　　15:19 ~ 15:25 　　   1721974740000 ~ 1721975100000
# ２　溝脇：　15:27 ~ 15:33        1721975220000 ~ 1721975580000
# ３　高野：　15:34 ~ 15:40        1721975640000 ~ 1721976000000
# ４　高橋：　15:41 ~ 15:47        1721976060000 ~ 1721976420000
# ５　金：　　15:48 ~ 15:54        1721976480000 ~ 1721976840000

base_path = '/Users/jiashurui/Desktop/Dataset_student/0726_lab/accelerometers.csv'
f = glob.glob(base_path)

df_data = pd.read_csv(f[0])
start_time = datetime_to_unixtime(2024, 7, 26, 15, 53, 9)
end_time = datetime_to_unixtime(2024, 7, 26, 15, 53, 42)

# jiashurui
filtered_values = df_data[(df_data['UNIX_time(milli)'] >= start_time)
                          & (df_data['UNIX_time(milli)'] <= end_time)]

plt.figure(figsize=(20, 20))
plt.subplot(1, 1, 1)

plt.plot(filtered_values['UNIX_time(milli)'], filtered_values['x(m/s2)'], label='x', color='r')
plt.plot(filtered_values['UNIX_time(milli)'], filtered_values['y(m/s2)'], label='y', color='g')
plt.plot(filtered_values['UNIX_time(milli)'], filtered_values['z(m/s2)'], label='z', color='b')

plt.xlabel('time')
plt.ylabel('acc')
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_scientific(False)
plt.gca().xaxis.get_major_formatter().set_useOffset(False)
plt.title('acc data jiashurui')
plt.legend()

plt.show()
