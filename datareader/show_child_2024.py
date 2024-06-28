import glob

from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

base_path = '/Users/jiashurui/Desktop/Dataset_labeled/acc_data/*.csv'
files = glob.glob(base_path)
show_count = 0

for file in files:
    if show_count == 3:
        break

    df_data = pd.read_csv(file)
    df = df_data.head(100)
    plt.figure(figsize=(20,20))

    plt.subplot(2, 2, 1)
    plt.plot(df['latitude'], df['longitude'],  label='geo')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.title('geo data' + ' ' + file)

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.gca().yaxis.get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    df['time'] = pd.to_datetime(df['UNIX_time(milli)'], unit='ms')

    plt.plot(df['time'], df['accx'], label='x')
    plt.plot(df['time'], df['accy'], label='y')
    plt.plot(df['time'], df['accz'], label='z')
    plt.xlabel('time')
    plt.ylabel('acc')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.grid(True)
    plt.rcParams['timezone'] = 'Asia/Tokyo'
    plt.title('acc data' + ' ' + file)

    plt.show()
    show_count +=1