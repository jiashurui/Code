import glob

from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

base_path = '/Users/jiashurui/Desktop/toyota_202404'
everyday_everyone_data = glob.glob(base_path)

# 21 days
everyday_list = [e.split('/')[-1] for e in glob.glob(base_path + '/*')]
everyday_list.sort()

# 50 person
everybody_list = [e.split('/')[-1] for e in glob.glob(base_path + '/' + everyday_list[0] + '/*')]

# choose one person (dc37bcf1456dd7fe's data)
person = everybody_list[8]

this_person_everyday_data = glob.glob(base_path + '/*/' + person + '/*')

show_count = 0
time_diff_threshold = 500000

for everyday in everyday_list:
    if show_count == 5:
        break
    this_day_data = glob.glob(base_path + '/' + everyday + '/' + person + '/*')
    this_day_geo_datafile = glob.glob(base_path + '/' + everyday + '/' + person + '/' + 'geopoints.csv')[0]
    this_day_acc_datafile = glob.glob(base_path + '/' + everyday + '/' + person + '/' + 'accelerometers.csv')[0]

    df_geo = pd.read_csv(this_day_geo_datafile)
    df_acc = pd.read_csv(this_day_acc_datafile)

    df_geo['time_diff'] = df_geo['UNIX_time(milli)'].diff().abs()

    threshold_index_list = df_geo[df_geo['time_diff'] > time_diff_threshold].index.tolist()
    print('threshold_index: ' + str(threshold_index_list))
    threshold_index = threshold_index_list[0]
    plt.figure(figsize=(40,40))

    plt.subplot(2, 2, 1)
    plt.plot(df_geo['latitude'], df_geo['longitude'],  label='geo')
    # plt.plot(df_geo['latitude'][threshold_index:], df_geo['longitude'][threshold_index:], color='blue',label='back')
    # plt.plot(df_geo['latitude'][:threshold_index], df_geo['longitude'][:threshold_index], color='red',label='go')

    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.gca().yaxis.get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.title('geo point' + ' ' + everyday + ' ' + person)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    df_acc['time'] = pd.to_datetime(df_acc['UNIX_time(milli)'], unit='ms')

    plt.plot(df_acc['time'], df_acc['x(m/s2)'], label='x')
    plt.plot(df_acc['time'], df_acc['y(m/s2)'], label='y')
    plt.plot(df_acc['time'], df_acc['z(m/s2)'], label='z')
    plt.xlabel('time')
    plt.ylabel('acc')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    plt.legend()
    plt.title('acc data' + ' ' + everyday + ' ' + person)
    plt.grid(True)
    plt.rcParams['timezone'] = 'Asia/Tokyo'

    plt.subplot(2, 2, 3)
    df_geo['time'] = pd.to_datetime(df_acc['UNIX_time(milli)'], unit='ms')
    plt.plot(df_geo.index, df_geo['time'], label='time')
    plt.xlabel('index')
    plt.ylabel('time')
    plt.legend()
    plt.title('time seria data' + ' ' + everyday + ' ' + person)
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.grid(True)


    plt.subplot(2, 2, 4)
    plt.plot(df_acc.index, df_acc['time'], label='time')
    plt.xlabel('index')
    plt.ylabel('time')
    plt.legend()
    plt.title('time seria data' + ' ' + everyday + ' ' + person)
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.grid(True)

    plt.show()
    show_count += 1

    # plt.subplot(2, 2, 4)


    print(this_day_geo_datafile)