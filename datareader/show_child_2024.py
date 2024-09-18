import glob

from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def show_child_2024():
    base_path = '/Users/jiashurui/Desktop/Dataset_labeled/acc_data/*.csv'
    files = glob.glob(base_path)
    show_count = 0

    for file in files:
        if show_count == 3:
            break

        df_data = pd.read_csv(file)
        df = df_data.head(100)
        plt.figure(figsize=(20, 20))

        plt.subplot(2, 2, 1)
        plt.plot(df['latitude'], df['longitude'], label='geo')
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
        show_count += 1


# (batch_size , seq , feature)
def show_tensor_data(tensor_before, tensor_after, loss):
    batch_size = tensor_before.shape[0]
    numpy_data = tensor_before.numpy()
    numpy_data_after = tensor_after.numpy()

    seq_data = numpy_data[0]
    seq_data_after = numpy_data_after[0]
    df = pd.DataFrame(seq_data, columns=['x', 'y', 'z'])
    df_after = pd.DataFrame(seq_data_after, columns=['x_after', 'y_after', 'z_after'])

    plt.figure(figsize=(10, 10))
    plt.plot(df.index, df['x'], label='x', color='red')
    plt.plot(df.index, df['y'], label='y', color='green')
    plt.plot(df.index, df['z'], label='z', color='blue')
    plt.plot(df_after.index, df_after['x_after'], label='x_after', color='#8B0000', linestyle='--')
    plt.plot(df_after.index, df_after['y_after'], label='y_after', color='#006400', linestyle='--')
    plt.plot(df_after.index, df_after['z_after'], label='z_after', color='#00008B', linestyle='--')

    # 文字说明
    # 获取当前坐标轴对象
    ax = plt.gca()
    plt.figtext(ax.get_xlim()[1], ax.get_ylim()[0],f"loss: {loss}", fontsize=12, color='black')
    plt.xlabel('time')
    plt.ylabel('acc data')
    plt.legend()
    plt.title('Data before and after')
    plt.show()
