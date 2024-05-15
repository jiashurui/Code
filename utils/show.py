import numpy as np
#
from matplotlib import pyplot as plt

from utils.report import save_report, save_plot


def show_me_data0(np_arr):
    fig, ax = plt.subplots()
    ax.plot(np_arr)
    ax.legend()
    plt.ylim(0, 2.5)

    plt.show()
    return plt

def show_me_data1(df, col_name):
    fig, ax = plt.subplots()
    for col in col_name:
        ax.plot(df.index, df[col], label=col)
    # 设置图例
    ax.legend()
    plt.show()
def show_me_data2(df_list, col_name):
    # too many plot is hard
    for df in df_list[0:5]:
        show_me_data1(df, col_name)
def show_me_hotmap(mat):
    label_map = {
        'waist': 0,
        'chest': 1,
        'forearm': 2,
        'head': 3,
        'shin': 4,
        'thigh': 5,
        'upperarm': 6
    }

    plt.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap using Matplotlib')
    # 在每个单元格的中心显示数字
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f'{mat[i, j]}', ha='center', va='center', color='black')

    # 添加颜色条
    plt.xticks(np.arange(mat.shape[0]), labels=list(label_map.keys()))
    plt.yticks(np.arange(mat.shape[1]), labels=list(label_map.keys()))
    plt.show()
    return plt