import numpy as np
#
from matplotlib import pyplot as plt

from utils.report import save_report, save_plot


def show_me_data0(np_arr):
    fig, ax = plt.subplots()
    ax.plot(np_arr)
    ax.legend()

    plt.show()
    return plt
def show_me_acc(np_arr):
    # 生成横坐标，数组索引加1
    x = list(range(1, len(np_arr) + 1))

    # 绘制图表
    plt.figure(figsize=(10, 5))
    plt.plot(x, np_arr, marker='o', linestyle='-')

    # 添加标题和标签
    plt.title('Test Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks([x[-1]])
    plt.tick_params(axis='both', which='both', length=0)

    # 显示图表
    plt.grid(True)
    plt.show()


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
def show_me_hotmap(mat, show=True):
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
    if show:
        plt.show()
    return plt

def show_me_mh_hotmap(mat, show=True):
    label_map = {
        'STANDING': 1,
        'Sitting': 2,
        'Lying': 3,
        'Walking': 4,
        'Climbing stairs': 5,
        'Waist bends forward': 6,
        'Frontal elevation of arms': 7,
        'Knees bending': 8,
        'Cycling': 9,
        'Jogging': 10,
        'Running': 11,
        'Jump front & back': 12
    }
    plt.figure(figsize=(10,10))

    plt.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap using Matplotlib')
    # 在每个单元格的中心显示数字
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f'{mat[i, j]}', ha='center', va='center', color='black')

    # 添加颜色条
    plt.xticks(np.arange(mat.shape[0]), labels=list(label_map.keys()), rotation=90)
    plt.yticks(np.arange(mat.shape[1]), labels=list(label_map.keys()))
    if show:
        plt.show()
    return plt

def show_me_child_hotmap(mat, show=True):
    label_map = {
        'walking': 1,
        'waiting': 2,
        'running': 3,
    }
    plt.figure(figsize=(7,7))

    plt.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')


    # 在每个单元格的中心显示数字
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f'{mat[i, j]}', ha='center', va='center', color='black')
    # comment
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 添加颜色条
    plt.xticks(np.arange(mat.shape[0]), labels=list(label_map.keys()), rotation=90)
    plt.yticks(np.arange(mat.shape[1]), labels=list(label_map.keys()))
    if show:
        plt.show()
    return plt

