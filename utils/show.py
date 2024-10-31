import glob
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prototype import global_tramsform, constant
from prototype.constant import Constant


def show_me_data0(np_arr):
    fig, ax = plt.subplots()
    ax.plot(np_arr)
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

# Param:{df: 全局变换之前的dataframe}
# Param:{df_transformed: 全局变换之后的dataframe}
# Param:{start_index : 开始的索引}
# Param:{end_index: 结束的索引}
# Result:{展示图片}
def show_acc_data_before_transformed(df, df_transformed, start_index = 0, end_index = 1000):

    # 对展示数据进行截断
    data = df.iloc[start_index:end_index, :]
    data_transformed = df_transformed.iloc[start_index:end_index, :]

    # 绘图
    plt.figure(figsize=(16, 8))

    # 以索引为横坐标,纵坐标为加速度数据(这里默认加速度数据处于前三个位置)
    plt.plot(data.index, data['accx'], color='r', linestyle='-', label='accx')
    plt.plot(data.index, data['accy'], color='g', linestyle='-', label='accy')
    plt.plot(data.index, data['accz'], color='b', linestyle='-', label='accz')

    # 进行了全局变换之后的图
    plt.plot(data_transformed.index, data_transformed['accx'], color='#8B0000', linestyle='--', label='accx_transformed')
    plt.plot(data_transformed.index, data_transformed['accy'], color='#006400', linestyle='--', label='accy_transformed')
    plt.plot(data_transformed.index, data_transformed['accz'], color='#00008B', linestyle='--', label='accz_transformed')

    plt.legend()
    # 展示图片
    plt.show()

def show_me_data1(df, col_name):
    fig, ax = plt.subplots()
    for col in col_name:
        ax.plot(df.index, df[col], label=col)
    # 设置图例
    ax.legend()
    plt.show()


def show_me_hotmap(mat, show=True, label_map=Constant.RealWorld.action_map):
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
    del Constant.mHealth.action_map['Null']
    label_map = Constant.mHealth.action_map
    plt.figure(figsize=(10,10))

    plt.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('mHealth-Confusion-Matrix-1D-CNN')
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

def show_me_stu_hotmap(mat, show=True):
    label_map = {
        'stand': 1,
        'wait': 2,
        'crouch': 3,
        'jump': 4,
        'walk': 5,
        'run': 6,
    }
    plt.figure(figsize=(7,7))

    plt.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Student Action Confusion Matrix')

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

def show_model_gradient_hotmap(model):
    gradient_norms = {name: [] for name, _ in model.named_parameters()}


class GradientUtils:
    def __init__(self, model):
        self.model = model
        self.gradient_norms = {name: [] for name, _ in model.named_parameters()}

    def show(self):
        # 绘制梯度范数曲线
        plt.figure(figsize=(12, 6))
        for name, norms in self.gradient_norms.items():
            plt.plot(norms, label=name)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training')
        plt.legend()
        plt.show()

    def record_gradient_norm(self):
        # 记录梯度范数
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()  # 计算梯度范数
                self.gradient_norms[name].append(grad_norm)



# 实时展示数据的函数，接收一个二维数组 float_matrix，并展示前三列
def real_time_show_phone_data(float_matrix ,transformed_data, model_pred, rpy, ground_truth = None):
    plt.ion()  # 开启交互模式
    # 获取当前数据的前三列
    x_data = np.arange(float_matrix.shape[0])
    y1_data = float_matrix[:, 0]  # 第一列
    y2_data = float_matrix[:, 1]  # 第二列
    y3_data = float_matrix[:, 2]  # 第三列

    y1_data_2 = transformed_data[:, 0]  # 数据集2的第一列
    y2_data_2 = transformed_data[:, 1]  # 数据集2的第二列
    y3_data_2 = transformed_data[:, 2]  # 数据集2的第三列

    # 如果是第一次调用，初始化图表
    if not hasattr(real_time_show_phone_data, 'initialized'):
        real_time_show_phone_data.fig, real_time_show_phone_data.ax = plt.subplots()
        plt.title('acc_data')
        real_time_show_phone_data.line1, = real_time_show_phone_data.ax.plot(x_data, y1_data, label='acc_x',color='red')
        real_time_show_phone_data.line2, = real_time_show_phone_data.ax.plot(x_data, y2_data, label='acc_y',color='green')
        real_time_show_phone_data.line3, = real_time_show_phone_data.ax.plot(x_data, y3_data, label='acc_z',color='blue')

        # 初始化第二组数据的线条
        real_time_show_phone_data.line2_1, = real_time_show_phone_data.ax.plot(x_data, y1_data_2,
                                                                                     label='acc_x_t',
                                                                                     color='#8B0000'
                                                                               , linestyle='--'
                                                                               )
        real_time_show_phone_data.line2_2, = real_time_show_phone_data.ax.plot(x_data, y2_data_2,
                                                                                     label='acc_y_t',
                                                                                     color='#006400'
                                                                               , linestyle='--'
                                                                               )
        real_time_show_phone_data.line2_3, = real_time_show_phone_data.ax.plot(x_data, y3_data_2,
                                                                                     label='acc_z_t',
                                                                                     color='#00008B'
                                                                               , linestyle='--')


        real_time_show_phone_data.ax.set_xlim(0, float_matrix.shape[0])
        real_time_show_phone_data.ax.set_ylim(np.min(float_matrix[:, :3]), np.max(float_matrix[:, :3]))
        real_time_show_phone_data.ax.legend()

        real_time_show_phone_data.initialized = True
    else:
        # 更新数据而不是重新绘制图表
        real_time_show_phone_data.line1.set_xdata(x_data)
        real_time_show_phone_data.line1.set_ydata(y1_data)
        real_time_show_phone_data.line2.set_xdata(x_data)
        real_time_show_phone_data.line2.set_ydata(y2_data)
        real_time_show_phone_data.line3.set_xdata(x_data)
        real_time_show_phone_data.line3.set_ydata(y3_data)

        # 更新第二组数据的线条
        real_time_show_phone_data.line2_1.set_xdata(x_data)
        real_time_show_phone_data.line2_1.set_ydata(y1_data_2)
        real_time_show_phone_data.line2_2.set_xdata(x_data)
        real_time_show_phone_data.line2_2.set_ydata(y2_data_2)
        real_time_show_phone_data.line2_3.set_xdata(x_data)
        real_time_show_phone_data.line2_3.set_ydata(y3_data_2)

        # 重新调整 x 和 y 轴的范围
        show_range_percent = 1.1  # 150%
        real_time_show_phone_data.ax.set_xlim(0, float_matrix.shape[0] * show_range_percent)
        real_time_show_phone_data.ax.set_ylim(
            min(np.min(float_matrix[:, :3]), np.min(transformed_data[:, :3])) * show_range_percent
            ,max(np.max(float_matrix[:, :3]), np.max(transformed_data[:, :3])) * show_range_percent)

        ground_truth_str = f"{ground_truth}" if ground_truth is not None else ""

        real_time_show_phone_data.ax.set_title(f'acc_data, '
                                               f'model_pred: {model_pred},\n'
                                               f'roll:{np.degrees(rpy[-1,0]):.2f},'
                                               f'pitch:{np.degrees(rpy[-1,1]):.2f},'
                                               f'yaw:{np.degrees(rpy[-1,2]):.2f}.'
                                               f'ground_truth:{ground_truth_str}'
                                               )

    plt.draw()  # 重绘当前图表
    plt.pause(0.01)  # 短暂停以确保图表刷新


# 实时展示异常检测结果
def real_time_show_abnormal_data(origin_data,transformed_data, model_recon, loss):
    plt.ion()  # 开启交互模式
    # 获取当前数据前三列
    x_data = np.arange(origin_data.shape[0])
    y1_data = origin_data[:, 0]  # 第一列
    y2_data = origin_data[:, 1]  # 第二列
    y3_data = origin_data[:, 2]  # 第三列

    y1_data_2 = transformed_data[:, 0]  # 变换后第1列
    y2_data_2 = transformed_data[:, 1]  # 变换后第2列
    y3_data_2 = transformed_data[:, 2]  # 变换后第3列

    y1_data_3 = model_recon[:, 0]  # 重建后第1列
    y2_data_3 = model_recon[:, 1]  # 重建后第2列
    y3_data_3 = model_recon[:, 2]  # 重建后第3列


    # 如果是第一次调用，初始化图表
    if not hasattr(real_time_show_phone_data, 'initialized'):
        real_time_show_phone_data.fig, real_time_show_phone_data.ax = plt.subplots(3, 1)
        plt.title('acc_data')

        # 第1行: 原始数据
        real_time_show_phone_data.line1, = real_time_show_phone_data.ax[0].plot(x_data, y1_data, label='acc_x',color='red')
        real_time_show_phone_data.line2, = real_time_show_phone_data.ax[0].plot(x_data, y2_data, label='acc_y',color='green')
        real_time_show_phone_data.line3, = real_time_show_phone_data.ax[0].plot(x_data, y3_data, label='acc_z',color='blue')

        real_time_show_phone_data.ax[0].legend()
        real_time_show_phone_data.ax[0].set_title('Origin Data')


        # 第2行: 全局变换后
        real_time_show_phone_data.line2_1, = real_time_show_phone_data.ax[1].plot(x_data, y1_data_2,
                                                                                     label='acc_x_t',
                                                                                     color='#8B0000'
                                                                               , linestyle='--'
                                                                               )
        real_time_show_phone_data.line2_2, = real_time_show_phone_data.ax[1].plot(x_data, y2_data_2,
                                                                                     label='acc_y_t',
                                                                                     color='#006400'
                                                                               , linestyle='--'
                                                                               )
        real_time_show_phone_data.line2_3, = real_time_show_phone_data.ax[1].plot(x_data, y3_data_2,
                                                                                     label='acc_z_t',
                                                                                     color='#00008B'
                                                                               , linestyle='--')
        real_time_show_phone_data.ax[1].legend()
        real_time_show_phone_data.ax[1].set_title('Transformed Data')

        # 第3行: 重建后
        real_time_show_phone_data.line3_1, = real_time_show_phone_data.ax[2].plot(x_data, y1_data_3,
                                                                                     label='acc_x_r',
                                                                                     color= '#8B0000'
                                                                               , linestyle='--'
                                                                               )
        real_time_show_phone_data.line3_2, = real_time_show_phone_data.ax[2].plot(x_data, y2_data_3,
                                                                                     label='acc_y_r',
                                                                                     color= '#006400'
                                                                               , linestyle='--'
                                                                               )
        real_time_show_phone_data.line3_3, = real_time_show_phone_data.ax[2].plot(x_data, y3_data_3,
                                                                                     label='acc_z_r',
                                                                                     color='#00008B'
                                                                               , linestyle='--')
        real_time_show_phone_data.ax[2].legend()
        real_time_show_phone_data.ax[2].set_title('Reconstruction Data')

        # 设置 x 和 y 轴的限制
        for ax in real_time_show_phone_data.ax:
            ax.set_xlim(0, origin_data.shape[0])
            ax.set_ylim(np.min(origin_data[:, :3]), np.max(origin_data[:, :3]))

        real_time_show_phone_data.initialized = True
    else:
        # 更新数据而不是重新绘制图表
        real_time_show_phone_data.line1.set_xdata(x_data)
        real_time_show_phone_data.line1.set_ydata(y1_data)
        real_time_show_phone_data.line2.set_xdata(x_data)
        real_time_show_phone_data.line2.set_ydata(y2_data)
        real_time_show_phone_data.line3.set_xdata(x_data)
        real_time_show_phone_data.line3.set_ydata(y3_data)

        # 更新第二组数据的线条
        real_time_show_phone_data.line2_1.set_xdata(x_data)
        real_time_show_phone_data.line2_1.set_ydata(y1_data_2)
        real_time_show_phone_data.line2_2.set_xdata(x_data)
        real_time_show_phone_data.line2_2.set_ydata(y2_data_2)
        real_time_show_phone_data.line2_3.set_xdata(x_data)
        real_time_show_phone_data.line2_3.set_ydata(y3_data_2)


        # 更新第三组数据的线条
        real_time_show_phone_data.line3_1.set_xdata(x_data)
        real_time_show_phone_data.line3_1.set_ydata(y1_data_3)
        real_time_show_phone_data.line3_2.set_xdata(x_data)
        real_time_show_phone_data.line3_2.set_ydata(y2_data_3)
        real_time_show_phone_data.line3_3.set_xdata(x_data)
        real_time_show_phone_data.line3_3.set_ydata(y3_data_3)

        # 判断 model_recon 是否超过阈值
        if np.max(model_recon) > 5000:
            # 修改最新 128 个数据点的颜色
            real_time_show_abnormal_data.line3_1.set_color('red')
            real_time_show_abnormal_data.line3_2.set_color('red')
            real_time_show_abnormal_data.line3_3.set_color('red')
        else:
            # 恢复正常颜色
            real_time_show_abnormal_data.line3_1.set_color('#8B0000')
            real_time_show_abnormal_data.line3_2.set_color('#006400')
            real_time_show_abnormal_data.line3_3.set_color('#00008B')

        # 重新调整 x 和 y 轴的范围
        show_range_percent = 1.2  # 120%

        real_time_show_phone_data.ax[0].set_xlim(0, origin_data.shape[0] * show_range_percent)
        real_time_show_phone_data.ax[0].set_ylim(np.min(origin_data[:, :3]) * show_range_percent,np.max(origin_data[:, :3]) * show_range_percent)
        real_time_show_phone_data.ax[1].set_xlim(0, transformed_data.shape[0] * show_range_percent)
        real_time_show_phone_data.ax[1].set_ylim(np.min(transformed_data[:, :3]) * show_range_percent,np.max(transformed_data[:, :3]) * show_range_percent)
        real_time_show_phone_data.ax[2].set_xlim(0, model_recon.shape[0] * show_range_percent)
        real_time_show_phone_data.ax[2].set_ylim(np.min(model_recon[:, :3]) * show_range_percent,np.max(model_recon[:, :3]) * show_range_percent)

    real_time_show_phone_data.ax[2].set_title(f'loss:{loss}')


    plt.tight_layout()  # 调整子图布局以避免重叠
    plt.draw()  # 重绘当前图表
    plt.pause(0.01)  # 短暂停以确保图表刷新

if __name__ == '__main__':
    real_time_show_file_data()
