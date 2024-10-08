import matplotlib.pyplot as plt
import numpy as np

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


def show_me_data1(df, col_name):
    fig, ax = plt.subplots()
    for col in col_name:
        ax.plot(df.index, df[col], label=col)
    # 设置图例
    ax.legend()
    plt.show()

def show_me_hotmap(mat, show=True):
    label_map = Constant.RealWorld.action_map

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

def real_time_show_phone_data(float_matrix ,transformed_data, model_pred, rpy):
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
        show_range_percent = 1.1  # 110%
        real_time_show_phone_data.ax.set_xlim(0, float_matrix.shape[0] * show_range_percent)
        real_time_show_phone_data.ax.set_ylim(np.min(float_matrix[:, :3]) * show_range_percent, np.max(float_matrix[:, :3]) * show_range_percent)
        real_time_show_phone_data.ax.set_title(f'acc_data, '
                                               f'model_pred: {model_pred},\n'
                                               f'roll:{np.degrees(rpy[-1,0]):.2f},'
                                               f'pitch:{np.degrees(rpy[-1,1]):.2f},'
                                               f'yaw:{np.degrees(rpy[-1,2]):.2f}.')

    plt.draw()  # 重绘当前图表
    plt.pause(0.01)  # 短暂停以确保图表刷新

