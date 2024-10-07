# 状态转移函数 f
import numpy as np


def f(x, u):
    roll, pitch, yaw = x.flatten()
    wx, wy, wz = u.flatten()

    # 计算角速度对欧拉角的影响
    roll_dot = wx + np.tan(pitch) * (wy * np.sin(roll) + wz * np.cos(roll))
    pitch_dot = wy * np.cos(roll) - wz * np.sin(roll)
    yaw_dot = (wy * np.sin(roll) + wz * np.cos(roll)) / np.cos(pitch)

    # 预测新的状态
    roll += roll_dot
    pitch += pitch_dot
    yaw += yaw_dot

    return np.array([[roll], [pitch], [yaw]])


# 测量函数 h
def h(acc, mag):
    roll = np.arctan2(acc[1], acc[2])
    pitch = np.arctan2(-acc[0], np.sqrt(acc[1] ** 2 + acc[2] ** 2))
    yaw = np.arctan2(
        mag[1] * np.cos(roll) - mag[2] * np.sin(roll),
        mag[0] * np.cos(pitch) + mag[1] * np.sin(pitch) * np.sin(roll)
        + mag[2] * np.sin(pitch) * np.cos(roll)
    )
    return np.array([[roll], [pitch], [yaw]])


# 预测步骤
def predict_state(x, P, omega, dt):
    u = omega * dt
    x_pred = f(x, u)
    F = np.eye(3) + dt * np.array([
        [1, np.tan(x[1, 0]) * np.sin(x[0, 0]), np.tan(x[1, 0]) * np.cos(x[0, 0])],
        [0, np.cos(x[0, 0]), -np.sin(x[0, 0])],
        [0, np.sin(x[0, 0]) / np.cos(x[1, 0]), np.cos(x[0, 0]) / np.cos(x[1, 0])]
    ])
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


# 更新步骤
def update_state(x_pred, P_pred, acc, mag):
    z = h(acc, mag)
    H = np.eye(3)
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(3) - K @ H) @ P_pred
    return x_upd, P_upd


# 卡尔曼滤波器
def kalman_filter(x, P, omega, acc, mag, dt):
    x_pred, P_pred = predict_state(x, P, omega, dt)
    x_upd, P_upd = update_state(x_pred, P_pred, acc, mag)
    return x_upd, P_upd

# 设置过程噪声和测量噪声协方差矩阵
Q = np.eye(3) * 0.01  # 过程噪声协方差矩阵
R = np.eye(3) * 0.1  # 测量噪声协方差矩阵

# 初始状态和协方差
x = np.zeros((3, 1))
P = np.eye(3)

# 时间步长
dt = 1 / 50  # 50 Hz

# 转换前和转换后的加速度存储

def transform_sensor_data(data):

    Q = np.eye(3) * 0.01  # 过程噪声协方差矩阵
    R = np.eye(3) * 0.1  # 测量噪声协方差矩阵

    # 初始状态和协方差
    x = np.zeros((3, 1))
    P = np.eye(3)

    # 时间步长
    dt = 1 / 50  # 50 Hz
    data = np.array(data)
    arr_acc = data[:,0:3]
    arr_gyo = data[:,3:6]
    arr_mag = data[:,6:9]

    # Global 変換後
    acc_global = []
    rpy = []

    for i in range(len(arr_acc)):
        acc = arr_acc[i]
        gyo = arr_gyo[i]
        mag = arr_mag[i]

        # 执行卡尔曼滤波
        x, P = kalman_filter(x, P, gyo, acc, mag, dt)

        # 获取姿态估计
        roll, pitch, yaw = x.flatten()

        # 构造旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_matrix = R_z @ R_y @ R_x

        # 转换到全局加速度
        a_global = R_matrix @ acc
        acc_global.append(np.append(a_global, [roll, pitch, yaw]))

    acc_global= np.array(acc_global)
    return acc_global

def transform_sensor_data_to_df(data):
    np_acc = transform_sensor_data(data)
    data.iloc[:, :3] = np_acc[:,:3]

    return data