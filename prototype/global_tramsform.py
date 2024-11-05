# 状态转移函数 f
import numpy as np
from scipy.spatial.transform import Rotation

HZ = 10
def f(x, u):
    quat = x.flatten()
    omega = u.flatten()
    dt = 1 / HZ

    # 计算四元数的变化
    r_delta = Rotation.from_rotvec(omega * dt)
    r_current = Rotation.from_quat(quat)
    r_new = r_delta * r_current

    # 返回更新后的四元数
    return r_new.as_quat().reshape((4, 1))


# 测量函数 h
def h(acc, mag):
    # 从加速度计和磁力计计算 roll, pitch, yaw
    roll = np.arctan2(acc[1], acc[2])
    pitch = np.arctan2(-acc[0], np.sqrt(acc[1] ** 2 + acc[2] ** 2))
    yaw = np.arctan2(
        mag[1] * np.cos(roll) - mag[2] * np.sin(roll),
        mag[0] * np.cos(pitch) + mag[1] * np.sin(pitch) * np.sin(roll)
        + mag[2] * np.sin(pitch) * np.cos(roll)
    )
    # 从欧拉角转换为四元数
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_quat().reshape((4, 1))


# 预测步骤
def predict_state(x, P, omega, dt):
    u = omega * dt
    x_pred = f(x, u)
    # 线性化状态转移函数的雅可比矩阵
    F = np.eye(4)  # 对于四元数的近似，这里使用单位矩阵
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


# 更新步骤
def update_state(x_pred, P_pred, acc, mag):
    # 测量值
    z = h(acc, mag)
    H = np.eye(4)
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(4) - K @ H) @ P_pred
    return x_upd, P_upd


# 卡尔曼滤波器
def kalman_filter(x, P, omega, acc, mag, dt):
    x_pred, P_pred = predict_state(x, P, omega, dt)
    x_upd, P_upd = update_state(x_pred, P_pred, acc, mag)
    return x_upd, P_upd


# 设置过程噪声和测量噪声协方差矩阵
Q = np.eye(4) * 0.01  # 过程噪声协方差矩阵
R = np.eye(4) * 0.1  # 测量噪声协方差矩阵

# 初始状态和协方差
x = np.array([[0], [0], [0], [1]])  # 初始四元数，表示无旋转
P = np.eye(4)

# 时间步长
dt = 1 / HZ

# 转换前和转换后的加速度存储
def transform_sensor_data(data):
    global Q, R
    Q = np.eye(4) * 0.01  # 过程噪声协方差矩阵
    R = np.eye(4) * 0.12  # 测量噪声协方差矩阵

    # 初始状态和协方差
    x = np.array([[0], [0], [0], [1]])  # 初始四元数，表示无旋转
    P = np.eye(4)

    # 时间步长
    dt = 1 / HZ
    data = np.array(data)
    arr_acc = data[:, 0:3]
    arr_gyo = data[:, 3:6]
    arr_mag = data[:, 6:9]

    # Global 変換後
    acc_global = []
    for i in range(len(arr_acc)):
        acc = arr_acc[i]
        gyo = arr_gyo[i]
        mag = arr_mag[i]

        # 执行卡尔曼滤波
        x, P = kalman_filter(x, P, gyo, acc, mag, dt)

        # 获取姿态估计
        r = Rotation.from_quat(x.flatten())
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        # 构造旋转矩阵
        R_matrix = r.as_matrix()

        # 转换到全局加速度
        a_global = R_matrix @ acc.reshape((3, 1))
        acc_global.append(np.append(a_global.flatten(), [roll, pitch, yaw]))

    acc_global = np.array(acc_global)
    return acc_global


def transform_sensor_data_to_df0(data):
    origin_data = data.copy()
    np_acc = transform_sensor_data(data)
    data.iloc[:, :3] = np_acc[:, :3]
    return origin_data, data

def transform_sensor_data_to_df(data):
    np_acc = transform_sensor_data(data)
    data.iloc[:, :3] = np_acc[:, :3]
    return data

def transform_sensor_data_to_df1(data):
    np_acc = transform_sensor_data(data)
    data.iloc[:, :3] = np_acc[:, :3]
    data['roll'] = np_acc[:, 3]
    data['pitch'] = np_acc[:, 4]
    data['yaw'] = np_acc[:, 5]

    return data

def transform_sensor_data_to_np(data):
    np_acc = transform_sensor_data(data)
    data[:, :3] = np_acc[:, :3]

    # np(128,9) , np(roll,pitch,yaw)
    return data, np_acc[:, 3:]

def fake_transform_sensor_data_to_np(data):
    np_acc = np.zeros((128,3))
    return data,np_acc