import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 状态转移函数 f
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




# 转换前和转换后的加速度存储
# 设置过程噪声和测量噪声协方差矩阵
Q = np.eye(3) * 0.01  # 过程噪声协方差矩阵
R = np.eye(3) * 0.1  # 测量噪声协方差矩阵

# 初始状态和协方差
x = np.zeros((3, 1))
P = np.eye(3)

# 时间步长
dt = 1 / 50  # 50 Hz


# 读取数据
file_path = '../data/mHealth/mHealth_subject1.log'
column_names = [
    'acc_chest_x', 'acc_chest_y', 'acc_chest_z', 'ecg_1', 'ecg_2',
    'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z', 'gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z',
    'mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z',
    'acc_arm_x', 'acc_arm_y', 'acc_arm_z', 'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
    'mag_arm_x', 'mag_arm_y', 'mag_arm_z', 'label'
]

data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)
acc_local = data[['acc_chest_x', 'acc_chest_y', 'acc_chest_z']].values

acc_global = []
rpy = []
for i in range(len(acc_local[:1000, :])):
    omega = data[['gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z']].values[i:i + 1].T
    acc = acc_local[i]
    mag = data[['mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z']].values[i]

    # 执行卡尔曼滤波
    x, P = kalman_filter(x, P, omega, acc, mag, dt)

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
    acc_global.append(np.append(a_global,[roll, pitch, yaw]))

# 转换为numpy数组
acc_global = np.array(acc_global)
rpy = np.array(rpy)
# 绘图
plt.figure(figsize=(12, 6))

# 原始加速度
plt.subplot(3, 1, 1)
plt.plot(acc_local[0:1000, 0], label='X-axis')
plt.plot(acc_local[0:1000, 1], label='Y-axis')
plt.plot(acc_local[0:1000, 2], label='Z-axis')
plt.title('Local Acceleration (Chest)')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
# 全局加速度
plt.subplot(3, 1, 2)
plt.plot(acc_global[0:1000, 0], label='X-axis')
plt.plot(acc_global[0:1000, 1], label='Y-axis')
plt.plot(acc_global[0:1000, 2], label='Z-axis')

plt.title('Global Acceleration')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()

# roll pitch yaw
plt.subplot(3, 1, 3)
plt.plot(rpy[0:1000, 0], label='X-axis')
plt.plot(rpy[0:1000, 1], label='Y-axis')
plt.plot(rpy[0:1000, 2], label='Z-axis')

plt.title('Roll Pitch Yaw')
plt.xlabel('Time Steps')
plt.ylabel('degree (m/s^2)')
plt.legend()


plt.tight_layout()
plt.show()