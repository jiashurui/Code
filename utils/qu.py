import numpy as np


class EKF:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.Q = process_noise
        self.R = measurement_noise
        self.x = np.array([1, 0, 0, 0, 0, 0, 0])  # 状态向量 [q_w, q_x, q_y, q_z, wx, wy, wz]
        self.P = np.eye(7)  # 状态协方差矩阵

    def predict(self, gyro):
        wx, wy, wz = gyro
        q = self.x[:4]

        # 四元数更新
        F = np.array([
            [1, -0.5 * self.dt * wx, -0.5 * self.dt * wy, -0.5 * self.dt * wz],
            [0.5 * self.dt * wx, 1, 0.5 * self.dt * wz, -0.5 * self.dt * wy],
            [0.5 * self.dt * wy, -0.5 * self.dt * wz, 1, 0.5 * self.dt * wx],
            [0.5 * self.dt * wz, 0.5 * self.dt * wy, -0.5 * self.dt * wx, 1]
        ])
        q = F @ q
        q /= np.linalg.norm(q)  # 归一化四元数

        # 更新状态向量
        self.x[:4] = q
        self.x[4:] = gyro

        # 预测状态协方差矩阵
        F_full = np.eye(7)
        F_full[:4, :4] = F
        self.P = F_full @ self.P @ F_full.T + self.Q

    def update(self, accel, mag):
        q = self.x[:4]
        accel_norm = np.linalg.norm(accel)
        mag_norm = np.linalg.norm(mag)

        if accel_norm == 0 or mag_norm == 0:
            return

        accel /= accel_norm
        mag /= mag_norm

        # 计算预测的重力方向和地磁方向
        g = np.array([2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[0] * q[1] + q[2] * q[3]),
                      q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2])
        b = np.array([np.sqrt(mag[0] ** 2 + mag[1] ** 2), 0, mag[2]])
        h = self.quaternion_rotate(q, b)

        # 计算测量误差
        z = np.concatenate((accel, mag))
        h_combined = np.concatenate((g, h))
        y = z - h_combined

        # 计算雅可比矩阵
        H = self.calculate_jacobian(q, accel, mag)

        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新状态
        self.x += K @ y
        self.x[:4] /= np.linalg.norm(self.x[:4])  # 归一化四元数
        self.P = (np.eye(7) - K @ H) @ self.P

    def calculate_jacobian(self, q, accel, mag):
        # 计算雅可比矩阵
        H = np.zeros((6, 7))
        qw, qx, qy, qz = q

        H[:3, :4] = np.array([
            [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
            [2 * qx, 2 * qw, 2 * qz, 2 * qy],
            [2 * qw, -2 * qx, -2 * qy, 2 * qz]
        ])

        b = np.array([np.sqrt(mag[0] ** 2 + mag[1] ** 2), 0, mag[2]])
        h = self.quaternion_rotate(q, b)
        hx, hy, hz = h

        H[3:, :4] = np.array([
            [2 * hz, 2 * hy, 2 * hx, 2 * hz],
            [2 * hx, 2 * hz, 2 * hy, 2 * hx],
            [2 * hy, 2 * hx, 2 * hz, 2 * hy]
        ])

        return H

    def quaternion_rotate(self, q, v):
        # 旋转向量v通过四元数q
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        v_quat = np.array([0] + list(v))
        return self.quaternion_product(self.quaternion_product(q, v_quat), q_conj)[1:]

    def quaternion_product(self, q1, q2):
        # 四元数乘法
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def get_euler_angles(self):
        q = self.x[:4]
        roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return roll, pitch, yaw


# 示例数据
dt = 0.01
process_noise = np.eye(7) * 0.01
measurement_noise = np.eye(6) * 0.1
ekf = EKF(dt, process_noise, measurement_noise)

accel_data = np.array([0.2, 0.1, 9.7])
gyro_data = np.array([0.01, 0.02, 0.03])
mag_data = np.array([20.0, 30.0, -40.0])

# 进行多次预测和更新步骤
for _ in range(10):
    ekf.predict(gyro_data)
    ekf.update(accel_data, mag_data)
    roll, pitch, yaw = ekf.get_euler_angles()
    print(
        f"Roll: {np.degrees(roll):.2f} degrees, Pitch: {np.degrees(pitch):.2f} degrees, Yaw: {np.degrees(yaw):.2f} degrees")
