import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    根据欧拉角计算旋转矩阵
    """
    # 计算各个方向上的旋转矩阵
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
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
        [0, 0, 1]
    ])

    # z y x
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# 示例欧拉角（以弧度为单位）
roll = np.radians(30)
pitch = np.radians(45)
yaw = np.radians(60)

accel_vector = np.array([1.0, 0.5, 0.8])

# 计算旋转矩阵
rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)

# 应用旋转矩阵进行加速度转换
transformed_accel_vector = np.dot(rotation_matrix, accel_vector)

# 输出结果
print("原始加速度向量:", accel_vector)
print("转换后的加速度向量:", transformed_accel_vector)

# 観測方程式
p = np.

