import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


# 定义长方体的顶点
def get_cube_vertices():
    # 创建长方体的 8 个顶点坐标
    return np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])


# 绘制长方体的边框
def get_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]


# 创建 roll, pitch, yaw 的序列
# 示例数据，实际使用中可以替换为自己的数据
roll_series = np.linspace(0, 2 * np.pi, 100)  # 旋转角度的序列
pitch_series = np.linspace(0, np.pi, 100)
yaw_series = np.linspace(0, np.pi / 2, 100)

# 创建画布和3D轴
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置轴的范围
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

# 初始化长方体
vertices = get_cube_vertices()


# 更新长方体的位置和旋转
def update(frame):
    ax.clear()  # 清除之前的绘制

    # 设置轴的范围和标签（每次清除之后需要重新设置）
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 获取当前的 roll, pitch, yaw
    roll, pitch, yaw = roll_series[frame], pitch_series[frame], yaw_series[frame]

    # 使用 scipy.spatial.transform.Rotation 进行旋转
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    rotated_vertices = rotation.apply(vertices)

    # 绘制长方体
    faces = get_faces(rotated_vertices)
    cube = Poly3DCollection(faces, alpha=0.5, edgecolor='k', facecolor='cyan')
    ax.add_collection3d(cube)


# 创建动画
ani = FuncAnimation(fig, update, frames=len(roll_series), interval=50)

# 显示动画
plt.show()
