import numpy as np

# 假设有一个形状为 [3, 2] 的数组和一个形状为 [3, 1] 的数组
array1 = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])

array2 = np.array([[7],
                   [8],
                   [9]])

# 使用 np.hstack 函数将这两个数组水平堆叠起来
stacked_array = np.hstack((array1, array2))

# 打印堆叠后的数组形状
print("Stacked array shape:", stacked_array.shape)
