# 实例化模型(加载模型参数)
from datetime import time, datetime

import numpy as np
import torch

from cnn.cnn import DeepOneDimCNN
from prototype.constant import Constant
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_load = DeepOneDimCNN().to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN-3CH.pth'))
label_map = Constant.RealWorld.action_map

def apply_1d_cnn(test_data):
    start_time = datetime.now()
    # 归一化(128, 9)
    test_data = standlize(test_data)
    tensor_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    data = tensor_data.unsqueeze(0).transpose(1, 2)[:, :9, :]
    outputs = model_load(data)
    pred = outputs.argmax(dim=1, keepdim=True)
    print(f"Model predict finished, start: {start_time} , end: {datetime.now()}")
    return pred

def standlize(arr):
    # 对每一列进行标准化到 [-1, 1]
    arr_min = np.min(arr, axis=0)  # 计算每一列的最小值
    arr_max = np.max(arr, axis=0)  # 计算每一列的最大值

    # 如果最大值和最小值相等，标准化会出现问题，需要检查
    standardized_arr = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        if arr_max[i] - arr_min[i] != 0:
            # 标准化公式：2 * (x - min) / (max - min) - 1
            standardized_arr[:, i] = 2 * (arr[:, i] - arr_min[i]) / (arr_max[i] - arr_min[i]) - 1
        else:
            standardized_arr[:, i] = 0  # 如果列中最大值等于最小值，全部置0

    return standardized_arr