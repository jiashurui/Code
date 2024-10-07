# 实例化模型(加载模型参数)
import numpy as np
import torch

from cnn.cnn import DeepOneDimCNN
from prototype.constant import Constant
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_load = DeepOneDimCNN().to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN-3CH.pth'))
label_map = Constant.RealWorld.action_map

def apply_1d_cnn(test_data):
    # (128, 3)
    tensor_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    data = tensor_data.unsqueeze(0).transpose(1, 2)[:, :3, :]
    outputs = model_load(data)
    pred = outputs.argmax(dim=1, keepdim=True)
    return pred