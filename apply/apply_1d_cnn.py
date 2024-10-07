# 实例化模型(加载模型参数)
import numpy as np
import torch

from cnn.cnn import DeepOneDimCNN
from prototype.constant import Constant
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

