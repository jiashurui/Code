# 定义自定义的 Dataset 类，用于加载源域和目标域的数据对
import random

from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data
        self.source_len = len(source_data)
        self.target_len = len(target_data)
        # 使用最小长度，避免索引超出范围
        self.length = min(self.source_len, self.target_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 随机选择源域和目标域的样本
        source_idx = random.randint(0, self.source_len - 1)
        target_idx = random.randint(0, self.target_len - 1)
        source_sample = self.source_data[source_idx]
        target_sample = self.target_data[target_idx]
        return source_sample, target_sample