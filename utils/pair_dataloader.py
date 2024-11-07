# 定义自定义的 Dataset 类，用于加载源域和目标域的数据对
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
        source_idx = idx % self.source_len  # 防止索引超出范围
        target_idx = idx % self.target_len  # 防止索引超出范围
        source_sample = self.source_data[source_idx]
        target_sample = self.target_data[target_idx]
        return source_sample, target_sample
