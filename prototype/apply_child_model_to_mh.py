import numpy as np
import torch
from torch import nn

from datareader.mh_datareader import get_mh_data_1d_3ch_for_test
from prototype.constant import Constant
from prototype.dataReader import get_data_1d_3ch_child
from prototype.model import Simple1DCNN
from utils import show, report
from sklearn.metrics import f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型(加载模型参数)
model_load = Simple1DCNN(in_channels=3,out_label=3).to(device)
model_load.load_state_dict(torch.load('../model/1D-CNN-3CH.pth'))
model_load.eval()

# Parameter
label_map = Constant.ChildWalk.action_map
batch_size = 128
num_sum = 0
correct = 0
test_loss = 0
confusion_matrix = np.zeros((len(label_map), len(label_map)))
all_pred = []
all_labels = []
slide_window_length = 40  # 序列长度

data, label = get_mh_data_1d_3ch_for_test(slide_window_length)
label -= 1

with torch.no_grad():
    for i in range(0, data.size()[0], batch_size):
        input_data, label = data[i: i + batch_size], label[i: i + batch_size]
        if label.size(0) != batch_size:
            continue

        outputs = model_load(input_data)
        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引
        for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
            confusion_matrix[actual, expected] += 1
            if actual == expected:
                correct += 1

        all_pred.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        num_sum += batch_size

print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

heatmap_plot = show.show_me_child_hotmap(confusion_matrix)
fig = heatmap_plot.gcf()
report.save_plot(heatmap_plot, 'heat-map')