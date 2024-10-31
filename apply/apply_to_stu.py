import numpy as np
import torch

from cnn.conv_lstm import ConvLSTM
from datareader.datareader_stu import simple_get_stu_all_features
from prototype.constant import Constant
from utils import show

# Parameter
in_channel = 6
out_channel = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

# Option

# 使用realworld进行训练
filtered = True
if filtered:
    filtered_label = [2,3]
    label_map = Constant.realworld_x_uStudent.action_map_en_reverse
    mapping_label = Constant.realworld_x_uStudent.mapping_stu

else:
    filtered_label = []
    label_map = Constant.uStudent.action_map_en

# Data
origin_data = simple_get_stu_all_features(25, filtered_label, mapping_label)
test_data = origin_data[:, :, :in_channel]
test_labels = origin_data[:, :, 9][:, 0].to(torch.long)

test_data = test_data.transpose(1, 2)
label_map = Constant.RealWorld.action_map

model_load = ConvLSTM(input_dim=in_channel, output_dim=out_channel).to(device)

model_load.eval()
num_sum = 0
correct = 0
test_loss = 0
confusion_matrix = np.zeros((len(label_map), len(label_map)))

with torch.no_grad():
    for i in range(0, test_data.size()[0], batch_size):
        input_data, label = test_data[i: i + batch_size], test_labels[i: i + batch_size]
        if label.size(0) != batch_size:
            continue

        outputs = model_load(input_data)
        pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引

        for (expected, actual) in zip(pred, label.reshape(batch_size, 1)):
            confusion_matrix[actual, expected] += 1
            if actual == expected:
                correct += 1

        num_sum += batch_size

print(f'\nTest set: Average loss: {test_loss / num_sum:.4f}, Accuracy: {correct}/{num_sum} ({100. * correct / num_sum:.0f}%)\n')

heatmap_plot = show.show_me_hotmap(confusion_matrix)



