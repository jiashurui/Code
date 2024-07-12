import glob

import pandas as pd
import torch
import torchvision.models as models

from prototype.model import Simple1DCNN
from utils.slidewindow import slide_window2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read 1d-CNN
# model_origin = Simple1DCNN(in_channels=3, out_label=12).to(device)

base_path = '/Users/jiashurui/Desktop/Dataset_labeled/acc_data/*.csv'
labeled_path = '/Users/jiashurui/Desktop/Dataset_labeled/label_data/*.csv'
file_path = glob.glob(labeled_path)


appended_data = []
final_data = []

file_change_dict = {}

for file_name in file_path:
    data = pd.read_csv(file_name)

    slice_point_list = []
    pre_val = -1
    for index, value in data['X'].items():
        if value != pre_val:
            slice_point_list.append(index * 10) # 1s ---> 10hz
        pre_val = value

    file_change_dict[file_name] = slice_point_list

print(file_change_dict)


