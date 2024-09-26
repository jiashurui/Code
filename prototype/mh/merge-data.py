import pandas as pd
import os

from utils.config_utils import get_value_from_config

# 文件夹路径
folder_path = get_value_from_config('child_origin_data')

# 获取文件名列表
# 获取文件夹中所有文件和文件夹的名字
all_items = os.listdir(folder_path + '/acc_data')
# 过滤掉文件夹，只保留文件
files = [item for item in all_items if os.path.isfile(os.path.join(folder_path+ '/acc_data', item))]

for file in files:

    # 读取原始数据文件
    data_df = pd.read_csv(folder_path + '/acc_data/' + file)

    # 读取标签文件
    labels_df = pd.read_csv(folder_path + '/label_data/' + file)

    # 为每个标签生成对应的行号索引
    labels_repeated = labels_df.loc[labels_df[0:len(labels_df)].index.repeat(10)].reset_index(drop=True)

    # 合并数据和标签
    merged_df = pd.concat([data_df, labels_repeated], axis=1)

    # Origin Dataの行数　!= Labels Files 行数のため
    if len(data_df) > len(labels_repeated):
        merged_df = merged_df[0:len(labels_repeated)]
    elif len(data_df) < len(labels_repeated):
        merged_df = merged_df[0:len(data_df)]

    # 为合并后的数据添加列名
    merged_df.columns = list(data_df.columns) + ['no','time','X','Y','Z','A','B']

    merged_df['object'] = file.replace('.csv', '')

    # 保存合并后的数据
    merged_df.to_csv(folder_path + '/merged_data/' + file, index=False)

