import glob

import pandas as pd

root_dir = f'../../data/realworld/1'

# 定义身体位置
body_parts = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']

# 定义传感器类型，按顺序排列
sensors = ['acc', 'Gyroscope', 'MagneticField']

# 动作映射表
action_map = {
    'climbingdown': 0,
    'climbingup': 1,
    'jumping': 2,
    'lying': 3,
    'running': 4,
    'sitting': 5,
    'standing': 6,
    'walking': 7,
}

def merge_data(index):
    base_path = f'../../data/realworld/{index}/'
    name_pattern = f'{base_path}acc_*_$placement.csv'
    dict_filename = {}

    for place in body_parts:
        name_pattern = name_pattern.replace('$placement', place)
        #  在这里搜索所有动作
        file_list = glob.glob(name_pattern)
        # 同一sensor 同一部位


        for file_name in file_list:

            gyro_file_name = file_name.replace('acc', 'Gyroscope')
            mag_file_name = file_name.replace('acc', 'MagneticField')
            try:
                df_acc = pd.read_csv(file_name).iloc[:, 2:].add_prefix('acc_')
                df_gyro = pd.read_csv(gyro_file_name).iloc[:, 2:].add_prefix('gyro_')
                df_mag = pd.read_csv(mag_file_name).iloc[:, 2:].add_prefix('mag_')
            except FileNotFoundError:
                continue

            df = pd.concat([df_acc,df_gyro,df_mag],axis=1)

            # 对于每一个dataframe , 按照文件名给其打上标签
            matched_substrings = [label for label in action_map.keys() if label in file_name]

            if not matched_substrings or len(matched_substrings) != 1:
                raise KeyError("无法生成标签")
            else:
                df['label'] = action_map.get(matched_substrings[0])

            print(f'{place}_merged , deleted null rows {df.isnull().any(axis=1).sum()}, {file_name}')

            # check data
            df = df.dropna()

            dict_filename[f'{place}_merged'] = df

    print(dict_filename.keys())


if __name__ == '__main__':
    for i in range(16):
        merge_data(i)