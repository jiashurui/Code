import os
import pandas as pd
import re

def merge_data(index):
    # 文件所在的根目录
    root_dir = f'../../data/realworld/{index}'

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

    # 正则表达式用于提取动作、部位、编号等信息
    pattern = re.compile(r'(\w+)_(\w+)(?:_(\d+)_(\w+))?')  # e.g., acc_standing_waist_1 or acc_standing_waist (没有编号)

    # 遍历每个身体位置
    for body_part in body_parts:
        # 存储每个动作的数据帧，用于不同动作的横向合并
        action_dataframes = []

        # 用于存储同一动作、同一编号（或无编号）的传感器数据
        data_groups = {}

        # 遍历文件夹，寻找匹配的文件
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # 解析文件名，确保其包含相应的部位和符合命名模式
                match = pattern.match(file)
                if match and body_part in file:
                    sensor_type, action, part, number = match.groups()

                    # 如果没有编号，则将编号置为空字符串，用于无编号文件的合并
                    number = number if number else ''

                    # 生成文件路径
                    file_path = os.path.join(root, file)

                    # 按照动作和编号（或者无编号）进行分组
                    group_key = (action, number)  # 动作和编号的组合键

                    if group_key not in data_groups:
                        data_groups[group_key] = {}

                    try:
                        # 读取CSV文件，舍弃时间和ID列
                        df = pd.read_csv(file_path, usecols=lambda x: x not in ['attr_time', 'id'])

                        stop_simple = 800  # 数据静止的个数

                        df = df[stop_simple: len(df)]

                        # 根据传感器类型，添加前缀并存储在相应的组中
                        if sensor_type == 'acc':
                            df = df.add_prefix('acc_')
                            df['label'] = action_map.get(action)  # 添加加速度的label
                        elif sensor_type == 'Gyroscope':
                            df = df.add_prefix('gyro_')
                        elif sensor_type == 'MagneticField':
                            df = df.add_prefix('mag_')

                        data_groups[group_key][sensor_type] = df

                    except Exception as e:
                        print(f'读取文件 {file_path} 时出错: {e}')

        # 将分组中的数据进行横向合并
        for group_key, sensor_data in data_groups.items():
            acc_df = sensor_data.get('acc')
            gyro_df = sensor_data.get('Gyroscope')
            mag_df = sensor_data.get('MagneticField')

            # 如果加速度、角速度和磁力计都存在，则合并
            if acc_df is not None and gyro_df is not None and mag_df is not None:
                dataframes = [acc_df, gyro_df, mag_df]  # 依次合并加速度、角速度、磁力计

                # 横向合并传感器数据，并保留列顺序：加速度 -> 角速度 -> 磁力计 -> label
                merged_action_data = pd.concat(dataframes, axis=1)

                # 检查是否有空值的行
                if merged_action_data.isnull().values.any():
                    # 打印存在空值的行数
                    missing_rows = merged_action_data.isnull().any(axis=1).sum()
                    print(f'动作 {group_key[0]}, 编号 {group_key[1]} 中存在空值的行数: {missing_rows}')

                    # 删除包含空值的行，并打印删除行数
                    merged_action_data = merged_action_data.dropna()
                    print(f'删除了 {missing_rows} 行')

                # 只保留加速度的label列
                label_col = merged_action_data.pop('label')
                merged_action_data['label'] = label_col

                # 保存合并后的数据帧
                action_dataframes.append(merged_action_data)

        # 合并所有动作的数据（纵向合并），生成对应身体部位的最终文件
        if action_dataframes:
            final_merged_data = pd.concat(action_dataframes, axis=0, ignore_index=True)  # 纵向合并，不保留原索引

            # 保存合并后的文件
            output_file = f'{body_part}_merged.csv'
            save_dir = root_dir  # 定义保存路径

            # 如果保存路径不存在，则创建
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            final_merged_data.to_csv(os.path.join(save_dir, output_file), index=False)
            print(f'已保存合并文件: {output_file}')

if __name__ == '__main__':
    for index in range(1, 16):
        merge_data(index)
