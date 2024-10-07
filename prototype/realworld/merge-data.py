import os
import pandas as pd


import os
import pandas as pd

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

    # 遍历每个身体位置
    for body_part in body_parts:
        # 存储每个传感器的文件路径
        sensor_files = {sensor: None for sensor in sensors}

        # 遍历文件夹，寻找匹配的文件
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # 检查文件名是否包含当前身体位置和传感器类型
                for sensor in sensors:
                    if file.startswith(sensor) and body_part in file and file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        sensor_files[sensor] = file_path

        # 读取每个传感器的数据，并按列名进行合并
        dataframes = []
        action_label = None  # 存储动作标签

        for sensor in sensors:  # 按照顺序读取加速度 -> 角速度 -> 磁力
            file_path = sensor_files.get(sensor)
            if file_path:
                try:
                    # 提取动作类型（例如'walking', 'standing'等）
                    for action in action_map:
                        if action in file_path:
                            action_label = action_map[action]
                            break

                    # 读取CSV文件，确保时间和ID列被舍弃，并保留x, y, z数据
                    df = pd.read_csv(file_path, usecols=lambda x: x not in ['attr_time', 'id'])  # 舍弃时间和ID列
                    df = df.add_prefix(sensor + '_')  # 添加前缀区分不同传感器的数据

                    # 添加动作标签列（相同动作标签）
                    df['label'] = action_label

                    dataframes.append(df)
                except Exception as e:
                    print(f'读取文件 {file_path} 时出错: {e}')

        # 合并所有传感器的数据，按列顺序加速度 -> 角速度 -> 磁力
        if dataframes:
            merged_data = pd.concat(dataframes, axis=1)

            # 删除多余的 `label` 列，只保留一个
            label_columns = [col for col in merged_data.columns if col == 'label']
            if len(label_columns) > 1:
                merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

            # 确保 `label` 列在最后
            label_col = merged_data.pop('label')  # 移除 `label` 列
            merged_data['label'] = label_col  # 再将其添加回最后一列

            # 保存合并后的文件
            output_file = f'{body_part}_merged.csv'
            save_dir = root_dir  # 定义保存路径

            # 如果保存路径不存在，则创建
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            merged_data.to_csv(os.path.join(save_dir, output_file), index=False)
            print(f'已保存合并文件: {output_file}')

if __name__ == '__main__':
    for index in range(1, 16):
        merge_data(index)